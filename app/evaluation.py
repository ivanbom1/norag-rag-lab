import json
import os
import time
from collections import defaultdict
from datetime import datetime

from openai import APIError
from pydantic import BaseModel, ValidationError

from app import bm25
from app.answering import (CANDIDATES, DEFAULT_RETRIEVERS, MODEL, RETRIEVERS, ask, llm, normalize, retrieve,
                           retrieve_with_cost)
from app.fusion import RRF_K
from app.reranking import RERANK_CANDIDATES, RERANK_MODEL

QUESTION_SETS = {
    "kestrel": "kestrel-kb/eval/questions.starter.jsonl",
    "monte-cristo": "eval/questions/monte-cristo.jsonl",
}
CORPUS_DESCRIPTIONS = {
    "kestrel": "Kestrel Analytics, a B2B dashboard software company",
    "monte-cristo": "the novel The Count of Monte Cristo by Alexandre Dumas",
}
RESULTS_DIR = "eval/results"
KS = (1, 3, 5, 10)
SCORE_KEYS = ("found_by", "rrf", "distance", "score", "rerank_score")


def scores(chunk):
    return {key: chunk[key] for key in SCORE_KEYS if key in chunk}


def load_questions(corpus):
    with open(QUESTION_SETS[corpus], encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def check_quotes(questions, corpus_dir):
    problems = []
    for question in questions:
        for item in question["evidence"]:
            for location in [item, *item.get("alternatives", [])]:
                path = os.path.join(corpus_dir, location["source"])
                text = open(path, encoding="utf-8").read() if os.path.exists(path) else ""
                if normalize(location["quote"]) not in normalize(text):
                    problems.append(f"{question['id']}: quote not found in {location['source']}: {location['quote']!r}")
    return problems


# Kestrel rule: a chunk is relevant to an evidence item if it comes from its source and contains its quote.
def is_relevant(chunk, item):
    return any(
        chunk["source"] == location["source"] and normalize(location["quote"]) in normalize(chunk["text"])
        for location in [item, *item.get("alternatives", [])]
    )


def first_relevant_rank(item, chunks):
    return next((rank for rank, chunk in enumerate(chunks, start=1) if is_relevant(chunk, item)), None)


def evaluate_retrieval(question, corpus, sources, settings, retrievers, rerank=False):
    chunks, cost = retrieve_with_cost(question["question"], corpus, sources, settings, max(KS), retrievers, rerank)
    ranks = [first_relevant_rank(item, chunks) for item in question["evidence"]]
    found = lambda k: [rank is not None and rank <= k for rank in ranks]
    return {
        "evidence_ranks": ranks,
        "hit": {k: any(found(k)) for k in KS},
        "recall": {k: round(sum(found(k)) / len(ranks), 3) for k in KS},
        "cost": cost,
        "retrieved": [
            {"rank": rank, "id": chunk["id"], "source": chunk["source"], "start": chunk["start"],
             "end": chunk["end"], **scores(chunk),
             "relevant_to": [i for i, item in enumerate(question["evidence"]) if is_relevant(chunk, item)]}
            for rank, chunk in enumerate(chunks, start=1)
        ],
    }


def evaluate_answer(question, corpus, sources, settings, top_k, retrievers, rerank=False):
    result = ask(question["question"], corpus, sources, settings, top_k, retrievers, rerank)
    return {
        "top_k": top_k,
        "status": result["status"],
        "answer": result["answer"],
        "citations": result["citations"],
        "retrieved": [{"label": c["label"], "id": c["id"], "source": c["source"], **scores(c)}
                      for c in result["retrieved"]],
        "evidence_in_context": any(is_relevant(c, item) for c in result["retrieved"] for item in question["evidence"])
        if question["answerable"] else None,
        "cost": result["cost"],
        "grade": None,
        "failure": None,
    }


COST_KEYS = ("retrieval_ms", "fusion_ms", "rerank_ms", "llm_calls", "prompt_tokens", "completion_tokens", "llm_ms")


def average_cost(costs):
    if not costs:
        return None
    n = len(costs)
    return {
        "questions": n,
        **{key: round(sum(cost.get(key, 0) for cost in costs) / n, 1) for key in COST_KEYS},
        "retriever_ms": {name: round(sum(cost["retriever_ms"].get(name, 0) for cost in costs) / n, 1)
                         for name in RETRIEVERS if any(name in cost["retriever_ms"] for cost in costs)},
    }


def format_cost(cost):
    parts = []
    if retrieval := cost["retrieval"]:
        per_retriever = ", ".join(f"{name} {ms}" for name, ms in retrieval["retriever_ms"].items())
        rerank = f", rerank {retrieval['rerank_ms']}" if retrieval.get("rerank_ms") else ""
        parts.append(f"retrieval {retrieval['retrieval_ms']} ms ({per_retriever}, fusion {retrieval['fusion_ms']}{rerank}), "
                     f"{retrieval['llm_calls']:g} LLM calls")
    if answer := cost["answer"]:
        parts.append(f"answer {answer['llm_calls']:g} LLM calls, {answer['prompt_tokens']:.0f} prompt + "
                     f"{answer['completion_tokens']:.0f} completion tokens, "
                     f"{answer['retrieval_ms'] + answer['llm_ms']:.0f} ms (LLM {answer['llm_ms']:.0f})")
    return "; ".join(parts)


def summarize(rows):
    def stats(group):
        group = [row for row in group if "retrieval" in row]
        return {
            "questions": len(group),
            **{f"hit@{k}": round(sum(row["retrieval"]["hit"][k] for row in group) / len(group), 3) for k in KS},
            **{f"recall@{k}": round(sum(row["retrieval"]["recall"][k] for row in group) / len(group), 3) for k in KS},
        } if group else None

    by_type = defaultdict(list)
    for row in rows:
        by_type[row["type"]].append(row)
    return {
        "overall": stats(rows),
        "by_type": {kind: stats(group) for kind, group in by_type.items() if stats(group)},
        "cost": {"retrieval": average_cost([row["retrieval"]["cost"] for row in rows if "retrieval" in row]),
                 "answer": average_cost([row["answer"]["cost"] for row in rows if "answer" in row])},
    }


def results_name(corpus, settings, retrievers, rerank=False):
    # Dense alone keeps the V2 file names. Otherwise: corpus_800_10_20_bm25, corpus_800_10_20_dense+bm25...
    if retrievers == ["dense"] and not rerank:
        return settings.collection_name(corpus)
    return f"{bm25.table_name(corpus, settings)}_{'+'.join(retrievers)}{'+rerank' if rerank else ''}"


def run_evaluation(corpus, sources, settings, answers=False, answer_top_k=5, delay=0, on_progress=None,
                   retrievers=DEFAULT_RETRIEVERS, rerank=False):
    retrievers = [name for name in RETRIEVERS if name in retrievers]
    # Warm-up: the first dense query loads the embedding model, which would inflate the first latency.
    retrieve("warm-up", corpus, sources, settings, 1, retrievers, rerank)
    rows = []
    for question in load_questions(corpus):
        row = {"id": question["id"], "type": question["type"], "question": question["question"],
               "answerable": question["answerable"], "expected_answer": question["expected_answer"]}
        if question["answerable"]:
            row["retrieval"] = evaluate_retrieval(question, corpus, sources, settings, retrievers, rerank)
        if answers:
            row["answer"] = evaluate_answer(question, corpus, sources, settings, answer_top_k, retrievers, rerank)
            time.sleep(delay)
        rows.append(row)
        if on_progress:
            on_progress(row)

    created = datetime.now()
    name = results_name(corpus, settings, retrievers, rerank) + (f"-answers-{created:%Y%m%d-%H%M%S}" if answers else "")
    report = {
        "corpus": corpus,
        "created": created.isoformat(timespec="seconds"),
        "settings": {"chunk_size": settings.chunk_size, "overlap_percent": settings.overlap_percent,
                     "min_paragraph_length": settings.min_paragraph_length, "retrievers": retrievers,
                     "rrf_k": RRF_K, "candidates_per_retriever": CANDIDATES,
                     "embedding": "all-MiniLM-L6-v2" if "dense" in retrievers else None,
                     "rerank": rerank, "rerank_model": RERANK_MODEL if rerank else None,
                     "rerank_candidates": RERANK_CANDIDATES if rerank else None,
                     "answer_top_k": answer_top_k if answers else None},
        "model": MODEL if answers else None,
        "sources": sources,
        "summary": summarize(rows),
        "results": rows,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return path, report


class Verdict(BaseModel):
    correct: bool
    reason: str


VERDICT_SCHEMA = {
    "type": "object",
    "properties": {"correct": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["correct", "reason"],
}


def judge(question, answer):
    prompt = (f"Question: {question['question']}\nReference answer: {question['expected_answer']}\n"
              f"Candidate answer: {answer}\n\nIs the candidate answer correct, meaning it states the key facts of "
              "the reference answer without contradicting it? Reply with correct and a one-sentence reason.")
    response = llm.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_schema", "json_schema": {"name": "verdict", "schema": VERDICT_SCHEMA}},
    )
    return Verdict.model_validate_json(response.choices[0].message.content or "")


def run_no_context(corpus, delay=0, on_progress=None):
    path = os.path.join(RESULTS_DIR, f"no-context-{corpus}.jsonl")
    previous = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            previous = {row["id"]: row for row in map(json.loads, f)}

    rows = []
    for question in load_questions(corpus):
        # Resumable: answers that succeeded before are kept, only errors and new questions are asked again.
        old = previous.get(question["id"])
        if old and not old["judge_reason"].startswith("error"):
            rows.append(old)
            continue
        row = {"id": question["id"], "type": question["type"], "answerable": question["answerable"],
               "question": question["question"], "expected_answer": question["expected_answer"]}
        try:
            response = llm.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": (
                f"This question is about {CORPUS_DESCRIPTIONS[corpus]}. Answer it in one or two sentences from "
                f"your own knowledge. If you do not know, say so.\n\nQuestion: {question['question']}")}])
            row["answer"] = response.choices[0].message.content or ""
            # Unanswerable questions have no correct answer to know in advance: they are not flagged.
            if question["answerable"]:
                verdict = judge(question, row["answer"])
                row["judge_correct"], row["judge_reason"] = verdict.correct, verdict.reason
            else:
                row["judge_correct"], row["judge_reason"] = None, "unanswerable: not judged"
        except (APIError, ValidationError) as error:
            row["answer"] = row.get("answer")
            row["judge_correct"], row["judge_reason"] = None, f"error: {error}"
        row["override"] = old.get("override") if old else None
        rows.append(row)
        if on_progress:
            on_progress(row)
        time.sleep(delay)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path, rows

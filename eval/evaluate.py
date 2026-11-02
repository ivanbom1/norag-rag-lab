import argparse
import json
import os
import re
import time
from collections import defaultdict

from openai import APIError

from app.answering import MODEL, ask, llm, retrieve
from app.indexing import ChunkSettings, index_document, is_indexed

CORPUS_DIR = "corpus"
QUESTIONS_PATH = "eval/questions.json"
RESULTS_DIR = "eval/results"


def locate(text, quote):
    pattern = r"\s+".join(re.escape(word) for word in quote.split())
    matches = list(re.finditer(pattern, text))
    if len(matches) != 1:
        raise ValueError(f"quote found {len(matches)} times, expected once: {quote!r}")
    return matches[0].start(), matches[0].end()


def load_questions():
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)
    texts = {}
    for question in questions:
        for part in question["evidence"]:
            for evidence in part:
                if evidence["file"] not in texts:
                    with open(os.path.join(CORPUS_DIR, evidence["file"]), encoding="utf-8") as f:
                        texts[evidence["file"]] = f.read()
                evidence["start"], evidence["end"] = locate(texts[evidence["file"]], evidence["quote"])
    return questions


def first_hit_rank(evidence, chunks):
    for rank, chunk in enumerate(chunks, start=1):
        if chunk["source"] == evidence["file"] and chunk["start"] < evidence["end"] and chunk["end"] > evidence["start"]:
            return rank
    return None


# Evidence is a list of required parts; each part lists alternative locations that state it.
def evaluate_retrieval(question, sources, settings, top_k):
    chunks = retrieve(question["question"], sources, settings, top_k)
    part_ranks = []
    for part in question["evidence"]:
        found = [rank for rank in (first_hit_rank(evidence, chunks) for evidence in part) if rank]
        part_ranks.append(min(found) if found else None)
    rank = None if None in part_ranks else max(part_ranks)
    return {"hit": rank is not None, "rank": rank, "part_ranks": part_ranks}


def contains_expected(answer, expected):
    answer = answer.lower()
    return all(any(option.lower() in answer for option in group.split("|")) for group in expected)


def evaluate_answer(question, sources, settings, top_k):
    result = ask(question["question"], sources, settings, top_k)
    if question["type"] == "unanswerable":
        correct = result["status"] == "not_found"
    else:
        correct = result["status"] == "answered" and contains_expected(result["answer"], question["expected"])
    return {"status": result["status"], "answer": result["answer"], "correct": correct}


def evaluate_no_context(question):
    try:
        response = llm.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Answer in one short sentence from your own knowledge of "
                       f"The Count of Monte Cristo by Alexandre Dumas.\n\nQuestion: {question['question']}"}],
        )
    except APIError as error:
        return {"answer": f"llm_error: {error.message}", "matches_corpus": False}
    answer = response.choices[0].message.content or ""
    matches = bool(question["expected"]) and contains_expected(answer, question["expected"])
    return {"answer": answer, "matches_corpus": matches}


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval (and optionally answers) on eval/questions.json.")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--answers", action="store_true", help="also run the full pipeline with the LLM")
    parser.add_argument("--no-context", action="store_true", help="also ask the LLM without any retrieval")
    parser.add_argument("--delay", type=float, default=0, help="seconds to wait between LLM calls")
    args = parser.parse_args()

    settings = ChunkSettings(chunk_size=args.chunk_size, overlap_percent=args.overlap)
    sources = sorted(os.listdir(CORPUS_DIR))
    for name in sources:
        if not is_indexed(name, settings):
            print(f"Indexing {name} for {settings.collection_name}...")
            index_document(os.path.join(CORPUS_DIR, name), settings)

    questions = load_questions()
    results = []
    for question in questions:
        row = {"id": question["id"], "type": question["type"], "question": question["question"]}
        if question["evidence"]:
            row["retrieval"] = evaluate_retrieval(question, sources, settings, args.top_k)
        if args.answers:
            row["answer"] = evaluate_answer(question, sources, settings, args.top_k)
            time.sleep(args.delay)
        if args.no_context:
            row["no_context"] = evaluate_no_context(question)
            time.sleep(args.delay)
        results.append(row)
        print_row(row)

    summary = summarize(results)
    print_summary(summary, settings, args.top_k)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{settings.collection_name}_k{args.top_k}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"settings": {"chunk_size": args.chunk_size, "overlap_percent": args.overlap, "top_k": args.top_k},
                   "model": MODEL if args.answers or args.no_context else None,
                   "summary": summary, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {path}")


def print_row(row):
    parts = [f"{row['id']:4} {row['type']:13}"]
    if "retrieval" in row:
        retrieval = row["retrieval"]
        parts.append(f"hit@k={'yes' if retrieval['hit'] else 'NO ':3} rank={retrieval['rank'] or '-'}")
    else:
        parts.append(" " * 17)
    if "answer" in row:
        parts.append(f"answer={row['answer']['status']:16} {'ok' if row['answer']['correct'] else 'WRONG'}")
    if "no_context" in row:
        parts.append(f"no-context={'matches' if row['no_context']['matches_corpus'] else 'differs'}")
    print("  ".join(parts))


def summarize(results):
    def retrieval_stats(rows):
        rows = [row for row in rows if "retrieval" in row]
        if not rows:
            return None
        return {
            "questions": len(rows),
            "hit_at_k": round(sum(row["retrieval"]["hit"] for row in rows) / len(rows), 3),
            "mrr": round(sum(1 / row["retrieval"]["rank"] for row in rows if row["retrieval"]["rank"]) / len(rows), 3),
        }

    by_type = defaultdict(list)
    for row in results:
        by_type[row["type"]].append(row)

    summary = {"retrieval": retrieval_stats(results),
               "retrieval_by_type": {kind: retrieval_stats(rows) for kind, rows in by_type.items() if retrieval_stats(rows)}}
    if any("answer" in row for row in results):
        summary["answer_accuracy"] = round(sum(row["answer"]["correct"] for row in results) / len(results), 3)
        summary["answer_accuracy_by_type"] = {
            kind: round(sum(row["answer"]["correct"] for row in rows) / len(rows), 3) for kind, rows in by_type.items()
        }
    if any("no_context" in row for row in results):
        summary["no_context_matches_corpus"] = sum(row["no_context"]["matches_corpus"] for row in results)
    return summary


def print_summary(summary, settings, top_k):
    print(f"\n{settings.collection_name}, top-k={top_k}")
    retrieval = summary["retrieval"]
    print(f"Retrieval: hit@{top_k}={retrieval['hit_at_k']}  MRR={retrieval['mrr']}  ({retrieval['questions']} answerable questions)")
    for kind, stats in summary["retrieval_by_type"].items():
        print(f"  {kind:13} hit@{top_k}={stats['hit_at_k']}  MRR={stats['mrr']}  (n={stats['questions']})")
    if "answer_accuracy" in summary:
        print(f"Answers: accuracy={summary['answer_accuracy']}  by type={summary['answer_accuracy_by_type']}")
    if "no_context_matches_corpus" in summary:
        print(f"No-context answers matching the corpus: {summary['no_context_matches_corpus']}")


if __name__ == "__main__":
    main()

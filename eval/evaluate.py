import argparse
import os
import sys

from app.evaluation import KS, QUESTION_SETS, check_quotes, load_questions, run_evaluation, run_no_context
from app.indexing import ChunkSettings, index_document, is_indexed
from app.main import corpus_dir, document_names


def ensure_indexed(corpus, settings):
    for name in document_names(corpus):
        if not is_indexed(name, corpus, settings):
            print(f"Indexing {corpus}/{name} for {settings.collection_name(corpus)}...")
            index_document(os.path.join(corpus_dir(corpus), name), corpus, settings)


def print_summary(summary):
    header = "  ".join(f"hit@{k:<2}" for k in KS) + "   " + "  ".join(f"rec@{k:<2}" for k in KS)
    print(f"{'type':16} {'n':>3}  {header}")
    for kind, stats in [("overall", summary["overall"]), *summary["by_type"].items()]:
        values = "  ".join(f"{stats[f'hit@{k}']:6.2f}" for k in KS) + "   " + \
                 "  ".join(f"{stats[f'recall@{k}']:6.2f}" for k in KS)
        print(f"{kind:16} {stats['questions']:>3}  {values}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval (and optionally answers) on a question set.")
    parser.add_argument("--corpus", choices=[*QUESTION_SETS, "all"], default="all")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=10)
    parser.add_argument("--answers", action="store_true", help="also generate answers with the LLM")
    parser.add_argument("--answer-top-k", type=int, default=5)
    parser.add_argument("--no-context", action="store_true", help="only run the no-context test (LLM without retrieval)")
    parser.add_argument("--delay", type=float, default=0, help="seconds to wait between LLM calls")
    args = parser.parse_args()

    settings = ChunkSettings(chunk_size=args.chunk_size, overlap_percent=args.overlap)
    for corpus in QUESTION_SETS if args.corpus == "all" else [args.corpus]:
        problems = check_quotes(load_questions(corpus), corpus_dir(corpus))
        if problems:
            sys.exit("\n".join(problems))

        if args.no_context:
            path, rows = run_no_context(corpus, args.delay, on_progress=lambda row: print(
                f"{row['id']:4} {row['type']:16} {row['judge_correct']!s:5}  {(row['answer'] or '')[:70]!r}"))
            print(f"{corpus}: {sum(row['judge_correct'] is True for row in rows)} answered correctly without "
                  f"retrieval. Saved {path}\n")
            continue

        ensure_indexed(corpus, settings)
        print(f"\n{corpus}, {settings.collection_name(corpus)}")
        path, report = run_evaluation(
            corpus, document_names(corpus), settings, args.answers, args.answer_top_k, args.delay,
            on_progress=lambda row: args.answers and print(f"  {row['id']:4} {row['answer']['status']}"))
        print_summary(report["summary"])
        print(f"Saved {path}")


if __name__ == "__main__":
    main()

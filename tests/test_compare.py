from app.evaluation import average_cost
from eval.compare import mark, question_cell, type_rows


def retrieval(ranks, recall5):
    return {"evidence_ranks": ranks, "recall": {"5": recall5}}


def test_mark_follows_recall_at_5_only():
    base = retrieval([2, 6], 0.5)
    assert mark(base, retrieval([1, 4], 1.0)) == " ▲"
    assert mark(base, retrieval([7, None], 0.0)) == " ▼"
    assert mark(base, retrieval([3, 9], 0.5)) == ""
    assert mark(None, base) == ""


def test_question_cell_shows_every_hop():
    assert question_cell(retrieval([6, None], 0.0), retrieval([1, None], 0.5)) == "6/— · R@5 0.00 ▼"


def test_type_rows_keep_unanswerable_types_with_dashes():
    stats = {"questions": 1, **{f"hit@{k}": 1.0 for k in (1, 3, 5, 10)}, **{f"recall@{k}": 0.5 for k in (1, 3, 5, 10)}}
    report = {"summary": {"overall": stats, "by_type": {"two_hop": stats}},
              "results": [{"type": "two_hop"}, {"type": "not_in_corpus"}, {"type": "not_in_corpus"}]}
    rows = type_rows("kestrel", "dense", report)
    assert [row[2:4] for row in rows] == [["**overall**", 1], ["two_hop", 1], ["not_in_corpus", 2]]
    assert rows[2][4:] == ["—"] * 8


def test_average_cost_per_retriever():
    costs = [{"retrieval_ms": 10.0, "retriever_ms": {"dense": 8.0, "bm25": 1.0}, "fusion_ms": 1.0, "llm_calls": 1,
              "prompt_tokens": 100, "completion_tokens": 10, "llm_ms": 500.0},
             {"retrieval_ms": 20.0, "retriever_ms": {"dense": 18.0}, "fusion_ms": 2.0, "llm_calls": 1,
              "prompt_tokens": 300, "completion_tokens": 30, "llm_ms": 700.0}]
    average = average_cost(costs)
    assert average["questions"] == 2
    assert average["retrieval_ms"] == 15.0 and average["prompt_tokens"] == 200.0 and average["llm_ms"] == 600.0
    assert average["retriever_ms"] == {"dense": 13.0, "bm25": 0.5}
    assert average_cost([]) is None

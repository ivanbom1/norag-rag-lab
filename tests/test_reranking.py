import pytest

from app import reranking


class FakeModel:
    def __init__(self, scores):
        self.scores = scores
        self.pairs = None

    def predict(self, pairs):
        self.pairs = pairs
        return [self.scores[text] for _, text in pairs]


def chunk(chunk_id, text, **found_by):
    return {"id": chunk_id, "text": text, "found_by": found_by or {"dense": 1}}


def test_rerank_orders_by_score_and_keeps_top_k():
    chunks = [chunk("a", "low"), chunk("b", "high"), chunk("c", "mid")]
    model = FakeModel({"low": 0.1, "high": 5.0, "mid": 2.0})
    result = reranking.rerank("question", chunks, 2, model=model)
    assert [c["id"] for c in result] == ["b", "c"]
    assert model.pairs == [("question", "low"), ("question", "high"), ("question", "mid")]


def test_rerank_adds_its_rank_and_score_without_losing_retriever_ranks():
    chunks = [chunk("a", "x", dense=1, bm25=4), chunk("b", "y", bm25=1)]
    result = reranking.rerank("q", chunks, 5, model=FakeModel({"x": -1.0, "y": 3.5}))
    assert result[0]["found_by"] == {"bm25": 1, "rerank": 1}
    assert result[1]["found_by"] == {"dense": 1, "bm25": 4, "rerank": 2}
    assert result[0]["rerank_score"] == 3.5
    assert chunks[0]["found_by"] == {"dense": 1, "bm25": 4}


def test_rerank_of_nothing_is_nothing():
    assert reranking.rerank("q", [], 5, model=FakeModel({})) == []


@pytest.fixture
def fake_retrievers(monkeypatch):
    from app import answering
    lists = {
        "dense": [chunk(f"d{i}", f"dense {i}") for i in range(20)],
        "bm25": [chunk(f"b{i}", f"bm25 {i}") for i in range(20)],
    }
    monkeypatch.setattr(answering, "RETRIEVERS", {name: (lambda q, c, s, st, k, name=name: lists[name][:k]) for name in lists})
    return answering


def test_retrieve_with_rerank_scores_more_candidates_than_top_k(fake_retrievers, monkeypatch):
    seen = {}

    def fake_rerank(question, candidates, top_k):
        seen["candidates"] = len(candidates)
        return list(reversed(candidates))[:top_k]

    monkeypatch.setattr(reranking, "rerank", fake_rerank)
    chunks, cost = fake_retrievers.retrieve_with_cost("q", "kestrel", [], None, 5, ["dense", "bm25"], rerank=True)
    assert seen["candidates"] == reranking.RERANK_CANDIDATES
    assert len(chunks) == 5
    assert [c["label"] for c in chunks] == ["C1", "C2", "C3", "C4", "C5"]
    assert cost["rerank_candidates"] == reranking.RERANK_CANDIDATES
    assert "rerank_error" not in cost


def test_retrieve_falls_back_to_search_order_when_reranking_fails(fake_retrievers, monkeypatch):
    def broken(question, candidates, top_k):
        raise RuntimeError("model not downloaded")

    monkeypatch.setattr(reranking, "rerank", broken)
    reranked, cost = fake_retrievers.retrieve_with_cost("q", "kestrel", [], None, 5, ["dense", "bm25"], rerank=True)
    plain, _ = fake_retrievers.retrieve_with_cost("q", "kestrel", [], None, 5, ["dense", "bm25"], rerank=False)
    assert [c["id"] for c in reranked] == [c["id"] for c in plain]
    assert "model not downloaded" in cost["rerank_error"]


def test_retrieve_without_rerank_does_not_call_the_model(fake_retrievers, monkeypatch):
    monkeypatch.setattr(reranking, "rerank", lambda *args: pytest.fail("rerank was called"))
    _, cost = fake_retrievers.retrieve_with_cost("q", "kestrel", [], None, 5, ["dense", "bm25"], rerank=False)
    assert cost["rerank_ms"] == 0.0

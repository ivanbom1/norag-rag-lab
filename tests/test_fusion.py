from app.fusion import rrf


def chunk(chunk_id, **scores):
    return {"id": chunk_id, "text": chunk_id, **scores}


def test_one_list_keeps_its_order():
    fused = rrf({"dense": [chunk("a"), chunk("b"), chunk("c")]})
    assert [c["id"] for c in fused] == ["a", "b", "c"]


def test_chunk_found_by_both_beats_chunks_found_by_one():
    fused = rrf({
        "dense": [chunk("a"), chunk("b"), chunk("c")],
        "bm25": [chunk("x"), chunk("y"), chunk("c")],
    })
    assert fused[0]["id"] == "c"
    assert fused[0]["found_by"] == {"dense": 3, "bm25": 3}


def test_ranks_and_scores_of_each_retriever_are_kept():
    fused = rrf({
        "dense": [chunk("a", distance=0.4), chunk("b", distance=0.6)],
        "bm25": [chunk("b", score=7.1)],
    })
    b = next(c for c in fused if c["id"] == "b")
    assert b["found_by"] == {"dense": 2, "bm25": 1}
    assert (b["distance"], b["score"]) == (0.6, 7.1)
    assert b["rrf"] == round(1 / 62 + 1 / 61, 5)


def test_ties_are_stable():
    # a and b both score 1/61 + 1/62 and both have a #1, so the id decides
    fused = rrf({"dense": [chunk("b"), chunk("a")], "bm25": [chunk("a"), chunk("b")]})
    assert [c["id"] for c in fused] == ["a", "b"]


def test_empty_lists():
    assert rrf({"dense": [], "bm25": []}) == []

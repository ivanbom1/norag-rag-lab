RRF_K = 60


def rrf(lists, k=RRF_K):
    """Reciprocal rank fusion: a chunk scores 1 / (k + rank) in every list it appears in.

    lists: {retriever name: ranked chunks}. Each fused chunk keeps the ranks it had per retriever
    in found_by, plus the retrievers' own scores (distance for dense, score for bm25).
    """
    fused = {}
    for name, chunks in lists.items():
        for rank, chunk in enumerate(chunks, start=1):
            entry = fused.setdefault(chunk["id"], {**chunk, "found_by": {}, "rrf": 0.0})
            entry.update({key: chunk[key] for key in ("distance", "score") if key in chunk})
            entry["found_by"][name] = rank
            entry["rrf"] += 1 / (k + rank)
    ranked = sorted(fused.values(), key=lambda c: (-c["rrf"], min(c["found_by"].values()), c["id"]))
    for chunk in ranked:
        chunk["rrf"] = round(chunk["rrf"], 5)
    return ranked

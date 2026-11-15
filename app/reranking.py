RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_CANDIDATES = 30
DEFAULT_RERANK = False

_model = None


def load_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder(RERANK_MODEL, max_length=512)
    return _model


def rerank(question, chunks, top_k, model=None):
    if not chunks:
        return []
    scores = (model or load_model()).predict([(question, chunk["text"]) for chunk in chunks])
    order = sorted(range(len(chunks)), key=lambda i: -float(scores[i]))[:top_k]
    return [
        {**chunks[i], "rerank_score": round(float(scores[i]), 3), "found_by": {**chunks[i]["found_by"], "rerank": rank}}
        for rank, i in enumerate(order, start=1)
    ]

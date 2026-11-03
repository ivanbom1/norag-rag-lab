import re
from collections import Counter

from app.evaluation import is_relevant
from app.indexing import get_collection

# Letters/digits joined by an inner - . ' or ’, so KST-2044, 203.0.113.10 and captain’s stay one word.
TOKEN = re.compile(r"[^\W_]+(?:[-.'’][^\W_]+)*")

STOPWORDS = set("""
a about after all also am an and any are as at be been before but by can could did do does for from had has have
he her here him his how i if in into is it its me my no not of on or our she so than that the their them then there
these they this those to up us was we were what when where which who whom why will with would you your
""".split())

_cache = {}


def tokenize(text):
    return [token.lower() for token in TOKEN.findall(text)]


def chunk_index(corpus, settings):
    collection = get_collection(corpus, settings)
    key = (collection.name, collection.count())
    if _cache.get(corpus, {}).get("key") != key:
        data = collection.get(include=["documents", "metadatas"])
        chunks = [{"id": chunk_id, "text": text, **metadata}
                  for chunk_id, text, metadata in zip(data["ids"], data["documents"], data["metadatas"])]
        chunks.sort(key=lambda chunk: (chunk["source"], chunk["chunk_index"]))
        counts = [Counter(tokenize(chunk["text"])) for chunk in chunks]
        vocabulary = Counter()
        chunk_frequency = Counter()
        for counter in counts:
            vocabulary.update(counter)
            chunk_frequency.update(counter.keys())
        _cache[corpus] = {"key": key, "chunks": chunks, "counts": counts,
                          "vocabulary": vocabulary, "chunk_frequency": chunk_frequency}
    return _cache[corpus]


def vocabulary(corpus, settings, contains="", show_stopwords=False, limit=200):
    index = chunk_index(corpus, settings)
    words = [(word, count) for word, count in index["vocabulary"].most_common()
             if contains.lower() in word and (show_stopwords or word not in STOPWORDS)]
    return {
        "chunks": len(index["chunks"]),
        "tokens": sum(index["vocabulary"].values()),
        "distinct_words": len(index["vocabulary"]),
        "matching_words": len(words),
        "words": [{"word": word, "count": count, "chunks": index["chunk_frequency"][word]} for word, count in words[:limit]],
    }


def word_chunks(corpus, settings, word, limit=100):
    index = chunk_index(corpus, settings)
    word = word.lower()
    hits = sorted(
        ({"id": chunk["id"], "source": chunk["source"], "start": chunk["start"], "end": chunk["end"],
          "count": counter[word], "text": chunk["text"]}
         for chunk, counter in zip(index["chunks"], index["counts"]) if counter[word]),
        key=lambda hit: -hit["count"],
    )
    return {"word": word, "count": index["vocabulary"][word], "chunk_count": len(hits), "chunks": hits[:limit]}


def question_words(corpus, settings, question, evidence=None):
    index = chunk_index(corpus, settings)
    evidence_chunks = [
        (chunk, counter) for chunk, counter in zip(index["chunks"], index["counts"])
        if any(is_relevant(chunk, item) for item in evidence or [])
    ]
    words = []
    for word in dict.fromkeys(tokenize(question)):
        words.append({
            "word": word,
            "stopword": word in STOPWORDS,
            "count": index["vocabulary"][word],
            "chunks": index["chunk_frequency"][word],
            "in_evidence_chunks": [chunk["id"] for chunk, counter in evidence_chunks if counter[word]],
        })
    return {
        "total_chunks": len(index["chunks"]),
        "words": words,
        "absent": [entry["word"] for entry in words if entry["count"] == 0],
        "evidence_chunks": [{"id": chunk["id"], "source": chunk["source"]} for chunk, _ in evidence_chunks],
    }

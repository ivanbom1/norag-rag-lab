"""BM25 retriever: the course's BM25Index scoring (ask_bm25_rag_google.py), on an index stored in SQLite.

Where the index lives: one SQLite file (BM25_PATH), next to chroma_db/. Each corpus x chunk setting is its own
index (table_name), the same layout as the Chroma collections, so word statistics are never mixed across corpora.
For each index it stores the chunks with their length in words, and the postings: how many times each word
appears in each chunk.

How it is built: index_document() in app/indexing.py writes the same chunks, with the same ids, to Chroma
and to this index in one pass. Indexes that exist only in Chroma are copied here at startup, without
re-embedding. Deleting a document removes its chunks from both.

Tokenization: app.tokens.tokenize (the Words tab tokenizer: lowercase, KST-2044 and 203.0.113.10 kept whole).
Ranking: the same formula as BM25Index in the course script, with k1 = 1.5 and b = 0.75.
"""
import math
import os
import sqlite3
from collections import Counter
from contextlib import closing

from app.tokens import STOPWORDS, tokenize

BM25_PATH = "./bm25_db/bm25.sqlite3"
BM25_K1 = 1.5
BM25_B = 0.75

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    index_name TEXT, corpus TEXT, id TEXT, source TEXT, chunk_index INTEGER, start INTEGER, "end" INTEGER,
    text TEXT, length INTEGER, PRIMARY KEY (index_name, id));
CREATE INDEX IF NOT EXISTS chunks_by_source ON chunks (index_name, source);
CREATE TABLE IF NOT EXISTS postings (
    index_name TEXT, word TEXT, chunk_id TEXT, count INTEGER, PRIMARY KEY (index_name, word, chunk_id)) WITHOUT ROWID;
"""


def table_name(corpus, settings):
    # BM25 does not depend on the embedding model, so unlike the Chroma collection there is no "_minilm".
    return f"{corpus}_{settings.chunk_size}_{settings.overlap_percent}_{settings.min_paragraph_length}"


def connect():
    os.makedirs(os.path.dirname(BM25_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(BM25_PATH)
    conn.executescript(SCHEMA)
    return closing(conn)


def _delete_chunks(conn, where, params):
    conn.execute(f"DELETE FROM postings WHERE (index_name, chunk_id) IN (SELECT index_name, id FROM chunks WHERE {where})",
                 params)
    conn.execute(f"DELETE FROM chunks WHERE {where}", params)


def index_chunks(source, corpus, settings, chunks):
    """Replace the chunks of one document. Each chunk: {id, text, chunk_index, start, end}."""
    name = table_name(corpus, settings)
    counts = [Counter(tokenize(c["text"])) for c in chunks]
    with connect() as conn, conn:
        _delete_chunks(conn, "index_name = ? AND source = ?", (name, source))
        conn.executemany(
            'INSERT INTO chunks (index_name, corpus, id, source, chunk_index, start, "end", text, length) '
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(name, corpus, c["id"], source, c["chunk_index"], c["start"], c["end"], c["text"], sum(n.values()))
             for c, n in zip(chunks, counts)],
        )
        conn.executemany(
            "INSERT INTO postings (index_name, word, chunk_id, count) VALUES (?, ?, ?, ?)",
            [(name, word, c["id"], count) for c, n in zip(chunks, counts) for word, count in n.items()],
        )
    return len(chunks)


def is_indexed(source, corpus, settings):
    with connect() as conn:
        return conn.execute("SELECT 1 FROM chunks WHERE index_name = ? AND source = ? LIMIT 1",
                            (table_name(corpus, settings), source)).fetchone() is not None


def delete_document(source, corpus):
    with connect() as conn, conn:
        _delete_chunks(conn, "corpus = ? AND source = ?", (corpus, source))


def query_terms(question):
    """Distinct question tokens without stopwords (all tokens if the question is only stopwords)."""
    terms = list(dict.fromkeys(tokenize(question)))
    return [term for term in terms if term not in STOPWORDS] or terms


def idf(n, df):
    return math.log((n - df + 0.5) / (df + 0.5) + 1)


def search(question, corpus, sources, settings, top_k):
    terms = query_terms(question)
    name = table_name(corpus, settings)
    if not terms or not sources:
        return []
    words = ", ".join("?" for _ in terms)
    with connect() as conn:
        # Word statistics come from the whole index, like BM25Index; the source filter only applies to the results.
        n, average_length = conn.execute("SELECT COUNT(*), AVG(length) FROM chunks WHERE index_name = ?",
                                         (name,)).fetchone()
        if not n:
            return []
        weights = {word: idf(n, df) for word, df in conn.execute(
            f"SELECT word, COUNT(*) FROM postings WHERE index_name = ? AND word IN ({words}) GROUP BY word",
            (name, *terms))}
        rows = conn.execute(
            f'SELECT c.id, c.text, c.source, c.chunk_index, c.start, c."end", c.length, p.word, p.count '
            f"FROM postings p JOIN chunks c ON c.index_name = p.index_name AND c.id = p.chunk_id "
            f"WHERE p.index_name = ? AND p.word IN ({words}) AND c.source IN ({', '.join('?' for _ in sources)})",
            (name, *terms, *sources),
        ).fetchall()

    found = {}
    for chunk_id, text, source, chunk_index, start, end, length, word, tf in rows:
        chunk = found.setdefault(chunk_id, {"id": chunk_id, "text": text, "score": 0.0, "source": source,
                                            "chunk_index": chunk_index, "start": start, "end": end})
        length_ratio = length / average_length
        chunk["score"] += weights[word] * tf * (BM25_K1 + 1) / (tf + BM25_K1 * (1 - BM25_B + BM25_B * length_ratio))
    ranked = sorted(found.values(), key=lambda c: (-c["score"], c["source"], c["chunk_index"]))[:top_k]
    for chunk in ranked:
        chunk["score"] = round(chunk["score"], 3)
    return ranked


def vocabulary(corpus, settings):
    """{token: total count} as stored in the index; used to check that it tokenizes exactly like app.tokens."""
    with connect() as conn:
        return dict(conn.execute("SELECT word, SUM(count) FROM postings WHERE index_name = ? GROUP BY word",
                                 (table_name(corpus, settings),)))

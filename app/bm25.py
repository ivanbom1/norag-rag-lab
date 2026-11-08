"""BM25 retriever backed by SQLite FTS5.

Where the index lives: one SQLite file (BM25_PATH), next to chroma_db/, with one FTS5 table per
corpus x chunk setting (the same layout as the Chroma collections, so IDF is never mixed across corpora).

How it is built: index_document() in app/indexing.py writes the same chunks, with the same ids, to Chroma
and to this index in one pass. Indexes that exist only in Chroma are copied here at startup, without
re-embedding. Deleting a document removes its chunks from both.

Tokenization: the text is tokenized in Python with app.tokens.tokenize (the Words tab tokenizer: lowercase,
KST-2044 and 203.0.113.10 kept whole) and stored as space-separated tokens. FTS5's unicode61 tokenizer is told
that - . ' ’ are token characters, so it only splits on the spaces and sees exactly our tokens.
Ranking: FTS5's built-in bm25() (k1 = 1.2, b = 0.75). It returns lower-is-better values; we expose -bm25().
"""
import os
import sqlite3
from contextlib import closing

from app.tokens import STOPWORDS, tokenize

BM25_PATH = "./bm25_db/bm25.sqlite3"
FTS_TOKENIZER = "unicode61 remove_diacritics 0 tokenchars '-.''’'"


def table_name(corpus, settings):
    # BM25 does not depend on the embedding model, so unlike the Chroma collection there is no "_minilm".
    return f"{corpus}_{settings.chunk_size}_{settings.overlap_percent}_{settings.min_paragraph_length}"


def connect():
    os.makedirs(os.path.dirname(BM25_PATH) or ".", exist_ok=True)
    return closing(sqlite3.connect(BM25_PATH))


def _ensure_table(conn, name):
    conn.execute(
        f'CREATE VIRTUAL TABLE IF NOT EXISTS "{name}" USING fts5('
        f'tokens, id UNINDEXED, source UNINDEXED, chunk_index UNINDEXED, start UNINDEXED, "end" UNINDEXED, '
        f'text UNINDEXED, tokenize = "{FTS_TOKENIZER}")'
    )


def _tables(conn):
    return [row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND sql LIKE 'CREATE VIRTUAL TABLE%'")]


def index_chunks(source, corpus, settings, chunks):
    """Replace the chunks of one document. Each chunk: {id, text, chunk_index, start, end}."""
    name = table_name(corpus, settings)
    with connect() as conn, conn:
        _ensure_table(conn, name)
        conn.execute(f'DELETE FROM "{name}" WHERE source = ?', (source,))
        conn.executemany(
            f'INSERT INTO "{name}" (tokens, id, source, chunk_index, start, "end", text) VALUES (?, ?, ?, ?, ?, ?, ?)',
            [(" ".join(tokenize(c["text"])), c["id"], source, c["chunk_index"], c["start"], c["end"], c["text"])
             for c in chunks],
        )
    return len(chunks)


def is_indexed(source, corpus, settings):
    name = table_name(corpus, settings)
    with connect() as conn:
        if name not in _tables(conn):
            return False
        return conn.execute(f'SELECT 1 FROM "{name}" WHERE source = ? LIMIT 1', (source,)).fetchone() is not None


def delete_document(source, corpus):
    with connect() as conn, conn:
        for name in _tables(conn):
            if name.startswith(f"{corpus}_"):
                conn.execute(f'DELETE FROM "{name}" WHERE source = ?', (source,))


def query_terms(question):
    """Distinct question tokens without stopwords (all tokens if the question is only stopwords)."""
    terms = list(dict.fromkeys(tokenize(question)))
    return [term for term in terms if term not in STOPWORDS] or terms


def search(question, corpus, sources, settings, top_k):
    terms = query_terms(question)
    name = table_name(corpus, settings)
    if not terms or not sources:
        return []
    # Tokens never contain a double quote, so each one can be passed as an FTS5 string ("kst-2044").
    match = "tokens : (" + " OR ".join(f'"{term}"' for term in terms) + ")"
    placeholders = ", ".join("?" for _ in sources)
    with connect() as conn:
        if name not in _tables(conn):
            return []
        rows = conn.execute(
            f'SELECT id, text, source, chunk_index, start, "end", bm25("{name}") AS rank FROM "{name}" '
            f'WHERE "{name}" MATCH ? AND source IN ({placeholders}) ORDER BY rank, source, chunk_index LIMIT ?',
            (match, *sources, top_k),
        ).fetchall()
    return [
        {"id": chunk_id, "text": text, "score": round(-rank, 3),
         "source": source, "chunk_index": chunk_index, "start": start, "end": end}
        for chunk_id, text, source, chunk_index, start, end, rank in rows
    ]


def vocabulary(corpus, settings):
    """{token: total count} as FTS5 sees it; used to check that it tokenizes exactly like app.tokens."""
    name = table_name(corpus, settings)
    with connect() as conn:
        conn.execute(f'CREATE VIRTUAL TABLE IF NOT EXISTS temp."{name}_vocab" USING fts5vocab(main, "{name}", row)')
        return dict(conn.execute(f'SELECT term, cnt FROM temp."{name}_vocab"'))

from collections import Counter
from types import SimpleNamespace

import pytest

from app import bm25
from app.tokens import tokenize

SETTINGS = SimpleNamespace(chunk_size=800, overlap_percent=10, min_paragraph_length=20)
OTHER_SETTINGS = SimpleNamespace(chunk_size=400, overlap_percent=10, min_paragraph_length=20)

RUNBOOK = [
    "Error KST-2040 means the data source timed out. Retry the connection later.",
    "Error KST-2044: the Kestrel addresses are missing from it. Add both 203.0.113.10 and 203.0.113.11.",
    "Dashboards refresh every hour. The Captain’s desk can't LOG in to the Château d’If portal.",
]


def chunks(texts, source):
    return [{"id": f"{source}#{i}", "text": text, "chunk_index": i, "start": 0, "end": len(text)}
            for i, text in enumerate(texts)]


@pytest.fixture(autouse=True)
def index_file(tmp_path, monkeypatch):
    monkeypatch.setattr(bm25, "BM25_PATH", str(tmp_path / "bm25.sqlite3"))
    bm25.index_chunks("runbook.md", "kestrel", SETTINGS, chunks(RUNBOOK, "runbook.md"))


def test_fts5_tokenizes_exactly_like_the_words_tab():
    expected = Counter(token for text in RUNBOOK for token in tokenize(text))
    assert bm25.vocabulary("kestrel", SETTINGS) == dict(expected)


def test_exact_identifier_ranks_first():
    results = bm25.search("What does error KST-2044 mean?", "kestrel", ["runbook.md"], SETTINGS, 5)
    assert results[0]["id"] == "runbook.md#1"
    assert results[0]["score"] > results[1]["score"]


def test_ip_address_is_one_token():
    results = bm25.search("203.0.113.10", "kestrel", ["runbook.md"], SETTINGS, 5)
    assert [r["id"] for r in results] == ["runbook.md#1"]


def test_unknown_words_return_nothing():
    assert bm25.search("zebra", "kestrel", ["runbook.md"], SETTINGS, 5) == []


def test_stopwords_are_dropped_unless_nothing_else_is_left():
    assert bm25.query_terms("What is the KST-2044 error?") == ["kst-2044", "error"]
    assert bm25.query_terms("what is the") == ["what", "is", "the"]


def test_source_filter():
    bm25.index_chunks("guide.md", "kestrel", SETTINGS, chunks(["Error KST-2044 in the guide."], "guide.md"))
    results = bm25.search("KST-2044", "kestrel", ["guide.md"], SETTINGS, 5)
    assert [r["source"] for r in results] == ["guide.md"]


def test_reindexing_a_document_replaces_its_chunks():
    bm25.index_chunks("runbook.md", "kestrel", SETTINGS, chunks(RUNBOOK, "runbook.md"))
    results = bm25.search("KST-2044", "kestrel", ["runbook.md"], SETTINGS, 10)
    assert [r["id"] for r in results] == ["runbook.md#1"]


def test_is_indexed_per_setting():
    assert bm25.is_indexed("runbook.md", "kestrel", SETTINGS)
    assert not bm25.is_indexed("runbook.md", "kestrel", OTHER_SETTINGS)
    assert not bm25.is_indexed("runbook.md", "monte-cristo", SETTINGS)


def test_delete_removes_the_document_from_every_setting_of_its_corpus_only():
    bm25.index_chunks("runbook.md", "kestrel", OTHER_SETTINGS, chunks(RUNBOOK, "runbook.md"))
    bm25.index_chunks("runbook.md", "monte-cristo", SETTINGS, chunks(RUNBOOK, "runbook.md"))
    bm25.delete_document("runbook.md", "kestrel")
    assert not bm25.is_indexed("runbook.md", "kestrel", SETTINGS)
    assert not bm25.is_indexed("runbook.md", "kestrel", OTHER_SETTINGS)
    assert bm25.is_indexed("runbook.md", "monte-cristo", SETTINGS)

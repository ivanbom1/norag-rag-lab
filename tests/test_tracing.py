import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from app import answering, llm, tracing


@pytest.fixture(autouse=True)
def trace_db(tmp_path, monkeypatch):
    monkeypatch.setattr(tracing, "TRACE_DB", str(tmp_path / "traces" / "traces.sqlite3"))


def test_saved_trace_can_be_read_back_and_listed():
    trace = tracing.Trace("question", corpus="kestrel")
    trace.add_span("dense", 12.5, chunk_ids=["a.md#0"])
    trace_id = tracing.save(trace, "answered")
    saved = tracing.get(trace_id)
    assert saved["status"] == "answered" and saved["corpus"] == "kestrel"
    assert saved["spans"] == [{"name": "dense", "duration_ms": 12.5, "chunk_ids": ["a.md#0"]}]
    assert tracing.recent("question")[0]["trace_id"] == trace_id
    assert tracing.recent("indexing") == []


def test_traces_older_than_the_retention_are_deleted():
    old = tracing.Trace("question")
    old.data["created"] = (datetime.now() - timedelta(days=tracing.RETENTION_DAYS + 1)).isoformat(timespec="seconds")
    tracing.save(old, "answered")
    kept = tracing.save(tracing.Trace("question"), "answered")
    assert tracing.delete_old() == 1
    assert tracing.get(old.id) is None and tracing.get(kept)


def test_llm_cost_uses_the_list_price():
    assert tracing.llm_cost("gemini-3.5-flash-lite", 1_000_000, 100_000) == 0.30 + 0.25
    assert tracing.llm_cost("unknown-model", 1000, 1000) == 0


CHUNK = {"id": "contacts.md#0", "source": "contacts.md", "text": "The billing contact is marie@example.com.",
         "found_by": {"dense": 1}}


def test_ask_sends_no_personal_data_and_traces_it_redacted(monkeypatch):
    sent = []

    def fake_llm(**kwargs):
        sent.append(kwargs["messages"][0]["content"])
        reply = {"answerable": True, "answer": "Write to [EMAIL_1].",
                 "citations": [{"chunk_id": "C1", "quote": "The billing contact is [EMAIL_1]."}]}
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(reply)))],
                               usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=50))

    def fake_retrieve(question, corpus, sources, settings, top_k, retrievers, rerank, trace):
        trace.add_span("dense", 5.0, chunk_ids=[CHUNK["id"]])
        return [{"label": "C1", **CHUNK}], answering.new_cost()

    monkeypatch.setattr(llm.client.chat.completions, "create", fake_llm)
    monkeypatch.setattr(answering, "retrieve_with_cost", fake_retrieve)
    trace = tracing.Trace("question")
    result = answering.ask("Who is marie@example.com?", "kestrel", ["contacts.md"], None, 5, trace=trace)

    assert "marie@example.com" not in sent[0] and "[EMAIL_1]" in sent[0]
    assert result["status"] == "answered" and result["answer"] == "Write to marie@example.com."
    assert result["citations"][0]["verified"]
    saved = tracing.get(tracing.save(trace, result["status"]))
    assert "marie@example.com" not in json.dumps(saved)
    assert saved["question"] == "Who is [EMAIL_1]?" and saved["answer"] == "Write to [EMAIL_1]."
    assert [span["name"] for span in saved["spans"]] == ["dense", "redact", "llm"]
    assert saved["spans"][1]["replaced"] == {"EMAIL": 1}
    assert saved["spans"][2]["prompt_tokens"] == 1000 and saved["spans"][2]["cost"] == tracing.llm_cost(
        answering.MODEL, 1000, 50)

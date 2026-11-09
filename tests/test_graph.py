import json

import httpx
from openai import RateLimitError

from app import graph

CHUNK = "### postroom\n\nSends every e-mail that leaves Kestrel.\nOwner: **Messaging**."


def triple(subject, subject_type, relation, obj, object_type, quote="Owner: **Messaging**."):
    return {"subject": subject, "subject_type": subject_type, "relation": relation,
            "object": obj, "object_type": object_type, "quote": quote}


def test_triples_outside_the_schema_are_dropped():
    kept, dropped = graph.validate_triples([
        triple("postroom", "Service", "owned_by", "Messaging", "Team"),
        triple("KST-5031", "ErrorCode", "owned_by", "Messaging", "Team"),
        triple("postroom", "Service", "sends", "e-mail", "Feature"),
    ], CHUNK)
    assert [t["subject"] for t in kept] == ["postroom"]
    assert len(dropped) == 2


def test_quote_is_checked_against_the_chunk():
    kept, _ = graph.validate_triples([
        triple("postroom", "Service", "owned_by", "Messaging", "Team"),
        triple("postroom", "Service", "owned_by", "Messaging", "Team", quote="postroom is run by Messaging"),
    ], CHUNK)
    assert [t["quote_verified"] for t in kept] == [True, False]


def test_names_are_normalized_per_type():
    assert graph.node_id("Analytics Engine team", "Team") == graph.node_id("the analytics engine", "Team")
    assert graph.node_id("oncall-messaging", "Channel") == "Channel:#oncall-messaging"
    assert graph.node_id("KST–2044", "ErrorCode") == graph.node_id("KST-2044", "ErrorCode")
    assert graph.node_id("Analytics Engine", "Team") != graph.node_id("Analytics Engine", "Service")


def test_resolve_merges_names_and_keeps_provenance():
    rows = [
        {"chunk_id": "catalog#1", "triples": [
            {**triple("postroom", "Service", "owned_by", "Messaging", "Team"), "quote_verified": True}]},
        {"chunk_id": "catalog#5", "triples": [
            {**triple("Messaging team", "Team", "on_call_channel", "#oncall-messaging", "Channel"), "quote_verified": True}]},
    ]
    nodes, edges, merges = graph.resolve(rows)
    messaging = next(node for node in nodes if node["id"] == "Team:messaging")
    assert messaging["aliases"] == ["Messaging", "Messaging team"]
    assert messaging["chunks"] == ["catalog#1", "catalog#5"]
    assert merges == [{"node": "Team:messaging", "names": ["Messaging", "Messaging team"]}]
    assert [(e["source"], e["relation"], e["target"], e["chunks"]) for e in edges] == [
        ("Service:postroom", "owned_by", "Team:messaging", ["catalog#1"]),
        ("Team:messaging", "on_call_channel", "Channel:#oncall-messaging", ["catalog#5"]),
    ]


def test_heading_context_gives_the_short_headings_the_chunk_lost():
    document = "# Catalog\n\n## 1. Services\n\n### auth\n\nIssues tokens.\nOwner: **Identity**.\n\n### postroom\n\nSends e-mail."
    start = document.index("Issues")
    context = graph.heading_context(document, start, len(document))
    assert "starts in: Catalog > 1. Services > auth" in context
    assert '### postroom (followed by: "Sends e-mail.…")' in context


def test_invalid_response_is_retried_once_then_the_chunk_fails(monkeypatch):
    replies = iter(["not json", json.dumps({"triples": [triple("postroom", "Service", "owned_by", "Messaging", "Team")]})])
    monkeypatch.setattr(graph, "call_llm", lambda messages, model, cost: next(replies))
    row = graph.extract({"id": "catalog#1", "source": "service-catalog.md", "text": CHUNK}, "Service Catalog", "m")
    assert row["status"] == "ok" and len(row["triples"]) == 1

    monkeypatch.setattr(graph, "call_llm", lambda messages, model, cost: "still not json")
    row = graph.extract({"id": "catalog#1", "source": "service-catalog.md", "text": CHUNK}, "Service Catalog", "m")
    assert row["status"] == "failed" and row["error"].startswith("invalid response")


def test_daily_quota_is_not_retried(monkeypatch):
    calls = []

    def quota_exceeded(**kwargs):
        calls.append(kwargs)
        response = httpx.Response(429, request=httpx.Request("POST", "https://example.test"))
        raise RateLimitError("GenerateRequestsPerDayPerProjectPerModel-FreeTier", response=response, body=None)

    monkeypatch.setattr(graph.llm.chat.completions, "create", quota_exceeded)
    row = graph.extract({"id": "service-catalog.md#1", "source": "service-catalog.md", "text": CHUNK}, "Service catalog",
                        graph.GRAPH_MODEL)
    assert row["status"] == "failed" and "llm error" in row["error"]
    assert len(calls) == 1 and row["cost"]["llm_calls"] == 1

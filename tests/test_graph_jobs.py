import json
import os
import time

from app import graph, graph_jobs, graph_retriever
from app.indexing import ChunkSettings


class FakeCollection:
    def count(self):
        return 3


def wait_until_not_building(corpus, settings, path):
    for _ in range(100):
        info = graph_jobs.status(corpus, settings, path)
        if info["state"] != "building":
            return info
        time.sleep(0.02)
    raise AssertionError("still building")


def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(graph_retriever, "GRAPH_DIR", str(tmp_path / "graph"))
    monkeypatch.setattr(graph_jobs, "get_collection", lambda corpus, settings: FakeCollection())
    monkeypatch.setattr(graph_jobs.tracing, "save", lambda trace, status: f"trace-{status}")
    graph_jobs._jobs.clear()
    documents = tmp_path / "docs"
    documents.mkdir()
    (documents / "a.md").write_text("Ada works at Acme.")
    return str(documents), ChunkSettings()


def fake_build(failed=0, nodes=2):
    def build(corpus, settings, corpus_path, model, delay, workers=1, on_progress=None, on_start=None):
        directory = graph_retriever.graph_dir(corpus, settings)
        os.makedirs(directory, exist_ok=True)
        on_start(3, 3)
        for i in range(3):
            on_progress({"status": "failed" if i < failed else "ok"})
        report = {"nodes": nodes, "edges": 1, "triples": 1, "dropped_triples": 0, "chunks": 3, "extracted_now": 3,
                  "created": "2026-11-15T15:00:00", "sources": ["a.md"],
                  "failed_chunks": [{"chunk_id": f"a.md#{i}", "error": "llm error: bad key"} for i in range(failed)],
                  "cost_this_run": {"llm_calls": 3, "prompt_tokens": 300, "completion_tokens": 30, "wall_seconds": 1.5}}
        with open(os.path.join(directory, "graph.json"), "w") as f:
            json.dump({"nodes": [], "edges": []}, f)
        with open(os.path.join(directory, "build-report.json"), "w") as f:
            json.dump(report, f)
        return directory, report
    return build


def test_no_graph_then_build_then_ready(tmp_path, monkeypatch):
    path, settings = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(graph, "build", fake_build())
    info = graph_jobs.status("notes", settings, path)
    assert info["state"] == "none" and info["passages"] == 3 and info["schema"] == "general"

    assert graph_jobs.start("notes", settings, path, {})
    info = wait_until_not_building("notes", settings, path)
    assert info["state"] == "ready" and info["nodes"] == 2 and not info["out_of_date"]
    assert info["trace_id"] == "trace-ok"


def test_a_new_document_makes_the_graph_out_of_date(tmp_path, monkeypatch):
    path, settings = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(graph, "build", fake_build())
    graph_jobs.start("notes", settings, path, {})
    wait_until_not_building("notes", settings, path)
    (tmp_path / "docs" / "b.md").write_text("Bob runs Beta.")
    assert graph_jobs.status("notes", settings, path)["out_of_date"]


def test_every_passage_failing_is_reported_as_a_failed_build(tmp_path, monkeypatch):
    path, settings = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(graph, "build", fake_build(failed=3, nodes=0))
    graph_jobs.start("notes", settings, path, {})
    info = wait_until_not_building("notes", settings, path)
    assert "bad key" in info["error"] and info["trace_id"] == "trace-error"


def test_a_second_build_of_the_same_graph_is_refused_while_one_runs(tmp_path, monkeypatch):
    path, settings = setup(tmp_path, monkeypatch)

    def slow_build(*args, on_start=None, **kwargs):
        time.sleep(0.3)
        return fake_build()(*args, on_start=on_start, **kwargs)

    monkeypatch.setattr(graph, "build", slow_build)
    assert graph_jobs.start("notes", settings, path, {})
    assert not graph_jobs.start("notes", settings, path, {})
    wait_until_not_building("notes", settings, path)

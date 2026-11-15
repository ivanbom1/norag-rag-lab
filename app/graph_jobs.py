import json
import os
import threading
import time

from app import graph, graph_retriever, tracing
from app.indexing import get_collection

_jobs = {}
_lock = threading.Lock()


def _key(corpus, settings):
    return graph_retriever.graph_dir(corpus, settings)


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _built_sources(directory):
    report = _read_json(os.path.join(directory, "build-report.json")) or {}
    if "sources" in report:
        return set(report["sources"])
    return {row["source"] for row in graph.load_cache(os.path.join(directory, "extractions.jsonl")).values()}


def status(corpus, settings, corpus_path):
    directory = _key(corpus, settings)
    job = _jobs.get(directory)
    result = {"state": "none", "schema": graph.schema_name(corpus), "model": graph.GRAPH_MODEL,
              "passages": get_collection(corpus, settings).count()}
    if job and job["state"] == "building":
        return {**result, **{key: job[key] for key in ("state", "done", "total", "failed", "started")}}
    if graph_retriever.available(corpus, settings):
        report = _read_json(os.path.join(directory, "build-report.json")) or {}
        built = os.path.getmtime(graph_retriever.graph_path(corpus, settings))
        documents = [name for name in os.listdir(corpus_path) if not name.startswith(".")]
        changed = any(os.path.getmtime(os.path.join(corpus_path, name)) > built for name in documents)
        result.update(
            state="ready", nodes=report.get("nodes"), edges=report.get("edges"), chunks=report.get("chunks"),
            failed=len(report.get("failed_chunks", [])), built=report.get("created"),
            out_of_date=changed or set(documents) != _built_sources(directory),
            first_error=next((item["error"] for item in report.get("failed_chunks", []) if item.get("error")), None),
        )
    if job and job["state"] == "failed":
        result["error"] = job["error"]
    if job:
        result["trace_id"] = job.get("trace_id")
    return result


def start(corpus, settings, corpus_path, settings_fields, workers=4, delay=0):
    directory = _key(corpus, settings)
    with _lock:
        if _jobs.get(directory, {}).get("state") == "building":
            return False
        job = {"state": "building", "done": 0, "total": None, "failed": 0, "started": time.time(), "error": None}
        _jobs[directory] = job
    thread = threading.Thread(target=_run, args=(job, corpus, settings, corpus_path, settings_fields, workers, delay),
                              daemon=True)
    thread.start()
    return True


def _run(job, corpus, settings, corpus_path, settings_fields, workers, delay):
    trace = tracing.Trace("graph_build", corpus=corpus, settings=settings_fields, model=graph.GRAPH_MODEL,
                          schema=graph.schema_name(corpus))

    def on_start(todo, total):
        job["total"] = todo
        trace.data["chunks"] = total
        trace.data["chunks_to_extract"] = todo

    def on_progress(row):
        job["done"] += 1
        job["failed"] += row["status"] != "ok"

    try:
        _, report = graph.build(corpus, settings, corpus_path, graph.GRAPH_MODEL, delay, workers=workers,
                                on_progress=on_progress, on_start=on_start)
    except Exception as error:
        job.update(state="failed", error=str(error))
        trace.data["error"] = str(error)
        job["trace_id"] = _save(trace, "error")
        return

    cost = report["cost_this_run"]
    trace.add_span("extract", round(cost.get("wall_seconds", 0) * 1000, 1), model=graph.GRAPH_MODEL,
                   llm_calls=cost.get("llm_calls", 0), prompt_tokens=int(cost.get("prompt_tokens", 0)),
                   completion_tokens=int(cost.get("completion_tokens", 0)),
                   cost=tracing.llm_cost(graph.GRAPH_MODEL, cost.get("prompt_tokens", 0), cost.get("completion_tokens", 0)),
                   chunks=report["extracted_now"], failed=len(report["failed_chunks"]))
    trace.data.update(nodes=report["nodes"], edges=report["edges"], triples=report["triples"],
                      dropped_triples=report["dropped_triples"])
    failed = report["failed_chunks"]
    if failed and len(failed) == report["chunks"]:
        job.update(state="failed", error=f"Every passage failed. First error: {failed[0]['error']}")
        job["trace_id"] = _save(trace, "error")
        return
    job["state"] = "ready"
    job["trace_id"] = _save(trace, "ok")


def _save(trace, status):
    try:
        return tracing.save(trace, status)
    except Exception as error:
        print(f"Trace {trace.id} not saved: {error}")
        return None


def concepts(corpus, settings, contains="", limit=50):
    data = graph_retriever.load_graph(corpus, settings)
    degree = {}
    for edge in data["edges"]:
        for end in (edge["source"], edge["target"]):
            degree[end] = degree.get(end, 0) + 1
    needle = contains.strip().lower()
    matching = [node for node in data["nodes"]
                if not needle or any(needle in name.lower() for name in [node["name"], *node["aliases"]])]
    matching.sort(key=lambda node: (-degree.get(node["id"], 0), node["name"].lower()))
    return {"total": len(data["nodes"]), "matching": len(matching),
            "concepts": [{"id": node["id"], "name": node["name"], "type": node["type"],
                          "links": degree.get(node["id"], 0)} for node in matching[:limit]]}


def concept(corpus, settings, node_id):
    data = graph_retriever.load_graph(corpus, settings)
    nodes = {node["id"]: node for node in data["nodes"]}
    if node_id not in nodes:
        return None
    links = []
    for edge in data["edges"]:
        if node_id not in (edge["source"], edge["target"]):
            continue
        other = edge["target"] if edge["source"] == node_id else edge["source"]
        links.append({"source": nodes[edge["source"]]["name"], "relation": edge["relation"],
                      "target": nodes[edge["target"]]["name"], "other": {"id": other, "name": nodes[other]["name"],
                                                                         "type": nodes[other]["type"]},
                      "outgoing": edge["source"] == node_id, "chunks": edge["chunks"],
                      "quotes": [quote["quote"] for quote in edge["quotes"]][:3]})
    node = nodes[node_id]
    return {"id": node_id, "name": node["name"], "type": node["type"], "aliases": node["aliases"],
            "chunks": node["chunks"], "links": links}

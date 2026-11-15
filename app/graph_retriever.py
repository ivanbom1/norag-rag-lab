"""Graph retriever: find the graph entities named in the question, walk up to HOPS edges from them in either
direction, and return the chunks those edges were extracted from, closest hop first.

Each returned chunk carries the edges that led to it in chunk["graph"].
"""
import json
import os

from app import bm25
from app.indexing import get_collection
from app.tokens import tokenize

GRAPH_DIR = "graph"
HOPS = 2
_loaded = {}


def graph_dir(corpus, settings):
    return os.path.join(GRAPH_DIR, bm25.table_name(corpus, settings))


def graph_path(corpus, settings):
    return os.path.join(graph_dir(corpus, settings), "graph.json")


def available(corpus, settings):
    return os.path.exists(graph_path(corpus, settings))


def load_graph(corpus, settings):
    path = graph_path(corpus, settings)
    modified = os.path.getmtime(path)
    # read again when the file changes, so a rebuild is picked up without restarting the app
    if _loaded.get(path, (None,))[0] != modified:
        with open(path, encoding="utf-8") as f:
            _loaded[path] = (modified, json.load(f))
    return _loaded[path][1]


def find_seeds(question, nodes):
    """Nodes whose name, or one of the names they were found under, has all its words in the question."""
    words = set(tokenize(question))
    # plan names are common words ("Which team should…"), so a plan also needs the word plan
    names_a_plan = bool(words & {"plan", "plans"})
    return [node["id"] for node in nodes
            if (node["type"] != "Plan" or names_a_plan)
            and any(tokenize(name) and set(tokenize(name)) <= words for name in [node["name"], *node["aliases"]])]


def walk(edges, seeds, hops=HOPS):
    """{edge index: the (edge index, node walked from) steps from a seed to that edge}, in both directions."""
    paths = {seed: [] for seed in seeds}
    frontier = set(seeds)
    reached = {}
    for _ in range(hops):
        found = {}
        for i, edge in enumerate(edges):
            start = next((end for end in (edge["source"], edge["target"]) if end in frontier), None)
            if i in reached or start is None:
                continue
            reached[i] = paths[start] + [(i, start)]
            other = edge["target"] if start == edge["source"] else edge["source"]
            if other not in paths:
                found.setdefault(other, reached[i])
        paths |= found
        frontier = set(found)
    return reached


def step(edge, nodes, walked_from):
    source, target = nodes[edge["source"]], nodes[edge["target"]]
    return {"source": source["name"], "relation": edge["relation"], "target": target["name"],
            "source_type": source["type"], "target_type": target["type"], "forward": walked_from == edge["source"]}


def search(question, corpus, sources, settings, top_k):
    graph = load_graph(corpus, settings)
    nodes = {node["id"]: node for node in graph["nodes"]}
    edges = [{**edge, "chunks": [c for c in edge["chunks"] if c.rsplit("#", 1)[0] in sources]} for edge in graph["edges"]]
    edges = [edge for edge in edges if edge["chunks"]]

    paths = {}
    for i, path in walk(edges, find_seeds(question, graph["nodes"])).items():
        for chunk_id in edges[i]["chunks"]:
            entry = paths.setdefault(chunk_id, {"hop": len(path), "paths": []})
            entry["hop"] = min(entry["hop"], len(path))
            entry["paths"].append([step(edges[j], nodes, start) for j, start in path])
    ranked = sorted(paths, key=lambda c: (paths[c]["hop"], -len(paths[c]["paths"]), c))[:top_k]
    if not ranked:
        return []

    found = get_collection(corpus, settings).get(ids=ranked)
    chunks = {chunk_id: {"id": chunk_id, "text": text, **metadata}
              for chunk_id, text, metadata in zip(found["ids"], found["documents"], found["metadatas"])}
    return [{**chunks[c], "graph": paths[c]} for c in ranked if c in chunks]

from app import graph_retriever

NODES = [
    {"id": "ErrorCode:kst-5031", "name": "KST-5031", "type": "ErrorCode", "aliases": ["KST-5031"]},
    {"id": "Service:postroom", "name": "postroom", "type": "Service", "aliases": ["postroom"]},
    {"id": "Team:messaging", "name": "Messaging", "type": "Team", "aliases": ["Messaging", "Messaging team"]},
    {"id": "Channel:#oncall-messaging", "name": "#oncall-messaging", "type": "Channel", "aliases": ["#oncall-messaging"]},
    {"id": "Plan:free", "name": "Free", "type": "Plan", "aliases": ["Free"]},
    {"id": "Plan:team", "name": "Team", "type": "Plan", "aliases": ["Team"]},
]
EDGES = [
    {"source": "ErrorCode:kst-5031", "relation": "raised_by", "target": "Service:postroom", "chunks": ["runbook.md#3"]},
    {"source": "Service:postroom", "relation": "owned_by", "target": "Team:messaging", "chunks": ["catalog.md#1"]},
    {"source": "Team:messaging", "relation": "on_call_channel", "target": "Channel:#oncall-messaging",
     "chunks": ["catalog.md#4"]},
]


def test_seeds_need_every_word_of_a_name():
    question = "Error KST-5031 happened again. Which messaging team should support contact?"
    assert graph_retriever.find_seeds(question, NODES) == ["ErrorCode:kst-5031", "Team:messaging"]


def test_a_plan_is_a_seed_only_when_the_question_says_plan():
    assert graph_retriever.find_seeds("Which team should support contact?", NODES) == []
    assert graph_retriever.find_seeds("Does the Team plan include it?", NODES) == ["Plan:team"]


def test_walk_goes_two_hops_in_both_directions():
    assert graph_retriever.walk(EDGES, ["ErrorCode:kst-5031"]) == {0: 1, 1: 2}
    assert graph_retriever.walk(EDGES, ["Channel:#oncall-messaging"]) == {2: 1, 1: 2}


class FakeCollection:
    def get(self, ids):
        return {"ids": ids, "documents": [f"text of {i}" for i in ids],
                "metadatas": [{"source": i.split("#")[0]} for i in ids]}


def test_search_ranks_by_hop_keeps_the_path_and_filters_sources(monkeypatch):
    monkeypatch.setattr(graph_retriever, "load_graph", lambda corpus, settings: {"nodes": NODES, "edges": EDGES})
    monkeypatch.setattr(graph_retriever, "get_collection", lambda corpus, settings: FakeCollection())
    chunks = graph_retriever.search("What raises KST-5031?", "kestrel", ["runbook.md", "catalog.md"], None, 5)
    assert [c["id"] for c in chunks] == ["runbook.md#3", "catalog.md#1"]
    assert chunks[1]["graph"] == {"hop": 2, "edges": [
        {"source": "postroom", "relation": "owned_by", "target": "Messaging", "hop": 2}]}

    only_runbook = graph_retriever.search("What raises KST-5031?", "kestrel", ["runbook.md"], None, 5)
    assert [c["id"] for c in only_runbook] == ["runbook.md#3"]

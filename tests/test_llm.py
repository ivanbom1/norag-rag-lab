import httpx
from openai import BadRequestError

from app import llm


class Message:
    def __init__(self, content):
        self.content = content


class Choice:
    def __init__(self, content):
        self.message = Message(content)


class Response:
    def __init__(self, content):
        self.choices = [Choice(content)]


def test_json_text_takes_the_object_out_of_fences_and_thinking():
    assert llm.json_text(Response('{"a": 1}')) == '{"a": 1}'
    assert llm.json_text(Response('```json\n{"a": 1}\n```')) == '{"a": 1}'
    assert llm.json_text(Response('<think>maybe {"b": 2}</think>Here it is: {"a": 1} hope it helps')) == '{"a": 1}'
    assert llm.json_text(Response(None)) == ""


def test_chat_retries_without_json_schema_when_the_server_rejects_it(monkeypatch):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if "response_format" in kwargs:
            response = httpx.Response(400, request=httpx.Request("POST", "https://example.test"))
            raise BadRequestError("response_format json_schema is not supported", response=response, body=None)
        return Response('{"ok": true}')

    monkeypatch.setattr(llm.client.chat.completions, "create", create)
    response = llm.chat([{"role": "user", "content": "hi"}], "my-model", {"type": "object"}, "test")
    assert llm.json_text(response) == '{"ok": true}'
    assert len(calls) == 2
    assert calls[1]["model"] == "my-model"
    assert "JSON schema" in calls[1]["messages"][-1]["content"]


def test_chat_without_schema_sends_no_response_format(monkeypatch):
    calls = []
    monkeypatch.setattr(llm.client.chat.completions, "create", lambda **kwargs: calls.append(kwargs) or Response("hi"))
    llm.chat([{"role": "user", "content": "hi"}])
    assert "response_format" not in calls[0] and calls[0]["model"] == llm.LLM_MODEL


def test_settings_come_from_the_environment(monkeypatch):
    import importlib
    monkeypatch.setenv("LLM_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("LLM_MODEL", "llama3.2")
    monkeypatch.delenv("LLM_GRAPH_MODEL", raising=False)
    reloaded = importlib.reload(llm)
    try:
        assert reloaded.LLM_URL == "http://localhost:11434/v1"
        assert reloaded.LLM_MODEL == "llama3.2"
        assert reloaded.LLM_GRAPH_MODEL == "llama3.2"
        assert str(reloaded.client.base_url).startswith("http://localhost:11434/v1")
        monkeypatch.setenv("LLM_URL", reloaded.GEMINI_URL)
        monkeypatch.setenv("LLM_MODEL", "gemini-3.5-flash-lite")
        assert importlib.reload(llm).LLM_GRAPH_MODEL == "gemini-3-flash-preview"
    finally:
        monkeypatch.undo()
        importlib.reload(llm)

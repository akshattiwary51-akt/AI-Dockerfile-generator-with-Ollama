"""Ollama client: availability, model checks, errors, structured output (HTTP fully mocked)."""
from __future__ import annotations

from typing import Any, Optional

import pytest
import requests

from app.agent.orchestrator import Orchestrator
from app.llm.ollama_client import ModelNotFoundError, OllamaClient, OllamaError, OllamaUnavailableError
from app.utils.config import Settings
from tests.conftest import FakeBuilder


class FakeResponse:
    def __init__(self, status: int = 200, data: Any = None, text: str = "") -> None:
        self.status_code = status
        self._data = data
        self.text = text or (str(data) if data is not None else "")

    def json(self) -> Any:
        if self._data is None:
            raise ValueError("no json")
        return self._data


class FakeSession:
    def __init__(self, get: Optional[list] = None, post: Optional[list] = None) -> None:
        self._get, self._post = list(get or []), list(post or [])
        self.posts: list[dict[str, Any]] = []

    @staticmethod
    def _next(queue: list) -> Any:
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def get(self, url: str, timeout: Any = None) -> Any:
        return self._next(self._get)

    def post(self, url: str, json: Any = None, timeout: Any = None) -> Any:
        import copy
        self.posts.append({"url": url, "json": copy.deepcopy(json), "timeout": timeout})
        return self._next(self._post)


def tags(*names: str) -> FakeResponse:
    return FakeResponse(200, {"models": [{"name": n} for n in names]})


def chat_ok(content: str = '{"a": 1}') -> FakeResponse:
    return FakeResponse(200, {"message": {"role": "assistant", "content": content}})


MSGS = [{"role": "user", "content": "hi"}]


def test_ollama_unavailable_gives_actionable_message(settings):
    client = OllamaClient(settings, FakeSession(get=[requests.ConnectionError("refused")] * 3,
                                                post=[requests.ConnectionError("refused")]))
    assert client.is_available() is False
    with pytest.raises(OllamaUnavailableError) as exc:
        client.list_models()
    msg = str(exc.value)
    assert "ollama serve" in msg and "ollama pull" in msg and settings.ollama_host in msg
    with pytest.raises(OllamaUnavailableError):
        client.ensure_ready()
    with pytest.raises(OllamaUnavailableError):
        client.chat(MSGS)


def test_environment_check_reports_ollama_down(settings):
    client = OllamaClient(settings, FakeSession(get=[requests.ConnectionError("x")] * 5))
    status = Orchestrator(settings, client=client, builder=FakeBuilder([])).check_environment()
    assert not status.ollama_available and not status.ready_to_generate
    assert "ollama serve" in status.ollama_message and status.docker_available


def test_model_not_installed(settings):
    client = OllamaClient(settings, FakeSession(get=[tags("llama3:8b"), tags("llama3:8b")]))
    assert client.has_model() is False
    with pytest.raises(ModelNotFoundError) as exc:
        client.ensure_ready()
    assert "ollama pull test-model:7b" in str(exc.value) and "llama3:8b" in str(exc.value)


def test_model_not_found_on_chat_404(settings):
    session = FakeSession(get=[tags("other:1b")], post=[FakeResponse(404, {"error": "model 'test-model:7b' not found"})])
    with pytest.raises(ModelNotFoundError) as exc:
        OllamaClient(settings, session).chat(MSGS)
    assert "ollama pull" in str(exc.value)


def test_model_name_normalization(settings):
    client = OllamaClient(settings, FakeSession(get=[tags("test-model:7b"), tags("mistral:latest")]))
    assert client.has_model("test-model:7b")
    assert client.has_model("mistral")  # bare name == :latest


def test_chat_sends_options_and_schema(settings):
    session = FakeSession(post=[chat_ok()])
    out = OllamaClient(settings, session).chat(MSGS, schema={"type": "object"})
    assert out == '{"a": 1}'
    sent = session.posts[0]
    assert sent["url"] == "http://ollama.test:11434/api/chat" and sent["timeout"] == settings.ollama_timeout
    body = sent["json"]
    assert body["model"] == "test-model:7b" and body["stream"] is False and body["format"] == {"type": "object"}
    assert body["options"] == {"temperature": 0.1, "num_ctx": 4096}


def test_chat_falls_back_to_json_mode_when_schema_rejected(settings):
    session = FakeSession(post=[FakeResponse(400, {"error": "invalid format"}), chat_ok("{}")])
    assert OllamaClient(settings, session).chat(MSGS, schema={"type": "object"}) == "{}"
    assert [p["json"]["format"] for p in session.posts] == [{"type": "object"}, "json"]


def test_chat_timeout_message_mentions_setting(settings):
    with pytest.raises(OllamaError) as exc:
        OllamaClient(settings, FakeSession(post=[requests.Timeout()])).chat(MSGS)
    assert "OLLAMA_TIMEOUT" in str(exc.value)


def test_model_load_failure(settings):
    resp = FakeResponse(500, {"error": "model failed to load: out of memory"})
    with pytest.raises(OllamaError) as exc:
        OllamaClient(settings, FakeSession(post=[resp])).chat(MSGS)
    assert "failed to load" in str(exc.value) and "smaller model" in str(exc.value)


@pytest.mark.parametrize("response", [FakeResponse(200, {"message": {"content": "  "}}), FakeResponse(200, {"x": 1}),
                                      FakeResponse(200, None, "not json"), FakeResponse(503, None, "busy")])
def test_bad_responses_raise_ollama_error(settings, response):
    with pytest.raises(OllamaError):
        OllamaClient(settings, FakeSession(post=[response])).chat(MSGS)


def test_configuration_comes_from_environment(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "myhost:9999/")
    monkeypatch.setenv("OLLAMA_MODEL", "deepseek-coder-v2")
    monkeypatch.setenv("MAX_RETRIES", "5")
    monkeypatch.setenv("DOCKER_BUILD_TIMEOUT", "123")
    s = Settings()
    assert s.ollama_host == "http://myhost:9999" and s.ollama_model == "deepseek-coder-v2"
    assert s.max_retries == 5 and s.docker_build_timeout == 123

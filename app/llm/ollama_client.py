"""Thin, defensive client for the local Ollama HTTP API."""
from __future__ import annotations

from typing import Any, Optional

import requests

from app.utils.config import Settings, get_settings
from app.utils.logger import get_logger

log = get_logger("dockerfile_ai.ollama")


class OllamaError(RuntimeError):
    """Base class for Ollama failures."""


class OllamaUnavailableError(OllamaError):
    """Ollama server cannot be reached."""


class ModelNotFoundError(OllamaError):
    """The selected model is not installed locally."""


def unavailable_message(host: str) -> str:
    return (f"Cannot reach Ollama at {host}.\n"
            "  1. Install Ollama: https://ollama.com/download\n"
            "  2. Start the server:  ollama serve\n"
            "  3. Pull a model:      ollama pull <model>   (e.g. ollama pull qwen2.5-coder:7b)\n"
            "  4. Check OLLAMA_HOST in your environment/.env if Ollama runs elsewhere.")


def not_found_message(model: str, installed: list[str]) -> str:
    have = ", ".join(installed) if installed else "none"
    return (f"Model '{model}' is not installed in Ollama (installed: {have}).\n"
            f"Install it with:  ollama pull {model}\n"
            "or choose another model via OLLAMA_MODEL.")


def _normalize(name: str) -> str:
    return name if ":" in name else f"{name}:latest"


class OllamaClient:
    def __init__(self, settings: Optional[Settings] = None, session: Optional[requests.Session] = None) -> None:
        self.settings = settings or get_settings()
        self.session = session or requests.Session()
        self.model = self.settings.ollama_model

    @property
    def host(self) -> str:
        return self.settings.ollama_host

    # ------------------------------------------------------------------ status
    def is_available(self) -> bool:
        try:
            return self.session.get(f"{self.host}/api/tags", timeout=5).status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> list[str]:
        try:
            resp = self.session.get(f"{self.host}/api/tags", timeout=10)
        except requests.RequestException as exc:
            raise OllamaUnavailableError(unavailable_message(self.host)) from exc
        if resp.status_code != 200:
            raise OllamaError(f"Ollama /api/tags returned HTTP {resp.status_code}: {resp.text[:200]}")
        try:
            return [m["name"] for m in resp.json().get("models", [])]
        except (ValueError, KeyError, TypeError) as exc:
            raise OllamaError("Ollama returned an unexpected /api/tags payload.") from exc

    def has_model(self, name: Optional[str] = None) -> bool:
        wanted = _normalize(name or self.model)
        return any(_normalize(m) == wanted for m in self.list_models())

    def ensure_ready(self, model: Optional[str] = None) -> None:
        """Raise a helpful error unless Ollama is up and the model is installed."""
        name = model or self.model
        installed = self.list_models()  # raises OllamaUnavailableError
        if _normalize(name) not in {_normalize(m) for m in installed}:
            raise ModelNotFoundError(not_found_message(name, installed))

    # ------------------------------------------------------------------ chat
    def chat(self, messages: list[dict[str, str]], *, schema: Optional[dict[str, Any]] = None,
             model: Optional[str] = None) -> str:
        """Send a conversation and return the assistant text.

        ``schema`` (a JSON schema) enables Ollama structured outputs; on servers that reject it the
        call transparently falls back to plain JSON mode.
        """
        name = model or self.model
        payload: dict[str, Any] = {
            "model": name, "messages": messages, "stream": False,
            "options": {"temperature": self.settings.temperature, "num_ctx": self.settings.num_ctx},
        }
        payload["format"] = schema if schema else "json"
        try:
            resp = self._post_chat(payload)
            if resp.status_code == 400 and schema:
                log.warning("Ollama rejected the JSON schema; retrying with plain JSON mode")
                payload["format"] = "json"
                resp = self._post_chat(payload)
        except requests.ConnectionError as exc:
            raise OllamaUnavailableError(unavailable_message(self.host)) from exc
        except requests.Timeout as exc:
            raise OllamaError(
                f"Ollama did not answer within {self.settings.ollama_timeout}s (model '{name}'). The model may "
                "still be loading or is too large for this machine; raise OLLAMA_TIMEOUT or pick a smaller model."
            ) from exc
        except requests.RequestException as exc:
            raise OllamaError(f"Request to Ollama failed: {exc}") from exc
        return self._extract_content(resp, name)

    def _post_chat(self, payload: dict[str, Any]) -> requests.Response:
        return self.session.post(f"{self.host}/api/chat", json=payload, timeout=self.settings.ollama_timeout)

    def _extract_content(self, resp: requests.Response, name: str) -> str:
        if resp.status_code == 404:
            raise ModelNotFoundError(not_found_message(name, self._safe_list()))
        if resp.status_code != 200:
            detail = self._error_text(resp)
            low = detail.lower()
            if "not found" in low and "model" in low:
                raise ModelNotFoundError(not_found_message(name, self._safe_list()))
            if any(k in low for k in ("memory", "failed to load", "load model", "llm server", "cuda")):
                raise OllamaError(f"Ollama failed to load model '{name}': {detail}\n"
                                  "Try a smaller model/quantization or lower OLLAMA_NUM_CTX.")
            raise OllamaError(f"Ollama returned HTTP {resp.status_code}: {detail}")
        try:
            body = resp.json()
            content = body["message"]["content"]
        except (ValueError, KeyError, TypeError) as exc:
            raise OllamaError("Ollama returned an unexpected response body (no message content).") from exc
        if not isinstance(content, str) or not content.strip():
            raise OllamaError(f"Model '{name}' returned an empty response.")
        return content

    @staticmethod
    def _error_text(resp: requests.Response) -> str:
        try:
            return str(resp.json().get("error", resp.text))[:500]
        except ValueError:
            return resp.text[:500]

    def _safe_list(self) -> list[str]:
        try:
            return self.list_models()
        except OllamaError:
            return []

"""Defensive client for the local Ollama HTTP API with latency/token tracking and model metadata."""
from __future__ import annotations

import math
import time
from typing import Any, Optional

import requests
from pydantic import BaseModel

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


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~3.5 characters per token for code/English); used only when Ollama reports none."""
    return max(1, math.ceil(len(text) / 3.5))


class ModelInfo(BaseModel):
    name: str
    size_bytes: int = 0
    parameter_size: str = ""      # e.g. "7.6B"
    quantization: str = ""        # e.g. "Q4_K_M"
    family: str = ""
    context_length: Optional[int] = None  # from /api/show when available

    @property
    def parameters_billion(self) -> Optional[float]:
        s = self.parameter_size.strip().upper()
        try:
            if s.endswith("B"):
                return float(s[:-1])
            if s.endswith("M"):
                return float(s[:-1]) / 1000
        except ValueError:
            return None
        return None


class ChatStats(BaseModel):
    """Per-call measurements. ``estimated`` is True when token counts are not reported by Ollama."""

    model: str
    latency_seconds: float
    load_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_per_second: Optional[float] = None
    context_window: int = 0
    estimated: bool = False

    @property
    def context_used_fraction(self) -> float:
        return (self.prompt_tokens + self.completion_tokens) / self.context_window if self.context_window else 0.0

    @property
    def prompt_may_be_truncated(self) -> bool:
        """Ollama silently drops the oldest context when the prompt exceeds num_ctx."""
        return bool(self.context_window) and self.prompt_tokens >= self.context_window


class OllamaClient:
    def __init__(self, settings: Optional[Settings] = None, session: Optional[requests.Session] = None) -> None:
        self.settings = settings or get_settings()
        self.session = session or requests.Session()
        self.model = self.settings.ollama_model
        self.last_stats: Optional[ChatStats] = None
        self.stats_log: list[ChatStats] = []

    @property
    def host(self) -> str:
        return self.settings.ollama_host

    # ------------------------------------------------------------------ status
    def is_available(self) -> bool:
        try:
            return self.session.get(f"{self.host}/api/tags", timeout=5).status_code == 200
        except requests.RequestException:
            return False

    def _tags(self) -> list[dict[str, Any]]:
        try:
            resp = self.session.get(f"{self.host}/api/tags", timeout=10)
        except requests.RequestException as exc:
            raise OllamaUnavailableError(unavailable_message(self.host)) from exc
        if resp.status_code != 200:
            raise OllamaError(f"Ollama /api/tags returned HTTP {resp.status_code}: {resp.text[:200]}")
        try:
            return list(resp.json().get("models", []))
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise OllamaError("Ollama returned an unexpected /api/tags payload.") from exc

    def list_models(self) -> list[str]:
        try:
            return [m["name"] for m in self._tags()]
        except KeyError as exc:
            raise OllamaError("Ollama returned an unexpected /api/tags payload.") from exc

    def list_model_details(self, with_context: bool = False) -> list[ModelInfo]:
        """Installed models with parameter size, quantization and file size (and context length if requested)."""
        infos: list[ModelInfo] = []
        for m in self._tags():
            d = m.get("details") or {}
            info = ModelInfo(name=m.get("name", "?"), size_bytes=int(m.get("size") or 0),
                             parameter_size=str(d.get("parameter_size") or ""),
                             quantization=str(d.get("quantization_level") or ""), family=str(d.get("family") or ""))
            if with_context:
                info.context_length = self.context_length(info.name)
            infos.append(info)
        return infos

    def context_length(self, name: str) -> Optional[int]:
        """Maximum context window the model supports (best effort via /api/show)."""
        try:
            resp = self.session.post(f"{self.host}/api/show", json={"model": name}, timeout=10)
            if resp.status_code != 200:
                return None
            for key, value in (resp.json().get("model_info") or {}).items():
                if key.endswith(".context_length") and isinstance(value, int):
                    return value
        except (requests.RequestException, ValueError, AttributeError):
            pass
        return None

    def has_model(self, name: Optional[str] = None) -> bool:
        wanted = _normalize(name or self.model)
        return any(_normalize(m) == wanted for m in self.list_models())

    def ensure_ready(self, model: Optional[str] = None) -> None:
        """Raise a helpful error unless Ollama is up and the model is installed."""
        name = model or self.model
        installed = self.list_models()  # raises OllamaUnavailableError
        if _normalize(name) not in {_normalize(m) for m in installed}:
            raise ModelNotFoundError(not_found_message(name, installed))

    def warm_up(self, model: Optional[str] = None) -> Optional[float]:
        """Load the model into memory (empty prompt) so the first real request does not time out.

        Returns the load time in seconds, or None if the warm-up failed (non-fatal).
        """
        name = model or self.model
        started = time.monotonic()
        try:
            resp = self.session.post(f"{self.host}/api/generate", json={"model": name, "prompt": "", "stream": False,
                                                                       "keep_alive": "10m"},
                                     timeout=self.settings.ollama_timeout * 2)
            return round(time.monotonic() - started, 2) if resp.status_code == 200 else None
        except requests.RequestException:
            log.warning("Model warm-up failed or timed out (continuing)")
            return None

    # ------------------------------------------------------------------ chat
    def chat(self, messages: list[dict[str, str]], *, schema: Optional[dict[str, Any]] = None,
             model: Optional[str] = None) -> str:
        """Send a conversation and return the assistant text (and record latency/token stats).

        ``schema`` (a JSON schema) enables Ollama structured outputs; on servers that reject it the call falls back
        to plain JSON mode. A timeout is retried ``OLLAMA_TIMEOUT_RETRIES`` times with a longer timeout.
        """
        name = model or self.model
        payload: dict[str, Any] = {
            "model": name, "messages": messages, "stream": False,
            "options": {"temperature": self.settings.temperature, "num_ctx": self.settings.num_ctx},
            "format": schema if schema else "json"}
        started = time.monotonic()
        timeout = float(self.settings.ollama_timeout)
        attempts_left = self.settings.ollama_timeout_retries
        while True:
            try:
                resp = self._post_chat(payload, timeout)
                if resp.status_code == 400 and schema:
                    log.warning("Ollama rejected the JSON schema; retrying with plain JSON mode")
                    payload["format"] = "json"
                    resp = self._post_chat(payload, timeout)
                break
            except requests.ConnectionError as exc:
                raise OllamaUnavailableError(unavailable_message(self.host)) from exc
            except requests.Timeout as exc:
                if attempts_left > 0:
                    attempts_left -= 1
                    timeout *= 1.5
                    log.warning("Ollama timed out; retrying with a %.0fs timeout (model may still be loading)", timeout)
                    continue
                raise OllamaError(
                    f"Ollama did not answer within {self.settings.ollama_timeout}s (model '{name}'). The model may "
                    "still be loading or is too large for this machine; raise OLLAMA_TIMEOUT or pick a smaller model."
                ) from exc
            except requests.RequestException as exc:
                raise OllamaError(f"Request to Ollama failed: {exc}") from exc
        content, body = self._extract_content(resp, name)
        self._record_stats(body, name, time.monotonic() - started, messages, content)
        return content

    def _post_chat(self, payload: dict[str, Any], timeout: float) -> requests.Response:
        return self.session.post(f"{self.host}/api/chat", json=payload, timeout=timeout)

    def _record_stats(self, body: dict[str, Any], name: str, wall: float, messages: list[dict[str, str]],
                      content: str) -> None:
        pt, ct = body.get("prompt_eval_count"), body.get("eval_count")
        estimated = not (isinstance(pt, int) and isinstance(ct, int))
        if estimated:
            pt = sum(estimate_tokens(m.get("content", "")) for m in messages)
            ct = estimate_tokens(content)
        eval_ns = body.get("eval_duration")
        tps = ct / (eval_ns / 1e9) if isinstance(eval_ns, (int, float)) and eval_ns > 0 else (
            ct / wall if wall > 0 and estimated else None)
        load_ns = body.get("load_duration")
        stats = ChatStats(model=name, latency_seconds=round(wall, 2), prompt_tokens=int(pt), completion_tokens=int(ct),
                          load_seconds=round(load_ns / 1e9, 2) if isinstance(load_ns, (int, float)) else 0.0,
                          tokens_per_second=round(tps, 1) if tps else None, context_window=self.settings.num_ctx,
                          estimated=estimated)
        self.last_stats = stats
        self.stats_log.append(stats)
        if stats.prompt_may_be_truncated:
            log.warning("Prompt (~%d tokens) fills the %d-token context window; Ollama may truncate it. "
                        "Raise OLLAMA_NUM_CTX or lower MAX_CONTEXT_CHARS.", stats.prompt_tokens, stats.context_window)

    def _extract_content(self, resp: requests.Response, name: str) -> tuple[str, dict[str, Any]]:
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
        return content, body

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

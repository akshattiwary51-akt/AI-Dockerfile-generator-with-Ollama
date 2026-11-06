"""Parse and validate structured LLM output. Raw model text is never trusted."""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from app.utils.logger import get_logger

log = get_logger("dockerfile_ai.parser")


class ResponseParseError(ValueError):
    """The model output could not be turned into valid structured data."""


class GeneratedArtifacts(BaseModel):
    dockerfile: str
    dockerignore: str = ""
    language: str = "unknown"
    framework: Optional[str] = None
    base_image: str = ""
    exposed_port: Optional[int] = None
    explanation: str = ""
    limitations: list[str] = Field(default_factory=list)

    @field_validator("dockerfile", "dockerignore", mode="before")
    @classmethod
    def _clean_text(cls, v: Any) -> Any:
        if v is None:
            return ""
        if not isinstance(v, str):
            raise ValueError("must be a string")
        v = _strip_fences(v)
        if "\n" not in v and "\\n" in v:  # model double-escaped newlines
            v = v.replace("\\n", "\n")
        return v.strip("\n") + "\n" if v.strip() else ""

    def _dockerfile_required(self) -> bool:
        return True

    @model_validator(mode="after")
    def _non_empty(self) -> "GeneratedArtifacts":
        if self._dockerfile_required() and not self.dockerfile.strip():
            raise ValueError("dockerfile is empty")
        return self

    @field_validator("exposed_port", mode="before")
    @classmethod
    def _port(cls, v: Any) -> Any:
        if v in (None, "", "none", "None", "N/A", "n/a"):
            return None
        if isinstance(v, str):
            m = re.search(r"\d+", v)
            return int(m.group()) if m else None
        if isinstance(v, list):
            return int(v[0]) if v else None
        return v

    @field_validator("limitations", mode="before")
    @classmethod
    def _listify(cls, v: Any) -> Any:
        if v is None or v == "":
            return []
        if isinstance(v, str):
            return [v]
        return [str(x) for x in v]

    @field_validator("language", "base_image", "explanation", "framework", mode="before")
    @classmethod
    def _stringify(cls, v: Any) -> Any:
        if v is None:
            return None
        return v if isinstance(v, str) else str(v)


class CorrectionResult(GeneratedArtifacts):
    dockerfile: str = ""  # may be empty when the failure is not recoverable
    root_cause: str = ""
    changes: list[str] = Field(default_factory=list)
    expected_effect: str = ""
    remaining_risks: list[str] = Field(default_factory=list)
    recoverable: bool = True

    def _dockerfile_required(self) -> bool:
        return self.recoverable

    @field_validator("expected_effect", "root_cause", mode="before")
    @classmethod
    def _text(cls, v: Any) -> Any:
        return "" if v is None else (v if isinstance(v, str) else str(v))

    @field_validator("changes", "remaining_risks", mode="before")
    @classmethod
    def _changes(cls, v: Any) -> Any:
        if v is None or v == "":
            return []
        if isinstance(v, str):
            return [v]
        return [str(x) for x in v]


def _strip_fences(text: str) -> str:
    t = text.strip()
    m = re.match(r"^```[A-Za-z0-9_-]*\n(.*?)\n?```$", t, re.S)
    return m.group(1) if m else text


def extract_json(raw: str) -> dict[str, Any]:
    """Return the first JSON object found in ``raw`` (handles code fences and surrounding prose)."""
    if not raw or not raw.strip():
        raise ResponseParseError("The model returned an empty response.")
    text = raw.strip()
    candidates = [text]
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates += fenced
    start = text.find("{")
    if start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start:i + 1])
                    break
    for cand in candidates:
        try:
            data = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    raise ResponseParseError("The model response did not contain a valid JSON object.")


def _salvage_from_fence(raw: str) -> Optional[dict[str, Any]]:
    """Last resort for weak models: a fenced Dockerfile block with no JSON wrapper."""
    m = re.search(r"```(?:dockerfile|Dockerfile|docker)?\s*\n(.*?FROM\s+\S+.*?)```", raw, re.S)
    if not m:
        return None
    ignore = re.search(r"```(?:dockerignore|gitignore)\s*\n(.*?)```", raw, re.S)
    return {"dockerfile": m.group(1), "dockerignore": ignore.group(1) if ignore else "",
            "limitations": ["Model did not return structured JSON; fields were inferred from a code block."]}


def _parse(raw: str, model: type[BaseModel]) -> Any:
    try:
        data = extract_json(raw)
    except ResponseParseError:
        data = _salvage_from_fence(raw)
        if data is None:
            raise
        log.warning("Structured output missing; salvaged Dockerfile from a fenced code block")
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
        raise ResponseParseError(f"The model response failed validation: {problems}") from exc


def parse_generation(raw: str) -> GeneratedArtifacts:
    return _parse(raw, GeneratedArtifacts)


def parse_correction(raw: str) -> CorrectionResult:
    return _parse(raw, CorrectionResult)

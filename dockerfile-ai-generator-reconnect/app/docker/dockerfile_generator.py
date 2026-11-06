"""LLM-backed Dockerfile generation/correction plus protected file writing."""
from __future__ import annotations

import difflib
import os
import shutil
from pathlib import Path
from typing import Any, Optional, Protocol

from app.agent.execution_history import ExecutionHistory
from app.analyzer.models import ProjectAnalysis
from app.docker.docker_builder import BuildResult
from app.docker.docker_validator import ValidationReport, validate_artifacts
from app.llm import prompt_builder as pb
from app.llm.response_parser import (CorrectionResult, GeneratedArtifacts, ResponseParseError, parse_correction,
                                     parse_generation)
from app.utils.config import Settings, get_settings
from app.utils.logger import get_logger
from app.utils.security import ensure_within

log = get_logger("dockerfile_ai.generator")

TARGET_FILES = ("Dockerfile", ".dockerignore")
MIN_TARGETED_SIMILARITY = 0.35


class GenerationError(RuntimeError):
    """The LLM could not produce a usable result."""


class FileProtectionError(PermissionError):
    """A write would overwrite existing files without approval."""

    def __init__(self, files: list[str]) -> None:
        self.files = files
        super().__init__("Refusing to overwrite existing file(s) without approval: " + ", ".join(files))


class ChatClient(Protocol):
    def chat(self, messages: list[dict[str, str]], *, schema: Optional[dict[str, Any]] = None,
             model: Optional[str] = None) -> str: ...


# ----------------------------------------------------------------------------- diff helpers
def similarity(old: str, new: str) -> float:
    return difflib.SequenceMatcher(None, old, new).ratio()


def summarize_diff(old: str, new: str, limit: int = 8) -> list[str]:
    out: list[str] = []
    for line in difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=0):
        if line.startswith(("---", "+++", "@@")):
            continue
        out.append(("Added: " if line.startswith("+") else "Removed: ") + line[1:].strip())
        if len(out) >= limit:
            out.append("... (more changes)")
            break
    return out


# ----------------------------------------------------------------------------- file protection
def existing_targets(project_dir: Path | str) -> list[str]:
    root = Path(project_dir)
    return [n for n in TARGET_FILES if (root / n).exists() or (root / n).is_symlink()]


def write_artifacts(project_dir: Path | str, dockerfile: str, dockerignore: str, *, overwrite: bool = False,
                    backup: bool = True) -> list[Path]:
    """Write Dockerfile/.dockerignore inside ``project_dir`` only; never overwrite without ``overwrite=True``."""
    root = Path(project_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Project directory does not exist: {root}")
    plan = [("Dockerfile", dockerfile)] + ([(".dockerignore", dockerignore)] if dockerignore.strip() else [])
    clashes = [n for n, _ in plan if (root / n).exists() or (root / n).is_symlink()]
    if clashes and not overwrite:
        raise FileProtectionError(clashes)
    written: list[Path] = []
    for name, content in plan:
        target = root / name
        if target.is_symlink():
            raise PermissionError(f"Refusing to write through symlink: {target}")
        ensure_within(root, target)
        if target.exists() and backup:
            shutil.copy2(target, target.with_name(name + ".bak"))
        tmp = root / f".{name}.tmp"
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, target)
        written.append(target)
        log.info("Wrote %s", target)
    return written


# ----------------------------------------------------------------------------- generator
class DockerfileGenerator:
    """Stateful per run: keeps the conversation so corrections build on earlier turns."""

    def __init__(self, client: ChatClient, settings: Optional[Settings] = None, max_format_retries: int = 2) -> None:
        self.client = client
        self.settings = settings or get_settings()
        self.max_format_retries = max_format_retries
        self.messages: list[dict[str, str]] = []

    def generate(self, analysis: ProjectAnalysis) -> GeneratedArtifacts:
        self.messages = pb.build_generation_messages(analysis, self.settings)
        result, _ = self._converse(correction=False)
        return result

    def correct(self, analysis: ProjectAnalysis, history: ExecutionHistory, current: GeneratedArtifacts,
                build: BuildResult) -> CorrectionResult:
        log_tail = build.combined_output
        msg = pb.build_correction_message(
            analysis, history, current.dockerfile, current.dockerignore, error_summary=build.error_summary,
            log_tail=log_tail, failed_stage=build.failed_stage, exit_code=build.exit_code, settings=self.settings)
        if not self.messages:  # corrections without a prior generate() in this process (e.g. UI reruns)
            self.messages = [{"role": "system", "content": pb.CORRECTION_SYSTEM}]
        else:
            self.messages[0] = {"role": "system", "content": pb.CORRECTION_SYSTEM}
        self.messages.append(msg)
        result, _ = self._converse(correction=True)
        assert isinstance(result, CorrectionResult)

        if result.recoverable:
            if result.dockerfile.strip() == current.dockerfile.strip():
                log.warning("Correction returned an unchanged Dockerfile; asking for a real change")
                self.messages.append({"role": "user", "content": (
                    "Your Dockerfile is identical to the one that failed. Identify a different root cause and "
                    "change the Dockerfile, or set recoverable=false if it cannot be fixed by editing it.")})
                result, _ = self._converse(correction=True)
                assert isinstance(result, CorrectionResult)
                if result.recoverable and result.dockerfile.strip() == current.dockerfile.strip():
                    result.recoverable = False
                    result.root_cause = (result.root_cause + " " if result.root_cause else "") + \
                        "(model could not propose a different fix)"
            elif similarity(current.dockerfile, result.dockerfile) < MIN_TARGETED_SIMILARITY:
                log.warning("Correction rewrote most of the Dockerfile; requesting a targeted fix")
                self.messages.append({"role": "user", "content": (
                    "You rewrote most of the Dockerfile. Keep the previous Dockerfile and change only the lines "
                    "needed to fix the root cause. Reply with the full JSON again.")})
                retry, _ = self._converse(correction=True)
                assert isinstance(retry, CorrectionResult)
                if retry.recoverable:
                    result = retry
        if result.recoverable and not result.changes:
            result.changes = summarize_diff(current.dockerfile, result.dockerfile)
        return result

    def _converse(self, *, correction: bool) -> tuple[GeneratedArtifacts, ValidationReport]:
        parser = parse_correction if correction else parse_generation
        schema = pb.json_schema_for(correction)
        last_problem = "unknown problem"
        for attempt in range(self.max_format_retries + 1):
            raw = self.client.chat(pb.trim_conversation(self.messages), schema=schema)
            self.messages.append({"role": "assistant", "content": raw})
            try:
                parsed = parser(raw)
            except ResponseParseError as exc:
                last_problem = str(exc)
                log.warning("Unusable LLM response (attempt %d): %s", attempt + 1, last_problem)
                self.messages.append(pb.parse_retry_message(last_problem))
                continue
            if isinstance(parsed, CorrectionResult) and not parsed.recoverable and not parsed.dockerfile.strip():
                return parsed, ValidationReport()
            report = validate_artifacts(parsed.dockerfile, parsed.dockerignore)
            if report.ok:
                return parsed, report
            last_problem = "validation errors:\n" + "\n".join(
                f"[{i.code}] {i.message}" + (f" (line {i.line})" if i.line else "") for i in report.errors)
            log.warning("Generated Dockerfile failed validation (attempt %d): %s", attempt + 1, last_problem)
            self.messages.append({"role": "user", "content": (
                "The Dockerfile you produced is invalid:\n" + last_problem +
                "\nFix exactly these problems and reply with the complete JSON object again.")})
        raise GenerationError(f"The model did not produce a valid Dockerfile after "
                              f"{self.max_format_retries + 1} attempts. Last problem: {last_problem}")

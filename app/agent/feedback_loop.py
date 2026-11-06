"""Build -> diagnose -> targeted-fix loop with retry limits, cancellation and history."""
from __future__ import annotations

import difflib
import threading
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Protocol

from pydantic import BaseModel

from app.agent.error_classifier import Diagnosis, classify_build_error
from app.agent.execution_history import AttemptRecord, ExecutionHistory
from app.analyzer.models import ProjectAnalysis
from app.docker.docker_builder import BuildResult
from app.docker.dockerfile_generator import GenerationError, write_artifacts
from app.llm.ollama_client import OllamaError
from app.llm.response_parser import CorrectionResult, GeneratedArtifacts
from app.utils.config import Settings, get_settings
from app.utils.logger import get_logger

log = get_logger("dockerfile_ai.loop")


class LoopStatus(str, Enum):
    SUCCESS = "success"
    MAX_RETRIES = "max_retries_reached"
    REPEATED_ERROR = "repeated_error"
    NON_RECOVERABLE = "non_recoverable_error"
    CANCELLED = "cancelled"


class LoopEvent(BaseModel):
    kind: str  # build_start | build_done | diagnosing | corrected | stopped
    message: str
    attempt: int = 0
    max_attempts: int = 0
    stage: Optional[str] = None


class LoopResult(BaseModel):
    status: LoopStatus
    message: str
    artifacts: GeneratedArtifacts
    history: ExecutionHistory
    last_build: Optional[BuildResult] = None

    @property
    def success(self) -> bool:
        return self.status == LoopStatus.SUCCESS


class Builder(Protocol):
    def build(self, project_dir: Path | str, image: str, *, on_output: Optional[Callable[[str], None]] = None,
              cancel_event: Optional[threading.Event] = None) -> BuildResult: ...


class Corrector(Protocol):
    def correct(self, analysis: ProjectAnalysis, history: ExecutionHistory, current: GeneratedArtifacts,
                build: BuildResult, diagnosis: Optional[Diagnosis] = None, repeat_count: int = 1,
                known_fixes: Optional[list[str]] = None) -> CorrectionResult: ...


class FeedbackLoop:
    def __init__(self, generator: Corrector, builder: Builder, settings: Optional[Settings] = None, *,
                 on_event: Optional[Callable[[LoopEvent], None]] = None,
                 on_log: Optional[Callable[[str], None]] = None,
                 cancel_event: Optional[threading.Event] = None,
                 knowledge: Optional[Callable[[str], list[str]]] = None) -> None:
        self.knowledge = knowledge  # category -> fixes that worked before (local history; never model training)
        self.generator = generator
        self.builder = builder
        self.settings = settings or get_settings()
        self.on_event = on_event
        self.on_log = on_log
        self.cancel_event = cancel_event or threading.Event()

    def _emit(self, kind: str, message: str, attempt: int = 0, stage: Optional[str] = None) -> None:
        log.info("[%s] %s", kind, message)
        if self.on_event:
            try:
                self.on_event(LoopEvent(kind=kind, message=message, attempt=attempt,
                                        max_attempts=self.settings.max_retries + 1, stage=stage))
            except Exception:  # UI errors must not break the loop
                log.debug("on_event callback failed", exc_info=True)

    def run(self, analysis: ProjectAnalysis, project_dir: Path | str, artifacts: GeneratedArtifacts, image: str,
            history: Optional[ExecutionHistory] = None) -> LoopResult:
        """Build ``artifacts`` (already written to ``project_dir``) and fix failures until done.

        At most ``1 + MAX_RETRIES`` builds are executed (the first build plus one per correction).
        """
        history = history or ExecutionHistory()
        current = artifacts
        max_builds = self.settings.max_retries + 1
        last: Optional[BuildResult] = None

        for attempt in range(1, max_builds + 1):
            if self.cancel_event.is_set():
                return self._finish(LoopStatus.CANCELLED, "Cancelled by the user.", current, history, last)
            self._emit("build_start", f"Build attempt {attempt}/{max_builds}", attempt)
            last = self.builder.build(project_dir, image, on_output=self.on_log, cancel_event=self.cancel_event)
            record = history.add(AttemptRecord(
                attempt=attempt, dockerfile=current.dockerfile, success=last.success, exit_code=last.exit_code,
                failed_stage=last.failed_stage, error="" if last.success else last.error_summary,
                duration_seconds=last.duration_seconds, timed_out=last.timed_out))
            self._emit("build_done", "Build succeeded." if last.success else "Build failed.", attempt,
                       last.failed_stage)

            if last.success:
                return self._finish(LoopStatus.SUCCESS, f"Image '{image}' built successfully after "
                                    f"{history.corrections_applied} correction(s).", current, history, last)
            if last.cancelled or self.cancel_event.is_set():
                return self._finish(LoopStatus.CANCELLED, "Cancelled by the user.", current, history, last)
            diag = classify_build_error(last.combined_output or last.error_summary, last.failed_stage)
            record.category, record.fixable = diag.category.value, diag.fixable
            record.failing_instruction, record.signature = diag.failing_instruction, diag.signature
            streak = 0
            for earlier in reversed(history.attempts):
                if earlier.signature != diag.signature:
                    break
                streak += 1
            record.repeated = streak
            if last.non_recoverable_reason:
                record.diagnosis = f"Not fixable by editing the Dockerfile: {last.non_recoverable_reason}"
                return self._finish(LoopStatus.NON_RECOVERABLE, record.diagnosis, current, history, last)
            if not diag.fixable:
                record.diagnosis = f"Not fixable by editing the Dockerfile ({diag.category.value}): {diag.hint}"
                return self._finish(LoopStatus.NON_RECOVERABLE, record.diagnosis, current, history, last)
            if streak >= self.settings.repeat_error_limit:
                record.diagnosis = (f"The same error ({diag.category.value}) occurred {streak} times in a row; the "
                                    "corrections are not addressing its root cause.")
                return self._finish(LoopStatus.REPEATED_ERROR, record.diagnosis, current, history, last)
            if attempt == max_builds:
                return self._finish(LoopStatus.MAX_RETRIES, f"Build still failing after {self.settings.max_retries} "
                                    f"correction attempt(s) (limit MAX_RETRIES={self.settings.max_retries}).",
                                    current, history, last)

            self._emit("diagnosing", f"Asking the LLM for a root-cause diagnosis and fix ({diag.category.value})...",
                       attempt, last.failed_stage)
            known = self.knowledge(diag.category.value)[:3] if self.knowledge else []
            try:
                correction = self.generator.correct(analysis, history, current, last, diagnosis=diag,
                                                    repeat_count=streak, known_fixes=known)
            except (GenerationError, OllamaError) as exc:
                record.diagnosis = f"LLM could not produce a correction: {exc}"
                return self._finish(LoopStatus.NON_RECOVERABLE, record.diagnosis, current, history, last)

            record.diagnosis = correction.root_cause
            record.changes = list(correction.changes)
            record.explanation, record.expected_effect = correction.explanation, correction.expected_effect
            record.remaining_risks = list(correction.remaining_risks)
            if not correction.recoverable:
                record.diagnosis = correction.root_cause or "The LLM judged the failure unrecoverable."
                return self._finish(LoopStatus.NON_RECOVERABLE, f"LLM reports the failure cannot be fixed in the "
                                    f"Dockerfile: {record.diagnosis}", current, history, last)

            current = GeneratedArtifacts(
                dockerfile=correction.dockerfile, dockerignore=correction.dockerignore or current.dockerignore,
                language=correction.language if correction.language != "unknown" else current.language,
                framework=correction.framework or current.framework,
                base_image=correction.base_image or current.base_image,
                exposed_port=correction.exposed_port or current.exposed_port,
                explanation=correction.explanation or current.explanation, limitations=correction.limitations)
            record.diff = "\n".join(difflib.unified_diff(record.dockerfile.splitlines(), current.dockerfile.splitlines(),
                                                         f"attempt {attempt}", f"attempt {attempt + 1}", lineterm=""))
            write_artifacts(project_dir, current.dockerfile, current.dockerignore, overwrite=True, backup=False)
            self._emit("corrected", "Applied correction: " + ("; ".join(correction.changes[:3]) or "see history"),
                       attempt)

        return self._finish(LoopStatus.MAX_RETRIES, "Retry limit reached.", current, history, last)  # pragma: no cover

    def _finish(self, status: LoopStatus, message: str, artifacts: GeneratedArtifacts, history: ExecutionHistory,
                last: Optional[BuildResult]) -> LoopResult:
        self._emit("stopped", message, len(history.attempts))
        return LoopResult(status=status, message=message, artifacts=artifacts, history=history, last_build=last)

"""Iteration history for the build-and-fix loop, with JSON/Markdown export."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class AttemptRecord(BaseModel):
    """One Docker build attempt and what the LLM did about its outcome."""

    attempt: int
    dockerfile: str
    success: bool
    exit_code: Optional[int] = None
    failed_stage: Optional[str] = None
    error: str = ""            # detected build error (redacted, trimmed)
    diagnosis: str = ""        # LLM root-cause analysis of this failure
    changes: list[str] = Field(default_factory=list)  # fixes applied to produce the next Dockerfile
    duration_seconds: float = 0.0
    timed_out: bool = False
    category: str = ""             # error_classifier.ErrorCategory value
    fixable: Optional[bool] = None
    failing_instruction: Optional[str] = None
    signature: str = ""
    repeated: int = 1              # consecutive occurrences of this exact error, including this one
    explanation: str = ""          # LLM explanation of the fix
    expected_effect: str = ""
    remaining_risks: list[str] = Field(default_factory=list)
    diff: str = ""                 # unified diff from this attempt's Dockerfile to the corrected one
    started_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    @property
    def result_label(self) -> str:
        if self.success:
            return "success"
        return "timeout" if self.timed_out else "failed"


class ExecutionHistory(BaseModel):
    attempts: list[AttemptRecord] = Field(default_factory=list)

    def add(self, record: AttemptRecord) -> AttemptRecord:
        self.attempts.append(record)
        return record

    @property
    def last(self) -> Optional[AttemptRecord]:
        return self.attempts[-1] if self.attempts else None

    @property
    def corrections_applied(self) -> int:
        """Number of LLM correction rounds that were requested (builds after the first)."""
        return max(0, len(self.attempts) - 1)

    @property
    def total_duration(self) -> float:
        return round(sum(a.duration_seconds for a in self.attempts), 2)

    @property
    def succeeded(self) -> bool:
        return bool(self.attempts) and self.attempts[-1].success

    def previous_failures_summary(self) -> str:
        """Compact text of earlier failed attempts, used to stop the LLM repeating itself."""
        lines = []
        for a in self.attempts:
            if a.success:
                continue
            err = a.error.strip().splitlines()
            lines.append(f"- Attempt {a.attempt}: failed"
                         + (f" [{a.category}]" if a.category else "")
                         + (f" at stage '{a.failed_stage}'" if a.failed_stage else "")
                         + f". Error: {err[-1][:300] if err else 'n/a'}"
                         + (f"\n  Diagnosis: {a.diagnosis[:400]}" if a.diagnosis else "")
                         + (f"\n  Fix applied: {'; '.join(a.changes)[:400]}" if a.changes else ""))
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(self.model_dump(), indent=2)

    def to_markdown(self) -> str:
        out = ["# Build history", ""]
        for a in self.attempts:
            out += [f"## Attempt {a.attempt} - {a.result_label} ({a.duration_seconds:.1f}s)", ""]
            if a.failed_stage:
                out.append(f"- Failed stage: `{a.failed_stage}`")
            if a.exit_code is not None:
                out.append(f"- Exit code: {a.exit_code}")
            if a.category:
                out.append(f"- Category: `{a.category}`" + ("" if a.fixable is None else
                                                          f" (fixable by Dockerfile edit: {'yes' if a.fixable else 'no'})"))
            if a.diagnosis:
                out.append(f"- Diagnosis: {a.diagnosis}")
            if a.expected_effect:
                out.append(f"- Expected effect: {a.expected_effect}")
            for r in a.remaining_risks:
                out.append(f"- Remaining risk: {r}")
            for c in a.changes:
                out.append(f"- Change: {c}")
            if a.error:
                out += ["", "```text", a.error.strip(), "```"]
            if a.diff:
                out += ["", "```diff", a.diff.rstrip(), "```"]
            out += ["", "<details><summary>Dockerfile</summary>", "", "```dockerfile", a.dockerfile.rstrip(), "```",
                    "", "</details>", ""]
        return "\n".join(out)

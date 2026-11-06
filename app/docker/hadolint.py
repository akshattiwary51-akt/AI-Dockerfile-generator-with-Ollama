"""Optional Hadolint integration. When the binary is missing the caller falls back to the internal validator."""
from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any, Callable, Optional

from pydantic import BaseModel

SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2, "style": 3}


class HadolintFinding(BaseModel):
    code: str
    level: str
    line: Optional[int] = None
    message: str


def hadolint_available() -> bool:
    return shutil.which("hadolint") is not None


def run_hadolint(dockerfile: str, *, runner: Callable[..., Any] = subprocess.run,
                 which: Callable[[str], Optional[str]] = shutil.which) -> Optional[list[HadolintFinding]]:
    """Lint with Hadolint via stdin. Returns ``None`` when Hadolint is unavailable or fails to run."""
    exe = which("hadolint")
    if not exe:
        return None
    try:
        proc = runner([exe, "--format", "json", "-"], input=dockerfile, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    # exit code 1 simply means "findings were reported"; anything without parsable JSON is a tool failure
    try:
        data = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    findings = [HadolintFinding(code=str(d.get("code", "?")), level=str(d.get("level", "info")),
                                line=d.get("line"), message=str(d.get("message", ""))) for d in data
                if isinstance(d, dict)]
    return sorted(findings, key=lambda f: (SEVERITY_ORDER.get(f.level, 9), f.line or 0))

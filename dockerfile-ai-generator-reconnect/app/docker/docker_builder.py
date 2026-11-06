"""Run ``docker build`` safely, capture its output and extract the failure cause."""
from __future__ import annotations

import re
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Optional

from pydantic import BaseModel, Field

from app.utils.config import Settings, get_settings
from app.utils.logger import get_logger
from app.utils.security import redact_secrets, truncate_log

log = get_logger("dockerfile_ai.builder")

DOCKER_INSTALL_HINT = ("Docker is not available. Install Docker Engine/Desktop (https://docs.docker.com/get-docker/), "
                       "start the daemon (e.g. `sudo systemctl start docker`) and verify with `docker version`.")

# Failures no Dockerfile edit can fix -> the loop must stop instead of burning retries.
NON_RECOVERABLE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"cannot connect to the docker daemon|failed to connect to the docker api|is the docker daemon running"
                r"|error during connect", re.I), "The Docker daemon is not reachable."),
    (re.compile(r"permission denied while trying to connect to the docker", re.I),
     "Permission denied for the Docker socket (add your user to the 'docker' group or use rootless Docker)."),
    (re.compile(r"no space left on device", re.I), "The host is out of disk space."),
    (re.compile(r"toomanyrequests|rate limit", re.I), "The registry rate-limited the pull (try `docker login`)."),
    (re.compile(r"temporary failure in name resolution|no such host|tls handshake timeout|network is unreachable"
                r"|i/o timeout.*(?:registry|docker\.io)|dial tcp.*(?:i/o timeout|connection refused)", re.I),
     "A network problem prevented reaching a registry/package index."),
]


class BuildResult(BaseModel):
    success: bool
    image: str
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    failed_stage: Optional[str] = None
    error_summary: str = ""
    duration_seconds: float = 0.0
    timed_out: bool = False
    cancelled: bool = False
    non_recoverable_reason: Optional[str] = None
    command: list[str] = Field(default_factory=list)

    @property
    def combined_output(self) -> str:
        return (self.stdout + "\n" + self.stderr).strip()


def sanitize_image_name(project_name: str) -> str:
    base = re.sub(r"[^a-z0-9._-]+", "-", project_name.lower()).strip("-._") or "app"
    return f"{base[:60]}:ai-generated"


def classify_failure(output: str) -> Optional[str]:
    for pattern, reason in NON_RECOVERABLE_PATTERNS:
        if pattern.search(output):
            return reason
    return None


def extract_build_error(output: str, max_lines: int = 40) -> tuple[Optional[str], str]:
    """Return ``(failed_stage, error_summary)`` from BuildKit (plain) or legacy builder output."""
    lines = output.splitlines()
    stage: Optional[str] = None
    summary: list[str] = []

    # BuildKit: "#7 ERROR: process ... did not complete successfully"
    err_idx = next((i for i, ln in enumerate(lines) if re.match(r"^#\d+\s+ERROR:", ln)), None)
    if err_idx is not None:
        step = re.match(r"^#(\d+)\s+ERROR:", lines[err_idx]).group(1)
        step_lines = [ln for ln in lines if ln.startswith(f"#{step} ")]
        header = next((ln for ln in step_lines if re.match(rf"^#{step}\s+\[", ln)), None)
        if header:
            stage = re.sub(rf"^#{step}\s+", "", header).strip()
        summary = [re.sub(rf"^#{step}\s+", "", ln) for ln in step_lines][-max_lines:]
    else:
        # Legacy builder: "Step 5/9 : RUN npm ci" + "The command ... returned a non-zero code"
        steps = [ln for ln in lines if re.match(r"^Step \d+/\d+\s*:", ln)]
        if steps and any("returned a non-zero code" in ln or ln.startswith("ERROR") for ln in lines):
            stage = steps[-1].strip()
            summary = lines[-max_lines:]

    # Parse errors, e.g. "ERROR: failed to solve: dockerfile parse error on line 3: ..."
    solve = [ln.strip() for ln in lines if re.search(r"failed to solve|dockerfile parse error|unknown instruction",
                                                      ln, re.I)]
    # BuildKit prints the offending Dockerfile excerpt between ------ markers
    block: list[str] = []
    for i, ln in enumerate(lines):
        if re.match(r"^Dockerfile:\d+\s*$", ln.strip()):
            block = [x for x in lines[i:i + 12]]
            if stage is None:
                stage = ln.strip()
            break
    for extra in (block, solve):
        for ln in extra:
            if ln not in summary:
                summary.append(ln)
    if not summary:
        summary = [ln for ln in lines if ln.strip()][-20:]
    return stage, redact_secrets("\n".join(summary)).strip()


class _BoundedLog:
    """Keeps the head and tail of a stream within a byte budget."""

    def __init__(self, max_bytes: int) -> None:
        self.head_budget = max_bytes // 2
        self.tail_budget = max_bytes - self.head_budget
        self.head: list[str] = []
        self.head_bytes = 0
        self.tail: deque[str] = deque()
        self.tail_bytes = 0
        self.dropped = 0

    def add(self, line: str) -> None:
        size = len(line.encode("utf-8", errors="replace")) + 1
        if self.head_bytes + size <= self.head_budget and not self.tail:
            self.head.append(line)
            self.head_bytes += size
            return
        self.tail.append(line)
        self.tail_bytes += size
        while self.tail_bytes > self.tail_budget and self.tail:
            old = self.tail.popleft()
            self.tail_bytes -= len(old.encode("utf-8", errors="replace")) + 1
            self.dropped += 1

    def text(self) -> str:
        mid = [f"... [{self.dropped} log line(s) omitted] ..."] if self.dropped else []
        return "\n".join([*self.head, *mid, *self.tail])


class DockerBuilder:
    """Wraps ``docker build``. Never privileged, never mounts anything, never runs the image."""

    def __init__(self, settings: Optional[Settings] = None,
                 popen: Callable[..., "subprocess.Popen[str]"] = subprocess.Popen,
                 run: Callable[..., "subprocess.CompletedProcess[str]"] = subprocess.run) -> None:
        self.settings = settings or get_settings()
        self._popen = popen
        self._run = run

    def check_docker(self) -> tuple[bool, str]:
        try:
            proc = self._run(["docker", "version", "--format", "{{.Server.Version}}"], capture_output=True,
                             text=True, timeout=20)
        except FileNotFoundError:
            return False, "The `docker` CLI was not found. " + DOCKER_INSTALL_HINT
        except subprocess.TimeoutExpired:
            return False, "`docker version` timed out; the daemon may be hung."
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip().splitlines()
            return False, f"Docker daemon not reachable: {detail[0] if detail else 'unknown error'}. " \
                          + DOCKER_INSTALL_HINT
        return True, f"Docker Engine {proc.stdout.strip()}"

    def build(self, project_dir: Path | str, image: str, *, on_output: Optional[Callable[[str], None]] = None,
              cancel_event: Optional[threading.Event] = None) -> BuildResult:
        project = Path(project_dir).resolve()
        cmd = ["docker", "build", "--progress=plain", "-t", image, "."]
        started = time.monotonic()
        if not (project / "Dockerfile").is_file():
            return BuildResult(success=False, image=image, command=cmd, exit_code=None,
                               error_summary=f"No Dockerfile found in {project}.",
                               non_recoverable_reason="Dockerfile missing")
        out_log = _BoundedLog(self.settings.max_log_bytes)
        err_log = _BoundedLog(self.settings.max_log_bytes)
        log.info("Starting build: %s (cwd=%s)", " ".join(cmd), project)
        try:
            proc = self._popen(cmd, cwd=str(project), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                               encoding="utf-8", errors="replace", bufsize=1, stdin=subprocess.DEVNULL,
                               env=self._env())
        except FileNotFoundError:
            return BuildResult(success=False, image=image, command=cmd, error_summary=DOCKER_INSTALL_HINT,
                               non_recoverable_reason="The `docker` CLI was not found.",
                               duration_seconds=round(time.monotonic() - started, 2))
        except OSError as exc:
            return BuildResult(success=False, image=image, command=cmd, error_summary=f"Cannot start docker: {exc}",
                               non_recoverable_reason=f"Cannot start docker: {exc}",
                               duration_seconds=round(time.monotonic() - started, 2))

        def pump(stream, sink: _BoundedLog) -> None:
            try:
                for raw in iter(stream.readline, ""):
                    line = redact_secrets(raw.rstrip("\n"))
                    sink.add(line)
                    if on_output:
                        try:
                            on_output(line)
                        except Exception:  # UI callbacks must never kill the build thread
                            log.debug("on_output callback failed", exc_info=True)
            except (ValueError, OSError):
                pass

        threads = [threading.Thread(target=pump, args=(proc.stdout, out_log), daemon=True),
                   threading.Thread(target=pump, args=(proc.stderr, err_log), daemon=True)]
        for t in threads:
            t.start()

        timed_out = cancelled = False
        deadline = started + self.settings.docker_build_timeout
        while proc.poll() is None:
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                self._stop(proc)
                break
            if time.monotonic() > deadline:
                timed_out = True
                self._stop(proc)
                break
            time.sleep(0.05)
        for t in threads:
            t.join(timeout=5)
        exit_code = proc.poll()
        stdout = truncate_log(out_log.text(), self.settings.max_log_bytes)
        stderr = truncate_log(err_log.text(), self.settings.max_log_bytes)
        duration = round(time.monotonic() - started, 2)
        success = exit_code == 0 and not timed_out and not cancelled
        result = BuildResult(success=success, image=image, exit_code=exit_code, stdout=stdout, stderr=stderr,
                             duration_seconds=duration, timed_out=timed_out, cancelled=cancelled, command=cmd)
        if not success:
            combined = result.combined_output
            result.failed_stage, result.error_summary = extract_build_error(combined)
            if timed_out:
                result.error_summary = (f"Build exceeded the {self.settings.docker_build_timeout}s timeout.\n"
                                        + result.error_summary)
            if cancelled:
                result.error_summary = "Build cancelled by the user."
            result.non_recoverable_reason = classify_failure(combined) if not (timed_out or cancelled) else None
        log.info("Build finished: success=%s exit=%s in %.1fs", success, exit_code, duration)
        return result

    @staticmethod
    def _stop(proc: "subprocess.Popen[str]") -> None:
        try:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        except (OSError, ProcessLookupError):
            pass

    @staticmethod
    def _env() -> dict[str, str]:
        import os
        env = dict(os.environ)
        env.setdefault("DOCKER_BUILDKIT", "1")
        return env

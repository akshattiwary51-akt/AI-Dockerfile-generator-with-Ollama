"""High-level workflow: analyze -> generate -> save (with approval) -> build-and-fix -> report."""
from __future__ import annotations

import difflib
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from pydantic import BaseModel, Field

from app.agent.execution_history import ExecutionHistory
from app.agent.feedback_loop import FeedbackLoop, LoopEvent, LoopResult, LoopStatus
from app.analyzer.models import ProjectAnalysis
from app.analyzer.project_scanner import ProjectScanner
from app.docker.docker_builder import DockerBuilder, sanitize_image_name
from app.docker.docker_validator import validate_artifacts
from app.docker.dockerfile_generator import DockerfileGenerator, FileProtectionError, existing_targets, write_artifacts
from app.llm.model_advisor import recommend_models
from app.llm.ollama_client import ModelNotFoundError, OllamaClient, OllamaError, OllamaUnavailableError
from app.llm.response_parser import GeneratedArtifacts
from app.utils.config import Settings, get_settings
from app.utils.logger import get_logger
from app.utils.security import redact_secrets

log = get_logger("dockerfile_ai.orchestrator")

BUILD_WARNING = ("Building a Dockerfile executes the instructions it contains (RUN commands, package scripts, "
                 "downloads). Only build projects you trust; the generated Dockerfile is produced by an LLM and "
                 "may contain mistakes or unsafe steps. Review it before approving the build.")
GENERATED_DIR = Path(__file__).resolve().parents[2] / "generated"


class ApprovalRequired(PermissionError):
    """An action needs explicit user approval that was not given."""


class EnvironmentStatus(BaseModel):
    ollama_available: bool = False
    ollama_message: str = ""
    model: str = ""
    model_installed: bool = False
    installed_models: list[str] = Field(default_factory=list)
    docker_available: bool = False
    docker_message: str = ""
    suggested_models: list[str] = Field(default_factory=list)

    @property
    def ready_to_generate(self) -> bool:
        return self.ollama_available and self.model_installed


class RunReport(BaseModel):
    project: str
    project_dir: str
    image: str
    model: str
    success: bool
    status: str
    message: str
    attempts: int
    corrections: int
    total_duration_seconds: float
    files: list[str] = Field(default_factory=list)
    dockerfile: str = ""
    dockerignore: str = ""
    quality_score: int = 0
    suggestions: list[str] = Field(default_factory=list)
    image_size_mb: Optional[float] = None
    history: ExecutionHistory = Field(default_factory=ExecutionHistory)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def to_markdown(self) -> str:
        head = [f"# Dockerfile AI report - {self.project}", "",
                f"- Result: **{'SUCCESS' if self.success else 'FAILED'}** ({self.status})",
                f"- Image: `{self.image}`", f"- Model: `{self.model}`",
                f"- Build attempts: {self.attempts} ({self.corrections} correction(s))",
                f"- Total build time: {self.total_duration_seconds:.1f}s",
                f"- Quality score: {self.quality_score}/100"]
        if self.image_size_mb is not None:
            head.append(f"- Image size: {self.image_size_mb:.0f} MB")
        head += [f"- Message: {self.message}", "", "## Files"] + [f"- {f}" for f in self.files]
        if self.suggestions:
            head += ["", "## Suggestions"] + [f"- {s}" for s in self.suggestions]
        head += ["", "## Final Dockerfile", "```dockerfile", self.dockerfile.rstrip(), "```", "",
                 self.history.to_markdown()]
        return "\n".join(head)


class Orchestrator:
    def __init__(self, settings: Optional[Settings] = None, *, client: Optional[OllamaClient] = None,
                 builder: Optional[DockerBuilder] = None, scanner: Optional[ProjectScanner] = None) -> None:
        self.settings = settings or get_settings()
        self.client = client or OllamaClient(self.settings)
        self.builder = builder or DockerBuilder(self.settings)
        self.scanner = scanner or ProjectScanner(self.settings)
        self.generator = DockerfileGenerator(self.client, self.settings)

    # ------------------------------------------------------------------ environment
    def check_environment(self) -> EnvironmentStatus:
        st = EnvironmentStatus(model=self.client.model)
        try:
            st.installed_models = self.client.list_models()
            st.ollama_available = True
            st.ollama_message = f"Ollama reachable at {self.client.host}"
            st.model_installed = self.client.has_model()
            if not st.model_installed:
                st.ollama_message += f"; model '{self.client.model}' is not installed (run: ollama pull {self.client.model})"
        except OllamaUnavailableError as exc:
            st.ollama_message = str(exc)
        except OllamaError as exc:
            st.ollama_message = str(exc)
        st.docker_available, st.docker_message = self.builder.check_docker()
        try:
            st.suggested_models = [f"{m} - {why}" for m, why in recommend_models()[:3]]
        except Exception:  # advisory only
            pass
        return st

    # ------------------------------------------------------------------ steps
    def analyze(self, project_dir: Path | str) -> ProjectAnalysis:
        return self.scanner.scan(project_dir)

    def generate(self, analysis: ProjectAnalysis) -> GeneratedArtifacts:
        self.client.ensure_ready()  # raises OllamaUnavailableError / ModelNotFoundError with instructions
        return self.generator.generate(analysis)

    def existing_files(self, project_dir: Path | str) -> list[str]:
        return existing_targets(project_dir)

    def diff_with_existing(self, project_dir: Path | str, artifacts: GeneratedArtifacts) -> str:
        path = Path(project_dir) / "Dockerfile"
        if not path.is_file():
            return ""
        old = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(difflib.unified_diff(old, artifacts.dockerfile.splitlines(), "Dockerfile (existing)",
                                              "Dockerfile (generated)", lineterm=""))

    def save(self, project_dir: Path | str, artifacts: GeneratedArtifacts, *, approve_overwrite: bool = False
             ) -> list[Path]:
        """Validate, then write. Existing files are never replaced without ``approve_overwrite``."""
        report = validate_artifacts(artifacts.dockerfile, artifacts.dockerignore)
        if not report.ok:
            raise ValueError("Refusing to save an invalid Dockerfile:\n" + report.format())
        return write_artifacts(project_dir, artifacts.dockerfile, artifacts.dockerignore, overwrite=approve_overwrite)

    def build_and_fix(self, project_dir: Path | str, analysis: ProjectAnalysis, artifacts: GeneratedArtifacts, *,
                      approve_build: bool, on_event: Optional[Callable[[LoopEvent], None]] = None,
                      on_log: Optional[Callable[[str], None]] = None,
                      cancel_event: Optional[threading.Event] = None,
                      history: Optional[ExecutionHistory] = None) -> RunReport:
        if not approve_build:
            raise ApprovalRequired("Docker build not approved. " + BUILD_WARNING)
        image = sanitize_image_name(analysis.name)
        loop = FeedbackLoop(self.generator, self.builder, self.settings, on_event=on_event, on_log=on_log,
                            cancel_event=cancel_event)
        started = time.monotonic()
        result = loop.run(analysis, project_dir, artifacts, image, history=history)
        log.info("Run finished in %.1fs: %s", time.monotonic() - started, result.status.value)
        return self._report(project_dir, analysis, image, result)

    def run_full(self, project_dir: Path | str, *, approve_overwrite: bool = False, approve_build: bool = False,
                 on_event: Optional[Callable[[LoopEvent], None]] = None,
                 on_log: Optional[Callable[[str], None]] = None,
                 cancel_event: Optional[threading.Event] = None) -> RunReport:
        """Non-interactive pipeline. Approvals must be passed in explicitly by the caller."""
        if not approve_build:
            raise ApprovalRequired("Docker build not approved. " + BUILD_WARNING)
        analysis = self.analyze(project_dir)
        artifacts = self.generate(analysis)
        self.save(project_dir, artifacts, approve_overwrite=approve_overwrite)
        return self.build_and_fix(project_dir, analysis, artifacts, approve_build=True, on_event=on_event,
                                  on_log=on_log, cancel_event=cancel_event)

    # ------------------------------------------------------------------ reporting
    def _report(self, project_dir: Path | str, analysis: ProjectAnalysis, image: str, result: LoopResult) -> RunReport:
        art = result.artifacts
        validation = validate_artifacts(art.dockerfile, art.dockerignore)
        suggestions = [i.message for i in validation.warnings]
        if analysis.secret_env_vars:
            suggestions.append("Provide secret variables at runtime (" + ", ".join(analysis.secret_env_vars[:5])
                               + "); they are intentionally not baked into the image.")
        if result.success and art.base_image and "alpine" not in art.base_image and "slim" not in art.base_image:
            suggestions.append("Consider a slim/alpine/distroless base image to reduce image size.")
        suggestions.append("Run a vulnerability scan on the image (e.g. `docker scout cves` or `trivy image`).")
        suggestions.extend(f"Assumption/limitation: {x}" for x in art.limitations[:3])
        size = self._image_size(image) if result.success else None
        root = Path(project_dir).resolve()
        files = [str(root / n) for n in ("Dockerfile", ".dockerignore") if (root / n).exists()]
        return RunReport(
            project=analysis.name, project_dir=str(root), image=image, model=self.client.model,
            success=result.success, status=result.status.value, message=redact_secrets(result.message),
            attempts=len(result.history.attempts), corrections=result.history.corrections_applied,
            total_duration_seconds=result.history.total_duration, files=files, dockerfile=art.dockerfile,
            dockerignore=art.dockerignore, quality_score=validation.score, suggestions=suggestions,
            image_size_mb=size, history=result.history)

    def _image_size(self, image: str) -> Optional[float]:
        try:
            proc = self.builder._run(["docker", "image", "inspect", "-f", "{{.Size}}", image], capture_output=True,
                                     text=True, timeout=20)
            return round(int(proc.stdout.strip()) / 1_000_000, 1) if proc.returncode == 0 else None
        except Exception:
            return None

    def export_report(self, report: RunReport, directory: Path | str = GENERATED_DIR) -> list[Path]:
        """Write JSON + Markdown reports to the tool's own ``generated/`` directory (never the project)."""
        out = Path(directory) / report.project
        out.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        paths = [out / f"report-{stamp}.json", out / f"report-{stamp}.md"]
        paths[0].write_text(redact_secrets(report.model_dump_json(indent=2)), encoding="utf-8")
        paths[1].write_text(redact_secrets(report.to_markdown()), encoding="utf-8")
        return paths

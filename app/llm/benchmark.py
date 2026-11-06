"""Compare local models on the same project: latency, throughput and the validity/quality of what they produce."""
from __future__ import annotations

import time
from typing import Any, Callable, Optional

from pydantic import BaseModel

from app.analyzer.models import ProjectAnalysis
from app.docker.docker_validator import validate_artifacts
from app.docker.quality import score_dockerfile
from app.docker.dockerfile_generator import DockerfileGenerator, GenerationError
from app.llm.ollama_client import OllamaClient, OllamaError
from app.utils.config import Settings, get_settings
from app.utils.logger import get_logger

log = get_logger("dockerfile_ai.benchmark")


class BenchResult(BaseModel):
    model: str
    ok: bool
    error: str = ""
    latency_seconds: float = 0.0
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_per_second: Optional[float] = None
    format_retries: int = 0      # extra round-trips needed to get valid structured output
    lint_score: int = 0          # static quality score of the produced Dockerfile (0-100, see docker/quality.py)
    errors: int = 0
    warnings: int = 0
    tokens_estimated: bool = False

    @property
    def rank_key(self) -> tuple:
        # valid output first, then fewer format retries, then quality, then speed
        return (not self.ok, self.format_retries, -self.lint_score, self.latency_seconds)


def benchmark_models(analysis: ProjectAnalysis, models: list[str], settings: Optional[Settings] = None, *,
                     client_factory: Optional[Callable[[Settings], Any]] = None,
                     on_progress: Optional[Callable[[str], None]] = None) -> list[BenchResult]:
    """Generate a Dockerfile for ``analysis`` with every model in ``models`` and return ranked results.

    Each model is exercised once; nothing is built or written to the project.
    """
    base = settings or get_settings()
    factory = client_factory or (lambda s: OllamaClient(s))
    results: list[BenchResult] = []
    for name in models:
        if on_progress:
            on_progress(f"Benchmarking {name} ...")
        s = base.model_copy(update={"ollama_model": name})
        client = factory(s)
        gen = DockerfileGenerator(client, s)
        started = time.monotonic()
        res = BenchResult(model=name, ok=False)
        try:
            client.ensure_ready(name)
            art = gen.generate(analysis)
            report = validate_artifacts(art.dockerfile, art.dockerignore)
            res.ok = True
            res.lint_score = score_dockerfile(art.dockerfile, art.dockerignore, analysis=analysis,
                                              use_hadolint=False).overall
            res.errors, res.warnings = len(report.errors), len(report.warnings)
        except (OllamaError, GenerationError) as exc:
            res.error = str(exc).splitlines()[0][:300]
            log.warning("Benchmark failed for %s: %s", name, res.error)
        res.latency_seconds = round(time.monotonic() - started, 2)
        res.format_retries = getattr(gen, "format_retries", 0)
        log_ = getattr(client, "stats_log", []) or []
        res.llm_calls = len(log_)
        res.prompt_tokens = sum(x.prompt_tokens for x in log_)
        res.completion_tokens = sum(x.completion_tokens for x in log_)
        res.tokens_estimated = any(x.estimated for x in log_)
        gen_secs = sum(x.completion_tokens / x.tokens_per_second for x in log_ if x.tokens_per_second)
        res.tokens_per_second = round(res.completion_tokens / gen_secs, 1) if gen_secs else None
        results.append(res)
    return sorted(results, key=lambda r: r.rank_key)

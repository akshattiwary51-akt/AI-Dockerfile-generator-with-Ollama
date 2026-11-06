"""Reject corrections that would silently make the Dockerfile worse than what it replaces."""
from __future__ import annotations

from typing import Optional

from app.analyzer.models import ProjectAnalysis
from app.docker.docker_validator import parse_instructions
from app.docker.quality import _stages, score_dockerfile
from app.llm.response_parser import GeneratedArtifacts
from app.utils.dockerignore import is_ignored


def _features(dockerfile: str) -> dict[str, bool]:
    instrs, _ = parse_instructions(dockerfile)
    stages = _stages(instrs)
    final = stages[-1] if stages else []
    users = [i.args.strip().lower() for i in final if i.name == "USER"]
    return {"non-root USER": bool(users) and users[-1] not in ("root", "0"),
            "HEALTHCHECK": any(i.name == "HEALTHCHECK" for i in final),
            "EXPOSE": any(i.name == "EXPOSE" for i in final),
            "multi-stage build": len(stages) > 1}


def find_regressions(old: GeneratedArtifacts, new: GeneratedArtifacts, analysis: Optional[ProjectAnalysis] = None
                     ) -> list[str]:
    """Human-readable reasons why ``new`` is inferior to ``old`` (empty list = acceptable)."""
    problems: list[str] = []
    root = analysis.root if analysis else None
    old_q = score_dockerfile(old.dockerfile, old.dockerignore, analysis=analysis, project_root=root, use_hadolint=False)
    new_q = score_dockerfile(new.dockerfile, new.dockerignore, analysis=analysis, project_root=root, use_hadolint=False)
    known = {(f.code, f.message) for f in old_q.findings if f.severity == "high"}
    for f in new_q.findings:
        if f.severity == "high" and f.category in ("security", "runtime") and (f.code, f.message) not in known:
            problems.append(f"introduces a new high-severity problem ({f.code}): {f.message}")
    before, after = _features(old.dockerfile), _features(new.dockerfile)
    for name, had in before.items():
        if had and not after[name]:
            problems.append(f"removes the {name} that the previous Dockerfile had")
    if new_q.categories["security"].score <= old_q.categories["security"].score - 15:
        problems.append(f"lowers the security score from {old_q.categories['security'].score} to "
                        f"{new_q.categories['security'].score}")
    if old.dockerignore and is_ignored(".env", old.dockerignore) and not is_ignored(".env", new.dockerignore):
        problems.append(".dockerignore no longer excludes .env files")
    return problems

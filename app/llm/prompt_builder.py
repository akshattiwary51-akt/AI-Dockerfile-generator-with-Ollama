"""Prompt engineering for Dockerfile generation and targeted build-failure correction."""
from __future__ import annotations

from typing import Any, Optional

from app.agent.execution_history import ExecutionHistory
from app.analyzer.models import ProjectAnalysis
from app.llm.response_parser import CorrectionResult, GeneratedArtifacts
from app.utils.config import Settings, get_settings
from app.utils.security import redact_secrets

BEST_PRACTICES = """\
Dockerfile requirements:
- Use an official, version-pinned base image (never `latest`); prefer slim/alpine variants that fit the stack.
- Use a multi-stage build when it shrinks the final image (compiled/bundled apps); keep it single-stage otherwise.
- Order instructions for layer caching: copy dependency manifests/lockfiles first, install, then copy the sources.
- Reproducible installs: `npm ci` / `yarn --frozen-lockfile` / `pnpm install --frozen-lockfile` when a lockfile exists,
  `pip install --no-cache-dir -r requirements.txt`, `go mod download`, etc.
- Run as a non-root user where practical (create it, `chown` what it needs, then `USER`).
- Install no unnecessary packages; clean package-manager caches in the same RUN layer.
- Never embed secrets. No secret ENV/ARG values; secrets are supplied at runtime. Never COPY .env files.
- Declare configurable settings with ENV defaults that are NOT secret (for example PORT).
- `EXPOSE` the application port and use the correct production start command in exec form (CMD [...]).
- Add a HEALTHCHECK only when the image contains a tool that can run it (slim images lack curl/wget; use a tiny
  python/node one-liner, or busybox `wget` on alpine). Skip it for non-HTTP programs.
- Production configuration: NODE_ENV=production, PYTHONUNBUFFERED=1, PYTHONDONTWRITEBYTECODE=1 where relevant.
- Forbidden: `--privileged`, mounting the Docker socket, `curl | sh`, `ADD` of remote URLs, `chmod 777`.
.dockerignore requirements: exclude VCS data, dependency directories (node_modules, venvs), build output, caches,
tests/docs not needed at runtime, `.env*` files, private keys and Docker files themselves."""

GENERATION_SCHEMA_HINT = """\
Respond with ONE JSON object and nothing else (no prose, no markdown fences) using exactly these keys:
{
  "dockerfile": "<full Dockerfile text>",
  "dockerignore": "<full .dockerignore text>",
  "language": "<detected language>",
  "framework": "<detected framework or null>",
  "base_image": "<final runtime base image tag>",
  "exposed_port": <integer or null>,
  "explanation": "<important design decisions, concise>",
  "limitations": ["<assumptions or limitations>"]
}"""

CORRECTION_SCHEMA_HINT = """\
Respond with ONE JSON object and nothing else (no prose, no markdown fences) using exactly these keys:
{
  "root_cause": "<what actually caused the failure, referring to the failing instruction and error text>",
  "explanation": "<why this root cause produces this error and why your change addresses it>",
  "changes": ["<each exact Dockerfile change you made, one per item, e.g. 'COPY requirement.txt -> requirements.txt'>"],
  "expected_effect": "<what the rebuilt image should do differently>",
  "remaining_risks": ["<anything that may still fail or that you could not verify>"],
  "recoverable": <true if a Dockerfile/.dockerignore change can fix it, false otherwise>,
  "dockerfile": "<the FULL corrected Dockerfile>",
  "dockerignore": "<the FULL .dockerignore>",
  "language": "<language>",
  "framework": "<framework or null>",
  "base_image": "<final runtime base image tag>",
  "exposed_port": <integer or null>,
  "explanation": "<short explanation of the fix>",
  "limitations": ["<remaining assumptions or risks>"]
}"""

GENERATION_SYSTEM = (
    "You are a senior DevOps engineer who writes production-grade, secure, minimal Dockerfiles. "
    "You are given a static analysis of a software project (facts extracted by tools, not guesses) and must "
    "produce a Dockerfile and .dockerignore that build successfully from the project directory as build context. "
    "Rely on the provided facts; if something is unknown, make the most conservative assumption and list it in "
    '"limitations". Secrets are never present in the context and must never be invented or embedded.\n\n'
    + BEST_PRACTICES + "\n\n" + GENERATION_SCHEMA_HINT)

CORRECTION_SYSTEM = (
    "You are a senior DevOps engineer fixing a Dockerfile that failed `docker build`. "
    "First identify the ROOT CAUSE from the build log, then apply the smallest TARGETED change that fixes it. "
    "Rules:\n"
    "- Keep every part of the Dockerfile that is not implicated by the error; do NOT rewrite it from scratch.\n"
    "- Do not repeat a fix listed under previous attempts - it already failed; try a different root-cause hypothesis.\n"
    "- Reference files that actually exist in the provided project structure; do not invent paths.\n"
    "- If the failure cannot be solved by editing the Dockerfile/.dockerignore (Docker daemon problems, no network, "
    "disk full, missing source files, registry authentication), set recoverable to false and explain.\n"
    "- Never add secrets, privileged mode, Docker socket mounts or `curl | sh`.\n\n"
    + BEST_PRACTICES + "\n\n" + CORRECTION_SCHEMA_HINT)


# ----------------------------------------------------------------------------- context rendering
def _facts(analysis: ProjectAnalysis) -> str:
    u = analysis.primary_unit
    lines = [f"Project name: {analysis.name}",
             f"Language: {analysis.language}",
             f"Framework: {analysis.framework or 'none detected'}"]
    if u:
        lines += [f"Role: {u.role}", f"Unit path (relative to build context): {u.path}",
                  f"Package manager: {u.package_manager or 'unknown'}",
                  f"Runtime version constraint: {u.runtime_version or 'not specified'}",
                  f"Lockfile: {u.lockfile or 'none'}",
                  f"Manifests: {', '.join(u.manifests) or 'none'}",
                  f"Entry points: {', '.join(u.entry_points) or 'unknown'}",
                  f"Application ports: {', '.join(map(str, u.ports)) or 'unknown'}"]
        if u.build_output_dir:
            lines.append(f"Build output directory: {u.build_output_dir}")
        if u.scripts:
            scripts = "; ".join(f"{k}: {v}" for k, v in list(u.scripts.items())[:10])
            lines.append(f"Scripts: {scripts}")
        deps = [d for d in u.dependencies][:40]
        if deps:
            more = f" (+{len(u.dependencies) - 40} more)" if len(u.dependencies) > 40 else ""
            lines.append(f"Dependencies ({len(u.dependencies)}): {', '.join(deps)}{more}")
        lines += [f"Note: {n}" for n in u.notes]
        if u.confirmed:
            lines.append("CONFIRMED (read from project files): " + "; ".join(u.confirmed))
        if u.inferred:
            lines.append("ASSUMPTIONS (not verified - mention them in limitations): " + "; ".join(u.inferred))
    if analysis.services:
        lines.append("Backing services the app appears to use (not part of this image): "
                     + ", ".join(f"{k} ({v})" for k, v in analysis.services.items()))
    if analysis.existing_compose:
        lines.append("Existing compose files: " + "; ".join(f"{f} -> services {', '.join(n) or '?'}"
                                                             for f, n in analysis.existing_compose.items()))
    if analysis.env_vars:
        plain = [e for e in analysis.env_vars if e not in analysis.secret_env_vars]
        lines.append(f"Environment variables used (names only): {', '.join(plain) or 'none'}")
        if analysis.secret_env_vars:
            lines.append("Secret-like variables (must be provided at runtime, NEVER set in the image): "
                         + ", ".join(analysis.secret_env_vars))
    if analysis.is_monorepo:
        lines.append("Monorepo/multi-unit layout: " + ", ".join(
            f"{x.path} ({x.framework or x.language}, {x.role})" for x in analysis.units))
    if analysis.has_frontend_backend_split:
        lines.append("Frontend and backend are separate units.")
    lines += [f"Warning: {w}" for w in analysis.warnings]
    return "\n".join(lines)


def _strategy(analysis: ProjectAnalysis) -> str:
    s = analysis.strategy
    if not s:
        return "none"
    out = [f"Suggested build/base image: {s.base_image}"]
    if s.runtime_image:
        out.append(f"Suggested runtime image: {s.runtime_image}")
    out.append(f"Multi-stage recommended: {'yes' if s.multi_stage else 'no'}")
    if s.start_command_hint:
        out.append(f"Start command hint: {s.start_command_hint}")
    out += [f"- {n}" for n in s.notes]
    return "\n".join(out)


def render_project_context(analysis: ProjectAnalysis, max_chars: int, *, include_files: bool = True) -> str:
    """Render analysis as text within ``max_chars``; low-priority sections are trimmed first."""
    parts: list[str] = [
        "## PROJECT FACTS\n" + _facts(analysis),
        "## SUGGESTED CONTAINERIZATION STRATEGY (starting point, verify against facts)\n" + _strategy(analysis)]
    text = "\n\n".join(parts)
    budget = max_chars - len(text)
    if include_files and analysis.key_files and budget > 300:
        chunks = []
        per_file = max(300, budget // max(1, len(analysis.key_files)) - 60)
        for name, content in analysis.key_files.items():
            body = content if len(content) <= per_file else content[:per_file] + "\n... [truncated]"
            chunks.append(f"### {name}\n```\n{body}\n```")
        block = "## KEY FILES\n" + "\n".join(chunks)
        if len(block) <= budget:
            text += "\n\n" + block
            budget -= len(block) + 2
    if analysis.structure and budget > 200:
        tree = analysis.structure
        if len(tree) + 40 > budget:
            tree = tree[: max(0, budget - 60)].rsplit("\n", 1)[0] + "\n... [structure truncated]"
        text += "\n\n## PROJECT STRUCTURE (files that exist in the build context)\n" + tree
    return redact_secrets(text)


# ----------------------------------------------------------------------------- message builders
def build_generation_messages(analysis: ProjectAnalysis, settings: Optional[Settings] = None) -> list[dict[str, str]]:
    settings = settings or get_settings()
    context = render_project_context(analysis, settings.max_context_chars)
    user = ("Create a Dockerfile and .dockerignore for the project below. The Docker build context is the project "
            "root directory and the command is `docker build -t <image> .`.\n\n" + context)
    if analysis.existing_dockerfile_content:
        user += ("\n\n## EXISTING DOCKERFILE (reference only; it will not be overwritten without approval)\n```\n"
                 + analysis.existing_dockerfile_content[:3000] + "\n```")
    return [{"role": "system", "content": GENERATION_SYSTEM}, {"role": "user", "content": user}]


def build_correction_message(analysis: ProjectAnalysis, history: ExecutionHistory, dockerfile: str,
                             dockerignore: str, *, error_summary: str, log_tail: str, failed_stage: Optional[str],
                             exit_code: Optional[int], settings: Optional[Settings] = None,
                             diagnosis: Optional[Any] = None, repeat_count: int = 1,
                             known_fixes: Optional[list[str]] = None) -> dict[str, str]:
    """User message asking for a targeted fix. Self-contained so conversation trimming is safe."""
    settings = settings or get_settings()
    facts = render_project_context(analysis, min(5000, settings.max_context_chars), include_files=False)
    tail_budget = min(6000, max(1500, settings.max_context_chars // 2))
    tail = redact_secrets(log_tail)[-tail_budget:]
    prior = history.previous_failures_summary()
    body = [
        "The Docker build failed. Diagnose the root cause and return a targeted correction.",
        f"## BUILD RESULT\nExit code: {exit_code}\nFailed stage/instruction: {failed_stage or 'unknown'}\n"
        f"Detected error:\n{redact_secrets(error_summary)}",
        f"## BUILD LOG (tail)\n```\n{tail}\n```",
        f"## CURRENT DOCKERFILE (failed)\n```dockerfile\n{dockerfile}\n```",
        f"## CURRENT .dockerignore\n```\n{dockerignore}\n```",
    ]
    if diagnosis is not None:
        cls = [f"Category: {diagnosis.category.value}",
               f"Fixable by editing the Dockerfile/.dockerignore: {'yes' if diagnosis.fixable else 'no'}",
               f"Failing instruction: {diagnosis.failing_instruction or 'unknown'}",
               f"Key evidence: {redact_secrets(diagnosis.evidence)}", f"Guidance: {diagnosis.hint}"]
        if repeat_count >= 2:
            cls.append(f"This exact error has occurred {repeat_count} times in a row: the previous fix did NOT "
                       "address the root cause. Form a different hypothesis.")
        body.append("## CLASSIFIED ERROR (from static log analysis)\n" + "\n".join(cls))
    if known_fixes:
        body.append("## FIXES THAT WORKED BEFORE FOR THIS ERROR CATEGORY (local history; reference only, they may "
                    "not apply here)\n" + "\n".join(f"- {k}" for k in known_fixes))
    if prior:
        body.append("## PREVIOUS ATTEMPTS (do not repeat fixes that already failed)\n" + prior)
    body.append("## BUILD CONSTRAINTS\n"
                f"- Command: docker build -t <image> . (context = project root, timeout {settings.docker_build_timeout}s)\n"
                "- Build runs unprivileged without extra mounts, build secrets or host network access.\n"
                "- Only the Dockerfile and .dockerignore may be changed; application sources cannot be edited.")
    body.append("## PROJECT CONTEXT\n" + facts)
    if analysis.structure:
        body.append("## PROJECT STRUCTURE\n" + analysis.structure[:2500])
    return {"role": "user", "content": "\n\n".join(body)}


def parse_retry_message(problem: str) -> dict[str, str]:
    """Follow-up when the model's previous reply was unusable."""
    return {"role": "user", "content": (
        f"Your previous reply could not be used: {problem}\n"
        "Reply again with ONLY the required JSON object, with every key present and the complete Dockerfile text.")}


def json_schema_for(correction: bool) -> dict[str, Any]:
    return (CorrectionResult if correction else GeneratedArtifacts).model_json_schema()


def trim_conversation(messages: list[dict[str, str]], keep_last: int = 4) -> list[dict[str, str]]:
    """Bound context growth: keep system + first user message + the most recent turns."""
    if len(messages) <= 2 + keep_last:
        return list(messages)
    return messages[:2] + messages[-keep_last:]

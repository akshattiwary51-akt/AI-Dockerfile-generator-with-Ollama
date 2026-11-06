"""Deterministic fault-injection harness: a simulated Docker engine plus a rule-based (non-LLM) corrector.

Nothing here touches Docker, Ollama or the network. It exists to prove that the production feedback loop
(classification -> correction -> rewrite -> rebuild) converges on its own, independent of any language model.
The SimulatedDocker only models the failure classes it reports; it is NOT a substitute for a real daemon.
"""
from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Any, Optional

from app.agent.error_classifier import Diagnosis, ErrorCategory
from app.docker.docker_builder import BuildResult, extract_build_error
from app.llm.response_parser import CorrectionResult
from app.utils.dockerignore import is_ignored

INSTRUCTIONS = ["FROM", "RUN", "CMD", "COPY", "ADD", "ENV", "ARG", "WORKDIR", "EXPOSE", "USER", "ENTRYPOINT", "VOLUME",
                "LABEL", "HEALTHCHECK", "SHELL", "STOPSIGNAL", "ONBUILD", "MAINTAINER"]
VALID_IMAGES = {"python:3.13-slim", "python:3.12-slim", "python:3.12", "node:22-alpine", "node:20-alpine", "alpine:3.20"}
KNOWN_PACKAGES = {"flask", "gunicorn", "requests", "fastapi", "uvicorn", "django", "numpy", "pydantic"}


class SimulatedDocker:
    """Builds by *reading* the Dockerfile and reporting the first failure BuildKit would (plain-progress style)."""

    def __init__(self, network_down: bool = False) -> None:
        self.network_down = network_down
        self.builds = 0
        self.dockerfiles: list[str] = []

    def check_docker(self) -> tuple[bool, str]:
        return True, "simulated"

    def build(self, project_dir: Any, image: str, *, on_output: Any = None, cancel_event: Any = None,
              dockerfile: str = "Dockerfile") -> BuildResult:
        self.builds += 1
        root = Path(project_dir)
        text = (root / dockerfile).read_text()
        self.dockerfiles.append(text)
        ignore = (root / ".dockerignore").read_text() if (root / ".dockerignore").exists() else ""
        out = self._run(root, text, ignore)
        if on_output:
            for line in out[0].splitlines():
                on_output(line)
        if out[1] is None:
            return BuildResult(success=True, image=image, exit_code=0, stdout=out[0], duration_seconds=0.1)
        stage, summary = extract_build_error(out[0])
        return BuildResult(success=False, image=image, exit_code=1, stdout=out[0], stderr=out[1],
                           failed_stage=stage, error_summary=summary, duration_seconds=0.1)

    # returns (log, error_text_or_None)
    def _run(self, root: Path, text: str, ignore: str) -> tuple[str, Optional[str]]:
        log: list[str] = []
        lines = [(i + 1, ln.strip()) for i, ln in enumerate(text.splitlines()) if ln.strip() and not ln.strip().startswith("#")]
        for n, ln in lines:
            word = ln.split()[0]
            if word.upper() not in INSTRUCTIONS:
                msg = f"ERROR: failed to solve: dockerfile parse error on line {n}: unknown instruction: {word}"
                return msg, msg
        if not lines or lines[0][1].split()[0].upper() != "FROM":
            msg = "ERROR: failed to solve: dockerfile parse error: file with no instructions"
            return msg, msg
        copied: set[str] = set()
        step = 0
        for n, ln in lines:
            word, _, rest = ln.partition(" ")
            word = word.upper()
            step += 1
            if word == "FROM":
                image = rest.split()[0]
                if self.network_down:
                    err = (f"#1 ERROR: failed to do request: Head \"https://registry-1.docker.io/v2/library/python/manifests/x\": "
                           "dial tcp: lookup registry-1.docker.io: no such host")
                    return "#1 [internal] load metadata\n" + err, err
                if image not in VALID_IMAGES:
                    err = (f"ERROR: failed to solve: {image}: failed to resolve source metadata for docker.io/library/{image}: "
                           f"docker.io/library/{image}: not found")
                    return err, err
                log.append(f"#{step} [1/n] FROM {image}")
            elif word in ("COPY", "ADD"):
                parts = [p for p in rest.split() if not p.startswith("--")]
                for src in parts[:-1]:
                    s = src.lstrip("./") or "."
                    if s == ".":
                        copied.add("*")
                        continue
                    if not (root / s).exists() or is_ignored(s, ignore):
                        hdr = f"#{step} [{step}/n] {ln}"
                        err = f"#{step} ERROR: failed to compute cache key: failed to calculate checksum of ref x: \"/{s}\": not found"
                        return "\n".join(log + [hdr, err]), "\n".join([hdr, err])
                    copied.add(s)
                log.append(f"#{step} [{step}/n] {ln}")
            elif word == "RUN" and "pip install" in rest:
                hdr = f"#{step} [{step}/n] {ln}"
                m = re.search(r"-r\s+(\S+)", rest)
                pkgs: list[str] = []
                if m:
                    req = m.group(1)
                    if "*" not in copied and req not in copied:
                        err = f"#{step} 1.2 ERROR: Could not open requirements file: [Errno 2] No such file or directory: '{req}'"
                        return "\n".join(log + [hdr, err]), "\n".join([hdr, err])
                    pkgs = [re.split(r"[=<>\[]", x.strip())[0].lower() for x in (root / req).read_text().splitlines() if x.strip()]
                else:
                    pkgs = [re.split(r"[=<>\[]", p)[0].lower() for p in rest.split("pip install")[1].split() if not p.startswith("-")]
                for p in pkgs:
                    if p not in KNOWN_PACKAGES:
                        err = (f"#{step} 1.4 ERROR: Could not find a version that satisfies the requirement {p} (from versions: none)\n"
                               f"#{step} 1.4 ERROR: No matching distribution found for {p}\n"
                               f"#{step} ERROR: process \"/bin/sh -c {rest}\" did not complete successfully: exit code: 1")
                        return "\n".join(log + [hdr, err]), "\n".join([hdr, err])
                log.append(f"#{step} [{step}/n] {ln}")
            else:
                log.append(f"#{step} [{step}/n] {ln}")
        log.append("#end DONE")
        return "\n".join(log), None


class RuleBasedCorrector:
    """A corrector with the same interface as DockerfileGenerator.correct, driven by rules instead of an LLM."""

    def __init__(self, project: Path, helpful: bool = True) -> None:
        self.project, self.helpful, self.calls = project, helpful, 0

    def correct(self, analysis: Any, history: Any, current: Any, build: BuildResult, diagnosis: Optional[Diagnosis] = None,
                repeat_count: int = 1, known_fixes: Optional[list[str]] = None) -> CorrectionResult:
        self.calls += 1
        text, evidence = current.dockerfile, build.combined_output
        new, change = text, ""
        if self.helpful and diagnosis:
            cat = diagnosis.category
            if cat == ErrorCategory.SYNTAX and (m := re.search(r"unknown instruction: (\S+)", evidence)):
                best = difflib.get_close_matches(m.group(1).upper(), INSTRUCTIONS, n=1)
                if best:
                    new, change = re.sub(rf"(?m)^{re.escape(m.group(1))}\b", best[0], text), f"{m.group(1)} -> {best[0]}"
            elif cat == ErrorCategory.BASE_IMAGE and (m := re.search(r"failed to solve: (\S+): failed to resolve", evidence)):
                repo = m.group(1).split(":")[0]
                options = sorted(i for i in VALID_IMAGES if i.startswith(repo + ":"))
                suffix = "-" + m.group(1).split("-", 1)[1] if "-" in m.group(1) else ""
                options = [i for i in options if i.endswith(suffix)] or options  # keep the variant (-slim, -alpine)
                if options:
                    new, change = text.replace(m.group(1), options[0]), f"base image {m.group(1)} -> {options[0]}"
            elif cat == ErrorCategory.MISSING_FILES and (m := re.search(r'"/([^"]+)": not found', evidence)):
                files = [p.name for p in self.project.iterdir() if p.is_file()]
                best = difflib.get_close_matches(m.group(1), files, n=1)
                if best:
                    new, change = re.sub(rf"\b{re.escape(m.group(1))}\b", best[0], text), f"COPY {m.group(1)} -> {best[0]}"
            elif cat == ErrorCategory.PYTHON_PKG and (m := re.search(r"No matching distribution found for (\S+)", evidence)):
                best = difflib.get_close_matches(m.group(1), sorted(KNOWN_PACKAGES), n=1)
                if best:
                    new, change = re.sub(rf"\b{re.escape(m.group(1))}\b", best[0], text), f"package {m.group(1)} -> {best[0]}"
        if new == text:
            return CorrectionResult(recoverable=False, root_cause="no rule applies", dockerfile="")
        return CorrectionResult(dockerfile=new, dockerignore=current.dockerignore, recoverable=True, changes=[change],
                                root_cause=(diagnosis.hint if diagnosis else ""), explanation=change,
                                expected_effect="the failing instruction should now succeed")

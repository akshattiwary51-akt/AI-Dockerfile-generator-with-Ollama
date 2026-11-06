"""Dockerfile quality scoring across seven categories with actionable recommendations."""
from __future__ import annotations

import glob as globmod
import json
import re
import shlex
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from app.analyzer.models import ProjectAnalysis
from app.docker.docker_validator import Instruction, parse_instructions, validate_artifacts
from app.docker.hadolint import HadolintFinding, run_hadolint
from app.utils.dockerignore import is_ignored

CATEGORIES = {  # name -> (label, weight)
    "security": ("Security", 0.22),
    "runtime": ("Runtime correctness", 0.20),
    "reproducibility": ("Reproducibility", 0.14),
    "build_efficiency": ("Build efficiency", 0.12),
    "image_size": ("Image size", 0.12),
    "best_practices": ("Docker best practices", 0.12),
    "maintainability": ("Maintainability", 0.08),
}
DEDUCT = {"high": 30, "medium": 15, "low": 6}
HIGH_SEVERITY_CAP = 59  # a weighted average must not hide a blocker such as an embedded secret
INSTALL_RE = re.compile(r"\b(pip3?\s+install|npm\s+(ci|install|i)\b|yarn(\s+install\b|\s*$|\s+--)|pnpm\s+(install|i)\b|"
                        r"go\s+mod\s+download|bundle\s+install|composer\s+install|mvn\b|gradle\b|\./gradlew|\./mvnw|"
                        r"poetry\s+install|uv\s+sync|pipenv\s+install)", re.I)
SLIM_HINTS = ("slim", "alpine", "distroless", "scratch", "busybox", "-jre", "jre-", "nginx", "chiseled", "minimal")
FULL_IMAGE_REPOS = {"python", "node", "ruby", "php", "golang", "openjdk", "ubuntu", "debian", "maven", "gradle"}
BUILD_TOOLS = re.compile(r"\b(build-essential|gcc|g\+\+|make|cmake|git)\b")

# validator issue code -> (category, severity)
VALIDATOR_MAP: dict[str, tuple[str, str]] = {
    "runs-as-root": ("security", "high"), "embedded-secret": ("security", "high"),
    "pipe-to-shell": ("security", "high"), "privileged-or-socket": ("security", "high"),
    "docker-socket": ("security", "high"), "copies-env-file": ("security", "high"),
    "unpinned-image": ("reproducibility", "medium"), "latest-tag": ("reproducibility", "medium"),
    "add-remote": ("security", "medium"), "chmod-777": ("security", "medium"), "sudo": ("security", "low"),
    "apt-recommends": ("image_size", "low"), "apt-cache": ("image_size", "low"), "pip-cache": ("image_size", "low"),
    "no-healthcheck": ("best_practices", "low"), "no-expose": ("best_practices", "low"),
    "no-cmd": ("runtime", "high"), "no-git-exclude": ("best_practices", "low"),
    "no-env-exclude": ("best_practices", "medium"), "empty-dockerignore": ("best_practices", "medium"),
}


class Finding(BaseModel):
    category: str
    severity: str  # high | medium | low
    code: str
    message: str
    recommendation: str
    line: Optional[int] = None


class CategoryScore(BaseModel):
    key: str
    label: str
    weight: float
    score: int = 100
    findings: list[Finding] = Field(default_factory=list)


class QualityReport(BaseModel):
    overall: int
    grade: str
    categories: dict[str, CategoryScore]
    recommendations: list[str] = Field(default_factory=list)
    hadolint_used: bool = False
    hadolint: list[HadolintFinding] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @property
    def findings(self) -> list[Finding]:
        return [f for c in self.categories.values() for f in c.findings]

    def format(self) -> str:
        lines = [f"Overall quality: {self.overall}/100 (grade {self.grade})"]
        for c in self.categories.values():
            lines.append(f"  {c.label:<22} {c.score:>3}  (weight {c.weight:.0%})")
            for f in c.findings:
                lines.append(f"      [{f.severity}] {f.message}")
        if self.recommendations:
            lines += ["Recommendations:"] + [f"  - {r}" for r in self.recommendations]
        lines += [f"Note: {n}" for n in self.notes]
        lines.append("Hadolint: " + (f"{len(self.hadolint)} finding(s)" if self.hadolint_used
                                     else "not installed - used internal checks only"))
        return "\n".join(lines)


def _grade(score: int) -> str:
    return "A" if score >= 90 else "B" if score >= 80 else "C" if score >= 65 else "D" if score >= 50 else "F"


def _stages(instrs: list[Instruction]) -> list[list[Instruction]]:
    stages: list[list[Instruction]] = []
    for ins in instrs:
        if ins.name == "FROM":
            stages.append([ins])
        elif stages:
            stages[-1].append(ins)
    return stages


def copy_sources(ins: Instruction) -> list[str]:
    """Source operands of a COPY/ADD (empty for ``--from`` copies, which read from another stage/image)."""
    args = ins.args.strip()
    if "--from=" in args:
        return []
    try:
        tokens = json.loads(args) if args.startswith("[") else shlex.split(args)
    except (ValueError, json.JSONDecodeError):
        return []
    tokens = [t for t in tokens if isinstance(t, str) and not t.startswith("--")]
    return tokens[:-1] if len(tokens) >= 2 else []


def _add(report: dict[str, CategoryScore], category: str, severity: str, code: str, message: str, rec: str,
         line: Optional[int] = None) -> None:
    cat = report[category]
    if any(f.code == code and f.line == line for f in cat.findings):
        return
    cat.findings.append(Finding(category=category, severity=severity, code=code, message=message,
                                recommendation=rec, line=line))


def _check_stage_logic(instrs: list[Instruction], cats: dict[str, CategoryScore], analysis: Optional[ProjectAnalysis]
                       ) -> None:
    stages = _stages(instrs)
    if not stages:
        return
    final = stages[-1]
    unit = analysis.primary_unit if analysis else None

    for stage in stages:
        first_copy_all = next((i for i in stage if i.name == "COPY" and copy_sources(i) and
                               all(s in (".", "./") for s in copy_sources(i))), None)
        first_install = next((i for i in stage if i.name == "RUN" and INSTALL_RE.search(i.args)), None)
        if first_copy_all and first_install and first_copy_all.line < first_install.line:
            _add(cats, "build_efficiency", "medium", "cache-order",
                 "All sources are copied before dependencies are installed, so every code change re-installs "
                 "dependencies.", "COPY only the dependency manifest/lockfile, install, then COPY the rest.",
                 first_copy_all.line)
        runs = [i for i in stage if i.name == "RUN"]
        for r in runs:
            if re.search(r"apt-get\s+update\s*$", r.args.strip()):
                _add(cats, "build_efficiency", "medium", "apt-update-alone",
                     "`apt-get update` runs in its own layer, which leads to stale package indexes from cache.",
                     "Combine `apt-get update && apt-get install ...` in one RUN.", r.line)
            if len(r.args) > 700:
                _add(cats, "maintainability", "low", "long-run", "A RUN instruction is very long.",
                     "Split unrelated steps or move logic into a script.", r.line)
        if len(runs) > 8:
            _add(cats, "build_efficiency", "low", "many-layers", f"{len(runs)} RUN instructions create many layers.",
                 "Merge related RUN steps with && to reduce layers.")
        if any(r.args and re.search(r"apt-get\s+(upgrade|dist-upgrade)", r.args) for r in runs):
            _add(cats, "reproducibility", "medium", "apt-upgrade",
                 "`apt-get upgrade` makes builds non-reproducible.", "Pin the base image tag instead of upgrading.")
        seen_user: Optional[str] = None
        for ins in stage:
            if ins.name == "USER":
                seen_user = ins.args.strip().lower()
            elif ins.name == "RUN" and seen_user not in (None, "root", "0") and \
                    re.search(r"\b(apt-get|apk\s+add|useradd|adduser|groupadd|yum|dnf)\b|npm\s+(i|install)\s+(-g|--global)",
                              ins.args):
                _add(cats, "runtime", "medium", "root-after-user",
                     "A RUN that needs root (package install/user management) comes after the non-root USER.",
                     "Move this RUN before the USER instruction.", ins.line)

    # reproducibility: install commands that ignore the lockfile
    text_runs = " \n".join(i.args for s in stages for i in s if i.name == "RUN")
    pm = unit.package_manager if unit else None
    lock = unit.lockfile if unit else None
    if re.search(r"\bnpm\s+(install|i)\b", text_runs) and lock in ("package-lock.json",):
        _add(cats, "reproducibility", "medium", "npm-install", "`npm install` ignores package-lock.json pinning.",
             "Use `npm ci` for reproducible installs.")
    if re.search(r"\byarn\s+install\b", text_runs) and lock == "yarn.lock" and "frozen-lockfile" not in text_runs \
            and "--immutable" not in text_runs:
        _add(cats, "reproducibility", "medium", "yarn-unfrozen", "`yarn install` may modify yarn.lock.",
             "Use `yarn install --frozen-lockfile` (or `--immutable` on Yarn 2+).")
    if re.search(r"\bpnpm\s+(install|i)\b", text_runs) and lock == "pnpm-lock.yaml" and "frozen-lockfile" not in text_runs:
        _add(cats, "reproducibility", "medium", "pnpm-unfrozen", "`pnpm install` may modify the lockfile.",
             "Use `pnpm install --frozen-lockfile`.")
    if re.search(r"pip3?\s+install\s+(?!-r|--requirement|-e|\.|--no-cache-dir\s+-r|--no-cache-dir\s+\.)[a-zA-Z]", text_runs) \
            and not re.search(r"pip3?\s+install[^\n]*(-r|--requirement)", text_runs):
        _add(cats, "reproducibility", "medium", "pip-adhoc",
             "Packages are installed ad hoc without a requirements file.", "Install from a pinned requirements.txt/lockfile.")
    if pm == "pip" and unit and unit.dependencies and not unit.lockfile:
        _add(cats, "reproducibility", "low", "no-lockfile", "No lockfile was found for this project.",
             "Pin dependency versions (requirements.txt with ==, or a lockfile).")

    # image size
    final_from = final[0].args.split()
    final_image = next((t for t in final_from if not t.startswith("--")), "")
    repo = final_image.split("/")[-1].split(":")[0].lower()
    is_slim = any(h in final_image.lower() for h in SLIM_HINTS)
    if len(stages) == 1 and repo in FULL_IMAGE_REPOS and not is_slim:
        _add(cats, "image_size", "medium", "full-base-image",
             f"Base image '{final_image}' is a full-size image.", "Use a -slim or -alpine variant (or distroless).")
    if analysis and analysis.strategy and analysis.strategy.multi_stage and len(stages) == 1:
        _add(cats, "image_size", "medium", "single-stage",
             "The project type benefits from a multi-stage build but only one stage is used.",
             "Build in one stage and copy only the artifacts into a minimal runtime stage.")
    if len(stages) == 1 and any(i.name == "RUN" and BUILD_TOOLS.search(i.args) and "install" in i.args for i in final):
        _add(cats, "image_size", "low", "build-tools-in-final",
             "Compilers/build tools are installed in the final image.", "Install them in a build stage only.")

    # maintainability / best practice
    if not any(i.name == "WORKDIR" for i in final):
        _add(cats, "maintainability", "medium", "no-workdir", "No WORKDIR is set; files land in the root directory.",
             "Set WORKDIR /app (or similar).")
    if not any(i.name == "LABEL" for s in stages for i in s):
        _add(cats, "maintainability", "low", "no-label", "No LABEL metadata (maintainer/source/version).",
             "Add OCI labels such as org.opencontainers.image.source.")
    for ins in final:
        if ins.name in ("CMD", "ENTRYPOINT") and not ins.args.lstrip().startswith("["):
            _add(cats, "maintainability", "low", "shell-form-cmd",
                 f"{ins.name} uses shell form, so signals reach a shell instead of the app.",
                 f'Use exec form, e.g. {ins.name} ["executable", "arg"].', ins.line)
        if ins.name == "ADD" and not re.search(r"https?://|\.(tar|tgz|gz|xz|zip)\b", ins.args):
            _add(cats, "maintainability", "low", "add-for-copy", "ADD is used for plain files.",
                 "Prefer COPY unless you need archive extraction.", ins.line)

    # runtime correctness
    cmd = next((i for i in reversed(final) if i.name in ("CMD", "ENTRYPOINT")), None)
    if cmd:
        lowered = cmd.args.lower()
        if re.search(r"(127\.0\.0\.1|localhost)", lowered) and re.search(r"(host|bind|-b|listen|:)\s*[= ]?", lowered):
            _add(cats, "runtime", "high", "loopback-bind",
                 "The start command binds to 127.0.0.1/localhost, which is unreachable from outside the container.",
                 "Bind to 0.0.0.0.", cmd.line)
        if re.search(r"\bflask\s+run\b", lowered) and "0.0.0.0" not in lowered:
            _add(cats, "runtime", "high", "flask-run-loopback",
                 "`flask run` binds to 127.0.0.1 by default.", "Add --host=0.0.0.0 or use gunicorn.", cmd.line)
    exposed = {int(p.split("/")[0]) for i in final if i.name == "EXPOSE" for p in i.args.split()
               if p.split("/")[0].isdigit()}
    if unit and unit.ports and exposed and not exposed & set(unit.ports):
        _add(cats, "runtime", "medium", "port-mismatch",
             f"EXPOSE {sorted(exposed)} does not match the detected application port {unit.ports}.",
             f"Expose and bind port {unit.ports[0]}.")
    if analysis and unit and unit.role in ("backend", "fullstack") and unit.ports and not any(
            i.name == "HEALTHCHECK" for i in final):
        _add(cats, "best_practices", "medium", "healthcheck-backend",
             "A network service without a HEALTHCHECK.", "Add a HEALTHCHECK using a tool present in the image.")


def check_copy_sources(instrs: list[Instruction], project_root: Path, dockerignore: str,
                       cats: dict[str, CategoryScore]) -> None:
    """Verify that every COPY/ADD source exists in the build context and is not excluded by .dockerignore."""
    root = project_root.resolve()
    for ins in instrs:
        if ins.name not in ("COPY", "ADD"):
            continue
        for src in copy_sources(ins):
            if "$" in src or src.startswith(("http://", "https://")):
                continue
            clean = src.lstrip("/").removeprefix("./")
            if clean in ("", "."):
                continue
            matches = [Path(p).relative_to(root).as_posix() for p in globmod.glob(str(root / clean), recursive=True)
                       if Path(p).resolve().is_relative_to(root)]
            if not matches:
                _add(cats, "runtime", "high", "copy-source-missing",
                     f"COPY source '{src}' does not exist in the build context; the build will fail.",
                     "Fix the path or add the file to the project.", ins.line)
            elif all(is_ignored(m, dockerignore) for m in matches):
                _add(cats, "runtime", "high", "copy-source-ignored",
                     f"COPY source '{src}' is excluded by .dockerignore; the build will fail.",
                     "Remove it from .dockerignore or stop copying it.", ins.line)


def score_dockerfile(dockerfile: str, dockerignore: str = "", *, analysis: Optional[ProjectAnalysis] = None,
                     project_root: Optional[Path | str] = None, use_hadolint: bool = True,
                     hadolint_runner: Optional[Callable[[str], Optional[list[HadolintFinding]]]] = None,
                     extra_security: Optional[Callable[..., list[Finding]]] = None) -> QualityReport:
    cats = {k: CategoryScore(key=k, label=v[0], weight=v[1]) for k, v in CATEGORIES.items()}
    notes: list[str] = []
    instrs, _ = parse_instructions(dockerfile)

    for issue in validate_artifacts(dockerfile, dockerignore).issues:
        if issue.code in VALIDATOR_MAP:
            cat, sev = VALIDATOR_MAP[issue.code]
            _add(cats, cat, sev, issue.code, issue.message, "See the message for the fix.", issue.line)
        elif issue.severity == "error":
            _add(cats, "runtime", "high", issue.code, issue.message, "Fix the Dockerfile syntax/semantics.", issue.line)
    if instrs:
        _check_stage_logic(instrs, cats, analysis)
        root = Path(project_root) if project_root else (Path(analysis.root) if analysis else None)
        if root and root.is_dir():
            check_copy_sources(instrs, root, dockerignore, cats)
        else:
            notes.append("COPY sources were not checked against the project (no project directory given).")
    if extra_security is None:
        from app.security.analyzer import quality_findings
        extra_security = quality_findings
    if extra_security:
        for f in extra_security(dockerfile, dockerignore, analysis, project_root):
            _add(cats, "security", f.severity, f.code, f.message, f.recommendation, f.line)

    lint: Optional[list[HadolintFinding]] = (hadolint_runner or run_hadolint)(dockerfile) if use_hadolint else None
    hadolint_penalty = 0
    if lint is not None:
        weights = {"error": 10, "warning": 4, "info": 1, "style": 0}
        hadolint_penalty = min(40, sum(weights.get(h.level, 0) for h in lint))
        if hadolint_penalty:
            cats["best_practices"].findings.append(Finding(
                category="best_practices", severity="medium" if hadolint_penalty >= 15 else "low", code="hadolint",
                message=f"Hadolint reported {len(lint)} finding(s) (e.g. {lint[0].code}: {lint[0].message}).",
                recommendation="Review the Hadolint findings."))
    for cat in cats.values():
        deduction = sum(DEDUCT[f.severity] for f in cat.findings if f.code != "hadolint")
        extra = hadolint_penalty if cat.key == "best_practices" else 0
        cat.score = max(0, 100 - deduction - extra)
    overall = round(sum(c.score * c.weight for c in cats.values()))
    blockers = [f for key in ("security", "runtime") for f in cats[key].findings if f.severity == "high"]
    if blockers and overall > HIGH_SEVERITY_CAP:
        notes.append(f"Overall score capped at {HIGH_SEVERITY_CAP}: {len(blockers)} high-severity security/runtime "
                     "finding(s) should be fixed before shipping this image.")
        overall = HIGH_SEVERITY_CAP
    order = {"high": 0, "medium": 1, "low": 2}
    recs: list[str] = []
    for f in sorted((f for c in cats.values() for f in c.findings), key=lambda f: order[f.severity]):
        if f.recommendation not in recs and f.recommendation != "See the message for the fix.":
            recs.append(f.recommendation)
        elif f.recommendation == "See the message for the fix." and f.message not in recs:
            recs.append(f.message)
    return QualityReport(overall=overall, grade=_grade(overall), categories=cats, recommendations=recs[:10],
                         hadolint_used=lint is not None, hadolint=lint or [], notes=notes)

"""Static validation of LLM-produced Dockerfiles / .dockerignore files (no Docker needed)."""
from __future__ import annotations

import json
import re
from typing import Optional

from pydantic import BaseModel, Field

VALID_INSTRUCTIONS = {"FROM", "RUN", "CMD", "LABEL", "MAINTAINER", "EXPOSE", "ENV", "ADD", "COPY", "ENTRYPOINT",
                      "VOLUME", "USER", "WORKDIR", "ARG", "ONBUILD", "STOPSIGNAL", "HEALTHCHECK", "SHELL"}
JSON_FORM = {"CMD", "ENTRYPOINT", "RUN", "SHELL", "VOLUME", "COPY", "ADD", "HEALTHCHECK"}
SECRET_KEY = re.compile(r"(?i)(password|passwd|secret|token|api[_-]?key|private[_-]?key|credential|access[_-]?key)")
MAX_DOCKERFILE_BYTES = 100_000


class ValidationIssue(BaseModel):
    severity: str  # error | warning
    code: str
    message: str
    line: Optional[int] = None


class ValidationReport(BaseModel):
    issues: list[ValidationIssue] = Field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def score(self) -> int:
        """Rule-based quality score 0-100 (errors weigh heavily, warnings lightly)."""
        return max(0, 100 - 25 * len(self.errors) - 5 * len(self.warnings))

    def format(self) -> str:
        return "\n".join(f"[{i.severity}] {i.code}" + (f" (line {i.line})" if i.line else "") + f": {i.message}"
                         for i in self.issues)

    def add(self, severity: str, code: str, message: str, line: Optional[int] = None) -> None:
        self.issues.append(ValidationIssue(severity=severity, code=code, message=message, line=line))


class Instruction(BaseModel):
    name: str
    args: str
    line: int


def parse_instructions(content: str) -> tuple[list[Instruction], list[str]]:
    """Split a Dockerfile into instructions, joining continuations and skipping comments/heredoc bodies."""
    instructions: list[Instruction] = []
    problems: list[str] = []
    lines = content.splitlines()
    i = 0
    escape = "\\"
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        m = re.match(r"^#\s*escape\s*=\s*(\S)", stripped, re.I)
        if m:
            escape = m.group(1)
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        start = i + 1
        buf = raw.rstrip()
        while buf.endswith(escape) and i + 1 < len(lines):
            i += 1
            nxt = lines[i]
            if nxt.strip().startswith("#"):
                buf = buf[:-1]
                continue
            buf = buf[:-1] + " " + nxt.strip()
        if buf.endswith(escape):
            buf = buf[:-1]
        parts = buf.strip().split(None, 1)
        name = parts[0].upper()
        args = parts[1] if len(parts) > 1 else ""
        instructions.append(Instruction(name=name, args=args, line=start))
        heredocs = re.findall(r"<<-?\s*[\"']?(\w+)[\"']?", args)
        i += 1
        for marker in heredocs:  # skip heredoc bodies up to their terminator
            while i < len(lines) and lines[i].strip() != marker:
                i += 1
            i += 1
    return instructions, problems


def _check_json_form(ins: Instruction, report: ValidationReport) -> None:
    if ins.name in JSON_FORM and ins.args.lstrip().startswith("["):
        try:
            parsed = json.loads(ins.args)
            if not isinstance(parsed, list) or not all(isinstance(x, str) for x in parsed):
                raise ValueError("array must contain only strings")
            if not parsed:
                report.add("error", "empty-exec-form", f"{ins.name} has an empty exec-form array.", ins.line)
        except ValueError as exc:
            report.add("error", "malformed-exec-form",
                       f"{ins.name} uses JSON exec form but it is invalid ({exc}). Use double quotes, e.g. "
                       f'["node", "server.js"].', ins.line)


def validate_dockerfile(content: str) -> ValidationReport:
    report = ValidationReport()
    if not content or not content.strip():
        report.add("error", "empty", "Dockerfile is empty.")
        return report
    if len(content.encode("utf-8")) > MAX_DOCKERFILE_BYTES:
        report.add("error", "too-large", "Dockerfile is unreasonably large.")
        return report
    if "\x00" in content:
        report.add("error", "binary", "Dockerfile contains NUL bytes.")
        return report
    if re.search(r"^\s*```", content, re.M):
        report.add("error", "markdown-fence", "Dockerfile contains markdown code fences.")

    instructions, _ = parse_instructions(content)
    if not instructions:
        report.add("error", "no-instructions", "Dockerfile contains no instructions.")
        return report

    first = next((x for x in instructions if x.name != "ARG"), None)
    if first is None or first.name != "FROM":
        report.add("error", "no-from", "A Dockerfile must start with FROM (only ARG may precede it).",
                   first.line if first else None)

    stages: list[str] = []
    from_count = 0
    last_stage_start = 0
    for idx, ins in enumerate(instructions):
        if ins.name not in VALID_INSTRUCTIONS:
            report.add("error", "unknown-instruction", f"Unknown instruction '{ins.name}'.", ins.line)
            continue
        if not ins.args and ins.name not in {"ONBUILD"}:
            report.add("error", "missing-arguments", f"{ins.name} requires arguments.", ins.line)
            continue
        _check_json_form(ins, report)
        a = ins.args
        if ins.name == "FROM":
            from_count += 1
            last_stage_start = idx
            m = re.match(r"^(?:--\S+\s+)*(\S+)(?:\s+[Aa][Ss]\s+(\S+))?\s*$", a)
            if not m:
                report.add("error", "bad-from", f"Invalid FROM syntax: '{a}'.", ins.line)
                continue
            image, alias = m.group(1), m.group(2)
            if alias:
                if alias.lower() in stages:
                    report.add("error", "duplicate-stage", f"Duplicate stage name '{alias}'.", ins.line)
                stages.append(alias.lower())
            if image.lower() != "scratch" and not image.startswith("$") and "${" not in image:
                tag = image.split("@")[0].rsplit("/", 1)[-1]
                if ":" not in tag and "@" not in image and image.lower() not in stages:
                    report.add("warning", "unpinned-image", f"Base image '{image}' has no tag (defaults to latest).",
                               ins.line)
                elif tag.endswith(":latest"):
                    report.add("warning", "latest-tag", f"Base image '{image}' uses the 'latest' tag.", ins.line)
        elif ins.name == "COPY" or ins.name == "ADD":
            m = re.search(r"--from=(\S+)", a)
            if m and ins.name == "COPY":
                ref = m.group(1).lower()
                if ref.isdigit():
                    if int(ref) >= from_count - 1:  # must reference an earlier stage than the current one
                        report.add("error", "bad-stage-ref", f"COPY --from={ref} refers to a stage that does not exist yet.",
                                   ins.line)
                elif ref not in stages and "/" not in ref and ":" not in ref and "$" not in ref:
                    report.add("error", "bad-stage-ref", f"COPY --from={m.group(1)} refers to an unknown stage.",
                               ins.line)
            if not a.lstrip().startswith("[") and len(re.sub(r"--\S+\s*", "", a).split()) < 2:
                report.add("error", "missing-arguments", f"{ins.name} needs at least a source and a destination.",
                           ins.line)
            if ins.name == "ADD" and re.search(r"https?://", a):
                report.add("warning", "add-remote", "ADD with a remote URL is discouraged; use RUN curl/COPY.", ins.line)
            if re.search(r"(^|[\s/])\.env(\.\w+)?(\s|$)", a) and ".example" not in a and ".sample" not in a:
                report.add("error", "copies-env-file", "Copying .env files into the image would embed secrets.", ins.line)
            if "docker.sock" in a:
                report.add("error", "docker-socket", "References the Docker socket.", ins.line)
        elif ins.name == "EXPOSE":
            for tok in a.split():
                port = tok.split("/")[0]
                if "$" in port:
                    continue
                if not port.isdigit() or not 1 <= int(port) <= 65535:
                    report.add("error", "bad-port", f"Invalid EXPOSE port '{tok}'.", ins.line)
                elif tok.count("/") and tok.split("/")[1].lower() not in {"tcp", "udp", "sctp"}:
                    report.add("error", "bad-port", f"Invalid protocol in EXPOSE '{tok}'.", ins.line)
        elif ins.name in ("ENV", "ARG"):
            pairs = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=(\"[^\"]*\"|'[^']*'|\S+)", a)
            if ins.name == "ENV" and not pairs and len(a.split(None, 1)) == 2:
                pairs = [(a.split(None, 1)[0], a.split(None, 1)[1])]
            for key, value in pairs:
                v = value.strip("\"'")
                if SECRET_KEY.search(key) and v and not v.startswith("$"):
                    report.add("error", "embedded-secret", f"{ins.name} {key} embeds a secret value in the image.",
                               ins.line)
        elif ins.name == "RUN":
            low = a.lower()
            if "--privileged" in low or "docker.sock" in low:
                report.add("error", "privileged-or-socket", "RUN must not use privileged mode or the Docker socket.",
                           ins.line)
            if re.search(r"(curl|wget)[^|;&]*\|\s*(sudo\s+)?(ba|z)?sh", low):
                report.add("error", "pipe-to-shell", "Piping a download into a shell is unsafe.", ins.line)
            if re.search(r"chmod\s+(-R\s+)?0?777", low):
                report.add("warning", "chmod-777", "chmod 777 is overly permissive.", ins.line)
            if "apt-get install" in low and "--no-install-recommends" not in low:
                report.add("warning", "apt-recommends", "apt-get install without --no-install-recommends.", ins.line)
            if "apt-get install" in low and "/var/lib/apt/lists" not in low:
                report.add("warning", "apt-cache", "apt lists are not cleaned in the same layer.", ins.line)
            if re.search(r"pip3?\s+install", low) and "--no-cache-dir" not in low and "PIP_NO_CACHE_DIR" not in content:
                report.add("warning", "pip-cache", "pip install without --no-cache-dir.", ins.line)
        elif ins.name == "USER" and len(a.split()) != 1:
            report.add("error", "bad-user", f"Invalid USER argument '{a}'.", ins.line)
        if re.search(r"(?i)\bsudo\b", a) and ins.name == "RUN":
            report.add("warning", "sudo", "Avoid sudo inside Dockerfile builds.", ins.line)

    if from_count == 0 and not any(i.code == "no-from" for i in report.issues):
        report.add("error", "no-from", "No FROM instruction found.")

    final = instructions[last_stage_start:] if from_count else instructions
    names = {x.name for x in final}
    if "CMD" not in names and "ENTRYPOINT" not in names:
        report.add("warning", "no-cmd", "Final stage has no CMD or ENTRYPOINT.")
    if "EXPOSE" not in names:
        report.add("warning", "no-expose", "Final stage does not EXPOSE a port.")
    users = [x for x in final if x.name == "USER"]
    if not users:
        report.add("warning", "runs-as-root", "Final stage never switches to a non-root USER.")
    elif users[-1].args.strip().lower() in ("root", "0"):
        report.add("warning", "runs-as-root", "Final stage runs as root.", users[-1].line)
    if "HEALTHCHECK" not in names:
        report.add("warning", "no-healthcheck", "No HEALTHCHECK defined (acceptable for non-HTTP workloads).")
    return report


def validate_dockerignore(content: str) -> ValidationReport:
    report = ValidationReport()
    if not content.strip():
        report.add("warning", "empty-dockerignore", ".dockerignore is empty; the whole context is sent to Docker.")
        return report
    patterns = {ln.strip().rstrip("/") for ln in content.splitlines() if ln.strip() and not ln.startswith("#")}
    if not {".git", ".git/*", ".git/**"} & patterns:
        report.add("warning", "no-git-exclude", ".dockerignore does not exclude .git.")
    if not any(p.startswith(".env") for p in patterns):
        report.add("warning", "no-env-exclude", ".dockerignore does not exclude .env files (possible secret leakage).")
    if "Dockerfile" in patterns and len(patterns) == 1:
        report.add("warning", "ignores-dockerfile", ".dockerignore only ignores the Dockerfile.")
    if "*" in patterns and not any(ln.strip().startswith("!") for ln in content.splitlines()):
        report.add("error", "ignores-everything", ".dockerignore excludes everything; the build context would be empty.")
    return report


def validate_artifacts(dockerfile: str, dockerignore: str) -> ValidationReport:
    merged = validate_dockerfile(dockerfile)
    merged.issues += validate_dockerignore(dockerignore).issues
    return merged

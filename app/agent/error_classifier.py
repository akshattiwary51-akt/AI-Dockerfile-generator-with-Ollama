"""Classify a failed ``docker build`` into a category, decide whether editing the Dockerfile can fix it, and
produce a stable signature used for repeated-error detection."""
from __future__ import annotations

import re
from enum import Enum
from typing import Optional

from pydantic import BaseModel


class ErrorCategory(str, Enum):
    DOCKER_DAEMON = "docker_daemon_failure"
    RESOURCE = "resource_exhaustion"
    NETWORK = "network_registry_failure"
    SYNTAX = "dockerfile_syntax"
    BASE_IMAGE = "base_image_problem"
    MISSING_FILES = "missing_application_files"
    BUILD_CONTEXT = "build_context_error"
    SYSTEM_DEP = "missing_system_dependency"
    PACKAGE_MANAGER = "package_manager_configuration"
    NODE_DEP = "invalid_node_dependency"
    PYTHON_PKG = "missing_python_package"
    PERMISSION = "permission_problem"
    WORKDIR = "incorrect_working_directory"
    PORT = "port_configuration"
    STARTUP = "invalid_startup_command"
    UNKNOWN = "unknown"


class Diagnosis(BaseModel):
    category: ErrorCategory
    fixable: bool                       # can a Dockerfile/.dockerignore edit plausibly fix it?
    evidence: str = ""                  # the log line that triggered the classification
    failing_instruction: Optional[str] = None
    hint: str = ""
    signature: str = ""                 # category + normalized evidence; identical across repeats of the same error


# (category, fixable, regex, hint) in priority order: the first rule with a matching log line wins.
RULES: list[tuple[ErrorCategory, bool, str, str]] = [
    (ErrorCategory.DOCKER_DAEMON, False,
     r"cannot connect to the docker daemon|failed to connect to the docker api|is the docker daemon running|"
     r"error during connect|permission denied while trying to connect to the docker|docker: command not found|"
     r"docker: not found",
     "The Docker daemon/CLI is not usable from here; start Docker or fix socket permissions. No Dockerfile change helps."),
    (ErrorCategory.RESOURCE, False,
     r"no space left on device|out of memory|oom-?kill|cannot allocate memory|exit code: 137|\bKilled\b$",
     "The host ran out of disk or memory. Free resources (docker system prune, more RAM); editing the Dockerfile will not help."),
    (ErrorCategory.NETWORK, False,
     r"toomanyrequests|rate limit|temporary failure in name resolution|no such host|tls handshake timeout|"
     r"network is unreachable|connection timed out|could not resolve host|i/o timeout|\bENOTFOUND\b|\bETIMEDOUT\b|"
     r"\bECONNRESET\b|\bEAI_AGAIN\b|failed to establish a new connection|max retries exceeded|"
     r"unauthorized: authentication required|connection refused|failed to do request",
     "A registry or package index could not be reached (network, DNS, proxy, rate limit or auth). "
     "Fix connectivity or run `docker login`; the Dockerfile is not the cause."),
    (ErrorCategory.SYNTAX, True,
     r"dockerfile parse error|unknown instruction|unexpected end of statement|unknown flag|failed to parse|"
     r"invalid reference format|dockerfile.*syntax|json: cannot unmarshal|requires at least one argument",
     "The Dockerfile itself is malformed; correct the instruction named in the error."),
    (ErrorCategory.BASE_IMAGE, True,
     r"failed to resolve source metadata|manifest unknown|manifest for .* not found|pull access denied|"
     r"repository does not exist|exec format error|release version \d+ not supported|invalid target release|"
     r"unsupported class file major version|image .* not found",
     "The base image tag/platform is wrong or unavailable; use an existing official tag (check spelling and version)."),
    (ErrorCategory.BUILD_CONTEXT, False,
     r"failed to read dockerfile|failed to solve with frontend dockerfile\.v0: failed to read|"
     r"unable to prepare context|path .* not found$|unable to evaluate symlinks",
     "The build context or Dockerfile path is wrong; run the build from the project directory."),
    (ErrorCategory.MISSING_FILES, True,
     r"failed to compute cache key|failed to calculate checksum|copy failed: file not found|"
     r"could not open requirements file|can't open file|no such file or directory|enoent|cannot find main module|"
     r"go\.mod file not found|lstat .*: no such file|not found in build context|\"/[^\"]+\": not found",
     "A file or directory referenced by COPY/RUN does not exist at that path in the build context or image "
     "(check spelling, WORKDIR, relative paths and .dockerignore)."),
    (ErrorCategory.SYSTEM_DEP, True,
     r"gcc: (command )?not found|command 'gcc' failed|error: command '(gcc|g\+\+|cc)'|pg_config executable not found|"
     r"python\.h: no such file|fatal error: .*\.h: no such file|cannot find -l\w+|unable to execute '(gcc|cc)'|"
     r"\bcc: not found|make: .*not found|g\+\+: not found|libpq-fe\.h|can't find rust compiler|cargo.*not found|"
     r"pkg-config.*not found|sh: \d*:? ?(?:gcc|g\+\+|make|cmake|git|curl|wget|pkg-config|cc): not found|"
     r"(?:git|curl|wget|make|cmake): (?:command )?not found|no matching package to install|unable to locate package|"
     r"failed building wheel",
     "A native build tool or system library is missing in the image. Install it (apt-get/apk) before the failing "
     "step, ideally in a build stage, in the same RUN layer as its cleanup."),
    (ErrorCategory.PACKAGE_MANAGER, True,
     r"npm ci` can only install|can only install packages when your package\.json|externally-managed-environment|"
     r"(?:pnpm|yarn|npm|pip3?|poetry|pipenv|uv|bundle|composer): (?:command )?not found|lockfile.*(?:out of date|not found)|"
     r"frozen-lockfile|err_pnpm|no required module provides package|your lock file|ambiguous.*package manager",
     "The package-manager invocation does not match the project (missing tool, outdated lockfile, or PEP 668). "
     "Install the tool (e.g. corepack enable) or use the command that matches the lockfile."),
    (ErrorCategory.NODE_DEP, True,
     r"npm (?:err!|error) code (?:e404|etarget|eresolve|enoent)|npm (?:err!|error) notarget|no matching version found|"
     r"cannot find module|couldn't find package|err_module_not_found|peer dep|eresolve",
     "A Node.js dependency cannot be resolved. Dependency versions live in package.json (not editable here); "
     "adjust install flags (e.g. --legacy-peer-deps) or the Node version."),
    (ErrorCategory.PYTHON_PKG, True,
     r"modulenotfounderror|no module named|could not find a version that satisfies|no matching distribution found|"
     r"requires a different python|requires-python|importerror",
     "A Python package/module is missing or incompatible with the interpreter. Check the Python base version and "
     "that the install step runs before the failing step."),
    (ErrorCategory.PERMISSION, True,
     r"permission denied|\beacces\b|operation not permitted|\bEPERM\b|errno 13|read-only file system|cannot create directory",
     "A step lacks permission. Create/chown directories before switching to the non-root USER, or run the step earlier."),
    (ErrorCategory.PORT, True,
     r"invalid containerport|invalid port specification|expose.*invalid|invalid.*expose",
     "The EXPOSE/port configuration is invalid."),
    (ErrorCategory.STARTUP, True,
     r"executable file not found in \$path|oci runtime create failed|starting container process caused",
     "The start command (CMD/ENTRYPOINT) refers to something that is not in the image."),
]
_COMPILED = [(c, f, re.compile(p, re.I), h) for c, f, p, h in RULES]

_STEP_PREFIX = re.compile(r"^\[(?:[\w.-]+\s+)?\d+/\d+\]\s*")
_NOISE = [(re.compile(r"^#\d+\s+(?:[\d.]+\s+)?"), ""), (re.compile(r"\b\d+(?:\.\d+)?\s?(?:ms|s|sec|seconds)\b"), "<t>"),
          (re.compile(r"\b[0-9a-f]{12,64}\b"), "<id>"), (re.compile(r"sha256:\S+"), "<sha>"),
          (re.compile(r"/tmp/\S+"), "/tmp/<x>"), (re.compile(r"\s+"), " ")]


def failing_instruction(failed_stage: Optional[str]) -> Optional[str]:
    if not failed_stage:
        return None
    return _STEP_PREFIX.sub("", failed_stage.strip()).rstrip(":") or None


def normalize_signature(category: ErrorCategory, evidence: str) -> str:
    text = evidence.strip().lower()
    for rx, repl in _NOISE:
        text = rx.sub(repl, text)
    return f"{category.value}:{text.strip()[:160]}"


def classify_build_error(output: str, failed_stage: Optional[str] = None) -> Diagnosis:
    lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
    for category, fixable, rx, hint in _COMPILED:
        for ln in lines:
            if rx.search(ln):
                evidence = ln[:300]
                return Diagnosis(category=category, fixable=fixable, evidence=evidence, hint=hint,
                                 failing_instruction=failing_instruction(failed_stage),
                                 signature=normalize_signature(category, evidence))
    error_lines = [ln for ln in lines if re.search(r"\berror\b|failed|did not complete", ln, re.I)]
    evidence = (error_lines[-1] if error_lines else (lines[-1] if lines else ""))[:300]
    return Diagnosis(category=ErrorCategory.UNKNOWN, fixable=True, evidence=evidence,
                     failing_instruction=failing_instruction(failed_stage),
                     hint="The cause is not recognised; read the log tail and fix the failing instruction.",
                     signature=normalize_signature(ErrorCategory.UNKNOWN, evidence))

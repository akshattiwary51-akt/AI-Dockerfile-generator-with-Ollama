"""Static security analysis of a Dockerfile (+ .dockerignore + build context).

This inspects *text and files only*. A clean report does NOT mean the image or the application is secure: it says
nothing about vulnerabilities in packages, application code or runtime configuration (use the scanners for packages).
"""
from __future__ import annotations

import json
import os
import re
import shlex
from datetime import date
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from app.analyzer.models import ProjectAnalysis
from app.docker.docker_validator import Instruction, parse_instructions
from app.utils.dockerignore import is_ignored
from app.utils.security import is_excluded_dir, is_sensitive_file, redact_secrets

DISCLAIMER = ("Static analysis of the Dockerfile and build context only. It cannot find vulnerable packages, flaws in "
              "the application, or runtime misconfiguration, so a good score does not mean the image is secure.")
SEVERITY_POINTS = {"critical": 40, "high": 25, "medium": 10, "low": 4, "info": 0}
SEVERITY_ORDER = ["critical", "high", "medium", "low", "info"]
SECRET_KEY = re.compile(r"(?i)(password|passwd|secret|token|api[_-]?key|private[_-]?key|credential|access[_-]?key)")

# Approximate end-of-life months (YYYY-MM). Always verify with https://endoflife.date.
EOL: dict[str, dict[str, str]] = {
    "python": {"2.7": "2020-01", "3.6": "2021-12", "3.7": "2023-06", "3.8": "2024-10", "3.9": "2025-10",
               "3.10": "2026-10", "3.11": "2027-10", "3.12": "2028-10", "3.13": "2029-10"},
    "node": {"12": "2022-04", "14": "2023-04", "16": "2023-09", "18": "2025-04", "20": "2026-04", "22": "2027-04",
             "24": "2028-04"},
    "ubuntu": {"14.04": "2019-04", "16.04": "2021-04", "18.04": "2023-05", "20.04": "2025-05", "22.04": "2027-04",
               "24.04": "2029-04"},
    "debian": {"jessie": "2020-06", "stretch": "2022-06", "buster": "2024-06", "bullseye": "2026-08",
               "bookworm": "2028-06", "8": "2020-06", "9": "2022-06", "10": "2024-06", "11": "2026-08", "12": "2028-06"},
    "alpine": {"3.14": "2023-05", "3.15": "2023-11", "3.16": "2024-05", "3.17": "2024-11", "3.18": "2025-05",
               "3.19": "2025-11", "3.20": "2026-04", "3.21": "2026-11", "3.22": "2027-05"},
    "php": {"7.4": "2022-11", "8.0": "2023-11", "8.1": "2025-12", "8.2": "2026-12", "8.3": "2027-12"},
    "ruby": {"2.7": "2023-03", "3.0": "2024-04", "3.1": "2025-03", "3.2": "2026-03", "3.3": "2027-03"},
}
RISKY_PACKAGES = {"openssh-server": ("high", "An SSH server inside a container is rarely needed and widens the attack surface."),
                  "openssh": ("medium", "SSH tooling in the image is rarely needed."),
                  "sudo": ("medium", "sudo is unnecessary in a container that already runs as a dedicated user."),
                  "telnet": ("medium", "telnet is an insecure legacy tool."), "netcat": ("low", "netcat is a common post-exploitation tool."),
                  "nmap": ("medium", "nmap is a scanning tool that does not belong in a runtime image."),
                  "vim": ("low", "Editors are unnecessary in runtime images."), "nano": ("low", "Editors are unnecessary in runtime images."),
                  "gdb": ("low", "Debuggers are unnecessary in runtime images."), "strace": ("low", "Debugging tools are unnecessary in runtime images.")}
DANGEROUS_PORTS = {22: "SSH", 23: "telnet", 3389: "RDP", 5900: "VNC", 2375: "unauthenticated Docker API",
                   2376: "Docker API", 445: "SMB", 3306: "MySQL", 5432: "PostgreSQL", 6379: "Redis", 27017: "MongoDB"}


class SecurityFinding(BaseModel):
    severity: str  # critical | high | medium | low | info
    category: str
    code: str
    message: str
    remediation: str
    line: Optional[int] = None


class BaseImageInfo(BaseModel):
    image: str
    stage: int
    pinned_tag: bool
    digest_pinned: bool
    variant: str            # slim | alpine | distroless | full | scratch | other
    eol: Optional[str] = None
    eol_status: str = "unknown"  # supported | near-eol | eol | unknown
    note: str = ""


class SecurityReport(BaseModel):
    findings: list[SecurityFinding] = Field(default_factory=list)
    base_images: list[BaseImageInfo] = Field(default_factory=list)
    leaked_files: list[str] = Field(default_factory=list)
    disclaimer: str = DISCLAIMER

    @property
    def counts(self) -> dict[str, int]:
        c = {s: 0 for s in SEVERITY_ORDER}
        for f in self.findings:
            c[f.severity] = c.get(f.severity, 0) + 1
        return c

    @property
    def score(self) -> int:
        return max(0, 100 - sum(SEVERITY_POINTS[f.severity] for f in self.findings))

    @property
    def risk_level(self) -> str:
        c = self.counts
        return "critical" if c["critical"] else "high" if c["high"] else "medium" if c["medium"] else \
            "low" if c["low"] else "no static findings"

    def recommendations(self) -> list[str]:
        seen: list[str] = []
        for f in sorted(self.findings, key=lambda f: SEVERITY_ORDER.index(f.severity)):
            if f.remediation not in seen and f.severity != "info":
                seen.append(f.remediation)
        return seen

    def format(self) -> str:
        lines = [f"Static security score: {self.score}/100 (highest severity: {self.risk_level})"]
        for f in sorted(self.findings, key=lambda f: SEVERITY_ORDER.index(f.severity)):
            lines.append(f"  [{f.severity}] {f.message}" + (f" (line {f.line})" if f.line else ""))
        lines += ["Recommendations:"] + [f"  - {r}" for r in self.recommendations()[:8]]
        lines.append(self.disclaimer)
        return "\n".join(lines)


# ----------------------------------------------------------------------------- helpers
def _split_stages(instrs: list[Instruction]) -> list[list[Instruction]]:
    stages: list[list[Instruction]] = []
    for ins in instrs:
        if ins.name == "FROM":
            stages.append([ins])
        elif stages:
            stages[-1].append(ins)
    return stages


def _month_to_date(ym: str) -> date:
    y, m = ym.split("-")
    return date(int(y), int(m), 1)


def _eol_key(repo: str, tag: str) -> Optional[str]:
    t = tag.lower()
    if repo == "python":
        m = re.match(r"(\d+\.\d+)", t)
    elif repo == "node":
        m = re.match(r"(\d+)", t)
    elif repo in ("ubuntu", "alpine", "php", "ruby"):
        m = re.match(r"(\d+\.\d+)", t)
    elif repo == "debian":
        m = re.match(r"(jessie|stretch|buster|bullseye|bookworm|\d+)", t)
    else:
        return None
    return m.group(1) if m else None


def analyze_base_image(image: str, stage: int, today: Optional[date] = None) -> Optional[BaseImageInfo]:
    today = today or date.today()
    if image.lower() == "scratch" or "$" in image:
        return BaseImageInfo(image=image, stage=stage, pinned_tag=True, digest_pinned=False, variant="scratch" if
                             image.lower() == "scratch" else "other", note="variable or scratch image; not analyzed")
    digest = "@sha256:" in image
    name = image.split("@")[0]
    repo_path, _, tag = name.rpartition(":") if ":" in name.split("/")[-1] else (name, "", "")
    repo = (repo_path or name).split("/")[-1].lower()
    low = (tag or "").lower()
    variant = ("distroless" if "distroless" in name.lower() else "alpine" if "alpine" in low or repo == "alpine" else
               "slim" if "slim" in low or "minimal" in low else "full" if repo in
               ("python", "node", "ruby", "php", "golang", "ubuntu", "debian", "openjdk") else "other")
    info = BaseImageInfo(image=image, stage=stage, pinned_tag=bool(tag) and tag != "latest", digest_pinned=digest,
                         variant=variant)
    key = _eol_key(repo, low) if tag else None
    table = EOL.get(repo)
    if table and key in table:
        info.eol = table[key]
        eol_date = _month_to_date(table[key])
        months_left = (eol_date.year - today.year) * 12 + eol_date.month - today.month
        info.eol_status = "eol" if months_left <= 0 else "near-eol" if months_left <= 6 else "supported"
    if repo == "node" and key and key.isdigit() and int(key) >= 17 and int(key) % 2 == 1:
        info.note = "odd-numbered Node.js releases are not LTS and are supported only briefly"
    if repo == "openjdk":
        info.note = "the openjdk image is deprecated; prefer eclipse-temurin"
    return info


def _sensitive_files_in_context(root: Path, limit: int = 200) -> list[str]:
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        rel_dir = Path(dirpath).relative_to(root)
        dirnames[:] = [d for d in dirnames if d != ".git" and not is_excluded_dir(d)]
        for name in filenames:
            p = Path(dirpath) / name
            if not p.is_symlink() and is_sensitive_file(p):
                found.append((rel_dir / name).as_posix())
        if len(found) >= limit:
            break
    return found


def _covered_by_copy(rel: str, sources: list[str]) -> bool:
    for src in sources:
        s = src.lstrip("/").removeprefix("./").rstrip("/")
        if s in ("", "."):
            return True
        if rel == s or rel.startswith(s + "/"):
            return True
        if any(ch in s for ch in "*?[") and re.fullmatch(re.escape(s).replace(r"\*", "[^/]*").replace(r"\?", "[^/]"), rel):
            return True
    return False


def _copy_sources(ins: Instruction) -> list[str]:
    if "--from=" in ins.args:
        return []
    try:
        toks = json.loads(ins.args) if ins.args.strip().startswith("[") else shlex.split(ins.args)
    except ValueError:
        return []
    toks = [t for t in toks if isinstance(t, str) and not t.startswith("--")]
    return toks[:-1] if len(toks) >= 2 else []


# ----------------------------------------------------------------------------- main analysis
def analyze_dockerfile_security(dockerfile: str, dockerignore: str = "", *, analysis: Optional[ProjectAnalysis] = None,
                                project_root: Optional[Path | str] = None, today: Optional[date] = None
                                ) -> SecurityReport:
    report = SecurityReport()
    add = lambda sev, cat, code, msg, rem, line=None: report.findings.append(  # noqa: E731
        SecurityFinding(severity=sev, category=cat, code=code, message=msg, remediation=rem, line=line))
    instrs, _ = parse_instructions(dockerfile)
    stages = _split_stages(instrs)
    if not stages:
        return report
    final = stages[-1]
    multi_stage = len(stages) > 1

    # 1. user
    users = [i for i in final if i.name == "USER"]
    if not users:
        add("high", "privilege", "runs-as-root", "The final stage never sets a non-root USER, so the app runs as root.",
            "Create an unprivileged user and switch to it with USER before CMD.")
    elif users[-1].args.strip().lower() in ("root", "0", "0:0"):
        add("high", "privilege", "runs-as-root", "The final stage runs as root.",
            "Switch to a non-root USER at the end of the final stage.", users[-1].line)

    # 2-9. instruction-level checks
    for ins in instrs:
        a, low = ins.args, ins.args.lower()
        if ins.name in ("ENV", "ARG"):
            pairs = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=(\"[^\"]*\"|'[^']*'|\S+)", a)
            if not pairs and ins.name == "ENV" and len(a.split(None, 1)) == 2:
                pairs = [tuple(a.split(None, 1))]  # type: ignore[list-item]
            for key, value in pairs:
                v = str(value).strip("\"'")
                if SECRET_KEY.search(key) and v and not v.startswith("$"):
                    add("critical", "secrets", "embedded-secret",
                        f"{ins.name} {key} stores a secret value in the image (visible in `docker history`).",
                        "Remove it; inject secrets at runtime (env, orchestrator secrets) or use BuildKit secret mounts.", ins.line)
            if ins.name == "ARG" and SECRET_KEY.search(a.split("=")[0]) and "=" not in a:
                add("medium", "secrets", "secret-build-arg",
                    f"ARG {a.split()[0]} looks like a secret; build args are recorded in image metadata.",
                    "Use `RUN --mount=type=secret` instead of build arguments for credentials.", ins.line)
        if redact_secrets(a) != a and ins.name != "ENV":
            add("critical", "secrets", "literal-credential", "A token/key/credential literal appears in the Dockerfile.",
                "Remove the credential and rotate it; use BuildKit secrets or runtime configuration.", ins.line)
        if re.search(r"https?://[^/\s:@]+:[^/\s@]+@", a):
            add("critical", "secrets", "credentials-in-url", "A URL embeds a username and password/token.",
                "Use a credential helper or BuildKit secret mount; rotate the exposed credential.", ins.line)
        if ins.name in ("COPY", "ADD"):
            for src in _copy_sources(ins):
                if is_sensitive_file(src) or re.search(r"(^|/)(\.ssh|\.aws|\.gnupg|\.docker)(/|$)", src):
                    add("high", "secrets", "copies-sensitive-file", f"Copies a sensitive file/dir into the image: {src}.",
                        "Never COPY credentials or keys; mount them at runtime or use build secrets.", ins.line)
            if re.search(r"--chmod=0?777|--chmod=a\+rwx", low):
                add("medium", "permissions", "chmod-777", "COPY sets world-writable permissions.",
                    "Use least-privilege modes such as 755/644 and chown to the app user.", ins.line)
            if ins.name == "ADD" and re.search(r"\bhttp://", a):
                add("high", "supply-chain", "add-insecure-url", "ADD downloads over plain HTTP.",
                    "Use HTTPS and verify a checksum/signature.", ins.line)
            elif ins.name == "ADD" and re.search(r"https://", a):
                add("medium", "supply-chain", "add-remote", "ADD fetches a remote URL without verification.",
                    "Download in a RUN with checksum verification, or vendor the file.", ins.line)
        if ins.name in ("RUN", "CMD", "ENTRYPOINT"):
            if re.search(r"(curl|wget)[^|;&\n]*\|\s*(sudo\s+)?(ba|z|da)?sh\b", low):
                add("high", "unsafe-shell", "pipe-to-shell", "Downloads and executes a script in one step (curl | sh).",
                    "Download, verify a checksum/signature, then execute.", ins.line)
            if re.search(r"\beval\b", low):
                add("medium", "unsafe-shell", "eval", "`eval` executes dynamically built commands.",
                    "Avoid eval; call the command directly.", ins.line)
            if re.search(r"--no-check-certificate|curl[^\n]*\s(-k|--insecure)\b|wget[^\n]*--no-check", low):
                add("high", "supply-chain", "tls-verification-disabled", "TLS certificate verification is disabled.",
                    "Remove the flag; install the proper CA certificates instead.", ins.line)
            if re.search(r"strict-ssl\s+false|node_tls_reject_unauthorized\s*=\s*0|http\.sslverify\s+false|"
                         r"--trusted-host|pip[^\n]*--index-url\s+http://|verify=false", low):
                add("high" if "trusted-host" not in low else "medium", "supply-chain", "insecure-package-source",
                    "Package installation bypasses TLS verification or uses an insecure index.",
                    "Use HTTPS indexes with verification enabled.", ins.line)
            if re.search(r"--allow-unauthenticated|--force-yes|--allow-insecure-repositories|gpg-check\s*=\s*0|--nogpgcheck", low):
                add("high", "supply-chain", "unauthenticated-packages", "Package signature verification is disabled.",
                    "Remove the flag so packages are authenticated.", ins.line)
            if re.search(r"(?:git\+|git://|npm\s+(?:i|install)[^\n]*\s)(?:git|http)://", low) or re.search(r"pip3?\s+install[^\n]*\bhttp://", low):
                add("medium", "supply-chain", "insecure-install-source", "Installs a dependency from an unencrypted source.",
                    "Use HTTPS (or a registry) and pin versions.", ins.line)
            if re.search(r"(?:curl|wget)[^\n|;&]*\bhttp://", low):
                add("medium", "supply-chain", "http-download", "Downloads over plain HTTP.", "Use HTTPS with checksum verification.", ins.line)
            if re.search(r"chmod\s+(-R\s+)?0?777|chmod\s+(-R\s+)?a\+rwx|chmod\s+(-R\s+)?o\+w", low):
                add("medium", "permissions", "chmod-777", "Files are made world-writable.",
                    "Grant only the required permissions to the app user.", ins.line)
            if re.search(r"chmod\s+[^\n]*[ugo]?\+s\b|chmod\s+[24]\d{3}\b", low):
                add("high", "permissions", "setuid", "Sets a setuid/setgid bit.", "Remove setuid/setgid bits.", ins.line)
            if re.search(r"--privileged|docker\.sock", low):
                add("high", "privilege", "privileged-or-socket", "Uses privileged mode or the Docker socket.",
                    "Remove it; containers must not control the Docker daemon.", ins.line)
            if re.search(r"\bsudo\b", low) and ins.name == "RUN" and "install" not in low:
                add("low", "privilege", "sudo", "sudo is used during the build.", "Perform root steps before USER; avoid sudo.", ins.line)
        if ins.name == "VOLUME" and "docker.sock" in low:
            add("high", "privilege", "docker-socket", "Declares the Docker socket as a volume.",
                "Never expose the Docker socket to containers.", ins.line)
        if ins.name == "EXPOSE":
            ports = [int(p.split("/")[0]) for p in a.split() if p.split("/")[0].isdigit()]
            for p in ports:
                if p in DANGEROUS_PORTS:
                    add("high" if p in (22, 23, 3389, 5900, 2375, 445) else "low", "network", "risky-port",
                        f"EXPOSE {p} ({DANGEROUS_PORTS[p]}) is unusual for an application image.",
                        "Expose only the application's service port.", ins.line)
            if len(ports) > 3:
                add("low", "network", "many-ports", f"{len(ports)} ports are exposed.",
                    "Expose only the ports the application needs.", ins.line)

    # 3. base images
    for idx, stage in enumerate(stages):
        parts = [t for t in stage[0].args.split() if not t.startswith("--")]
        image = parts[0] if parts else ""
        stage_names = {s[0].args.split()[-1].lower() for s in stages[:idx] if " as " in s[0].args.lower()}
        if not image or image.lower() in stage_names:
            continue
        bi = analyze_base_image(image, idx, today)
        if not bi:
            continue
        report.base_images.append(bi)
        line = stage[0].line
        if bi.variant == "scratch" or "$" in image:
            continue
        if not bi.pinned_tag:
            add("medium", "base-image", "unpinned-image", f"Base image '{image}' is unpinned (latest).",
                "Pin an explicit version tag (and optionally a digest).", line)
        elif not bi.digest_pinned and idx == len(stages) - 1:
            add("info", "base-image", "no-digest", f"'{image}' is pinned by tag, not digest; tags can be re-pushed.",
                "For high-assurance builds pin the image digest (image@sha256:...).", line)
        if bi.eol_status == "eol":
            add("high", "base-image", "eol-base-image",
                f"Base image '{image}' reached end of life around {bi.eol} (approximate) and no longer gets security fixes.",
                "Move to a supported release (check https://endoflife.date).", line)
        elif bi.eol_status == "near-eol":
            add("low", "base-image", "near-eol-base-image", f"Base image '{image}' nears end of life ({bi.eol}, approximate).",
                "Plan an upgrade to a newer supported release.", line)
        if bi.note:
            add("low", "base-image", "base-image-note", f"{image}: {bi.note}.", "Choose a maintained image.", line)
        if bi.variant == "full" and not multi_stage:
            add("low", "base-image", "full-base-image", f"'{image}' is a full image with a large package surface.",
                "Prefer -slim/-alpine/distroless runtime images.", line)

    # 4. unnecessary packages in the final stage
    for ins in final:
        if ins.name != "RUN":
            continue
        m = re.search(r"(?:apt-get|apt)\s+install\b(.*?)(?:&&|;|$)|apk\s+add\b(.*?)(?:&&|;|$)", ins.args, re.S)
        if not m:
            continue
        pkgs = set(re.findall(r"[a-z0-9][a-z0-9.+-]*", (m.group(1) or m.group(2) or "").lower()))
        for pkg, (sev, why) in RISKY_PACKAGES.items():
            if pkg in pkgs:
                add(sev, "attack-surface", "risky-package", f"Package '{pkg}' is installed in the final image. {why}",
                    f"Remove {pkg} from the final stage.", ins.line)
        if not multi_stage and pkgs & {"gcc", "g++", "build-essential", "make", "cmake"}:
            add("low", "attack-surface", "build-tools-in-final", "Compilers/build tools remain in the final image.",
                "Compile in a build stage and copy only the artifacts into a minimal runtime stage.", ins.line)
        if "apt" in ins.args and "--no-install-recommends" not in ins.args:
            add("low", "attack-surface", "install-recommends", "apt installs recommended packages that are not needed.",
                "Use --no-install-recommends.", ins.line)

    # 10. build context
    root = Path(project_root) if project_root else (Path(analysis.root) if analysis else None)
    if root and root.is_dir():
        sources = [s for ins in instrs if ins.name in ("COPY", "ADD") for s in _copy_sources(ins)]
        leaked = [f for f in _sensitive_files_in_context(root) if not is_ignored(f, dockerignore) and
                  _covered_by_copy(f, sources)]
        report.leaked_files = leaked
        for f in leaked[:10]:
            private_key = bool(re.search(r"(id_rsa|id_dsa|id_ecdsa|id_ed25519|\.pem|\.key|\.p12|\.pfx|\.jks)$", f))
            add("critical" if private_key or Path(f).name.startswith(".env") else "high", "secrets", "sensitive-file-in-image",
                f"Sensitive file '{f}' is not excluded by .dockerignore and is copied into the image.",
                f"Add '{f}' to .dockerignore (and avoid COPY of directories that contain it).")
        if len(leaked) > 10:
            add("high", "secrets", "sensitive-files-in-image", f"{len(leaked) - 10} more sensitive files would be copied.",
                "Review .dockerignore.")
        if (root / ".git").is_dir() and not is_ignored(".git/config", dockerignore) and \
                _covered_by_copy(".git/config", sources):
            add("medium", "secrets", "git-dir-in-image", "The .git directory (full history, possibly credentials) would be copied.",
                "Add .git to .dockerignore.")
        if not dockerignore.strip() and _sensitive_files_in_context(root, 1):
            add("medium", "secrets", "no-dockerignore", "There is no .dockerignore, so sensitive files are sent to the Docker daemon.",
                "Create a .dockerignore that excludes .env*, keys and VCS data.")
    return report


OVERLAP_WITH_VALIDATOR = {"runs-as-root", "embedded-secret", "pipe-to-shell", "privileged-or-socket", "docker-socket",
                          "add-remote", "chmod-777", "sudo", "unpinned-image", "latest-tag", "copies-env-file"}


def quality_findings(dockerfile: str, dockerignore: str, analysis: Optional[ProjectAnalysis],
                     project_root: Optional[Path | str]):
    """Adapter for the quality scorer: security findings the lint validator does not already report."""
    from app.docker.quality import Finding  # local import avoids a module cycle

    out = []
    for f in analyze_dockerfile_security(dockerfile, dockerignore, analysis=analysis, project_root=project_root).findings:
        if f.code in OVERLAP_WITH_VALIDATOR or f.severity == "info":
            continue
        sev = "high" if f.severity in ("critical", "high") else f.severity
        out.append(Finding(category="security", severity=sev, code=f.code, message=f.message,
                           recommendation=f.remediation, line=f.line))
    return out

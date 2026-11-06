"""Optional image vulnerability scanners (Trivy, Docker Scout). Both are optional external tools.

Scans only ever run against an image the user already built; nothing is executed from inside the image.
Trivy parsing follows its documented JSON schema. Docker Scout SARIF parsing is best-effort and has not been
verified against a live Scout installation.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from typing import Callable, Optional

from pydantic import BaseModel, Field

from app.utils.security import redact_secrets

SEVERITIES = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "UNKNOWN"]
Runner = Callable[..., "subprocess.CompletedProcess[str]"]


class Vulnerability(BaseModel):
    id: str
    package: str
    installed: str = ""
    fixed: str = ""
    severity: str = "UNKNOWN"
    title: str = ""
    target_class: str = ""  # os-pkgs | lang-pkgs | other


class ScanReport(BaseModel):
    tool: str
    available: bool = True
    ok: bool = True
    error: str = ""
    image: str = ""
    counts: dict[str, int] = Field(default_factory=lambda: {s: 0 for s in SEVERITIES})
    vulnerabilities: list[Vulnerability] = Field(default_factory=list)
    os_family: str = ""
    os_name: str = ""
    os_eol: Optional[bool] = None
    by_class: dict[str, int] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    @property
    def fixable(self) -> int:
        return sum(1 for v in self.vulnerabilities if v.fixed)

    def top(self, n: int = 10) -> list[Vulnerability]:
        order = {s: i for i, s in enumerate(SEVERITIES)}
        return sorted(self.vulnerabilities, key=lambda v: (order.get(v.severity, 9), not v.fixed, v.id))[:n]

    def remediation(self) -> list[str]:
        out: list[str] = []
        if self.os_eol:
            out.append(f"The base OS ({self.os_family} {self.os_name}) is end-of-life: move to a supported base image.")
        if self.by_class.get("os-pkgs"):
            out.append("Rebuild on a newer/slimmer base image to pick up OS package fixes.")
        if self.by_class.get("lang-pkgs"):
            out.append("Upgrade the vulnerable application dependencies listed (fixed versions are shown where known).")
        if self.fixable:
            out.append(f"{self.fixable} finding(s) have a fixed version available.")
        return out


def _unavailable(tool: str, image: str, why: str) -> ScanReport:
    return ScanReport(tool=tool, available=False, ok=False, image=image, error=why)


def parse_trivy(raw: str | dict, image: str = "") -> ScanReport:
    data = json.loads(raw) if isinstance(raw, str) else raw
    rep = ScanReport(tool="trivy", image=image or data.get("ArtifactName", ""))
    os_meta = (data.get("Metadata") or {}).get("OS") or {}
    rep.os_family, rep.os_name = os_meta.get("Family", ""), str(os_meta.get("Name", ""))
    rep.os_eol = os_meta.get("EOSL") if isinstance(os_meta.get("EOSL"), bool) else None
    for res in data.get("Results") or []:
        cls = res.get("Class", "other")
        for v in res.get("Vulnerabilities") or []:
            sev = str(v.get("Severity", "UNKNOWN")).upper()
            sev = sev if sev in SEVERITIES else "UNKNOWN"
            rep.counts[sev] += 1
            rep.by_class[cls] = rep.by_class.get(cls, 0) + 1
            rep.vulnerabilities.append(Vulnerability(
                id=v.get("VulnerabilityID", "?"), package=v.get("PkgName", ""), installed=v.get("InstalledVersion", ""),
                fixed=v.get("FixedVersion", "") or "", severity=sev, title=redact_secrets(v.get("Title", ""))[:200],
                target_class=cls))
    return rep


def parse_scout_sarif(raw: str | dict, image: str = "") -> ScanReport:
    """Best-effort SARIF parsing (level error->HIGH, warning->MEDIUM, note->LOW; CVSS in rule properties if present)."""
    data = json.loads(raw) if isinstance(raw, str) else raw
    rep = ScanReport(tool="docker-scout", image=image,
                     notes=["Docker Scout SARIF parsing is best-effort and unverified against a live Scout."])
    for run in data.get("runs") or []:
        rules = {r.get("id"): r for r in (run.get("tool", {}).get("driver", {}).get("rules") or [])}
        for res in run.get("results") or []:
            rid = res.get("ruleId", "?")
            props = (rules.get(rid) or {}).get("properties") or {}
            sev = str(props.get("cvssSeverity") or "").upper() or {"error": "HIGH", "warning": "MEDIUM", "note": "LOW"}.get(
                res.get("level", ""), "UNKNOWN")
            sev = sev if sev in SEVERITIES else "UNKNOWN"
            rep.counts[sev] += 1
            rep.vulnerabilities.append(Vulnerability(
                id=rid, package=str(props.get("purls", [""])[0] if props.get("purls") else ""), severity=sev,
                fixed=str(props.get("fixed_version", "") or ""),
                title=redact_secrets((res.get("message") or {}).get("text", ""))[:200]))
    return rep


def _run(tool: str, cmd: list[str], image: str, runner: Runner, timeout: int, parser) -> ScanReport:
    try:
        proc = runner(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return ScanReport(tool=tool, ok=False, image=image, error=f"{tool} timed out after {timeout}s")
    except OSError as exc:
        return ScanReport(tool=tool, ok=False, image=image, error=f"could not run {tool}: {exc}")
    if proc.returncode != 0 and not (proc.stdout or "").strip():
        return ScanReport(tool=tool, ok=False, image=image, error=redact_secrets((proc.stderr or "")[-500:]))
    try:
        return parser(proc.stdout, image)
    except (ValueError, AttributeError, TypeError) as exc:
        return ScanReport(tool=tool, ok=False, image=image, error=f"unparseable {tool} output: {exc}")


def run_trivy(image: str, *, runner: Runner = subprocess.run, which: Callable[[str], Optional[str]] = shutil.which,
              timeout: int = 600) -> ScanReport:
    if not which("trivy"):
        return _unavailable("trivy", image, "Trivy is not installed (optional). See https://trivy.dev")
    return _run("trivy", ["trivy", "image", "--quiet", "--format", "json", image], image, runner, timeout, parse_trivy)


def run_scout(image: str, *, runner: Runner = subprocess.run, which: Callable[[str], Optional[str]] = shutil.which,
              timeout: int = 600) -> ScanReport:
    if not which("docker"):
        return _unavailable("docker-scout", image, "Docker is not installed.")
    return _run("docker-scout", ["docker", "scout", "cves", "--format", "sarif", image], image, runner, timeout,
                parse_scout_sarif)


def scan_image(image: str, **kw) -> list[ScanReport]:
    """Run every available scanner; unavailable ones are reported, not hidden."""
    return [run_trivy(image, **kw), run_scout(image, **kw)]

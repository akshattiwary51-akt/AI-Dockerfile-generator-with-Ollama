import json
import shutil
from pathlib import Path

import pytest

from app.main import main
from app.security.scanners import ScanReport, Vulnerability

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "flask_app"


@pytest.fixture
def proj(tmp_path, monkeypatch):
    d = tmp_path / "flask_app"
    shutil.copytree(FIXTURE, d)
    monkeypatch.setenv("HISTORY_DB", str(tmp_path / "h.db"))
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")
    return d


def test_score_text_and_json_and_gate(proj, capsys):
    assert main(["score", str(proj)]) == 0
    out = capsys.readouterr().out
    assert "Overall quality" in out and "Static security score" in out and "cannot find vulnerable packages" in out
    assert main(["score", str(proj), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert 0 <= data["quality"]["overall"] <= 100 and "findings" in data["security"]
    assert main(["score", str(proj), "--min-score", "101"]) == 1


def test_score_missing_dockerfile(proj, capsys):
    (proj / "Dockerfile").unlink()
    assert main(["score", str(proj)]) == 2 and "No Dockerfile" in capsys.readouterr().err


def test_score_flags_a_bad_dockerfile(proj, capsys):
    (proj / "Dockerfile").write_text("FROM python\nENV API_TOKEN=abc123\nCOPY . .\nCMD python app.py\n")
    assert main(["score", str(proj), "--min-score", "60"]) == 1
    out = capsys.readouterr().out
    assert "API_TOKEN" in out and "[high]" in out


def test_scan_gate_and_missing_tools(monkeypatch, capsys):
    bad = ScanReport(tool="trivy", image="x", counts={"CRITICAL": 1, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "UNKNOWN": 0},
                     vulnerabilities=[Vulnerability(id="CVE-1", package="openssl", severity="CRITICAL", fixed="1.2")])
    gone = ScanReport(tool="docker-scout", available=False, ok=False, error="no docker")
    monkeypatch.setattr("app.security.scanners.scan_image", lambda image: [bad, gone])
    assert main(["scan", "x"]) == 0 and "CVE-1" in capsys.readouterr().out
    assert main(["scan", "x", "--fail-on", "critical"]) == 1
    monkeypatch.setattr("app.security.scanners.scan_image", lambda image: [gone])
    assert main(["scan", "x"]) == 2


def test_optimize_writes_variant_and_never_touches_dockerfile(proj, capsys):
    original = (proj / "Dockerfile").read_text()
    (proj / "Dockerfile").write_text(original.replace("--no-cache-dir ", ""))
    before = (proj / "Dockerfile").read_text()
    assert main(["optimize", str(proj), "--write"]) == 0
    assert (proj / "Dockerfile").read_text() == before and "--no-cache-dir" in (proj / "Dockerfile.optimized").read_text()
    assert main(["optimize", str(proj), "--build"]) == 2  # needs --image


def test_optimize_build_requires_approval(proj, monkeypatch):
    (proj / "Dockerfile").write_text((proj / "Dockerfile").read_text().replace("--no-cache-dir ", ""))
    monkeypatch.setattr("sys.stdin", None)
    assert main(["optimize", str(proj), "--build", "--image", "x:1"]) == 3  # non-interactive, not approved


def test_history_listing_export_and_stats(proj, tmp_path, capsys):
    from app.history.store import HistoryStore
    from tests.unit.test_history import report
    s = HistoryStore(tmp_path / "h.db")
    s.record(report("alpha", True))
    s.record(report("beta", False, category="network", error="dns failure"))
    assert main(["history"]) == 0 and "beta" in capsys.readouterr().out
    assert main(["history", "--failed"]) == 0
    out = capsys.readouterr().out
    assert "beta" in out and "alpha" not in out
    assert main(["history", "--export", "csv"]) == 0 and capsys.readouterr().out.startswith("id,created_at")
    assert main(["history", "--stats"]) == 0 and json.loads(capsys.readouterr().out)["executions"] == 2
    assert main(["history", "--search", "dns"]) == 0 and "beta" in capsys.readouterr().out


def test_models_without_ollama_degrades_gracefully(proj, capsys):
    assert main(["models"]) == 0
    assert "Installed-model assessment unavailable" in capsys.readouterr().out
    assert main(["models", "--benchmark", str(proj)]) == 2


def test_backward_compatible_commands_still_registered():
    from app.main import build_parser
    parser = build_parser()
    for cmd in ("doctor", "models", "analyze", "run", "fix", "selftest"):
        assert parser.parse_args([cmd] + ([] if cmd in ("doctor", "models", "selftest") else ["x"])).command == cmd

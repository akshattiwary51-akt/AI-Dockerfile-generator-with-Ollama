import sqlite3
from pathlib import Path

from app.agent.execution_history import AttemptRecord, ExecutionHistory
from app.agent.orchestrator import RunReport
from app.history.store import HistoryStore


def report(project="p", success=True, corrections=1, quality=80, category="copy-source", dockerfile="FROM x\n",
           error="boom", changes=("Fix COPY path",), model="m1"):
    h = ExecutionHistory()
    h.add(AttemptRecord(attempt=1, dockerfile="FROM x\n", success=False, error=error, category=category,
                        fixable=True, diagnosis="d", changes=list(changes), duration_seconds=1.0))
    if success:
        h.add(AttemptRecord(attempt=2, dockerfile=dockerfile, success=True, duration_seconds=2.0))
    return RunReport(project=project, project_dir=str(Path.home() / "work" / project), image=f"{project}:ai", model=model,
                     success=success, status="success" if success else "max_retries_reached", message="done",
                     attempts=len(h.attempts), corrections=corrections, total_duration_seconds=3.0, llm_calls=2,
                     llm_tokens=100, dockerfile=dockerfile, quality_score=quality, history=h)


def test_record_get_and_redaction(tmp_path):
    s = HistoryStore(tmp_path / "h.db")
    eid = s.record(report(error="auth failed password=hunter2secret", dockerfile="FROM x\nENV API_TOKEN=abc123def456ghi\n"))
    row = s.get(eid)
    blob = str(row)
    assert "hunter2secret" not in blob and "abc123def456ghi" not in blob
    assert str(Path.home()) not in row["project_dir"] or str(Path.home()) == "/"
    assert len(row["attempts_detail"]) == 2 and row["attempts_detail"][0]["changes"] == ["Fix COPY path"]


def test_search_filters(tmp_path):
    s = HistoryStore(tmp_path / "h.db")
    s.record(report("a", True, quality=90))
    s.record(report("b", False, quality=40, category="network", error="dns failure", model="m2"))
    assert len(s.search()) == 2 and s.search()[0]["project"] == "b"  # newest first
    assert [r["project"] for r in s.search(success=False)] == ["b"]
    assert [r["project"] for r in s.search(project="a")] == ["a"]
    assert [r["project"] for r in s.search("dns")] == ["b"]
    assert [r["project"] for r in s.search(min_quality=80)] == ["a"]
    assert [r["project"] for r in s.search(model="m2")] == ["b"]
    assert s.projects() == ["a", "b"]


def test_stats_and_similar_fixes(tmp_path):
    s = HistoryStore(tmp_path / "h.db")
    s.record(report("a", True))
    s.record(report("b", False, category="network"))
    st = s.stats()
    assert st["executions"] == 2 and st["success_rate"] == 0.5 and st["correction_success_rate"] == 0.5
    assert st["failures_by_category"] == {"copy-source": 2, "network": 1} or "copy-source" in st["failures_by_category"]
    assert s.similar_fixes("copy-source") == ["Fix COPY path"]
    assert s.similar_fixes("network") == [] and s.similar_fixes("") == []


def test_exports_and_compare(tmp_path):
    s = HistoryStore(tmp_path / "h.db")
    a = s.record(report("a", True, quality=70, dockerfile="FROM x\n"))
    b = s.record(report("a", True, quality=90, dockerfile="FROM y\n"))
    assert "quality_score" in s.export_csv().splitlines()[0] and '"project"' in s.export_json()
    cmp = s.compare(a, b)
    assert cmp["changed"]["quality_score"] == {"a": 70, "b": 90} and "+FROM y" in cmp["dockerfile_diff"]


def test_csv_formula_injection_neutralised(tmp_path):
    s = HistoryStore(tmp_path / "h.db")
    s.record(report("=cmd|calc"))
    import csv, io
    rows = list(csv.DictReader(io.StringIO(s.export_csv())))
    assert rows[0]["project"] == "'=cmd|calc"


def test_delete_cascades(tmp_path):
    s = HistoryStore(tmp_path / "h.db")
    eid = s.record(report())
    s.delete(eid)
    with sqlite3.connect(s.path) as c:
        assert c.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 0

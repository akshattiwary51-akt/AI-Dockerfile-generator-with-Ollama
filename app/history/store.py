"""Persistent SQLite execution history (local only). Everything is redacted before it is stored; the store never
holds credentials, and paths under the user's home directory are shortened to ``~``."""
from __future__ import annotations

import csv
import io
import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

from app.utils.security import redact_secrets

DEFAULT_DB = Path(__file__).resolve().parents[2] / "generated" / "history.db"
SCHEMA = """
CREATE TABLE IF NOT EXISTS executions (
  id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, project TEXT, project_dir TEXT, image TEXT, model TEXT,
  success INTEGER, status TEXT, message TEXT, attempts INTEGER, corrections INTEGER, duration REAL,
  quality_score INTEGER, llm_calls INTEGER, llm_seconds REAL, llm_tokens INTEGER, image_size_mb REAL,
  dockerfile TEXT, dockerignore TEXT);
CREATE TABLE IF NOT EXISTS attempts (
  id INTEGER PRIMARY KEY AUTOINCREMENT, execution_id INTEGER REFERENCES executions(id) ON DELETE CASCADE,
  n INTEGER, success INTEGER, category TEXT, fixable INTEGER, failed_stage TEXT, error TEXT, diagnosis TEXT,
  explanation TEXT, changes TEXT, diff TEXT, duration REAL);
CREATE INDEX IF NOT EXISTS idx_exec_project ON executions(project);
CREATE INDEX IF NOT EXISTS idx_att_cat ON attempts(category);
"""
EXEC_COLS = ["id", "created_at", "project", "image", "model", "success", "status", "attempts", "corrections",
             "duration", "quality_score", "llm_calls", "llm_seconds", "llm_tokens", "image_size_mb", "message"]


def _clean(text: Any) -> str:
    s = redact_secrets(str(text or ""))
    home = str(Path.home())
    return s.replace(home, "~") if home and home != "/" else s


class HistoryStore:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else DEFAULT_DB
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys=ON")
        return c

    # ------------------------------------------------------------------ write
    def record(self, report) -> int:
        """Store a RunReport (duck-typed) and its attempts; returns the execution id."""
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO executions (created_at,project,project_dir,image,model,success,status,message,attempts,"
                "corrections,duration,quality_score,llm_calls,llm_seconds,llm_tokens,image_size_mb,dockerfile,"
                "dockerignore) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (report.created_at, _clean(report.project), _clean(report.project_dir), _clean(report.image),
                 _clean(report.model), int(report.success), report.status, _clean(report.message), report.attempts,
                 report.corrections, report.total_duration_seconds, report.quality_score, report.llm_calls,
                 report.llm_seconds, report.llm_tokens, report.image_size_mb, _clean(report.dockerfile),
                 _clean(report.dockerignore)))
            eid = cur.lastrowid
            for a in report.history.attempts:
                c.execute("INSERT INTO attempts (execution_id,n,success,category,fixable,failed_stage,error,diagnosis,"
                          "explanation,changes,diff,duration) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                          (eid, a.attempt, int(a.success), a.category, None if a.fixable is None else int(a.fixable),
                           a.failed_stage, _clean(a.error), _clean(a.diagnosis), _clean(a.explanation),
                           json.dumps([_clean(x) for x in a.changes]), _clean(a.diff), a.duration_seconds))
        return int(eid)

    def delete(self, execution_id: int) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM executions WHERE id=?", (execution_id,))

    # ------------------------------------------------------------------ read
    def search(self, text: str = "", *, project: str = "", success: Optional[bool] = None, model: str = "",
               status: str = "", since: str = "", min_quality: Optional[int] = None, limit: int = 200
               ) -> list[dict[str, Any]]:
        where, args = [], []  # type: list[str], list[Any]
        if project:
            where.append("project = ?"); args.append(project)
        if success is not None:
            where.append("success = ?"); args.append(int(success))
        if model:
            where.append("model = ?"); args.append(model)
        if status:
            where.append("status = ?"); args.append(status)
        if since:
            where.append("created_at >= ?"); args.append(since)
        if min_quality is not None:
            where.append("quality_score >= ?"); args.append(min_quality)
        if text:
            like = f"%{text}%"
            where.append("(project LIKE ? OR message LIKE ? OR image LIKE ? OR id IN (SELECT execution_id FROM attempts "
                         "WHERE error LIKE ? OR diagnosis LIKE ? OR category LIKE ?))")
            args += [like] * 6
        sql = f"SELECT {','.join(EXEC_COLS)} FROM executions" + (" WHERE " + " AND ".join(where) if where else "") + \
              " ORDER BY id DESC LIMIT ?"
        with self._conn() as c:
            return [dict(r) for r in c.execute(sql, args + [limit])]

    def get(self, execution_id: int) -> Optional[dict[str, Any]]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM executions WHERE id=?", (execution_id,)).fetchone()
            if not row:
                return None
            out = dict(row)
            out["attempts_detail"] = []
            for a in c.execute("SELECT * FROM attempts WHERE execution_id=? ORDER BY n", (execution_id,)):
                d = dict(a)
                d["changes"] = json.loads(d["changes"] or "[]")
                out["attempts_detail"].append(d)
            return out

    def projects(self) -> list[str]:
        with self._conn() as c:
            return [r[0] for r in c.execute("SELECT DISTINCT project FROM executions ORDER BY project")]

    def stats(self) -> dict[str, Any]:
        with self._conn() as c:
            total, ok = c.execute("SELECT COUNT(*), COALESCE(SUM(success),0) FROM executions").fetchone()
            fixed = c.execute("SELECT COUNT(*), COALESCE(SUM(success),0) FROM executions WHERE corrections>0").fetchone()
            avg_q = c.execute("SELECT AVG(quality_score) FROM executions WHERE quality_score>0").fetchone()[0]
            cats = c.execute("SELECT category, COUNT(*) n FROM attempts WHERE success=0 AND category!='' "
                             "GROUP BY category ORDER BY n DESC").fetchall()
        return {"executions": total, "successes": ok, "success_rate": round(ok / total, 3) if total else None,
                "runs_needing_correction": fixed[0], "corrected_successfully": fixed[1],
                "correction_success_rate": round(fixed[1] / fixed[0], 3) if fixed[0] else None,
                "avg_quality": round(avg_q, 1) if avg_q is not None else None,
                "failures_by_category": {r["category"]: r["n"] for r in cats}}

    def similar_fixes(self, category: str, limit: int = 3) -> list[str]:
        """Changes applied after this error category in runs that ended successfully (local reference only; the
        model is not trained or modified by this)."""
        if not category:
            return []
        with self._conn() as c:
            rows = c.execute("SELECT a.changes FROM attempts a JOIN executions e ON e.id=a.execution_id "
                             "WHERE a.category=? AND a.success=0 AND e.success=1 AND a.changes!='[]' "
                             "ORDER BY a.id DESC LIMIT 20", (category,)).fetchall()
        out: list[str] = []
        for r in rows:
            for ch in json.loads(r["changes"]):
                if ch and ch not in out:
                    out.append(ch)
        return out[:limit]

    # ------------------------------------------------------------------ export / compare
    def export_json(self, **filters: Any) -> str:
        return json.dumps(self.search(**filters), indent=2)

    def export_csv(self, **filters: Any) -> str:
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=EXEC_COLS)
        w.writeheader()
        for row in self.search(**filters):
            w.writerow({k: ("'" + v if isinstance(v, str) and v[:1] in ("=", "+", "-", "@") else v)
                        for k, v in row.items()})  # neutralise spreadsheet-formula injection
        return buf.getvalue()

    def compare(self, a_id: int, b_id: int) -> dict[str, Any]:
        a, b = self.get(a_id), self.get(b_id)
        if not a or not b:
            raise KeyError("execution not found")
        fields = ["success", "status", "attempts", "corrections", "duration", "quality_score", "image_size_mb",
                  "llm_tokens", "model"]
        diff = {f: {"a": a[f], "b": b[f]} for f in fields if a[f] != b[f]}
        import difflib
        df = "\n".join(difflib.unified_diff(a["dockerfile"].splitlines(), b["dockerfile"].splitlines(),
                                            f"run {a_id}", f"run {b_id}", lineterm=""))
        return {"changed": diff, "dockerfile_diff": df}

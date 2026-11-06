"""Disposable faulty projects for testing the self-correction loop (never touches real projects)."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

APP_PY = 'from flask import Flask\n\napp = Flask(__name__)\n\n\n@app.get("/")\ndef hi():\n    return "ok"\n'

GOOD_BODY = """\
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY {req} .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
"""


class Fault(BaseModel):
    name: str
    description: str
    expected_fix: str
    difficulty: str
    requirements: str
    dockerfile: str


FAULTS: dict[str, Fault] = {
    "copy-typo": Fault(
        name="copy-typo", difficulty="easy",
        description="COPY references 'requirement.txt' but the file is 'requirements.txt'.",
        expected_fix="Change the COPY source to requirements.txt.",
        requirements="flask==3.0.0\ngunicorn==21.2.0\n",
        dockerfile="FROM python:3.12-slim\n" + GOOD_BODY.format(req="requirement.txt")),
    "bad-tag": Fault(
        name="bad-tag", difficulty="easy",
        description="Base image tag 'python:3.12-slimm' does not exist (typo).",
        expected_fix="Use a valid tag such as python:3.12-slim.",
        requirements="flask==3.0.0\ngunicorn==21.2.0\n",
        dockerfile="FROM python:3.12-slimm\n" + GOOD_BODY.format(req="requirements.txt")),
    "missing-dep": Fault(
        name="missing-dep", difficulty="hard",
        description="psycopg2 compiles from source but the slim image has no gcc / libpq-dev (pg_config not found).",
        expected_fix="apt-get install --no-install-recommends gcc libpq-dev before pip install (same layer, clean apt lists).",
        requirements="flask==3.0.0\ngunicorn==21.2.0\npsycopg2==2.9.9\n",
        dockerfile="FROM python:3.12-slim\n" + GOOD_BODY.format(req="requirements.txt")),
}


def create_faulty_project(fault: str, base_dir: Optional[Path] = None) -> Path:
    """Create a fresh disposable Flask project containing the chosen fault; returns its path."""
    if fault not in FAULTS:
        raise ValueError(f"Unknown fault '{fault}'. Choose from: {', '.join(FAULTS)}")
    f = FAULTS[fault]
    root = Path(tempfile.mkdtemp(prefix=f"selftest-{fault}-", dir=str(base_dir) if base_dir else None))
    project = root / f"selftest-{fault}"
    project.mkdir()
    (project / "app.py").write_text(APP_PY, encoding="utf-8")
    (project / "requirements.txt").write_text(f.requirements, encoding="utf-8")
    (project / "Dockerfile").write_text(f.dockerfile, encoding="utf-8")
    (project / ".dockerignore").write_text(".git\n.env\n.env.*\n__pycache__/\n*.pyc\n.venv/\n", encoding="utf-8")
    return project

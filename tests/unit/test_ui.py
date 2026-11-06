"""Headless Streamlit checks (AppTest). Ollama and Docker are absent, so pages must degrade gracefully."""
import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

from app.ui.dashboard import PAGES  # noqa: E402

from pathlib import Path  # noqa: E402

APP = str(Path(__file__).resolve().parents[2] / "app" / "ui" / "dashboard.py")


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("HISTORY_DB", str(tmp_path / "h.db"))
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")  # nothing listens here
    from app.utils import config
    if hasattr(config.get_settings, "cache_clear"):
        config.get_settings.cache_clear()


@pytest.mark.parametrize("label", [p[0] for p in PAGES])
def test_every_page_renders_without_exception(label, tmp_path):
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["nav"] = label
    at.session_state["project_dir"] = str(tmp_path)
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_analyzer_with_project(tmp_path):
    (tmp_path / "requirements.txt").write_text("flask\n")
    (tmp_path / "app.py").write_text("from flask import Flask\napp = Flask(__name__)\n")
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["nav"] = "Project Analyzer"
    at.session_state["project_dir"] = str(tmp_path)
    at.run()
    next(b for b in at.button if "Analyze" in b.label).click().run()
    assert not at.exception and at.session_state["analysis"].framework == "Flask"


@pytest.mark.parametrize("label", ["Dockerfile Studio", "Security Center", "Image Optimization Center",
                                   "Execution History", "Deployment Center", "AI Debugging Center", "Build Center", "AI Model Center"])
def test_pages_with_loaded_project_and_history(label, tmp_path):
    from app.analyzer.project_scanner import ProjectScanner
    from app.llm.response_parser import GeneratedArtifacts
    from tests.unit.test_history import report
    from app.history.store import HistoryStore
    from app.utils.config import get_settings

    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "requirements.txt").write_text("flask\n")
    (proj / "app.py").write_text("from flask import Flask\napp = Flask(__name__)\n")
    (proj / ".env").write_text("SECRET=1\n")
    df = "FROM python:3.12\nCOPY . .\nRUN pip install flask\nEXPOSE 5000\nCMD [\"python\",\"app.py\"]\n"
    store = HistoryStore(get_settings().history_db)
    store.record(report("a", True))
    store.record(report("b", False, category="network", error="dns failure"))
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["nav"] = label
    at.session_state["project_dir"] = str(proj)
    at.session_state["analysis"] = ProjectScanner(get_settings()).scan(proj) if hasattr(ProjectScanner, "scan") else None
    at.session_state["artifacts"] = GeneratedArtifacts(dockerfile=df, dockerignore="")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    marker = {"Dockerfile Studio": "Quality score", "Security Center": "Static security score",
              "Deployment Center": "Docker Compose", "Execution History": None}.get(label)
    if marker:
        assert any(marker in m.value for m in at.markdown), f"{label} did not render its content"
    if label == "Execution History":
        assert len(at.dataframe) >= 1

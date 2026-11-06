"""The feedback loop must converge on deterministic faults WITHOUT any LLM (no Ollama, no network)."""
import shutil
from pathlib import Path

import pytest
import requests

from app.agent.feedback_loop import FeedbackLoop, LoopStatus
from app.analyzer.project_scanner import ProjectScanner
from app.docker.docker_validator import validate_artifacts
from app.llm.response_parser import GeneratedArtifacts
from tests.fault_injection import RuleBasedCorrector, SimulatedDocker

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "flask_app"
GOOD = (FIXTURE / "Dockerfile").read_text()


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("an HTTP request was attempted: the loop must not need Ollama here")
    monkeypatch.setattr(requests.sessions.Session, "request", boom)


@pytest.fixture
def project(tmp_path):
    dst = tmp_path / "app"
    shutil.copytree(FIXTURE, dst)
    return dst


def run_loop(project, settings, dockerfile, *, corrector=None, docker=None, **settings_kw):
    (project / "Dockerfile").write_text(dockerfile)
    s = settings.model_copy(update={"max_retries": 5, **settings_kw})
    analysis = ProjectScanner(s).scan(project)
    docker = docker or SimulatedDocker()
    corrector = corrector or RuleBasedCorrector(project)
    loop = FeedbackLoop(corrector, docker, s)
    result = loop.run(analysis, project, GeneratedArtifacts(dockerfile=dockerfile, dockerignore=""), "app:ai-generated")
    return result, docker, corrector


FAULTS = {
    "invalid-instruction": (GOOD.replace("RUN pip", "RUNN pip"), "dockerfile_syntax", "RUN pip"),
    "bad-base-tag": (GOOD.replace("python:3.12-slim", "python:3.99-slim"), "base_image_problem", "python:3.12-slim"),
    "copy-typo": (GOOD.replace("COPY requirements.txt .", "COPY requirement.txt ."), "missing_application_files",
                  "COPY requirements.txt ."),
    "bad-package": (GOOD.replace("COPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt",
                                 "RUN pip install --no-cache-dir flaskk"), "missing_python_package", "flask"),
}


def test_simulator_accepts_the_known_good_dockerfile(project, settings):
    result, docker, corrector = run_loop(project, settings, GOOD)
    assert result.status == LoopStatus.SUCCESS and docker.builds == 1 and corrector.calls == 0


@pytest.mark.parametrize("name", list(FAULTS))
def test_single_fault_is_diagnosed_corrected_and_rebuilt(name, project, settings):
    faulty, category, expected_fragment = FAULTS[name]
    result, docker, corrector = run_loop(project, settings, faulty)
    assert result.status == LoopStatus.SUCCESS, result.message
    first = result.history.attempts[0]
    assert not first.success and first.category == category and first.fixable is True and first.diff
    assert result.history.attempts[-1].success and docker.builds == 2 and corrector.calls == 1
    assert expected_fragment in (project / "Dockerfile").read_text()  # the corrected file is what is on disk
    assert validate_artifacts(result.artifacts.dockerfile, "").ok
    assert first.changes and first.diff.startswith("--- attempt 1")


def test_invalid_dockerfile_with_several_faults_converges(project, settings):
    faulty = GOOD.replace("RUN pip", "RUNN pip").replace("python:3.12-slim", "python:3.99-slim") \
        .replace("COPY requirements.txt .", "COPY requirement.txt .")
    result, docker, corrector = run_loop(project, settings, faulty)
    assert result.success and docker.builds == 4
    cats = [a.category for a in result.history.attempts[:-1]]
    assert cats == ["dockerfile_syntax", "base_image_problem", "missing_application_files"]
    assert all(a.signature for a in result.history.attempts[:-1])  # signatures recorded for repeat detection


def test_retry_limit_is_enforced(project, settings):
    faulty = GOOD.replace("RUN pip", "RUNN pip").replace("python:3.12-slim", "python:3.99-slim")
    result, docker, _ = run_loop(project, settings, faulty, max_retries=1)
    assert result.status == LoopStatus.MAX_RETRIES and docker.builds == 2


def test_network_failure_is_not_fixable_and_makes_no_correction(project, settings):
    result, docker, corrector = run_loop(project, settings, GOOD, docker=SimulatedDocker(network_down=True))
    assert result.status == LoopStatus.NON_RECOVERABLE and corrector.calls == 0 and docker.builds == 1
    att = result.history.attempts[0]
    assert att.category == "network_registry_failure" and att.fixable is False


def test_unhelpful_corrector_is_stopped_not_looped_forever(project, settings):
    faulty = GOOD.replace("RUN pip", "RUNN pip")
    result, docker, corrector = run_loop(project, settings, faulty, corrector=RuleBasedCorrector(project, helpful=False))
    assert result.status == LoopStatus.NON_RECOVERABLE and docker.builds == 1 and corrector.calls == 1


def test_repeated_identical_error_stops_the_loop(project, settings):
    class Sloppy(RuleBasedCorrector):  # "fixes" something unrelated, so the same error repeats
        def correct(self, analysis, history, current, build, **kw):
            self.calls += 1
            from app.llm.response_parser import CorrectionResult
            return CorrectionResult(dockerfile=current.dockerfile + f"\n# attempt {self.calls}\n", recoverable=True,
                                    changes=["comment"], root_cause="?")
    result, docker, _ = run_loop(project, settings, GOOD.replace("RUN pip", "RUNN pip"), corrector=Sloppy(project),
                                 repeat_error_limit=3)
    assert result.status == LoopStatus.REPEATED_ERROR and docker.builds == 3
    assert [a.repeated for a in result.history.attempts] == [1, 2, 3]


def test_history_survives_into_the_persistent_store(project, settings, tmp_path):
    from app.agent.orchestrator import Orchestrator
    from app.history.store import HistoryStore
    faulty = FAULTS["copy-typo"][0]
    (project / "Dockerfile").write_text(faulty)
    s = settings.model_copy(update={"max_retries": 3, "history_db": str(tmp_path / "h.db")})
    orch = Orchestrator(s, builder=SimulatedDocker())
    orch.generator = RuleBasedCorrector(project)  # swap the LLM-backed generator for the rule-based one
    analysis = orch.analyze(project)
    report = orch.build_and_fix(project, analysis, GeneratedArtifacts(dockerfile=faulty, dockerignore=""), approve_build=True)
    assert report.success and report.corrections == 1
    store = HistoryStore(tmp_path / "h.db")
    assert store.stats()["correction_success_rate"] == 1.0
    assert store.similar_fixes("missing_application_files") == ["COPY requirement.txt -> requirements.txt"]

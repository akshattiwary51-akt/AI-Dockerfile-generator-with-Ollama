import subprocess

from app.agent.feedback_loop import FeedbackLoop, LoopStatus
from app.analyzer.project_scanner import ProjectScanner
from app.llm.response_parser import GeneratedArtifacts
from app.optimization.optimizer import measure_image
from tests.fault_injection import RuleBasedCorrector


def test_real_build_of_the_fixture_succeeds(docker, project, cleanup_images):
    cleanup_images.append("it-flask:ok")
    res = docker.build(project, "it-flask:ok")
    assert res.success, res.error_summary
    m = measure_image("it-flask:ok")
    assert m.available and m.size_bytes and m.layers  # MEASURED, from the real daemon


def test_real_buildkit_error_is_classified_and_fixed_deterministically(docker, real_settings, project, cleanup_images):
    """Real BuildKit error text -> real classifier -> rule-based (non-LLM) fix -> real rebuild."""
    cleanup_images.append("it-flask:fixed")
    faulty = (project / "Dockerfile").read_text().replace("COPY requirements.txt .", "COPY requirement.txt .")
    (project / "Dockerfile").write_text(faulty)
    analysis = ProjectScanner(real_settings).scan(project)
    result = FeedbackLoop(RuleBasedCorrector(project), docker, real_settings).run(
        analysis, project, GeneratedArtifacts(dockerfile=faulty, dockerignore=""), "it-flask:fixed")
    first = result.history.attempts[0]
    assert first.category == "missing_application_files", first.error  # validates the classifier on real output
    assert result.status == LoopStatus.SUCCESS


def test_compose_config_with_real_docker(real_settings, project):
    from app.compose.generator import generate_compose, validate_compose
    r = generate_compose(ProjectScanner(real_settings).scan(project), project)
    v = validate_compose(r.compose, r.env_example)
    if not v.docker_compose_checked:
        import pytest
        pytest.skip("docker compose plugin not available: " + "; ".join(v.warnings))
    assert v.ok, v.errors


def test_generated_stack_is_never_started(docker, real_settings, project):
    out = subprocess.run(["docker", "ps", "-q"], capture_output=True, text=True).stdout.split()
    from app.compose.generator import generate_compose, write_compose
    write_compose(project, generate_compose(ProjectScanner(real_settings).scan(project), project))
    assert subprocess.run(["docker", "ps", "-q"], capture_output=True, text=True).stdout.split() == out

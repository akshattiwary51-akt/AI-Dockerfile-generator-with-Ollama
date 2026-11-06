import pytest

from app.agent.orchestrator import Orchestrator
from app.docker.docker_validator import validate_artifacts
from app.docker.quality import score_dockerfile


@pytest.fixture
def orch(real_settings):
    o = Orchestrator(real_settings)
    env = o.check_environment()
    if not env.ready_to_generate:
        pytest.skip(f"Ollama/model not ready: {env.ollama_message}")
    return o


def test_real_model_generates_a_valid_dockerfile(orch, project):
    analysis = orch.analyze(project)
    art = orch.generate(analysis)
    assert validate_artifacts(art.dockerfile, art.dockerignore).ok, art.dockerfile
    q = score_dockerfile(art.dockerfile, art.dockerignore, analysis=analysis, project_root=project)
    print(f"\nmodel={orch.client.model} quality={q.overall} retries={orch.generator.format_retries}")
    assert orch.client.stats_log and orch.client.stats_log[-1].latency_seconds > 0


def test_model_assessment_uses_installed_models(orch):
    hw, assessed = orch.assess_models()
    assert any(a.name == orch.client.model for a in assessed)

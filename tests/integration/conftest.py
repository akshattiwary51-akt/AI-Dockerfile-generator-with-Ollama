"""Opt-in integration tests. They need real services and are skipped unless RUN_INTEGRATION=1:

    RUN_INTEGRATION=1 python -m pytest -m integration tests/integration -v

Docker tests need a reachable daemon (and network to pull python:3.12-slim); the Ollama test needs a running
Ollama with the configured model installed. Whatever is missing is reported as a skip, never as a pass.
"""
import os
import shutil
from pathlib import Path

import pytest

from app.docker.docker_builder import DockerBuilder
from app.utils.config import Settings

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "flask_app"


def pytest_collection_modifyitems(items):
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)


@pytest.fixture(autouse=True)
def _opt_in():
    if os.environ.get("RUN_INTEGRATION") != "1":
        pytest.skip("set RUN_INTEGRATION=1 to run integration tests")


@pytest.fixture
def real_settings(tmp_path) -> Settings:
    from app.utils.config import get_settings
    return get_settings().model_copy(update={"max_retries": 3, "history_db": str(tmp_path / "h.db"),
                                              "docker_build_timeout": 600})


@pytest.fixture
def docker(real_settings):
    ok, msg = DockerBuilder(real_settings).check_docker()
    if not ok:
        pytest.skip(f"Docker not available: {msg}")
    return DockerBuilder(real_settings)


@pytest.fixture
def project(tmp_path) -> Path:
    dst = tmp_path / "flaskapp"
    shutil.copytree(FIXTURE, dst)
    return dst


@pytest.fixture
def cleanup_images():
    images: list[str] = []
    yield images
    import subprocess
    for i in images:
        subprocess.run(["docker", "rmi", "-f", i], capture_output=True, timeout=60)

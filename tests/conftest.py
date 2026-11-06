"""Shared fixtures: fixture projects, a scripted fake Ollama, a fake builder and a fake Popen."""
from __future__ import annotations

import copy
import io
import json
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.docker.docker_builder import BuildResult  # noqa: E402
from app.utils.config import Settings  # noqa: E402

GOOD_DOCKERFILE = """\
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd --create-home appuser
USER appuser
EXPOSE 5000
HEALTHCHECK CMD python -c "import urllib.request as u; u.urlopen('http://localhost:5000')"
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
"""
GOOD_DOCKERIGNORE = ".git\n.env\n.env.*\n__pycache__/\n*.pyc\n.venv/\nDockerfile\n"

FLASK_FILES = {
    "requirements.txt": "flask==3.0.0\ngunicorn==21.2.0\n",
    "app.py": ('import os\nfrom flask import Flask\n\napp = Flask(__name__)\n'
               'DB = os.environ.get("DATABASE_URL")\nKEY = os.getenv("SECRET_KEY")\n\n'
               'if __name__ == "__main__":\n    app.run(host="0.0.0.0", port=5050)\n'),
}
FASTAPI_FILES = {
    "requirements.txt": "fastapi>=0.110\nuvicorn[standard]>=0.29\n",
    "main.py": "from fastapi import FastAPI\n\napi = FastAPI()\n\n@api.get('/')\ndef root():\n    return {}\n",
    "pyproject.toml": '[project]\nname = "svc"\nrequires-python = ">=3.10"\n',
}
VITE_FILES = {
    "package.json": json.dumps({"name": "web", "scripts": {"build": "vite build", "dev": "vite"},
                                "dependencies": {"react": "^18.2.0", "react-dom": "^18.2.0"},
                                "devDependencies": {"vite": "^5.0.0", "@vitejs/plugin-react": "^4.2.0"}}),
    "package-lock.json": "{}",
    "src/main.jsx": "console.log(import.meta.env.VITE_API_URL)\n",
    "index.html": "<html></html>",
}
EXPRESS_FILES = {
    "package.json": json.dumps({"name": "api", "main": "server.js", "scripts": {"start": "node server.js"},
                                "dependencies": {"express": "^4.18.2"}, "engines": {"node": ">=18"}}),
    "yarn.lock": "# yarn lockfile v1\n",
    "server.js": ("const express = require('express');\nconst app = express();\n"
                  "app.listen(process.env.PORT || 4000);\nconsole.log(process.env.API_TOKEN);\n"),
}


def llm_json(dockerfile: str = GOOD_DOCKERFILE, dockerignore: str = GOOD_DOCKERIGNORE, **extra: Any) -> str:
    data = {"dockerfile": dockerfile, "dockerignore": dockerignore, "language": "Python", "framework": "Flask",
            "base_image": "python:3.12-slim", "exposed_port": 5000, "explanation": "slim image, non-root user",
            "limitations": ["assumes app:app"]}
    data.update(extra)
    return json.dumps(data)


def correction_json(dockerfile: str, root_cause: str = "missing build dependency", changes: Optional[list] = None,
                    recoverable: bool = True, **extra: Any) -> str:
    base = json.loads(llm_json(dockerfile))
    base.update({"root_cause": root_cause, "changes": changes if changes is not None else ["added gcc"],
                 "recoverable": recoverable})
    base.update(extra)
    return json.dumps(base)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(history_db=str(tmp_path / "history.db"), ollama_host="http://ollama.test:11434", ollama_model="test-model:7b", temperature=0.1,
                    ollama_timeout=30, ollama_timeout_retries=0, num_ctx=4096, max_retries=3, docker_build_timeout=30, max_log_bytes=20_000,
                    max_context_chars=6000, max_file_bytes=100_000)


@pytest.fixture
def make_project(tmp_path: Path) -> Callable[..., Path]:
    def _make(files: dict[str, str], name: str = "proj") -> Path:
        root = tmp_path / name
        root.mkdir(parents=True, exist_ok=True)
        for rel, content in files.items():
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        return root
    return _make


class ScriptedClient:
    """Fake Ollama client returning scripted replies (or raising scripted exceptions)."""

    def __init__(self, replies: list[Any], model: str = "test-model:7b") -> None:
        self.replies = list(replies)
        self.calls: list[list[dict[str, str]]] = []
        self.schemas: list[Any] = []
        self.model = model
        self.host = "http://ollama.test:11434"

    def chat(self, messages: list[dict[str, str]], *, schema: Any = None, model: Optional[str] = None) -> str:
        self.calls.append(copy.deepcopy(messages))
        self.schemas.append(schema)
        if not self.replies:
            raise AssertionError("ScriptedClient ran out of replies")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def ensure_ready(self, model: Optional[str] = None) -> None:
        return None

    def list_models(self) -> list[str]:
        return [self.model]

    def has_model(self, name: Optional[str] = None) -> bool:
        return True


def failed_build(error: str = "ERROR: process did not complete successfully", stage: str = "[3/5] RUN pip install",
                 **extra: Any) -> BuildResult:
    data = dict(success=False, image="proj:ai-generated", exit_code=1, failed_stage=stage, error_summary=error,
                stderr=error, duration_seconds=1.5)
    data.update(extra)
    return BuildResult(**data)


def ok_build() -> BuildResult:
    return BuildResult(success=True, image="proj:ai-generated", exit_code=0, duration_seconds=2.0)


class FakeBuilder:
    """Returns scripted BuildResults; records calls. Mirrors DockerBuilder's interface."""

    def __init__(self, results: list[BuildResult]) -> None:
        self.results = list(results)
        self.calls = 0
        self.dockerfiles_built: list[str] = []
        self._run = lambda *a, **k: type("P", (), {"returncode": 1, "stdout": "", "stderr": ""})()

    def check_docker(self) -> tuple[bool, str]:
        return True, "Docker Engine fake"

    def build(self, project_dir: Any, image: str, *, on_output: Any = None, cancel_event: Any = None,
              dockerfile: str = "Dockerfile") -> BuildResult:
        self.calls += 1
        path = Path(project_dir) / dockerfile
        self.dockerfiles_built.append(path.read_text() if path.exists() else "")
        if on_output:
            on_output(f"fake build output #{self.calls}")
        if not self.results:
            raise AssertionError("FakeBuilder ran out of results")
        return self.results.pop(0)


class FakeProc:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: Optional[int] = 0, hang: bool = False) -> None:
        self.stdout = io.StringIO(stdout)
        self.stderr = io.StringIO(stderr)
        self._rc = returncode
        self.hang = hang
        self.returncode: Optional[int] = None
        self.terminated = False

    def poll(self) -> Optional[int]:
        if self.hang and not self.terminated:
            return None
        self.returncode = -15 if self.terminated else self._rc
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.terminated = True

    def wait(self, timeout: Optional[float] = None) -> Optional[int]:
        return self.poll()


@pytest.fixture
def popen_factory() -> Callable[..., Any]:
    """Build a fake ``subprocess.Popen`` that records its call arguments."""

    def factory(proc: Optional[FakeProc] = None, raises: Optional[Exception] = None) -> Any:
        calls: list[dict[str, Any]] = []

        def fake(cmd: list[str], **kwargs: Any) -> FakeProc:
            calls.append({"cmd": cmd, "kwargs": kwargs})
            if raises:
                raise raises
            return proc or FakeProc()

        fake.calls = calls  # type: ignore[attr-defined]
        return fake

    return factory

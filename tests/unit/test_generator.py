"""Generation, invalid LLM output, file protection and the Docker builder (subprocess mocked)."""
from __future__ import annotations

import threading
from pathlib import Path

import pytest

from app.agent.orchestrator import ApprovalRequired, Orchestrator
from app.analyzer.project_scanner import ProjectScanner
from app.docker.docker_builder import (DockerBuilder, classify_failure, extract_build_error, sanitize_image_name)
from app.docker.dockerfile_generator import (DockerfileGenerator, FileProtectionError, GenerationError, write_artifacts)
from app.llm.ollama_client import ModelNotFoundError
from app.llm.prompt_builder import build_generation_messages
from app.llm.response_parser import ResponseParseError, parse_generation
from tests.conftest import (FLASK_FILES, GOOD_DOCKERFILE, GOOD_DOCKERIGNORE, FakeBuilder, FakeProc, ScriptedClient,
                            llm_json, ok_build)


def analysis_for(make_project, settings, files=FLASK_FILES):
    root = make_project(files)
    return root, ProjectScanner(settings).scan(root)


# ----------------------------------------------------------------------------- generation
def test_generate_valid_output(make_project, settings):
    _, analysis = analysis_for(make_project, settings)
    client = ScriptedClient([llm_json()])
    art = DockerfileGenerator(client, settings).generate(analysis)
    assert art.dockerfile.startswith("FROM python:3.12-slim") and art.exposed_port == 5000
    assert art.base_image == "python:3.12-slim" and art.limitations == ["assumes app:app"]
    assert client.schemas[0] is not None  # structured output requested


def test_prompt_contains_facts_but_no_secret_values(make_project, settings):
    files = dict(FLASK_FILES)
    files[".env"] = "SECRET_KEY=topsecretvalue987\n"
    files["config.py"] = 'PASSWORD = "hunter2hunter2"\n'
    root, analysis = analysis_for(make_project, settings, files)
    text = "\n".join(m["content"] for m in build_generation_messages(analysis, settings))
    assert "Flask" in text and "app:app" in text and "SECRET_KEY" in text
    assert "topsecretvalue987" not in text and "hunter2hunter2" not in text
    assert len(build_generation_messages(analysis, settings)[1]["content"]) < settings.max_context_chars + 1500


def test_invalid_llm_response_then_repair(make_project, settings):
    _, analysis = analysis_for(make_project, settings)
    client = ScriptedClient(["I cannot do that, sorry", "", llm_json()])
    art = DockerfileGenerator(client, settings).generate(analysis)
    assert art.dockerfile and len(client.calls) == 3
    assert "could not be used" in client.calls[1][-1]["content"]  # the model is told what was wrong


def test_invalid_llm_response_gives_up(make_project, settings):
    _, analysis = analysis_for(make_project, settings)
    client = ScriptedClient(["nope", "still nope", '{"language": "py"}'])
    with pytest.raises(GenerationError) as exc:
        DockerfileGenerator(client, settings).generate(analysis)
    assert "3 attempts" in str(exc.value)


def test_malformed_dockerfile_is_sent_back_for_repair(make_project, settings):
    _, analysis = analysis_for(make_project, settings)
    bad = "FROM python:3.12-slim\nRUNN pip install flask\nCMD [gunicorn]\n"
    client = ScriptedClient([llm_json(bad), llm_json(GOOD_DOCKERFILE)])
    art = DockerfileGenerator(client, settings).generate(analysis)
    assert "RUNN" not in art.dockerfile
    feedback = client.calls[1][-1]["content"]
    assert "unknown-instruction" in feedback and "malformed-exec-form" in feedback


def test_malformed_dockerfile_never_repaired_raises(make_project, settings):
    _, analysis = analysis_for(make_project, settings)
    client = ScriptedClient([llm_json("not a dockerfile at all")] * 3)
    with pytest.raises(GenerationError):
        DockerfileGenerator(client, settings).generate(analysis)


def test_parser_handles_fences_prose_and_salvage():
    wrapped = "Sure!\n```json\n" + llm_json() + "\n```\nHope it helps"
    assert parse_generation(wrapped).framework == "Flask"
    salvaged = parse_generation("Here:\n```dockerfile\nFROM alpine:3.20\nCMD [\"sh\"]\n```\n")
    assert salvaged.dockerfile.startswith("FROM alpine") and "did not return structured JSON" in salvaged.limitations[0]
    escaped = parse_generation(llm_json("FROM alpine:3.20\\nCMD [\\\"sh\\\"]").replace("\\\\n", "\\n"))
    assert escaped.dockerfile
    for bad in ("", "   ", "plain text", "[1, 2]"):
        with pytest.raises(ResponseParseError):
            parse_generation(bad)


def test_model_not_installed_blocks_generation(make_project, settings):
    _, analysis = analysis_for(make_project, settings)

    class Missing(ScriptedClient):
        def ensure_ready(self, model=None):
            raise ModelNotFoundError("Model 'x' is not installed; run: ollama pull x")

    orch = Orchestrator(settings, client=Missing([]), builder=FakeBuilder([]))
    with pytest.raises(ModelNotFoundError):
        orch.generate(analysis)


# ----------------------------------------------------------------------------- file protection
def test_existing_dockerfile_is_protected(make_project, settings):
    root = make_project({**FLASK_FILES, "Dockerfile": "FROM scratch\n# mine\n"})
    with pytest.raises(FileProtectionError) as exc:
        write_artifacts(root, GOOD_DOCKERFILE, GOOD_DOCKERIGNORE)
    assert "Dockerfile" in exc.value.files
    assert (root / "Dockerfile").read_text() == "FROM scratch\n# mine\n"  # untouched
    assert not (root / ".dockerignore").exists()  # nothing partially written


def test_overwrite_with_approval_makes_backup(make_project):
    root = make_project({"Dockerfile": "FROM scratch\n# mine\n"})
    written = write_artifacts(root, GOOD_DOCKERFILE, GOOD_DOCKERIGNORE, overwrite=True)
    assert (root / "Dockerfile").read_text() == GOOD_DOCKERFILE
    assert (root / "Dockerfile.bak").read_text() == "FROM scratch\n# mine\n"
    assert {p.name for p in written} == {"Dockerfile", ".dockerignore"}
    assert not list(root.glob(".*.tmp"))


def test_write_refuses_symlink_and_missing_dir(make_project, tmp_path):
    root = make_project({"x.txt": "1"})
    outside = tmp_path / "outside.txt"
    outside.write_text("keep")
    (root / "Dockerfile").symlink_to(outside)
    with pytest.raises(FileProtectionError):
        write_artifacts(root, GOOD_DOCKERFILE, "")
    with pytest.raises(PermissionError):
        write_artifacts(root, GOOD_DOCKERFILE, "", overwrite=True)
    assert outside.read_text() == "keep"
    with pytest.raises(FileNotFoundError):
        write_artifacts(tmp_path / "missing", GOOD_DOCKERFILE, "")


def test_orchestrator_save_validates_and_protects(make_project, settings):
    root = make_project(FLASK_FILES)
    orch = Orchestrator(settings, client=ScriptedClient([]), builder=FakeBuilder([]))
    art = parse_generation(llm_json())
    bad = art.model_copy(update={"dockerfile": "RUNN x\n"})
    with pytest.raises(ValueError):
        orch.save(root, bad)
    orch.save(root, art)
    assert orch.existing_files(root) == ["Dockerfile", ".dockerignore"]
    with pytest.raises(FileProtectionError):
        orch.save(root, art)
    orch.save(root, art, approve_overwrite=True)
    assert "---" in orch.diff_with_existing(root, art.model_copy(update={"dockerfile": "FROM alpine:3.20\nCMD [\"sh\"]\n"}))


def test_build_requires_explicit_approval(make_project, settings):
    root, analysis = analysis_for(make_project, settings)
    builder = FakeBuilder([ok_build()])
    orch = Orchestrator(settings, client=ScriptedClient([llm_json()]), builder=builder)
    with pytest.raises(ApprovalRequired) as exc:
        orch.run_full(root)
    assert "executes the instructions" in str(exc.value)
    assert builder.calls == 0 and not (root / "Dockerfile").exists()
    with pytest.raises(ApprovalRequired):
        orch.build_and_fix(root, analysis, parse_generation(llm_json()), approve_build=False)
    assert builder.calls == 0


# ----------------------------------------------------------------------------- docker builder (mocked subprocess)
def test_docker_build_success(make_project, settings, popen_factory):
    root = make_project({"Dockerfile": GOOD_DOCKERFILE})
    popen = popen_factory(FakeProc(stdout="#1 [1/3] FROM python\n#5 DONE\n", stderr="", returncode=0))
    lines: list[str] = []
    res = DockerBuilder(settings, popen=popen).build(root, "proj:ai-generated", on_output=lines.append)
    assert res.success and res.exit_code == 0 and res.failed_stage is None and res.non_recoverable_reason is None
    call = popen.calls[0]
    assert call["cmd"] == ["docker", "build", "--progress=plain", "-t", "proj:ai-generated", "."]
    assert call["kwargs"]["cwd"] == str(root.resolve())
    assert lines and "#1 [1/3] FROM python" in lines


def test_docker_build_never_privileged_or_mounting(make_project, settings, popen_factory):
    root = make_project({"Dockerfile": GOOD_DOCKERFILE})
    popen = popen_factory(FakeProc())
    DockerBuilder(settings, popen=popen).build(root, "x:y")
    cmd = " ".join(popen.calls[0]["cmd"])
    for forbidden in ("--privileged", "-v ", "--volume", "docker.sock", "--network=host", "--secret", "run "):
        assert forbidden not in cmd
    assert popen.calls[0]["kwargs"]["stdin"] is not None


BUILDKIT_FAIL = """#6 [3/5] RUN pip install -r requirements.txt
#6 1.1 ERROR: Could not find a version that satisfies the requirement nopkg==9.9
#6 ERROR: process "/bin/sh -c pip install -r requirements.txt" did not complete successfully: exit code: 1
------
 > [3/5] RUN pip install -r requirements.txt:
------
Dockerfile:5
ERROR: failed to solve: process did not complete successfully: exit code: 1"""


def test_docker_build_failure_captures_stage_and_error(make_project, settings, popen_factory):
    root = make_project({"Dockerfile": GOOD_DOCKERFILE})
    popen = popen_factory(FakeProc(stdout="", stderr=BUILDKIT_FAIL, returncode=1))
    res = DockerBuilder(settings, popen=popen).build(root, "x:y")
    assert not res.success and res.exit_code == 1
    assert res.failed_stage == "[3/5] RUN pip install -r requirements.txt"
    assert "Could not find a version" in res.error_summary and res.non_recoverable_reason is None


def test_legacy_builder_output_parsed():
    out = "Step 4/8 : RUN npm ci\n ---> Running in abc\nnpm ERR! boom\nThe command '/bin/sh -c npm ci' returned a non-zero code: 1"
    stage, summary = extract_build_error(out)
    assert stage == "Step 4/8 : RUN npm ci" and "npm ERR! boom" in summary


def test_secrets_redacted_in_logs(make_project, settings, popen_factory):
    root = make_project({"Dockerfile": GOOD_DOCKERFILE})
    leaked = "#4 RUN echo ghp_abcdefghijklmnopqrstuvwxyz123456 DB_PASSWORD=hunter2 postgres://u:pw123@db/x\n"
    seen: list[str] = []
    res = DockerBuilder(settings, popen=popen_factory(FakeProc(stderr=leaked, returncode=1))).build(
        root, "x:y", on_output=seen.append)
    blob = res.combined_output + "\n".join(seen) + res.error_summary
    assert "ghp_abcdefghij" not in blob and "hunter2" not in blob and "pw123" not in blob


def test_log_size_is_bounded(make_project, settings, popen_factory):
    root = make_project({"Dockerfile": GOOD_DOCKERFILE})
    noisy = "".join(f"line {i} " + "x" * 80 + "\n" for i in range(5000)) + "FINAL ERROR LINE\n"
    res = DockerBuilder(settings, popen=popen_factory(FakeProc(stdout=noisy, returncode=1))).build(root, "x:y")
    assert len(res.stdout.encode()) <= settings.max_log_bytes + 200
    assert "FINAL ERROR LINE" in res.stdout and "omitted" in res.stdout  # keeps the tail, drops the middle


def test_timeout_terminates_build(make_project, settings, popen_factory):
    root = make_project({"Dockerfile": GOOD_DOCKERFILE})
    proc = FakeProc(hang=True)
    short = settings.model_copy(update={"docker_build_timeout": 1})
    res = DockerBuilder(short, popen=popen_factory(proc)).build(root, "x:y")
    assert res.timed_out and not res.success and proc.terminated and "timeout" in res.error_summary


def test_cancel_terminates_build(make_project, settings, popen_factory):
    root = make_project({"Dockerfile": GOOD_DOCKERFILE})
    proc, cancel = FakeProc(hang=True), threading.Event()
    cancel.set()
    res = DockerBuilder(settings, popen=popen_factory(proc)).build(root, "x:y", cancel_event=cancel)
    assert res.cancelled and proc.terminated and not res.success


def test_docker_missing_and_no_dockerfile(make_project, settings, popen_factory):
    root = make_project({"Dockerfile": GOOD_DOCKERFILE})
    res = DockerBuilder(settings, popen=popen_factory(raises=FileNotFoundError())).build(root, "x:y")
    assert not res.success and "not found" in res.non_recoverable_reason
    empty = make_project({"a.txt": "1"}, "empty")
    res = DockerBuilder(settings, popen=popen_factory()).build(empty, "x:y")
    assert not res.success and res.non_recoverable_reason == "Dockerfile missing"


def test_check_docker_states(settings):
    class P:
        def __init__(self, rc, out="", err=""):
            self.returncode, self.stdout, self.stderr = rc, out, err

    ok, msg = DockerBuilder(settings, run=lambda *a, **k: P(0, "27.1\n")).check_docker()
    assert ok and "27.1" in msg
    ok, msg = DockerBuilder(settings, run=lambda *a, **k: P(1, "", "Cannot connect to the Docker daemon")).check_docker()
    assert not ok and "daemon" in msg

    def boom(*a, **k):
        raise FileNotFoundError()

    ok, msg = DockerBuilder(settings, run=boom).check_docker()
    assert not ok and "docker" in msg.lower() and "install" in msg.lower()


@pytest.mark.parametrize("text,expected", [
    ("Cannot connect to the Docker daemon at unix:///var/run/docker.sock", "daemon"),
    ("write /var/lib/docker/x: no space left on device", "disk"),
    ("toomanyrequests: You have reached your pull rate limit", "rate-limited"),
    ("dial tcp: lookup registry-1.docker.io: no such host", "network"),
    ("npm ERR! code E404", None),
])
def test_non_recoverable_classification(text, expected):
    reason = classify_failure(text)
    assert (expected in reason) if expected else reason is None


def test_sanitize_image_name():
    assert sanitize_image_name("My App_v2!") == "my-app_v2:ai-generated"
    assert sanitize_image_name("***") == "app:ai-generated"

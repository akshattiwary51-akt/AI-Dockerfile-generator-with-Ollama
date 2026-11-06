"""The build -> diagnose -> targeted fix loop, end to end with fake Ollama and fake Docker."""
from __future__ import annotations

import threading
from pathlib import Path

import pytest

from app.agent.feedback_loop import LoopStatus
from app.agent.orchestrator import Orchestrator
from app.docker.docker_builder import BuildResult
from app.llm.ollama_client import OllamaUnavailableError
from app.llm.response_parser import parse_generation
from tests.conftest import (FLASK_FILES, GOOD_DOCKERFILE, FakeBuilder, ScriptedClient, correction_json, failed_build,
                            llm_json, ok_build)

FIXED = GOOD_DOCKERFILE.replace("RUN pip install --no-cache-dir -r requirements.txt",
                                "RUN apt-get update && apt-get install -y --no-install-recommends gcc "
                                "&& rm -rf /var/lib/apt/lists/* \\\n    && pip install --no-cache-dir -r requirements.txt")
FIXED2 = FIXED.replace("python:3.12-slim", "python:3.11-slim")


def setup(make_project, settings, replies, builds, name="proj", **overrides):
    root = make_project(FLASK_FILES, name)
    s = settings.model_copy(update=overrides)
    client, builder = ScriptedClient(replies), FakeBuilder(builds)
    return root, Orchestrator(s, client=client, builder=builder), client, builder


def test_build_success_first_try(make_project, settings):
    root, orch, client, builder = setup(make_project, settings, [llm_json()], [ok_build()])
    report = orch.run_full(root, approve_build=True)
    assert report.success and report.attempts == 1 and report.corrections == 0 and report.status == "success"
    assert builder.calls == 1 and len(client.calls) == 1
    assert (root / "Dockerfile").read_text() == GOOD_DOCKERFILE and (root / ".dockerignore").exists()
    assert report.quality_score == 100 and any("vulnerability scan" in s for s in report.suggestions)


def test_build_failure_then_successful_correction(make_project, settings):
    err = "ERROR: Failed building wheel for mypkg\ngcc: command not found"
    replies = [llm_json(), correction_json(FIXED, "pip needs gcc to compile mypkg; slim image lacks it",
                                           ["Install gcc before pip install"])]
    events = []
    root, orch, client, builder = setup(make_project, settings, replies, [failed_build(err), ok_build()])
    report = orch.run_full(root, approve_build=True, on_event=events.append)

    assert report.success and report.attempts == 2 and report.corrections == 1
    # the corrected Dockerfile is what was written and rebuilt
    assert builder.dockerfiles_built == [GOOD_DOCKERFILE, FIXED]
    assert (root / "Dockerfile").read_text() == FIXED
    # history keeps the failure, diagnosis and applied change
    first = report.history.attempts[0]
    assert not first.success and first.dockerfile == GOOD_DOCKERFILE and "gcc" in first.error
    assert "needs gcc" in first.diagnosis and first.changes == ["Install gcc before pip install"]
    assert report.history.attempts[1].success and report.history.attempts[1].dockerfile == FIXED
    assert [e.kind for e in events if e.kind != "stopped"] == [
        "build_start", "build_done", "diagnosing", "corrected", "build_start", "build_done"]
    # the correction prompt carried the log, failing Dockerfile and constraints
    prompt = client.calls[1][-1]["content"]
    assert "gcc: command not found" in prompt and "CURRENT DOCKERFILE" in prompt and "BUILD CONSTRAINTS" in prompt
    assert "Failed stage/instruction: [3/5] RUN pip install" in prompt
    # conversation context is retained across the correction (original generation is still there)
    assert client.calls[1][1]["role"] == "user" and "PROJECT FACTS" in client.calls[1][1]["content"]
    assert any(m["role"] == "assistant" for m in client.calls[1])


def test_max_retry_limit_reached(make_project, settings):
    fixes = [FIXED, FIXED2, FIXED.replace("gcc", "build-essential")]
    replies = [llm_json()] + [correction_json(f, f"cause {i}", [f"fix {i}"]) for i, f in enumerate(fixes)]
    builds = [failed_build(f"error {i}") for i in range(4)]
    root, orch, client, builder = setup(make_project, settings, replies, builds, max_retries=3)
    report = orch.run_full(root, approve_build=True)
    assert not report.success and report.status == LoopStatus.MAX_RETRIES.value
    assert builder.calls == 4 and report.attempts == 4 and report.corrections == 3  # 1 build + 3 retries, no more
    assert "MAX_RETRIES=3" in report.message
    assert len(client.calls) == 4  # generation + exactly 3 corrections


def test_zero_retries_means_single_build(make_project, settings):
    root, orch, client, builder = setup(make_project, settings, [llm_json()], [failed_build()], max_retries=0)
    report = orch.run_full(root, approve_build=True)
    assert builder.calls == 1 and report.status == "max_retries_reached" and len(client.calls) == 1


def test_previous_attempts_are_sent_to_avoid_repeating_fixes(make_project, settings):
    replies = [llm_json(), correction_json(FIXED, "cause A", ["added gcc"]), correction_json(FIXED2, "cause B", ["py3.11"])]
    root, orch, client, _ = setup(make_project, settings, replies, [failed_build("e1"), failed_build("e2"), ok_build()])
    report = orch.run_full(root, approve_build=True)
    assert report.success and report.corrections == 2
    second_prompt = client.calls[2][-1]["content"]
    assert "PREVIOUS ATTEMPTS" in second_prompt and "cause A" in second_prompt and "added gcc" in second_prompt


def test_unrecoverable_environment_error_stops_without_llm(make_project, settings):
    bad = failed_build("Cannot connect to the Docker daemon at unix:///var/run/docker.sock",
                       non_recoverable_reason="The Docker daemon is not reachable.")
    root, orch, client, builder = setup(make_project, settings, [llm_json()], [bad])
    report = orch.run_full(root, approve_build=True)
    assert report.status == LoopStatus.NON_RECOVERABLE.value and builder.calls == 1 and len(client.calls) == 1
    assert "daemon" in report.message


def test_llm_declares_failure_unrecoverable(make_project, settings):
    reply = correction_json("", "the source tree lacks requirements.txt", [], recoverable=False)
    root, orch, _, builder = setup(make_project, settings, [llm_json(), reply], [failed_build()])
    report = orch.run_full(root, approve_build=True)
    assert report.status == "non_recoverable_error" and builder.calls == 1
    assert (root / "Dockerfile").read_text() == GOOD_DOCKERFILE  # not modified


def test_unchanged_dockerfile_triggers_second_ask_then_stops(make_project, settings):
    same = correction_json(GOOD_DOCKERFILE, "unsure", ["none"])
    root, orch, client, builder = setup(make_project, settings, [llm_json(), same, same], [failed_build()])
    report = orch.run_full(root, approve_build=True)
    assert report.status == "non_recoverable_error" and builder.calls == 1 and len(client.calls) == 3
    assert "identical" in client.calls[2][-1]["content"]


def test_full_rewrite_is_pushed_back_to_targeted_fix(make_project, settings):
    rewrite = "FROM node:22-alpine\nWORKDIR /srv\nCOPY . .\nEXPOSE 3000\nCMD [\"node\", \"x.js\"]\n"
    replies = [llm_json(), correction_json(rewrite, "x", ["rewrote"]), correction_json(FIXED, "needs gcc", ["gcc"])]
    root, orch, client, builder = setup(make_project, settings, replies, [failed_build(), ok_build()])
    report = orch.run_full(root, approve_build=True)
    assert report.success and builder.dockerfiles_built[1] == FIXED
    assert "rewrote most of the Dockerfile" in client.calls[2][-1]["content"]


def test_invalid_correction_is_repaired(make_project, settings):
    broken = correction_json("FROM python:3.12-slim\nRUNN oops\n")
    replies = [llm_json(), broken, correction_json(FIXED, "gcc", ["gcc"])]
    root, orch, client, builder = setup(make_project, settings, replies, [failed_build(), ok_build()])
    report = orch.run_full(root, approve_build=True)
    assert report.success and "unknown-instruction" in client.calls[2][-1]["content"]


def test_llm_failure_during_correction_ends_cleanly(make_project, settings):
    replies = [llm_json(), OllamaUnavailableError("ollama went away")]
    root, orch, _, _ = setup(make_project, settings, replies, [failed_build()])
    report = orch.run_full(root, approve_build=True)
    assert report.status == "non_recoverable_error" and "could not produce a correction" in report.history.attempts[0].diagnosis


def test_user_cancellation(make_project, settings):
    root, orch, _, builder = setup(make_project, settings, [llm_json()], [BuildResult(
        success=False, image="x:y", cancelled=True, error_summary="Build cancelled by the user.")])
    report = orch.run_full(root, approve_build=True, cancel_event=threading.Event())
    assert report.status == "cancelled" and builder.calls == 1
    root2, orch2, _, builder2 = setup(make_project, settings, [llm_json()], [], name="proj2")
    cancel = threading.Event()
    cancel.set()
    analysis = orch2.analyze(root2)
    art = parse_generation(llm_json())
    orch2.save(root2, art)
    rep = orch2.build_and_fix(root2, analysis, art, approve_build=True, cancel_event=cancel)
    assert rep.status == "cancelled" and builder2.calls == 0


def test_reports_export_without_secrets(make_project, settings, tmp_path):
    secret_err = "ERROR: login failed token=ghp_abcdefghijklmnopqrstuvwxyz123456"
    replies = [llm_json(), correction_json(FIXED, "auth", ["x"])]
    root, orch, _, _ = setup(make_project, settings, replies, [failed_build(secret_err), ok_build()])
    report = orch.run_full(root, approve_build=True)
    paths = orch.export_report(report, tmp_path / "out")
    assert {p.suffix for p in paths} == {".json", ".md"}
    for p in paths:
        text = p.read_text()
        assert "ghp_abcdefghij" not in text and p.is_relative_to(tmp_path)
    assert "Attempt 1" in paths[1].read_text() and "SUCCESS" in paths[1].read_text()


def test_generated_files_stay_inside_project(make_project, settings, tmp_path):
    root, orch, _, _ = setup(make_project, settings, [llm_json()], [ok_build()])
    before = {p for p in tmp_path.rglob("*")}
    orch.run_full(root, approve_build=True)
    new = {p for p in tmp_path.rglob("*")} - before
    assert all(p.is_relative_to(root) for p in new)

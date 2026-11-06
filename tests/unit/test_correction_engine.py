"""Phase 5: error classification, repeated-error detection, regression guard, history-informed prompts."""
from __future__ import annotations

import pytest

from app.agent.error_classifier import ErrorCategory as C, classify_build_error, failing_instruction
from app.agent.feedback_loop import LoopStatus
from app.agent.orchestrator import Orchestrator
from app.agent.patch_guard import find_regressions
from app.llm.response_parser import GeneratedArtifacts, parse_correction
from tests.conftest import (FLASK_FILES, GOOD_DOCKERFILE, GOOD_DOCKERIGNORE, FakeBuilder, ScriptedClient,
                            correction_json, failed_build, llm_json, ok_build)

CASES = [
    ("ERROR: failed to solve: python:3.12-slimm: failed to resolve source metadata for docker.io/library/python:3.12-slimm: not found", C.BASE_IMAGE, True),
    ('failed to compute cache key: failed to calculate checksum of ref a::b: "/requirement.txt": not found', C.MISSING_FILES, True),
    ("ERROR: Could not open requirements file: [Errno 2] No such file or directory: 'requirements.txt'", C.MISSING_FILES, True),
    ("Error: pg_config executable not found.", C.SYSTEM_DEP, True),
    ("/bin/sh: 1: gcc: not found", C.SYSTEM_DEP, True),
    ("ERROR: Failed building wheel for mypkg", C.SYSTEM_DEP, True),
    ("npm error code E404", C.NODE_DEP, True),
    ("npm error `npm ci` can only install packages when your package.json and package-lock.json are in sync.", C.PACKAGE_MANAGER, True),
    ("error: externally-managed-environment", C.PACKAGE_MANAGER, True),
    ("ERROR: No matching distribution found for nopkg-xyz==9.9", C.PYTHON_PKG, True),
    ("EACCES: permission denied, mkdir '/app/node_modules'", C.PERMISSION, True),
    ("ERROR: failed to solve: dockerfile parse error on line 3: unknown instruction: RUNN", C.SYNTAX, True),
    ("dial tcp: lookup registry-1.docker.io: no such host", C.NETWORK, False),
    ("toomanyrequests: You have reached your pull rate limit", C.NETWORK, False),
    ("npm error code ENOTFOUND", C.NETWORK, False),
    ("Cannot connect to the Docker daemon at unix:///var/run/docker.sock", C.DOCKER_DAEMON, False),
    ("write /var/lib/docker/x: no space left on device", C.RESOURCE, False),
    ("process did not complete successfully: exit code: 137", C.RESOURCE, False),
    ("ERROR: failed to solve: failed to read dockerfile: open Dockerfile: no such file or directory", C.BUILD_CONTEXT, False),
    ("ERROR: something strange", C.UNKNOWN, True),
]


@pytest.mark.parametrize("log,category,fixable", CASES)
def test_error_classification(log, category, fixable):
    d = classify_build_error(log, "[3/6] RUN pip install -r requirements.txt")
    assert d.category == category and d.fixable is fixable
    assert d.evidence and d.hint and d.failing_instruction == "RUN pip install -r requirements.txt"


def test_classifier_prefers_root_cause_over_symptoms():
    log = "#5 ERROR: Failed building wheel for psycopg2\n#5 Error: pg_config executable not found.\nModuleNotFoundError: x"
    assert classify_build_error(log).category == C.SYSTEM_DEP
    net = "pip._vendor.urllib3: Failed to establish a new connection: [Errno -3]\nERROR: No matching distribution found for flask"
    assert classify_build_error(net).category == C.NETWORK  # the network failure is the cause, not the missing dist


def test_signature_ignores_noise_but_not_substance():
    a = classify_build_error("#7 12.4s ERROR: copy failed abcdef0123456789abcd")
    b = classify_build_error("#9 3.1s ERROR: copy failed 0123456789abcdef0123")
    c = classify_build_error("#7 ERROR: copy failed: other.txt: not found")
    assert a.signature == b.signature and a.signature != c.signature
    assert failing_instruction("[builder 2/5] COPY x .") == "COPY x ." and failing_instruction(None) is None


FIXED = GOOD_DOCKERFILE.replace("RUN pip install", "RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/* && pip install")


def setup(make_project, settings, replies, builds, name="proj", **overrides):
    root = make_project(FLASK_FILES, name)
    s = settings.model_copy(update=overrides)
    client, builder = ScriptedClient(replies), FakeBuilder(builds)
    return root, Orchestrator(s, client=client, builder=builder), client, builder


def test_unfixable_error_stops_without_calling_the_llm(make_project, settings):
    root, orch, client, builder = setup(make_project, settings, [llm_json()],
                                        [failed_build("Temporary failure in name resolution")])
    report = orch.run_full(root, approve_build=True)
    a = report.history.attempts[0]
    assert report.status == LoopStatus.NON_RECOVERABLE.value and builder.calls == 1 and len(client.calls) == 1
    assert a.category == "network_registry_failure" and a.fixable is False and "network" in a.diagnosis.lower()


def test_attempt_records_classification_and_llm_explanation(make_project, settings):
    reply = correction_json(FIXED, "slim image has no gcc", ["install gcc"], explanation="wheel needs a compiler",
                            expected_effect="pip can build mypkg", remaining_risks=["image gets larger"])
    root, orch, client, _ = setup(make_project, settings, [llm_json(), reply],
                                  [failed_build("ERROR: Failed building wheel for mypkg\ngcc: command not found"), ok_build()])
    report = orch.run_full(root, approve_build=True)
    first = report.history.attempts[0]
    assert first.category == "missing_system_dependency" and first.fixable and first.repeated == 1
    assert first.failing_instruction == "RUN pip install" and "gcc" in first.diff and first.diff.startswith("--- attempt 1")
    assert (first.explanation, first.expected_effect, first.remaining_risks) == (
        "wheel needs a compiler", "pip can build mypkg", ["image gets larger"])
    prompt = client.calls[1][-1]["content"]
    assert "CLASSIFIED ERROR" in prompt and "Category: missing_system_dependency" in prompt
    md = report.history.to_markdown()
    assert "missing_system_dependency" in md and "```diff" in md and "Expected effect" in md


def test_repeated_identical_error_stops_the_loop(make_project, settings):
    same = "ERROR: could not resolve the thing: foo"
    fixes = [FIXED, FIXED.replace("gcc", "build-essential")]
    replies = [llm_json()] + [correction_json(f, f"try {i}", [f"fix {i}"]) for i, f in enumerate(fixes)]
    root, orch, client, builder = setup(make_project, settings, replies, [failed_build(same)] * 3, max_retries=5,
                                        repeat_error_limit=3)
    report = orch.run_full(root, approve_build=True)
    assert report.status == LoopStatus.REPEATED_ERROR.value and builder.calls == 3 and len(client.calls) == 3
    assert [a.repeated for a in report.history.attempts] == [1, 2, 3]
    assert "3 times in a row" in report.message
    assert "occurred 2 times in a row" in client.calls[2][-1]["content"]  # the model is told its fix did not work


def test_non_identical_errors_do_not_count_as_repeats(make_project, settings):
    replies = [llm_json(), correction_json(FIXED, "a", ["a"]), correction_json(FIXED.replace("gcc", "make"), "b", ["b"])]
    root, orch, _, builder = setup(make_project, settings, replies,
                                   [failed_build("error one"), failed_build("error two"), ok_build()], max_retries=5)
    assert orch.run_full(root, approve_build=True).success and builder.calls == 3


def test_history_knowledge_is_included_as_reference_only(make_project, settings):
    seen = []

    def knowledge(category):
        seen.append(category)
        return ["install gcc and libpq-dev before pip install", "second", "third", "fourth"]

    root = make_project(FLASK_FILES)
    client = ScriptedClient([correction_json(FIXED, "gcc", ["gcc"])])
    orch = Orchestrator(settings, client=client, builder=FakeBuilder([failed_build("gcc: command not found"), ok_build()]))
    analysis = orch.analyze(root)
    art = parse_correction(correction_json(GOOD_DOCKERFILE)).model_copy()
    orch.save(root, GeneratedArtifacts(dockerfile=GOOD_DOCKERFILE, dockerignore=GOOD_DOCKERIGNORE))
    report = orch.build_and_fix(root, analysis, GeneratedArtifacts(dockerfile=GOOD_DOCKERFILE, dockerignore=GOOD_DOCKERIGNORE),
                                approve_build=True, knowledge=knowledge)
    assert report.success and seen == ["missing_system_dependency"]
    prompt = client.calls[0][-1]["content"]
    assert "FIXES THAT WORKED BEFORE" in prompt and "libpq-dev" in prompt and "reference only" in prompt
    assert "fourth" not in prompt  # capped at three suggestions


# ----------------------------------------------------------------------------- regression guard
GOOD = GeneratedArtifacts(dockerfile=GOOD_DOCKERFILE, dockerignore=GOOD_DOCKERIGNORE)


def test_find_regressions_flags_dropped_user_healthcheck_and_secrets():
    worse = GOOD_DOCKERFILE.replace("USER appuser\n", "").replace("ENV PYTHONUNBUFFERED=1", "ENV API_TOKEN=abc PYTHONUNBUFFERED=1")
    worse = "\n".join(l for l in worse.splitlines() if not l.startswith("HEALTHCHECK")) + "\n"
    problems = find_regressions(GOOD, GeneratedArtifacts(dockerfile=worse, dockerignore="node_modules\n"))
    text = " | ".join(problems)
    assert "non-root USER" in text and "HEALTHCHECK" in text and "embedded-secret" in text and ".env" in text
    assert find_regressions(GOOD, GOOD) == []
    # fixing a build error with a *better* file is never a regression
    assert find_regressions(GOOD, GeneratedArtifacts(dockerfile=FIXED, dockerignore=GOOD_DOCKERIGNORE)) == []


def test_correction_that_drops_user_is_pushed_back_then_accepted(make_project, settings):
    no_user = FIXED.replace("USER appuser\n", "")
    replies = [llm_json(), correction_json(no_user, "gcc", ["gcc"]), correction_json(FIXED, "gcc", ["gcc, keep USER"])]
    root, orch, client, builder = setup(make_project, settings, replies, [failed_build("gcc: command not found"), ok_build()])
    report = orch.run_full(root, approve_build=True)
    assert report.success and builder.dockerfiles_built[1] == FIXED
    assert "worse than the previous one" in client.calls[2][-1]["content"] and "non-root USER" in client.calls[2][-1]["content"]


def test_correction_that_stays_worse_is_rejected_and_original_kept(make_project, settings):
    no_user = FIXED.replace("USER appuser\n", "")
    replies = [llm_json(), correction_json(no_user, "gcc", ["gcc"]), correction_json(no_user, "gcc again", ["gcc"])]
    root, orch, _, builder = setup(make_project, settings, replies, [failed_build("gcc: command not found")])
    report = orch.run_full(root, approve_build=True)
    assert report.status == "non_recoverable_error" and builder.calls == 1
    assert "Correction rejected" in report.history.attempts[0].diagnosis
    assert (root / "Dockerfile").read_text() == GOOD_DOCKERFILE  # never silently replaced


def test_correction_pointing_at_a_nonexistent_file_is_rejected(make_project, settings):
    bad = FIXED.replace("COPY requirements.txt .", "COPY reqs-new.txt .")
    replies = [llm_json(), correction_json(bad, "x", ["x"]), correction_json(bad, "x", ["x"])]
    root, orch, _, _ = setup(make_project, settings, replies, [failed_build("gcc: command not found")])
    report = orch.run_full(root, approve_build=True)
    assert report.status == "non_recoverable_error" and "reqs-new.txt" in report.history.attempts[0].diagnosis

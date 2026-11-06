"""Phase 4: seven-category quality scoring, COPY-source verification, .dockerignore semantics, Hadolint adapter."""
from __future__ import annotations

import json

import pytest

from app.analyzer.project_scanner import ProjectScanner
from app.docker.hadolint import HadolintFinding, run_hadolint
from app.docker.quality import CATEGORIES, score_dockerfile
from app.utils.dockerignore import is_ignored
from tests.conftest import FLASK_FILES, GOOD_DOCKERFILE, GOOD_DOCKERIGNORE, VITE_FILES, EXPRESS_FILES


def codes(report, category=None):
    return {f.code for c in report.categories.values() for f in c.findings if category in (None, c.key)}


def test_good_dockerfile_scores_high_with_all_categories():
    r = score_dockerfile(GOOD_DOCKERFILE, GOOD_DOCKERIGNORE, use_hadolint=False)
    assert r.overall >= 97 and r.grade == "A" and set(r.categories) == set(CATEGORIES)
    assert abs(sum(c.weight for c in r.categories.values()) - 1.0) < 1e-9
    assert "Overall quality" in r.format()


BAD = """\
FROM python
COPY . .
RUN pip install flask
RUN apt-get update
RUN apt-get install -y gcc
ENV API_TOKEN=abc123
EXPOSE 22
CMD python app.py --host 127.0.0.1
"""


def test_bad_dockerfile_is_penalised_in_the_right_categories():
    r = score_dockerfile(BAD, "", use_hadolint=False)
    assert r.overall <= 59 and r.grade in ("D", "F")  # never above 59 with high-severity findings
    assert "risky-port" in codes(r, "security")
    assert {"embedded-secret", "runs-as-root"} <= codes(r, "security")
    assert {"unpinned-image", "pip-adhoc"} <= codes(r, "reproducibility")
    assert {"apt-update-alone"} <= codes(r, "build_efficiency")
    assert {"apt-recommends", "full-base-image"} <= codes(r, "image_size")
    assert "loopback-bind" in codes(r, "runtime")
    assert {"no-workdir", "shell-form-cmd"} <= codes(r, "maintainability")
    assert r.categories["security"].score <= 40 and r.recommendations and len(r.recommendations) <= 10
    assert r.recommendations[0]  # most severe first


def test_high_severity_cap_only_applies_to_blockers():
    clean = score_dockerfile(GOOD_DOCKERFILE, GOOD_DOCKERIGNORE, use_hadolint=False)
    assert not any("capped" in n for n in clean.notes)
    secret = GOOD_DOCKERFILE.replace("ENV PYTHONUNBUFFERED=1", "ENV DB_PASSWORD=hunter2 PYTHONUNBUFFERED=1")
    capped = score_dockerfile(secret, GOOD_DOCKERIGNORE, use_hadolint=False)
    assert capped.overall == 59 and "embedded-secret" in codes(capped, "security")


def test_cache_order_detection():
    bad = "FROM python:3.12-slim\nWORKDIR /app\nCOPY . .\nRUN pip install -r requirements.txt\nCMD [\"x\"]\n"
    assert "cache-order" in codes(score_dockerfile(bad, use_hadolint=False))
    assert "cache-order" not in codes(score_dockerfile(GOOD_DOCKERFILE, use_hadolint=False))


def test_user_before_root_only_commands():
    bad = "FROM node:22-alpine\nWORKDIR /a\nUSER node\nRUN apk add --no-cache git\nCMD [\"node\"]\n"
    assert "root-after-user" in codes(score_dockerfile(bad, use_hadolint=False), "runtime")


def test_lockfile_awareness_from_analysis(make_project, settings):
    root = make_project(EXPRESS_FILES | {"package-lock.json": "{}"}, "ex")
    del_yarn = ScriptedAnalysis = ProjectScanner(settings).scan(root)
    df = "FROM node:22-alpine\nWORKDIR /app\nCOPY package*.json ./\nRUN npm install\nCOPY . .\nEXPOSE 4000\nUSER node\nCMD [\"node\", \"server.js\"]\n"
    assert del_yarn.primary_unit.lockfile in ("package-lock.json", "yarn.lock")
    r = score_dockerfile(df, GOOD_DOCKERIGNORE, analysis=del_yarn, use_hadolint=False)
    if del_yarn.primary_unit.lockfile == "package-lock.json":
        assert "npm-install" in codes(r)
    else:
        assert "npm-install" not in codes(r)


def test_port_mismatch_and_missing_healthcheck_use_analysis(make_project, settings):
    root = make_project(FLASK_FILES)  # app.run(port=5050)
    analysis = ProjectScanner(settings).scan(root)
    df = GOOD_DOCKERFILE.replace('HEALTHCHECK CMD python -c "import urllib.request as u; u.urlopen(\'http://localhost:5000\')"\n', "")
    r = score_dockerfile(df, GOOD_DOCKERIGNORE, analysis=analysis, use_hadolint=False)
    assert "port-mismatch" in codes(r, "runtime") and "healthcheck-backend" in codes(r, "best_practices")


def test_multi_stage_recommended_but_single_stage(make_project, settings):
    analysis = ProjectScanner(settings).scan(make_project(VITE_FILES, "vite"))
    df = "FROM node:22-alpine\nWORKDIR /a\nCOPY package*.json ./\nRUN npm ci\nCOPY . .\nRUN npm run build\nEXPOSE 80\nUSER node\nCMD [\"npx\", \"serve\"]\n"
    assert "single-stage" in codes(score_dockerfile(df, GOOD_DOCKERIGNORE, analysis=analysis, use_hadolint=False), "image_size")


def test_copy_sources_are_verified_against_the_project(make_project):
    root = make_project({"requirements.txt": "flask\n", "app.py": "x=1\n", "secret.txt": "s", "src/a.py": "1"}, "cp")
    df = ("FROM python:3.12-slim\nWORKDIR /app\nCOPY requirement.txt .\nCOPY secret.txt .\nCOPY src/*.py ./\n"
          "COPY nothing/*.js ./\nCOPY requirements.txt app.py ./\nCOPY --from=base /x /y\nUSER nobody\nCMD [\"python\"]\n")
    r = score_dockerfile(df, "secret.txt\n", project_root=root, use_hadolint=False)
    msgs = [f.message for f in r.categories["runtime"].findings]
    assert any("'requirement.txt' does not exist" in m for m in msgs)
    assert any("'secret.txt' is excluded by .dockerignore" in m for m in msgs)
    assert any("'nothing/*.js' does not exist" in m for m in msgs)
    assert not any("src/*.py" in m or "app.py" in m for m in msgs)
    assert r.categories["runtime"].score <= 40
    assert "COPY sources were not checked" in score_dockerfile(df, use_hadolint=False).notes[0]


@pytest.mark.parametrize("path,patterns,expected", [
    ("a.pyc", "*.pyc", True), ("sub/a.pyc", "*.pyc", False), ("sub/a.pyc", "**/*.pyc", True),
    ("node_modules/x/y.js", "node_modules", True), ("node_modules/x/y.js", "node_modules/", True),
    ("src/a.py", "*\n!src", False), ("lib/a.py", "*\n!src", True), (".env", ".env*", True),
    (".env.example", ".env*\n!.env.example", False), ("a/b/c.txt", "a/*/c.txt", True), ("x.txt", "# c\n\n", False),
    ("docs/a.md", "/docs", True), ("deep/docs/a.md", "/docs", False), ("t.log", "?.log", True), ("tt.log", "?.log", False),
])
def test_dockerignore_matching_follows_docker_semantics(path, patterns, expected):
    assert is_ignored(path, patterns) is expected


def test_hadolint_integration_and_fallback(monkeypatch):
    sample = [{"code": "DL3008", "level": "warning", "line": 3, "message": "Pin versions in apt-get install"},
              {"code": "DL3002", "level": "error", "line": 5, "message": "Last USER should not be root"},
              {"code": "DL3059", "level": "style", "line": 6, "message": "Multiple consecutive RUN"}]

    class Proc:
        returncode, stdout = 1, json.dumps(sample)

    found = run_hadolint("FROM x\n", runner=lambda *a, **k: Proc(), which=lambda n: "/usr/bin/hadolint")
    assert [f.code for f in found] == ["DL3002", "DL3008", "DL3059"]  # errors first
    assert run_hadolint("FROM x\n", which=lambda n: None) is None  # absent -> caller falls back
    class Broken:
        returncode, stdout = 2, "not json"
    assert run_hadolint("x", runner=lambda *a, **k: Broken(), which=lambda n: "h") is None
    def boom(*a, **k):
        raise OSError("nope")
    assert run_hadolint("x", runner=boom, which=lambda n: "h") is None

    base = score_dockerfile(GOOD_DOCKERFILE, GOOD_DOCKERIGNORE, use_hadolint=False)
    lint = [HadolintFinding(code="DL1", level="error", line=1, message="m")] * 3
    with_lint = score_dockerfile(GOOD_DOCKERFILE, GOOD_DOCKERIGNORE, hadolint_runner=lambda d: lint)
    assert with_lint.hadolint_used and with_lint.categories["best_practices"].score == base.categories["best_practices"].score - 30
    assert with_lint.overall < base.overall and "Hadolint: 3 finding(s)" in with_lint.format()
    fallback = score_dockerfile(GOOD_DOCKERFILE, GOOD_DOCKERIGNORE, hadolint_runner=lambda d: None)
    assert not fallback.hadolint_used and fallback.overall == base.overall and "not installed" in fallback.format()

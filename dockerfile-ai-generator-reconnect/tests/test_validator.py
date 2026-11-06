"""Dockerfile / .dockerignore validation, including malformed LLM output."""
from __future__ import annotations

import pytest

from app.docker.docker_validator import validate_artifacts, validate_dockerfile, validate_dockerignore
from tests.conftest import GOOD_DOCKERFILE, GOOD_DOCKERIGNORE


def codes(content: str) -> set[str]:
    return {i.code for i in validate_dockerfile(content).issues}


def error_codes(content: str) -> set[str]:
    return {i.code for i in validate_dockerfile(content).errors}


def test_good_dockerfile_passes_cleanly():
    report = validate_dockerfile(GOOD_DOCKERFILE)
    assert report.ok and report.issues == [] and report.score == 100


MULTISTAGE = """\
FROM node:22-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:1.27-alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
USER nginx
HEALTHCHECK CMD wget -qO- http://localhost/ || exit 1
CMD ["nginx", "-g", "daemon off;"]
"""


def test_multistage_and_line_continuations_are_valid():
    assert validate_dockerfile(MULTISTAGE).ok
    cont = "FROM python:3.12-slim\nRUN apt-get update && \\\n    apt-get install -y --no-install-recommends gcc \\\n    && rm -rf /var/lib/apt/lists/*\nCMD [\"python\"]\n"
    assert validate_dockerfile(cont).ok


def test_heredoc_body_is_not_parsed_as_instructions():
    text = "FROM alpine:3.20\nRUN <<EOF\nnot an instruction\nFOO bar\nEOF\nCMD [\"sh\"]\n"
    assert validate_dockerfile(text).ok


@pytest.mark.parametrize("content,code", [
    ("", "empty"),
    ("   \n\n", "empty"),
    ("RUN echo hi\n", "no-from"),
    ("# only a comment\n", "no-instructions"),
    ("FROM\n", "missing-arguments"),
    ("FROM alpine:3.20\nRUNN echo hi\n", "unknown-instruction"),
    ('FROM alpine:3.20\nCMD [gunicorn, app:app]\n', "malformed-exec-form"),
    ("FROM alpine:3.20\nCMD ['a', 'b']\n", "malformed-exec-form"),
    ('FROM alpine:3.20\nCMD []\n', "empty-exec-form"),
    ("FROM alpine:3.20\nENV API_SECRET=hunter2\n", "embedded-secret"),
    ("FROM alpine:3.20\nARG DB_PASSWORD=hunter2\n", "embedded-secret"),
    ("FROM alpine:3.20\nRUN curl -fsSL https://x.sh | sh\n", "pipe-to-shell"),
    ("FROM alpine:3.20\nRUN docker run --privileged x\n", "privileged-or-socket"),
    ("FROM alpine:3.20\nCOPY /var/run/docker.sock /s\n", "docker-socket"),
    ("FROM alpine:3.20\nCOPY .env /app/.env\n", "copies-env-file"),
    ("FROM alpine:3.20\nEXPOSE 99999\n", "bad-port"),
    ("FROM alpine:3.20\nEXPOSE abc\n", "bad-port"),
    ("FROM alpine:3.20\nCOPY --from=ghost /a /b\n", "bad-stage-ref"),
    ("FROM alpine:3.20\nCOPY --from=0 /a /b\n", "bad-stage-ref"),
    ("FROM alpine:3.20 AS a\nFROM alpine:3.20 AS a\n", "duplicate-stage"),
    ("FROM alpine:3.20\nCOPY onlyonearg\n", "missing-arguments"),
    ("FROM alpine:3.20\n```\n", "markdown-fence"),
    ("FROM alpine:3.20\nUSER a b\n", "bad-user"),
])
def test_malformed_dockerfiles_are_rejected(content, code):
    assert code in error_codes(content)
    assert not validate_dockerfile(content).ok


def test_null_bytes_and_size_limits():
    assert "binary" in error_codes("FROM alpine\x00\n")
    assert "too-large" in error_codes("FROM alpine:3.20\n" + "# x\n" * 40_000)


def test_warnings_do_not_block_but_lower_score():
    report = validate_dockerfile("FROM python\nRUN apt-get install -y gcc\nRUN pip install flask\nCMD [\"python\"]\n")
    assert report.ok
    assert {"unpinned-image", "apt-recommends", "apt-cache", "pip-cache", "runs-as-root", "no-expose",
            "no-healthcheck"} <= {i.code for i in report.warnings}
    assert report.score <= 65


def test_latest_tag_and_root_user_warnings():
    c = codes("FROM node:latest\nUSER root\nEXPOSE 3000\nCMD [\"node\"]\n")
    assert "latest-tag" in c and "runs-as-root" in c


def test_secret_references_via_variables_are_allowed():
    assert "embedded-secret" not in codes("FROM alpine:3.20\nARG API_TOKEN\nENV API_TOKEN=$API_TOKEN\n")


def test_dockerignore_checks():
    assert validate_dockerignore(GOOD_DOCKERIGNORE).issues == []
    assert {i.code for i in validate_dockerignore("node_modules\n").issues} == {"no-git-exclude", "no-env-exclude"}
    assert {i.code for i in validate_dockerignore("").issues} == {"empty-dockerignore"}
    assert [i.code for i in validate_dockerignore("*\n").errors] == ["ignores-everything"]
    assert validate_dockerignore("*\n!src\n.git\n.env\n").ok


def test_validate_artifacts_merges_reports():
    merged = validate_artifacts("FROM alpine:3.20\nEXPOSE 80\nCMD [\"sh\"]\n", "node_modules\n")
    assert merged.ok and any(i.code == "no-env-exclude" for i in merged.issues)
    assert "no-env-exclude" in merged.format()

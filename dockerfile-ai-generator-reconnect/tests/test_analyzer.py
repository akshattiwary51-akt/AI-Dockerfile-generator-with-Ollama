"""Project analysis: framework detection, missing manifests, secret exclusion."""
from __future__ import annotations

import json

import pytest

from app.analyzer import framework_detector as fd
from app.analyzer.project_scanner import ProjectScanner
from tests.conftest import EXPRESS_FILES, FASTAPI_FILES, FLASK_FILES, VITE_FILES


def scan(root, settings):
    return ProjectScanner(settings).scan(root)


def test_flask_project_detection(make_project, settings):
    a = scan(make_project(FLASK_FILES), settings)
    assert (a.language, a.framework) == ("Python", "Flask")
    u = a.primary_unit
    assert u.role == "backend" and u.entry_points == ["app:app"] and u.ports == [5050]
    assert "flask==3.0.0" in u.dependencies and u.package_manager == "pip"
    assert "DATABASE_URL" in a.env_vars and "SECRET_KEY" in a.secret_env_vars
    assert a.strategy.base_image == "python:3.12-slim"
    assert "gunicorn" in a.strategy.start_command_hint and "app:app" in a.strategy.start_command_hint


def test_fastapi_project_detection(make_project, settings):
    a = scan(make_project(FASTAPI_FILES), settings)
    assert (a.language, a.framework) == ("Python", "FastAPI")
    assert a.primary_unit.entry_points == ["main:api"] and a.primary_unit.ports == [8000]
    assert a.primary_unit.runtime_version == ">=3.10"
    assert a.strategy.start_command_hint.startswith("uvicorn main:api")


def test_react_vite_project_detection(make_project, settings):
    a = scan(make_project(VITE_FILES), settings)
    assert (a.language, a.framework) == ("JavaScript", "React (Vite)")
    u = a.primary_unit
    assert u.role == "frontend" and u.build_output_dir == "dist" and u.lockfile == "package-lock.json"
    assert a.strategy.multi_stage and a.strategy.runtime_image.startswith("nginx")
    assert "VITE_API_URL" in a.env_vars


def test_express_project_detection(make_project, settings):
    a = scan(make_project(EXPRESS_FILES), settings)
    assert (a.language, a.framework) == ("JavaScript", "Express")
    u = a.primary_unit
    assert u.role == "backend" and u.ports == [4000] and u.entry_points == ["server.js"]
    assert u.package_manager == "yarn" and u.runtime_version == ">=18"
    assert a.strategy.base_image == "node:22-alpine"
    assert "API_TOKEN" in a.secret_env_vars


def test_django_nextjs_go_java(make_project, settings):
    django = scan(make_project({"requirements.txt": "Django==5.0\n", "manage.py": "", "site/wsgi.py": ""}, "dj"),
                  settings)
    assert django.framework == "Django" and django.primary_unit.ports == [8000]
    nxt = scan(make_project({"package.json": json.dumps({"dependencies": {"next": "14", "react": "18"}})}, "nx"),
               settings)
    assert nxt.framework == "Next.js" and nxt.primary_unit.ports == [3000]
    go = scan(make_project({"go.mod": "module ex.com/app\n\ngo 1.22\n\nrequire github.com/gin-gonic/gin v1.9.1\n",
                            "main.go": 'package main\nfunc main(){ r.Run(":9090") }\n'}, "go"), settings)
    assert go.framework == "Gin" and go.primary_unit.ports == [9090] and go.strategy.base_image == "golang:1.22-alpine"
    java = scan(make_project({"pom.xml": ("<project><parent><artifactId>spring-boot-starter-parent</artifactId></parent>"
                                          "<properties><java.version>17</java.version></properties></project>"),
                              "src/main/resources/application.properties": "server.port=8181\n"}, "jv"), settings)
    assert java.framework == "Spring Boot" and java.primary_unit.ports == [8181]
    assert java.strategy.runtime_image == "eclipse-temurin:17-jre"


def test_missing_manifest_python_and_node(make_project, settings):
    py = scan(make_project({"script.py": "print('hi')\n"}, "py"), settings)
    assert py.language == "Python" and py.framework is None
    assert any("No dependency manifest" in w for w in py.warnings)
    js = scan(make_project({"index.js": "console.log(1)\n"}, "js"), settings)
    assert js.language == "JavaScript" and any("package.json" in w for w in js.warnings)


def test_empty_and_missing_directories(make_project, settings, tmp_path):
    empty = scan(make_project({}, "empty"), settings)
    assert empty.language == "unknown" and any("Could not detect" in w for w in empty.warnings)
    with pytest.raises(FileNotFoundError):
        scan(tmp_path / "nope", settings)
    f = tmp_path / "file.txt"
    f.write_text("x")
    with pytest.raises(NotADirectoryError):
        scan(f, settings)


def test_malformed_manifests_do_not_crash(make_project, settings):
    a = scan(make_project({"package.json": "{ not json", "pyproject.toml": "= broken"}), settings)
    assert a.language in {"Python", "JavaScript"}  # degraded but alive


def test_sensitive_files_are_excluded(make_project, settings):
    files = dict(FLASK_FILES)
    files.update({".env": "SECRET_KEY=supersecretvalue123\n", "id_rsa": "-----BEGIN PRIVATE KEY-----\nabc\n",
                  "certs/server.pem": "pem-data", "secrets.json": '{"k": "supersecretvalue123"}',
                  ".env.local": "TOKEN=supersecretvalue123", ".env.example": "DATABASE_URL=\nSTRIPE_API_KEY=\n",
                  "node_modules/dep/index.js": "x", ".git/config": "[core]", "logo.png": "\x89PNG",
                  "Procfile": "web: gunicorn app:app --token=ghp_abcdefghijklmnopqrstuvwxyz123456\n"})
    a = scan(make_project(files), settings)
    dump = a.model_dump_json()
    assert "supersecretvalue123" not in dump
    assert set(a.skipped_sensitive_files) == {".env", "id_rsa", "certs/server.pem", "secrets.json", ".env.local"}
    entries = {ln.strip().rstrip("/") for ln in a.structure.splitlines()}
    for hidden in (".env", ".env.local", "id_rsa", "server.pem", "certs", "secrets.json", "node_modules", ".git",
                   "logo.png"):
        assert hidden not in entries, hidden
    assert ".env.example" in entries  # templates are safe and useful
    assert not set(a.key_files) & set(a.skipped_sensitive_files)
    assert "ghp_abcdefghijklmnopqrstuvwxyz123456" not in dump and "[REDACTED]" in a.key_files["Procfile"]
    # names (not values) are collected from the safe template
    assert "STRIPE_API_KEY" in a.env_vars and "STRIPE_API_KEY" in a.secret_env_vars


def test_monorepo_frontend_backend_split(make_project, settings):
    files = {f"frontend/{k}": v for k, v in VITE_FILES.items()}
    files.update({f"backend/{k}": v for k, v in EXPRESS_FILES.items()})
    a = scan(make_project(files), settings)
    assert a.is_monorepo and a.has_frontend_backend_split
    assert a.primary_unit.path == "backend" and a.framework == "Express"
    assert {u.path for u in a.units} == {"frontend", "backend"}
    assert any("primary unit lives in 'backend'" in w for w in a.warnings)


def test_existing_docker_files_detected(make_project, settings):
    files = dict(FLASK_FILES)
    files.update({"Dockerfile": "FROM python:3.12\n", ".dockerignore": ".git\n", "docker-compose.yml": "services: {}\n"})
    a = scan(make_project(files), settings)
    assert a.has_dockerfile and {"Dockerfile", ".dockerignore", "docker-compose.yml"} <= set(a.existing_docker_files)
    assert a.existing_dockerfile_content.startswith("FROM python")


@pytest.mark.parametrize("spec,expected", [(None, "3.12"), (">=3.9", "3.12"), ("==3.11.4", "3.11"), ("~=3.10", "3.10"),
                                           (">=3.8,<3.12", "3.11"), ("3.9", "3.9"), (">=3.13", "3.13")])
def test_python_version_picker(spec, expected):
    assert fd.pick_python_version(spec) == expected


@pytest.mark.parametrize("spec,expected", [(None, "22"), (">=18", "22"), ("20.x", "20"), ("^20.11.0", "20"),
                                           ("18", "18")])
def test_node_version_picker(spec, expected):
    assert fd.pick_node_version(spec) == expected

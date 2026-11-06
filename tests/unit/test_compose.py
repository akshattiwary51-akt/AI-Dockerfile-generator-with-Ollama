import subprocess

import pytest

from app.analyzer.project_scanner import ProjectScanner
from app.compose.generator import (ComposeProtectionError, generate_compose, generate_unit_dockerfiles, validate_compose,
                                   write_compose)
from app.utils import yamlish

yaml = pytest.importorskip("yaml")

APP = '''import os
from flask import Flask
app = Flask(__name__)
DB = os.environ["DATABASE_URL"]
KEY = os.environ.get("SECRET_KEY")
CACHE = os.environ.get("REDIS_URL")
DEBUG = os.getenv("APP_DEBUG")
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
'''


def analysis_for(make_project, settings, files=None):
    root = make_project(files or {"requirements.txt": "flask\npsycopg2-binary\nredis\n", "app.py": APP,
                                  "Dockerfile": "FROM python:3.13-slim\n"})
    return root, ProjectScanner(settings).scan(root)


# ----------------------------------------------------------------------------- emitter
@pytest.mark.parametrize("value", [
    {"a": 1, "b": [1, 2, {"c": "x y"}], "d": {"e": None, "f": True}, "g": {}, "h": []},
    {"ports": ["127.0.0.1:${P:-5000}:5000"], "cmd": ["CMD", "python", "-c", "import a;b('x',2)"]},
    {"s": ["yes", "no", "null", "123", "", " lead", "a: b", "# c", "multi\nline", "5000:5000", "é"]},
    [{"k": [{"x": 1}, [2, 3]]}, "tail"],
])
def test_yamlish_round_trips(value):
    assert yaml.safe_load(yamlish.dump(value)) == value


def test_yamlish_dump_all():
    assert list(yaml.safe_load_all(yamlish.dump_all([{"a": 1}, {"b": 2}]))) == [{"a": 1}, {"b": 2}]


# ----------------------------------------------------------------------------- generation
def test_flask_postgres_redis_stack(make_project, settings):
    root, a = analysis_for(make_project, settings)
    r = generate_compose(a, root)
    doc = yaml.safe_load(r.compose)
    assert set(doc["services"]) >= {"proj", "postgres", "redis"} or len(doc["services"]) == 3
    app = next(v for k, v in doc["services"].items() if "build" in v)
    assert app["ports"][0].startswith("127.0.0.1:") and app["ports"][0].endswith(":5000")
    assert app["environment"]["DATABASE_URL"].startswith("postgresql://") and "@postgres:5432" in app["environment"]["DATABASE_URL"]
    assert app["environment"]["REDIS_URL"] == "redis://redis:6379/0"
    assert app["environment"]["SECRET_KEY"] == "${SECRET_KEY:?Set SECRET_KEY in .env}"
    assert app["depends_on"]["postgres"]["condition"] == "service_healthy"
    pg = doc["services"]["postgres"]
    assert "ports" not in pg and pg["networks"] == ["data_net"] and "healthcheck" in pg
    assert doc["networks"]["data_net"] == {"internal": True} and set(doc["volumes"]) == {"postgres_data", "redis_data"}
    assert app["healthcheck"]["test"][1] == "python"


def test_no_secret_values_anywhere(make_project, settings):
    root, a = analysis_for(make_project, settings)
    r = generate_compose(a, root)
    for line in r.env_example.splitlines():
        if line and not line.startswith("#") and any(w in line.upper() for w in ("PASSWORD", "SECRET", "TOKEN")):
            assert line.split("=", 1)[1] == ""
    assert "POSTGRES_PASSWORD=" in r.env_example
    v = validate_compose(r.compose, r.env_example, use_docker=False)
    assert v.ok, v.errors


def test_missing_dockerfile_is_reported(make_project, settings):
    root, a = analysis_for(make_project, settings, {"requirements.txt": "flask\n", "app.py": APP})
    r = generate_compose(a, root)
    assert r.missing_dockerfiles == ["."] and any("No Dockerfile" in w for w in r.warnings)


def test_no_services_raises(make_project, settings):
    root = make_project({"README.md": "hi"})
    a = ProjectScanner(settings).scan(root)
    with pytest.raises(ValueError):
        generate_compose(a, root)


def test_monorepo_unique_host_ports(make_project, settings):
    files = {"backend/requirements.txt": "flask\n", "backend/app.py": APP, "backend/Dockerfile": "FROM x\n",
             "worker/requirements.txt": "flask\n", "worker/app.py": APP, "worker/Dockerfile": "FROM x\n"}
    root, a = analysis_for(make_project, settings, files)
    r = generate_compose(a, root)
    doc = yaml.safe_load(r.compose)
    ports = [s["ports"][0] for s in doc["services"].values() if "ports" in s]
    assert len(ports) == len(set(ports)) >= 2


# ----------------------------------------------------------------------------- validation
def test_validator_rejects_bad_compose():
    bad = yamlish.dump({"services": {"a": {"image": "x", "privileged": True, "volumes": ["/var/run/docker.sock:/s"],
                                           "environment": {"DB_PASSWORD": "hunter2"}, "depends_on": {"zz": {}},
                                           "networks": ["nope"]}}})
    v = validate_compose(bad, use_docker=False)
    text = " ".join(v.errors)
    assert not v.ok and "privileged" in text and "not allowed" in text and "literal value" in text \
        and "unknown service" in text and "undefined network" in text


def test_validator_flags_env_example_secret():
    ok = yamlish.dump({"services": {"a": {"image": "x"}}})
    v = validate_compose(ok, "DB_PASSWORD=realvalue\n", use_docker=False)
    assert not v.ok and "env.example" in v.errors[0]


def test_docker_compose_config_invocation(make_project, settings):
    root, a = analysis_for(make_project, settings)
    r = generate_compose(a, root)
    calls = []

    def runner(cmd, **k):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")
    v = validate_compose(r.compose, r.env_example, runner=runner, which=lambda n: "/usr/bin/docker")
    assert v.docker_compose_checked and calls[0][:3] == ["docker", "compose", "--env-file"] and calls[0][-2:] == ["config", "-q"]
    assert "up" not in calls[0]
    failing = lambda cmd, **k: subprocess.CompletedProcess(cmd, 1, "", "services.app.ports invalid")  # noqa: E731
    v2 = validate_compose(r.compose, r.env_example, runner=failing, which=lambda n: "x")
    assert not v2.ok and "docker compose config" in v2.errors[0]
    noplugin = lambda cmd, **k: subprocess.CompletedProcess(cmd, 1, "", "docker: 'compose' is not a docker command")  # noqa: E731
    assert validate_compose(r.compose, r.env_example, runner=noplugin, which=lambda n: "x").ok


# ----------------------------------------------------------------------------- writing
def test_write_protects_existing_and_never_writes_dotenv(make_project, settings):
    root, a = analysis_for(make_project, settings)
    r = generate_compose(a, root)
    paths = write_compose(root, r)
    assert {p.name for p in paths} == {"docker-compose.yml", ".env.example"} and not (root / ".env").exists()
    with pytest.raises(ComposeProtectionError):
        write_compose(root, r)
    write_compose(root, r, overwrite=True)
    assert (root / "docker-compose.yml.bak").exists()
    with pytest.raises(PermissionError):
        write_compose(root, r, filename="../x.yml", overwrite=True)


def test_write_refuses_symlink(make_project, settings, tmp_path):
    root, a = analysis_for(make_project, settings)
    (tmp_path / "victim").write_text("keep")
    (root / "docker-compose.yml").symlink_to(tmp_path / "victim")
    with pytest.raises(PermissionError):
        write_compose(root, generate_compose(a, root), overwrite=True)
    assert (tmp_path / "victim").read_text() == "keep"


def test_generate_unit_dockerfiles_reports_per_unit(make_project, settings):
    files = {"backend/requirements.txt": "flask\n", "backend/app.py": APP,
             "web/requirements.txt": "flask\n", "web/app.py": APP, "web/Dockerfile": "FROM x\n"}
    root, a = analysis_for(make_project, settings, files)

    class Orch:
        def analyze(self, d): return ProjectScanner(settings).scan(d)
        def generate(self, an): return type("A", (), {"dockerfile": "FROM python:3.13-slim\nUSER 1000\n", "dockerignore": ""})()
        def save(self, d, art, approve_overwrite=False):
            (d / "Dockerfile").write_text(art.dockerfile)
    out = generate_unit_dockerfiles(Orch(), a, root)
    assert out.get("backend") == "generated" and out.get("web") == "exists"


def test_literal_credential_in_url_is_still_caught():
    bad = yamlish.dump({"services": {"a": {"image": "x", "environment": {"DATABASE_URL": "postgresql://app:hunter2@db/x"}}}})
    v = validate_compose(bad, use_docker=False)
    assert not v.ok and any("credential" in e for e in v.errors)


# ----------------------------------------------------------------------------- CLI
def test_cli_compose_prints_then_writes(make_project, settings, capsys, monkeypatch):
    from app.main import main
    root, _ = analysis_for(make_project, settings)
    monkeypatch.setenv("HISTORY_DB", str(root.parent / "h.db"))
    assert main(["compose", str(root)]) == 0
    out = capsys.readouterr()
    assert "services:" in out.out and "validation: ok" in out.err and not (root / "docker-compose.yml").exists()
    assert main(["compose", str(root), "--write"]) == 0
    assert (root / "docker-compose.yml").exists() and not (root / ".env").exists()
    assert main(["compose", str(root), "--write"]) == 3  # protected
    assert main(["compose", str(root), "--write", "--yes-overwrite"]) == 0


def test_dockerfile_expose_is_authoritative_and_mismatch_is_reported(make_project, settings):
    src = APP.replace("port=5000", "port=5000")  # source says 5000
    root, a = analysis_for(make_project, settings, {"requirements.txt": "flask\n", "app.py": src,
                                                    "Dockerfile": "FROM python:3.13-slim\nEXPOSE 7070\n"})
    r = generate_compose(a, root)
    app = next(v for v in yaml.safe_load(r.compose)["services"].values() if "build" in v)
    assert app["ports"][0].endswith(":7070") and "7070" in app["healthcheck"]["test"][-1]
    assert any("EXPOSEs 7070" in w for w in r.warnings)


def test_plain_env_vars_are_passed_through_not_forced_empty(make_project, settings):
    root, a = analysis_for(make_project, settings)
    app = next(v for v in yaml.safe_load(generate_compose(a, root).compose)["services"].values() if "build" in v)
    assert "APP_DEBUG" in app["environment"] and app["environment"]["APP_DEBUG"] is None  # null = pass through if set
    assert validate_compose(generate_compose(a, root).compose, use_docker=False).ok

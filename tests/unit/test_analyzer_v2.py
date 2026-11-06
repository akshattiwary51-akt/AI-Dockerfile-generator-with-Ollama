"""Phase 2: PHP/Ruby, poetry carets, src layout, workspaces, services, compose parsing, provenance."""
from __future__ import annotations

import json

from app.analyzer.project_scanner import ProjectScanner
from app.llm.prompt_builder import build_generation_messages


def scan(make_project, settings, files, name="p"):
    return ProjectScanner(settings).scan(make_project(files, name))


def test_poetry_caret_versions_detect_framework(make_project, settings):
    a = scan(make_project, settings, {
        "pyproject.toml": '[tool.poetry]\nname="p"\n[tool.poetry.dependencies]\npython="^3.11"\nfastapi="^0.110"\n',
        "main.py": "from fastapi import FastAPI\napp = FastAPI()\n"})
    assert a.framework == "FastAPI" and a.primary_unit.package_manager == "poetry"
    assert a.primary_unit.runtime_version == "^3.11"
    assert any("poetry install" in n for n in a.strategy.notes)


def test_uv_and_pipenv_strategy_notes(make_project, settings):
    uv = scan(make_project, settings, {"pyproject.toml": '[project]\nname="u"\ndependencies=["flask>=3"]\n',
                                       "uv.lock": "", "app.py": "from flask import Flask\napp=Flask(__name__)\n"}, "uv")
    assert uv.primary_unit.package_manager == "uv" and any("uv sync" in n for n in uv.strategy.notes)
    pe = scan(make_project, settings, {"Pipfile": '[packages]\nflask = "*"\n', "app.py": "x=1\n"}, "pe")
    assert pe.primary_unit.package_manager == "pipenv" and any("pipenv install" in n for n in pe.strategy.notes)


def test_src_layout_entry_point_and_note(make_project, settings):
    a = scan(make_project, settings, {"requirements.txt": "flask\n",
                                      "src/myapp/app.py": "from flask import Flask\napp = Flask(__name__)\n"})
    assert a.primary_unit.entry_points == ["myapp.app:app"]
    assert any("PYTHONPATH" in n for n in a.primary_unit.notes)
    assert "myapp.app:app" in a.strategy.start_command_hint


def test_npm_workspaces_pick_a_member_not_the_root(make_project, settings):
    a = scan(make_project, settings, {
        "package.json": json.dumps({"name": "root", "private": True, "workspaces": ["packages/*"]}),
        "packages/web/package.json": json.dumps({"dependencies": {"react": "18"}, "devDependencies": {"vite": "5"}}),
        "packages/api/package.json": json.dumps({"dependencies": {"express": "4"}})})
    assert a.is_monorepo and a.has_frontend_backend_split
    assert a.primary_unit.path == "packages/api" and a.framework == "Express"
    assert any("workspace member" in n for n in a.primary_unit.notes)
    assert next(u for u in a.units if u.path == ".").workspace_root


def test_php_laravel(make_project, settings):
    a = scan(make_project, settings, {
        "composer.json": json.dumps({"require": {"php": "^8.2", "laravel/framework": "^11.0", "ext-pdo": "*"}}),
        "composer.lock": "{}", "public/index.php": "<?php\n$k = env('APP_KEY'); getenv('DB_PASSWORD');\n",
        "app/Http/Controller.php": "<?php\n$x = env('MAIL_HOST');\n"})
    assert (a.language, a.framework) == ("PHP", "Laravel") and a.primary_unit.runtime_version == "8.2"
    assert a.primary_unit.package_manager == "composer" and a.primary_unit.lockfile == "composer.lock"
    assert a.primary_unit.entry_points == ["public/index.php"] and a.primary_unit.ports == [80]
    assert {"APP_KEY", "DB_PASSWORD", "MAIL_HOST"} <= set(a.env_vars) and "DB_PASSWORD" in a.secret_env_vars
    assert a.strategy.runtime_image == "php:8.2-apache" and a.strategy.multi_stage
    assert all("ext-pdo" not in d for d in a.primary_unit.dependencies)


def test_ruby_rails_and_sinatra(make_project, settings):
    rails = scan(make_project, settings, {
        "Gemfile": "source 'https://rubygems.org'\nruby '3.2.2'\ngem 'rails', '~> 7.1'\ngem 'puma'\n",
        "Gemfile.lock": "", "config.ru": "run Rails.application\n",
        "config/app.rb": "ENV.fetch('SECRET_KEY_BASE'); ENV['RAILS_ENV']\n"}, "rails")
    assert (rails.language, rails.framework) == ("Ruby", "Rails") and rails.primary_unit.runtime_version == "3.2.2"
    assert rails.primary_unit.ports == [3000] and rails.strategy.base_image == "ruby:3.2-slim"
    assert "rails server" in rails.strategy.start_command_hint and "SECRET_KEY_BASE" in rails.secret_env_vars
    sin = scan(make_project, settings, {"Gemfile": "gem 'sinatra'\n", "config.ru": "run App\n"}, "sin")
    assert sin.framework == "Sinatra" and sin.primary_unit.ports == [4567]
    assert "rackup" in sin.strategy.start_command_hint


def test_confirmed_vs_inferred_provenance(make_project, settings):
    a = scan(make_project, settings, {"requirements.txt": "flask==3.0.0\n",
                                      "app.py": "from flask import Flask\napp = Flask(__name__)\n"}, "a")
    u = a.primary_unit
    assert any("framework Flask" in c for c in u.confirmed) and any("entry point" in c for c in u.confirmed)
    assert any("port 5000 is the Flask default" in i for i in u.inferred)
    assert any("no runtime version declared" in i for i in u.inferred)
    b = scan(make_project, settings, {"requirements.txt": "flask\n",
                                      "app.py": 'from flask import Flask\napp = Flask(__name__)\napp.run(port=5050)\n'}, "b")
    assert b.primary_unit.ports == [5050]
    assert any("5050" in c and "read from source" in c for c in b.primary_unit.confirmed)
    assert not any("port" in i for i in b.primary_unit.inferred)


def test_prompt_separates_confirmed_from_assumptions(make_project, settings):
    a = scan(make_project, settings, {"requirements.txt": "flask\n", "app.py": "from flask import Flask\napp=Flask(__name__)\n"})
    text = build_generation_messages(a, settings)[1]["content"]
    assert "CONFIRMED (read from project files)" in text and "ASSUMPTIONS (not verified" in text


def test_infra_services_detected_from_dependencies_and_env(make_project, settings):
    a = scan(make_project, settings, {
        "requirements.txt": "flask\npsycopg2-binary==2.9.9\nredis==5.0\n",
        "app.py": "import os\nfrom flask import Flask\napp=Flask(__name__)\nos.environ.get('MONGO_URL')\n"})
    assert set(a.services) == {"postgres", "redis", "mongo"}
    assert "psycopg2" in a.services["postgres"] and "inferred" in a.services["mongo"]
    node = scan(make_project, settings, {"package.json": json.dumps({"dependencies": {
        "express": "4", "pg": "8", "mysql2": "3", "ioredis": "5"}})}, "n")
    assert set(node.services) == {"postgres", "mysql", "redis"}
    none = scan(make_project, settings, {"requirements.txt": "flask\n", "app.py": "x=1\n"}, "none")
    assert none.services == {}


def test_existing_compose_services_parsed(make_project, settings):
    compose = "services:\n  web:\n    build: .\n    ports:\n      - '80:80'\n  db:\n    image: postgres:16\nvolumes:\n  x: {}\n"
    a = scan(make_project, settings, {"requirements.txt": "flask\n", "app.py": "x=1\n", "docker-compose.yml": compose})
    assert a.existing_compose == {"docker-compose.yml": ["web", "db"]}
    assert "services web, db" in build_generation_messages(a, settings)[1]["content"]


def test_php_ruby_missing_manifest_fallback(make_project, settings):
    php = scan(make_project, settings, {"index.php": "<?php echo 1;"}, "ph")
    assert php.language == "PHP" and any("composer.json" in w for w in php.warnings)
    rb = scan(make_project, settings, {"app.rb": "puts 1"}, "rb")
    assert rb.language == "Ruby" and any("Gemfile" in w for w in rb.warnings)


def test_python_port_default_read_from_environ_get(make_project, settings):
    from app.analyzer.project_scanner import ProjectScanner
    src = ('import os\nfrom flask import Flask\napp = Flask(__name__)\n'
           'if __name__ == "__main__":\n    app.run(port=int(os.environ.get("PORT", "5050")))\n')
    a = ProjectScanner(settings).scan(make_project({"requirements.txt": "flask\n", "app.py": src}))
    assert a.primary_unit.ports == [5050] and any("read from source" in c for c in a.primary_unit.confirmed)
    src2 = src.replace('os.environ.get("PORT", "5050")', "os.getenv('PORT', 8123)")
    b = ProjectScanner(settings).scan(make_project({"requirements.txt": "flask\n", "app.py": src2}, name="p2"))
    assert b.primary_unit.ports == [8123]

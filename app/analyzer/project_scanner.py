"""Filesystem scanner that produces a compact, secret-free ``ProjectAnalysis``."""
from __future__ import annotations

import fnmatch
import os
import re
from collections import Counter
from pathlib import Path
from typing import Optional

from app.analyzer import framework_detector as fd
from app.analyzer.models import DependencyInfo, ProjectAnalysis, ProjectUnit
from app.utils.config import Settings, get_settings
from app.utils.logger import get_logger
from app.utils.security import (SAFE_ENV_TEMPLATES, ensure_within, is_binary_extension, is_excluded_dir,
                                is_sensitive_file, redact_secrets)

log = get_logger("dockerfile_ai.scanner")

MAX_FILES = 5000
MAX_UNIT_DEPTH = 3
NON_UNIT_DIRS = {"test", "tests", "__tests__", "example", "examples", "docs", "doc", "samples", "fixtures",
                 "e2e", "benchmarks", "scripts", "tools"}
DOCKER_FILE_PATTERNS = ("Dockerfile", "Dockerfile.*", "*.Dockerfile", ".dockerignore", "docker-compose*.yml",
                        "docker-compose*.yaml", "compose*.yml", "compose*.yaml")
CODE_EXTS = {".py", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".go", ".java", ".kt", ".php", ".rb"}
ENV_PATTERNS = [
    re.compile(r"os\.environ(?:\.get)?\s*[\(\[]\s*['\"]([A-Z][A-Z0-9_]+)['\"]"),
    re.compile(r"os\.getenv\(\s*['\"]([A-Z][A-Z0-9_]+)['\"]"),
    re.compile(r"process\.env\.([A-Z][A-Z0-9_]+)"),
    re.compile(r"process\.env\[\s*['\"]([A-Z][A-Z0-9_]+)['\"]\s*\]"),
    re.compile(r"import\.meta\.env\.([A-Z][A-Z0-9_]+)"),
    re.compile(r"os\.(?:Getenv|LookupEnv)\(\s*\"([A-Z][A-Z0-9_]+)\""),
    re.compile(r"System\.getenv\(\s*\"([A-Z][A-Z0-9_]+)\""),
    re.compile(r"(?:getenv|env)\(\s*['\"]([A-Z][A-Z0-9_]+)['\"]"),           # PHP getenv()/Laravel env()
    re.compile(r"\$_(?:ENV|SERVER)\[\s*['\"]([A-Z][A-Z0-9_]+)['\"]\s*\]"),      # PHP superglobals
    re.compile(r"ENV(?:\.fetch)?[\[\(]\s*['\"]([A-Z][A-Z0-9_]+)['\"]"),           # Ruby ENV['X'] / ENV.fetch('X')
]

SERVICE_SIGNATURES: dict[str, tuple[str, ...]] = {
    "postgres": ("psycopg", "psycopg2", "psycopg2-binary", "asyncpg", "pg", "pg-promise", "postgres", "pgx",
                 "lib/pq", "org.postgresql", "postgresql", "pdo_pgsql", "node-postgres", "sequelize-pg"),
    "mysql": ("mysqlclient", "pymysql", "mysql-connector-python", "aiomysql", "mysql2", "mysql", "go-sql-driver/mysql",
              "mysql-connector-j", "mysql-connector-java", "mariadb"),
    "redis": ("redis", "aioredis", "ioredis", "go-redis", "jedis", "lettuce", "spring-boot-starter-data-redis",
              "predis/predis", "redis-rb", "django-redis", "celery-redis"),
    "mongo": ("pymongo", "motor", "mongoose", "mongodb", "mongo-driver", "mongoid", "spring-boot-starter-data-mongodb"),
}
ENV_NOISE = {"PATH", "HOME", "PWD", "USER", "SHELL", "TERM", "LANG", "TMPDIR", "HOSTNAME", "PYTHONPATH"}
SECRET_HINTS = ("KEY", "SECRET", "TOKEN", "PASSWORD", "PASSWD", "CREDENTIAL", "PRIVATE", "AUTH")
KEY_CONFIG_GLOBS = ("next.config.*", "vite.config.*", "nuxt.config.*", "tsconfig.json", "application.properties",
                    "application.yml", "application.yaml", ".python-version", ".nvmrc", "Procfile", "manage.py")
NON_TEXT_MANIFESTS = {"mvnw", "gradlew"}


class ProjectScanner:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    # ------------------------------------------------------------------ public
    def scan(self, root: Path | str) -> ProjectAnalysis:
        root_path = Path(root).expanduser()
        if not root_path.exists():
            raise FileNotFoundError(f"Project directory does not exist: {root_path}")
        if not root_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {root_path}")
        root_path = root_path.resolve()

        files: list[str] = []
        ext_counts: Counter = Counter()
        skipped_sensitive: list[str] = []
        docker_files: list[str] = []
        env_templates: list[Path] = []
        unit_candidates: list[tuple[Path, str, str]] = []  # (dir, rel, language)
        warnings: list[str] = []
        truncated = False

        for dirpath, dirnames, filenames in os.walk(root_path, followlinks=False):
            current = Path(dirpath)
            rel_dir = current.relative_to(root_path)
            depth = len(rel_dir.parts)
            dirnames[:] = sorted(d for d in dirnames if not is_excluded_dir(d) and not (current / d).is_symlink())

            if depth <= MAX_UNIT_DEPTH and (depth == 0 or not (set(rel_dir.parts) & NON_UNIT_DIRS)):
                for lang in fd.detect_languages_in_dir(current):
                    unit_candidates.append((current, rel_dir.as_posix() or ".", lang))

            for fname in sorted(filenames):
                path = current / fname
                if path.is_symlink():
                    continue
                rel = (rel_dir / fname).as_posix()
                if is_sensitive_file(path):
                    skipped_sensitive.append(rel)
                    continue
                if fname.lower() in SAFE_ENV_TEMPLATES:
                    env_templates.append(path)
                if depth <= MAX_UNIT_DEPTH and any(fnmatch.fnmatch(fname, p) for p in DOCKER_FILE_PATTERNS):
                    docker_files.append(rel)
                if is_binary_extension(path):
                    continue
                if len(files) >= MAX_FILES:
                    truncated = True
                    continue
                files.append(rel)
                ext_counts[path.suffix.lower()] += 1

        if truncated:
            warnings.append(f"Project has more than {MAX_FILES} files; analysis covers the first {MAX_FILES}.")

        units = self._build_units(unit_candidates, root_path)
        primary = self._choose_primary(units)

        if primary is None:
            guessed = fd.guess_language_from_extensions(ext_counts)
            if guessed:
                primary = ProjectUnit(path=".", language=guessed, role="unknown")
                try:
                    fd._DETECTORS[guessed](root_path, DependencyInfo(), primary)
                except Exception as exc:  # defensive: detection must never crash a scan
                    log.warning("Fallback detection failed: %s", exc)
                units = [primary]
                manifest_hint = {"Python": "requirements.txt / pyproject.toml / Pipfile / setup.py",
                                 "JavaScript": "package.json", "Go": "go.mod", "Java": "pom.xml / build.gradle",
                                 "PHP": "composer.json", "Ruby": "Gemfile"}[guessed]
                warnings.append(f"No dependency manifest found ({manifest_hint}); language guessed from file "
                                f"extensions as {guessed}. Dependencies cannot be determined.")
            else:
                warnings.append("Could not detect a supported language (Python, JavaScript/Node.js, Go, Java).")

        analysis = ProjectAnalysis(
            name=root_path.name, root=str(root_path), file_count=len(files),
            skipped_sensitive_files=sorted(skipped_sensitive), existing_docker_files=sorted(docker_files),
            warnings=warnings, units=units, primary_unit=primary)

        if primary:
            analysis.language = primary.language
            analysis.framework = primary.framework
            analysis.strategy = fd.suggest_strategy(primary)
        analysis.is_monorepo = len({u.path for u in units}) > 1
        roles = {u.role for u in units}
        analysis.has_frontend_backend_split = "frontend" in roles and bool(roles & {"backend", "fullstack"})
        if analysis.is_monorepo:
            analysis.warnings.append(
                "Multiple sub-projects detected (" + ", ".join(f"{u.path}:{u.framework or u.language}" for u in units)
                + "); the primary unit is used for the Dockerfile.")
        ws_roots = [u for u in units if u.workspace_root]
        if ws_roots and primary and primary.path != ".":
            primary.notes.append("npm/yarn/pnpm workspace member: the lockfile lives at the repository root, so run the "
                                 "install from the root (copy root package.json + lockfile + member manifests).")
        if primary and primary.path != ".":
            analysis.warnings.append(f"No root manifest; primary unit lives in '{primary.path}'. "
                                     "Dockerfile paths must account for this directory.")

        analysis.env_vars, analysis.secret_env_vars = self._collect_env_vars(root_path, files, env_templates)
        analysis.services = self._detect_services(units, analysis.env_vars)
        analysis.existing_compose = self._read_compose_services(root_path, docker_files)
        analysis.structure = self._render_structure(files)
        analysis.key_files = self._collect_key_files(root_path, primary, files)
        analysis.existing_dockerfile_content = self._read_existing_dockerfile(root_path)
        if skipped_sensitive:
            log.info("Excluded %d sensitive file(s) from analysis", len(skipped_sensitive))
        log.info("Scanned %s: %s", root_path.name, analysis.summary())
        return analysis

    # ------------------------------------------------------------------ helpers
    def _build_units(self, candidates: list[tuple[Path, str, str]], root: Path) -> list[ProjectUnit]:
        units: list[ProjectUnit] = []
        for directory, rel, lang in candidates:
            try:
                units.append(fd.detect_unit(directory, rel, lang))
            except Exception as exc:  # malformed manifests must not abort the scan
                log.warning("Skipping unit %s (%s): %s", rel, lang, exc)
        return units

    @staticmethod
    def _detect_services(units: list[ProjectUnit], env_vars: list[str]) -> dict[str, str]:
        found: dict[str, str] = {}
        for unit in units:
            names = {re.split(r"[<>=!~^*,(\[\s@;]", d.strip(), maxsplit=1)[0].lower().replace("_", "-")
                     for d in unit.dependencies}
            names |= {n.split(":")[-1] for n in names}
            for service, sigs in SERVICE_SIGNATURES.items():
                for sig in sigs:
                    if service not in found and any(n == sig or n.endswith("/" + sig) or sig in n.split("-") and
                                                    len(sig) > 4 for n in names):
                        found[service] = f"dependency '{sig}' in {unit.path}"
        for var in env_vars:
            for service in SERVICE_SIGNATURES:
                if service not in found and service.upper() in var.upper().replace("POSTGRES", "POSTGRES"):
                    found[service] = f"environment variable name {var} (inferred)"
        return found

    @staticmethod
    def _read_compose_services(root: Path, docker_files: list[str]) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for rel in docker_files:
            if not re.search(r"(docker-)?compose.*\.ya?ml$", rel):
                continue
            try:
                text = (root / rel).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            names: list[str] = []
            in_services, indent = False, None
            for line in text.splitlines():
                if re.match(r"^services:\s*$", line):
                    in_services = True
                    continue
                if in_services:
                    if line.strip() and not line.startswith((" ", "\t", "#")):
                        break
                    m = re.match(r"^(\s+)([A-Za-z0-9_.-]+):\s*$", line)
                    if m and (indent is None or len(m.group(1)) == indent):
                        indent = len(m.group(1))
                        names.append(m.group(2))
            result[rel] = names
        return result

    @staticmethod
    def _choose_primary(units: list[ProjectUnit]) -> Optional[ProjectUnit]:
        if not units:
            return None
        members = [u for u in units if not u.workspace_root]
        if members:  # a workspace root is only a container; build one of its members
            units = members
        rank = {"backend": 0, "fullstack": 1, "frontend": 2, "cli": 3, "library": 4, "unknown": 5}
        return sorted(units, key=lambda u: (u.path != ".", rank.get(u.role, 9), len(u.path), u.path))[0]

    def _collect_env_vars(self, root: Path, files: list[str], templates: list[Path]) -> tuple[list[str], list[str]]:
        names: set[str] = set()
        for tpl in templates:  # names only: values in templates are never read into the result
            try:
                text = ensure_within(root, tpl).read_text(encoding="utf-8", errors="replace")
            except (OSError, PermissionError):
                continue
            for line in text.splitlines():
                m = re.match(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=", line)
                if m:
                    names.add(m.group(1))
        scanned = 0
        for rel in files:
            path = root / rel
            if path.suffix.lower() not in CODE_EXTS or scanned >= 400:
                continue
            try:
                if path.stat().st_size > self.settings.max_file_bytes:
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            scanned += 1
            for pat in ENV_PATTERNS:
                names.update(pat.findall(text))
        names -= ENV_NOISE
        ordered = sorted(names)
        secrets = [n for n in ordered if any(h in n.upper() for h in SECRET_HINTS)]
        return ordered, secrets

    @staticmethod
    def _render_structure(files: list[str], max_lines: int = 120, max_depth: int = 3, per_dir: int = 15) -> str:
        tree: dict = {}
        for rel in files:
            node = tree
            parts = rel.split("/")
            for part in parts[:-1]:
                node = node.setdefault(part + "/", {})
            node.setdefault(parts[-1], None)
        lines: list[str] = []

        def walk(node: dict, depth: int) -> None:
            entries = sorted(node.items(), key=lambda kv: (kv[1] is None, kv[0]))
            for i, (name, child) in enumerate(entries):
                if len(lines) >= max_lines:
                    return
                if i >= per_dir:
                    lines.append("  " * depth + f"... ({len(entries) - per_dir} more)")
                    return
                lines.append("  " * depth + name)
                if child is not None and depth + 1 < max_depth:
                    walk(child, depth + 1)
                elif child:
                    lines.append("  " * (depth + 1) + "...")

        walk(tree, 0)
        if len(lines) >= max_lines:
            lines.append("... (structure truncated)")
        return "\n".join(lines)

    def _read_text(self, root: Path, rel: str, limit: int) -> Optional[str]:
        path = root / rel
        try:
            ensure_within(root, path)
            if not path.is_file() or path.is_symlink() or is_sensitive_file(path):
                return None
            if path.stat().st_size > self.settings.max_file_bytes:
                return f"[file too large: {path.stat().st_size} bytes, omitted]"
            text = path.read_text(encoding="utf-8", errors="replace")
        except (OSError, PermissionError):
            return None
        text = redact_secrets(text)
        return text if len(text) <= limit else text[:limit] + "\n... [truncated]"

    def _collect_key_files(self, root: Path, primary: Optional[ProjectUnit], files: list[str]) -> dict[str, str]:
        result: dict[str, str] = {}
        if primary is None:
            return result
        base = "" if primary.path == "." else primary.path + "/"
        wanted = [base + m for m in primary.manifests if m not in NON_TEXT_MANIFESTS]
        wanted += [f for f in files if f.startswith(base) and "/" not in f[len(base):]
                   and any(fnmatch.fnmatch(Path(f).name, g) for g in KEY_CONFIG_GLOBS)]
        wanted += [f for f in files if f in (base + "src/main/resources/application.properties",
                                              base + "src/main/resources/application.yml")]
        for rel in dict.fromkeys(wanted):
            text = self._read_text(root, rel, 3000)
            if text is not None:
                result[rel] = text
        for entry in primary.entry_points[:2]:
            rel = base + entry.split(":")[0].split(" ")[0]
            for candidate in (rel, rel.replace(".", "/") + ".py"):
                if candidate in files:
                    text = self._read_text(root, candidate, 1500)
                    if text is not None:
                        result[candidate] = text
                    break
        return result

    def _read_existing_dockerfile(self, root: Path) -> Optional[str]:
        return self._read_text(root, "Dockerfile", 6000)

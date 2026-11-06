"""Extract dependencies, runtime versions and scripts from language manifests."""
from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path
from typing import Optional

from app.analyzer.models import DependencyInfo
from app.utils.logger import get_logger

log = get_logger("dockerfile_ai.deps")

PYTHON_MANIFESTS = ("requirements.txt", "pyproject.toml", "Pipfile", "setup.py", "setup.cfg")
NODE_MANIFESTS = ("package.json",)
GO_MANIFESTS = ("go.mod",)
PHP_MANIFESTS = ("composer.json",)
RUBY_MANIFESTS = ("Gemfile",)
JAVA_MANIFESTS = ("pom.xml", "build.gradle", "build.gradle.kts", "settings.gradle", "settings.gradle.kts")

NODE_LOCKFILES = (("pnpm-lock.yaml", "pnpm"), ("yarn.lock", "yarn"), ("package-lock.json", "npm"),
                  ("bun.lockb", "bun"))


def _read(path: Path, limit: int = 200_000) -> str:
    try:
        if path.stat().st_size > limit:
            return ""
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _req_name(line: str) -> Optional[str]:
    line = line.split("#", 1)[0].strip()
    if not line or line.startswith(("-", "git+", "http")):
        return None
    m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._\-]*)(\[[^\]]*\])?\s*(.*)$", line)
    if not m:
        return None
    return f"{m.group(1)}{m.group(3).split(';')[0].strip().replace(' ', '')}".strip()


def _pep508_name(spec: str) -> str:
    return _req_name(spec) or spec


def base_name(dep: str) -> str:
    """Return the lowercase package name of 'Flask==2.0' -> 'flask'."""
    return re.split(r"[<>=!~^*,(\[\s@;]", dep.strip(), maxsplit=1)[0].lower().replace("_", "-")


# ----------------------------------------------------------------------------- Python
def detect_python(directory: Path) -> DependencyInfo:
    info = DependencyInfo()
    deps: list[str] = []
    dev: list[str] = []

    for req in sorted(directory.glob("requirements*.txt")):
        info.manifests.append(req.name)
        target = deps if req.name == "requirements.txt" else dev
        for line in _read(req).splitlines():
            name = _req_name(line)
            if name:
                target.append(name)
    if info.manifests:
        info.package_manager = "pip"

    pyproject = directory / "pyproject.toml"
    if pyproject.is_file():
        info.manifests.append("pyproject.toml")
        try:
            data = tomllib.loads(_read(pyproject))
        except tomllib.TOMLDecodeError as exc:
            log.warning("Invalid pyproject.toml in %s: %s", directory, exc)
            data = {}
        proj = data.get("project", {})
        deps += [_pep508_name(d) for d in proj.get("dependencies", [])]
        for group in proj.get("optional-dependencies", {}).values():
            dev += [_pep508_name(d) for d in group]
        if proj.get("requires-python"):
            info.runtime_version = proj["requires-python"]
        info.module_name = proj.get("name")
        info.scripts.update({k: str(v) for k, v in proj.get("scripts", {}).items()})
        poetry = data.get("tool", {}).get("poetry", {})
        if poetry:
            info.package_manager = "poetry"
            for name, spec in poetry.get("dependencies", {}).items():
                if name.lower() == "python":
                    info.runtime_version = info.runtime_version or str(spec)
                else:
                    deps.append(f"{name}{spec if isinstance(spec, str) and spec != '*' else ''}")
            info.module_name = info.module_name or poetry.get("name")
        if "uv" in data.get("tool", {}) or (directory / "uv.lock").exists():
            info.package_manager = "uv"
        elif not info.package_manager:
            info.package_manager = "pip"
        for lock in ("poetry.lock", "uv.lock", "pdm.lock"):
            if (directory / lock).exists():
                info.lockfile = lock

    pipfile = directory / "Pipfile"
    if pipfile.is_file():
        info.manifests.append("Pipfile")
        info.package_manager = "pipenv"
        try:
            data = tomllib.loads(_read(pipfile))
        except tomllib.TOMLDecodeError:
            data = {}
        deps += [f"{k}{v if isinstance(v, str) and v != '*' else ''}" for k, v in data.get("packages", {}).items()]
        dev += list(data.get("dev-packages", {}).keys())
        pv = data.get("requires", {}).get("python_version")
        if pv:
            info.runtime_version = info.runtime_version or pv
        if (directory / "Pipfile.lock").exists():
            info.lockfile = "Pipfile.lock"

    setup_py = directory / "setup.py"
    if setup_py.is_file():
        info.manifests.append("setup.py")
        text = _read(setup_py)
        m = re.search(r"install_requires\s*=\s*\[(.*?)\]", text, re.S)
        if m:
            deps += [_pep508_name(s) for s in re.findall(r"['\"]([^'\"]+)['\"]", m.group(1))]
        pm = re.search(r"python_requires\s*=\s*['\"]([^'\"]+)['\"]", text)
        if pm:
            info.runtime_version = info.runtime_version or pm.group(1)
        info.package_manager = info.package_manager or "pip"

    setup_cfg = directory / "setup.cfg"
    if setup_cfg.is_file():
        info.manifests.append("setup.cfg")

    for fname in (".python-version", "runtime.txt"):
        f = directory / fname
        if f.is_file():
            txt = _read(f).strip()
            m = re.search(r"(\d+\.\d+(?:\.\d+)?)", txt)
            if m:
                info.runtime_version = m.group(1)
                info.manifests.append(fname)

    if not info.lockfile and (directory / "requirements.txt").exists():
        info.lockfile = "requirements.txt"
    info.dependencies = list(dict.fromkeys(deps))
    info.dev_dependencies = list(dict.fromkeys(dev))
    return info


# ----------------------------------------------------------------------------- Node
def detect_node(directory: Path) -> DependencyInfo:
    info = DependencyInfo(manifests=["package.json"])
    try:
        data = json.loads(_read(directory / "package.json") or "{}")
        if not isinstance(data, dict):
            data = {}
    except json.JSONDecodeError as exc:
        log.warning("Invalid package.json in %s: %s", directory, exc)
        data = {}
    info.module_name = data.get("name")
    ws = data.get("workspaces")
    if isinstance(ws, dict):
        ws = ws.get("packages")
    info.workspaces = [str(w) for w in ws] if isinstance(ws, list) else []
    main = data.get("main")
    info.main_entry = main if isinstance(main, str) else None
    info.dependencies = [f"{k}@{v}" for k, v in (data.get("dependencies") or {}).items()]
    info.dev_dependencies = [f"{k}@{v}" for k, v in (data.get("devDependencies") or {}).items()]
    info.scripts = {k: str(v) for k, v in (data.get("scripts") or {}).items()}
    engines = data.get("engines") or {}
    if isinstance(engines, dict) and engines.get("node"):
        info.runtime_version = str(engines["node"])
    nvmrc = directory / ".nvmrc"
    if nvmrc.is_file():
        m = re.search(r"(\d+(?:\.\d+)*)", _read(nvmrc))
        if m:
            info.runtime_version = info.runtime_version or m.group(1)
            info.manifests.append(".nvmrc")
    for lock, pm in NODE_LOCKFILES:
        if (directory / lock).exists():
            info.lockfile = lock
            info.package_manager = pm
            break
    else:
        pm_field = str(data.get("packageManager", ""))
        info.package_manager = pm_field.split("@")[0] if pm_field else "npm"
    return info


# ----------------------------------------------------------------------------- Go
def detect_go(directory: Path) -> DependencyInfo:
    info = DependencyInfo(manifests=["go.mod"], package_manager="go modules")
    text = _read(directory / "go.mod")
    m = re.search(r"^module\s+(\S+)", text, re.M)
    info.module_name = m.group(1) if m else None
    gv = re.search(r"^go\s+(\d+\.\d+(?:\.\d+)?)", text, re.M)
    info.runtime_version = gv.group(1) if gv else None
    deps: list[str] = []
    for block in re.findall(r"require\s*\((.*?)\)", text, re.S):
        for line in block.splitlines():
            parts = line.split("//")[0].split()
            if len(parts) >= 2:
                deps.append(f"{parts[0]}@{parts[1]}")
    deps += [f"{a}@{b}" for a, b in re.findall(r"^require\s+(\S+)\s+(\S+)", text, re.M)]
    info.dependencies = deps
    if (directory / "go.sum").exists():
        info.lockfile = "go.sum"
    return info


# ----------------------------------------------------------------------------- Java
def detect_java(directory: Path) -> DependencyInfo:
    info = DependencyInfo()
    deps: list[str] = []
    pom = directory / "pom.xml"
    if pom.is_file():
        info.manifests.append("pom.xml")
        info.package_manager = "maven"
        text = _read(pom)
        for g, a in re.findall(r"<dependency>\s*<groupId>([^<]+)</groupId>\s*<artifactId>([^<]+)</artifactId>", text):
            deps.append(f"{g.strip()}:{a.strip()}")
        for tag in ("maven.compiler.release", "java.version", "maven.compiler.source", "maven.compiler.target"):
            m = re.search(rf"<{re.escape(tag)}>\s*([\d.]+)\s*</{re.escape(tag)}>", text)
            if m:
                info.runtime_version = m.group(1)
                break
        parent = re.search(r"<parent>.*?<artifactId>([^<]+)</artifactId>", text, re.S)
        if parent:
            deps.append(f"parent:{parent.group(1).strip()}")
        am = re.search(r"</parent>.*?<artifactId>([^<]+)</artifactId>", text, re.S) or re.search(
            r"<artifactId>([^<]+)</artifactId>", text)
        info.module_name = am.group(1).strip() if am else None
    for gname in ("build.gradle", "build.gradle.kts", "settings.gradle", "settings.gradle.kts"):
        g = directory / gname
        if g.is_file():
            info.manifests.append(gname)
            info.package_manager = info.package_manager or "gradle"
            if gname.startswith("build"):
                text = _read(g)
                deps += [d for d in re.findall(
                    r"(?:implementation|api|compileOnly|runtimeOnly)\s*\(?\s*['\"]([^'\"]+)['\"]", text)]
                deps += re.findall(r"id\s*\(?['\"](org\.springframework\.boot|io\.quarkus|io\.micronaut\.[a-z.]+)['\"]", text)
                m = (re.search(r"JavaVersion\.VERSION_(\d+)", text)
                     or re.search(r"jvmToolchain\s*\(\s*(\d+)\s*\)", text)
                     or re.search(r"languageVersion\s*(?:=|\.set\()\s*JavaLanguageVersion\.of\(\s*(\d+)\s*\)", text)
                     or re.search(r"sourceCompatibility\s*=?\s*['\"]?(\d+(?:\.\d+)?)", text))
                if m and not info.runtime_version:
                    info.runtime_version = m.group(1)
    for wrapper in ("mvnw", "gradlew"):
        if (directory / wrapper).exists():
            info.manifests.append(wrapper)
    info.dependencies = list(dict.fromkeys(deps))
    return info


# ----------------------------------------------------------------------------- PHP
def detect_php(directory: Path) -> DependencyInfo:
    info = DependencyInfo(manifests=["composer.json"], package_manager="composer")
    try:
        data = json.loads(_read(directory / "composer.json") or "{}")
        if not isinstance(data, dict):
            data = {}
    except json.JSONDecodeError as exc:
        log.warning("Invalid composer.json in %s: %s", directory, exc)
        data = {}
    require = data.get("require") or {}
    dev = data.get("require-dev") or {}
    info.module_name = data.get("name")
    for name, spec in require.items():
        if name == "php":
            m = re.search(r"(\d+\.\d+)", str(spec))
            info.runtime_version = m.group(1) if m else None
        elif not name.startswith("ext-"):
            info.dependencies.append(f"{name}@{spec}")
    info.dev_dependencies = [f"{k}@{v}" for k, v in dev.items()]
    info.scripts = {k: str(v) for k, v in (data.get("scripts") or {}).items()}
    if (directory / "composer.lock").exists():
        info.lockfile = "composer.lock"
    return info


# ----------------------------------------------------------------------------- Ruby
def detect_ruby(directory: Path) -> DependencyInfo:
    info = DependencyInfo(manifests=["Gemfile"], package_manager="bundler")
    text = _read(directory / "Gemfile")
    for name, ver in re.findall(r"^\s*gem\s+['\"]([^'\"]+)['\"](?:\s*,\s*['\"]([^'\"]+)['\"])?", text, re.M):
        info.dependencies.append(f"{name}@{ver}" if ver else name)
    m = re.search(r"^\s*ruby\s+['\"]([\d.]+)['\"]", text, re.M)
    if m:
        info.runtime_version = m.group(1)
    rv = directory / ".ruby-version"
    if rv.is_file():
        mm = re.search(r"(\d+\.\d+(?:\.\d+)?)", _read(rv))
        if mm:
            info.runtime_version = info.runtime_version or mm.group(1)
            info.manifests.append(".ruby-version")
    if (directory / "Gemfile.lock").exists():
        info.lockfile = "Gemfile.lock"
    return info

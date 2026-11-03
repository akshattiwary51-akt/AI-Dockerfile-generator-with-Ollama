"""Framework, entry-point and port detection plus base-image strategy suggestions."""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Optional

from app.analyzer import dependency_detector as dd
from app.analyzer.models import DependencyInfo, ProjectUnit, Strategy
from app.utils.security import is_excluded_dir, is_sensitive_file

DEFAULT_PYTHON = (3, 12)
DEFAULT_NODE = 22
DEFAULT_JAVA = 21

EXT_LANGUAGE = {".py": "Python", ".js": "JavaScript", ".mjs": "JavaScript", ".jsx": "JavaScript",
                ".ts": "JavaScript", ".tsx": "JavaScript", ".go": "Go", ".java": "Java", ".kt": "Java"}


def guess_language_from_extensions(ext_counts: Counter) -> Optional[str]:
    tally: Counter = Counter()
    for ext, n in ext_counts.items():
        lang = EXT_LANGUAGE.get(ext)
        if lang:
            tally[lang] += n
    return tally.most_common(1)[0][0] if tally else None


def _read(path: Path, limit: int = 100_000) -> str:
    try:
        if path.stat().st_size > limit:
            return ""
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _existing(directory: Path, candidates: list[str]) -> list[Path]:
    return [directory / c for c in candidates if (directory / c).is_file()]


def _ports_from_text(text: str, patterns: list[str]) -> list[int]:
    found: list[int] = []
    for pat in patterns:
        for m in re.finditer(pat, text):
            p = int(m.group(1))
            if 1 <= p <= 65535 and p not in found:
                found.append(p)
    return found


def _node_name(dep: str) -> str:
    return dep.rsplit("@", 1)[0].lower() if "@" in dep[1:] else dep.lower()


# ----------------------------------------------------------------------------- versions
def pick_python_version(spec: Optional[str]) -> str:
    """Map a version constraint onto a concrete ``major.minor`` for a Docker tag."""
    default = DEFAULT_PYTHON
    if not spec:
        return f"{default[0]}.{default[1]}"
    pinned = re.match(r"^\s*(?:==|~=)?\s*(\d+)\.(\d+)", spec)
    if pinned and not spec.strip().startswith((">", "<")) and "," not in spec:
        return f"{pinned.group(1)}.{pinned.group(2)}"
    chosen = default
    lower = re.search(r">=?\s*(\d+)\.(\d+)", spec)
    upper = re.search(r"<(=?)\s*(\d+)\.(\d+)", spec)
    exact = re.search(r"==\s*(\d+)\.(\d+)", spec)
    if exact:
        return f"{exact.group(1)}.{exact.group(2)}"
    if lower:
        chosen = max(chosen, (int(lower.group(1)), int(lower.group(2))))
    if upper:
        up = (int(upper.group(2)), int(upper.group(3)))
        if not upper.group(1):  # strictly less than
            up = (up[0], up[1] - 1)
        if chosen > up:
            chosen = up
    return f"{chosen[0]}.{chosen[1]}"


def pick_node_version(spec: Optional[str]) -> str:
    if not spec:
        return str(DEFAULT_NODE)
    m = re.search(r"(\d+)", spec)
    if not m:
        return str(DEFAULT_NODE)
    major = int(m.group(1))
    if re.match(r"^\s*>=?", spec):
        return str(max(major, DEFAULT_NODE))
    return str(major)


def pick_java_version(spec: Optional[str]) -> str:
    if not spec:
        return str(DEFAULT_JAVA)
    if spec.startswith("1."):
        return spec.split(".")[1]
    m = re.match(r"(\d+)", spec)
    return m.group(1) if m else str(DEFAULT_JAVA)


def _major_minor(spec: Optional[str]) -> Optional[str]:
    if not spec:
        return None
    m = re.match(r"(\d+)\.(\d+)", spec)
    return f"{m.group(1)}.{m.group(2)}" if m else None


# ----------------------------------------------------------------------------- per-language detection
def _detect_python(directory: Path, info: DependencyInfo, unit: ProjectUnit) -> None:
    names = {dd.base_name(d) for d in info.dependencies + info.dev_dependencies}
    candidates = ["main.py", "app.py", "wsgi.py", "asgi.py", "server.py", "run.py", "application.py", "api.py",
                  "manage.py", "app/main.py", "app/__init__.py", "src/main.py", "src/app.py", "app/app.py"]
    files = _existing(directory, candidates)

    def module_path(p: Path) -> str:
        rel = p.relative_to(directory).with_suffix("")
        parts = list(rel.parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts)

    def find_app(pattern: str) -> Optional[tuple[Path, str]]:
        for f in files:
            m = re.search(pattern, _read(f), re.M)
            if m:
                return f, m.group(1)
        return None

    if "django" in names:
        unit.framework, unit.role = "Django", "backend"
        unit.ports = [8000]
        if (directory / "manage.py").is_file():
            unit.entry_points.append("manage.py")
        for wsgi in sorted(directory.glob("*/wsgi.py"))[:2]:
            unit.entry_points.append(f"{wsgi.parent.name}.wsgi:application")
        unit.notes.append("Run with gunicorn in production; run collectstatic if STATIC_ROOT is configured.")
    elif "fastapi" in names:
        unit.framework, unit.role = "FastAPI", "backend"
        found = find_app(r"^(\w+)\s*=\s*FastAPI\(")
        if found:
            unit.entry_points.append(f"{module_path(found[0])}:{found[1]}")
        for f in files:
            unit.ports = unit.ports or _ports_from_text(_read(f), [r"port\s*=\s*(\d{2,5})", r"--port\s+(\d{2,5})"])
        unit.ports = unit.ports or [8000]
    elif "flask" in names:
        unit.framework, unit.role = "Flask", "backend"
        found = find_app(r"^(\w+)\s*=\s*Flask\(")
        if found:
            unit.entry_points.append(f"{module_path(found[0])}:{found[1]}")
        elif (directory / "app" / "__init__.py").is_file() and "create_app" in _read(directory / "app" / "__init__.py"):
            unit.entry_points.append("app:create_app()")
        for f in files:
            unit.ports = unit.ports or _ports_from_text(_read(f), [r"\.run\([^)]*port\s*=\s*(\d{2,5})"])
        unit.ports = unit.ports or [5000]
    elif "streamlit" in names:
        unit.framework, unit.role, unit.ports = "Streamlit", "fullstack", [8501]
        unit.entry_points = [f.name for f in files][:1]
    else:
        for fw, label, port in (("starlette", "Starlette", 8000), ("aiohttp", "aiohttp", 8080),
                                ("tornado", "Tornado", 8888), ("sanic", "Sanic", 8000),
                                ("pyramid", "Pyramid", 6543), ("gradio", "Gradio", 7860)):
            if fw in names:
                unit.framework, unit.role, unit.ports = label, "backend", [port]
                break
    if not unit.entry_points:
        unit.entry_points = [str(f.relative_to(directory)) for f in files if f.name != "__init__.py"][:2]
    if "gunicorn" in names:
        unit.notes.append("gunicorn is already a dependency.")
    if "uvicorn" in names:
        unit.notes.append("uvicorn is already a dependency.")
    unit.role = unit.role if unit.role != "unknown" else ("cli" if info.scripts else "library")


def _detect_node(directory: Path, info: DependencyInfo, unit: ProjectUnit) -> None:
    all_deps = {_node_name(d) for d in info.dependencies + info.dev_dependencies}
    has = all_deps.__contains__
    scripts = info.scripts

    if has("next"):
        unit.framework, unit.role, unit.ports, unit.build_output_dir = "Next.js", "fullstack", [3000], ".next"
        for cfg in ("next.config.js", "next.config.mjs", "next.config.ts"):
            if "standalone" in _read(directory / cfg):
                unit.notes.append("next.config sets output: 'standalone' (copy .next/standalone and .next/static).")
    elif has("vite"):
        label = "React (Vite)" if has("react") else "Vue (Vite)" if has("vue") else \
            "Svelte (Vite)" if has("svelte") else "Vite"
        unit.framework, unit.role, unit.build_output_dir = label, "frontend", "dist"
        unit.ports = [80]
        unit.notes.append("Static frontend: build with `vite build`, serve dist/ via nginx on port 80 "
                          "(Vite dev server default is 5173 and is not for production).")
    elif has("react-scripts"):
        unit.framework, unit.role, unit.build_output_dir = "React (CRA)", "frontend", "build"
        unit.ports = [80]
        unit.notes.append("Static frontend: build output is build/, serve via nginx on port 80.")
    elif has("@nestjs/core"):
        unit.framework, unit.role, unit.ports, unit.build_output_dir = "NestJS", "backend", [3000], "dist"
    elif has("express"):
        unit.framework, unit.role = "Express", "backend"
    elif has("fastify"):
        unit.framework, unit.role = "Fastify", "backend"
    elif has("koa"):
        unit.framework, unit.role = "Koa", "backend"
    elif has("react"):
        unit.framework, unit.role = "React", "frontend"
    else:
        unit.framework, unit.role = None, "backend" if "start" in scripts else "library"

    entries: list[str] = []
    if info.main_entry:
        entries.append(info.main_entry)
    start = scripts.get("start", "")
    m = re.search(r"node\s+(?:--[\w-]+\s+)*(\S+\.(?:c|m)?js)", start)
    if m and m.group(1) not in entries:
        entries.append(m.group(1))
    cand =["server.js", "index.js", "app.js", "src/index.js", "src/server.js", "src/app.js", "server/index.js",
            "main.js", "src/main.ts", "src/index.ts", "src/server.ts", "server.ts"]
    for c in cand:
        if (directory / c).is_file() and c not in entries:
            entries.append(c)
    unit.entry_points = [e for e in entries if (directory / e).is_file() or e == start][:3]
    if "start" in scripts and unit.role != "frontend":
        unit.notes.append(f"package.json start script: {scripts['start']}")
    if "build" in scripts:
        unit.notes.append(f"package.json build script: {scripts['build']}")

    if unit.role == "backend" and not unit.ports:
        found: list[int] = []
        for e in unit.entry_points:
            found += _ports_from_text(_read(directory / e), [
                r"\.listen\(\s*(?:process\.env\.\w+\s*(?:\|\||\?\?)\s*)?(\d{2,5})",
                r"PORT\s*(?:\|\||\?\?|=)\s*['\"]?(\d{2,5})"])
        unit.ports = list(dict.fromkeys(found)) or [3000]
    if unit.framework in ("Express", "Fastify", "Koa", "NestJS") and "typescript" in all_deps:
        unit.notes.append("TypeScript project: compile before running (see build script).")


def _detect_go(directory: Path, info: DependencyInfo, unit: ProjectUnit) -> None:
    deps = " ".join(info.dependencies)
    for needle, label in (("gin-gonic/gin", "Gin"), ("labstack/echo", "Echo"), ("gofiber/fiber", "Fiber"),
                          ("go-chi/chi", "Chi"), ("gorilla/mux", "Gorilla Mux")):
        if needle in deps:
            unit.framework = label
            break
    mains = [p for p in [directory / "main.go", *sorted(directory.glob("cmd/*/main.go"))] if p.is_file()]
    unit.entry_points = [str(p.relative_to(directory)) for p in mains][:3]
    src = "\n".join(_read(p) for p in mains)
    if not unit.framework and "http.ListenAndServe" in src:
        unit.framework = "net/http"
    unit.ports = _ports_from_text(src, [r"(?:ListenAndServe|Run|Listen|Start)\(\s*\"[^\":]*:(\d{2,5})\""])
    unit.role = "backend" if (unit.framework or unit.ports) else "cli"
    if unit.role == "backend" and not unit.ports:
        unit.ports = [8080]
    if "github.com/mattn/go-sqlite3" in deps:
        unit.notes.append("go-sqlite3 requires CGO (gcc + musl-dev on Alpine, or use a Debian base).")


def _detect_java(directory: Path, info: DependencyInfo, unit: ProjectUnit) -> None:
    joined = " ".join(info.dependencies).lower()
    if "spring-boot" in joined or "org.springframework.boot" in joined:
        unit.framework, unit.role = "Spring Boot", "backend"
    elif "quarkus" in joined:
        unit.framework, unit.role = "Quarkus", "backend"
    elif "micronaut" in joined:
        unit.framework, unit.role = "Micronaut", "backend"
    else:
        unit.role = "library"
    port = None
    for cfg in ("src/main/resources/application.properties", "src/main/resources/application.yml",
                "src/main/resources/application.yaml"):
        text = _read(directory / cfg)
        m = re.search(r"server\.port\s*=\s*(\d+)", text) or re.search(r"^\s*port:\s*(\d+)", text, re.M)
        if m:
            port = int(m.group(1))
            break
    if unit.role == "backend":
        unit.ports = [port or 8080]
    for java in list(directory.glob("src/main/java/**/*.java"))[:200]:
        if "@SpringBootApplication" in _read(java, 50_000) or "public static void main" in _read(java, 50_000):
            unit.entry_points.append(str(java.relative_to(directory)))
            break
    wrapper = [w for w in ("mvnw", "gradlew") if (directory / w).exists()]
    if wrapper:
        unit.notes.append(f"Build wrapper present: {', '.join(wrapper)} (needs executable bit / chmod +x).")
    unit.build_output_dir = "target" if info.package_manager == "maven" else "build/libs"


_DETECTORS = {"Python": _detect_python, "JavaScript": _detect_node, "Go": _detect_go, "Java": _detect_java}
_DEP_DETECTORS = {"Python": dd.detect_python, "JavaScript": dd.detect_node, "Go": dd.detect_go,
                  "Java": dd.detect_java}


def detect_languages_in_dir(directory: Path) -> list[str]:
    """Languages whose manifests exist directly in ``directory``."""
    langs: list[str] = []
    if any((directory / m).is_file() for m in dd.NODE_MANIFESTS):
        langs.append("JavaScript")
    if any((directory / m).is_file() for m in dd.GO_MANIFESTS):
        langs.append("Go")
    if any((directory / m).is_file() for m in dd.JAVA_MANIFESTS):
        langs.append("Java")
    if any((directory / m).is_file() for m in dd.PYTHON_MANIFESTS) or list(directory.glob("requirements*.txt")):
        langs.append("Python")
    return langs


def detect_unit(directory: Path, rel_path: str, language: str) -> ProjectUnit:
    info = _DEP_DETECTORS[language](directory)
    unit = ProjectUnit(
        path=rel_path, language=language, package_manager=info.package_manager, lockfile=info.lockfile,
        runtime_version=info.runtime_version, manifests=info.manifests, dependencies=info.dependencies,
        dev_dependencies=info.dev_dependencies, scripts=info.scripts)
    _DETECTORS[language](directory, info, unit)
    return unit


# ----------------------------------------------------------------------------- strategy
def suggest_strategy(unit: ProjectUnit) -> Strategy:
    lang, fw = unit.language, unit.framework
    port = unit.ports[0] if unit.ports else None
    entry = next((e for e in unit.entry_points if ":" in e), unit.entry_points[0] if unit.entry_points else None)

    if lang == "Python":
        ver = pick_python_version(unit.runtime_version)
        notes = [f"Install dependencies via {unit.package_manager or 'pip'}"
                 + (f" using lockfile {unit.lockfile}" if unit.lockfile else "") + "."]
        hint = None
        if fw == "Django":
            mod = next((e.split(":")[0] for e in unit.entry_points if "wsgi" in e), "<project>.wsgi")
            hint = f"gunicorn {mod}:application --bind 0.0.0.0:{port or 8000}"
        elif fw == "FastAPI" and entry and ":" in entry:
            hint = f"uvicorn {entry} --host 0.0.0.0 --port {port or 8000}"
        elif fw == "Flask" and entry and ":" in entry:
            hint = f"gunicorn --bind 0.0.0.0:{port or 5000} {entry}"
        elif entry:
            hint = f"python {entry}"
        return Strategy(base_image=f"python:{ver}-slim", multi_stage=False, start_command_hint=hint,
                        notes=notes + ["Set PYTHONDONTWRITEBYTECODE=1 and PYTHONUNBUFFERED=1."])

    if lang == "JavaScript":
        ver = pick_node_version(unit.runtime_version)
        pm = unit.package_manager or "npm"
        install = {"npm": "npm ci", "yarn": "yarn install --frozen-lockfile", "pnpm": "pnpm install --frozen-lockfile",
                   "bun": "bun install --frozen-lockfile"}.get(pm, "npm install")
        if not unit.lockfile and pm == "npm":
            install = "npm install"
        notes = [f"Package manager: {pm}; reproducible install: `{install}`."]
        if pm in ("yarn", "pnpm"):
            notes.append(f"Enable via `corepack enable` before using {pm}.")
        if unit.role == "frontend":
            return Strategy(base_image=f"node:{ver}-alpine", runtime_image="nginx:1.27-alpine", multi_stage=True,
                            start_command_hint="nginx -g 'daemon off;'",
                            notes=notes + [f"Copy /{unit.build_output_dir or 'dist'} into /usr/share/nginx/html; "
                                           "add an SPA fallback (try_files) nginx config."])
        if fw == "Next.js":
            return Strategy(base_image=f"node:{ver}-alpine", multi_stage=True,
                            start_command_hint=unit.scripts.get("start", "npm run start"),
                            notes=notes + ["Build with `next build`; run as non-root user."])
        hint = unit.scripts.get("start") and "npm start" or (f"node {entry}" if entry else None)
        return Strategy(base_image=f"node:{ver}-alpine", multi_stage=fw in ("NestJS",) or "typescript" in
                        " ".join(unit.dev_dependencies).lower(), start_command_hint=hint,
                        notes=notes + ["Install production dependencies only in the final image (NODE_ENV=production)."])

    if lang == "Go":
        ver = _major_minor(unit.runtime_version) or "1.23"
        return Strategy(base_image=f"golang:{ver}-alpine", runtime_image="alpine:3.22", multi_stage=True,
                        start_command_hint=f"./app" + (f" (listens on {port})" if port else ""),
                        notes=["Build a static binary with CGO_ENABLED=0 unless CGO deps are used.",
                               "Copy go.mod/go.sum first and run `go mod download` for layer caching."])

    if lang == "Java":
        ver = pick_java_version(unit.runtime_version)
        if unit.package_manager == "gradle":
            build = f"gradle:8-jdk{ver}"
            cmd = "gradle bootJar --no-daemon -x test" if fw == "Spring Boot" else "gradle build --no-daemon -x test"
        else:
            build = f"maven:3.9-eclipse-temurin-{ver}"
            cmd = "mvn -B -DskipTests package"
        return Strategy(base_image=build, runtime_image=f"eclipse-temurin:{ver}-jre", multi_stage=True,
                        start_command_hint="java -jar app.jar",
                        notes=[f"Build with `{cmd}`; copy only the built jar to the JRE runtime image.",
                               "Resolve dependencies before copying sources for caching (e.g. mvn dependency:go-offline)."])

    return Strategy(base_image="debian:bookworm-slim", notes=["Language not recognized; choose the image from context."])

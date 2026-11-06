"""Deterministic Docker Compose generation from a ProjectAnalysis (no LLM involved, so it is reproducible).

Safety rules enforced here and re-checked by ``validate_compose``:
  * no real passwords/API keys: secrets are ``${VAR:?message}`` references and ``.env.example`` holds empty values;
  * databases/caches are NOT published to the host and live on an internal network;
  * app ports bind to 127.0.0.1 by default;
  * no privileged mode, host networking, docker.sock mounts, or ``compose up`` (we never run the stack).
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Optional

from pydantic import BaseModel, Field

from app.analyzer.models import ProjectAnalysis, ProjectUnit
from app.docker.docker_validator import exposed_ports
from app.utils import yamlish
from app.utils.credcheck import has_embedded_credential
from app.utils.security import ensure_within, redact_secrets

SECRET_HINT = re.compile(r"(?i)(password|passwd|secret|token|api[_-]?key|private[_-]?key|credential)")
COMPOSE_NAME = "docker-compose.yml"
ENV_EXAMPLE_NAME = ".env.example"

INFRA = {
    "postgres": {"image": "postgres:16-alpine", "port": 5432, "volume": "/var/lib/postgresql/data",
                 "env": {"POSTGRES_USER": "${POSTGRES_USER:-app}", "POSTGRES_DB": "${POSTGRES_DB:-app}",
                            "POSTGRES_PASSWORD": "${POSTGRES_PASSWORD:?Set POSTGRES_PASSWORD in .env}"},
                 "health": ["CMD-SHELL", "pg_isready -U $${POSTGRES_USER} -d $${POSTGRES_DB}"]},
    "mysql": {"image": "mysql:8.4", "port": 3306, "volume": "/var/lib/mysql",
              "env": {"MYSQL_DATABASE": "${MYSQL_DATABASE:-app}", "MYSQL_USER": "${MYSQL_USER:-app}",
                         "MYSQL_PASSWORD": "${MYSQL_PASSWORD:?Set MYSQL_PASSWORD in .env}",
                         "MYSQL_ROOT_PASSWORD": "${MYSQL_ROOT_PASSWORD:?Set MYSQL_ROOT_PASSWORD in .env}"},
              "health": ["CMD-SHELL", "mysqladmin ping -h 127.0.0.1 -u root -p$${MYSQL_ROOT_PASSWORD} --silent"]},
    "redis": {"image": "redis:7-alpine", "port": 6379, "volume": "/data", "env": {},
              "health": ["CMD", "redis-cli", "ping"]},
    "mongo": {"image": "mongo:7", "port": 27017, "volume": "/data/db", "env": {},
              "health": ["CMD-SHELL", "mongosh --quiet --eval \"db.adminCommand('ping')\""]},
}
ALIASES = {"postgresql": "postgres", "pg": "postgres", "mariadb": "mysql", "mongodb": "mongo"}


class ComposeResult(BaseModel):
    compose: str
    env_example: str
    services: list[str] = Field(default_factory=list)
    missing_dockerfiles: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ComposeValidation(BaseModel):
    ok: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    docker_compose_checked: bool = False


def _svc_name(text: str) -> str:
    n = re.sub(r"[^a-z0-9_-]+", "-", text.lower()).strip("-_")
    return n or "app"


def _tcp_healthcheck(unit: ProjectUnit, port: int) -> Optional[dict]:
    """TCP-connect probe using only the language runtime already in the image (no curl assumed)."""
    if unit.language == "Python":
        cmd = ["CMD", "python", "-c", f"import socket;socket.create_connection(('127.0.0.1',{port}),2)"]
    elif unit.language == "JavaScript" and unit.role != "frontend":
        cmd = ["CMD", "node", "-e", f"require('net').connect({port},'127.0.0.1').on('connect',()=>process.exit(0))"
                                    ".on('error',()=>process.exit(1))"]
    else:
        return None
    return {"test": cmd, "interval": "15s", "timeout": "5s", "retries": 5, "start_period": "20s"}


def _infra_kinds(analysis: ProjectAnalysis) -> list[str]:
    kinds = []
    for name in analysis.services:
        k = ALIASES.get(name.lower(), name.lower())
        if k in INFRA and k not in kinds:
            kinds.append(k)
    return kinds


def _derived_env(name: str, kinds: list[str]) -> Optional[str]:
    """Wire well-known connection variables to the generated infra services (inferred; flagged in notes)."""
    n = name.upper()
    if n in ("DATABASE_URL", "DB_URL", "SQLALCHEMY_DATABASE_URI"):
        if "postgres" in kinds:
            return "postgresql://${POSTGRES_USER:-app}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB:-app}"
        if "mysql" in kinds:
            return "mysql://${MYSQL_USER:-app}:${MYSQL_PASSWORD}@mysql:3306/${MYSQL_DATABASE:-app}"
    if n in ("REDIS_URL", "CACHE_URL") and "redis" in kinds:
        return "redis://redis:6379/0"
    if n == "REDIS_HOST" and "redis" in kinds:
        return "redis"
    if n in ("MONGO_URL", "MONGODB_URI", "MONGO_URI") and "mongo" in kinds:
        return "mongodb://mongo:27017/app"
    if n in ("DB_HOST", "DATABASE_HOST"):
        return "postgres" if "postgres" in kinds else "mysql" if "mysql" in kinds else None
    return None


def generate_compose(analysis: ProjectAnalysis, project_dir: Path | str | None = None) -> ComposeResult:
    root = Path(project_dir or analysis.root)
    notes: list[str] = []
    warnings: list[str] = []
    kinds = _infra_kinds(analysis)
    units = [u for u in analysis.units if not u.workspace_root and u.role in ("backend", "frontend", "fullstack", "unknown")
             and u.language != "unknown"]
    if not units and analysis.primary_unit:
        units = [analysis.primary_unit]
    if not units:
        warnings.append("No runnable application component was detected; only infrastructure services are generated.")

    services: dict = {}
    env_lines: dict[str, str] = {}
    volumes: dict = {}
    host_ports_used: set[int] = set()
    missing: list[str] = []
    multi = len(units) > 1

    for u in units:
        base = u.path if u.path not in (".", "") else analysis.name
        name = _svc_name(base if multi or u.path in (".", "") else base)
        name = name if name not in services else f"{name}-{len(services)}"
        context = u.path if u.path not in (".", "") else "."
        if not (root / u.path / "Dockerfile").is_file():
            missing.append(u.path or ".")
            warnings.append(f"No Dockerfile in '{u.path or '.'}': generate one first (compose --generate-dockerfiles) "
                            "or `docker compose build` will fail.")
        svc: dict = {"build": {"context": context, "dockerfile": "Dockerfile"}, "restart": "unless-stopped"}
        port = u.ports[0] if u.ports else None
        port_from_source = bool(port)
        df_file = root / u.path / "Dockerfile"
        if df_file.is_file() and not df_file.is_symlink() and df_file.stat().st_size < 200_000:
            exposed = exposed_ports(df_file.read_text(encoding="utf-8", errors="replace"))
            if exposed:  # the Dockerfile decides what the image listens on
                if port and port != exposed[0]:
                    warnings.append(f"{name}: the source suggests port {port} but the Dockerfile EXPOSEs {exposed[0]}; "
                                    f"using {exposed[0]}. Check which one the app really binds.")
                port, port_from_source = exposed[0], False
                notes.append(f"{name}: container port {port} taken from the Dockerfile's EXPOSE.")
        if port:
            var = f"{name.upper().replace('-', '_')}_HOST_PORT"
            host = port
            while host in host_ports_used:
                host += 1
            host_ports_used.add(host)
            svc["ports"] = [f"127.0.0.1:${{{var}:-{host}}}:{port}"]
            env_lines[var] = str(host)
            if port_from_source and u.inferred and any(str(port) in x for x in u.inferred):
                notes.append(f"{name}: container port {port} is a framework default, not read from the source.")
            hc = _tcp_healthcheck(u, port)
            if hc:
                svc["healthcheck"] = hc
            else:
                notes.append(f"{name}: no healthcheck generated (no safe probe available in a {u.language} image).")
        else:
            warnings.append(f"{name}: no listening port detected, so none is published.")
        env: dict = {}
        for var in analysis.env_vars:
            derived = _derived_env(var, kinds)
            if derived is not None:
                env[var] = derived
                notes.append(f"{name}: {var} is wired to the generated service (inferred from its name).")
            elif SECRET_HINT.search(var):
                env[var] = f"${{{var}:?Set {var} in .env}}"
                env_lines.setdefault(var, "")
            else:
                env[var] = None  # pass through only when set (.env/shell); unset => the app's own default applies
                env_lines.setdefault(var, "")
        if env:
            svc["environment"] = env
        depends = {k: {"condition": "service_healthy"} for k in kinds}
        if depends and u.role != "frontend":
            svc["depends_on"] = depends
        svc["networks"] = ["app_net"] + (["data_net"] if kinds and u.role != "frontend" else [])
        services[name] = svc

    for k in kinds:
        spec = INFRA[k]
        vol = f"{k}_data"
        svc = {"image": spec["image"], "restart": "unless-stopped", "volumes": [f"{vol}:{spec['volume']}"],
               "healthcheck": {"test": spec["health"], "interval": "10s", "timeout": "5s", "retries": 5,
                                "start_period": "20s"},
               "networks": ["data_net"]}
        if spec["env"]:
            svc["environment"] = dict(spec["env"])
            for m in re.findall(r"\$\{([A-Z_]+)", " ".join(spec["env"].values())):
                env_lines.setdefault(m, "")
        services[k] = svc
        volumes[vol] = {}
        notes.append(f"{k}: not published to the host and attached only to the internal data_net network.")
    if "redis" in kinds:
        notes.append("redis runs without authentication on an internal network; add a password for shared environments.")
    for k, ev in analysis.services.items():
        if ALIASES.get(k.lower(), k.lower()) not in INFRA:
            warnings.append(f"Detected service '{k}' ({ev}) has no Compose template and was not generated.")
    if analysis.existing_compose:
        warnings.append("The project already has Compose file(s): " + ", ".join(analysis.existing_compose) +
                        ". The generated file will not overwrite them without approval.")
    if not services:
        raise ValueError("Nothing to generate: no application components or infrastructure services detected.")

    doc: dict = {"services": services,
                 "networks": {"app_net": {}, **({"data_net": {"internal": True}} if kinds else {})}}
    if volumes:
        doc["volumes"] = volumes
    header = ("# Generated by the AI DevOps Agent. Review before use.\n"
              "# Secrets are NOT stored here: copy .env.example to .env and fill in the values (keep .env out of git).\n")
    env_text = ("# Copy to .env and fill in. Never commit .env. Empty values below are required secrets or optional "
                "settings.\n" + "\n".join(f"{k}={v}" for k, v in sorted(env_lines.items())) + "\n")
    notes.append("Run `docker compose up --build` yourself after reviewing; this tool never starts the stack.")
    return ComposeResult(compose=header + yamlish.dump(doc), env_example=env_text, services=list(services),
                         missing_dockerfiles=missing, notes=notes, warnings=warnings)


# ----------------------------------------------------------------------------- validation
def _parse_back(text: str) -> dict:
    import yaml  # optional; used for validation only when installed
    return yaml.safe_load(text)


def validate_compose(compose: str, env_example: str = "", *, runner: Callable[..., "subprocess.CompletedProcess[str]"] = subprocess.run,
                     use_docker: bool = True, which: Callable[[str], Optional[str]] = lambda n: __import__("shutil").which(n)
                     ) -> ComposeValidation:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        doc = _parse_back(compose)
    except ImportError:
        return ComposeValidation(ok=True, warnings=["PyYAML not installed: structural validation skipped."])
    except Exception as exc:  # yaml.YAMLError
        return ComposeValidation(ok=False, errors=[f"Not valid YAML: {exc}"])
    if not isinstance(doc, dict) or not isinstance(doc.get("services"), dict) or not doc["services"]:
        return ComposeValidation(ok=False, errors=["No services defined."])
    services = doc["services"]
    nets, vols = set(doc.get("networks") or {}), set(doc.get("volumes") or {})
    for name, s in services.items():
        if not isinstance(s, dict) or not (s.get("image") or s.get("build")):
            errors.append(f"{name}: needs 'image' or 'build'.")
            continue
        for dep in (s.get("depends_on") or {}):
            if dep not in services:
                errors.append(f"{name}: depends_on unknown service '{dep}'.")
        for n in s.get("networks") or []:
            if n not in nets:
                errors.append(f"{name}: uses undefined network '{n}'.")
        for v in s.get("volumes") or []:
            src = str(v).split(":")[0]
            if src.startswith(("/", ".", "~")) or "docker.sock" in str(v):
                errors.append(f"{name}: host path mount '{v}' is not allowed.")
            elif src not in vols:
                errors.append(f"{name}: uses undefined volume '{src}'.")
        if s.get("privileged"):
            errors.append(f"{name}: privileged mode is not allowed.")
        if s.get("network_mode") == "host":
            errors.append(f"{name}: host networking is not allowed.")
        for key, val in (s.get("environment") or {}).items():
            sval = str(val)
            if SECRET_HINT.search(str(key)) and sval and not sval.startswith("${"):
                errors.append(f"{name}: {key} holds a literal value; use a ${{VAR}} reference.")
        for p in s.get("ports") or []:
            if not str(p).startswith("127.0.0.1:"):
                warnings.append(f"{name}: port '{p}' is published on all interfaces.")
    if has_embedded_credential(doc):
        errors.append("A value looks like an embedded credential (token or user:password in a URL).")
    for line in env_example.splitlines():
        if "=" in line and not line.startswith("#") and SECRET_HINT.search(line.split("=")[0]) \
                and line.split("=", 1)[1].strip():
            errors.append(f".env.example must not contain secret values ({line.split('=')[0]}).")
    checked = False
    if use_docker and not errors and which("docker"):
        with tempfile.TemporaryDirectory() as td:
            f, e = Path(td, "compose.yml"), Path(td, "dummy.env")
            f.write_text(compose, encoding="utf-8")
            # only variables WITHOUT a default need a dummy value; defaults are validated as written
            names = set(re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::\?[^}]*)?\}", compose))
            e.write_text("".join(f"{n}=x\n" for n in sorted(names)), encoding="utf-8")
            try:
                p = runner(["docker", "compose", "--env-file", str(e), "-f", str(f), "config", "-q"],
                           capture_output=True, text=True, timeout=30, cwd=td)
                if p.returncode != 0 and "not a docker command" in (p.stderr or ""):
                    warnings.append("The Docker Compose plugin is not installed; `docker compose config` was skipped.")
                else:
                    checked = True
                    if p.returncode != 0:
                        errors.append("docker compose config: " + redact_secrets((p.stderr or "").strip()[-300:]))
            except (OSError, subprocess.TimeoutExpired):
                warnings.append("docker compose config could not be run.")
    return ComposeValidation(ok=not errors, errors=errors, warnings=warnings, docker_compose_checked=checked)


# ----------------------------------------------------------------------------- writing
class ComposeProtectionError(PermissionError):
    pass


def write_compose(project_dir: Path | str, result: ComposeResult, *, overwrite: bool = False,
                  filename: str = COMPOSE_NAME) -> list[Path]:
    """Write the Compose file and .env.example (never .env). Existing files need ``overwrite=True``."""
    root = Path(project_dir).resolve()
    if filename != Path(filename).name:
        raise PermissionError("filename must be a plain file name")
    plan = [(filename, result.compose), (ENV_EXAMPLE_NAME, result.env_example)]
    clashes = [n for n, _ in plan if (root / n).exists() or (root / n).is_symlink()]
    if clashes and not overwrite:
        raise ComposeProtectionError("Refusing to overwrite existing file(s) without approval: " + ", ".join(clashes))
    written = []
    for name, content in plan:
        target = root / name
        if target.is_symlink():
            raise PermissionError(f"Refusing to write through symlink: {target}")
        ensure_within(root, target)
        if target.exists():
            target.with_name(name + ".bak").write_text(target.read_text(encoding="utf-8"), encoding="utf-8")
        tmp = root / f".{name}.tmp"
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, target)
        written.append(target)
    return written


def generate_unit_dockerfiles(orch, analysis: ProjectAnalysis, project_dir: Path | str, *,
                              approve_overwrite: bool = False) -> dict[str, str]:
    """Optional: ask the local LLM for a Dockerfile for each unit that has none. Returns path -> status."""
    from app.docker.dockerfile_generator import FileProtectionError

    root = Path(project_dir)
    out: dict[str, str] = {}
    for u in analysis.units:
        if u.workspace_root or u.role in ("library", "cli"):
            continue
        unit_dir = root / u.path
        if (unit_dir / "Dockerfile").is_file() and not approve_overwrite:
            out[u.path] = "exists"
            continue
        try:
            sub = orch.analyze(unit_dir)
            art = orch.generate(sub)
            orch.save(unit_dir, art, approve_overwrite=approve_overwrite)
            out[u.path] = "generated"
        except FileProtectionError as exc:
            out[u.path] = f"protected: {exc}"
        except Exception as exc:  # LLM/validation failures are reported per unit
            out[u.path] = f"failed: {exc}"
    return out

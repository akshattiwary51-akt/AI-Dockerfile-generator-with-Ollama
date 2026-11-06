"""Image size/cache optimization.

MEASURED values come only from `docker image inspect` / `docker history` of images that exist locally.
ESTIMATED values come from a static table of typical base-image sizes and are always labelled as estimates.
Nothing here invents a size reduction: the final before/after comparison uses two measured sizes only.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Callable, Optional

from pydantic import BaseModel, Field

from app.docker.docker_validator import parse_instructions

Runner = Callable[..., "subprocess.CompletedProcess[str]"]
MB = 1_000_000

# ESTIMATE ONLY: approximate uncompressed sizes in MB of common base images (vary by tag/arch/date).
BASE_SIZE_ESTIMATE_MB = {
    "python": {"full": 1000, "slim": 130, "alpine": 60}, "node": {"full": 1100, "slim": 230, "alpine": 160},
    "golang": {"full": 800, "alpine": 250}, "ruby": {"full": 900, "slim": 220, "alpine": 100},
    "php": {"full": 500, "alpine": 80}, "eclipse-temurin": {"full": 450, "alpine": 200},
}


class LayerInfo(BaseModel):
    created_by: str
    size_bytes: int


class SizeMeasurement(BaseModel):
    image: str
    available: bool
    size_bytes: Optional[int] = None
    layers: list[LayerInfo] = Field(default_factory=list)
    error: str = ""

    @property
    def size_mb(self) -> Optional[float]:
        return None if self.size_bytes is None else round(self.size_bytes / MB, 1)


class Suggestion(BaseModel):
    code: str
    message: str
    estimated_saving_mb: Optional[int] = None  # ESTIMATE, from the static table; None when unknown
    line: Optional[int] = None


class OptimizationReport(BaseModel):
    measured: Optional[SizeMeasurement] = None
    suggestions: list[Suggestion] = Field(default_factory=list)
    largest_layers: list[LayerInfo] = Field(default_factory=list)
    note: str = ("Measured sizes come from the local Docker daemon. 'Estimated' savings are rough figures from a "
                 "static table and are NOT guaranteed; rebuild and compare measured sizes.")


class Comparison(BaseModel):
    before: SizeMeasurement
    after: SizeMeasurement

    @property
    def valid(self) -> bool:
        return self.before.size_bytes is not None and self.after.size_bytes is not None

    @property
    def delta_bytes(self) -> Optional[int]:
        return self.after.size_bytes - self.before.size_bytes if self.valid else None  # type: ignore[operator]

    @property
    def percent(self) -> Optional[float]:
        return round(100 * self.delta_bytes / self.before.size_bytes, 1) if self.valid and self.before.size_bytes else None

    def summary(self) -> str:
        if not self.valid:
            return "Comparison unavailable: both images must be built locally to be measured."
        d = self.delta_bytes
        word = "smaller" if d < 0 else "larger" if d > 0 else "unchanged"  # type: ignore[operator]
        return (f"Measured: {self.before.size_mb} MB -> {self.after.size_mb} MB "
                f"({abs(self.percent or 0)}% {word}).")


def _parse_size(text: str) -> int:
    m = re.match(r"\s*([\d.]+)\s*([kKMG]?B)", text)
    if not m:
        return 0
    return int(float(m.group(1)) * {"B": 1, "KB": 1000, "kB": 1000, "MB": MB, "GB": 1000 * MB}[m.group(2)])


def measure_image(image: str, *, runner: Runner = subprocess.run) -> SizeMeasurement:
    try:
        p = runner(["docker", "image", "inspect", "--format", "{{.Size}}", image], capture_output=True, text=True,
                   timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return SizeMeasurement(image=image, available=False, error=str(exc))
    if p.returncode != 0:
        return SizeMeasurement(image=image, available=False, error=(p.stderr or "image not found").strip()[:300])
    try:
        size = int(p.stdout.strip())
    except ValueError:
        return SizeMeasurement(image=image, available=False, error="unexpected docker output")
    layers: list[LayerInfo] = []
    try:
        h = runner(["docker", "history", "--no-trunc", "--format", "{{json .}}", image], capture_output=True,
                   text=True, timeout=30)
        if h.returncode == 0:
            for line in h.stdout.splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                layers.append(LayerInfo(created_by=str(row.get("CreatedBy", ""))[:200],
                                        size_bytes=_parse_size(str(row.get("Size", "0B")))))
    except (OSError, subprocess.TimeoutExpired):
        pass
    return SizeMeasurement(image=image, available=True, size_bytes=size, layers=layers)


def analyze_optimizations(dockerfile: str, dockerignore: str = "", measured: Optional[SizeMeasurement] = None,
                          project_root: Optional[Path | str] = None) -> OptimizationReport:
    rep = OptimizationReport(measured=measured)
    if measured and measured.available:
        rep.largest_layers = sorted(measured.layers, key=lambda l: -l.size_bytes)[:5]
    instrs, _ = parse_instructions(dockerfile)
    froms = [i for i in instrs if i.name == "FROM"]
    add = lambda **k: rep.suggestions.append(Suggestion(**k))  # noqa: E731
    if froms:
        last = froms[-1]
        image = last.args.split()[0] if last.args.split() else ""
        repo = image.split(":")[0].split("/")[-1].lower()
        tag = image.split(":")[1].lower() if ":" in image else ""
        table = BASE_SIZE_ESTIMATE_MB.get(repo)
        if table and "slim" not in tag and "alpine" not in tag:
            saving = table["full"] - table.get("slim", table.get("alpine", table["full"]))
            add(code="slim-base", message=f"'{image}' is a full image; a -slim variant is typically much smaller "
                f"(estimate, not measured).", estimated_saving_mb=saving, line=last.line)
        if table and "alpine" in tag and repo in ("python", "node"):
            add(code="alpine-caveat", message="Alpine uses musl libc; some Python wheels/native modules need compiling "
                "and may be slower or larger. Measure before assuming it helps.", line=last.line)
        if len(froms) == 1 and any(re.search(r"\b(gcc|g\+\+|build-essential|make|cmake)\b", i.args) for i in instrs
                                   if i.name == "RUN"):
            add(code="multi-stage", message="Build tools are installed in the only stage; a multi-stage build keeps "
                "compilers out of the runtime image.", line=last.line)
    for i in instrs:
        if i.name != "RUN":
            continue
        a = i.args
        if re.search(r"apt-get install", a) and "rm -rf /var/lib/apt/lists" not in a:
            add(code="apt-cache", message="Remove /var/lib/apt/lists in the same RUN as apt-get install.", line=i.line)
        if re.search(r"apt-get install", a) and "--no-install-recommends" not in a:
            add(code="apt-recommends", message="Use --no-install-recommends to skip optional packages.", line=i.line)
        if re.search(r"\bpip3? install\b", a) and "--no-cache-dir" not in a:
            add(code="pip-cache", message="Use pip install --no-cache-dir to avoid storing the wheel cache.", line=i.line)
        if re.search(r"\bnpm (install|i)\b", a) and "npm ci" not in a:
            add(code="npm-ci", message="Prefer `npm ci --omit=dev` for smaller, reproducible installs.", line=i.line)
        if re.search(r"apk add", a) and "--no-cache" not in a:
            add(code="apk-cache", message="Use apk add --no-cache.", line=i.line)
    # layer cache: dependency manifest copied together with source
    copies = [i for i in instrs if i.name == "COPY"]
    for idx, c in enumerate(copies):
        if re.match(r"(--\S+\s+)*\.\s+\S+", c.args) and not any(
                re.search(r"(requirements|package(-lock)?\.json|go\.(mod|sum)|pom\.xml|composer\.json|Gemfile|pyproject)",
                          x.args) for x in copies[:idx]):
            add(code="cache-order", message="Source is copied before the dependency manifest, so every code change "
                "invalidates the dependency-install layer. Copy the manifest first, install, then copy the source.",
                line=c.line)
            break
    if not dockerignore.strip():
        add(code="no-dockerignore", message="No .dockerignore: the whole directory is sent as build context.")
    elif project_root:
        big = [d for d in ("node_modules", ".git", "venv", ".venv", "__pycache__", "dist", "build")
               if (Path(project_root) / d).exists() and d not in dockerignore]
        if big:
            add(code="context-bloat", message="Large directories are not in .dockerignore: " + ", ".join(big) + ".")
    return rep


def optimize_dockerfile(dockerfile: str) -> str:
    """Deterministic, conservative rewrites only (no behaviour change): pip/apk cache flags and apt recommends."""
    out = []
    for line in dockerfile.splitlines():
        if re.match(r"\s*RUN\b", line):
            if re.search(r"\bpip3? install\b", line) and "--no-cache-dir" not in line:
                line = re.sub(r"\b(pip3? install)\b", r"\1 --no-cache-dir", line, count=1)
            if re.search(r"\bapk add\b", line) and "--no-cache" not in line:
                line = re.sub(r"\b(apk add)\b", r"\1 --no-cache", line, count=1)
            if re.search(r"apt-get install\b", line) and "--no-install-recommends" not in line and line.rstrip().endswith("\\") is False:
                line = re.sub(r"(apt-get install)\b", r"\1 --no-install-recommends", line, count=1)
        out.append(line)
    return "\n".join(out) + ("\n" if dockerfile.endswith("\n") else "")


def write_optimized(project_dir: Path | str, dockerfile: str, name: str = "Dockerfile.optimized") -> Path:
    """Write the variant next to the original; never overwrites Dockerfile itself."""
    root = Path(project_dir).resolve()
    if name == "Dockerfile" or name != Path(name).name:
        raise PermissionError("The optimized variant must not overwrite Dockerfile")
    target = root / name
    if target.is_symlink():
        raise PermissionError(f"Refusing to write through symlink: {target}")
    target.write_text(optimize_dockerfile(dockerfile), encoding="utf-8")
    return target


def build_and_compare(builder, project_dir: Path | str, before_image: str, after_image: str, *,
                      runner: Runner = subprocess.run, approved: bool = False):
    """Build Dockerfile.optimized as ``after_image`` and compare measured sizes. Requires explicit approval."""
    if not approved:
        raise PermissionError("Building the optimized image requires explicit approval.")
    result = builder.build(project_dir, after_image, dockerfile="Dockerfile.optimized")
    before = measure_image(before_image, runner=runner)
    after = measure_image(after_image, runner=runner) if result.success else SizeMeasurement(
        image=after_image, available=False, error="optimized build failed")
    return result, Comparison(before=before, after=after)

"""Regenerate samples/output/python-flask/ (docker-compose.yml, .env.example, k8s/*) WITHOUT an LLM or Docker.

Run:  python samples/generate_deployment_samples.py
The files are produced by the same deterministic generators the CLI (`compose`, `k8s`) and dashboard use.
"""
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.analyzer.project_scanner import ProjectScanner  # noqa: E402
from app.compose.generator import generate_compose, validate_compose, write_compose  # noqa: E402
from app.kubernetes.generator import (generate_manifests, load_unit_dockerfiles, validate_manifests,  # noqa: E402
                                      write_manifests)
from app.utils.config import Settings  # noqa: E402

SRC = ROOT / "samples" / "python-flask"
OUT = ROOT / "samples" / "output" / "python-flask"

with tempfile.TemporaryDirectory() as tmp:
    proj = Path(tmp) / "flask-shop"
    shutil.copytree(SRC, proj)
    analysis = ProjectScanner(Settings()).scan(proj)
    compose = generate_compose(analysis, proj)
    cv = validate_compose(compose.compose, compose.env_example)
    dockerfiles = load_unit_dockerfiles(analysis, proj)
    k8s = generate_manifests(analysis, image="registry.example.com/flask-shop:1.0.0", dockerfile=dockerfiles.get(".", ""),
                             units_dockerfiles=dockerfiles, ingress_host="shop.example.com", health_path="/")
    kv = validate_manifests(k8s.docs, yaml_text=k8s.combined)
    print(f"compose validation ok={cv.ok} (docker compose config checked={cv.docker_compose_checked})")
    print(f"k8s validation ok={kv.ok} (kubectl dry-run checked={kv.kubectl_checked})")
    if not (cv.ok and kv.ok):
        sys.exit(1)
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    write_compose(OUT, compose)
    write_manifests(OUT, k8s)
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            print("wrote", p.relative_to(ROOT))

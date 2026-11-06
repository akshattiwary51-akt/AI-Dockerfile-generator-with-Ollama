import json
import subprocess

import pytest

from app.optimization.optimizer import (Comparison, SizeMeasurement, analyze_optimizations, build_and_compare,
                                        measure_image, optimize_dockerfile, write_optimized)
from tests.conftest import FakeBuilder, failed_build, ok_build

DF = """FROM python:3.12
COPY . .
RUN apt-get update && apt-get install -y gcc
RUN pip install -r requirements.txt
CMD ["python","a.py"]
"""


def fake_runner(size="500000000"):
    def run(cmd, **k):
        if "inspect" in cmd:
            return subprocess.CompletedProcess(cmd, 0, size + "\n", "")
        rows = [json.dumps({"CreatedBy": "RUN pip install", "Size": "300MB"}), json.dumps({"CreatedBy": "CMD", "Size": "0B"})]
        return subprocess.CompletedProcess(cmd, 0, "\n".join(rows), "")
    return run


def test_measure_and_layers():
    m = measure_image("x", runner=fake_runner())
    assert m.size_mb == 500.0 and m.layers[0].size_bytes == 300_000_000


def test_measure_missing_image():
    r = lambda cmd, **k: subprocess.CompletedProcess(cmd, 1, "", "No such image")  # noqa: E731
    m = measure_image("x", runner=r)
    assert not m.available and m.size_bytes is None


def test_suggestions_label_estimates():
    rep = analyze_optimizations(DF, "")
    codes = {s.code for s in rep.suggestions}
    assert {"slim-base", "multi-stage", "apt-cache", "pip-cache", "cache-order", "no-dockerignore"} <= codes
    slim = next(s for s in rep.suggestions if s.code == "slim-base")
    assert slim.estimated_saving_mb and "estimate" in slim.message and "NOT guaranteed" in rep.note


def test_no_invented_saving_for_unknown_base():
    rep = analyze_optimizations("FROM mycorp/base:1\nUSER 1000\n", ".git\n")
    assert all(s.estimated_saving_mb is None for s in rep.suggestions)


def test_optimize_is_conservative():
    out = optimize_dockerfile(DF)
    assert "pip install --no-cache-dir -r" in out and "FROM python:3.12" in out and out.count("\n") == DF.count("\n")


def test_write_optimized_never_overwrites_dockerfile(tmp_path):
    (tmp_path / "Dockerfile").write_text(DF)
    write_optimized(tmp_path, DF)
    assert (tmp_path / "Dockerfile").read_text() == DF and (tmp_path / "Dockerfile.optimized").exists()
    with pytest.raises(PermissionError):
        write_optimized(tmp_path, DF, "Dockerfile")


def test_comparison_uses_measured_only(tmp_path):
    (tmp_path / "Dockerfile.optimized").write_text("FROM x\n")
    b = FakeBuilder([ok_build()])
    with pytest.raises(PermissionError):
        build_and_compare(b, tmp_path, "a", "b")
    res, cmp = build_and_compare(b, tmp_path, "a", "b", runner=fake_runner(), approved=True)
    assert cmp.valid and cmp.delta_bytes == 0 and "unchanged" in cmp.summary()


def test_failed_optimized_build_gives_no_comparison(tmp_path):
    (tmp_path / "Dockerfile.optimized").write_text("FROM x\n")
    _, cmp = build_and_compare(FakeBuilder([failed_build()]), tmp_path, "a", "b", runner=fake_runner(), approved=True)
    assert not cmp.valid and "unavailable" in cmp.summary()


def test_percent():
    c = Comparison(before=SizeMeasurement(image="a", available=True, size_bytes=1000),
                   after=SizeMeasurement(image="b", available=True, size_bytes=600))
    assert c.percent == -40.0 and "smaller" in c.summary()


def test_docker_builder_alternate_dockerfile(make_project, settings, popen_factory):
    from app.docker.docker_builder import DockerBuilder
    from tests.conftest import FakeProc
    root = make_project({"Dockerfile": "FROM x\n", "Dockerfile.optimized": "FROM x\n"})
    popen = popen_factory(FakeProc())
    DockerBuilder(settings, popen=popen).build(root, "p:opt", dockerfile="Dockerfile.optimized")
    assert popen.calls[0]["cmd"] == ["docker", "build", "--progress=plain", "-t", "p:opt", "-f", "Dockerfile.optimized", "."]
    with pytest.raises(ValueError):
        DockerBuilder(settings, popen=popen).build(root, "p:x", dockerfile="../evil")

"""Offline demo: real analyzer/validator/feedback loop with a scripted LLM and a simulated Docker build.

The first Dockerfile fails (a C extension needs gcc); the "LLM" diagnoses it and the loop applies a targeted fix.
Run:  python samples/demo_end_to_end.py
"""
import json, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.agent.orchestrator import Orchestrator
from app.docker.docker_builder import BuildResult
from app.utils.config import Settings

DF1 = (Path(__file__).parent / "python-flask" / "Dockerfile").read_text()
DF2 = DF1.replace("RUN pip install --no-cache-dir -r requirements.txt",
                  "RUN apt-get update && apt-get install -y --no-install-recommends gcc \\\n"
                  "    && rm -rf /var/lib/apt/lists/* \\\n    && pip install --no-cache-dir -r requirements.txt")
IGN = (Path(__file__).parent / "python-flask" / ".dockerignore").read_text()
base = dict(language="Python", framework="Flask", base_image="python:3.12-slim", exposed_port=5050,
            explanation="Slim image, cached dependency layer, non-root user.", limitations=[])


class DemoLLM:
    model, host = "demo-model", "scripted"
    def __init__(self): self.n = 0
    def ensure_ready(self, model=None): pass
    def chat(self, messages, *, schema=None, model=None):
        self.n += 1
        if self.n == 1:
            return json.dumps({**base, "dockerfile": DF1, "dockerignore": IGN})
        return json.dumps({**base, "dockerfile": DF2, "dockerignore": IGN, "recoverable": True,
                           "root_cause": "mypkg builds a C extension; python:3.12-slim has no compiler (gcc not found).",
                           "changes": ["Install gcc in the same RUN layer as pip install and clean apt lists"]})


class DemoDocker:
    n = 0
    def check_docker(self): return True, "simulated"
    def _run(self, *a, **k): return type("P", (), {"returncode": 1, "stdout": "", "stderr": ""})()
    def build(self, project_dir, image, *, on_output=None, cancel_event=None):
        self.n += 1
        if "gcc" not in (Path(project_dir) / "Dockerfile").read_text():
            err = "#6 [3/6] RUN pip install ...\n#6 ERROR: Failed building wheel for mypkg\n#6 gcc: command not found"
            if on_output: [on_output(l) for l in err.splitlines()]
            return BuildResult(success=False, image=image, exit_code=1, stderr=err, failed_stage="[3/6] RUN pip install",
                               error_summary=err, duration_seconds=4.2)
        if on_output: on_output("#6 DONE\n#7 exporting to image")
        return BuildResult(success=True, image=image, exit_code=0, duration_seconds=11.8)


with tempfile.TemporaryDirectory() as tmp:
    proj = Path(tmp) / "flask-demo"; proj.mkdir()
    (proj / "requirements.txt").write_text("flask==3.0.0\ngunicorn==21.2.0\nmypkg==1.0\n")
    (proj / "app.py").write_text("from flask import Flask\napp = Flask(__name__)\n")
    orch = Orchestrator(Settings(max_retries=3), client=DemoLLM(), builder=DemoDocker())
    rep = orch.run_full(proj, approve_build=True, on_event=lambda e: print(f"  [{e.kind}] {e.message}"),
                        on_log=lambda l: print("     |", l))
    print(f"\nRESULT: {rep.status} | attempts={rep.attempts} corrections={rep.corrections} score={rep.quality_score}")
    for a in rep.history.attempts:
        print(f"  attempt {a.attempt}: {a.result_label}; diagnosis={a.diagnosis!r}; changes={a.changes}")
    print("\nFinal Dockerfile:\n" + (proj / "Dockerfile").read_text())

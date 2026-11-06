# AI DevOps Agent — Dockerfiles, Compose and Kubernetes from a local LLM

Analyzes a software project, asks a **local Ollama model** (default `qwen2.5-coder:7b`) for an optimized Dockerfile,
validates and scores the result, builds it with Docker **after your approval**, diagnoses build failures and applies
targeted fixes in a bounded loop, scans and optimizes the image, and generates Docker Compose and Kubernetes files.

* **No cloud LLM APIs.** All inference goes to your own Ollama server.
* **Facts first, LLM second.** The analyzer extracts facts from files; the model only proposes. Every model answer is
  parsed, schema-validated, linted, scored and regression-checked before it is used.
* **Nothing risky runs by itself.** Docker builds need approval. Containers are never started. Kubernetes manifests are
  never applied. Existing files are never overwritten without approval.

> **Honest status.** The test suite (296 tests) mocks Ollama and Docker, plus a deterministic fault-injection harness.
> The code paths that talk to a real Ollama, Docker daemon, Hadolint, Trivy, Docker Scout and `kubectl` are covered by
> mocked tests and by opt-in integration tests (see [Testing](#testing)); output quality of a real model varies and must be
> judged by you. See [Limitations](#limitations).

## Architecture

```
                         ┌──────────────────────── Streamlit control center / CLI ───────────────────────┐
                         │ Overview · Analyzer · Model Center · Dockerfile Studio · Build Center ·       │
                         │ AI Debugging · Security · Image Optimization · Deployment · History           │
                         └───────────────────────────────────┬───────────────────────────────────────────┘
                                                             │ Orchestrator (app/agent)
   project dir ──► analyzer ──► facts ──► prompt_builder ──► Ollama client ──► response_parser (Pydantic)
   (app/analyzer)   confirmed vs          (app/llm)          latency/tokens         │
                    inferred                                                         ▼
                                              validator + quality score (app/docker) + security (app/security)
                                                             │ approval gate
                                                             ▼
        ┌─────────────────────────── feedback loop (app/agent) ───────────────────────────┐
        │ docker build ─► fail ─► error_classifier ─► fixable? ─► LLM correction ─► patch_guard ─┐
        │      ▲  (timeout, bounded logs,   not fixable / repeated error / retry limit ─► STOP   │
        │      └──────────────── rewrite Dockerfile (atomic, backup) ◄──────────────────────────┘
        └──────────────────────────────────────┬────────────────────────────────────────────────┘
                                               ▼
   SQLite history (app/history) ◄── runs/attempts (redacted)        image: scan (Trivy/Scout) · optimize (measured sizes)
                                               │
   compose generator (app/compose) · kubernetes generator (app/kubernetes)   ← deterministic, no LLM
```

| Package | Responsibility |
|---|---|
| `app/analyzer` | Language/framework/package-manager/port/env-var/service detection (Python incl. Poetry/Pipenv/uv, JS/TS incl. NestJS/pnpm/workspaces, Java, Go, PHP/Composer, Ruby/Bundler, monorepos, existing Docker/Compose files). Separates **confirmed** from **inferred** facts. |
| `app/llm` | Ollama client (multi-model, latency/token stats, timeouts+retry), prompt builder, response parser, hardware-aware model advisor, model benchmark. |
| `app/docker` | Dockerfile validator, 7-category quality scoring, optional Hadolint, bounded/redacted build executor, protected file writes. |
| `app/agent` | Orchestrator, build→diagnose→fix loop, error classifier, regression guard, in-run history, self-test. |
| `app/security` | Static Dockerfile/context security analysis; optional Trivy / Docker Scout image scanners. |
| `app/optimization` | Image size (measured) and cache analysis, estimated savings (labelled), `Dockerfile.optimized`, before/after comparison. |
| `app/history` | SQLite execution history: search, filters, JSON/CSV export, run comparison, success statistics. |
| `app/compose`, `app/kubernetes` | Deterministic Compose and Kubernetes generation with validation. |
| `app/ui` | Streamlit control center (one module per section in `app/ui/pages`). |
| `app/utils` | Settings, secret redaction, path safety, `.dockerignore` matching, YAML emitter, credential checks. |

## Requirements

* Python 3.10+
* [Ollama](https://ollama.com) running locally with a code model pulled (e.g. `ollama pull qwen2.5-coder:7b`)
* Docker Engine/Desktop — only needed to **build** images; analysis, scoring, Compose and Kubernetes generation work without it
* Optional: Hadolint, Trivy, Docker Scout (`docker scout`), `kubectl`, Docker Compose plugin (extra validation)

## Install

**Windows (PowerShell)**
```powershell
cd dockerfile-ai-generator
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1        # if blocked: Set-ExecutionPolicy -Scope Process Bypass
pip install -r requirements.txt
Copy-Item .env.example .env
ollama pull qwen2.5-coder:7b
python run.py doctor
python run.py ui
```

**Linux / macOS**
```bash
cd dockerfile-ai-generator
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
ollama pull qwen2.5-coder:7b
python run.py doctor
python run.py ui
```

`python run.py doctor` checks Ollama, the configured model and Docker. `python run.py ui` opens the dashboard
(http://localhost:8501).

## Configuration reference

Set in `.env` or the environment (`app/utils/config.py`):

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server |
| `OLLAMA_MODEL` | `qwen2.5-coder:7b` | Model used for generation/correction |
| `OLLAMA_TEMPERATURE` | `0.1` | Sampling temperature (0–2) |
| `OLLAMA_TIMEOUT` | `300` | Seconds per LLM request |
| `OLLAMA_TIMEOUT_RETRIES` | `1` | Extra attempts (1.5× timeout) after a timeout, e.g. while a model loads |
| `OLLAMA_NUM_CTX` | `8192` | Context window requested |
| `MAX_RETRIES` | `3` | Correction attempts; total builds = 1 + `MAX_RETRIES` |
| `REPEAT_ERROR_LIMIT` | `3` | Stop when the same error repeats this many times in a row |
| `DOCKER_BUILD_TIMEOUT` | `600` | Seconds before a build is killed |
| `MAX_LOG_BYTES` | `200000` | Build-log buffer (head+tail kept) |
| `MAX_CONTEXT_CHARS` / `MAX_FILE_BYTES` | `12000` / `100000` | Limits on what is sent to the LLM |
| `HISTORY_DB` | `generated/history.db` | SQLite history path |

## CLI

```
python run.py <command> [options]
```

| Command | What it does |
|---|---|
| `doctor` | Check Ollama, model, Docker |
| `models [--benchmark PROJECT --models a,b]` | Recommend models for your RAM/VRAM, assess installed ones, optionally compare them on a project (real generations). Never downloads or switches models. |
| `analyze PROJECT` | Print the analysis (JSON) |
| `run PROJECT [--model M] [--max-retries N] [--yes-overwrite] [--yes-build] [-v] [--export]` | Analyze → generate → save → build → self-correct |
| `fix PROJECT [...]` | Build the Dockerfile already in the project; the LLM repairs failures |
| `selftest [--fault copy-typo\|bad-tag\|missing-dep] [--keep]` | Inject a fault into a disposable copy and watch the loop repair it |
| `score PROJECT [--dockerfile F] [--json] [--min-score N]` | Quality + static security score; no LLM or Docker; exit 1 below `--min-score` (CI gate) |
| `scan IMAGE [--top N] [--fail-on high\|critical]` | Trivy / Docker Scout scan of a built image |
| `optimize PROJECT [--image I] [--write] [--build --yes-build]` | Size/cache suggestions; `--write` creates `Dockerfile.optimized`; `--build` compares **measured** sizes |
| `compose PROJECT [--write] [--yes-overwrite] [--generate-dockerfiles --yes-generate]` | Generate `docker-compose.yml` + `.env.example` (prints by default; never starts the stack) |
| `k8s PROJECT [--image I] [--namespace N] [--ingress-host H] [--health-path P] [--replicas N] [--write]` | Generate Kubernetes manifests; prints `kubectl` instructions; never deploys |
| `history [--search T] [--project P] [--failed] [--stats] [--export json\|csv]` | Browse the local execution history |

Exit codes: `0` success · `1` build/validation/gate failed · `2` environment problem (Ollama/Docker/bad path) ·
`3` declined or protected (approval missing, file exists).

## Dashboard sections

Overview (health, hardware, statistics) · Project Analyzer (confirmed vs inferred facts, services, monorepo units) ·
AI Model Center (hardware assessment, measured latency/tokens, benchmark) · Dockerfile Studio (generate/edit, quality
breakdown, protected save) · Build Center (approval, live logs, cancel) · AI Debugging Center (category, root cause,
proposed change, diff, remaining risks) · Security Center (static findings, optional scanners) · Image Optimization Center
(measured sizes, suggestions, optimized variant) · Deployment Center (Compose + Kubernetes) · Execution History
(search, filters, export, compare).

## Self-correction

1. A failed build is **classified** (daemon, resources, network, syntax, base image, missing files, system/Python/Node
   dependencies, permissions, workdir, port, startup, unknown) and judged **fixable by editing the Dockerfile or not**.
   Network/daemon/resource failures stop immediately: no LLM call can fix them.
2. For fixable errors the model gets the Dockerfile, the redacted error, the diagnosis, earlier failed attempts and
   (as reference text only) fixes that worked earlier for the same category in your local history.
3. The proposal must parse, validate, differ from the failing Dockerfile, stay a targeted change, and not regress
   (dropped `USER`, new blockers, missing `COPY` sources). Otherwise it is re-requested once and then rejected; the
   original Dockerfile is kept.
4. The loop stops on success, retry limit, repeated identical error, non-fixable error, or cancel.

"Learning" means a local history used as prompt reference. **The model is never trained or modified.**

## Security model

* No cloud LLM; `.env`, keys and credentials are never read into prompts; secrets are redacted from logs, reports,
  history and generated files.
* Docker builds require explicit approval (checkbox / `--yes-build`) and show a warning: building runs the
  instructions of an LLM-written file. Builds are unprivileged, time-limited, with bounded logs. Containers and
  Compose stacks are never started; Kubernetes is never deployed.
* Writes are restricted to the project (and `generated/`), refuse symlinks, are atomic, and back up existing files as
  `.bak`. Overwrites need approval.
* Static analysis cannot prove an image is secure; reports say so. Image scanners (Trivy/Scout) cover packages only.
* Compose: secrets are `${VAR:?}` references, `.env.example` has empty values, databases are not published and sit on an
  internal network, app ports bind to `127.0.0.1`, no host mounts/privileged/host networking.
* Kubernetes: Secret manifests are placeholder templates (`REPLACE_ME`), `allowPrivilegeEscalation: false`, all
  capabilities dropped, seccomp `RuntimeDefault`, no service-account token; `runAsNonRoot` only when a numeric `USER` is
  known.
* History stores no credentials and shortens home paths to `~`.

## Example workflows

```bash
# 1. Score an existing Dockerfile in CI (no LLM/Docker needed)
python run.py score ./myapp --min-score 70

# 2. Generate, build and self-correct
python run.py run ./myapp --yes-build -v --export

# 3. Fix a Dockerfile that already exists but fails to build
python run.py fix ./myapp --yes-build

# 4. Scan and slim the resulting image
python run.py scan myapp:ai-generated --fail-on high
python run.py optimize ./myapp --image myapp:ai-generated --write

# 5. Deployment files (reviewed, then applied by you)
python run.py compose ./myapp --write
python run.py k8s ./myapp --image registry.example.com/myapp:1.0.0 --ingress-host app.example.com --write

# 6. See what happened before
python run.py history --failed
python run.py history --export csv > history.csv
```

Reproducible deployment samples (no LLM or Docker needed): `python samples/generate_deployment_samples.py` →
`samples/output/python-flask/`. Offline end-to-end demo with a scripted LLM: `python samples/demo_end_to_end.py`.

## Testing

```bash
python -m pytest -q                                          # 296 unit/integration-style tests, no Ollama/Docker needed
RUN_INTEGRATION=1 python -m pytest -m integration tests/integration -v   # opt-in: real Docker / Ollama
```

* `tests/unit` mocks Ollama (`FakeSession`, `ScriptedClient`) and Docker (`FakeBuilder`, `FakeProc`).
* `tests/unit/test_fault_injection.py` proves the loop converges on invalid instructions, bad tags, missing `COPY`
  sources and unknown packages **with a rule-based fixer and no LLM or network**, and that it stops on network errors,
  retry limits and repeated errors. The `SimulatedDocker` only models the failure classes it reports.
* `tests/integration` runs against a real Docker daemon (build, classify real BuildKit errors, `docker compose config`,
  measured sizes) and a real Ollama model. Missing services are reported as skips, never passes.
* `tests/fixtures/flask_app` is a small real project used by those tests.

## Project layout

```
app/{analyzer,llm,docker,agent,security,optimization,compose,kubernetes,history,ui,utils}
tests/{unit,integration,fixtures}      generated/   logs/   samples/   docs/AUDIT.md
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `doctor`: Ollama unavailable | Start it (`ollama serve`), check `OLLAMA_HOST`; from WSL/containers use the host's address |
| Model not installed | `ollama pull <model>` (this tool never downloads models) |
| LLM requests time out | Raise `OLLAMA_TIMEOUT`; smaller model or GPU offload; first call includes model load time |
| Docker unavailable | Start Docker Desktop/daemon; on Linux add yourself to the `docker` group |
| `permission denied` on `/var/run/docker.sock` | Group membership or rootless Docker; not fixable from the Dockerfile (classified as such) |
| PowerShell blocks activation | `Set-ExecutionPolicy -Scope Process Bypass` |
| Port 8501 busy | `python run.py ui --server.port 8502` |
| Existing `Dockerfile` refused | Approve overwrite in the UI / `--yes-overwrite` (a `.bak` is kept) |
| Compose validation "PyYAML not installed" | `pip install PyYAML` (declared in `requirements.txt`) |
| `docker compose config` skipped | Install the Docker Compose plugin |

## Limitations

* The LLM can still produce Dockerfiles that are valid but suboptimal or wrong for your app; always review. Quality
  scores are heuristics, not guarantees.
* Port/entry-point/env-var detection is static and pattern-based; unusual layouts need manual correction (the output
  marks inferred values and warns when a Dockerfile `EXPOSE` disagrees with the source).
* End-of-life dates and base-image size estimates are approximate tables; image sizes after a build are measured.
* Docker Scout SARIF parsing is best-effort and unverified against a live Scout; Trivy parsing follows its JSON schema.
* Compose/Kubernetes generation does not create databases for Kubernetes, TLS, or network policies; resource requests are
  starting points; probes are TCP-based unless a health path is given.
* Static security analysis does not find vulnerable packages or application flaws.
* The code was developed in a sandbox without a Docker daemon or Ollama; real-service behaviour is verified only by the
  opt-in integration tests, which you should run on your machine.

## Suggested next phase

HTTP verification of built images (run on a random loopback port with no mounts, expect HTTP 200, always clean up),
a LangGraph orchestration layer, Kubernetes `kustomize` overlays, and CI templates that call `score`/`scan` as gates.

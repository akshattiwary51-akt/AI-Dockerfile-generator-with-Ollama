# Dockerfile AI Generator

A local AI DevOps assistant. It scans a project, asks a **locally running Ollama model** to write a `Dockerfile` and
`.dockerignore`, builds the image, and - if the build fails - feeds the error back to the model for a *targeted* fix,
repeating until the build succeeds or the retry limit is reached. No cloud LLM APIs are used.

## Quick start

```bash
# 1. Python 3.10+ environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Docker (verify the daemon is running)
docker version

# 3. Ollama + a code model
curl -fsSL https://ollama.com/install.sh | sh      # or https://ollama.com/download
ollama serve &                                      # skip if it already runs as a service
python run.py models                                # suggests models that fit your RAM/VRAM
ollama pull qwen2.5-coder:7b                        # default; smaller: qwen2.5-coder:3b, larger: qwen2.5-coder:14b

# 4. Configure (optional)
cp .env.example .env

# 5. Check everything
python run.py doctor
```

## Usage

```bash
python run.py ui                          # Streamlit dashboard (default)
python run.py analyze /path/to/project    # JSON project analysis, no LLM needed
python run.py run /path/to/project        # interactive: generate -> approve -> build -> self-correct
python run.py run ./proj --model qwen2.5-coder:14b --max-retries 5 --export
python run.py run ./proj --yes-overwrite --yes-build   # explicit non-interactive approvals
python samples/demo_end_to_end.py         # offline demo of the correction loop (no Ollama/Docker needed)
python -m pytest tests -q                 # 112 tests, no live Ollama or Docker required
```

Exit codes: `0` success, `1` build failed / generation failed, `2` environment problem (Ollama/Docker/path),
`3` declined or cancelled.

## Configuration (environment variables / `.env`)

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API |
| `OLLAMA_MODEL` | `qwen2.5-coder:7b` | Model name (any pulled model) |
| `MAX_RETRIES` | `3` | Correction attempts (total builds = 1 + retries) |
| `DOCKER_BUILD_TIMEOUT` | `600` | Seconds per build |
| `OLLAMA_TEMPERATURE` / `OLLAMA_TIMEOUT` / `OLLAMA_NUM_CTX` | `0.1` / `300` / `8192` | Generation settings |
| `MAX_LOG_BYTES` | `200000` | Cap on captured build output |
| `MAX_CONTEXT_CHARS` / `MAX_FILE_BYTES` | `12000` / `100000` | Limits on context sent to the LLM |

## Architecture

```
app/
  analyzer/   project_scanner (walk, exclusions, env-var names, structure)
              dependency_detector (manifests, versions, scripts)
              framework_detector (framework, entry points, ports, base-image strategy)
  llm/        ollama_client (availability, model check, structured output, errors)
              prompt_builder (generation + correction prompts)
              response_parser (JSON extraction + Pydantic validation)
              model_advisor (optional RAM/VRAM model suggestions; no extra dependency)
  docker/     dockerfile_generator (LLM round-trips, repair, protected atomic writes)
              docker_validator (static lint: errors block, warnings score)
              docker_builder (docker build, timeout, cancel, log bounds, error extraction)
  agent/      feedback_loop (build -> diagnose -> fix), execution_history, orchestrator
  ui/         dashboard (Streamlit)        utils/  config, logger, security
```

Key decisions:

- **Facts first, LLM second.** Static analysis extracts versions, entry points, ports and a suggested strategy; the
  prompt presents them as facts, so small local models have less to guess.
- **Never trust model output.** Replies are parsed and Pydantic-validated, then linted (`FROM` first, valid
  instructions and exec-form JSON, no embedded secrets, no `curl | sh`, no Docker socket, ...). Problems are sent back
  to the model for repair; nothing invalid reaches disk.
- **Targeted corrections.** The correction prompt includes the log tail, failed stage, current Dockerfile and earlier
  failed attempts. An unchanged Dockerfile, or a near-total rewrite (similarity < 0.35), is rejected and re-requested.
- **Stops when editing cannot help.** Docker daemon down, no disk space, registry rate limits or network failures end the
  loop instead of burning retries.
- **Conversation context** is kept across corrections (trimmed to bound growth).

## Security model

- Excluded from analysis and never sent to the LLM: `.git`, `node_modules`, virtualenvs, build/cache dirs, `.env*`
  (except `.env.example`-style templates, read for *names only*), private keys/certs, credential files, binaries.
- Secrets patterns (tokens, keys, passwords, URL credentials) are redacted from prompts, logs, build output and reports.
- Writes only to the project's `Dockerfile` / `.dockerignore` (symlinks refused, atomic writes, `.bak` backups).
  Existing files are **never overwritten without explicit approval**; reports go to this tool's `generated/` folder.
- Builds are explicit-approval only, unprivileged, with no mounts, no secrets passed, and a timeout. Containers are
  **never run**. Because a Dockerfile executes arbitrary `RUN` commands, only build projects you trust.

## What was verified (and what was not)

Verified here: the whole pipeline and 112 tests (analyzer, Ollama client errors, parsing, validation, protected writes,
builder success/failure/timeout/cancel with mocked subprocess, correction after failure, retry limit), the
offline end-to-end demo, and a headless render of the Streamlit dashboard.

**Not verified in the build environment:** a real Ollama model and a real Docker daemon were unavailable, so real model
output quality and real `docker build` behavior are untested. The BuildKit and legacy error parsers were tested against
representative output samples only. Try it on a small project with `python run.py doctor` first. Small models (<7B)
may need more retries; raise `MAX_RETRIES` or use a larger model.

## Troubleshooting

- *Ollama unreachable* - run `ollama serve`, check `OLLAMA_HOST`.
- *Model not installed* - `ollama pull <model>` (the error message prints the exact command).
- *Timeouts on CPU* - raise `OLLAMA_TIMEOUT`, use a smaller model, or lower `OLLAMA_NUM_CTX`.
- *Docker permission denied* - add your user to the `docker` group, or use rootless Docker.

## Possible next steps

Docker Compose / Kubernetes manifest generation, model benchmarking across installed models, a LangGraph state machine
for the agent, and CI/CD suggestions. These are intentionally left out so the core loop stays reliable.

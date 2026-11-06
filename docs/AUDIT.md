# Repository audit (baseline: 115 passing tests, ~3.7k lines; see the end for the final state)

Findings come from reading the code and probing it with fixtures, not from the README.

## Works and is kept
Scanner with sensitive-file exclusion, Pydantic validation of LLM output, static Dockerfile linter, bounded/redacted
Docker builds with timeout and cancel, protected atomic writes, feedback loop with retry limit, CLI (`doctor`, `models`,
`analyze`, `run`, `fix`, `selftest`), Streamlit dashboard (4 tabs).

## Defects found
| # | Finding | Evidence |
|---|---|---|
| 1 | Poetry caret specs break framework detection | `fastapi = "^0.110"` -> dependency name `fastapi^0.110`, framework `None` |
| 2 | `src/` layout apps get no entry point or start command | Flask app in `src/myapp/app.py` -> entry `[]`, strategy hint `None` |
| 3 | npm/yarn workspace roots are treated as a unit with no framework and no strategy | root `package.json` with `workspaces` |
| 4 | PHP (Composer) and Ruby (Bundler) are not detected | Laravel/Rails fixtures -> language `unknown` |
| 5 | Analysis does not separate confirmed facts from assumptions | defaults (e.g. port 5000) look identical to detected values |
| 6 | Existing compose files are listed but never interpreted; no infra (DB/cache) detection | |
| 7 | Dashboard calls `check_environment()` twice per rerun (sidebar + tab), each spawning `docker version` | `dashboard.py` lines 73 and 86 |
| 8 | Ollama client discards latency/token metadata; model advisor uses a static table unrelated to installed models | |
| 9 | Failures are not classified; repeated identical errors are not detected; a "fix" may silently drop `USER`/`HEALTHCHECK` | `feedback_loop.py`, `dockerfile_generator.py` |
| 10 | No check that `COPY` sources exist or that the build context leaks sensitive files | |
| 11 | History is in-memory only; nothing is learned from earlier runs | |
| 12 | No security scoring beyond lint, no image-size analysis, no Compose/Kubernetes output | |
| 13 | Flat test layout; no opt-in integration tests; fault injection relied on a scripted LLM only | |

## Roadmap (maps to phases)
2 analyzer (1-6) - 3 LLM (8) - 4 quality scoring (10) - 5 correction engine (9, 11) - 6 security (10, 12) -
7 optimization (12) - 8 history + dashboard (7, 11) - 9 compose - 10 kubernetes - 11 integration tests (13).

## Found while building the upgrade (all fixed, each with a regression test)
| # | Finding | How it was found |
|---|---|---|
| 14 | Python port detection ignored the very common `os.environ.get("PORT", "5050")` default and fell back to the framework default (5000) | Generating the Compose sample for `samples/python-flask`: the stack mapped 5000 while the app and its Dockerfile used 5050 |
| 15 | Compose/Kubernetes generators ignored the project's existing Dockerfile `EXPOSE`, the best evidence of the container port | Same sample; they now prefer `EXPOSE` and warn when it disagrees with the source |
| 16 | Compose forced detected plain env vars to empty strings (`${PORT:-}`), which would override an app's own default (`int("")` crash) | Same sample; plain variables are now passed through only when set |
| 17 | A static security check flagged a Compose file merely containing `${VAR}` password references | Compose generator tests; credential checks now judge values and ignore references (`app/utils/credcheck.py`) |

## Result (final)
296 unit tests plus 6 opt-in integration tests (real Docker/Ollama; skipped when unavailable). Real-service behaviour of
Ollama, Docker builds, Hadolint, Trivy, Docker Scout and `kubectl` was not exercised in the development sandbox beyond
`docker compose config`.

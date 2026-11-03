"""Streamlit dashboard. Run with:  python run.py ui   (or streamlit run app/ui/dashboard.py)."""
from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.agent.feedback_loop import LoopEvent  # noqa: E402
from app.agent.orchestrator import BUILD_WARNING, Orchestrator, RunReport  # noqa: E402
from app.docker.docker_validator import validate_artifacts  # noqa: E402
from app.docker.dockerfile_generator import FileProtectionError, GenerationError  # noqa: E402
from app.llm.ollama_client import OllamaError  # noqa: E402
from app.llm.response_parser import GeneratedArtifacts  # noqa: E402
from app.utils.config import get_settings  # noqa: E402

st.set_page_config(page_title="Dockerfile AI Generator", page_icon="🐳", layout="wide")

CSS = """
<style>
.block-container {padding-top: 1.6rem; max-width: 1250px;}
div[data-testid="stMetric"] {background: rgba(128,128,128,.08); border-radius: 10px; padding: .6rem .9rem;}
.status-pill {display:inline-block; padding:.15rem .7rem; border-radius:999px; font-size:.82rem; font-weight:600;}
.pill-ok {background:#1f8f4e22; color:#1f8f4e;} .pill-bad {background:#c0392b22; color:#c0392b;}
.pill-run {background:#2563eb22; color:#2563eb;}
</style>
"""


class BuildJob:
    """Background build state shared between the worker thread and the UI fragment."""

    def __init__(self) -> None:
        self.logs: list[str] = []
        self.events: list[LoopEvent] = []
        self.cancel = threading.Event()
        self.report: Optional[RunReport] = None
        self.error: Optional[str] = None
        self.done = False
        self.noted = False
        self.thread: Optional[threading.Thread] = None


def pill(ok: Optional[bool], text: str) -> str:
    cls = "pill-run" if ok is None else "pill-ok" if ok else "pill-bad"
    return f'<span class="status-pill {cls}">{text}</span>'


def init_state() -> None:
    for key, default in {"analysis": None, "artifacts": None, "saved": False, "job": None, "gen_error": None,
                         "save_msg": None, "project_dir": str(Path.cwd())}.items():
        st.session_state.setdefault(key, default)


def make_orchestrator(model: str, retries: int) -> Orchestrator:
    settings = get_settings().model_copy(update={"ollama_model": model, "max_retries": retries})
    return Orchestrator(settings)


# ----------------------------------------------------------------------------- sidebar
def sidebar() -> tuple[Orchestrator, str, bool]:
    st.sidebar.title("🐳 Dockerfile AI")
    st.sidebar.caption("Local LLM (Ollama) · build · self-correct")
    base = get_settings()
    st.sidebar.text_input("Project directory", key="project_dir", help="Absolute path of the project to containerize.")
    probe = Orchestrator(base.model_copy(update={"ollama_model": base.ollama_model}))
    env = probe.check_environment()
    options = env.installed_models or [base.ollama_model]
    default_idx = options.index(base.ollama_model) if base.ollama_model in options else 0
    model = st.sidebar.selectbox("Ollama model", options, index=default_idx,
                                 help="Models installed in Ollama; set OLLAMA_MODEL for the default.")
    retries = st.sidebar.slider("Max correction retries", 0, 10, base.max_retries)
    if st.sidebar.button("Re-check environment"):
        st.rerun()
    return make_orchestrator(model, retries), model, env.ready_to_generate


# ----------------------------------------------------------------------------- tabs
def tab_dashboard(orch: Orchestrator) -> None:
    env = orch.check_environment()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Ollama** " + pill(env.ollama_available, "connected" if env.ollama_available else "unavailable"),
                    unsafe_allow_html=True)
        st.markdown(f"Model `{env.model}` " + pill(env.model_installed, "installed" if env.model_installed
                                                    else "not installed"), unsafe_allow_html=True)
        if not env.ollama_available or not env.model_installed:
            st.error(env.ollama_message)
            if env.suggested_models:
                st.info("Models that fit this machine:\n\n" + "\n".join(f"- `ollama pull {m}`" for m in
                                                                          env.suggested_models))
    with c2:
        st.markdown("**Docker** " + pill(env.docker_available, "ready" if env.docker_available else "unavailable"),
                    unsafe_allow_html=True)
        if not env.docker_available:
            st.warning(env.docker_message)

    st.divider()
    path = st.session_state.project_dir
    if st.button("🔍 Analyze project", type="primary"):
        try:
            st.session_state.analysis = orch.analyze(path)
            st.session_state.artifacts = None
            st.session_state.saved = False
            st.session_state.job = None
        except (FileNotFoundError, NotADirectoryError) as exc:
            st.session_state.analysis = None
            st.error(str(exc))
    a = st.session_state.analysis
    if not a:
        st.info("Enter a project directory in the sidebar and click **Analyze project**.")
        return
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Project", a.name)
    m2.metric("Language", a.language)
    m3.metric("Framework", a.framework or "—")
    m4.metric("Dependencies", len(a.primary_unit.dependencies) if a.primary_unit else 0)
    for w in a.warnings:
        st.warning(w)
    left, right = st.columns(2)
    with left:
        u = a.primary_unit
        if u:
            st.markdown("**Runtime details**")
            st.write({"role": u.role, "package manager": u.package_manager, "runtime version": u.runtime_version,
                      "entry points": u.entry_points, "ports": u.ports, "lockfile": u.lockfile})
        if a.strategy:
            st.markdown(f"**Suggested base image:** `{a.strategy.base_image}`"
                        + (f" → `{a.strategy.runtime_image}`" if a.strategy.runtime_image else ""))
    with right:
        st.markdown("**Environment variables (names only)**")
        st.write(", ".join(f"`{e}`" for e in a.env_vars) or "none detected")
        if a.skipped_sensitive_files:
            st.caption(f"🔒 {len(a.skipped_sensitive_files)} sensitive file(s) excluded from analysis and never "
                       "sent to the LLM.")
        if a.existing_docker_files:
            st.caption("Existing Docker files: " + ", ".join(a.existing_docker_files))
    with st.expander("Project structure sent to the LLM"):
        st.code(a.structure or "(empty)", language="text")


def tab_generate(orch: Orchestrator, ready: bool) -> None:
    a = st.session_state.analysis
    if not a:
        st.info("Analyze a project first (Dashboard tab).")
        return
    if st.button("✨ Generate Dockerfile", type="primary", disabled=not ready):
        st.session_state.gen_error = None
        with st.spinner(f"Asking {orch.client.model} ... this can take a while on CPU"):
            try:
                st.session_state.artifacts = orch.generate(a)
                st.session_state.saved = False
                st.session_state.job = None
                st.session_state.pop("df_edit", None)
                st.session_state.pop("di_edit", None)
            except (OllamaError, GenerationError) as exc:
                st.session_state.gen_error = str(exc)
    if not ready:
        st.warning("Ollama/model is not ready - see the Dashboard tab for instructions.")
    if st.session_state.gen_error:
        st.error(st.session_state.gen_error)
    art: Optional[GeneratedArtifacts] = st.session_state.artifacts
    if not art:
        return
    st.session_state.setdefault("df_edit", art.dockerfile)
    st.session_state.setdefault("di_edit", art.dockerignore)
    c1, c2 = st.columns([3, 2])
    c1.text_area("Dockerfile (editable)", key="df_edit", height=420)
    c2.text_area(".dockerignore (editable)", key="di_edit", height=420)
    report = validate_artifacts(st.session_state.df_edit, st.session_state.di_edit)
    if report.errors:
        st.error("Validation errors:\n\n" + "\n".join(f"- {i.message}" for i in report.errors))
    for w in report.warnings:
        st.caption(f"⚠ {w.message}")
    st.progress(report.score / 100, text=f"Lint score {report.score}/100")
    with st.expander("LLM explanation", expanded=True):
        st.write(art.explanation or "—")
        st.caption(f"Base image: `{art.base_image}` · port: {art.exposed_port or '—'} · "
                   f"{art.language} / {art.framework or '—'}")
        for lim in art.limitations:
            st.markdown(f"- {lim}")

    existing = orch.existing_files(a.root)
    approve = False
    if existing:
        st.warning(f"Existing file(s) in the project: {', '.join(existing)}. They are protected and will be backed "
                   "up as `.bak` if you approve overwriting.")
        diff = orch.diff_with_existing(a.root, GeneratedArtifacts(dockerfile=st.session_state.df_edit or "FROM scratch\n"))
        if diff:
            with st.expander("Diff vs existing Dockerfile"):
                st.code(diff, language="diff")
        approve = st.checkbox("I approve overwriting the existing file(s)")
    if st.button("💾 Save to project", disabled=bool(report.errors) or (bool(existing) and not approve)):
        edited = art.model_copy(update={"dockerfile": st.session_state.df_edit,
                                        "dockerignore": st.session_state.di_edit})
        try:
            paths = orch.save(a.root, edited, approve_overwrite=approve)
            st.session_state.artifacts = edited
            st.session_state.saved = True
            st.session_state.save_msg = "Saved: " + ", ".join(str(p) for p in paths)
        except (FileProtectionError, ValueError, PermissionError) as exc:
            st.error(str(exc))
    if st.session_state.save_msg and st.session_state.saved:
        st.success(st.session_state.save_msg)


def start_build(orch: Orchestrator) -> None:
    job = BuildJob()
    a, art = st.session_state.analysis, st.session_state.artifacts

    def worker() -> None:
        try:
            job.report = orch.build_and_fix(a.root, a, art, approve_build=True, on_event=job.events.append,
                                            on_log=job.logs.append, cancel_event=job.cancel)
        except Exception as exc:  # surfaced in the UI
            job.error = f"{type(exc).__name__}: {exc}"
        finally:
            job.done = True

    job.thread = threading.Thread(target=worker, daemon=True)
    st.session_state.job = job
    job.thread.start()


@st.fragment(run_every=1.0)
def build_monitor(max_attempts: int) -> None:
    job: Optional[BuildJob] = st.session_state.job
    if job is None:
        return
    last = job.events[-1] if job.events else None
    attempt = max((e.attempt for e in job.events if e.kind == "build_start"), default=0)
    running = not job.done
    st.markdown("**Status** " + (pill(None, "running") if running else
                                  pill(bool(job.report and job.report.success), "finished")), unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Attempt", f"{attempt}/{max_attempts}" if attempt else "—")
    c2.metric("Phase", last.kind.replace("_", " ") if last else "starting")
    c3.metric("Current stage", (last.stage or "—")[:40] if last else "—")
    st.progress(min(1.0, attempt / max(1, max_attempts)))
    if running and st.button("⛔ Cancel build"):
        job.cancel.set()
    st.code("\n".join(job.logs[-200:]) or "(waiting for output...)", language="text")
    corrections = [e for e in job.events if e.kind in ("diagnosing", "corrected")]
    if corrections:
        st.markdown("**Correction activity**")
        for e in corrections:
            st.write(f"• attempt {e.attempt}: {e.message}")
    if job.error:
        st.error(job.error)
    if job.done and not job.noted:
        job.noted = True
        st.rerun()


def tab_build(orch: Orchestrator) -> None:
    if not st.session_state.artifacts:
        st.info("Generate a Dockerfile first.")
        return
    job: Optional[BuildJob] = st.session_state.job
    running = bool(job and not job.done)
    st.warning(BUILD_WARNING)
    st.caption("Containers are never run automatically; the build is unprivileged with no extra mounts.")
    if not st.session_state.saved:
        st.info("Save the Dockerfile to the project (Dockerfile Generation tab) before building.")
    approved = st.checkbox("I reviewed the Dockerfile and approve starting the Docker build",
                           disabled=running)
    if st.button("🔨 Build Docker image", type="primary",
                 disabled=running or not approved or not st.session_state.saved):
        start_build(orch)
        st.rerun()
    build_monitor(orch.settings.max_retries + 1)


def tab_report() -> None:
    job: Optional[BuildJob] = st.session_state.job
    if not job or not job.done or not job.report:
        st.info("The final report appears here after a build finishes.")
        return
    r = job.report
    (st.success if r.success else st.error)(("Build succeeded. " if r.success else "Build failed. ") + r.message)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Image", r.image)
    m2.metric("Attempts", f"{r.attempts} ({r.corrections} fixes)")
    m3.metric("Duration", f"{r.total_duration_seconds:.1f}s")
    m4.metric("Lint score", f"{r.quality_score}/100")
    if r.image_size_mb:
        st.caption(f"Image size ≈ {r.image_size_mb:.0f} MB")
    st.markdown("**Generated files**\n\n" + "\n".join(f"- `{f}`" for f in r.files))
    if r.suggestions:
        st.markdown("**Suggestions**\n\n" + "\n".join(f"- {s}" for s in r.suggestions))
    st.markdown("**Correction history**")
    for att in r.history.attempts:
        label = f"Attempt {att.attempt} - {att.result_label} ({att.duration_seconds:.1f}s)"
        with st.expander(label, expanded=not att.success and att is r.history.attempts[-1]):
            if att.failed_stage:
                st.write(f"Failed stage: `{att.failed_stage}`")
            if att.error:
                st.code(att.error, language="text")
            if att.diagnosis:
                st.info("Diagnosis: " + att.diagnosis)
            for c in att.changes:
                st.markdown(f"- {c}")
            st.code(att.dockerfile, language="dockerfile")
    st.download_button("⬇ Report (Markdown)", r.to_markdown(), file_name=f"{r.project}-report.md")
    st.download_button("⬇ Report (JSON)", r.model_dump_json(indent=2), file_name=f"{r.project}-report.json")


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()
    orch, _model, ready = sidebar()
    st.title("AI Dockerfile Generator")
    t1, t2, t3, t4 = st.tabs(["📊 Dashboard", "✨ Dockerfile Generation", "🔨 Docker Build", "📄 Final Report"])
    with t1:
        tab_dashboard(orch)
    with t2:
        tab_generate(orch, ready)
    with t3:
        tab_build(orch)
    with t4:
        tab_report()


main()

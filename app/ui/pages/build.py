import threading
from typing import Optional

import streamlit as st

from app.agent.orchestrator import BUILD_WARNING, Orchestrator
from app.ui.common import BuildJob, Ctx, need_artifacts, pill


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
    c = st.columns(3)
    c[0].metric("Attempt", f"{attempt}/{max_attempts}" if attempt else "—")
    c[1].metric("Phase", last.kind.replace("_", " ") if last else "starting")
    c[2].metric("Current stage", (last.stage or "—")[:40] if last else "—")
    st.progress(min(1.0, attempt / max(1, max_attempts)))
    if running and st.button("⛔ Cancel build"):
        job.cancel.set()
    st.code("\n".join(job.logs[-200:]) or "(waiting for output...)", language="text")
    for e in (e for e in job.events if e.kind in ("diagnosing", "corrected")):
        st.write(f"• attempt {e.attempt}: {e.message}")
    if job.error:
        st.error(job.error)
    if job.done and not job.noted:
        job.noted = True
        st.rerun()


def render(ctx: Ctx) -> None:
    if not need_artifacts():
        return
    job: Optional[BuildJob] = st.session_state.job
    running = bool(job and not job.done)
    st.warning(BUILD_WARNING)
    st.caption("Containers are never run automatically; the build is unprivileged with no extra mounts.")
    if not st.session_state.saved:
        st.info("Save the Dockerfile to the project (Dockerfile Studio) before building.")
    approved = st.checkbox("I reviewed the Dockerfile and approve starting the Docker build", disabled=running)
    if st.button("🔨 Build Docker image", type="primary",
                 disabled=running or not approved or not st.session_state.saved or not ctx.env.docker_available):
        start_build(ctx.orch)
        st.rerun()
    if not ctx.env.docker_available:
        st.warning(ctx.env.docker_message)
    build_monitor(ctx.orch.settings.max_retries + 1)
    if job and job.done and job.report:
        r = job.report
        (st.success if r.success else st.error)(("Build succeeded. " if r.success else "Build failed. ") + r.message)
        m = st.columns(4)
        m[0].metric("Image", r.image)
        m[1].metric("Attempts", f"{r.attempts} ({r.corrections} fixes)")
        m[2].metric("Duration", f"{r.total_duration_seconds:.1f}s")
        m[3].metric("Quality", f"{r.quality_score}/100")
        if r.image_size_mb:
            st.caption(f"Measured image size: {r.image_size_mb:.0f} MB")
        st.download_button("⬇ Report (Markdown)", r.to_markdown(), file_name=f"{r.project}-report.md")
        st.download_button("⬇ Report (JSON)", r.model_dump_json(indent=2), file_name=f"{r.project}-report.json")

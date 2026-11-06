from typing import Optional

import streamlit as st

from app.docker.dockerfile_generator import FileProtectionError, GenerationError
from app.docker.docker_validator import validate_artifacts
from app.docker.quality import score_dockerfile
from app.llm.ollama_client import OllamaError
from app.llm.response_parser import GeneratedArtifacts
from app.ui.common import Ctx, need_analysis


def _reset_edits() -> None:
    for k in ("df_edit", "di_edit"):
        st.session_state.pop(k, None)


def render(ctx: Ctx) -> None:
    if not need_analysis():
        return
    a = st.session_state.analysis
    b1, b2 = st.columns(2)
    if b1.button("✨ Generate Dockerfile", type="primary", disabled=not ctx.ready):
        st.session_state.gen_error = None
        with st.spinner(f"Asking {ctx.orch.client.model} ... this can take a while on CPU"):
            try:
                st.session_state.artifacts = ctx.orch.generate(a)
                st.session_state.saved, st.session_state.job = False, None
                _reset_edits()
            except (OllamaError, GenerationError) as exc:
                st.session_state.gen_error = str(exc)
    if ctx.orch.existing_files(a.root) and b2.button("📂 Load existing Dockerfile (for fixing / scoring)"):
        st.session_state.artifacts = ctx.orch.load_existing_artifacts(a.root)
        st.session_state.saved, st.session_state.job = True, None
        _reset_edits()
    if not ctx.ready:
        st.warning("Ollama/model is not ready - see Overview. You can still load an existing Dockerfile.")
    if st.session_state.gen_error:
        st.error(st.session_state.gen_error)
    art: Optional[GeneratedArtifacts] = st.session_state.artifacts
    if not art:
        return
    st.session_state.setdefault("df_edit", art.dockerfile)
    st.session_state.setdefault("di_edit", art.dockerignore)
    c1, c2 = st.columns([3, 2])
    c1.text_area("Dockerfile (editable)", key="df_edit", height=380)
    c2.text_area(".dockerignore (editable)", key="di_edit", height=380)
    df, di = st.session_state.df_edit, st.session_state.di_edit
    report = validate_artifacts(df, di)
    if report.errors:
        st.error("Validation errors:\n\n" + "\n".join(f"- {i.message}" for i in report.errors))

    q = score_dockerfile(df, di, analysis=a, project_root=a.root)
    st.markdown(f"### Quality score: {q.overall}/100 (grade {q.grade})")
    cols = st.columns(len(q.categories))
    for col, c in zip(cols, q.categories.values()):
        col.metric(c.label, f"{c.score}")
    for n in q.notes:
        st.caption("ℹ " + n)
    with st.expander("Findings and recommendations", expanded=bool(q.findings)):
        for f in sorted(q.findings, key=lambda f: ("high", "medium", "low").index(f.severity)):
            st.markdown(f"- **{f.severity}** [{f.category}] {f.message}" + (f" _(line {f.line})_" if f.line else "")
                        + f"  \n  ↳ {f.recommendation}")
        if q.hadolint_used:
            st.caption(f"Hadolint findings: {len(q.hadolint)}")
        else:
            st.caption("Hadolint not installed (optional); the built-in checks were used.")
    with st.expander("LLM explanation"):
        st.write(art.explanation or "—")
        st.caption(f"Base image: `{art.base_image}` · port: {art.exposed_port or '—'} · "
                   f"{art.language} / {art.framework or '—'}")
        for lim in art.limitations:
            st.markdown(f"- {lim}")

    existing = ctx.orch.existing_files(a.root)
    approve = False
    if existing:
        st.warning(f"Existing file(s) in the project: {', '.join(existing)}. They are protected and will be backed "
                   "up as `.bak` if you approve overwriting.")
        diff = ctx.orch.diff_with_existing(a.root, GeneratedArtifacts(dockerfile=df or "FROM scratch\n"))
        if diff:
            with st.expander("Diff vs existing Dockerfile"):
                st.code(diff, language="diff")
        approve = st.checkbox("I approve overwriting the existing file(s)")
    if st.button("💾 Save to project", disabled=bool(report.errors) or (bool(existing) and not approve)):
        edited = art.model_copy(update={"dockerfile": df, "dockerignore": di})
        try:
            paths = ctx.orch.save(a.root, edited, approve_overwrite=approve)
            st.session_state.artifacts, st.session_state.saved = edited, True
            st.session_state.save_msg = "Saved: " + ", ".join(str(p) for p in paths)
        except (FileProtectionError, ValueError, PermissionError) as exc:
            st.error(str(exc))
    if st.session_state.save_msg and st.session_state.saved:
        st.success(st.session_state.save_msg)

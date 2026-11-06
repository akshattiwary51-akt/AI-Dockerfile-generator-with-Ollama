import streamlit as st

from app.ui.common import Ctx


def render(ctx: Ctx) -> None:
    if st.button("🔍 Analyze project", type="primary"):
        try:
            st.session_state.analysis = ctx.orch.analyze(st.session_state.project_dir)
            st.session_state.artifacts = None
            st.session_state.saved = False
            st.session_state.job = None
            st.session_state.scan_reports = st.session_state.opt_result = None
            for k in ("df_edit", "di_edit"):
                st.session_state.pop(k, None)
        except (FileNotFoundError, NotADirectoryError) as exc:
            st.session_state.analysis = None
            st.error(str(exc))
    a = st.session_state.analysis
    if not a:
        st.info("Enter a project directory in the sidebar and click **Analyze project**.")
        return
    m = st.columns(4)
    m[0].metric("Project", a.name)
    m[1].metric("Language", a.language)
    m[2].metric("Framework", a.framework or "—")
    m[3].metric("Dependencies", len(a.primary_unit.dependencies) if a.primary_unit else 0)
    for w in a.warnings:
        st.warning(w)
    if len(a.units) > 1:
        st.markdown("**Components (monorepo)**")
        st.dataframe([{"path": u.path, "language": u.language, "framework": u.framework or "—", "role": u.role,
                       "package manager": u.package_manager or "—", "ports": ", ".join(map(str, u.ports)),
                       "workspace root": u.workspace_root} for u in a.units], hide_index=True,
                     use_container_width=True)
    left, right = st.columns(2)
    u = a.primary_unit
    with left:
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
        if a.secret_env_vars:
            st.caption("Secret-looking variables (never given values): " + ", ".join(a.secret_env_vars[:8]))
        if a.services:
            st.markdown("**Backing services detected**")
            for name, why in a.services.items():
                st.write(f"- `{name}` — {why}")
        if a.existing_compose:
            st.caption("Existing Compose files: " + "; ".join(f"{f}: {', '.join(s)}"
                                                              for f, s in a.existing_compose.items()))
        if a.skipped_sensitive_files:
            st.caption(f"🔒 {len(a.skipped_sensitive_files)} sensitive file(s) excluded from analysis and never "
                       "sent to the LLM.")
        if a.existing_docker_files:
            st.caption("Existing Docker files: " + ", ".join(a.existing_docker_files))
    if u and (u.confirmed or u.inferred):
        c, i = st.columns(2)
        c.markdown("**Confirmed from files**\n\n" + ("\n".join(f"- {x}" for x in u.confirmed) or "—"))
        i.markdown("**Inferred / assumed**\n\n" + ("\n".join(f"- {x}" for x in u.inferred) or "—"))
    with st.expander("Project structure sent to the LLM"):
        st.code(a.structure or "(empty)", language="text")

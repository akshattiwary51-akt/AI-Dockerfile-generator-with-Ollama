import streamlit as st

from app.ui.common import Ctx, pill


def render(ctx: Ctx) -> None:
    env = ctx.env
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Ollama** " + pill(env.ollama_available, "connected" if env.ollama_available else "unavailable"),
                    unsafe_allow_html=True)
        st.markdown(f"Model `{env.model}` " + pill(env.model_installed, "installed" if env.model_installed
                                                    else "not installed"), unsafe_allow_html=True)
        if not env.ollama_available or not env.model_installed:
            st.error(env.ollama_message)
            if env.suggested_models:
                st.info("Models that fit this machine (nothing is downloaded automatically):\n\n" +
                        "\n".join(f"- `ollama pull {m}`" for m in env.suggested_models))
    with c2:
        st.markdown("**Docker** " + pill(env.docker_available, "ready" if env.docker_available else "unavailable"),
                    unsafe_allow_html=True)
        if not env.docker_available:
            st.warning(env.docker_message)

    st.divider()
    s = ctx.orch.store.stats()
    m = st.columns(4)
    m[0].metric("Recorded runs", s["executions"])
    m[1].metric("Success rate", "—" if s["success_rate"] is None else f"{s['success_rate']:.0%}")
    m[2].metric("Runs that needed fixes", s["runs_needing_correction"])
    m[3].metric("…ended in success", "—" if s["correction_success_rate"] is None
                else f"{s['correction_success_rate']:.0%}")
    st.caption("Statistics come from the local history database only. The model is never trained or modified; "
               "earlier fixes are only shown to it as reference text in later prompts.")
    st.markdown("**Recent runs**")
    rows = ctx.orch.store.search(limit=5)
    if rows:
        st.dataframe([{k: r[k] for k in ("id", "created_at", "project", "model", "status", "attempts", "quality_score")}
                      for r in rows], hide_index=True, use_container_width=True)
    else:
        st.info("No runs recorded yet.")
    st.markdown("**Workflow state**")
    a, art, job = st.session_state.analysis, st.session_state.artifacts, st.session_state.job
    st.write(f"1. Analysis: {'done — ' + a.name if a else 'not done'}  \n"
             f"2. Dockerfile: {'ready' if art else 'not generated'}{' (saved)' if st.session_state.saved else ''}  \n"
             f"3. Build: {'running' if job and not job.done else 'finished' if job else 'not started'}")

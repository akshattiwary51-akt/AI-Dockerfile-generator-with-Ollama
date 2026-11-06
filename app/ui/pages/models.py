import streamlit as st

from app.llm.ollama_client import OllamaError
from app.ui.common import Ctx


def render(ctx: Ctx) -> None:
    if not ctx.env.ollama_available:
        st.error(ctx.env.ollama_message)
        return
    st.caption("Advice only: no model is downloaded or switched automatically. Choose the model in the sidebar.")
    try:
        hw, assessed = ctx.orch.assess_models()
    except OllamaError as exc:
        st.error(str(exc))
        return
    h = st.columns(4)
    h[0].metric("RAM (free)", "?" if hw.ram_gb is None else f"{hw.ram_gb:.0f} GB ({(hw.ram_available_gb or 0):.0f})")
    h[1].metric("GPU", hw.gpu_name or ("unified memory" if hw.unified_memory else "none detected"))
    h[2].metric("VRAM (free)", "—" if hw.vram_gb is None else f"{hw.vram_gb:.0f} GB ({(hw.vram_free_gb or 0):.0f})")
    h[3].metric("OS", hw.os or "?")
    if assessed:
        st.markdown("**Installed models for this machine** (speed figures are estimates)")
        st.dataframe([{"model": m.name, "verdict": m.verdict, "params (B)": m.parameters_b, "quant": m.quantization,
                       "memory GB": round(m.total_gb, 1), "placement": m.placement,
                       "est. tok/s": m.est_tokens_per_second, "context": m.context_supported,
                       "context ok": m.context_ok, "notes": "; ".join(m.notes)} for m in assessed],
                     hide_index=True, use_container_width=True)
    else:
        st.info("No models installed in Ollama.")
    if ctx.env.suggested_models:
        st.markdown("**Suggested to pull (manual):** " + ", ".join(f"`ollama pull {m}`" for m in ctx.env.suggested_models))

    st.divider()
    st.markdown("**Measured usage in this session**")
    stats = ctx.orch.client.stats_log
    if stats:
        st.dataframe([{"model": s.model, "latency s": round(s.latency_seconds, 1), "load s": round(s.load_seconds, 1),
                       "prompt tok": s.prompt_tokens, "completion tok": s.completion_tokens,
                       "tok/s": s.tokens_per_second, "estimated": s.estimated} for s in stats], hide_index=True,
                     use_container_width=True)
    else:
        st.caption("No LLM calls yet in this session.")

    st.divider()
    st.markdown("**Compare models on the analyzed project**")
    names = [m.name for m in assessed]
    picks = st.multiselect("Models to benchmark", names, default=names[:2])
    st.caption("Runs a real generation on each selected model (can take minutes on CPU). Nothing is written to the project.")
    if st.button("Run benchmark", disabled=not picks or not st.session_state.analysis):
        with st.spinner("Benchmarking..."):
            st.session_state.bench = ctx.orch.benchmark(st.session_state.analysis, picks)
    if not st.session_state.analysis:
        st.info("Analyze a project first to enable benchmarking.")
    if st.session_state.bench:
        st.dataframe([{"model": r.model, "ok": r.ok, "latency s": round(r.latency_seconds, 1), "quality": r.lint_score,
                       "tokens": r.prompt_tokens + r.completion_tokens, "tok/s": r.tokens_per_second,
                       "format retries": r.format_retries, "error": r.error} for r in st.session_state.bench],
                     hide_index=True, use_container_width=True)

import streamlit as st

from app.optimization.optimizer import (analyze_optimizations, build_and_compare, measure_image, optimize_dockerfile,
                                        write_optimized)
from app.ui.common import Ctx, current_files, need_analysis, need_artifacts


def render(ctx: Ctx) -> None:
    if not (need_analysis() and need_artifacts()):
        return
    a = st.session_state.analysis
    df, di = current_files()
    job = st.session_state.job
    image = job.report.image if job and job.report and job.report.success else ""
    image = st.text_input("Built image to measure", value=image)
    measured = measure_image(image) if image else None
    if measured and measured.available:
        st.metric("Measured size", f"{measured.size_mb} MB")
    elif measured:
        st.caption(f"Not measured: {measured.error}")
    rep = analyze_optimizations(df, di, measured, a.root)
    st.caption(rep.note)
    if rep.largest_layers:
        st.markdown("**Largest layers (measured)**")
        st.dataframe([{"MB": round(l.size_bytes / 1e6, 1), "created by": l.created_by} for l in rep.largest_layers],
                     hide_index=True)
    st.markdown("**Suggestions**")
    for s in rep.suggestions:
        st.markdown(f"- {s.message}" + (f" _(estimated saving ≈ {s.estimated_saving_mb} MB, not measured)_"
                                        if s.estimated_saving_mb else ""))
    if not rep.suggestions:
        st.success("No optimization suggestions.")
    new = optimize_dockerfile(df)
    st.divider()
    if new == df:
        st.info("The automatic safe rewrites would not change this Dockerfile.")
        return
    import difflib
    st.code("\n".join(difflib.unified_diff(df.splitlines(), new.splitlines(), "Dockerfile", "Dockerfile.optimized",
                                           lineterm="")), language="diff")
    st.caption("Written as Dockerfile.optimized next to your Dockerfile; the original is never modified.")
    approve = st.checkbox("I approve building the optimized variant (a Docker build)")
    if st.button("Build optimized image and compare", disabled=not (approve and image and ctx.env.docker_available)):
        write_optimized(a.root, df)
        with st.spinner("Building..."):
            res, cmp = build_and_compare(ctx.orch.builder, a.root, image, f"{image}-optimized", approved=True)
        st.session_state.opt_result = (res.success, cmp.summary(), res.error_summary)
    if st.session_state.opt_result:
        ok, summary, err = st.session_state.opt_result
        (st.success if ok else st.error)(summary if ok else f"Optimized build failed: {err}")

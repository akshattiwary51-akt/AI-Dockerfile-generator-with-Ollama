import streamlit as st

from app.ui.common import Ctx


def render(ctx: Ctx) -> None:
    store = ctx.orch.store
    st.caption(f"Local database: `{store.path}` — secrets are redacted before storing.")
    f1, f2, f3, f4 = st.columns(4)
    text = f1.text_input("Search")
    project = f2.selectbox("Project", [""] + store.projects(), format_func=lambda x: x or "all")
    outcome = f3.selectbox("Outcome", ["all", "success", "failed"])
    min_q = f4.slider("Min quality", 0, 100, 0)
    filters = dict(text=text, project=project, success=None if outcome == "all" else outcome == "success",
                   min_quality=min_q or None)
    rows = store.search(**filters)
    s = store.stats()
    m = st.columns(4)
    m[0].metric("Runs", s["executions"])
    m[1].metric("Success rate", "—" if s["success_rate"] is None else f"{s['success_rate']:.0%}")
    m[2].metric("Fix success rate", "—" if s["correction_success_rate"] is None else f"{s['correction_success_rate']:.0%}")
    m[3].metric("Avg quality", s["avg_quality"] or "—")
    if s["failures_by_category"]:
        st.bar_chart(s["failures_by_category"])
    if not rows:
        st.info("No matching runs.")
        return
    st.dataframe(rows, hide_index=True, use_container_width=True)
    d1, d2 = st.columns(2)
    d1.download_button("⬇ Export JSON", store.export_json(**filters), file_name="history.json")
    d2.download_button("⬇ Export CSV", store.export_csv(**filters), file_name="history.csv")
    ids = [r["id"] for r in rows]
    pick = st.selectbox("Run details", ids)
    ex = store.get(pick)
    if ex:
        with st.expander("Final Dockerfile"):
            st.code(ex["dockerfile"], language="dockerfile")
        st.write(ex["message"])
    if len(ids) >= 2:
        st.markdown("**Compare two runs**")
        c1, c2 = st.columns(2)
        a, b = c1.selectbox("Run A", ids, key="cmp_a"), c2.selectbox("Run B", ids, index=1, key="cmp_b")
        if a != b:
            cmp = store.compare(a, b)
            st.write(cmp["changed"] or "No differences in the summary metrics.")
            if cmp["dockerfile_diff"]:
                st.code(cmp["dockerfile_diff"], language="diff")

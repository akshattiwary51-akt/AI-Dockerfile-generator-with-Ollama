import streamlit as st

from app.ui.common import Ctx


def _show_attempts(attempts) -> None:
    for att in attempts:
        label = f"Attempt {att.attempt} — {att.result_label}" + (f" · {att.category}" if att.category else "")
        with st.expander(label, expanded=not att.success and att is attempts[-1]):
            if att.category:
                fixable = {None: "unknown", True: "yes", False: "no"}[att.fixable]
                st.caption(f"Category `{att.category}` · fixable by editing the Dockerfile: {fixable}"
                           + (f" · failing instruction: `{att.failing_instruction}`" if att.failing_instruction else "")
                           + (f" · same error {att.repeated}× in a row" if att.repeated > 1 else ""))
            if att.error:
                st.code(att.error, language="text")
            if att.diagnosis:
                st.info("Root cause: " + att.diagnosis)
            if att.explanation:
                st.write("Proposed correction: " + att.explanation)
            for c in att.changes:
                st.markdown(f"- {c}")
            if att.expected_effect:
                st.write("Expected effect: " + att.expected_effect)
            for r in att.remaining_risks:
                st.warning("Remaining risk: " + r)
            if att.diff:
                st.code(att.diff, language="diff")


def render(ctx: Ctx) -> None:
    job = st.session_state.job
    if job and job.done and job.report:
        r = job.report
        st.markdown(f"### Latest run: {r.project} — {r.status}")
        _show_attempts(r.history.attempts)
        st.divider()
    st.markdown("### Past runs")
    rows = ctx.orch.store.search(success=False, limit=20)
    if not rows:
        st.info("No failed runs recorded.")
        return
    choice = st.selectbox("Failed run", [r["id"] for r in rows],
                          format_func=lambda i: next(f"#{r['id']} {r['project']} — {r['status']} ({r['created_at']})"
                                                     for r in rows if r["id"] == i))
    ex = ctx.orch.store.get(choice)
    if ex:
        for att in ex["attempts_detail"]:
            with st.expander(f"Attempt {att['n']} — {'success' if att['success'] else 'failed'}"
                             + (f" · {att['category']}" if att["category"] else "")):
                if att["error"]:
                    st.code(att["error"], language="text")
                if att["diagnosis"]:
                    st.info(att["diagnosis"])
                for ch in att["changes"]:
                    st.markdown(f"- {ch}")
                if att["diff"]:
                    st.code(att["diff"], language="diff")

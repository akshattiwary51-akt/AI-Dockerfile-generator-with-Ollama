import streamlit as st

from app.security.analyzer import SEVERITY_ORDER, analyze_dockerfile_security
from app.security.scanners import scan_image
from app.ui.common import Ctx, current_files, need_analysis, need_artifacts

ICON = {"critical": "🟥", "high": "🟧", "medium": "🟨", "low": "🟦", "info": "⬜"}


def render(ctx: Ctx) -> None:
    if not (need_analysis() and need_artifacts()):
        return
    a = st.session_state.analysis
    df, di = current_files()
    rep = analyze_dockerfile_security(df, di, analysis=a, project_root=a.root)
    st.markdown(f"### Static security score: {rep.score}/100 — highest severity: {rep.risk_level}")
    st.caption(rep.disclaimer)
    cols = st.columns(5)
    for col, sev in zip(cols, SEVERITY_ORDER):
        col.metric(sev, rep.counts[sev])
    for f in sorted(rep.findings, key=lambda f: SEVERITY_ORDER.index(f.severity)):
        st.markdown(f"{ICON[f.severity]} **{f.severity}** · {f.message}" + (f" _(line {f.line})_" if f.line else "")
                    + f"  \n  ↳ {f.remediation}")
    if not rep.findings:
        st.success("No static findings (this does not prove the image is secure).")
    if rep.base_images:
        with st.expander("Base images"):
            st.dataframe([{"image": b.image, "stage": b.stage, "variant": b.variant, "pinned": b.pinned_tag,
                           "digest": b.digest_pinned, "EOL (approx.)": b.eol or "?", "status": b.eol_status}
                          for b in rep.base_images], hide_index=True)
            st.caption("End-of-life dates are approximate; verify at endoflife.date.")

    st.divider()
    st.markdown("### Image vulnerability scan (optional tools)")
    job = st.session_state.job
    image = job.report.image if job and job.report and job.report.success else ""
    image = st.text_input("Image to scan (must already be built locally)", value=image)
    st.caption("Trivy / Docker Scout analyse the image's packages; the image is never run.")
    if st.button("Scan image", disabled=not image):
        with st.spinner("Scanning..."):
            st.session_state.scan_reports = scan_image(image)
    for r in st.session_state.scan_reports or []:
        st.markdown(f"**{r.tool}**")
        if not r.available:
            st.info(r.error)
            continue
        if not r.ok:
            st.error(r.error)
            continue
        st.write({k: v for k, v in r.counts.items() if v} or "no vulnerabilities reported")
        if r.top():
            st.dataframe([{"id": v.id, "severity": v.severity, "package": v.package, "installed": v.installed,
                           "fixed in": v.fixed or "—", "title": v.title} for v in r.top(15)], hide_index=True)
        for x in r.remediation() + r.notes:
            st.caption("• " + x)

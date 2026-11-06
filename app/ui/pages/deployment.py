import streamlit as st

from app.compose.generator import ComposeProtectionError, generate_compose, validate_compose, write_compose
from app.ui.common import Ctx, need_analysis


def render(ctx: Ctx) -> None:
    if not need_analysis():
        return
    tab_c, tab_k = st.tabs(["Docker Compose", "Kubernetes"])
    with tab_c:
        _compose(ctx)
    with tab_k:
        _kubernetes()


def _kubernetes() -> None:
    from pathlib import Path

    from app.kubernetes.generator import (K8sProtectionError, apply_instructions, generate_manifests,
                                          load_unit_dockerfiles, validate_manifests, write_manifests)
    a = st.session_state.analysis
    st.caption("Manifests are generated deterministically and are never applied by this tool. Secrets are placeholder "
               "templates only.")
    c = st.columns(4)
    image = c[0].text_input("Image (registry reference)", placeholder="registry.example.com/app:1.0.0") or None
    namespace = c[1].text_input("Namespace") or None
    host = c[2].text_input("Ingress host (optional)") or None
    health = c[3].text_input("HTTP health path (optional)") or None
    replicas = st.slider("Replicas", 1, 10, 2)
    try:
        docs = load_unit_dockerfiles(a, a.root)
        result = generate_manifests(a, image=image, dockerfile=docs.get(".", ""), units_dockerfiles=docs,
                                    namespace=namespace, ingress_host=host, health_path=health, replicas=replicas)
    except ValueError as exc:
        st.info(str(exc))
        return
    check = validate_manifests(result.docs, yaml_text=result.combined)
    if check.ok:
        st.success("Validation passed" + (" (kubectl dry-run checked)" if check.kubectl_checked else " (built-in checks only)"))
    for e in check.errors:
        st.error(e)
    for w in result.warnings + check.warnings:
        st.warning(w)
    for n in result.notes:
        st.caption("ℹ " + n)
    for path, text in result.files.items():
        with st.expander(f"k8s/{path}"):
            st.code(text, language="yaml")
    st.markdown("**To deploy (run these yourself after review):**")
    st.code("\n".join(apply_instructions(result, namespace=namespace)), language="bash")
    existing = [p for p in result.files if (Path(a.root) / "k8s" / p).exists()]
    approve = st.checkbox("I approve overwriting existing manifest file(s)", key="k8s_ow") if existing else False
    if st.button("💾 Write manifests to k8s/", disabled=not check.ok or (bool(existing) and not approve)):
        try:
            st.success("Wrote " + str(len(write_manifests(a.root, result, overwrite=approve))) + " file(s) under k8s/")
        except (K8sProtectionError, PermissionError) as exc:
            st.error(str(exc))


def _compose(ctx: Ctx) -> None:
    a = st.session_state.analysis
    st.markdown("### Docker Compose")
    st.caption("Generated deterministically from the analysis. Secrets are `${VAR}` references; `.env.example` has "
               "empty values. The stack is never started from here.")
    try:
        result = generate_compose(a, a.root)
    except ValueError as exc:
        st.info(str(exc))
        return
    check = validate_compose(result.compose, result.env_example)
    (st.success if check.ok else st.error)("Validation passed" + (" (docker compose config checked)"
                                                                  if check.docker_compose_checked else " (structure only)")
                                           if check.ok else "Validation failed")
    for e in check.errors:
        st.error(e)
    for w in result.warnings + check.warnings:
        st.warning(w)
    for n in result.notes:
        st.caption("ℹ " + n)
    c1, c2 = st.columns([3, 2])
    c1.code(result.compose, language="yaml")
    c2.code(result.env_example, language="bash")
    existing = [n for n in ("docker-compose.yml", ".env.example") if (__import__("pathlib").Path(a.root) / n).exists()]
    approve = st.checkbox(f"I approve overwriting existing file(s): {', '.join(existing)}") if existing else False
    if st.button("💾 Write Compose files to project", disabled=not check.ok or (bool(existing) and not approve)):
        try:
            st.success("Wrote: " + ", ".join(str(p) for p in write_compose(a.root, result, overwrite=approve)))
        except (ComposeProtectionError, PermissionError) as exc:
            st.error(str(exc))

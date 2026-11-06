"""Streamlit control center. Run with:  python run.py ui   (or streamlit run app/ui/dashboard.py)."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.ui.common import CSS, build_context, init_state  # noqa: E402
from app.utils.config import get_settings  # noqa: E402

# (label, module) - only the selected page is rendered, which keeps reruns fast.
PAGES = [("Overview", "overview"), ("Project Analyzer", "analyzer"), ("AI Model Center", "models"),
         ("Dockerfile Studio", "studio"), ("Build Center", "build"), ("AI Debugging Center", "debugging"),
         ("Security Center", "security"), ("Image Optimization Center", "optimization"),
         ("Deployment Center", "deployment"), ("Execution History", "history")]


def main() -> None:
    st.set_page_config(page_title="AI DevOps Agent", page_icon="🐳", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()
    base = get_settings()
    st.sidebar.title("🐳 AI DevOps Agent")
    st.sidebar.caption("Local LLM (Ollama) · no cloud APIs")
    st.sidebar.text_input("Project directory", key="project_dir", help="Absolute path of the project to containerize.")
    probe = build_context()
    options = probe.env.installed_models or [base.ollama_model]
    model = st.sidebar.selectbox("Ollama model", options,
                                 index=options.index(base.ollama_model) if base.ollama_model in options else 0)
    retries = st.sidebar.slider("Max correction retries", 0, 10, base.max_retries)
    ctx = build_context(model, retries)
    st.sidebar.divider()
    labels = [p[0] for p in PAGES]
    page = st.sidebar.radio("Section", labels, key="nav")
    st.title(page)
    module = importlib.import_module("app.ui.pages." + dict(PAGES)[page])
    module.render(ctx)


main()

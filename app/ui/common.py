"""Shared UI state and helpers for the Streamlit control center."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import streamlit as st

from app.agent.feedback_loop import LoopEvent
from app.agent.orchestrator import EnvironmentStatus, Orchestrator, RunReport
from app.utils.config import get_settings

CSS = """
<style>
.block-container {padding-top: 1.6rem; max-width: 1250px;}
div[data-testid="stMetric"] {background: rgba(128,128,128,.08); border-radius: 10px; padding: .6rem .9rem;}
.status-pill {display:inline-block; padding:.15rem .7rem; border-radius:999px; font-size:.82rem; font-weight:600;}
.pill-ok {background:#1f8f4e22; color:#1f8f4e;} .pill-bad {background:#c0392b22; color:#c0392b;}
.pill-run {background:#2563eb22; color:#2563eb;}
</style>
"""
ENV_CACHE_SECONDS = 15.0


class BuildJob:
    """Background build state shared between the worker thread and the UI fragment."""

    def __init__(self) -> None:
        self.logs: list[str] = []
        self.events: list[LoopEvent] = []
        self.cancel = threading.Event()
        self.report: Optional[RunReport] = None
        self.error: Optional[str] = None
        self.done = False
        self.noted = False
        self.thread: Optional[threading.Thread] = None


@dataclass
class Ctx:
    orch: Orchestrator
    env: EnvironmentStatus
    model: str

    @property
    def ready(self) -> bool:
        return self.env.ready_to_generate


def pill(ok: Optional[bool], text: str) -> str:
    cls = "pill-run" if ok is None else "pill-ok" if ok else "pill-bad"
    return f'<span class="status-pill {cls}">{text}</span>'


def init_state() -> None:
    for key, default in {"analysis": None, "artifacts": None, "saved": False, "job": None, "gen_error": None,
                         "save_msg": None, "project_dir": str(Path.cwd()), "nav": "Overview",
                         "scan_reports": None, "opt_result": None, "bench": None}.items():
        st.session_state.setdefault(key, default)


def get_orchestrator(model: str, retries: int) -> Orchestrator:
    """One Orchestrator per (model, retries) so the environment cache and the LLM stats survive reruns."""
    cache = st.session_state.setdefault("_orchs", {})
    key = (model, retries)
    if key not in cache:
        cache[key] = Orchestrator(get_settings().model_copy(update={"ollama_model": model, "max_retries": retries}))
    return cache[key]


def build_context(model: Optional[str] = None, retries: Optional[int] = None) -> Ctx:
    base = get_settings()
    model = model or base.ollama_model
    orch = get_orchestrator(model, base.max_retries if retries is None else retries)
    return Ctx(orch=orch, env=orch.check_environment(max_age=ENV_CACHE_SECONDS), model=model)


def need_analysis() -> bool:
    if not st.session_state.analysis:
        st.info("Analyze a project first (Project Analyzer section).")
        return False
    return True


def need_artifacts() -> bool:
    if not st.session_state.artifacts:
        st.info("Generate or load a Dockerfile first (Dockerfile Studio section).")
        return False
    return True


def current_files() -> tuple[str, str]:
    art = st.session_state.artifacts
    return (st.session_state.get("df_edit") or art.dockerfile, st.session_state.get("di_edit") or art.dockerignore)

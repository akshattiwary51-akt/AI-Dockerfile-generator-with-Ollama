#!/usr/bin/env python3
"""Entry point.

    python run.py ui                 # Streamlit dashboard (default)
    python run.py doctor             # check Ollama / model / Docker
    python run.py models             # model suggestions for your RAM/VRAM
    python run.py analyze <dir>      # print project analysis
    python run.py run <dir> [flags]  # CLI pipeline (see --help)
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def launch_ui(extra: list[str]) -> int:
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print("Streamlit is not installed. Run: pip install -r requirements.txt", file=sys.stderr)
        return 2
    cmd = [sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "ui" / "dashboard.py"),
           "--server.headless", "true", *extra]
    return subprocess.call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] == "ui":
        raise SystemExit(launch_ui(args[1:]))
    from app.main import main

    raise SystemExit(main(args))

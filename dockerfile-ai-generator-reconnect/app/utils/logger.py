"""Logging setup with automatic secret redaction."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.utils.security import redact_secrets

_CONFIGURED = False
LOG_DIR = Path(__file__).resolve().parents[2] / "logs"


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg = redact_secrets(record.getMessage())
            record.args = ()
        except Exception:  # never let logging break the app
            pass
        return True


def get_logger(name: str = "dockerfile_ai") -> logging.Logger:
    global _CONFIGURED
    root = logging.getLogger("dockerfile_ai")
    if not _CONFIGURED:
        root.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
        stream = logging.StreamHandler()
        stream.setFormatter(fmt)
        stream.addFilter(RedactingFilter())
        root.addHandler(stream)
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            fh = RotatingFileHandler(LOG_DIR / "app.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8")
            fh.setFormatter(fmt)
            fh.addFilter(RedactingFilter())
            root.addHandler(fh)
        except OSError:  # read-only FS etc.
            root.warning("File logging disabled (cannot write to %s)", LOG_DIR)
        root.propagate = False
        _CONFIGURED = True
    return root if name == "dockerfile_ai" else root.getChild(name.replace("dockerfile_ai.", ""))

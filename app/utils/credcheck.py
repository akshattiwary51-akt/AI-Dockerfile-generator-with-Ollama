"""Detect embedded credentials in parsed YAML documents, ignoring ``${VAR}`` references."""
from __future__ import annotations

import re
from typing import Any, Iterator

from app.utils.security import redact_secrets

_REF = re.compile(r"\$\{[^}]*\}")
_USERINFO = re.compile(r"://([^/@\s]+)@")
_REF_ONLY_PASSWORD = re.compile(r"(?:\$\{[^}]*\}|[^:$])*:(?:\$\{[^}]*\})+")


def iter_strings(node: Any) -> Iterator[str]:
    if isinstance(node, dict):
        for v in node.values():
            yield from iter_strings(v)
    elif isinstance(node, list):
        for v in node:
            yield from iter_strings(v)
    elif isinstance(node, str):
        yield node


def value_has_credential(value: str) -> bool:
    def strip(m: "re.Match[str]") -> str:
        return "://" if _REF_ONLY_PASSWORD.fullmatch(m.group(1)) else m.group(0)
    masked = _REF.sub("VAR", _USERINFO.sub(strip, value))
    return redact_secrets(masked) != masked


def has_embedded_credential(doc: Any) -> bool:
    return any(value_has_credential(v) for v in iter_strings(doc))

"""Tiny deterministic YAML emitter for the generated Compose/Kubernetes files (no PyYAML dependency).

Supports dict/list/str/int/float/bool/None. Strings that are not plainly safe are emitted as double-quoted JSON
strings, which are valid YAML. Multiple documents are joined with ``---``.
"""
from __future__ import annotations

import json
import re
from typing import Any

_PLAIN = re.compile(r"^[A-Za-z_][A-Za-z0-9_./-]*$")
_RESERVED = {"true", "false", "yes", "no", "on", "off", "null", "~", "y", "n"}


def scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    s = str(value)
    if _PLAIN.match(s) and s.lower() not in _RESERVED and not s.endswith(("-", ".", "/")):
        return s
    return json.dumps(s, ensure_ascii=False)


def _emit(value: Any, indent: int, out: list[str]) -> None:
    pad = "  " * indent
    if isinstance(value, dict):
        for k, v in value.items():
            key = scalar(k)
            if isinstance(v, dict) and v:
                out.append(f"{pad}{key}:")
                _emit(v, indent + 1, out)
            elif isinstance(v, list) and v:
                out.append(f"{pad}{key}:")
                _emit(v, indent, out)  # block sequences at the same indent as the key
            else:
                out.append(f"{pad}{key}: {_inline(v)}")
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict) and item:
                sub: list[str] = []
                _emit(item, indent + 1, sub)
                sub[0] = f"{pad}- " + sub[0].lstrip()
                out.extend(sub)
            elif isinstance(item, list) and item:
                out.append(f"{pad}-")
                _emit(item, indent + 1, out)
            else:
                out.append(f"{pad}- {_inline(item)}")
    else:
        out.append(f"{pad}{scalar(value)}")


def _inline(v: Any) -> str:
    if isinstance(v, dict):
        return "{}" if not v else json.dumps(v)
    if isinstance(v, list):
        return "[]" if not v else json.dumps(v)
    return scalar(v)


def dump(value: Any) -> str:
    out: list[str] = []
    _emit(value, 0, out)
    return "\n".join(out) + "\n"


def dump_all(docs: list[Any]) -> str:
    return "---\n".join(dump(d) for d in docs)

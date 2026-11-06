"""Docker-compatible .dockerignore matching (patterns are relative to the build-context root)."""
from __future__ import annotations

import re
from functools import lru_cache


@lru_cache(maxsize=512)
def _compile(pattern: str) -> re.Pattern[str]:
    """Translate a .dockerignore glob to a regex: ``*`` stays inside one path segment, ``**`` crosses them."""
    out, i = [], 0
    while i < len(pattern):
        c = pattern[i]
        if c == "*":
            if pattern[i:i + 2] == "**":
                i += 2
                if pattern[i:i + 1] == "/":
                    i += 1
                    out.append("(?:.*/)?")
                else:
                    out.append(".*")
                continue
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        else:
            out.append(re.escape(c))
        i += 1
    return re.compile("".join(out) + r"\Z")


def parse_patterns(content: str) -> list[tuple[bool, str]]:
    """Return ``(negated, pattern)`` pairs in file order."""
    result: list[tuple[bool, str]] = []
    for raw in content.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        negated = line.startswith("!")
        if negated:
            line = line[1:].strip()
        line = line.lstrip("/").rstrip("/")
        line = re.sub(r"^\./", "", line)
        if line:
            result.append((negated, line))
    return result


def is_ignored(rel_path: str, content: str) -> bool:
    """True when ``rel_path`` (POSIX, relative to the context root) is excluded. The last matching pattern wins."""
    rel = rel_path.strip("/").lstrip("./") if rel_path.startswith("./") else rel_path.strip("/")
    parts = rel.split("/")
    ancestors = ["/".join(parts[:i]) for i in range(1, len(parts) + 1)]  # a, a/b, a/b/c
    ignored = False
    for negated, pattern in parse_patterns(content):
        rx = _compile(pattern)
        if any(rx.match(a) for a in ancestors):  # matching a parent directory excludes everything below it
            ignored = not negated
    return ignored

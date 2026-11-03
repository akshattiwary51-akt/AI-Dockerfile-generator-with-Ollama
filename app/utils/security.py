"""Security helpers: secret redaction, sensitive-file detection, path confinement."""
from __future__ import annotations

import fnmatch
import re
from pathlib import Path

SENSITIVE_FILE_PATTERNS = [
    ".env", ".env.*", "*.env", "*.pem", "*.key", "*.p12", "*.pfx", "*.jks", "*.keystore",
    "id_rsa", "id_rsa.*", "id_dsa", "id_ecdsa", "id_ed25519", "*.crt", "*.cer",
    "credentials", "credentials.json", "secrets.json", "secrets.yml", "secrets.yaml",
    ".npmrc", ".pypirc", ".netrc", ".htpasswd", "*.kdbx", "service-account*.json",
]
# Templates are safe and useful to read for env-var discovery.
SAFE_ENV_TEMPLATES = {".env.example", ".env.sample", ".env.template", ".env.dist"}

EXCLUDED_DIRS = {
    ".git", ".hg", ".svn", "node_modules", ".venv", "venv", "env", "__pycache__", ".mypy_cache",
    ".pytest_cache", ".tox", "dist", "build", "target", ".next", ".nuxt", ".cache", ".idea",
    ".vscode", "coverage", ".gradle", "vendor", "site-packages", ".terraform", ".ssh",
}
BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".webp", ".pdf", ".zip", ".tar", ".gz", ".tgz", ".7z",
    ".jar", ".war", ".class", ".so", ".dll", ".exe", ".bin", ".dylib", ".pyc", ".woff", ".woff2",
    ".ttf", ".eot", ".mp3", ".mp4", ".mov", ".sqlite", ".db", ".whl", ".o", ".a",
}

_REDACTED = "[REDACTED]"

_REDACTION_PATTERNS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{16,}"),
    # scheme://user:password@host
    re.compile(r"(?i)\b([a-z][a-z0-9+.\-]*://[^\s:/@]+:)([^\s@/]+)(@)"),
]
# KEY=value / "key": "value" where key looks sensitive
_KV_PATTERN = re.compile(
    r"(?i)\b([A-Za-z0-9_.\-]*(?:password|passwd|secret|token|api[_-]?key|access[_-]?key|private[_-]?key|credential)[A-Za-z0-9_.\-]*)"
    r"(\s*[=:]\s*)(\"[^\"\n]*\"|'[^'\n]*'|[^\s\"',;]+)"
)


def redact_secrets(text: str) -> str:
    """Mask credentials/secrets found in free text (logs, Dockerfiles, errors)."""
    if not text:
        return text
    out = _REDACTION_PATTERNS[0].sub(_REDACTED, text)
    for pat in _REDACTION_PATTERNS[1:-1]:
        out = pat.sub(_REDACTED, out)
    out = _REDACTION_PATTERNS[-1].sub(lambda m: f"{m.group(1)}{_REDACTED}{m.group(3)}", out)
    out = _KV_PATTERN.sub(lambda m: f"{m.group(1)}{m.group(2)}{_REDACTED}", out)
    return out


def is_sensitive_file(path: Path | str) -> bool:
    name = Path(path).name
    if name.lower() in SAFE_ENV_TEMPLATES:
        return False
    lowered = name.lower()
    return any(fnmatch.fnmatch(lowered, pat.lower()) for pat in SENSITIVE_FILE_PATTERNS)


def is_excluded_dir(name: str) -> bool:
    return name in EXCLUDED_DIRS or name.endswith(".egg-info")


def is_binary_extension(path: Path | str) -> bool:
    return Path(path).suffix.lower() in BINARY_EXTENSIONS


def ensure_within(base: Path, target: Path) -> Path:
    """Resolve ``target`` and assert it stays inside ``base`` (blocks traversal and symlink escape)."""
    base_r = base.resolve()
    target_r = target.resolve()
    try:
        target_r.relative_to(base_r)
    except ValueError as exc:
        raise PermissionError(f"Refusing to access '{target}': outside project directory '{base_r}'.") from exc
    return target_r


def truncate_log(text: str, max_bytes: int) -> str:
    """Keep head and tail of oversized logs (errors are usually at the end)."""
    raw = text.encode("utf-8", errors="replace")
    if len(raw) <= max_bytes:
        return text
    half = max_bytes // 2
    head = raw[:half].decode("utf-8", errors="ignore")
    tail = raw[-half:].decode("utf-8", errors="ignore")
    return f"{head}\n... [log truncated: {len(raw) - max_bytes} bytes omitted] ...\n{tail}"

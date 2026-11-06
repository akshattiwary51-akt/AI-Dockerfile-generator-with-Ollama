"""Environment-driven configuration validated with Pydantic."""
from __future__ import annotations

import os
from functools import lru_cache

from pydantic import BaseModel, Field, field_validator

try:  # optional: load a local .env file if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


class Settings(BaseModel):
    ollama_host: str = Field(default_factory=lambda: _env("OLLAMA_HOST", "http://localhost:11434"), validate_default=True)
    ollama_model: str = Field(default_factory=lambda: _env("OLLAMA_MODEL", "qwen2.5-coder:7b"))
    temperature: float = Field(default_factory=lambda: float(_env("OLLAMA_TEMPERATURE", "0.1")), ge=0, le=2)
    ollama_timeout: int = Field(default_factory=lambda: int(_env("OLLAMA_TIMEOUT", "300")), gt=0)
    num_ctx: int = Field(default_factory=lambda: int(_env("OLLAMA_NUM_CTX", "8192")), ge=512)
    max_retries: int = Field(default_factory=lambda: int(_env("MAX_RETRIES", "3")), ge=0, le=20)
    docker_build_timeout: int = Field(default_factory=lambda: int(_env("DOCKER_BUILD_TIMEOUT", "600")), gt=0)
    max_log_bytes: int = Field(default_factory=lambda: int(_env("MAX_LOG_BYTES", "200000")), gt=1000)
    max_context_chars: int = Field(default_factory=lambda: int(_env("MAX_CONTEXT_CHARS", "12000")), gt=500)
    max_file_bytes: int = Field(default_factory=lambda: int(_env("MAX_FILE_BYTES", "100000")), gt=100)

    @field_validator("ollama_host")
    @classmethod
    def _strip_host(cls, v: str) -> str:
        v = v.strip().rstrip("/")
        if not v.startswith(("http://", "https://")):
            v = "http://" + v
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reload_settings() -> Settings:
    get_settings.cache_clear()
    return get_settings()

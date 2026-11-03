"""Pydantic models describing the result of project analysis."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class DependencyInfo(BaseModel):
    """Raw dependency facts extracted from manifests of a single project unit."""

    manifests: list[str] = Field(default_factory=list)
    package_manager: Optional[str] = None
    lockfile: Optional[str] = None
    dependencies: list[str] = Field(default_factory=list)
    dev_dependencies: list[str] = Field(default_factory=list)
    runtime_version: Optional[str] = None
    scripts: dict[str, str] = Field(default_factory=dict)
    module_name: Optional[str] = None
    main_entry: Optional[str] = None


class ProjectUnit(BaseModel):
    """One buildable component (a repo root or a sub-project in a monorepo)."""

    path: str = "."
    language: str = "unknown"
    framework: Optional[str] = None
    role: str = "unknown"  # backend | frontend | fullstack | cli | library | unknown
    package_manager: Optional[str] = None
    lockfile: Optional[str] = None
    runtime_version: Optional[str] = None
    manifests: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    dev_dependencies: list[str] = Field(default_factory=list)
    scripts: dict[str, str] = Field(default_factory=dict)
    entry_points: list[str] = Field(default_factory=list)
    ports: list[int] = Field(default_factory=list)
    build_output_dir: Optional[str] = None
    notes: list[str] = Field(default_factory=list)


class Strategy(BaseModel):
    """Containerization strategy suggested by static analysis (an input to the LLM, not a command)."""

    base_image: str
    runtime_image: Optional[str] = None
    multi_stage: bool = False
    start_command_hint: Optional[str] = None
    notes: list[str] = Field(default_factory=list)


class ProjectAnalysis(BaseModel):
    name: str
    root: str
    language: str = "unknown"
    framework: Optional[str] = None
    is_monorepo: bool = False
    has_frontend_backend_split: bool = False
    primary_unit: Optional[ProjectUnit] = None
    units: list[ProjectUnit] = Field(default_factory=list)
    strategy: Optional[Strategy] = None
    env_vars: list[str] = Field(default_factory=list)  # names only, never values
    secret_env_vars: list[str] = Field(default_factory=list)
    existing_docker_files: list[str] = Field(default_factory=list)
    existing_dockerfile_content: Optional[str] = None
    structure: str = ""
    key_files: dict[str, str] = Field(default_factory=dict)
    file_count: int = 0
    skipped_sensitive_files: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def has_dockerfile(self) -> bool:
        return any(f == "Dockerfile" or f.endswith("/Dockerfile") for f in self.existing_docker_files)

    def summary(self) -> str:
        pu = self.primary_unit
        fw = self.framework or "no framework detected"
        deps = len(pu.dependencies) if pu else 0
        return f"{self.language} / {fw} ({deps} dependencies)"

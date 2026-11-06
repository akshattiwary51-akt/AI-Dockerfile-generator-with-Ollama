"""Hardware-aware model selection aid (no extra dependencies, never downloads or switches models by itself).

All memory/speed figures are *estimates* derived from file size, parameter count and assumed memory bandwidth; they are
labelled as such wherever they are displayed.
"""
from __future__ import annotations

import math
import os
import platform
import shutil
import subprocess
from typing import Optional

from pydantic import BaseModel, Field

from app.llm.ollama_client import ModelInfo, estimate_tokens
from app.utils.config import Settings, get_settings

# (tag, approx. download size in GB for the default 4-bit quantization, note)
CODE_MODELS: list[tuple[str, float, str]] = [
    ("qwen2.5-coder:1.5b", 1.0, "tiny; weak at multi-step fixes"),
    ("qwen2.5-coder:3b", 1.9, "small; acceptable for simple projects"),
    ("qwen2.5-coder:7b", 4.7, "recommended default balance"),
    ("deepseek-coder-v2:16b", 8.9, "strong code model (MoE)"),
    ("qwen2.5-coder:14b", 9.0, "better reasoning for build fixes"),
    ("qwen2.5-coder:32b", 20.0, "best quality; needs a large GPU / lots of RAM"),
]
OVERHEAD_GB = 1.0              # runtime buffers
# Assumed memory bandwidth (GB/s). Token generation is memory-bound: tokens/s ~ efficiency * bandwidth / weights.
ASSUMED_BANDWIDTH = {"gpu": 300.0, "unified": 100.0, "cpu": 40.0, "partial": 30.0}
EFFICIENCY = 0.6
EXPECTED_OUTPUT_TOKENS = 1200  # typical Dockerfile JSON reply
PROMPT_OVERHEAD_TOKENS = 3500  # system prompt + schema + the reply itself


class Hardware(BaseModel):
    ram_gb: Optional[float] = None
    ram_available_gb: Optional[float] = None
    vram_gb: Optional[float] = None
    vram_free_gb: Optional[float] = None
    gpu_name: str = ""
    unified_memory: bool = False   # Apple Silicon: GPU shares system RAM
    os: str = ""


def parse_linux_meminfo(text: str) -> tuple[Optional[float], Optional[float]]:
    vals: dict[str, float] = {}
    for line in text.splitlines():
        parts = line.replace(":", " ").split()
        if len(parts) >= 2 and parts[0] in ("MemTotal", "MemAvailable") and parts[1].isdigit():
            vals[parts[0]] = int(parts[1]) / 1024 ** 2  # kB -> GB
    return (round(vals["MemTotal"], 1) if "MemTotal" in vals else None,
            round(vals["MemAvailable"], 1) if "MemAvailable" in vals else None)


def _windows_memory() -> tuple[Optional[float], Optional[float]]:
    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]

    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
    return round(stat.ullTotalPhys / 1024 ** 3, 1), round(stat.ullAvailPhys / 1024 ** 3, 1)


def _memory() -> tuple[Optional[float], Optional[float]]:
    system = platform.system()
    try:
        if system == "Linux":
            with open("/proc/meminfo", encoding="utf-8") as fh:
                return parse_linux_meminfo(fh.read())
        if system == "Windows":
            return _windows_memory()
        if system == "Darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5).stdout
            return round(int(out.strip()) / 1024 ** 3, 1), None
    except (OSError, ValueError, subprocess.SubprocessError, AttributeError):
        pass
    try:  # POSIX fallback
        return round(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024 ** 3, 1), None
    except (ValueError, OSError, AttributeError):
        return None, None


def parse_nvidia_smi(output: str) -> tuple[str, Optional[float], Optional[float]]:
    best: tuple[str, float, float] = ("", 0.0, 0.0)
    for line in output.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                total, free = float(parts[1]) / 1024, float(parts[2]) / 1024
            except ValueError:
                continue
            if total > best[1]:
                best = (parts[0], total, free)
    return (best[0], round(best[1], 1), round(best[2], 1)) if best[1] else ("", None, None)


def detect_hardware() -> Hardware:
    ram, avail = _memory()
    hw = Hardware(ram_gb=ram, ram_available_gb=avail, os=platform.system())
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        hw.unified_memory = True
        hw.gpu_name = "Apple Silicon (unified memory)"
        return hw
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.run([smi, "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5, check=True).stdout
            hw.gpu_name, hw.vram_gb, hw.vram_free_gb = parse_nvidia_smi(out)
        except (subprocess.SubprocessError, OSError):
            pass
    return hw


def _budgets(hw: Hardware) -> tuple[Optional[float], Optional[float]]:
    """(gpu_budget_gb, cpu_budget_gb) of memory that can realistically be used for a model."""
    ram_budget = (hw.ram_available_gb or (hw.ram_gb * 0.6 if hw.ram_gb else None))
    if hw.ram_available_gb is not None:
        ram_budget = hw.ram_available_gb * 0.85
    if hw.unified_memory and hw.ram_gb:
        return hw.ram_gb * 0.66, hw.ram_gb * 0.66
    gpu = (hw.vram_free_gb or hw.vram_gb) if hw.vram_gb else None
    return (gpu * 0.95 if gpu else None), ram_budget


class ModelAssessment(BaseModel):
    name: str
    parameters_b: Optional[float] = None
    quantization: str = ""
    weights_gb: float = 0.0
    kv_cache_gb: float = 0.0
    total_gb: float = 0.0
    placement: str = "unknown"       # gpu | unified | partial | cpu | does not fit | unknown
    est_tokens_per_second: Optional[float] = None
    est_generation_seconds: Optional[float] = None
    context_supported: Optional[int] = None
    required_context: int = 0
    context_ok: bool = True
    verdict: str = "unknown"         # recommended | ok | slow | too large | unknown
    score: float = 0.0
    notes: list[str] = Field(default_factory=list)


def required_context_tokens(settings: Optional[Settings] = None) -> int:
    s = settings or get_settings()
    return estimate_tokens("x" * s.max_context_chars) + PROMPT_OVERHEAD_TOKENS


def assess_model(info: ModelInfo, hw: Hardware, settings: Optional[Settings] = None) -> ModelAssessment:
    s = settings or get_settings()
    params = info.parameters_billion
    weights = info.size_bytes / 1e9 if info.size_bytes else (params * 0.6 if params else 0.0)
    kv = round((params or 7) * 0.012 * (s.num_ctx / 1000), 2)  # rough GQA-era estimate
    total = round(weights + kv + OVERHEAD_GB, 2)
    req = required_context_tokens(s)
    a = ModelAssessment(name=info.name, parameters_b=params, quantization=info.quantization,
                        weights_gb=round(weights, 2), kv_cache_gb=kv, total_gb=total,
                        context_supported=info.context_length, required_context=req)
    if info.context_length and info.context_length < s.num_ctx:
        a.notes.append(f"model supports only {info.context_length} tokens of context; OLLAMA_NUM_CTX is {s.num_ctx}")
    if s.num_ctx < req:
        a.context_ok = False
        a.notes.append(f"this app needs about {req} tokens of context but OLLAMA_NUM_CTX is {s.num_ctx}; "
                       "raise it or lower MAX_CONTEXT_CHARS")
    gpu_budget, cpu_budget = _budgets(hw)
    bw_key: Optional[str] = None
    if hw.unified_memory and gpu_budget and total <= gpu_budget:
        a.placement, bw_key = "unified", "unified"
    elif gpu_budget and total <= gpu_budget:
        a.placement, bw_key = "gpu", "gpu"
    elif gpu_budget and cpu_budget and total <= gpu_budget + cpu_budget and total <= (cpu_budget or 0) + gpu_budget:
        a.placement, bw_key = "partial", "partial"
        a.notes.append("only partly fits in VRAM; layers spill to system RAM, which is much slower")
    elif cpu_budget and total <= cpu_budget:
        a.placement, bw_key = "cpu", "cpu"
    elif gpu_budget is None and cpu_budget is None:
        a.placement = "unknown"
        a.notes.append("could not detect RAM/VRAM; fit unknown")
    else:
        a.placement = "does not fit"
    if bw_key and weights > 0:
        tps = EFFICIENCY * ASSUMED_BANDWIDTH[bw_key] / weights
        a.est_tokens_per_second = round(tps, 1)
        a.est_generation_seconds = round(EXPECTED_OUTPUT_TOKENS / tps, 0)
        a.notes.append(f"speed is an estimate assuming ~{ASSUMED_BANDWIDTH[bw_key]:.0f} GB/s memory bandwidth")
    if a.placement == "does not fit":
        a.verdict = "too large"
    elif a.placement == "unknown":
        a.verdict = "unknown"
    elif a.est_generation_seconds and a.est_generation_seconds > 300:
        a.verdict = "slow"
    else:
        a.verdict = "ok"
    if a.verdict in ("ok", "slow"):
        code_bonus = 1.3 if any(k in info.name.lower() for k in ("coder", "code")) else 1.0
        a.score = min(params or 3, 32) * code_bonus * (0.4 if a.verdict == "slow" else 1.0) * (1.0 if a.context_ok else 0.8)
    return a


def assess_installed(models: list[ModelInfo], hw: Optional[Hardware] = None,
                     settings: Optional[Settings] = None) -> list[ModelAssessment]:
    """Rank installed models for this machine; the best 'ok' model is marked ``recommended`` (advice only)."""
    hw = hw or detect_hardware()
    results = sorted((assess_model(m, hw, settings) for m in models), key=lambda r: -r.score)
    for r in results:
        if r.verdict == "ok":
            r.verdict = "recommended"
            break
    return results


def recommend_models(hw: Optional[Hardware] = None) -> list[tuple[str, str]]:
    """Models worth ``ollama pull``-ing for this machine (largest that fits first). Advice only."""
    hw = hw or detect_hardware()
    gpu_budget, cpu_budget = _budgets(hw)
    budget = gpu_budget or cpu_budget
    where = "VRAM" if gpu_budget else "RAM"
    if budget is None:
        return [(tag, f"~{size} GB; hardware not detected - verify it fits") for tag, size, _ in CODE_MODELS[:3]]
    fits = sorted(((t, s, n) for t, s, n in CODE_MODELS if s + OVERHEAD_GB + 0.7 <= budget), key=lambda x: -x[1])
    return [(t, f"~{s} GB of {budget:.1f} GB usable {where}; {n}") for t, s, n in fits]

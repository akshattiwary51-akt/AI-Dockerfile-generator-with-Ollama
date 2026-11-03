"""Dependency-free model-selection aid: recommend Ollama models that fit available RAM/VRAM.

This is an optional convenience; nothing else in the application depends on it.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

# (tag, approx. size in GB for the default 4-bit quantization, note)
CODE_MODELS: list[tuple[str, float, str]] = [
    ("qwen2.5-coder:1.5b", 1.0, "tiny; weak at multi-step fixes"),
    ("qwen2.5-coder:3b", 1.9, "small; acceptable for simple projects"),
    ("qwen2.5-coder:7b", 4.7, "recommended default balance"),
    ("deepseek-coder-v2:16b", 8.9, "strong code model (MoE)"),
    ("qwen2.5-coder:14b", 9.0, "better reasoning for build fixes"),
    ("qwen2.5-coder:32b", 20.0, "best quality; needs a large GPU / lots of RAM"),
]
OVERHEAD_GB = 2.0  # KV cache and runtime headroom


@dataclass
class Hardware:
    ram_gb: Optional[float]
    vram_gb: Optional[float]


def detect_hardware() -> Hardware:
    ram = None
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        size = os.sysconf("SC_PAGE_SIZE")
        ram = round(pages * size / 1024 ** 3, 1)
    except (ValueError, OSError, AttributeError):
        pass
    vram = None
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.run([smi, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5, check=True).stdout
            vram = round(max(float(x) for x in out.split()) / 1024, 1)
        except (subprocess.SubprocessError, ValueError, OSError):
            pass
    return Hardware(ram_gb=ram, vram_gb=vram)


def recommend_models(hw: Optional[Hardware] = None) -> list[tuple[str, str]]:
    """Return models (largest that fits first) with a short reason."""
    hw = hw or detect_hardware()
    budget = hw.vram_gb if hw.vram_gb else (hw.ram_gb * 0.7 if hw.ram_gb else None)
    where = "VRAM" if hw.vram_gb else "RAM"
    if budget is None:
        return [(tag, f"~{size} GB; hardware not detected - verify it fits") for tag, size, _ in CODE_MODELS[:3]]
    fits = [(t, s, n) for t, s, n in CODE_MODELS if s + OVERHEAD_GB <= budget]
    fits.sort(key=lambda x: -x[1])
    return [(t, f"~{s} GB of {budget:.1f} GB usable {where}; {n}") for t, s, n in fits]

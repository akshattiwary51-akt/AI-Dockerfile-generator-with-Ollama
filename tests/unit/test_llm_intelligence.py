"""Phase 3: latency/token tracking, model details, warm-up, timeout retry, hardware-aware assessment, benchmark."""
from __future__ import annotations

import pytest
import requests

from app.agent.orchestrator import Orchestrator
from app.analyzer.project_scanner import ProjectScanner
from app.llm.benchmark import benchmark_models
from app.llm.model_advisor import (Hardware, assess_installed, assess_model, parse_linux_meminfo, parse_nvidia_smi,
                                   recommend_models, required_context_tokens)
from app.llm.ollama_client import ModelInfo, OllamaClient, OllamaError, estimate_tokens
from tests.conftest import FLASK_FILES, FakeBuilder, ScriptedClient, llm_json
from tests.unit.test_ollama import FakeResponse, FakeSession, chat_ok


def chat_with_stats(content="{}", prompt=1200, completion=300, eval_ns=3_000_000_000, load_ns=2_000_000_000):
    return FakeResponse(200, {"message": {"content": content}, "prompt_eval_count": prompt, "eval_count": completion,
                              "eval_duration": eval_ns, "load_duration": load_ns})


def test_chat_records_latency_and_tokens(settings):
    client = OllamaClient(settings, FakeSession(post=[chat_with_stats()]))
    client.chat([{"role": "user", "content": "hi"}])
    s = client.last_stats
    assert (s.prompt_tokens, s.completion_tokens, s.tokens_per_second) == (1200, 300, 100.0)
    assert s.load_seconds == 2.0 and not s.estimated and s.context_window == 4096
    assert 0.3 < s.context_used_fraction < 0.4 and len(client.stats_log) == 1


def test_tokens_estimated_when_ollama_reports_none(settings):
    client = OllamaClient(settings, FakeSession(post=[chat_ok("x" * 700)]))
    client.chat([{"role": "user", "content": "y" * 350}])
    assert client.last_stats.estimated and client.last_stats.completion_tokens == estimate_tokens("x" * 700) == 200
    assert client.last_stats.prompt_tokens == 100


def test_truncation_warning_flag(settings):
    client = OllamaClient(settings, FakeSession(post=[chat_with_stats(prompt=4096)]))
    client.chat([{"role": "user", "content": "hi"}])
    assert client.last_stats.prompt_may_be_truncated


def test_timeout_is_retried_with_longer_timeout(settings):
    s = settings.model_copy(update={"ollama_timeout_retries": 1})
    session = FakeSession(post=[requests.Timeout(), chat_ok("{}")])
    assert OllamaClient(s, session).chat([{"role": "user", "content": "x"}]) == "{}"
    assert [p["timeout"] for p in session.posts] == [30.0, 45.0]
    with pytest.raises(OllamaError):
        OllamaClient(s, FakeSession(post=[requests.Timeout(), requests.Timeout()])).chat([{"role": "user", "content": "x"}])


def test_model_details_and_context_length(settings):
    tags = FakeResponse(200, {"models": [
        {"name": "qwen2.5-coder:7b", "size": 4_683_000_000,
         "details": {"parameter_size": "7.6B", "quantization_level": "Q4_K_M", "family": "qwen2"}},
        {"name": "tiny:latest", "size": 700_000_000, "details": {"parameter_size": "900M"}}]})
    show = FakeResponse(200, {"model_info": {"qwen2.context_length": 32768, "general.x": 1}})
    client = OllamaClient(settings, FakeSession(get=[tags], post=[show, FakeResponse(404, {})]))
    infos = client.list_model_details(with_context=True)
    assert infos[0].parameters_billion == 7.6 and infos[0].quantization == "Q4_K_M" and infos[0].context_length == 32768
    assert infos[1].parameters_billion == 0.9 and infos[1].context_length is None


def test_warm_up_is_non_fatal(settings):
    assert OllamaClient(settings, FakeSession(post=[FakeResponse(200, {})])).warm_up() is not None
    assert OllamaClient(settings, FakeSession(post=[requests.ConnectionError()])).warm_up() is None


def test_hardware_parsers():
    meminfo = "MemTotal:       16384000 kB\nMemFree: 1 kB\nMemAvailable:    8192000 kB\n"
    assert parse_linux_meminfo(meminfo) == (15.6, 7.8)
    assert parse_linux_meminfo("garbage") == (None, None)
    name, total, free = parse_nvidia_smi("NVIDIA RTX 3060, 12288, 11000\nNVIDIA T4, 16384, 100\n")
    assert (name, total) == ("NVIDIA T4", 16.0) and free == 0.1
    assert parse_nvidia_smi("") == ("", None, None)


def info(name, size_gb, params, quant="Q4_K_M", ctx=None):
    return ModelInfo(name=name, size_bytes=int(size_gb * 1e9), parameter_size=params, quantization=quant,
                     context_length=ctx)


CPU_16 = Hardware(ram_gb=16, ram_available_gb=10)
GPU_12 = Hardware(ram_gb=32, ram_available_gb=24, vram_gb=12, vram_free_gb=11, gpu_name="RTX")
TINY_4 = Hardware(ram_gb=4, ram_available_gb=2)


def test_assess_cpu_gpu_and_too_large(settings):
    big = settings.model_copy(update={"num_ctx": 8192})
    cpu = assess_model(info("qwen2.5-coder:7b", 4.7, "7.6B"), CPU_16, big)
    assert cpu.placement == "cpu" and cpu.verdict == "ok" and 5 < cpu.est_tokens_per_second < 8
    assert any("estimate" in n for n in cpu.notes) and cpu.context_ok
    gpu = assess_model(info("qwen2.5-coder:7b", 4.7, "7.6B"), GPU_12, big)
    assert gpu.placement == "gpu" and gpu.est_tokens_per_second > cpu.est_tokens_per_second * 3
    huge = assess_model(info("qwen2.5-coder:32b", 20, "32.8B"), CPU_16, big)
    assert huge.placement == "does not fit" and huge.verdict == "too large" and huge.score == 0
    part = assess_model(info("qwen2.5-coder:14b", 9.0, "14.8B"), Hardware(ram_gb=32, ram_available_gb=24, vram_gb=6,
                                                                       vram_free_gb=6), big)
    assert part.placement == "partial" and any("spill" in n for n in part.notes)
    nohw = assess_model(info("m", 4.7, "7B"), Hardware(), big)
    assert nohw.verdict == "unknown"


def test_assess_slow_and_context_warnings(settings):
    s8 = settings.model_copy(update={"num_ctx": 8192})
    tight = assess_model(info("qwen2.5-coder:14b", 9.0, "14.8B"), Hardware(ram_gb=16, ram_available_gb=12), s8)
    assert tight.placement == "does not fit"  # 9 GB weights + ~1.5 GB KV cache + overhead exceeds 85% of 12 GB free
    slow = assess_model(info("qwen2.5-coder:14b", 9.0, "14.8B"), Hardware(ram_gb=32, ram_available_gb=20), s8)
    assert slow.placement == "cpu" and slow.verdict == "slow" and slow.est_generation_seconds > 300
    small_ctx = assess_model(info("m", 2, "3B", ctx=2048), CPU_16, settings.model_copy(update={"num_ctx": 2048}))
    assert not small_ctx.context_ok and any("needs about" in n for n in small_ctx.notes)
    assert required_context_tokens(settings) > 3500
    capped = assess_model(info("m", 2, "3B", ctx=2048), CPU_16, settings.model_copy(update={"num_ctx": 8192}))
    assert any("supports only 2048" in n for n in capped.notes)


def test_ranking_prefers_best_model_that_fits(settings):
    s = settings.model_copy(update={"num_ctx": 8192})
    ranked = assess_installed([info("qwen2.5-coder:3b", 1.9, "3.1B"), info("qwen2.5-coder:7b", 4.7, "7.6B"),
                               info("qwen2.5-coder:32b", 20, "32.8B"), info("llama3:8b", 4.7, "8B")], CPU_16, s)
    assert ranked[0].name == "qwen2.5-coder:7b" and ranked[0].verdict == "recommended"  # code model beats llama3:8b
    assert ranked[-1].name == "qwen2.5-coder:32b" and ranked[-1].verdict == "too large"
    assert sum(r.verdict == "recommended" for r in ranked) == 1


def test_pull_recommendations_respect_memory():
    assert recommend_models(TINY_4) == []  # nothing listed fits in ~1.7 GB usable RAM
    on_12 = [m for m, _ in recommend_models(GPU_12)]
    assert on_12[0] == "qwen2.5-coder:7b" and "qwen2.5-coder:14b" not in on_12  # 14B + KV cache > 11 GB free
    on_16 = [m for m, _ in recommend_models(Hardware(ram_gb=32, ram_available_gb=24, vram_gb=16, vram_free_gb=16))]
    assert "qwen2.5-coder:14b" in on_16 and "qwen2.5-coder:32b" not in on_16
    assert len(recommend_models(Hardware())) == 3  # unknown hardware -> cautious defaults


def test_benchmark_ranks_models_and_survives_failures(make_project, settings):
    analysis = ProjectScanner(settings).scan(make_project(FLASK_FILES))
    bad_json = "not json"
    scripts = {"good:7b": [llm_json()], "retry:3b": [bad_json, llm_json()],
               "broken:1b": [bad_json, bad_json, bad_json]}

    def factory(s):
        return ScriptedClient(list(scripts[s.ollama_model]), model=s.ollama_model)

    results = benchmark_models(analysis, ["broken:1b", "retry:3b", "good:7b"], settings, client_factory=factory)
    assert [r.model for r in results] == ["good:7b", "retry:3b", "broken:1b"]
    good, retry, broken = results
    assert good.ok and good.format_retries == 0 and good.lint_score > 0
    assert retry.ok and retry.format_retries == 1
    assert not broken.ok and "3 attempts" in broken.error


def test_environment_check_is_cached(settings):
    calls = {"n": 0}

    class Counting(FakeBuilder):
        def check_docker(self):
            calls["n"] += 1
            return True, "ok"

    orch = Orchestrator(settings, client=OllamaClient(settings, FakeSession(get=[requests.ConnectionError()] * 10)),
                        builder=Counting([]))
    orch.check_environment(max_age=60)
    orch.check_environment(max_age=60)
    assert calls["n"] == 1
    orch.check_environment()  # no max_age -> fresh probe
    assert calls["n"] == 2

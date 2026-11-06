"""Command-line interface: doctor | models | analyze | run."""
from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path
from typing import Optional, Sequence

from app.agent.feedback_loop import LoopEvent
from app.agent.orchestrator import BUILD_WARNING, ApprovalRequired, Orchestrator
from app.docker.dockerfile_generator import FileProtectionError, GenerationError
from app.docker.docker_validator import validate_artifacts
from app.llm.model_advisor import detect_hardware, recommend_models
from app.llm.ollama_client import OllamaError
from app.utils.config import get_settings, reload_settings
from app.utils.logger import get_logger

log = get_logger("dockerfile_ai.cli")
EXIT_OK, EXIT_BUILD_FAILED, EXIT_ENV, EXIT_DECLINED = 0, 1, 2, 3


def _confirm(question: str, assume_yes: bool) -> bool:
    if assume_yes:
        print(f"{question} [auto-approved by flag]")
        return True
    if not sys.stdin or not sys.stdin.isatty():
        print(f"{question} -> not approved (non-interactive; pass the matching --yes-* flag to approve)")
        return False
    return input(f"{question} [y/N]: ").strip().lower() in {"y", "yes"}


def cmd_doctor(_: argparse.Namespace) -> int:
    orch = Orchestrator()
    st = orch.check_environment()
    print(f"Ollama : {'OK' if st.ollama_available else 'UNAVAILABLE'} - {st.ollama_message}")
    print(f"Model  : {st.model} - {'installed' if st.model_installed else 'NOT installed'}")
    if st.installed_models:
        print(f"         installed models: {', '.join(st.installed_models)}")
    print(f"Docker : {'OK' if st.docker_available else 'UNAVAILABLE'} - {st.docker_message}")
    return EXIT_OK if (st.ready_to_generate and st.docker_available) else EXIT_ENV


def cmd_models(_: argparse.Namespace) -> int:
    hw = detect_hardware()
    ram = f"{hw.ram_gb} GB" if hw.ram_gb else "unknown"
    vram = f"{hw.vram_gb} GB" if hw.vram_gb else "none detected"
    print(f"Detected RAM: {ram}, GPU VRAM: {vram}")
    recs = recommend_models(hw)
    if not recs:
        print("No listed code model fits comfortably; try qwen2.5-coder:1.5b or add RAM/VRAM.")
    for tag, why in recs:
        print(f"  ollama pull {tag:<26} # {why}")
    return EXIT_OK


def cmd_analyze(args: argparse.Namespace) -> int:
    try:
        analysis = Orchestrator().analyze(args.project)
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ENV
    print(analysis.model_dump_json(indent=2, exclude={"key_files"}))
    return EXIT_OK


def cmd_run(args: argparse.Namespace) -> int:
    if args.model:
        import os
        os.environ["OLLAMA_MODEL"] = args.model
    if args.max_retries is not None:
        import os
        os.environ["MAX_RETRIES"] = str(args.max_retries)
    settings = reload_settings()
    orch = Orchestrator(settings)
    project = Path(args.project).expanduser()

    try:
        print(f"[1/5] Analyzing {project} ...")
        analysis = orch.analyze(project)
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ENV
    print(f"      {analysis.summary()}; {analysis.file_count} files scanned, "
          f"{len(analysis.skipped_sensitive_files)} sensitive file(s) excluded")
    for w in analysis.warnings:
        print(f"      warning: {w}")

    print(f"[2/5] Generating Dockerfile with Ollama model '{settings.ollama_model}' ...")
    try:
        artifacts = orch.generate(analysis)
    except OllamaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ENV
    except GenerationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_BUILD_FAILED
    print("\n----- Dockerfile -----\n" + artifacts.dockerfile + "----- .dockerignore -----\n" + artifacts.dockerignore)
    print(f"Explanation: {artifacts.explanation}")
    for lim in artifacts.limitations:
        print(f"Limitation : {lim}")
    report = validate_artifacts(artifacts.dockerfile, artifacts.dockerignore)
    for w in report.warnings:
        print(f"lint warning: {w.message}")

    print("[3/5] Saving files ...")
    existing = orch.existing_files(project)
    approve_overwrite = False
    if existing:
        print(f"      Existing file(s) found: {', '.join(existing)}")
        diff = orch.diff_with_existing(project, artifacts)
        if diff:
            print(diff)
        approve_overwrite = _confirm("Overwrite the existing file(s)? A .bak copy will be kept.", args.yes_overwrite)
        if not approve_overwrite:
            print("Existing files were not modified. Aborting.")
            return EXIT_DECLINED
    try:
        written = orch.save(project, artifacts, approve_overwrite=approve_overwrite)
    except FileProtectionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_DECLINED
    print("      wrote: " + ", ".join(str(p) for p in written))

    print("[4/5] Docker build\n      WARNING: " + BUILD_WARNING)
    if not _confirm("Start the Docker build now?", args.yes_build):
        print("Build skipped. Files were saved; build later with `docker build -t <name> .`")
        return EXIT_DECLINED
    ok, msg = orch.builder.check_docker()
    if not ok:
        print(f"error: {msg}", file=sys.stderr)
        return EXIT_ENV

    cancel = threading.Event()

    def on_event(e: LoopEvent) -> None:
        print(f"  >> [{e.kind}] {e.message}")

    def on_log(line: str) -> None:
        if args.verbose:
            print("     | " + line)

    print("[5/5] Building and self-correcting ...")
    try:
        result = orch.build_and_fix(project, analysis, artifacts, approve_build=True, on_event=on_event,
                                    on_log=on_log, cancel_event=cancel)
    except KeyboardInterrupt:
        cancel.set()
        print("\nCancelled.")
        return EXIT_DECLINED
    except ApprovalRequired as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_DECLINED

    print(f"\n===== FINAL REPORT =====\nResult   : {'SUCCESS' if result.success else 'FAILED'} ({result.status})\n"
          f"Image    : {result.image}\nAttempts : {result.attempts} build(s), {result.corrections} correction(s)\n"
          f"Duration : {result.total_duration_seconds:.1f}s\nFiles    : {', '.join(result.files)}\n"
          f"Message  : {result.message}")
    for a in result.history.attempts:
        if a.diagnosis:
            print(f"  attempt {a.attempt}: {a.diagnosis}")
            for c in a.changes[:5]:
                print(f"      - {c}")
    for s in result.suggestions:
        print(f"Suggestion: {s}")
    if args.export:
        for p in orch.export_report(result):
            print(f"Exported: {p}")
    return EXIT_OK if result.success else EXIT_BUILD_FAILED


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dockerfile-ai", description="Generate and self-correct Dockerfiles with a "
                                                                   "local Ollama model.")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="check Ollama, the model and Docker").set_defaults(func=cmd_doctor)
    sub.add_parser("models", help="suggest Ollama models for this machine").set_defaults(func=cmd_models)
    a = sub.add_parser("analyze", help="print the project analysis as JSON")
    a.add_argument("project")
    a.set_defaults(func=cmd_analyze)
    r = sub.add_parser("run", help="analyze, generate, build and self-correct")
    r.add_argument("project")
    r.add_argument("--model", help="Ollama model (overrides OLLAMA_MODEL)")
    r.add_argument("--max-retries", type=int, help="correction attempts (overrides MAX_RETRIES)")
    r.add_argument("--yes-overwrite", action="store_true", help="approve overwriting an existing Dockerfile")
    r.add_argument("--yes-build", action="store_true", help="approve starting the Docker build")
    r.add_argument("--verbose", "-v", action="store_true", help="stream docker build output")
    r.add_argument("--export", action="store_true", help="write JSON/Markdown reports to generated/")
    r.set_defaults(func=cmd_run)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return EXIT_DECLINED


if __name__ == "__main__":
    raise SystemExit(main())

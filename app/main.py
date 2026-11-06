"""Command-line interface: doctor | models | analyze | run | fix | selftest | score | scan | optimize | compose | k8s | history."""
from __future__ import annotations

import argparse
import json
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


def cmd_models(args: argparse.Namespace) -> int:
    hw = detect_hardware()
    ram = f"{hw.ram_gb} GB" if hw.ram_gb else "unknown"
    vram = f"{hw.vram_gb} GB" if hw.vram_gb else "none detected"
    print(f"Detected RAM: {ram}, GPU VRAM: {vram}")
    recs = recommend_models(hw)
    if not recs:
        print("No listed code model fits comfortably; try qwen2.5-coder:1.5b or add RAM/VRAM.")
    for tag, why in recs:
        print(f"  ollama pull {tag:<26} # {why}  (suggestion only; nothing is downloaded automatically)")
    orch = Orchestrator(reload_settings())
    try:
        _, assessed = orch.assess_models()
    except OllamaError as exc:
        print(f"\nInstalled-model assessment unavailable: {exc}")
        return EXIT_ENV if args.benchmark else EXIT_OK
    if assessed:
        print("\nInstalled models (speed figures are estimates):")
        for a in assessed:
            speed = f"~{a.est_tokens_per_second:.0f} tok/s" if a.est_tokens_per_second else ""
            print(f"  {a.name:<32} {a.verdict:<12} ~{a.total_gb:.1f} GB  {a.placement:<10} {speed}")
    if args.benchmark:
        names = [m.strip() for m in args.models.split(",")] if args.models else [a.name for a in assessed[:2]]
        print(f"\nBenchmarking {', '.join(names)} on {args.benchmark} (real generations; may take minutes) ...")
        try:
            analysis = orch.analyze(args.benchmark)
        except (FileNotFoundError, NotADirectoryError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_ENV
        for r in orch.benchmark(analysis, names, on_progress=lambda m: print("  " + m)):
            print(f"  {r.model:<32} ok={r.ok} latency={r.latency_seconds:.1f}s quality={r.lint_score} "
                  f"tokens={r.prompt_tokens + r.completion_tokens} format_retries={r.format_retries}"
                  + (f" error={r.error}" if r.error else ""))
    return EXIT_OK


def _read_artifacts(project: Path, dockerfile: Optional[str]) -> tuple[str, str]:
    path = Path(dockerfile) if dockerfile else project / "Dockerfile"
    if not path.is_file():
        raise FileNotFoundError(f"No Dockerfile found at {path}")
    ignore = project / ".dockerignore"
    return (path.read_text(encoding="utf-8", errors="replace"),
            ignore.read_text(encoding="utf-8", errors="replace") if ignore.is_file() else "")


def cmd_score(args: argparse.Namespace) -> int:
    from app.docker.quality import score_dockerfile
    from app.security.analyzer import analyze_dockerfile_security
    try:
        analysis = Orchestrator(reload_settings()).analyze(args.project)
        df, di = _read_artifacts(Path(args.project), args.dockerfile)
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ENV
    q = score_dockerfile(df, di, analysis=analysis, project_root=args.project)
    sec = analyze_dockerfile_security(df, di, analysis=analysis, project_root=args.project)
    if args.json:
        print(json.dumps({"quality": q.model_dump(), "security": {**sec.model_dump(), "score": sec.score}}, indent=2))
    else:
        print(q.format())
        print()
        print(sec.format())
    return EXIT_BUILD_FAILED if args.min_score is not None and q.overall < args.min_score else EXIT_OK


def cmd_scan(args: argparse.Namespace) -> int:
    from app.security.scanners import scan_image
    reports = scan_image(args.image)
    worst = 0
    for r in reports:
        print(f"[{r.tool}]", end=" ")
        if not r.available:
            print(f"not available: {r.error}")
            continue
        if not r.ok:
            print(f"failed: {r.error}")
            continue
        print(", ".join(f"{k}={v}" for k, v in r.counts.items() if v) or "no vulnerabilities reported")
        for v in r.top(args.top):
            print(f"    {v.severity:<8} {v.id:<18} {v.package} {v.installed}"
                  + (f" -> fixed in {v.fixed}" if v.fixed else ""))
        for line in r.remediation() + r.notes:
            print("  - " + line)
        worst = max(worst, 2 if r.counts.get("CRITICAL") else 1 if r.counts.get("HIGH") else 0)
    if not any(r.ok for r in reports):
        print("No scanner produced results. Install Trivy (https://trivy.dev) or enable `docker scout`.", file=sys.stderr)
        return EXIT_ENV
    gate = {"critical": 2, "high": 1}.get(args.fail_on or "", 99)
    return EXIT_BUILD_FAILED if worst >= gate else EXIT_OK


def cmd_optimize(args: argparse.Namespace) -> int:
    from app.optimization.optimizer import (analyze_optimizations, build_and_compare, measure_image,
                                            optimize_dockerfile, write_optimized)
    project = Path(args.project)
    try:
        df, di = _read_artifacts(project, None)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ENV
    measured = measure_image(args.image) if args.image else None
    if measured:
        print(f"Measured size of {args.image}: {measured.size_mb} MB" if measured.available
              else f"Image not measured: {measured.error}")
    rep = analyze_optimizations(df, di, measured, project)
    for s in rep.suggestions:
        print(f"  - {s.message}" + (f" (estimated saving ~{s.estimated_saving_mb} MB, not measured)"
                                     if s.estimated_saving_mb else ""))
    print(rep.note)
    if not (args.write or args.build):
        return EXIT_OK
    if optimize_dockerfile(df) == df:
        print("The safe automatic rewrites would not change this Dockerfile; nothing to write.")
        return EXIT_OK
    path = write_optimized(project, df)
    print(f"wrote {path} (your Dockerfile is untouched)")
    if not args.build:
        return EXIT_OK
    if not args.image:
        print("error: --build needs --image (the already-built image to compare against)", file=sys.stderr)
        return EXIT_ENV
    if not _confirm("Build the optimized variant with Docker?", args.yes_build):
        return EXIT_DECLINED
    res, cmp = build_and_compare(Orchestrator(reload_settings()).builder, project, args.image,
                                 f"{args.image}-optimized", approved=True)
    print(cmp.summary() if res.success else f"Optimized build failed: {res.error_summary}")
    return EXIT_OK if res.success else EXIT_BUILD_FAILED


def cmd_history(args: argparse.Namespace) -> int:
    from app.history.store import HistoryStore
    store = HistoryStore(args.db or reload_settings().history_db or None)
    flt = dict(text=args.search or "", project=args.project or "", success=False if args.failed else None,
               limit=args.limit)
    if args.export == "json":
        print(store.export_json(**flt))
    elif args.export == "csv":
        print(store.export_csv(**flt), end="")
    elif args.stats:
        print(json.dumps(store.stats(), indent=2))
    else:
        rows = store.search(**flt)
        if not rows:
            print("No matching runs.")
        for r in rows:
            print(f"#{r['id']:<4} {r['created_at']}  {r['project']:<20} {r['status']:<22} attempts={r['attempts']} "
                  f"quality={r['quality_score']} model={r['model']}")
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


def _print_report(result, orch, args) -> None:
    print(f"\n===== FINAL REPORT =====\nResult   : {'SUCCESS' if result.success else 'FAILED'} ({result.status})\n"
          f"Image    : {result.image}\nAttempts : {result.attempts} build(s), {result.corrections} correction(s)\n"
          f"Duration : {result.total_duration_seconds:.1f}s\nMessage  : {result.message}")
    for a in result.history.attempts:
        print(f"\n--- attempt {a.attempt}: {a.result_label} ---")
        if a.failed_stage:
            print(f"failed stage: {a.failed_stage}")
        if a.error:
            print("error:\n" + "\n".join("   " + ln for ln in a.error.splitlines()[-12:]))
        if a.diagnosis:
            print(f"LLM diagnosis: {a.diagnosis}")
        for c in a.changes[:6]:
            print(f"   change: {c}")
    if getattr(args, "export", False):
        for p in orch.export_report(result):
            print(f"Exported: {p}")


def _run_fix(project: Path, args: argparse.Namespace, orch: Orchestrator) -> object:
    print("WARNING: " + BUILD_WARNING)
    if not _confirm("Build this Dockerfile and let the LLM repair it on failure?", args.yes_build):
        print("Not approved; nothing was built.")
        return None
    ok, msg = orch.builder.check_docker()
    if not ok:
        print(f"error: {msg}", file=sys.stderr)
        return None
    cancel = threading.Event()
    try:
        return orch.fix_existing(
            project, approve_build=True, cancel_event=cancel,
            on_event=lambda e: print(f"  >> [{e.kind}] {e.message}"),
            on_log=(lambda ln: print("     | " + ln)) if args.verbose else None)
    except KeyboardInterrupt:
        cancel.set()
        print("\nCancelled.")
        return None


def _apply_overrides(args: argparse.Namespace):
    import os
    if getattr(args, "model", None):
        os.environ["OLLAMA_MODEL"] = args.model
    if getattr(args, "max_retries", None) is not None:
        os.environ["MAX_RETRIES"] = str(args.max_retries)
    return reload_settings()


def cmd_fix(args: argparse.Namespace) -> int:
    orch = Orchestrator(_apply_overrides(args))
    project = Path(args.project).expanduser()
    try:
        before = orch.load_existing_artifacts(project).dockerfile
        result = _run_fix(project, args, orch)
    except (FileNotFoundError, NotADirectoryError, OllamaError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ENV
    if result is None:
        return EXIT_DECLINED
    _print_report(result, orch, args)
    if result.success and result.corrections:
        print("\nOriginal Dockerfile kept as Dockerfile.orig; corrected Dockerfile is Dockerfile.")
    return EXIT_OK if result.success else EXIT_BUILD_FAILED


def cmd_selftest(args: argparse.Namespace) -> int:
    import difflib
    import shutil
    from app.agent.selftest import FAULTS, create_faulty_project
    if args.list:
        for f in FAULTS.values():
            print(f"{f.name:<12} [{f.difficulty}] {f.description}\n{'':<12} expected fix: {f.expected_fix}")
        return EXIT_OK
    orch = Orchestrator(_apply_overrides(args))
    fault = FAULTS[args.fault]
    project = create_faulty_project(args.fault)
    print(f"Disposable test project: {project}\nInjected fault ({fault.difficulty}): {fault.description}\n"
          f"Expected fix: {fault.expected_fix}\n\n----- faulty Dockerfile -----\n"
          + (project / "Dockerfile").read_text())
    faulty = (project / "Dockerfile").read_text()
    try:
        result = _run_fix(project, args, orch)
    except (FileNotFoundError, OllamaError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        result = None
    code = EXIT_DECLINED if result is None else EXIT_OK if result.success else EXIT_BUILD_FAILED
    if result is not None:
        _print_report(result, orch, args)
        fixed = (project / "Dockerfile").read_text()
        print("\n----- diff: faulty -> final Dockerfile -----")
        print("\n".join(difflib.unified_diff(faulty.splitlines(), fixed.splitlines(), "faulty", "final", lineterm="")) or "(no change)")
        verdict = ("PASS: the first build failed and the LLM-corrected Dockerfile built successfully."
                   if result.success and result.corrections >= 1 else
                   "INCONCLUSIVE: the faulty Dockerfile built without a failure." if result.success else
                   "FAIL: the loop did not reach a successful build.")
        print("\nSELF-CORRECTION TEST: " + verdict)
        if result.success:
            print(f"Remove the test image with:  docker rmi {result.image}")
    if args.keep:
        print(f"Kept test project at {project}")
    else:
        shutil.rmtree(project.parent, ignore_errors=True)
    return code


def cmd_compose(args: argparse.Namespace) -> int:
    from app.compose.generator import (ComposeProtectionError, generate_compose, generate_unit_dockerfiles,
                                       validate_compose, write_compose)
    settings = reload_settings()
    orch = Orchestrator(settings)
    try:
        analysis = orch.analyze(args.project)
        if args.generate_dockerfiles:
            if not _confirm("Ask the local LLM to generate a Dockerfile for each component that lacks one?",
                            args.yes_generate):
                print("Skipped Dockerfile generation.")
            else:
                for unit, status in generate_unit_dockerfiles(orch, analysis, args.project).items():
                    print(f"  {unit or '.'}: {status}")
        result = generate_compose(analysis, args.project)
    except (FileNotFoundError, NotADirectoryError, ValueError, OllamaError, GenerationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ENV
    check = validate_compose(result.compose, result.env_example)
    for w in result.warnings + check.warnings:
        print(f"warning: {w}", file=sys.stderr)
    for n in result.notes:
        print(f"note: {n}", file=sys.stderr)
    if not check.ok:
        for e in check.errors:
            print(f"validation error: {e}", file=sys.stderr)
        return EXIT_BUILD_FAILED
    print(f"validation: ok ({'docker compose config checked' if check.docker_compose_checked else 'structure only'})",
          file=sys.stderr)
    if not args.write:
        print(result.compose)
        print("# --- .env.example ---\n" + result.env_example)
        return EXIT_OK
    try:
        for path in write_compose(args.project, result, overwrite=args.yes_overwrite):
            print(f"wrote {path}")
    except ComposeProtectionError as exc:
        print(f"error: {exc}. Re-run with --yes-overwrite to approve (existing files are backed up as .bak).",
              file=sys.stderr)
        return EXIT_DECLINED
    print("Next: copy .env.example to .env, fill in the values, then run `docker compose up --build` yourself.")
    return EXIT_OK


def cmd_k8s(args: argparse.Namespace) -> int:
    from app.kubernetes.generator import (K8sProtectionError, apply_instructions, generate_manifests,
                                          load_unit_dockerfiles, validate_manifests, write_manifests)
    try:
        analysis = Orchestrator(reload_settings()).analyze(args.project)
        dockerfiles = load_unit_dockerfiles(analysis, args.project)
        result = generate_manifests(analysis, image=args.image, dockerfile=dockerfiles.get(".", ""),
                                    units_dockerfiles=dockerfiles, namespace=args.namespace,
                                    ingress_host=args.ingress_host, health_path=args.health_path,
                                    replicas=args.replicas)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ENV
    check = validate_manifests(result.docs, yaml_text=result.combined)
    for w in result.warnings + check.warnings:
        print(f"warning: {w}", file=sys.stderr)
    for n in result.notes:
        print(f"note: {n}", file=sys.stderr)
    if not check.ok:
        for e in check.errors:
            print(f"validation error: {e}", file=sys.stderr)
        return EXIT_BUILD_FAILED
    print(f"validation: ok ({'kubectl dry-run checked' if check.kubectl_checked else 'built-in checks only'})",
          file=sys.stderr)
    if not args.write:
        print(result.combined)
    else:
        try:
            for path in write_manifests(args.project, result, overwrite=args.yes_overwrite):
                print(f"wrote {path}")
        except K8sProtectionError as exc:
            print(f"error: {exc}. Re-run with --yes-overwrite to approve (existing files are backed up as .bak).",
                  file=sys.stderr)
            return EXIT_DECLINED
    print("\nNothing was deployed. To deploy, review the files, then run yourself:", file=sys.stderr)
    for step in apply_instructions(result, namespace=args.namespace):
        print("  " + step, file=sys.stderr)
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dockerfile-ai", description="Generate and self-correct Dockerfiles with a "
                                                                   "local Ollama model.")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="check Ollama, the model and Docker").set_defaults(func=cmd_doctor)
    m = sub.add_parser("models", help="recommend models for this machine and assess the installed ones")
    m.add_argument("--benchmark", metavar="PROJECT", help="also compare models by generating for this project")
    m.add_argument("--models", help="comma-separated model names to benchmark (default: top 2 installed)")
    m.set_defaults(func=cmd_models)
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
    f = sub.add_parser("fix", help="build the Dockerfile already in the project; let the LLM repair build failures")
    f.add_argument("project")
    f.add_argument("--model")
    f.add_argument("--max-retries", type=int)
    f.add_argument("--yes-build", action="store_true", help="approve starting the Docker build")
    f.add_argument("--verbose", "-v", action="store_true")
    f.add_argument("--export", action="store_true")
    f.set_defaults(func=cmd_fix)
    t = sub.add_parser("selftest", help="test self-correction on a disposable project with an injected fault")
    t.add_argument("--fault", default="copy-typo", choices=["copy-typo", "bad-tag", "missing-dep"])
    t.add_argument("--list", action="store_true", help="describe the available faults")
    t.add_argument("--model")
    t.add_argument("--max-retries", type=int)
    t.add_argument("--yes-build", action="store_true")
    t.add_argument("--verbose", "-v", action="store_true")
    t.add_argument("--keep", action="store_true", help="keep the temp project for inspection")
    t.add_argument("--export", action="store_true")
    t.set_defaults(func=cmd_selftest)
    q = sub.add_parser("score", help="quality + static security score of the project's Dockerfile (no LLM, no Docker)")
    q.add_argument("project")
    q.add_argument("--dockerfile", help="score this file instead of <project>/Dockerfile")
    q.add_argument("--json", action="store_true")
    q.add_argument("--min-score", type=int, help="exit 1 when the quality score is below this (for CI)")
    q.set_defaults(func=cmd_score)
    sc = sub.add_parser("scan", help="vulnerability-scan a built image with Trivy / Docker Scout (if installed)")
    sc.add_argument("image")
    sc.add_argument("--top", type=int, default=10)
    sc.add_argument("--fail-on", choices=["high", "critical"], help="exit 1 at/above this severity (for CI)")
    sc.set_defaults(func=cmd_scan)
    o = sub.add_parser("optimize", help="image-size/cache suggestions; optionally write Dockerfile.optimized")
    o.add_argument("project")
    o.add_argument("--image", help="already-built image to measure / compare against")
    o.add_argument("--write", action="store_true", help="write Dockerfile.optimized (original untouched)")
    o.add_argument("--build", action="store_true", help="also build it and compare measured sizes (needs approval)")
    o.add_argument("--yes-build", action="store_true")
    o.set_defaults(func=cmd_optimize)
    h = sub.add_parser("history", help="browse/search/export the local execution history")
    h.add_argument("--search")
    h.add_argument("--project")
    h.add_argument("--failed", action="store_true")
    h.add_argument("--limit", type=int, default=20)
    h.add_argument("--stats", action="store_true")
    h.add_argument("--export", choices=["json", "csv"])
    h.add_argument("--db", help="history database path (default: HISTORY_DB or generated/history.db)")
    h.set_defaults(func=cmd_history)
    c = sub.add_parser("compose", help="generate docker-compose.yml and .env.example (never starts the stack)")
    c.add_argument("project")
    c.add_argument("--write", action="store_true", help="write the files into the project (default: print)")
    c.add_argument("--yes-overwrite", action="store_true", help="approve overwriting existing compose/.env.example")
    c.add_argument("--generate-dockerfiles", action="store_true", help="also create missing per-component Dockerfiles with the LLM")
    c.add_argument("--yes-generate", action="store_true", help="approve LLM Dockerfile generation")
    c.set_defaults(func=cmd_compose)
    k = sub.add_parser("k8s", help="generate Kubernetes manifests (never deploys; prints kubectl instructions)")
    k.add_argument("project")
    k.add_argument("--image", help="registry image reference, e.g. registry.example.com/app:1.0.0")
    k.add_argument("--namespace")
    k.add_argument("--ingress-host", help="generate an Ingress for this host name")
    k.add_argument("--health-path", help="HTTP health endpoint (default: TCP probes)")
    k.add_argument("--replicas", type=int, default=2)
    k.add_argument("--write", action="store_true", help="write manifests to <project>/k8s/ (default: print)")
    k.add_argument("--yes-overwrite", action="store_true")
    k.set_defaults(func=cmd_k8s)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return EXIT_DECLINED


if __name__ == "__main__":
    raise SystemExit(main())

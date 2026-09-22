"""Command-line interface for SSMForge.

Subcommands:
  arch MODEL      Inspect a HuggingFace model and report architectural quirks.
  doctor          Print ssmforge + environment info (cache, tokens, versions).

The `arch` subcommand supports:
  - --format {json,markdown,md}: report output format (default: json)
  - --output PATH / -o PATH: write to file; '-' means stdout
  - --quiet: suppress the human-readable summary on stderr
  - --diff OTHER: compare against another model and print a diff
  - --dry-run: fetch only config, estimate memory, skip weight download
"""

from __future__ import annotations

import argparse
import contextlib
import os
import signal
import sys
from pathlib import Path


@contextlib.contextmanager
def _arch_load_progress(model_id: str, dry_run: bool = False):
    """Load a HuggingFace model for `ssmforge arch`, with progress on stderr.

    If `dry_run=True`, fetches only the config (no weight download).
    """
    if dry_run:
        print(f"Loading config for {model_id}... (dry-run, no weights)", file=sys.stderr)
    else:
        print(f"Loading {model_id}... (downloading if not cached)", file=sys.stderr)

    try:
        if dry_run:
            # Config only — no weight download. AutoConfig fetches config.json.
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_id)
            yield config
        else:
            from transformers import AutoModelForCausalLM
            # `dtype=` is the new transformers ≥ 4.50 kwarg; `torch_dtype` is deprecated
            # and emits a warning on every load. Try `dtype` first, fall back if older.
            try:
                model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")
            except TypeError:
                # transformers < 4.50
                model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
            yield model
    finally:
        pass


def _is_local_path(source: str) -> bool:
    """Return True if `source` looks like a local filesystem path.

    Heuristic: starts with `.`, `/`, `~`, or contains a path separator on Windows.
    HF model ids don't contain `/` at the start of a path component and have
    the form `org/name` (org has no leading dot).
    """
    if source.startswith(("./", "../", "/", "~")):
        return True
    # Windows absolute path like "C:\..." or "C:/..."
    if len(source) >= 3 and source[1] == ":" and source[2] in ("/", "\\"):
        return True
    return False


def _format_diff_markdown(report_a: dict, report_b: dict, diff: dict) -> str:
    """Render a diff between two reports as Markdown."""
    lines = []
    lines.append(f"# Architectural diff")
    lines.append("")
    lines.append(f"- **Model A:** `{report_a.get('model_id', '?')}`")
    lines.append(f"- **Model B:** `{report_b.get('model_id', '?')}`")
    lines.append("")
    if diff["identical"]:
        lines.append("**Identical** — architectures match exactly.")
        return "\n".join(lines)

    lines.append(f"**{len(diff['differences'])} differences found.**")
    lines.append("")
    if diff["added_quirks"]:
        lines.append(f"### Added quirks (in B but not A)")
        for q in diff["added_quirks"]:
            lines.append(f"- `{q}`")
        lines.append("")
    if diff["removed_quirks"]:
        lines.append(f"### Removed quirks (in A but not B)")
        for q in diff["removed_quirks"]:
            lines.append(f"- `{q}`")
        lines.append("")
    lines.append("### All differences")
    lines.append("")
    lines.append("| Field | A | B |")
    lines.append("|-------|---|---|")
    for d in diff["differences"]:
        lines.append(f"| `{d['field']}` | `{d['a']}` | `{d['b']}` |")
    return "\n".join(lines)


def _format_dry_run_summary(report: dict) -> str:
    """Format a dry-run report as a human-readable summary."""
    cfg = report.get("config", {})
    mem = report.get("memory_estimate", {})
    profile = report.get("profile", {})

    lines = []
    lines.append(f"SSMForge arch DRY-RUN: {report['model_id']}")
    lines.append(f"  family:              {profile.get('family', '?')}")
    lines.append(f"  attention_type:      {profile.get('attention_type', '?')}")
    lines.append(f"  mlp_type:            {profile.get('mlp_type', '?')}")
    lines.append(f"  norm_type:           {profile.get('norm_type', '?')}")
    lines.append("")
    lines.append("Model config:")
    lines.append(f"  model_type:          {report.get('model_type', '?')}")
    lines.append(f"  hidden_size:         {cfg.get('hidden_size')}")
    lines.append(f"  num_hidden_layers:   {cfg.get('num_hidden_layers')}")
    lines.append(f"  num_attention_heads: {cfg.get('num_attention_heads')}")
    lines.append(f"  num_kv_heads:        {cfg.get('num_key_value_heads')}")
    lines.append(f"  intermediate_size:   {cfg.get('intermediate_size')}")
    lines.append(f"  vocab_size:          {cfg.get('vocab_size')}")
    lines.append(f"  max_position:        {cfg.get('max_position_embeddings')}")
    lines.append(f"  torch_dtype:         {cfg.get('torch_dtype')}")
    lines.append(f"  tie_word_embeddings: {cfg.get('tie_word_embeddings')}")
    lines.append(f"  attention_bias:      {cfg.get('attention_bias')}")
    lines.append("")
    if mem.get("estimated_params"):
        n_params = mem["estimated_params"]
        n_bytes = mem["estimated_bytes"]
        # Human-format
        if n_params >= 1e9:
            human = f"{n_params / 1e9:.2f}B"
        elif n_params >= 1e6:
            human = f"{n_params / 1e6:.2f}M"
        elif n_params >= 1e3:
            human = f"{n_params / 1e3:.2f}K"
        else:
            human = str(n_params)
        if n_bytes >= 1024 ** 3:
            sz = f"{n_bytes / (1024 ** 3):.2f} GB"
        elif n_bytes >= 1024 ** 2:
            sz = f"{n_bytes / (1024 ** 2):.2f} MB"
        elif n_bytes >= 1024:
            sz = f"{n_bytes / 1024:.2f} KB"
        else:
            sz = f"{n_bytes} B"
        lines.append(f"Memory estimate: ~{sz} in {mem.get('dtype', 'unknown')} ({human} params)")
        lines.append("")

    # Quirks we could detect from config
    lines.append("Detected quirks (config-only):")
    q = report.get("quirks", {})
    if q.get("grouped_attention"):
        lines.append("  ✓ GQA")
    if q.get("mqa"):
        lines.append("  ✓ MQA (kv_heads=1)")
    if q.get("moe"):
        lines.append(f"  ✓ MoE (num_experts={q.get('num_experts')}, top_k={q.get('moe_top_k')})")
    if q.get("sliding_window"):
        lines.append(f"  ✓ sliding_window={q['sliding_window']}")
    if q.get("soft_capping"):
        caps = q["soft_capping"]
        cap_str = ", ".join(f"{k}={v}" for k, v in caps.items())
        lines.append(f"  ✓ soft-capping: {cap_str}")
    if q.get("partial_rope_factor"):
        lines.append(f"  ✓ partial RoPE factor: {q['partial_rope_factor']}")
    if q.get("rope_theta") is not None:
        lines.append(f"  ✓ rope_theta={q['rope_theta']} (from {cfg.get('rope_theta_source', '?')})")

    # Quirks NOT detectable without state_dict
    lines.append("")
    lines.append("Quirks requiring --no-dry-run (state_dict inspection):")
    lines.append("  · attention_bias (config truth vs state_dict truth)")
    lines.append("  · tied_embeddings (lm_head tensor identity)")
    lines.append("  · fused_qkv (qkv_proj vs q/k/v split)")
    lines.append("  · fused_gate_up (gate_up_proj vs gate/up split)")
    lines.append("  · mlp_type / norm_type (state_dict structure)")
    lines.append("  · layer_scale (ls1/ls2 keys)")
    lines.append("")

    compat = report.get("compatibility", {})
    if compat.get("is_compatible"):
        lines.append("Compatibility: OK (no blockers)")
    else:
        lines.append(f"Compatibility: BLOCKED by: {', '.join(compat.get('blockers', []))}")

    return "\n".join(lines)


def _write_or_print(text: str, output_path: str | None) -> None:
    """Write text to a file if path given, else print to stdout.

    Special-case: if output_path is '-', print to stdout (Unix convention).
    """
    if output_path and output_path != "-":
        try:
            Path(output_path).write_text(text, encoding="utf-8")
        except (PermissionError, FileNotFoundError, IsADirectoryError, OSError) as e:
            print(f"Error: cannot write to {output_path!r}: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"Report written to {output_path}", file=sys.stderr)
    else:
        print(text)


def _check_local_path(source: str) -> None:
    """If `source` looks like a local path, fail fast if it doesn't exist.

    HF model ids pass through this unchanged.
    """
    if not _is_local_path(source):
        return
    expanded = os.path.expanduser(source)
    if not os.path.exists(expanded):
        print(f"Error: local path does not exist: {source}", file=sys.stderr)
        print(f"  expanded: {expanded}", file=sys.stderr)
        print(f"  Hint: if you meant a HuggingFace model id, omit any leading '/' or './'", file=sys.stderr)
        sys.exit(1)


def _install_signal_handlers() -> None:
    """Make Ctrl+C print a clean message instead of a traceback."""
    def _sigint_handler(sig, frame):
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)  # 130 = SIGINT convention
    try:
        signal.signal(signal.SIGINT, _sigint_handler)
    except (ValueError, OSError):
        # Not in main thread, or platform doesn't support it. Best-effort.
        pass


def _quiet_transformers_warnings() -> None:
    """Reduce noise from transformers' internal logging.

    The `torch_dtype` deprecation warning fires on every from_pretrained
    call. For a tool that's already migrated to `dtype=`, the warning
    is just clutter. We move the transformers logger to ERROR-level
    when --quiet is on, and leave WARNING-level visible otherwise so
    users still see real problems.

    Also silences the 'pad_token_id' / 'generation_config' warnings that
    fire on some small test models.
    """
    try:
        import logging
        import transformers
        # Don't downgrade below WARNING — we want to see actual errors.
        # But the deprecation flood is at INFO/DEBUG.
        for name in ("transformers", "transformers.modeling_utils",
                     "transformers.configuration_utils", "transformers.tokenization_utils_base"):
            logging.getLogger(name).setLevel(logging.ERROR)
    except ImportError:
        pass


def main(argv: list[str] | None = None) -> None:
    """Entry point for the `ssmforge` command."""
    _install_signal_handlers()

    # Show version on --version
    if argv and argv[0] in ("--version", "-V", "version") and (len(argv) == 1 or argv[1].startswith("-")):
        try:
            from ssmforge import __version__
            print(f"ssmforge {__version__}")
            sys.exit(0)
        except ImportError:
            print("ssmforge (unknown version)")
            sys.exit(0)

    parser = argparse.ArgumentParser(
        prog="ssmforge",
        description=(
            "Architecture analyzer for HuggingFace models.\n\n"
            "Subcommands:\n"
            "  arch MODEL      Inspect a HuggingFace model — reports quirks like\n"
            "                  attention biases, fused QKV, MoE, sliding window,\n"
            "                  LayerScale, soft-capping, partial RoPE, MLP type,\n"
            "                  norm type. Outputs JSON or Markdown.\n"
            "  doctor          Print ssmforge + environment info (cache dir, transformers version, etc.)\n\n"
            "Try: ssmforge arch hf-internal-testing/tiny-random-LlamaForCausalLM --dry-run"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", "-V", action="version",
        version=f"%(prog)s {_get_version()}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- arch subcommand ----
    arch_p = subparsers.add_parser(
        "arch",
        help=(
            "Inspect a HuggingFace model's architecture and report quirks. "
            "Detects attention biases, fused tensors, MoE, sliding window, "
            "LayerScale, soft-capping, partial RoPE, MLP type, and norm type."
        ),
    )
    arch_p.add_argument(
        "source", nargs="?", default=None,
        help="HF model id or local path. Optional when --compare is used (the "
             "first model can come from --compare instead).",
    )
    arch_p.add_argument(
        "--output", "-o", default=None,
        help="Write report to this file (use '-' for stdout). Format chosen by --format.",
    )
    arch_p.add_argument(
        "--format", "-f", default="json", choices=["json", "markdown", "md"],
        help="Output format (default: json). markdown/md is GitHub-friendly.",
    )
    arch_p.add_argument(
        "--quiet", action="store_true",
        help="Suppress the human-readable summary; print the report only",
    )
    arch_p.add_argument(
        "--diff", metavar="OTHER_MODEL", default=None,
        help="Compare against another model. Loads both and prints a diff report.",
    )
    arch_p.add_argument(
        "--dry-run", action="store_true",
        help="Fetch only the config (no weight download). Reports config-only quirks "
             "and memory estimate. Useful as a pre-flight check.",
    )
    arch_p.add_argument(
        "--compare", metavar="MODEL", nargs="+", default=None,
        help="Compare 2+ models side-by-side. The first model comes from the "
             "`source` positional (or --compare if source is omitted). Renders "
             "a multi-model comparison table.",
    )

    # ---- doctor subcommand ----
    doctor_p = subparsers.add_parser(
        "doctor",
        help="Print ssmforge + environment info (cache dir, transformers version, etc.)",
    )
    doctor_p.add_argument(
        "--format", "-f", default="text", choices=["text", "json"],
        help="Output format (default: text)",
    )

    args = parser.parse_args(argv)

    if args.command == "arch":
        # Suppress transformers' torch_dtype deprecation noise when user wants clean output
        if getattr(args, "quiet", False):
            _quiet_transformers_warnings()
        return _cmd_arch(args)
    elif args.command == "doctor":
        return _cmd_doctor(args)
    else:
        parser.print_help()
        sys.exit(1)


def _cmd_arch(args) -> None:
    """Handle the `arch` subcommand."""
    from ssmforge.analyze import (
        build_report,
        build_report_from_config,
        compare_reports,
        diff_reports,
        format_compare_markdown,
        format_report_json,
        format_report_markdown,
    )
    from ssmforge.analyze.summary import render_summary

    # Resolve the list of models to compare / inspect.
    # If --compare is given, models = [source?] + args.compare (deduped, preserving order)
    # If only --diff is given, models = [source, args.diff]
    # Otherwise models = [source]
    if args.compare:
        # Build the full model list: source first (if given), then --compare args
        all_models = []
        if args.source:
            all_models.append(args.source)
        for m in args.compare:
            if m not in all_models:
                all_models.append(m)
        # Need at least 2 for compare
        if len(all_models) < 2:
            print(
                "Error: --compare requires at least 2 models. "
                "Pass model ids as arguments after --compare, or "
                "provide the first as a positional argument.",
                file=sys.stderr,
            )
            sys.exit(1)
        args._compare_models = all_models
    elif args.diff:
        args._compare_models = [args.source, args.diff]
    else:
        args._compare_models = [args.source] if args.source else []

    # Fail fast on missing local paths
    for m in args._compare_models:
        _check_local_path(m)

    # ---- Multi-model dispatch (compare, diff, single) ----
    models_to_load = args._compare_models
    n_models = len(models_to_load)

    # Load all reports
    reports = []
    for model_id in models_to_load:
        try:
            with _arch_load_progress(model_id, dry_run=args.dry_run) as obj:
                if args.dry_run:
                    r = build_report_from_config(model_id=model_id, config=obj)
                else:
                    r = build_report(
                        model_id=model_id,
                        config=obj.config,
                        state_dict=dict(obj.state_dict()),
                    )
            reports.append(r)
        except KeyboardInterrupt:
            print("\nInterrupted.", file=sys.stderr)
            sys.exit(130)
        except MemoryError:
            print(f"Error: out of memory while loading {model_id}", file=sys.stderr)
            if not args.dry_run:
                print("  Try --dry-run to preview without loading weights.", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            err_name = type(e).__name__
            msg = str(e)
            if not args.dry_run and ("memory allocation" in msg.lower() or "out of memory" in msg.lower()):
                print(f"Error: insufficient memory to load {model_id}", file=sys.stderr)
                print(f"  Detail: {msg}", file=sys.stderr)
                print(f"  Try --dry-run to preview without loading weights.", file=sys.stderr)
                sys.exit(1)
            print(f"Error analyzing {model_id}: {err_name}: {msg}", file=sys.stderr)
            sys.exit(1)

    # Single model
    if n_models == 1:
        report_a = reports[0]
        if args.format in ("markdown", "md"):
            output_text = format_report_markdown(report_a)
        else:
            output_text = format_report_json(report_a)
            if not args.quiet and not args.output:
                if args.dry_run:
                    print(_format_dry_run_summary(report_a), file=sys.stderr)
                else:
                    print(render_summary(report_a), file=sys.stderr)

        _write_or_print(output_text, args.output)
        sys.exit(0 if report_a["compatibility"]["is_compatible"] else 2)

    # Two models (legacy --diff behavior, preserved for back-compat)
    if n_models == 2 and not args.compare:
        report_a, report_b = reports
        diff = diff_reports(report_a, report_b)
        if args.format in ("markdown", "md"):
            md = _format_diff_markdown(report_a, report_b, diff)
            _write_or_print(md, args.output)
        else:
            import json as json_mod
            output = {
                "model_a": args._compare_models[0],
                "model_b": args._compare_models[1],
                "identical": diff["identical"],
                "differences": diff["differences"],
                "added_quirks": diff["added_quirks"],
                "removed_quirks": diff["removed_quirks"],
            }
            _write_or_print(json_mod.dumps(output, indent=2), args.output)
        sys.exit(0)

    # Multi-model comparison (3+ models, or --compare with 2 models)
    comparison = compare_reports(reports)
    if args.format in ("markdown", "md"):
        md = format_compare_markdown(comparison)
        _write_or_print(md, args.output)
    else:
        import json as json_mod
        # Add a small header so JSON consumers know it's a comparison
        output = {
            "comparison_type": "multi_model",
            "model_count": len(comparison["models"]),
            "all_identical": comparison["all_identical"],
            "identical_field_count": comparison["identical_field_count"],
            "different_field_count": comparison["different_field_count"],
            "models": comparison["models"],
            "fields": comparison["fields"],
        }
        _write_or_print(json_mod.dumps(output, indent=2), args.output)
    sys.exit(0)


def _cmd_doctor(args) -> None:
    """Handle the `doctor` subcommand."""
    import os as _os
    try:
        from ssmforge import __version__ as ssmforge_version
    except ImportError:
        ssmforge_version = "(unknown)"

    info = {
        "ssmforge_version": ssmforge_version,
        "python_version": sys.version.split()[0],
        "platform": sys.platform,
        "hf_home": _os.environ.get("HF_HOME", "(not set — using ~/.cache/huggingface)"),
        "hf_hub_cache": _os.environ.get("HF_HUB_CACHE", "(not set — using HF_HOME/hub)"),
        "hf_token_set": bool(_os.environ.get("HF_TOKEN")),
    }
    try:
        import transformers
        info["transformers_version"] = transformers.__version__
    except ImportError:
        info["transformers_version"] = "(not installed)"
    try:
        import huggingface_hub
        info["huggingface_hub_version"] = huggingface_hub.__version__
    except ImportError:
        info["huggingface_hub_version"] = "(not installed)"

    if args.format == "json":
        import json as json_mod
        print(json_mod.dumps(info, indent=2))
    else:
        print("ssmforge doctor")
        print("-" * 40)
        for k, v in info.items():
            print(f"  {k}: {v}")
    sys.exit(0)


def _get_version() -> str:
    try:
        from ssmforge import __version__
        return __version__
    except ImportError:
        return "(unknown)"


if __name__ == "__main__":
    main()

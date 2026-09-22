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
    arch_p.add_argument("source", help="HF model id or local path")
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
        diff_reports,
        format_report_json,
        format_report_markdown,
    )
    from ssmforge.analyze.summary import render_summary

    # Fail fast on missing local paths
    _check_local_path(args.source)
    if args.diff:
        _check_local_path(args.diff)

    # Dry-run: config only
    if args.dry_run:
        try:
            with _arch_load_progress(args.source, dry_run=True) as config:
                report_a = build_report_from_config(
                    model_id=args.source,
                    config=config,
                )
        except KeyboardInterrupt:
            print("\nInterrupted.", file=sys.stderr)
            sys.exit(130)
        except MemoryError:
            print(f"Error: out of memory while loading config for {args.source}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error analyzing {args.source}: {e}", file=sys.stderr)
            sys.exit(1)

        # Diff in dry-run mode
        if args.diff:
            try:
                with _arch_load_progress(args.diff, dry_run=True) as config_b:
                    report_b = build_report_from_config(
                        model_id=args.diff,
                        config=config_b,
                    )
            except KeyboardInterrupt:
                print("\nInterrupted.", file=sys.stderr)
                sys.exit(130)
            except MemoryError:
                print(f"Error: out of memory while loading config for {args.diff}", file=sys.stderr)
                sys.exit(1)
            except Exception as e:
                print(f"Error analyzing {args.diff}: {e}", file=sys.stderr)
                sys.exit(1)

            diff = diff_reports(report_a, report_b)
            if args.format in ("markdown", "md"):
                md = _format_diff_markdown(report_a, report_b, diff)
                _write_or_print(md, args.output)
            else:
                import json as json_mod
                output = {
                    "model_a": args.source,
                    "model_b": args.diff,
                    "identical": diff["identical"],
                    "differences": diff["differences"],
                    "added_quirks": diff["added_quirks"],
                    "removed_quirks": diff["removed_quirks"],
                }
                _write_or_print(json_mod.dumps(output, indent=2), args.output)
            sys.exit(0)

        # Single-model dry-run
        if args.format in ("markdown", "md"):
            output_text = format_report_markdown(report_a)
        else:
            output_text = format_report_json(report_a)
            if not args.quiet and not args.output:
                print(_format_dry_run_summary(report_a), file=sys.stderr)

        _write_or_print(output_text, args.output)
        sys.exit(0 if report_a["compatibility"]["is_compatible"] else 2)

    # Full mode: load weights
    try:
        with _arch_load_progress(args.source, dry_run=False) as model:
            report_a = build_report(
                model_id=args.source,
                config=model.config,
                state_dict=dict(model.state_dict()),
            )
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except MemoryError:
        print(f"Error: out of memory while loading {args.source}", file=sys.stderr)
        print("  Try --dry-run to preview the model without loading weights,", file=sys.stderr)
        print("  or set HF_HOME to a different drive.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        err_name = type(e).__name__
        # Detect the "memory allocation failed" message that Rust-backended
        # safetensors emits when the model doesn't fit in RAM.
        msg = str(e)
        if "memory allocation" in msg.lower() or "out of memory" in msg.lower():
            print(f"Error: insufficient memory to load {args.source}", file=sys.stderr)
            print(f"  Detail: {msg}", file=sys.stderr)
            print(f"  Try --dry-run to preview without loading weights.", file=sys.stderr)
            sys.exit(1)
        print(f"Error analyzing {args.source}: {err_name}: {msg}", file=sys.stderr)
        sys.exit(1)

    # Diff mode
    if args.diff:
        try:
            with _arch_load_progress(args.diff, dry_run=False) as model_b:
                report_b = build_report(
                    model_id=args.diff,
                    config=model_b.config,
                    state_dict=dict(model_b.state_dict()),
                )
        except KeyboardInterrupt:
            print("\nInterrupted.", file=sys.stderr)
            sys.exit(130)
        except MemoryError:
            print(f"Error: out of memory while loading {args.diff}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            err_name = type(e).__name__
            msg = str(e)
            if "memory allocation" in msg.lower() or "out of memory" in msg.lower():
                print(f"Error: insufficient memory to load {args.diff}", file=sys.stderr)
                sys.exit(1)
            print(f"Error analyzing {args.diff}: {err_name}: {msg}", file=sys.stderr)
            sys.exit(1)

        diff = diff_reports(report_a, report_b)
        if args.format in ("markdown", "md"):
            md = _format_diff_markdown(report_a, report_b, diff)
            _write_or_print(md, args.output)
        else:
            import json as json_mod
            output = {
                "model_a": args.source,
                "model_b": args.diff,
                "identical": diff["identical"],
                "differences": diff["differences"],
                "added_quirks": diff["added_quirks"],
                "removed_quirks": diff["removed_quirks"],
            }
            _write_or_print(json_mod.dumps(output, indent=2), args.output)
        sys.exit(0)

    # Single-model full mode
    if args.format in ("markdown", "md"):
        output_text = format_report_markdown(report_a)
    else:
        output_text = format_report_json(report_a)
        if not args.quiet and not args.output:
            print(render_summary(report_a), file=sys.stderr)

    _write_or_print(output_text, args.output)
    sys.exit(0 if report_a["compatibility"]["is_compatible"] else 2)


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

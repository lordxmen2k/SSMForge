"""Command-line interface for SSMForge.

The only subcommand is `arch`, which runs the architecture analyzer
on any HuggingFace model and reports on quirks that affect downstream
conversion or fine-tuning.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path


@contextlib.contextmanager
def _arch_load_progress(model_id: str):
    """Load a HuggingFace model for `ssmforge arch`, with a progress note on stderr."""
    print(f"Loading {model_id}... (downloading if not cached)", file=sys.stderr)
    try:
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


def main(argv: list[str] | None = None) -> None:
    """Entry point for the `ssmforge` command."""
    parser = argparse.ArgumentParser(
        prog="ssmforge",
        description=(
            "Architecture analyzer for HuggingFace models.\n\n"
            "Subcommands:\n"
            "  arch MODEL      Inspect a HuggingFace model — reports quirks like\n"
            "                  attention biases, fused QKV, MoE, sliding window,\n"
            "                  LayerScale, soft-capping, partial RoPE, MLP type,\n"
            "                  norm type. Outputs JSON or Markdown.\n"
            "  doctor          Print ssmforge + environment info (cache dir, transformers version, etc.)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

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
        help="Write report to this file. Format chosen by --format.",
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
        from ssmforge.analyze import (
            build_report,
            diff_reports,
            format_report_json,
            format_report_markdown,
        )
        from ssmforge.analyze.summary import render_summary

        try:
            with _arch_load_progress(args.source) as model:
                report_a = build_report(
                    model_id=args.source,
                    config=model.config,
                    state_dict=dict(model.state_dict()),
                )
        except Exception as e:
            print(f"Error analyzing {args.source}: {e}", file=sys.stderr)
            sys.exit(1)

        # Diff mode: load second model and compare
        if args.diff:
            try:
                with _arch_load_progress(args.diff) as model_b:
                    report_b = build_report(
                        model_id=args.diff,
                        config=model_b.config,
                        state_dict=dict(model_b.state_dict()),
                    )
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

        # Single-model mode
        if args.format in ("markdown", "md"):
            output_text = format_report_markdown(report_a)
        else:
            output_text = format_report_json(report_a)
            if not args.quiet and not args.output:
                print(render_summary(report_a), file=sys.stderr)

        _write_or_print(output_text, args.output)
        sys.exit(0 if report_a["compatibility"]["is_compatible"] else 2)

    elif args.command == "doctor":
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


def _write_or_print(text: str, output_path: str | None) -> None:
    """Write text to a file if path given, else print to stdout."""
    if output_path:
        Path(output_path).write_text(text, encoding="utf-8")
        print(f"Report written to {output_path}", file=sys.stderr)
    else:
        print(text)


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


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()

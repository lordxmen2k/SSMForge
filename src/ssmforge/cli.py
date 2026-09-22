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
            "Use `ssmforge arch MODEL` to inspect any HuggingFace model — reports\n"
            "quirks like attention biases, fused QKV, MoE, sliding window,\n"
            "LayerScale, soft-capping, partial RoPE, MLP type, and norm type.\n"
            "Outputs structured JSON + human-readable summary."
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
        help="Write JSON report to this file (UTF-8) instead of stdout",
    )
    arch_p.add_argument(
        "--quiet", action="store_true",
        help="Suppress the human-readable summary; print JSON only",
    )

    args = parser.parse_args(argv)

    if args.command == "arch":
        from ssmforge.analyze import build_report, format_report_json
        from ssmforge.analyze.summary import render_summary

        try:
            with _arch_load_progress(args.source) as model:
                report = build_report(
                    model_id=args.source,
                    config=model.config,
                    state_dict=dict(model.state_dict()),
                )
        except Exception as e:
            print(f"Error analyzing {args.source}: {e}", file=sys.stderr)
            sys.exit(1)

        json_text = format_report_json(report)

        if args.output:
            Path(args.output).write_text(json_text, encoding="utf-8")
            print(f"Report written to {args.output}", file=sys.stderr)
        else:
            if not args.quiet:
                print(render_summary(report), file=sys.stderr)
            print(json_text)
        sys.exit(0 if report["compatibility"]["is_compatible"] else 2)


if __name__ == "__main__":
    main()

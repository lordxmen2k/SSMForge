"""Command-line interface for SSMForge."""

from __future__ import annotations

import argparse
import sys

from ssmforge import convert
from ssmforge.exceptions import SSMForgeError
from ssmforge.recipes import list_recipes


def main(argv: list[str] | None = None) -> None:
    """Entry point for the `ssmforge` command."""
    parser = argparse.ArgumentParser(
        prog="ssmforge",
        description=(
            "Convert pretrained transformers to hybrid SSM/attention models.\n\n"
            "WARNING: experimental. Output quality unverified. Not for production.\n"
            "SSMForge GGUFs only load via `ssmforge run` (pure PyTorch) — they\n"
            "do NOT load in ollama / LM Studio / stock llama.cpp (yet)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_p = subparsers.add_parser("convert", help="Convert a model to hybrid SSM/attention + GGUF")
    convert_p.add_argument("source", help="HF model id or local path")
    convert_p.add_argument("--recipe", default="hybrid-25", help="Recipe name (default: hybrid-25)")
    convert_p.add_argument(
        "--quantize",
        default="Q4_K_M",
        choices=["F16", "Q8_0", "Q5_K_M", "Q4_K_M", "Q4_K_S"],
        help="GGUF quantization type (default: Q4_K_M)",
    )
    convert_p.add_argument("--output", default="./out", help="Output directory (default: ./out)")
    convert_p.add_argument("--calibration-data", default=None, help="Calibration data source")
    convert_p.add_argument("--verify", action="store_true", help="Run Stage 6 verification (slow)")
    convert_p.add_argument("--dry-run", action="store_true", help="Plan only, no export")
    convert_p.add_argument("--no-distill", action="store_true",
                           help="Skip distillation stage (default: 2-step stub; --no-distill skips it entirely)")
    convert_p.add_argument("--experimental", action="store_true", help="Allow experimental recipes")
    convert_p.add_argument("--strict-verify", action="store_true", help="Promote verify warnings to errors")
    convert_p.add_argument("--debug", action="store_true", help="Show full tracebacks on error")

    subparsers.add_parser("list-recipes", help="List registered recipes")

    run_p = subparsers.add_parser(
        "run",
        help="Run inference on an SSMForge GGUF (self-contained, no ollama needed)",
    )
    run_p.add_argument("--model", required=True, help="Path to SSMForge GGUF")
    run_p.add_argument("--prompt", default=None, help="Text prompt (omit for --interactive)")
    run_p.add_argument("--interactive", action="store_true", help="REPL mode")
    run_p.add_argument("--max-new-tokens", type=int, default=30)
    run_p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    run_p.add_argument("--no-quiet", action="store_true", help="Print model metadata")

    args = parser.parse_args(argv)

    if args.command == "list-recipes":
        print("Registered recipes:")
        for name in list_recipes():
            print(f"  - {name}")
        sys.exit(0)
    elif args.command == "run":
        from ssmforge.runtime.run import main as run_main
        # Pass through only the args after "run"
        run_argv = []
        seen = False
        for a in (argv or sys.argv[1:]):
            if seen:
                run_argv.append(a)
            elif a == "run":
                seen = True
        sys.exit(run_main(run_argv))
    elif args.command == "convert":
        print(
            "WARNING: experimental software. Output quality not verified. "
            "Not for production environments.",
            file=sys.stderr,
        )
        try:
            result = convert(
                source=args.source,
                recipe=args.recipe,
                quantize=args.quantize,
                output_dir=args.output,
                calibration_data=args.calibration_data,
                verify=args.verify,
                dry_run=args.dry_run,
                experimental=args.experimental,
                no_distill=args.no_distill,
            )
        except SSMForgeError as e:
            if args.debug:
                raise
            print(str(e), file=sys.stderr)
            sys.exit(1)
        print(f"GGUF: {result.gguf_path}")
        print(f"Manifest: {result.manifest_path}")
        print(f"Stats: {result.stats}")
        sys.exit(0)


if __name__ == "__main__":
    main()

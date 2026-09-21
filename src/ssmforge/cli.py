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
        description="Convert pretrained transformers to hybrid SSM/attention models.",
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

    args = parser.parse_args(argv)

    try:
        if args.command == "list-recipes":
            print("Registered recipes:")
            for name in list_recipes():
                print(f"  - {name}")
            sys.exit(0)
        elif args.command == "convert":
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
            print(f"GGUF: {result.gguf_path}")
            print(f"Manifest: {result.manifest_path}")
            print(f"Stats: {result.stats}")
            sys.exit(0)
    except SSMForgeError as e:
        if args.debug:
            raise
        print(str(e), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

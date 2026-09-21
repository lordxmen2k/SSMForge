"""Example: benchmark a converted model at long context."""

from pathlib import Path

from ssmforge import convert
from ssmforge.benchmark.long_context import benchmark_long_context


def main():
    """Convert a model and benchmark it at various context lengths."""
    convert(
        source="meta-llama/Llama-3.2-1B",
        recipe="hybrid-25",
        quantize="Q4_K_M",
        output_dir=Path("./out"),
    )

    # Note: full benchmark requires loading the GGUF with llama-cpp-python.
    # This example demonstrates the API surface; full integration
    # with llama.cpp's loader is in the spec roadmap.

    print("Conversion complete. See ./out/ for artifacts.")
    print("Use ssmforge.benchmark.benchmark_long_context() with a loaded model to measure.")


if __name__ == "__main__":
    main()

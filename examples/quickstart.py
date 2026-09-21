"""Quickstart example: convert Llama-3.2-1B to hybrid-25 + Q4_K_M GGUF."""

from pathlib import Path

from ssmforge import convert


if __name__ == "__main__":
    result = convert(
        source="meta-llama/Llama-3.2-1B",
        recipe="hybrid-25",
        quantize="Q4_K_M",
        output_dir=Path("./out"),
    )
    print(f"GGUF: {result.gguf_path}")
    print(f"Stats: {result.stats}")

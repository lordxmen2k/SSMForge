"""Long-context benchmark: memory + speed at increasing context lengths."""

from __future__ import annotations

import time
from typing import Any


def benchmark_long_context(
    model: Any,
    tokenizer: Any,
    context_lengths: list[int] | None = None,
    device: str = "cpu",
) -> dict[int, dict[str, float]]:
    """Measure peak memory and tokens/sec at each context length."""
    if context_lengths is None:
        context_lengths = [4096, 32768, 131072]
    results = {}
    for ctx in context_lengths:
        try:
            import torch

            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            input_ids = torch.randint(0, tokenizer.vocab_size, (1, ctx))

            start = time.time()
            with torch.no_grad():
                _ = model(input_ids=input_ids)
            elapsed = time.time() - start

            results[ctx] = {
                "elapsed_sec": elapsed,
                "tokens_per_sec": ctx / elapsed if elapsed > 0 else 0,
                "peak_memory_mb": (
                    torch.cuda.max_memory_allocated() / 1e6
                    if device == "cuda" and torch.cuda.is_available() else 0
                ),
            }
        except Exception as e:
            results[ctx] = {"error": str(e)}
    return results

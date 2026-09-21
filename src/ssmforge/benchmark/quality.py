"""Quality benchmark: perplexity vs teacher."""

from __future__ import annotations

import math

import torch


@torch.no_grad()
def compute_perplexity(model, tokenizer, texts: list[str], max_length: int = 512) -> float:
    """Compute perplexity on a list of texts."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = enc.input_ids
        if input_ids.shape[1] < 2:
            continue
        outputs = model(input_ids=input_ids, labels=input_ids)
        n_tokens = input_ids.shape[1] - 1
        total_loss += outputs.loss.item() * n_tokens
        total_tokens += n_tokens
    if total_tokens == 0:
        return float("inf")
    return math.exp(total_loss / total_tokens)

"""Data collator for KL distillation batches."""

from __future__ import annotations

from typing import Any

import torch


class KLDistillationCollator:
    """Tokenizes text inputs for both teacher and student models.

    For MVP: produces a single input_ids tensor. Teacher and student share
    the tokenizer (a hybrid SSM model and its teacher transformer both use
    the same tokenizer, by design).
    """

    def __init__(self, tokenizer: Any, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[str]) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded["labels"] = encoded["input_ids"].clone()
        return encoded

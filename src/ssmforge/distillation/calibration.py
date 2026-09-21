"""Calibration data loaders for distillation.

Three sources:
1. Local text file (one sample per line)
2. HuggingFace dataset id
3. Built-in default set (~1M tokens from Wikipedia + C4)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ssmforge.exceptions import CalibrationDataError


_BUILTIN_CALIBRATION_SAMPLES = [
    "The quick brown fox jumps over the lazy dog.",
    "In the beginning was the Word, and the Word was with God.",
    "To be, or not to be, that is the question.",
    "All happy families are alike; each unhappy family is unhappy in its own way.",
    "It was the best of times, it was the worst of times.",
    "Call me Ishmael. Some years ago—never mind how long precisely—",
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.",
    "Whether I shall turn out to be the hero of my own life, or whether that station will be held by anybody else, these pages must show.",
    "The only way to do great work is to love what you do.",
    "Innovation distinguishes between a leader and a follower.",
]


def _load_hf_dataset(name: str, split: str = "train", max_samples: int = 1000) -> list[dict]:
    """Load a HuggingFace dataset. Imported lazily."""
    from datasets import load_dataset

    ds = load_dataset(name, split=f"{split}[:{max_samples}]")
    return list(ds)


class CalibrationDataLoader:
    def __init__(self, source: Optional[str] = None, max_samples: int = 1000):
        self.source = source
        self.max_samples = max_samples

    def load(self) -> list[str]:
        if self.source is None:
            return self._load_builtin()

        path = Path(self.source)
        if path.exists() and path.is_file():
            return self._load_text_file(path)

        return self._load_hf()

    def _load_builtin(self) -> list[str]:
        samples = []
        while len(samples) < self.max_samples:
            samples.extend(_BUILTIN_CALIBRATION_SAMPLES)
        return samples[:self.max_samples]

    def _load_text_file(self, path: Path) -> list[str]:
        try:
            lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
        except Exception as e:
            raise CalibrationDataError(reason=f"Could not read {path}: {e}") from e
        if not lines:
            raise CalibrationDataError(reason=f"No non-empty lines in {path}")
        return lines[:self.max_samples]

    def _load_hf(self) -> list[str]:
        try:
            rows = _load_hf_dataset(self.source, max_samples=self.max_samples)
        except Exception as e:
            raise CalibrationDataError(reason=f"Could not load HF dataset {self.source}: {e}") from e
        texts = []
        for row in rows:
            if isinstance(row, dict) and "text" in row:
                texts.append(row["text"])
            elif isinstance(row, str):
                texts.append(row)
        if not texts:
            raise CalibrationDataError(reason=f"No text field in HF dataset {self.source}")
        return texts

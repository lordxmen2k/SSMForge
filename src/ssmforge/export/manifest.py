"""Manifest schema and writer for SSMForge conversions.

Every conversion produces a manifest JSON file alongside the GGUF, capturing
provenance: source model, recipe, layer mapping, calibration data, training
stats, output file SHA, optional quality metrics, optional long-context benchmarks.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class Manifest(BaseModel):
    ssmforge_version: str
    source_model: str
    source_revision: str
    recipe: str
    quant_type: str
    layer_mapping: list[dict]
    calibration_data_sha: Optional[str]
    training_stats: Optional[dict]
    output_gguf_path: str
    output_gguf_sha: str
    output_gguf_bytes: int
    created_at: datetime
    quality_metrics: Optional[dict] = None
    long_context_benchmark: Optional[dict] = None


def compute_file_sha(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(manifest: Manifest, path: Path) -> Path:
    """Write manifest to JSON file. Returns the path."""
    path.write_text(manifest.model_dump_json(indent=2))
    return path

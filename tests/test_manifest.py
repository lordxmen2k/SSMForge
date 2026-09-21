import hashlib
from datetime import datetime
from pathlib import Path

import pytest
from ssmforge.export import Manifest, write_manifest, compute_file_sha


def test_manifest_round_trip(tmp_path):
    m = Manifest(
        ssmforge_version="0.1.0",
        source_model="meta-llama/Llama-3.2-1B",
        source_revision="abc123",
        recipe="hybrid-25",
        quant_type="Q4_K_M",
        layer_mapping=[{"index": 0, "layer_type": "attention"}],
        calibration_data_sha=None,
        training_stats=None,
        output_gguf_path="out/model.gguf",
        output_gguf_sha="deadbeef",
        output_gguf_bytes=12345,
        created_at=datetime.now(),
        quality_metrics=None,
        long_context_benchmark=None,
    )
    out = write_manifest(m, tmp_path / "manifest.json")
    assert out.exists()
    loaded = Manifest.model_validate_json(out.read_text())
    assert loaded.source_model == m.source_model
    assert loaded.recipe == m.recipe
    assert loaded.output_gguf_bytes == 12345


def test_compute_file_sha(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("hello")
    sha = compute_file_sha(f)
    assert sha == hashlib.sha256(b"hello").hexdigest()


def test_manifest_required_fields():
    with pytest.raises(Exception):
        Manifest()


def test_compute_file_sha_large_file(tmp_path):
    f = tmp_path / "big.bin"
    f.write_bytes(b"x" * (1 << 22))  # 4MB
    sha = compute_file_sha(f)
    assert len(sha) == 64  # sha256 hex length

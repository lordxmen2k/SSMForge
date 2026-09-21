from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.integration


def test_write_f16_gguf_creates_valid_file(tmp_path):
    from ssmforge.export.gguf_writer import write_f16_gguf

    fake_sd = {
        "model.embed_tokens.weight": torch.randn(100, 64),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(64, 64),
    }
    fake_config = type("C", (), {
        "name_or_path": "test-model",
        "max_position_embeddings": 128,
        "hidden_size": 64,
        "num_hidden_layers": 1,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "rope_theta": 10000.0,
    })()

    out = tmp_path / "test.gguf"
    result = write_f16_gguf(fake_sd, fake_config, None, out)
    assert result.exists()
    assert result.stat().st_size > 0
    # Verify GGUF magic header
    with open(out, "rb") as f:
        magic = f.read(4)
    assert magic == b"GGUF", f"Expected GGUF magic, got {magic}"


def test_write_f16_gguf_round_trips_through_gguf_py(tmp_path):
    """Write a GGUF then read it back to verify integrity."""
    from gguf import GGUFReader

    from ssmforge.export.gguf_writer import write_f16_gguf

    fake_sd = {
        "model.embed_tokens.weight": torch.randn(100, 32),
    }
    fake_config = type("C", (), {
        "name_or_path": "round-trip-test",
        "max_position_embeddings": 64,
        "hidden_size": 32,
        "num_hidden_layers": 1,
        "intermediate_size": 64,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "rope_theta": 10000.0,
    })()

    out = tmp_path / "rt.gguf"
    write_f16_gguf(fake_sd, fake_config, None, out)
    reader = GGUFReader(str(out))
    # The reader should at least have parsed fields
    assert reader.fields  # non-empty fields dict

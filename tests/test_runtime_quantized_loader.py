"""Tests that the runtime loader handles quantized GGUFs (Q4_K_M).

These tests verify that:
- The dequant math works for at least one quantization type
- The full pipeline (write F16 -> quantize to Q4_K_M -> load back -> forward) works

These are NOT run unless quantized dependencies + a working llama-quantize
binary are present, since they require a full end-to-end round-trip.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from ssmforge.export.gguf_writer import write_f16_gguf
from ssmforge.runtime.loader import load_hybrid_model_from_gguf


REPO_ROOT = Path(__file__).resolve().parents[1]
LLAMA_QUANTIZE = REPO_ROOT / "vendor" / "llama.cpp" / "build" / "bin" / "llama-quantize"


def _build_minimal_state_dict(hidden=256, n_layers=4, inter=1024, vocab=100):
    sd = {
        "model.embed_tokens.weight": torch.randn(vocab, hidden),
        "model.norm.weight": torch.ones(hidden),
        "lm_head.weight": torch.randn(vocab, hidden),
    }
    for i in range(n_layers):
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            sd[f"model.layers.{i}.self_attn.{proj}.weight"] = torch.randn(hidden, hidden)
        sd[f"model.layers.{i}.mlp.gate_proj.weight"] = torch.randn(inter, hidden)
        sd[f"model.layers.{i}.mlp.up_proj.weight"] = torch.randn(inter, hidden)
        sd[f"model.layers.{i}.mlp.down_proj.weight"] = torch.randn(hidden, inter)
        sd[f"model.layers.{i}.input_layernorm.weight"] = torch.ones(hidden)
        sd[f"model.layers.{i}.post_attention_layernorm.weight"] = torch.ones(hidden)
    return sd


def _build_minimal_config(hidden=256, n_layers=4, inter=1024, vocab=100, n_heads=4):
    from types import SimpleNamespace
    return SimpleNamespace(
        name_or_path="test",
        max_position_embeddings=2048,
        hidden_size=hidden,
        num_hidden_layers=n_layers,
        intermediate_size=inter,
        num_attention_heads=n_heads,
        num_key_value_heads=n_heads,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        vocab_size=vocab,
    )


@pytest.mark.skipif(
    not LLAMA_QUANTIZE.exists(),
    reason=f"llama-quantize not built at {LLAMA_QUANTIZE}",
)
def test_loader_loads_q4km_gguf(tmp_path):
    """F16 → quantize → Q4_K_M → load → forward works end-to-end."""
    sd = _build_minimal_state_dict()
    cfg = _build_minimal_config()
    f16_path = tmp_path / "test.f16.gguf"
    write_f16_gguf(sd, cfg, tokenizer=None, output_path=f16_path)
    assert f16_path.exists()

    # Quantize via vendored llama-quantize
    q4_path = tmp_path / "test.q4km.gguf"
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = (
        str(LLAMA_QUANTIZE.parent) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    )
    proc = subprocess.run(
        [str(LLAMA_QUANTIZE), str(f16_path), str(q4_path), "Q4_K_M"],
        capture_output=True, text=True, env=env, timeout=60,
    )
    assert proc.returncode == 0, f"quantize failed: {proc.stderr}"
    assert q4_path.exists()

    # Load the quantized GGUF
    model, metadata = load_hybrid_model_from_gguf(q4_path)
    assert metadata["general.architecture"] == "ssmforge"

    # Forward pass
    input_ids = torch.randint(0, 100, (1, 8))
    out = model(input_ids=input_ids)
    assert out.logits.shape == (1, 8, 100)
    assert torch.isfinite(out.logits).all()


@pytest.mark.skipif(
    not LLAMA_QUANTIZE.exists(),
    reason=f"llama-quantize not built at {LLAMA_QUANTIZE}",
)
def test_q4km_gguf_is_smaller_than_f16(tmp_path):
    """Sanity check: Q4_K_M GGUF should be substantially smaller than F16."""
    sd = _build_minimal_state_dict()
    cfg = _build_minimal_config()
    f16_path = tmp_path / "test.f16.gguf"
    write_f16_gguf(sd, cfg, tokenizer=None, output_path=f16_path)
    q4_path = tmp_path / "test.q4km.gguf"
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = (
        str(LLAMA_QUANTIZE.parent) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    )
    proc = subprocess.run(
        [str(LLAMA_QUANTIZE), str(f16_path), str(q4_path), "Q4_K_M"],
        capture_output=True, text=True, env=env, timeout=60,
    )
    assert proc.returncode == 0, f"quantize failed: {proc.stderr}"

    f16_size = f16_path.stat().st_size
    q4_size = q4_path.stat().st_size
    # Q4_K_M is roughly 4-bit-per-weight vs F16 16-bit → ~4x smaller
    assert q4_size < f16_size / 2, \
        f"Q4_K_M ({q4_size}) should be much smaller than F16 ({f16_size})"

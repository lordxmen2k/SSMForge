"""Tests that ssmforge-produced F16 GGUFs can be quantized by our vendored
llama-quantize fork (the real round-trip validation).
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from ssmforge.export.gguf_writer import write_f16_gguf


REPO_ROOT = Path(__file__).resolve().parents[1]
LLAMA_QUANTIZE = REPO_ROOT / "vendor" / "llama.cpp" / "build" / "bin" / "llama-quantize"


pytestmark = pytest.mark.skipif(
    not LLAMA_QUANTIZE.exists(),
    reason=f"llama-quantize not built at {LLAMA_QUANTIZE}. Run: bash scripts/build_vendor.sh",
)


def _fake_hybrid_state_dict(hidden: int = 256, n_layers: int = 4, inter: int = 1024, vocab: int = 100):
    """Build a hybrid state dict (all attention blocks for the test)."""
    sd = {
        "model.embed_tokens.weight": torch.randn(vocab, hidden),
        "model.norm.weight": torch.ones(hidden),
        "lm_head.weight": torch.randn(vocab, hidden),
    }
    for i in range(n_layers):
        # Llama-style attention
        sd[f"model.layers.{i}.self_attn.q_proj.weight"] = torch.randn(hidden, hidden)
        sd[f"model.layers.{i}.self_attn.k_proj.weight"] = torch.randn(hidden, hidden)
        sd[f"model.layers.{i}.self_attn.v_proj.weight"] = torch.randn(hidden, hidden)
        sd[f"model.layers.{i}.self_attn.o_proj.weight"] = torch.randn(hidden, hidden)
        sd[f"model.layers.{i}.mlp.gate_proj.weight"] = torch.randn(inter, hidden)
        sd[f"model.layers.{i}.mlp.up_proj.weight"] = torch.randn(inter, hidden)
        sd[f"model.layers.{i}.mlp.down_proj.weight"] = torch.randn(hidden, inter)
        sd[f"model.layers.{i}.input_layernorm.weight"] = torch.ones(hidden)
        sd[f"model.layers.{i}.post_attention_layernorm.weight"] = torch.ones(hidden)
    return sd


def test_gguf_roundtrip_quantize(tmp_path):
    """An F16 GGUF from ssmforge can be quantized to Q4_K_M by the vendored llama-quantize."""
    # 1. Build a fake HybridLlamaMambaConfig-like config
    from types import SimpleNamespace
    config = SimpleNamespace(
        name_or_path="test-hybrid",
        max_position_embeddings=2048,
        hidden_size=256,
        num_hidden_layers=4,
        intermediate_size=1024,
        num_attention_heads=4,
        num_key_value_heads=4,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
    )
    state_dict = _fake_hybrid_state_dict()

    # 2. Write F16 GGUF
    f16_path = tmp_path / "test-hybrid.f16.gguf"
    write_f16_gguf(state_dict, config, tokenizer=None, output_path=f16_path)
    assert f16_path.exists()
    f16_size = f16_path.stat().st_size

    # 3. Quantize via vendored llama-quantize
    q4_path = tmp_path / "test-hybrid.q4km.gguf"
    env = os.environ.copy()
    # Make sure llama-quantize can find libllama.so
    env["LD_LIBRARY_PATH"] = (
        str(LLAMA_QUANTIZE.parent) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    )
    proc = subprocess.run(
        [str(LLAMA_QUANTIZE), str(f16_path), str(q4_path), "Q4_K_M"],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert proc.returncode == 0, f"llama-quantize failed:\nSTDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
    assert q4_path.exists()
    q4_size = q4_path.stat().st_size

    # 4. Verify compression ratio is in the expected ballpark (Q4 ~ 4-5 BPW)
    #    F16 is 16 BPW so Q4 should be ~3-4x smaller
    assert q4_size < f16_size, "Q4 GGUF should be smaller than F16 GGUF"
    print(f"\n  F16: {f16_size / 1e6:.2f} MiB, Q4_K_M: {q4_size / 1e6:.2f} MiB "
          f"(ratio {q4_size / f16_size:.3f})")

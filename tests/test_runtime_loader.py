"""Tests for the runtime GGUF loader (self-contained inference path)."""

from pathlib import Path

import numpy as np
import pytest
import torch
from types import SimpleNamespace

from ssmforge.export.gguf_writer import write_f16_gguf
from ssmforge.runtime.loader import load_hybrid_model_from_gguf
from gguf import GGUFReader


def _build_minimal_hybrid_state_dict(hidden: int = 256, n_layers: int = 4, inter: int = 1024, vocab: int = 100):
    """Source-style state dict (Llama key names)."""
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


def _build_minimal_config(hidden: int = 256, n_layers: int = 4, inter: int = 1024, vocab: int = 100, n_heads: int = 4):
    """Build a minimal HybridLlamaMambaConfig-like object."""
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


def test_loader_loads_f16_gguf_clean(tmp_path):
    """End-to-end: write an F16 GGUF, load it back into a HybridLlamaMambaModel."""
    hidden, n_layers, inter, vocab, n_heads = 256, 4, 1024, 100, 4
    sd = _build_minimal_hybrid_state_dict(hidden, n_layers, inter, vocab)
    cfg = _build_minimal_config(hidden, n_layers, inter, vocab, n_heads)

    f16_path = tmp_path / "test_hybrid.f16.gguf"
    write_f16_gguf(sd, cfg, tokenizer=None, output_path=f16_path)
    assert f16_path.exists()
    assert f16_path.stat().st_size > 0

    # Load it
    model, metadata = load_hybrid_model_from_gguf(f16_path)
    assert model is not None
    assert metadata["general.architecture"] == "ssmforge"

    # Run a forward pass
    input_ids = torch.randint(0, vocab, (1, 8))
    out = model(input_ids=input_ids)
    assert out.logits.shape == (1, 8, vocab)
    assert torch.isfinite(out.logits).all()


def test_loader_detects_ssm_layers_from_gguf(tmp_path):
    """When mamba.* tensors exist in the GGUF, the loader marks those layers as SSM."""
    hidden, n_layers, inter, vocab = 256, 4, 1024, 100
    sd = _build_minimal_hybrid_state_dict(hidden, n_layers, inter, vocab)

    # Inject the FULL set of mamba.* tensors for layer 2 so the model's
    # load_state_dict() doesn't complain. Shape contract matches the model's
    # defaults from mamba_ssm.Mamba2:
    #   d_inner = hidden * 2 = 512
    #   d_state = 128
    #   n_group = 1
    #   headdim = 64
    #   nheads = d_inner / headdim = 8
    #   d_in_proj = 2*d_inner + 2*n_group*d_state + nheads
    #            = 1024 + 256 + 8 = 1288
    #   conv_dim = d_inner + 2*n_group*d_state = 768
    d_inner = hidden * 2  # 512
    d_state = 128
    n_group = 1
    headdim = 64
    nheads = d_inner // headdim
    d_in_proj = 2 * d_inner + 2 * n_group * d_state + nheads  # 1288
    conv_dim = d_inner + 2 * n_group * d_state  # 768

    m = {
        f"model.layers.2.mamba.in_proj.weight":  torch.zeros(d_in_proj, hidden),
        f"model.layers.2.mamba.conv1d.weight":   torch.zeros(conv_dim, 1, 4),
        f"model.layers.2.mamba.conv1d.bias":     torch.zeros(conv_dim),
        f"model.layers.2.mamba.dt_bias":         torch.zeros(nheads),
        f"model.layers.2.mamba.A_log":           torch.zeros(nheads),
        f"model.layers.2.mamba.D":               torch.ones(nheads),
        f"model.layers.2.mamba.norm.weight":     torch.ones(d_inner),
        f"model.layers.2.mamba.out_proj.weight": torch.zeros(hidden, d_inner),
    }
    sd.update(m)

    cfg = _build_minimal_config(hidden, n_layers, inter, vocab)

    f16_path = tmp_path / "test_ssm_layer.f16.gguf"
    write_f16_gguf(sd, cfg, tokenizer=None, output_path=f16_path)

    model, _ = load_hybrid_model_from_gguf(f16_path)
    assert 2 in model.config.ssm_layer_indices


def test_loader_rejects_unknown_architecture(tmp_path):
    """A GGUF with general.architecture != 'ssmforge' is rejected with a clear error."""
    from gguf import GGUFWriter

    writer = GGUFWriter(str(tmp_path / "bogus.gguf"), "llama")  # wrong arch
    writer.add_embedding_length(256)
    writer.add_tensor("test", np.zeros(256, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    with pytest.raises(ValueError, match="Expected architecture='ssmforge'"):
        load_hybrid_model_from_gguf(tmp_path / "bogus.gguf")


def test_loader_returns_metadata(tmp_path):
    """The metadata dict is returned for inspection."""
    sd = _build_minimal_hybrid_state_dict(hidden=256, n_layers=4)
    cfg = _build_minimal_config(hidden=256, n_layers=4)
    f16_path = tmp_path / "test_metadata.f16.gguf"
    write_f16_gguf(sd, cfg, tokenizer=None, output_path=f16_path)

    model, metadata = load_hybrid_model_from_gguf(f16_path)
    assert metadata["general.architecture"] == "ssmforge"
    assert int(metadata["ssmforge.embedding_length"]) == 256
    assert int(metadata["ssmforge.block_count"]) == 4
    assert int(metadata["ssmforge.feed_forward_length"]) == 1024


def test_loader_handles_default_ssm_hparams(tmp_path):
    """If a GGUF doesn't have ssmforge.ssm.* keys, the loader uses defaults matching mamba_ssm.Mamba2."""
    sd = _build_minimal_hybrid_state_dict()
    cfg = _build_minimal_config()
    f16_path = tmp_path / "test_defaults.f16.gguf"
    write_f16_gguf(sd, cfg, tokenizer=None, output_path=f16_path)

    model, _ = load_hybrid_model_from_gguf(f16_path)
    assert model.config.ssm_d_conv == 4
    assert model.config.ssm_d_state == 128
    assert model.config.ssm_dt_rank == 64


def test_loader_loads_qwen2_style_shapes_clean(tmp_path):
    """Regression: loader must handle gguf-py's metadata-vs-data shape mismatch.

    gguf.Writer.add_tensor records shape in llama.cpp convention
    ([in_features, out_features] for Linear, [hidden, vocab] for embeddings)
    but t.data returns the array in PyTorch-native byte order. The loader
    must use t.data.shape (PyTorch order) and ignore the metadata shape.

    This simulates the Qwen2-1.5B case (hidden=1536, intermediate=8960,
    vocab=151936, head_dim=128, kv_heads=2) at small scale.
    """
    hidden, n_layers, inter, vocab, n_heads, n_kv_heads = 64, 2, 256, 1000, 4, 2
    head_dim = hidden // n_heads  # 16

    sd = {
        "model.embed_tokens.weight": torch.arange(vocab * hidden, dtype=torch.float32).reshape(vocab, hidden),
        "model.norm.weight": torch.ones(hidden),
        "lm_head.weight": torch.zeros(vocab, hidden),  # tied embeddings — zero is fine, test only checks shape match
    }
    for i in range(n_layers):
        sd[f"model.layers.{i}.self_attn.q_proj.weight"] = torch.zeros(hidden, hidden)
        sd[f"model.layers.{i}.self_attn.k_proj.weight"] = torch.zeros(head_dim * n_kv_heads, hidden)
        sd[f"model.layers.{i}.self_attn.v_proj.weight"] = torch.zeros(head_dim * n_kv_heads, hidden)
        sd[f"model.layers.{i}.self_attn.o_proj.weight"] = torch.zeros(hidden, hidden)
        sd[f"model.layers.{i}.mlp.gate_proj.weight"] = torch.zeros(inter, hidden)
        sd[f"model.layers.{i}.mlp.up_proj.weight"] = torch.zeros(inter, hidden)
        sd[f"model.layers.{i}.mlp.down_proj.weight"] = torch.zeros(hidden, inter)
        sd[f"model.layers.{i}.input_layernorm.weight"] = torch.ones(hidden)
        sd[f"model.layers.{i}.post_attention_layernorm.weight"] = torch.ones(hidden)

    cfg = SimpleNamespace(
        name_or_path="test",
        max_position_embeddings=2048,
        hidden_size=hidden,
        num_hidden_layers=n_layers,
        intermediate_size=inter,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv_heads,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        vocab_size=vocab,
    )

    f16_path = tmp_path / "qwen2_style.f16.gguf"
    write_f16_gguf(sd, cfg, tokenizer=None, output_path=f16_path)

    # Verify the gguf-py shape-vs-data mismatch exists in the file we just wrote.
    # (We expect this — it's how llama.cpp / gguf-py works.)
    reader = GGUFReader(str(f16_path), mode="r")
    for t in reader.tensors:
        if "embed_tokens" in t.name:
            data_shape = tuple(int(s) for s in t.data.shape)
            meta_shape = tuple(int(s) for s in t.shape)
            # t.data should be PyTorch order (vocab, hidden); metadata reversed
            assert data_shape == (vocab, hidden), f"data shape {data_shape}"
            assert meta_shape == (hidden, vocab), f"meta shape {meta_shape}"
            break

    # The loader should now load WITHOUT size-mismatch errors even though
    # the metadata shape is transposed relative to PyTorch expectations.
    model, _ = load_hybrid_model_from_gguf(f16_path)
    assert model is not None

    # Confirm vocab size inferred correctly
    assert model.config.vocab_size == vocab

    # Forward pass on a Qwen2-shaped input
    input_ids = torch.randint(0, vocab, (1, 8))
    out = model(input_ids=input_ids)
    assert out.logits.shape == (1, 8, vocab)
    assert torch.isfinite(out.logits).all()

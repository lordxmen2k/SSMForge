"""Tests for Gemma2 → hybrid Mamba2 state-dict conversion.

Gemma2 has identical state-dict layout to Llama. The converter is
mostly a delegate but should also strip any bias tensors (since the
hybrid Llama target doesn't have them in current transformers versions).
"""

import torch

from ssmforge.converters.gemma2_to_hybrid import Gemma2ToHybridConverter
from ssmforge.config import LayerSpec, LayerType


def _fake_gemma2_sd(hidden=256, n_layers=4, n_heads=4, n_kv_heads=4, head_dim=64, inter=1024, vocab=100):
    q_size = n_heads * head_dim
    kv_size = n_kv_heads * head_dim
    sd = {
        "model.embed_tokens.weight": torch.randn(vocab, hidden),
        "model.norm.weight": torch.ones(hidden),
        "lm_head.weight": torch.randn(vocab, hidden),
        "_hidden_size": hidden,
    }
    for i in range(n_layers):
        # Gemma2 bias-free (default)
        sd[f"model.layers.{i}.self_attn.q_proj.weight"] = torch.randn(q_size, hidden)
        sd[f"model.layers.{i}.self_attn.k_proj.weight"] = torch.randn(kv_size, hidden)
        sd[f"model.layers.{i}.self_attn.v_proj.weight"] = torch.randn(kv_size, hidden)
        sd[f"model.layers.{i}.self_attn.o_proj.weight"] = torch.randn(hidden, q_size)
        sd[f"model.layers.{i}.mlp.gate_proj.weight"] = torch.randn(inter, hidden)
        sd[f"model.layers.{i}.mlp.up_proj.weight"] = torch.randn(inter, hidden)
        sd[f"model.layers.{i}.mlp.down_proj.weight"] = torch.randn(hidden, inter)
        sd[f"model.layers.{i}.input_layernorm.weight"] = torch.ones(hidden)
        sd[f"model.layers.{i}.post_attention_layernorm.weight"] = torch.ones(hidden)
    return sd


def test_gemma2_converter_handles_bias_free():
    """Standard Gemma2 with bias-free attention and Llama-style MLP works."""
    converter = Gemma2ToHybridConverter()
    src = _fake_gemma2_sd()
    plan = [LayerSpec(index=i, layer_type=LayerType.SSM if i == 1 else LayerType.ATTENTION) for i in range(4)]
    target = converter.convert_state_dict(src, plan)

    # Layer 0 (attention) has the original weights, no mamba
    assert "model.layers.0.self_attn.q_proj.weight" in target
    assert "model.layers.0.mamba.in_proj.weight" not in target
    # Layer 1 (SSM) has mamba
    assert "model.layers.1.mamba.in_proj.weight" in target
    assert "model.layers.1.self_attn.q_proj.weight" not in target


def test_gemma2_converter_strips_attention_biases():
    """If Gemma2 ships with bias tensors (some variants), strip them."""
    converter = Gemma2ToHybridConverter()
    src = _fake_gemma2_sd()
    # Add bias tensors (Gemma2 config allows bias=True in some versions)
    for i in range(4):
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            src[f"model.layers.{i}.self_attn.{proj}.bias"] = torch.zeros(int(256 * 4 / 4))  # shape varies

    plan = [LayerSpec(index=0, layer_type=LayerType.ATTENTION)]
    target = converter.convert_state_dict(src, plan)
    # No .bias keys should appear in target
    bias_keys = [k for k in target if k.endswith(".bias")]
    assert bias_keys == [], f"Bias keys leaked: {bias_keys}"


def test_gemma2_converter_preserves_gqa_shapes():
    """Gemma2-9B uses GQA (n_kv_heads=1 vs n_heads=8). Verify shape preservation."""
    converter = Gemma2ToHybridConverter()
    src = _fake_gemma2_sd(hidden=512, n_heads=8, n_kv_heads=1, head_dim=64)
    plan = [LayerSpec(index=0, layer_type=LayerType.ATTENTION)]
    target = converter.convert_state_dict(src, plan)
    # q_proj: (q_size, hidden) = (512, 512)
    # k_proj, v_proj: (kv_size, hidden) = (64, 512)
    assert target["model.layers.0.self_attn.q_proj.weight"].shape == (512, 512)
    assert target["model.layers.0.self_attn.k_proj.weight"].shape == (64, 512)
    assert target["model.layers.0.self_attn.v_proj.weight"].shape == (64, 512)


def test_gemma2_converter_ssm_shape_contract():
    """SSM layers must match the mamba_ssm.Mamba2 shape contract."""
    converter = Gemma2ToHybridConverter()
    hidden = 256
    src = _fake_gemma2_sd(hidden=hidden)
    plan = [LayerSpec(index=0, layer_type=LayerType.SSM)]
    target = converter.convert_state_dict(src, plan)
    # out_proj must be (hidden, hidden*2)
    assert target["model.layers.0.mamba.out_proj.weight"].shape == (hidden, hidden * 2)


def test_gemma2_converter_preserves_mlp_intermediate():
    """Non-default intermediate_size preserved through conversion."""
    converter = Gemma2ToHybridConverter()
    hidden = 256
    custom_inter = int(hidden * 3.5)  # not the default 4x
    src = _fake_gemma2_sd(hidden=hidden, inter=custom_inter)
    plan = [LayerSpec(index=0, layer_type=LayerType.ATTENTION)]
    target = converter.convert_state_dict(src, plan)
    assert target["model.layers.0.mlp.gate_proj.weight"].shape == (custom_inter, hidden)

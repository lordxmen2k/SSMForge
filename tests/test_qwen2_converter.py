"""Tests for Qwen2 → hybrid Mamba2 state-dict conversion.

Qwen2 differs from Llama mainly in tied embeddings (no lm_head.weight
in many checkpoints) and bias-free attention projections. We verify
the Qwen2ToHybridConverter handles these correctly with REAL tensors.
"""

import torch

from ssmforge.converters.qwen2_to_hybrid import Qwen2ToHybridConverter
from ssmforge.config import LayerSpec, LayerType


def _fake_qwen2_sd(
    hidden: int = 256,
    n_layers: int = 4,
    inter: int = 1024,
    vocab: int = 100,
    tied_embeddings: bool = True,
):
    """Build a Qwen2-shaped state dict (similar to Llama)."""
    sd = {
        "model.embed_tokens.weight": torch.randn(vocab, hidden),
        "model.norm.weight": torch.ones(hidden),
    }
    if not tied_embeddings:
        sd["lm_head.weight"] = torch.randn(vocab, hidden)
    for i in range(n_layers):
        # Qwen2 attention: bias=False on q_proj/k_proj/v_proj/o_proj
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            sd[f"model.layers.{i}.self_attn.{proj}.weight"] = torch.randn(hidden, hidden)
        sd[f"model.layers.{i}.mlp.gate_proj.weight"] = torch.randn(inter, hidden)
        sd[f"model.layers.{i}.mlp.up_proj.weight"] = torch.randn(inter, hidden)
        sd[f"model.layers.{i}.mlp.down_proj.weight"] = torch.randn(hidden, inter)
        sd[f"model.layers.{i}.input_layernorm.weight"] = torch.ones(hidden)
        sd[f"model.layers.{i}.post_attention_layernorm.weight"] = torch.ones(hidden)
    return sd


def test_qwen2_converter_handles_tied_embeddings():
    """When lm_head.weight is missing, the converter synthesizes it from embed_tokens."""
    converter = Qwen2ToHybridConverter()
    src = _fake_qwen2_sd(tied_embeddings=True)
    plan = []
    target = converter.convert_state_dict(src, plan)
    # lm_head must now exist and reference the same tensor object as embed_tokens
    assert "lm_head.weight" in target
    assert "model.embed_tokens.weight" in target
    assert torch.equal(target["lm_head.weight"], target["model.embed_tokens.weight"])


def test_qwen2_converter_handles_explicit_lm_head():
    """When lm_head.weight exists, don't overwrite it."""
    converter = Qwen2ToHybridConverter()
    src = _fake_qwen2_sd(tied_embeddings=False)
    original_head = src["lm_head.weight"]
    plan = []
    target = converter.convert_state_dict(src, plan)
    assert target["lm_head.weight"].data_ptr() == original_head.data_ptr()


def test_qwen2_converter_produces_mamba_for_ssm_layers():
    """SSM layers get Mamba2 weights with correct shape contract."""
    converter = Qwen2ToHybridConverter()
    hidden = 256
    src = _fake_qwen2_sd(hidden=hidden, n_layers=4)
    plan = [LayerSpec(index=i, layer_type=LayerType.SSM if i == 1 else LayerType.ATTENTION) for i in range(4)]
    target = converter.convert_state_dict(src, plan)

    # SSM layer (1) has mamba tensors
    assert "model.layers.1.mamba.in_proj.weight" in target
    assert "model.layers.1.mamba.out_proj.weight" in target
    assert "model.layers.1.mamba.A_log" in target
    assert "model.layers.1.mamba.D" in target
    assert "model.layers.1.mamba.conv1d.weight" in target
    assert "model.layers.1.mamba.norm.weight" in target

    # Critical: out_proj must be (hidden, hidden*expand=hidden*2)
    assert target["model.layers.1.mamba.out_proj.weight"].shape == (hidden, hidden * 2)


def test_qwen2_converter_attention_layers_kept_verbatim():
    """Attention slots get the original attention/MLP tensors."""
    converter = Qwen2ToHybridConverter()
    src = _fake_qwen2_sd(n_layers=4)
    plan = [LayerSpec(index=i, layer_type=LayerType.ATTENTION) for i in range(4)]
    target = converter.convert_state_dict(src, plan)

    # All 4 layers have attention; none have mamba
    for i in range(4):
        assert f"model.layers.{i}.self_attn.q_proj.weight" in target
        assert f"model.layers.{i}.mlp.gate_proj.weight" in target
        assert f"model.layers.{i}.mamba.in_proj.weight" not in target


def test_qwen2_converter_preserves_mlp_shape():
    """Non-default MLP intermediate_size preserved through conversion."""
    converter = Qwen2ToHybridConverter()
    hidden = 256
    custom_inter = int(hidden * 2.75)  # Qwen2 uses 2.75x by default actually
    src = _fake_qwen2_sd(hidden=hidden, inter=custom_inter)
    plan = [LayerSpec(index=0, layer_type=LayerType.ATTENTION)]
    target = converter.convert_state_dict(src, plan)
    assert target["model.layers.0.mlp.gate_proj.weight"].shape == (custom_inter, hidden)


def test_qwen2_converter_handles_no_hidden_size_metadata():
    """Converter must infer _hidden_size from embed_tokens if not provided."""
    converter = Qwen2ToHybridConverter()
    src = _fake_qwen2_sd(hidden=512)
    src.pop("_hidden_size", None)  # ensure it's NOT set
    # _fake_qwen2_sd doesn't set _hidden_size — the converter should detect this
    plan = [LayerSpec(index=0, layer_type=LayerType.SSM)]
    target = converter.convert_state_dict(src, plan)
    # Should still produce a valid mamba out_proj
    assert target["model.layers.0.mamba.out_proj.weight"].shape == (512, 1024)

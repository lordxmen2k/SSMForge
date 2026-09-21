"""Tests for Mistral → hybrid Mamba2 state-dict conversion.

Mistral uses the same state-dict layout as Llama, so the converter is a
thin alias over LlamaToHybridConverter. These tests verify the alias
preserves the real tensor shapes (not just sentinel strings).
"""

import torch

from ssmforge.converters.mistral_to_hybrid import MistralToHybridConverter
from ssmforge.config import LayerSpec, LayerType


def _fake_mistral_sd(hidden: int = 256, n_layers: int = 4, inter: int = 1024, vocab: int = 100):
    """Mistral uses same keys as Llama. Build a realistic state dict."""
    sd = {
        "model.embed_tokens.weight": torch.randn(vocab, hidden),
        "model.norm.weight": torch.ones(hidden),
        "lm_head.weight": torch.randn(vocab, hidden),
    }
    for i in range(n_layers):
        # Mistral doesn't have q/k/v biases (unlike some LLamas)
        sd[f"model.layers.{i}.self_attn.q_proj.weight"] = torch.randn(hidden, hidden)
        sd[f"model.layers.{i}.self_attn.k_proj.weight"] = torch.randn(hidden, hidden)
        sd[f"model.layers.{i}.self_attn.v_proj.weight"] = torch.randn(hidden, hidden)
        sd[f"model.layers.{i}.self_attn.o_proj.weight"] = torch.randn(hidden, hidden)
        sd[f"model.layers.{i}.mlp.gate_proj.weight"] = torch.randn(inter, hidden)
        sd[f"model.layers.{i}.mlp.up_proj.weight"] = torch.randn(inter, hidden)
        sd[f"model.layers.{i}.mlp.down_proj.weight"] = torch.randn(hidden, inter)
        sd[f"model.layers.{i}.input_layernorm.weight"] = torch.ones(hidden)
        sd[f"model.layers.{i}.post_attention_layernorm.weight"] = torch.ones(hidden)
    sd["_hidden_size"] = hidden
    return sd


def test_mistral_converter_handles_real_tensors():
    """End-to-end with real tensors — would have caught the out_proj shape bug."""
    hidden, n_layers, inter, vocab = 256, 4, 1024, 100
    converter = MistralToHybridConverter()
    src = _fake_mistral_sd(hidden, n_layers, inter, vocab)
    plan = [LayerSpec(index=i, layer_type=LayerType.SSM if i == 1 else LayerType.ATTENTION) for i in range(n_layers)]
    target = converter.convert_state_dict(src, plan)

    # Token embeddings / norms / head are copies
    assert torch.equal(target["model.embed_tokens.weight"], src["model.embed_tokens.weight"])
    assert torch.equal(target["model.norm.weight"], src["model.norm.weight"])
    # _hidden_size is stripped from target
    assert "_hidden_size" not in target

    # SSM slot (layer 1) has mamba keys, no attention q/k/v
    assert "model.layers.1.mamba.in_proj.weight" in target
    assert "model.layers.1.mamba.conv1d.weight" in target
    assert "model.layers.1.mamba.dt_bias" in target
    assert "model.layers.1.mamba.A_log" in target
    assert "model.layers.1.mamba.D" in target
    assert "model.layers.1.mamba.norm.weight" in target
    assert "model.layers.1.mamba.out_proj.weight" in target
    assert "model.layers.1.self_attn.q_proj.weight" not in target

    # Attention slot (layer 0) has q/k/v but no mamba
    assert "model.layers.0.self_attn.q_proj.weight" in target
    assert "model.layers.0.mamba.out_proj.weight" not in target

    # MLP weights are preserved for both slots (Mistral has full Llama-style MLP)
    assert "model.layers.1.mlp.gate_proj.weight" in target
    assert torch.equal(target["model.layers.0.mlp.gate_proj.weight"], src["model.layers.0.mlp.gate_proj.weight"])
    assert "model.layers.1.mlp.down_proj.weight" in target


def test_mistral_converter_preserves_intermediate_size():
    """Non-default MLP ratio (Mistral 7B is 4x, but verify)."""
    converter = MistralToHybridConverter()
    src = _fake_mistral_sd(hidden=256, n_layers=2, inter=int(256 * 3.5))  # non-default
    plan = [LayerSpec(index=i, layer_type=LayerType.ATTENTION) for i in range(2)]
    target = converter.convert_state_dict(src, plan)
    assert target["model.layers.0.mlp.gate_proj.weight"].shape == (int(256 * 3.5), 256)


def test_mistral_converter_strips_underscore_metadata():
    """Any _-prefixed key from the source state dict should not propagate."""
    converter = MistralToHybridConverter()
    src = {
        "model.embed_tokens.weight": torch.zeros(10, 32),
        "model.norm.weight": torch.ones(32),
        "_hidden_size": 32,
        "_custom_config_flag": True,  # arbitrary underscore key
    }
    plan = []
    target = converter.convert_state_dict(src, plan)
    for key in target:
        assert not key.startswith("_"), f"underscore key leaked: {key}"

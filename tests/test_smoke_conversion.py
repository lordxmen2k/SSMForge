"""End-to-end smoke test: convert a fake Llama state dict into a hybrid
Mamba2 model and run a forward pass. This is the regression test that
would have caught the v0.1.x out_proj shape bug — run before any PyPI
release.

Kept small (hidden=256) to run on CPU in CI; the shape contract is the
same regardless of hidden_size.
"""

import torch

from ssmforge.config import LayerSpec, LayerType
from ssmforge.converters.llama_to_hybrid import LlamaToHybridConverter
from ssmforge.models.hybrid_llama_mamba import (
    HybridLlamaMambaConfig,
    HybridLlamaMambaModel,
    _MAMBA_AVAILABLE,
)


def _build_fake_llama_state_dict(hidden: int, n_layers: int, vocab: int):
    """Construct a complete Llama-shaped state dict with random tensors."""
    sd = {
        "model.embed_tokens.weight": torch.randn(vocab, hidden),
        "model.norm.weight": torch.ones(hidden),
        "lm_head.weight": torch.randn(vocab, hidden),
    }
    for i in range(n_layers):
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            sd[f"model.layers.{i}.self_attn.{proj}.weight"] = torch.randn(hidden, hidden)
        inter = hidden * 4
        sd[f"model.layers.{i}.mlp.gate_proj.weight"] = torch.randn(inter, hidden)
        sd[f"model.layers.{i}.mlp.up_proj.weight"] = torch.randn(inter, hidden)
        sd[f"model.layers.{i}.mlp.down_proj.weight"] = torch.randn(hidden, inter)
        sd[f"model.layers.{i}.input_layernorm.weight"] = torch.ones(hidden)
        sd[f"model.layers.{i}.post_attention_layernorm.weight"] = torch.ones(hidden)
    return sd


def test_smoke_conversion_tiny_model():
    """Build a tiny hybrid model from a fake Llama state dict and forward it."""
    hidden = 256
    n_layers = 8
    n_heads = 4
    vocab = 100

    cfg = HybridLlamaMambaConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=hidden * 4,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_heads,
        ssm_layer_indices=[2, 5],
    )
    model = HybridLlamaMambaModel(cfg)
    print(f"  Mamba backend: {'CUDA Mamba' if _MAMBA_AVAILABLE else 'Pure-PyTorch fallback'}")

    src_sd = _build_fake_llama_state_dict(hidden, n_layers, vocab)
    src_sd["_hidden_size"] = hidden

    plan = [
        LayerSpec(index=i, layer_type=LayerType.SSM if i in {2, 5} else LayerType.ATTENTION)
        for i in range(n_layers)
    ]

    converter = LlamaToHybridConverter()
    target_sd = converter.convert_state_dict(src_sd, plan)
    assert len(target_sd) > 0

    result = model.load_state_dict(target_sd, strict=False)
    assert result.missing_keys == [], f"Missing keys after conversion: {result.missing_keys[:5]}"
    assert result.unexpected_keys == [], f"Unexpected keys: {result.unexpected_keys[:5]}"

    input_ids = torch.randint(0, vocab, (1, 16))
    out = model(input_ids=input_ids)
    assert out.logits.shape == (1, 16, vocab), f"Bad output shape: {out.logits.shape}"

    out.logits.sum().backward()
    print(f"  Forward+backward OK; output shape: {tuple(out.logits.shape)}")

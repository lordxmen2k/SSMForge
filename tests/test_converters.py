import pytest
from ssmforge.converters import LlamaToHybridConverter, ArchitectureConverterRegistry
from ssmforge.config import LayerSpec, LayerType
from ssmforge.exceptions import UnsupportedArchitectureError


def test_llama_converter_is_registered():
    assert "llama" in ArchitectureConverterRegistry.list_supported()


def test_convert_state_dict_keeps_embeddings():
    converter = LlamaToHybridConverter()
    src_sd = {
        "model.embed_tokens.weight": "embed",
        "model.layers.0.self_attn.q_proj.weight": "q",
        "model.layers.0.mlp.gate_proj.weight": "mlp",
        "model.norm.weight": "norm",
        "lm_head.weight": "head",
        "_hidden_size": 2048,
    }
    plan = [
        LayerSpec(layer_type=LayerType.ATTENTION, index=0),
        LayerSpec(layer_type=LayerType.SSM, index=1),
    ]
    target_sd = converter.convert_state_dict(src_sd, plan)
    assert target_sd["model.embed_tokens.weight"] == "embed"
    assert target_sd["model.layers.0.mlp.gate_proj.weight"] == "mlp"
    assert target_sd["model.norm.weight"] == "norm"
    assert target_sd["lm_head.weight"] == "head"


def test_convert_state_dict_handles_ssm_layer():
    converter = LlamaToHybridConverter()
    src_sd = {
        "model.layers.0.self_attn.q_proj.weight": "q",
        "model.layers.0.self_attn.k_proj.weight": "k",
        "model.layers.0.self_attn.v_proj.weight": "v",
        "model.layers.0.self_attn.o_proj.weight": "o",
        "model.layers.0.mlp.gate_proj.weight": "mlp",
        "model.layers.0.input_layernorm.weight": "ln1",
        "model.layers.0.post_attention_layernorm.weight": "ln2",
        "_hidden_size": 2048,
    }
    plan = [LayerSpec(layer_type=LayerType.SSM, index=0)]
    target_sd = converter.convert_state_dict(src_sd, plan)
    # MLP and LN preserved
    assert target_sd["model.layers.0.mlp.gate_proj.weight"] == "mlp"
    assert target_sd["model.layers.0.input_layernorm.weight"] == "ln1"
    assert target_sd["model.layers.0.post_attention_layernorm.weight"] == "ln2"
    # Mamba weights initialized (at least one mamba.* key present)
    assert any("mamba." in k for k in target_sd.keys())


def test_unsupported_architecture_raises():
    with pytest.raises(UnsupportedArchitectureError):
        ArchitectureConverterRegistry.get("this-arch-does-not-exist-xyz")


def test_attention_layer_passes_through_verbatim():
    converter = LlamaToHybridConverter()
    src_sd = {
        "model.layers.0.self_attn.q_proj.weight": "q",
        "model.layers.0.mlp.gate_proj.weight": "mlp",
        "_hidden_size": 2048,
    }
    plan = [LayerSpec(layer_type=LayerType.ATTENTION, index=0)]
    target_sd = converter.convert_state_dict(src_sd, plan)
    assert target_sd["model.layers.0.self_attn.q_proj.weight"] == "q"
    assert target_sd["model.layers.0.mlp.gate_proj.weight"] == "mlp"
    # No mamba weights for attention layer
    assert not any("mamba." in k for k in target_sd.keys())

from ssmforge.converters import ArchitectureConverterRegistry
from ssmforge.converters.mistral_to_hybrid import MistralToHybridConverter
from ssmforge.config import LayerSpec, LayerType


def test_mistral_converter_is_registered():
    assert "mistral" in ArchitectureConverterRegistry.list_supported()


def test_mistral_converter_copies_non_layer_weights():
    converter = MistralToHybridConverter()
    src_sd = {
        "model.embed_tokens.weight": "embed",
        "model.norm.weight": "norm",
        "lm_head.weight": "head",
        "_hidden_size": 2048,
    }
    plan = []
    target_sd = converter.convert_state_dict(src_sd, plan)
    assert target_sd["model.embed_tokens.weight"] == "embed"
    assert target_sd["model.norm.weight"] == "norm"


def test_mistral_converter_handles_sliding_window_attention():
    converter = MistralToHybridConverter()
    src_sd = {
        "model.layers.0.self_attn.q_proj.weight": "q",
        "model.layers.0.self_attn.k_proj.weight": "k",
        "model.layers.0.self_attn.v_proj.weight": "v",
        "model.layers.0.self_attn.o_proj.weight": "o",
        "model.layers.0.mlp.gate_proj.weight": "mlp",
        "_hidden_size": 2048,
    }
    plan = [LayerSpec(layer_type=LayerType.SSM, index=0)]
    target_sd = converter.convert_state_dict(src_sd, plan)
    assert target_sd["model.layers.0.mlp.gate_proj.weight"] == "mlp"
    assert any("mamba" in k for k in target_sd.keys())

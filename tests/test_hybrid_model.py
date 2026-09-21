import pytest
import torch

pytestmark = pytest.mark.integration


def test_hybrid_model_constructs_from_config():
    from ssmforge.models import HybridLlamaMambaConfig, HybridLlamaMambaModel

    config = HybridLlamaMambaConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        ssm_layer_indices=[1, 3],
        intermediate_size=128,
    )
    model = HybridLlamaMambaModel(config)
    assert model.config.ssm_layer_indices == [1, 3]


def test_hybrid_model_forward_pass_shape():
    from ssmforge.models import HybridLlamaMambaConfig, HybridLlamaMambaModel

    config = HybridLlamaMambaConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        ssm_layer_indices=[1, 3],
        intermediate_size=128,
    )
    model = HybridLlamaMambaModel(config)
    model.eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        output = model(input_ids=input_ids)
    assert "logits" in output
    assert output.logits.shape == (1, 4, 128)


def test_ssm_layer_indices_in_config():
    from ssmforge.models import HybridLlamaMambaConfig

    config = HybridLlamaMambaConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        ssm_layer_indices=[0],
        intermediate_size=64,
    )
    assert config.ssm_layer_indices == [0]
    assert config.ssm_expand == 2  # default

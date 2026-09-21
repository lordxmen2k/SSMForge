from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from ssmforge import convert


def _fake_llama_model():
    m = MagicMock()
    m.config = MagicMock()
    m.config.num_hidden_layers = 16
    m.config.hidden_size = 2048
    m.config.model_type = "llama"
    m.config.architectures = ["LlamaForCausalLM"]
    return m


def _fake_sd():
    return {
        "model.embed_tokens.weight": MagicMock(),
        "model.layers.0.self_attn.q_proj.weight": MagicMock(),
        "model.norm.weight": MagicMock(),
        "lm_head.weight": MagicMock(),
    }


def test_convert_returns_stats_with_dry_run(tmp_path):
    with patch("ssmforge.pipeline._load_model", return_value=(_fake_llama_model(), _fake_sd())):
        result = convert(
            source="fake/model",
            recipe="hybrid-25",
            quantize="F16",
            output_dir=tmp_path,
            dry_run=True,
        )
    assert "layer_mapping" in result.stats
    assert result.stats["layer_count"] == 16
    assert result.stats["ssm_count"] > 0
    assert result.stats["dry_run"] is True
    assert result.stats["arch"] == "llama"


def test_convert_unknown_recipe_raises_unknown(tmp_path):
    with patch("ssmforge.pipeline._load_model", return_value=(_fake_llama_model(), _fake_sd())):
        with pytest.raises(Exception) as exc:
            convert(source="fake", recipe="this-recipe-does-not-exist-xyz", output_dir=tmp_path, dry_run=True)
    msg = str(exc.value).lower()
    assert "this-recipe-does-not-exist-xyz" in msg or "unknownrecipe" in msg


def test_convert_pure_mamba_without_experimental_raises(tmp_path):
    with patch("ssmforge.pipeline._load_model", return_value=(_fake_llama_model(), _fake_sd())):
        with pytest.raises(Exception) as exc:
            convert(source="fake", recipe="pure-mamba", output_dir=tmp_path, dry_run=True)
    assert "experimental" in str(exc.value).lower()


def test_convert_unsupported_arch_raises(tmp_path):
    fake = _fake_llama_model()
    fake.config.model_type = "this-arch-does-not-exist-xyz"
    with patch("ssmforge.pipeline._load_model", return_value=(fake, _fake_sd())):
        with pytest.raises(Exception) as exc:
            convert(source="fake", recipe="hybrid-25", output_dir=tmp_path, dry_run=True)
    assert "not supported" in str(exc.value).lower() or "UnsupportedArchitecture" in str(exc.value)

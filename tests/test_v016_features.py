"""Tests for v0.1.6 features: warning suppression, expanded family inference."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ssmforge.analyze import build_report, build_report_from_config
from ssmforge.analyze.report import _infer_family
from ssmforge.cli import main, _quiet_transformers_warnings


def _fake_sd_for_llama():
    """Build a minimal Llama-shaped state dict for testing family inference."""
    import torch
    embed = torch.zeros(100, 64)
    sd = {"model.embed_tokens.weight": embed}
    for i in range(2):
        sd[f"model.layers.{i}.self_attn.q_proj.weight"] = torch.zeros(64, 64)
        sd[f"model.layers.{i}.self_attn.k_proj.weight"] = torch.zeros(64, 64)
        sd[f"model.layers.{i}.self_attn.v_proj.weight"] = torch.zeros(64, 64)
        sd[f"model.layers.{i}.self_attn.o_proj.weight"] = torch.zeros(64, 64)
        sd[f"model.layers.{i}.mlp.gate_proj.weight"] = torch.zeros(128, 64)
        sd[f"model.layers.{i}.mlp.up_proj.weight"] = torch.zeros(128, 64)
        sd[f"model.layers.{i}.mlp.down_proj.weight"] = torch.zeros(64, 128)
        sd[f"model.layers.{i}.input_layernorm.weight"] = torch.ones(64)
        sd[f"model.layers.{i}.post_attention_layernorm.weight"] = torch.ones(64)
    sd["model.norm.weight"] = torch.ones(64)
    sd["lm_head.weight"] = embed
    return sd


def _llama_cfg(**overrides):
    base = dict(
        model_type="llama",
        architectures=["LlamaForCausalLM"],
        vocab_size=100, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=2048,
        rope_theta=None,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        rope_scaling=None,
        rms_norm_eps=1e-5, tie_word_embeddings=False, attention_bias=False,
        torch_dtype="float32",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# ---------- expanded family inference ----------

def test_family_starcoder():
    """StarCoder / multi-query with model_type=gpt_bigcode should report StarCoder."""
    cfg = _llama_cfg(model_type="gpt_bigcode", tie_word_embeddings=True, attention_bias=False)
    sd = _fake_sd_for_llama()
    r = build_report("bigcode/starcoder", cfg, sd)
    assert "StarCoder" in r["profile"]["family"]


def test_family_starcoder2():
    """StarCoder2 (model_type=starcoder2) should also report StarCoder family."""
    cfg = _llama_cfg(model_type="starcoder2")
    sd = _fake_sd_for_llama()
    r = build_report("bigcode/starcoder2-3b", cfg, sd)
    assert "StarCoder" in r["profile"]["family"]


def test_family_olmo():
    """OLMo (non-MoE) should report 'OLMo' family."""
    cfg = _llama_cfg(model_type="olmo")
    sd = _fake_sd_for_llama()
    r = build_report("allenai/OLMo-1B", cfg, sd)
    assert "OLMo" in r["profile"]["family"]


def test_family_olmoe_when_moe():
    """OLMoE is MoE — should report 'OLMoE (MoE)'."""
    cfg = _llama_cfg(
        model_type="olmoe",
        num_local_experts=64,
        num_experts_per_tok=8,
    )
    import torch
    sd = _fake_sd_for_llama()
    # Add MoE expert tensors
    for i in range(2):
        sd[f"model.layers.{i}.mlp.experts.0.gate_proj.weight"] = torch.zeros(128, 64)
        sd[f"model.layers.{i}.mlp.experts.0.up_proj.weight"] = torch.zeros(128, 64)
        sd[f"model.layers.{i}.mlp.experts.0.down_proj.weight"] = torch.zeros(64, 128)
        sd[f"model.layers.{i}.mlp.experts.63.gate_proj.weight"] = torch.zeros(128, 64)
        sd[f"model.layers.{i}.mlp.experts.63.up_proj.weight"] = torch.zeros(128, 64)
        sd[f"model.layers.{i}.mlp.experts.63.down_proj.weight"] = torch.zeros(64, 128)
        sd[f"model.layers.{i}.mlp.gate.weight"] = torch.zeros(64, 64)
    r = build_report("allenai/OLMoE-1B-7B-0924", cfg, sd)
    assert "OLMoE" in r["profile"]["family"]
    assert r["quirks"]["moe"] is True
    assert r["quirks"]["num_experts"] == 64
    assert r["quirks"]["moe_top_k"] == 8


def test_family_deepseek_moe():
    """DeepSeek-MoE should report DeepSeek-MoE family with expert count."""
    cfg = _llama_cfg(
        model_type="deepseek",
        num_local_experts=60,
        num_experts_per_tok=6,
    )
    import torch
    sd = _fake_sd_for_llama()
    for i in range(2):
        sd[f"model.layers.{i}.mlp.experts.0.gate_proj.weight"] = torch.zeros(128, 64)
        sd[f"model.layers.{i}.mlp.experts.0.up_proj.weight"] = torch.zeros(128, 64)
        sd[f"model.layers.{i}.mlp.experts.0.down_proj.weight"] = torch.zeros(64, 128)
        sd[f"model.layers.{i}.mlp.experts.59.gate_proj.weight"] = torch.zeros(128, 64)
        sd[f"model.layers.{i}.mlp.experts.59.up_proj.weight"] = torch.zeros(128, 64)
        sd[f"model.layers.{i}.mlp.experts.59.down_proj.weight"] = torch.zeros(64, 128)
        sd[f"model.layers.{i}.mlp.gate.weight"] = torch.zeros(60, 64)
    r = build_report("deepseek-ai/deepseek-moe-16b-base", cfg, sd)
    assert "DeepSeek-MoE" in r["profile"]["family"]
    assert "60" in r["profile"]["family"]


def test_family_deepseek_dense():
    """DeepSeek (non-MoE) should report as 'DeepSeek (model_type)'."""
    cfg = _llama_cfg(model_type="deepseek")
    sd = _fake_sd_for_llama()
    r = build_report("deepseek-ai/deepseek-llm-7b-base", cfg, sd)
    assert "DeepSeek" in r["profile"]["family"]


def test_family_qwen_moe():
    """Qwen-MoE should report as Qwen-MoE family."""
    cfg = _llama_cfg(
        model_type="qwen2_moe",
        num_local_experts=60,
        num_experts_per_tok=4,
    )
    import torch
    sd = _fake_sd_for_llama()
    for i in range(2):
        sd[f"model.layers.{i}.mlp.experts.0.gate_proj.weight"] = torch.zeros(128, 64)
        sd[f"model.layers.{i}.mlp.experts.0.up_proj.weight"] = torch.zeros(128, 64)
        sd[f"model.layers.{i}.mlp.experts.0.down_proj.weight"] = torch.zeros(64, 128)
        sd[f"model.layers.{i}.mlp.gate.weight"] = torch.zeros(60, 64)
    r = build_report("Qwen/Qwen1.5-MoE-A2.7B", cfg, sd)
    assert "Qwen-MoE" in r["profile"]["family"]


def test_family_falcon_specific():
    """Falcon with MQA should report 'Falcon (MQA)' specifically, not generic 'MQA'."""
    # Build a Falcon-shaped state dict
    import torch
    sd = {"transformer.word_embeddings.weight": torch.zeros(100, 64)}
    for i in range(2):
        sd[f"transformer.h.{i}.self_attention.query_key_value.weight"] = torch.zeros(192, 64)
        sd[f"transformer.h.{i}.self_attention.dense.weight"] = torch.zeros(64, 64)
        sd[f"transformer.h.{i}.mlp.dense_h_to_4h.weight"] = torch.zeros(256, 64)
        sd[f"transformer.h.{i}.mlp.dense_4h_to_h.weight"] = torch.zeros(64, 256)
        sd[f"transformer.h.{i}.ln_attn.weight"] = torch.ones(64)
        sd[f"transformer.h.{i}.ln_mlp.weight"] = torch.ones(64)
    sd["transformer.ln_f.weight"] = torch.ones(64)
    sd["lm_head.weight"] = sd["transformer.word_embeddings.weight"]

    cfg = SimpleNamespace(
        model_type="falcon",
        architectures=["FalconForCausalLM"],
        vocab_size=100, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=1,
        max_position_embeddings=2048,
        rope_theta=None,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        rope_scaling=None,
        layer_norm_eps=1e-5, tie_word_embeddings=True, attention_bias=False,
        torch_dtype="float32",
    )
    r = build_report("tiiuae/falcon-7b", cfg, sd)
    assert r["profile"]["family"] == "Falcon (MQA)"


def test_family_pythia_specific():
    """Pythia / GPT-NeoX with MQA should report specifically."""
    import torch
    embed = torch.zeros(100, 64)
    sd = {
        "gpt_neox.embed_in.weight": embed,
        "gpt_neox.embed_out.weight": embed,  # tied
    }
    for i in range(2):
        sd[f"gpt_neox.layers.{i}.attention.query_key_value.weight"] = torch.zeros(192, 64)
        sd[f"gpt_neox.layers.{i}.attention.dense.weight"] = torch.zeros(64, 64)
        sd[f"gpt_neox.layers.{i}.mlp.dense_h_to_4h.weight"] = torch.zeros(256, 64)
        sd[f"gpt_neox.layers.{i}.mlp.dense_4h_to_h.weight"] = torch.zeros(64, 256)
        sd[f"gpt_neox.layers.{i}.input_layernorm.weight"] = torch.ones(64)
        sd[f"gpt_neox.layers.{i}.post_attention_layernorm.weight"] = torch.ones(64)
    sd["gpt_neox.final_layer_norm.weight"] = torch.ones(64)

    cfg = SimpleNamespace(
        model_type="gpt_neox",
        architectures=["GPTNeoXForCausalLM"],
        vocab_size=100, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=1,
        max_position_embeddings=2048,
        rope_theta=None,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        rope_scaling=None,
        layer_norm_eps=1e-5, tie_word_embeddings=True, attention_bias=False,
        torch_dtype="float32",
    )
    r = build_report("EleutherAI/pythia-70m", cfg, sd)
    assert r["profile"]["family"] == "Pythia / GPT-NeoX (MQA)"


# ---------- quiet warning suppression ----------

def test_quiet_transformers_warnings_runs():
    """_quiet_transformers_warnings should run without error."""
    _quiet_transformers_warnings()  # should not raise


def test_cli_quiet_flag_does_not_crash_with_warning_suppression(capsys):
    """`ssmforge arch MODEL --quiet --dry-run` should not crash even when transformers warns."""
    mock_config = _llama_cfg()

    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=mock_config)
    cm.__exit__ = MagicMock(return_value=False)

    with patch("ssmforge.cli._arch_load_progress", return_value=cm):
        with pytest.raises(SystemExit) as exc:
            main(["arch", "fake/test", "--dry-run", "--quiet"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    # JSON should still be on stdout
    assert "fake/test" in captured.out

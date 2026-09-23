"""Tests for ssmforge arch — the architecture analyzer."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from ssmforge.analyze import (
    build_report,
    format_report_json,
    scan_state_dict,
    count_state_dict_summary,
    render_summary,
)
from ssmforge.analyze.state_dict_scan import QuirkReport


# ---------- Fixtures ----------

def _fake_state_dict_qwen2(hidden=64, n_layers=2, inter=128, vocab=100, n_heads=4, n_kv_heads=2):
    """Build a fake Qwen2-style state dict with biases and tied embeddings."""
    head_dim = hidden // n_heads
    kv_dim = head_dim * n_kv_heads
    sd = {
        "model.embed_tokens.weight": torch.randn(vocab, hidden),
    }
    for i in range(n_layers):
        sd[f"model.layers.{i}.self_attn.q_proj.weight"] = torch.randn(hidden, hidden)
        sd[f"model.layers.{i}.self_attn.q_proj.bias"] = torch.zeros(hidden)
        sd[f"model.layers.{i}.self_attn.k_proj.weight"] = torch.randn(kv_dim, hidden)
        sd[f"model.layers.{i}.self_attn.k_proj.bias"] = torch.zeros(kv_dim)
        sd[f"model.layers.{i}.self_attn.v_proj.weight"] = torch.randn(kv_dim, hidden)
        sd[f"model.layers.{i}.self_attn.v_proj.bias"] = torch.zeros(kv_dim)
        sd[f"model.layers.{i}.self_attn.o_proj.weight"] = torch.randn(hidden, hidden)
        sd[f"model.layers.{i}.mlp.gate_proj.weight"] = torch.randn(inter, hidden)
        sd[f"model.layers.{i}.mlp.up_proj.weight"] = torch.randn(inter, hidden)
        sd[f"model.layers.{i}.mlp.down_proj.weight"] = torch.randn(hidden, inter)
        sd[f"model.layers.{i}.input_layernorm.weight"] = torch.ones(hidden)
        sd[f"model.layers.{i}.post_attention_layernorm.weight"] = torch.ones(hidden)
    sd["model.norm.weight"] = torch.ones(hidden)
    # Tied: lm_head shares storage with embed_tokens
    sd["lm_head.weight"] = sd["model.embed_tokens.weight"]
    return sd


def _fake_state_dict_llama(hidden=64, n_layers=2, inter=128, vocab=100, n_heads=4):
    """Build a fake Llama-style state dict (no biases, separate lm_head, no grouped attention)."""
    sd = {
        "model.embed_tokens.weight": torch.randn(vocab, hidden),
    }
    for i in range(n_layers):
        sd[f"model.layers.{i}.self_attn.q_proj.weight"] = torch.randn(hidden, hidden)
        sd[f"model.layers.{i}.self_attn.k_proj.weight"] = torch.randn(hidden, hidden)
        sd[f"model.layers.{i}.self_attn.v_proj.weight"] = torch.randn(hidden, hidden)
        sd[f"model.layers.{i}.self_attn.o_proj.weight"] = torch.randn(hidden, hidden)
        sd[f"model.layers.{i}.mlp.gate_proj.weight"] = torch.randn(inter, hidden)
        sd[f"model.layers.{i}.mlp.up_proj.weight"] = torch.randn(inter, hidden)
        sd[f"model.layers.{i}.mlp.down_proj.weight"] = torch.randn(hidden, inter)
        sd[f"model.layers.{i}.input_layernorm.weight"] = torch.ones(hidden)
        sd[f"model.layers.{i}.post_attention_layernorm.weight"] = torch.ones(hidden)
    sd["model.norm.weight"] = torch.ones(hidden)
    sd["lm_head.weight"] = torch.randn(vocab, hidden)  # separate, not tied
    return sd


def _fake_state_dict_phi3(hidden=64, n_layers=2, inter=128, vocab=100):
    """Phi-3 style: fused QKV + fused gate_up, no biases."""
    sd = {
        "model.embed_tokens.weight": torch.randn(vocab, hidden),
    }
    for i in range(n_layers):
        # Phi-3 has fused qkv_proj with shape (q_dim + 2*kv_dim, hidden)
        # Simplified: just use a single fused weight
        sd[f"model.layers.{i}.self_attn.qkv_proj.weight"] = torch.randn(hidden * 3, hidden)
        sd[f"model.layers.{i}.self_attn.o_proj.weight"] = torch.randn(hidden, hidden)
        # Phi-3 fused gate_up_proj
        sd[f"model.layers.{i}.mlp.gate_up_proj.weight"] = torch.randn(inter * 2, hidden)
        sd[f"model.layers.{i}.mlp.down_proj.weight"] = torch.randn(hidden, inter)
        sd[f"model.layers.{i}.input_layernorm.weight"] = torch.ones(hidden)
        sd[f"model.layers.{i}.post_attention_layernorm.weight"] = torch.ones(hidden)
    sd["model.norm.weight"] = torch.ones(hidden)
    sd["lm_head.weight"] = torch.randn(vocab, hidden)
    return sd


def _fake_config(hidden=64, n_layers=2, inter=128, vocab=100, n_heads=4, n_kv_heads=None):
    return SimpleNamespace(
        model_type="test",
        architectures=["TestForCausalLM"],
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=inter,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv_heads or n_heads,
        max_position_embeddings=2048,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        attention_bias=False,
        torch_dtype=torch.float32,
    )


# ---------- scan_state_dict ----------

def test_scan_detects_attention_bias():
    sd = _fake_state_dict_qwen2()
    # _fake_state_dict_qwen2 uses zeros for biases (vestigial, like real Qwen2).
    # v0.2.1 fix: attention_bias stays False when biases are all zero.
    # We still record that bias keys are present (informational).
    report = scan_state_dict(sd)
    assert report.attention_bias is False
    assert any("q_proj.bias" in k for k in report.bias_keys_found)


def test_scan_detects_nonzero_attention_bias():
    """When bias tensors are non-zero (Phi-3 / Mistral-style), quirk = True."""
    sd = _fake_state_dict_qwen2()
    # Replace zero biases with non-zero ones to simulate a real bias-using model
    for k in list(sd.keys()):
        if k.endswith(".bias"):
            sd[k] = torch.randn_like(sd[k])
    report = scan_state_dict(sd)
    assert report.attention_bias is True
    assert any("q_proj.bias" in k for k in report.bias_keys_found)


def test_scan_no_bias_for_llama_style():
    sd = _fake_state_dict_llama()
    report = scan_state_dict(sd)
    assert report.attention_bias is False
    assert report.bias_keys_found == []


def test_scan_detects_tied_embeddings():
    sd = _fake_state_dict_qwen2()
    report = scan_state_dict(sd)
    assert report.tied_embeddings is True


def test_scan_no_tied_when_separate():
    sd = _fake_state_dict_llama()
    report = scan_state_dict(sd)
    assert report.tied_embeddings is False


def test_scan_detects_fused_qkv():
    sd = _fake_state_dict_phi3()
    report = scan_state_dict(sd)
    assert report.fused_qkv is True
    assert any("qkv_proj" in k for k in report.fused_keys_found)


def test_scan_detects_fused_gate_up():
    sd = _fake_state_dict_phi3()
    report = scan_state_dict(sd)
    assert report.fused_gate_up is True


def test_scan_no_fused_for_standard():
    sd = _fake_state_dict_llama()
    report = scan_state_dict(sd)
    assert report.fused_qkv is False
    assert report.fused_gate_up is False


def test_scan_detects_grouped_attention():
    sd = _fake_state_dict_qwen2(n_kv_heads=2)
    report = scan_state_dict(sd)
    assert report.grouped_attention is True


def test_scan_no_grouped_for_dense():
    sd = _fake_state_dict_llama()  # n_heads == n_kv_heads
    report = scan_state_dict(sd)
    assert report.grouped_attention is False


def test_scan_handles_empty_state_dict():
    report = scan_state_dict({})
    assert isinstance(report, QuirkReport)
    assert report.attention_bias is False


def test_scan_handles_missing_lm_head():
    """If lm_head.weight is missing, treat as tied embeddings."""
    sd = _fake_state_dict_llama()
    del sd["lm_head.weight"]
    report = scan_state_dict(sd)
    assert report.tied_embeddings is True


def test_scan_detects_moe():
    """MoE models have expert tensors — should be detected."""
    sd = _fake_state_dict_llama()
    # Add MoE-style tensors
    sd["model.layers.0.block_sparse_moe.gate.weight"] = torch.randn(8, 64)
    sd["model.layers.0.block_sparse_moe.experts.0.w1.weight"] = torch.randn(128, 64)
    sd["model.layers.0.block_sparse_moe.experts.0.w2.weight"] = torch.randn(64, 128)
    report = scan_state_dict(sd)
    assert report.moe is True


def test_scan_handles_numpy_arrays():
    """Some state dicts come back as numpy (e.g. via gguf-py). Should not crash."""
    sd = _fake_state_dict_llama()
    sd_np = {k: v.numpy() if isinstance(v, torch.Tensor) else v for k, v in sd.items()}
    report = scan_state_dict(sd_np)
    assert report.attention_bias is False
    assert report.tied_embeddings is False


# ---------- count_state_dict_summary ----------

def test_count_summary_basic():
    sd = _fake_state_dict_llama()
    summary = count_state_dict_summary(sd)
    assert summary["total_tensors"] == len(sd)
    assert summary["total_params"] > 0
    assert summary["tensor_breakdown"]["embeddings"] == 1
    assert summary["tensor_breakdown"]["attention_weights"] == 4 * 2  # 4 per layer * 2 layers
    assert summary["tensor_breakdown"]["mlp_weights"] == 3 * 2
    assert summary["tensor_breakdown"]["layer_norms"] == 2 * 2
    assert summary["tensor_breakdown"]["final_norm"] == 1
    assert summary["tensor_breakdown"]["lm_head"] == 1


def test_count_summary_handles_numpy():
    sd = _fake_state_dict_llama()
    sd_np = {k: v.numpy() if isinstance(v, torch.Tensor) else v for k, v in sd.items()}
    summary = count_state_dict_summary(sd_np)
    # numpy arrays have .size attribute
    assert summary["total_params"] > 0


def test_count_summary_counts_biases():
    sd = _fake_state_dict_qwen2()
    summary = count_state_dict_summary(sd)
    assert summary["tensor_breakdown"]["attention_biases"] == 3 * 2  # q/k/v per layer


# ---------- build_report ----------

def test_build_report_qwen2():
    sd = _fake_state_dict_qwen2(n_layers=2)
    cfg = _fake_config(n_layers=2, n_kv_heads=2)
    report = build_report(
        model_id="Qwen/Qwen2-test",
        config=cfg,
        state_dict=sd,
    )
    assert report["model_id"] == "Qwen/Qwen2-test"
    assert report["model_type"] == "test"
    # Qwen2 has config.attention_bias=None (the actual config ships it that way)
    # AND the weights include vestigial zero-initialized bias tensors.
    # Config wins per the v0.2.1 fix — quirk reports False (Qwen convention).
    # The bias_keys_found list still records that the tensors are present.
    assert report["quirks"]["attention_bias"] is False
    assert len(report["quirks"]["bias_keys_found"]) > 0
    assert report["quirks"]["tied_embeddings"] is True
    assert report["quirks"]["grouped_attention"] is True
    assert report["compatibility"] is not None
    assert report["compatibility"]["is_compatible"] is True


def test_build_report_llama_no_quirks():
    sd = _fake_state_dict_llama()
    cfg = _fake_config()
    report = build_report("test/llama", cfg, sd)
    assert report["quirks"]["attention_bias"] is False
    assert report["quirks"]["tied_embeddings"] is False
    assert report["quirks"]["grouped_attention"] is False


def test_build_report_moe_is_incompatible():
    sd = _fake_state_dict_llama()
    sd["model.layers.0.block_sparse_moe.experts.0.w1.weight"] = torch.randn(128, 64)
    cfg = _fake_config()
    report = build_report("test/moe", cfg, sd)
    assert report["quirks"]["moe"] is True
    assert report["compatibility"]["is_compatible"] is False
    issues = report["compatibility"]["issues"]
    assert any("moe" in i["quirk"].lower() for i in issues)


def test_build_report_uses_cfg_defaults():
    """Missing config fields should default, not crash."""
    cfg = SimpleNamespace(
        model_type="custom",
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        # missing: num_key_value_heads, rope_theta, rms_norm_eps, etc.
    )
    sd = _fake_state_dict_llama()
    report = build_report("test/custom", cfg, sd)
    assert report["config"]["max_position_embeddings"] is None  # missing → None


def test_build_report_json_serializable():
    sd = _fake_state_dict_llama()
    cfg = _fake_config()
    report = build_report("test", cfg, sd)
    json_text = format_report_json(report)
    parsed = json.loads(json_text)
    assert parsed["model_id"] == "test"
    assert "quirks" in parsed


def test_render_summary_contains_key_fields():
    sd = _fake_state_dict_qwen2()
    cfg = _fake_config(n_kv_heads=2)
    report = build_report("test", cfg, sd)
    summary = render_summary(report)
    assert "test" in summary
    assert "Qwen" in summary or "model_type" in summary
    assert "attention_bias" in summary
    assert "tied_embeddings" in summary
    assert "grouped_attention" in summary


def test_compatibility_notes_have_required_fields():
    """Every issue must have severity, quirk, message."""
    sd = _fake_state_dict_llama()
    sd["model.layers.0.block_sparse_moe.experts.0.w1.weight"] = torch.randn(128, 64)
    cfg = _fake_config()
    report = build_report("test", cfg, sd)
    for issue in report["compatibility"]["issues"]:
        assert "severity" in issue
        assert "quirk" in issue
        assert "message" in issue
        assert issue["severity"] in ("info", "warning", "error")


def test_resolves_rope_theta_from_scaling_dict():
    """Qwen2 stores rope_theta in rope_scaling dict, not as a direct field.

    This was the silent bug that produced gibberish output: GGUF writer
    was defaulting to 10000.0 for Qwen2-1.5B when its actual rope_theta
    is 1,000,000.0.
    """
    from types import SimpleNamespace

    # Qwen2-style: direct field is None, rope_scaling has the value
    cfg = SimpleNamespace(
        model_type="qwen2", architectures=["Qwen2ForCausalLM"],
        vocab_size=100, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=2048,
        rope_theta=None,
        rope_scaling={"rope_theta": 1000000.0, "rope_type": "default"},
        rms_norm_eps=1e-6, tie_word_embeddings=True, attention_bias=True,
        torch_dtype="bfloat16",
    )
    sd = _fake_state_dict_qwen2()
    report = build_report("test/qwen2", cfg, sd)
    assert report["config"]["rope_theta"] == 1000000.0
    assert report["config"]["rope_theta_source"] == "config.rope_scaling.rope_theta"


def test_resolves_rope_theta_from_parameters_dict():
    """transformers ≥ 5.0 stores rope_theta in `rope_parameters` (unified).

    Bug surfaced by running `ssmforge arch TinyLlama/TinyLlama-1.1B-Chat-v1.0`:
    the report said source = "config.rope_scaling.rope_theta" but the actual
    transformers source for that value was `rope_parameters.rope_theta`.
    `config.rope_theta` returned None because newer transformers moves the
    top-level rope_theta into `rope_parameters` during config load.
    """
    from types import SimpleNamespace

    # transformers ≥ 5.0 style: rope_theta only in rope_parameters
    cfg = SimpleNamespace(
        model_type="llama", architectures=["LlamaForCausalLM"],
        vocab_size=100, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=2048,
        rope_theta=None,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        rope_scaling=None,
        rms_norm_eps=1e-5, tie_word_embeddings=False, attention_bias=False,
        torch_dtype="bfloat16",
    )
    sd = _fake_state_dict_llama()
    report = build_report("test/llama5x", cfg, sd)
    assert report["config"]["rope_theta"] == 10000.0
    assert report["config"]["rope_theta_source"] == "config.rope_parameters.rope_theta"


def test_rope_theta_resolution_priority():
    """When multiple rope_theta locations exist, direct field wins."""
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        model_type="llama", architectures=["LlamaForCausalLM"],
        vocab_size=100, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=2048,
        rope_theta=10000.0,
        rope_parameters={"rope_theta": 99999.0},  # should be ignored
        rope_scaling={"rope_theta": 88888.0},     # should be ignored
        rms_norm_eps=1e-5, tie_word_embeddings=False, attention_bias=False,
        torch_dtype="float32",
    )
    sd = _fake_state_dict_llama()
    report = build_report("test/llama", cfg, sd)
    # Direct field takes priority
    assert report["config"]["rope_theta"] == 10000.0
    assert report["config"]["rope_theta_source"] == "config.rope_theta"


def test_resolves_rope_theta_from_direct_field():
    """Llama-style: rope_theta is a direct field, no rope_scaling."""
    from types import SimpleNamespace
    cfg = SimpleNamespace(
        model_type="llama", architectures=["LlamaForCausalLM"],
        vocab_size=100, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=2048,
        rope_theta=10000.0,
        rope_scaling=None,
        rms_norm_eps=1e-6, tie_word_embeddings=False, attention_bias=False,
        torch_dtype="float32",
    )
    sd = _fake_state_dict_llama()
    report = build_report("test/llama", cfg, sd)
    assert report["config"]["rope_theta"] == 10000.0
    assert report["config"]["rope_theta_source"] == "config.rope_theta"


def test_reports_attention_bias_from_state_dict_when_config_missing():
    """Config doesn't expose attention_bias but state_dict has biases.

    Some HF configs (Qwen2) have attention_bias=None at the config level
    but ship with vestigial zero-initialized bias tensors. Per v0.2.1 fix:
    config.attention_bias=None + zero biases → quirk is False (vestigial).

    We verify that the bias keys are still recorded for transparency.
    """
    from types import SimpleNamespace
    cfg = SimpleNamespace(
        model_type="qwen2", architectures=["Qwen2ForCausalLM"],
        vocab_size=100, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=2048,
        rope_theta=None, rope_scaling={"rope_theta": 1000000.0, "rope_type": "default"},
        rms_norm_eps=1e-6, tie_word_embeddings=True,
        attention_bias=None,  # config doesn't expose this
        torch_dtype="bfloat16",
    )
    sd = _fake_state_dict_qwen2()  # has biases (zero in fake_qwen2 helper)
    report = build_report("test/qwen2", cfg, sd)
    # Zero biases + None config = quirk is False (vestigial like real Qwen2)
    assert report["config"]["attention_bias"] is False
    assert report["quirks"]["attention_bias"] is False
    # But the bias_keys_found list still records that the tensors exist
    assert any("q_proj.bias" in k for k in report["quirks"]["bias_keys_found"])

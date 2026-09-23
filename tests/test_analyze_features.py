"""Tests for ssmforge arch subcommands: markdown output, diff mode, doctor."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from ssmforge.analyze import (
    build_report,
    diff_reports,
    format_report_json,
    format_report_markdown,
)
from ssmforge.cli import main


def _fake_qwen2_sd(hidden=64, n_layers=2, inter=128, vocab=100, n_kv_heads=2):
    sd = {"model.embed_tokens.weight": torch.zeros(vocab, hidden)}
    head_dim = hidden // 14  # Qwen2-0.5B has 14 heads
    kv_dim = head_dim * n_kv_heads
    for i in range(n_layers):
        n_heads = 14
        sd[f"model.layers.{i}.self_attn.q_proj.weight"] = torch.zeros(hidden, hidden)
        sd[f"model.layers.{i}.self_attn.q_proj.bias"] = torch.zeros(hidden)
        sd[f"model.layers.{i}.self_attn.k_proj.weight"] = torch.zeros(kv_dim, hidden)
        sd[f"model.layers.{i}.self_attn.k_proj.bias"] = torch.zeros(kv_dim)
        sd[f"model.layers.{i}.self_attn.v_proj.weight"] = torch.zeros(kv_dim, hidden)
        sd[f"model.layers.{i}.self_attn.v_proj.bias"] = torch.zeros(kv_dim)
        sd[f"model.layers.{i}.self_attn.o_proj.weight"] = torch.zeros(hidden, hidden)
        sd[f"model.layers.{i}.mlp.gate_proj.weight"] = torch.zeros(inter, hidden)
        sd[f"model.layers.{i}.mlp.up_proj.weight"] = torch.zeros(inter, hidden)
        sd[f"model.layers.{i}.mlp.down_proj.weight"] = torch.zeros(hidden, inter)
        sd[f"model.layers.{i}.input_layernorm.weight"] = torch.ones(hidden)
        sd[f"model.layers.{i}.post_attention_layernorm.weight"] = torch.ones(hidden)
    sd["model.norm.weight"] = torch.ones(hidden)
    sd["lm_head.weight"] = sd["model.embed_tokens.weight"]
    return sd, 14, n_kv_heads


def _fake_qwen2_config(hidden=64, n_layers=2, inter=128, vocab=100, n_kv_heads=2):
    return SimpleNamespace(
        model_type="qwen2", architectures=["Qwen2ForCausalLM"],
        vocab_size=vocab, hidden_size=hidden, intermediate_size=inter,
        num_hidden_layers=n_layers, num_attention_heads=14,
        num_key_value_heads=n_kv_heads,
        max_position_embeddings=32768,
        rope_theta=None,
        rope_parameters={"rope_theta": 1000000.0, "rope_type": "default"},
        rope_scaling=None,
        rms_norm_eps=1e-6, tie_word_embeddings=True, attention_bias=None,
        torch_dtype="bfloat16",
    )


def _fake_llama_sd(hidden=64, n_layers=2, inter=128, vocab=100):
    sd = {"model.embed_tokens.weight": torch.zeros(vocab, hidden)}
    for i in range(n_layers):
        sd[f"model.layers.{i}.self_attn.q_proj.weight"] = torch.zeros(hidden, hidden)
        sd[f"model.layers.{i}.self_attn.k_proj.weight"] = torch.zeros(hidden, hidden)
        sd[f"model.layers.{i}.self_attn.v_proj.weight"] = torch.zeros(hidden, hidden)
        sd[f"model.layers.{i}.self_attn.o_proj.weight"] = torch.zeros(hidden, hidden)
        sd[f"model.layers.{i}.mlp.gate_proj.weight"] = torch.zeros(inter, hidden)
        sd[f"model.layers.{i}.mlp.up_proj.weight"] = torch.zeros(inter, hidden)
        sd[f"model.layers.{i}.mlp.down_proj.weight"] = torch.zeros(hidden, inter)
        sd[f"model.layers.{i}.input_layernorm.weight"] = torch.ones(hidden)
        sd[f"model.layers.{i}.post_attention_layernorm.weight"] = torch.ones(hidden)
    sd["model.norm.weight"] = torch.ones(hidden)
    sd["lm_head.weight"] = torch.zeros(vocab, hidden)
    return sd


def _fake_llama_config(hidden=64, n_layers=2, inter=128, vocab=100):
    return SimpleNamespace(
        model_type="llama", architectures=["LlamaForCausalLM"],
        vocab_size=vocab, hidden_size=hidden, intermediate_size=inter,
        num_hidden_layers=n_layers, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=2048,
        rope_theta=10000.0,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        rope_scaling=None,
        rms_norm_eps=1e-5, tie_word_embeddings=False, attention_bias=False,
        torch_dtype="float32",
    )


# ---------- markdown rendering ----------

def test_markdown_output_contains_key_sections():
    """Markdown report should have title, profile, config, quirks, compatibility, state_dict."""
    sd, _, _ = _fake_qwen2_sd(hidden=896, n_layers=2, inter=128, vocab=100, n_kv_heads=2)
    # Adjust k/v size for n_heads=14
    head_dim = 896 // 14
    sd2 = {"model.embed_tokens.weight": torch.zeros(100, 896)}
    for i in range(2):
        sd2[f"model.layers.{i}.self_attn.q_proj.weight"] = torch.zeros(896, 896)
        sd2[f"model.layers.{i}.self_attn.q_proj.bias"] = torch.zeros(896)
        sd2[f"model.layers.{i}.self_attn.k_proj.weight"] = torch.zeros(head_dim * 2, 896)
        sd2[f"model.layers.{i}.self_attn.k_proj.bias"] = torch.zeros(head_dim * 2)
        sd2[f"model.layers.{i}.self_attn.v_proj.weight"] = torch.zeros(head_dim * 2, 896)
        sd2[f"model.layers.{i}.self_attn.v_proj.bias"] = torch.zeros(head_dim * 2)
        sd2[f"model.layers.{i}.self_attn.o_proj.weight"] = torch.zeros(896, 896)
        sd2[f"model.layers.{i}.mlp.gate_proj.weight"] = torch.zeros(4864, 896)
        sd2[f"model.layers.{i}.mlp.up_proj.weight"] = torch.zeros(4864, 896)
        sd2[f"model.layers.{i}.mlp.down_proj.weight"] = torch.zeros(896, 4864)
        sd2[f"model.layers.{i}.input_layernorm.weight"] = torch.ones(896)
        sd2[f"model.layers.{i}.post_attention_layernorm.weight"] = torch.ones(896)
    sd2["model.norm.weight"] = torch.ones(896)
    sd2["lm_head.weight"] = sd2["model.embed_tokens.weight"]

    cfg = SimpleNamespace(
        model_type="qwen2", architectures=["Qwen2ForCausalLM"],
        vocab_size=100, hidden_size=896, intermediate_size=4864,
        num_hidden_layers=2, num_attention_heads=14, num_key_value_heads=2,
        max_position_embeddings=32768,
        rope_theta=None,
        rope_parameters={"rope_theta": 1000000.0, "rope_type": "default"},
        rope_scaling=None,
        rms_norm_eps=1e-6, tie_word_embeddings=True, attention_bias=None,
        torch_dtype="bfloat16",
    )
    report = build_report("test/qwen2", cfg, sd2)
    md = format_report_markdown(report)
    # Must contain all major sections
    assert "# `ssmforge arch`" in md
    assert "## Profile" in md
    assert "## Model config" in md
    assert "## Detected quirks" in md
    assert "## Compatibility" in md
    assert "## State dict" in md
    # Must include quirk checklist with markdown table
    assert "| Quirk | Detected |" in md
    # Must include rope_theta with source
    assert "rope_parameters.rope_theta" in md


def test_cli_arch_markdown_format(tmp_path, capsys):
    """`ssmforge arch MODEL --format markdown` writes markdown to stdout."""
    sd, _, _ = _fake_qwen2_sd()
    cfg = _fake_qwen2_config()

    mock_model = MagicMock()
    mock_model.config = cfg
    mock_model.state_dict.return_value = sd

    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=mock_model)
    cm.__exit__ = MagicMock(return_value=False)
    with patch("ssmforge.cli._arch_load_progress", return_value=cm):
        with pytest.raises(SystemExit) as exc:
            main(["arch", "fake/test", "--format", "markdown", "--quiet"])
    assert exc.value.code in (0, 2)
    captured = capsys.readouterr()
    assert "# `ssmforge arch`" in captured.out


# ---------- diff mode ----------

def test_diff_identical_models():
    """Diff of a model against itself should be identical."""
    sd, _, _ = _fake_qwen2_sd()
    cfg = _fake_qwen2_config()
    report = build_report("test/qwen2", cfg, sd)
    diff = diff_reports(report, report)
    assert diff["identical"] is True
    assert diff["differences"] == []
    assert diff["added_quirks"] == []
    assert diff["removed_quirks"] == []


def test_diff_different_architectures():
    """Diff between Qwen2 and Llama should show tied_embeddings and grouped_attention."""
    sd_q, _, _ = _fake_qwen2_sd()
    cfg_q = _fake_qwen2_config()
    report_q = build_report("Qwen/Qwen2", cfg_q, sd_q)

    sd_l = _fake_llama_sd()
    cfg_l = _fake_llama_config()
    report_l = build_report("meta-llama/Llama", cfg_l, sd_l)

    diff = diff_reports(report_q, report_l)
    assert diff["identical"] is False
    assert len(diff["differences"]) > 0
    diff_fields = [d["field"] for d in diff["differences"]]
    # Qwen2 has tied embeddings (config says True) + grouped attention (kv:2/4); Llama doesn't
    assert "tie_word_embeddings" in diff_fields
    assert "grouped_attention" in diff_fields
    assert "rope_theta" in diff_fields
    # Note: attention_bias is False for both in v0.2.1 (Qwen2 has zero biases),
    # so it's no longer a diff field. grouped_attention IS the differentiator.


def test_cli_arch_diff_mode(capsys):
    """`ssmforge arch A --diff B` should produce a diff report."""
    sd_q, _, _ = _fake_qwen2_sd()
    cfg_q = _fake_qwen2_config()
    report_q = build_report("Qwen/Qwen2", cfg_q, sd_q)

    sd_l = _fake_llama_sd()
    cfg_l = _fake_llama_config()
    report_l = build_report("meta-llama/Llama", cfg_l, sd_l)

    # Mock two separate model loads
    mock_q = MagicMock()
    mock_q.config = cfg_q
    mock_q.state_dict.return_value = sd_q
    mock_l = MagicMock()
    mock_l.config = cfg_l
    mock_l.state_dict.return_value = sd_l

    cm_q = MagicMock()
    cm_q.__enter__ = MagicMock(return_value=mock_q)
    cm_q.__exit__ = MagicMock(return_value=False)
    cm_l = MagicMock()
    cm_l.__enter__ = MagicMock(return_value=mock_l)
    cm_l.__exit__ = MagicMock(return_value=False)

    # First call returns Qwen2, second returns Llama
    with patch("ssmforge.cli._arch_load_progress", side_effect=[cm_q, cm_l]):
        with pytest.raises(SystemExit) as exc:
            main(["arch", "Qwen/Qwen2", "--diff", "meta-llama/Llama", "--quiet"])
    assert exc.value.code in (0, 2)
    captured = capsys.readouterr()
    # Output is JSON when --format=json (default)
    parsed = json.loads(captured.out)
    assert parsed["model_a"] == "Qwen/Qwen2"
    assert parsed["model_b"] == "meta-llama/Llama"
    assert parsed["identical"] is False
    assert "differences" in parsed


def test_cli_arch_diff_markdown_format(capsys):
    """`ssmforge arch A --diff B --format markdown` should produce markdown diff."""
    sd_q, _, _ = _fake_qwen2_sd()
    cfg_q = _fake_qwen2_config()
    report_q = build_report("Qwen/Qwen2", cfg_q, sd_q)

    sd_l = _fake_llama_sd()
    cfg_l = _fake_llama_config()
    report_l = build_report("meta-llama/Llama", cfg_l, sd_l)

    mock_q = MagicMock()
    mock_q.config = cfg_q
    mock_q.state_dict.return_value = sd_q
    mock_l = MagicMock()
    mock_l.config = cfg_l
    mock_l.state_dict.return_value = sd_l

    cm_q = MagicMock()
    cm_q.__enter__ = MagicMock(return_value=mock_q)
    cm_q.__exit__ = MagicMock(return_value=False)
    cm_l = MagicMock()
    cm_l.__enter__ = MagicMock(return_value=mock_l)
    cm_l.__exit__ = MagicMock(return_value=False)

    with patch("ssmforge.cli._arch_load_progress", side_effect=[cm_q, cm_l]):
        with pytest.raises(SystemExit) as exc:
            main(["arch", "Qwen/Qwen2", "--diff", "meta-llama/Llama",
                  "--format", "markdown", "--quiet"])
    assert exc.value.code in (0, 2)
    captured = capsys.readouterr()
    assert "# Architectural diff" in captured.out
    assert "Model A:" in captured.out
    assert "Model B:" in captured.out


# ---------- doctor ----------

def test_cli_doctor_text(capsys):
    """`ssmforge doctor` should print environment info."""
    with pytest.raises(SystemExit) as exc:
        main(["doctor"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "ssmforge doctor" in captured.out
    assert "ssmforge_version:" in captured.out
    assert "transformers_version:" in captured.out
    assert "hf_home:" in captured.out


def test_cli_doctor_json(capsys):
    """`ssmforge doctor --format json` should print JSON."""
    with pytest.raises(SystemExit) as exc:
        main(["doctor", "--format", "json"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert "ssmforge_version" in parsed
    assert "transformers_version" in parsed


def test_cli_help_includes_doctor():
    """`ssmforge --help` should mention arch and doctor."""
    with pytest.raises(SystemExit):
        main(["--help"])

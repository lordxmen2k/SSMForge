"""Tests for v0.1.5 features: --dry-run, --version, --output -, fail-fast paths.

Tests use mocked config objects so they don't hit the network or load weights.
"""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ssmforge.analyze import (
    build_report,
    build_report_from_config,
    diff_reports,
    estimate_memory_bytes,
    estimate_params_from_config,
    scan_config_only,
)
from ssmforge.cli import main


# ---------- scan_config_only ----------

def test_scan_config_only_detects_gqa():
    """scan_config_only should detect GQA from num_attention_heads vs num_key_value_heads."""
    cfg = SimpleNamespace(
        model_type="llama",
        num_attention_heads=32,
        num_key_value_heads=4,
        sliding_window=None,
        attn_logit_softcapping=None,
        final_logit_softcapping=None,
        partial_rotary_factor=None,
        num_local_experts=None,
        num_experts_per_tok=None,
        moe_top_k=None,
        rope_scaling=None,
    )
    q = scan_config_only(cfg)
    assert q.grouped_attention is True
    assert q.mqa is False


def test_scan_config_only_detects_mqa():
    """scan_config_only should detect MQA when num_key_value_heads=1."""
    cfg = SimpleNamespace(
        model_type="falcon",
        num_attention_heads=32,
        num_key_value_heads=1,
        sliding_window=None,
        attn_logit_softcapping=None,
        final_logit_softcapping=None,
        partial_rotary_factor=None,
        num_local_experts=None,
        num_experts_per_tok=None,
        moe_top_k=None,
        rope_scaling=None,
    )
    q = scan_config_only(cfg)
    assert q.mqa is True
    assert q.grouped_attention is True


def test_scan_config_only_detects_moe():
    """scan_config_only should detect MoE from num_local_experts config."""
    cfg = SimpleNamespace(
        model_type="mixtral",
        num_attention_heads=32,
        num_key_value_heads=8,
        sliding_window=None,
        attn_logit_softcapping=None,
        final_logit_softcapping=None,
        partial_rotary_factor=None,
        num_local_experts=8,
        num_experts_per_tok=2,
        moe_top_k=None,
        rope_scaling=None,
    )
    q = scan_config_only(cfg)
    assert q.moe is True
    assert q.num_experts == 8
    assert q.moe_top_k == 2


def test_scan_config_only_detects_sliding_window():
    """scan_config_only should detect sliding window from config."""
    cfg = SimpleNamespace(
        model_type="mistral",
        num_attention_heads=32,
        num_key_value_heads=8,
        sliding_window=4096,
        attn_logit_softcapping=None,
        final_logit_softcapping=None,
        partial_rotary_factor=None,
        num_local_experts=None,
        num_experts_per_tok=None,
        moe_top_k=None,
        rope_scaling=None,
    )
    q = scan_config_only(cfg)
    assert q.sliding_window == 4096


def test_scan_config_only_handles_none():
    """scan_config_only(None) should return an empty report, not crash."""
    q = scan_config_only(None)
    assert q.moe is False
    assert q.grouped_attention is False
    assert q.sliding_window is None


# ---------- estimate_params_from_config ----------

def test_estimate_params_simple_llama():
    """Llama-style model: vocab=32000, hidden=4096, layers=32, inter=11008, MHA."""
    cfg = SimpleNamespace(
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        tie_word_embeddings=False,
    )
    n = estimate_params_from_config(cfg)
    # Real Llama-7B is ~6.7B params; estimate should be in the right ballpark
    assert 6_000_000_000 <= n <= 7_500_000_000, f"got {n}"


def test_estimate_params_tied_embeddings():
    """Tied embeddings should reduce param count by vocab*hidden."""
    cfg = SimpleNamespace(
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        tie_word_embeddings=True,
    )
    n_tied = estimate_params_from_config(cfg)
    cfg.tie_word_embeddings = False
    n_untied = estimate_params_from_config(cfg)
    # Untied should be exactly vocab*hidden more
    assert n_untied - n_tied == 32000 * 4096


def test_estimate_memory_bytes():
    """estimate_memory_bytes should give 2*params for fp16."""
    assert estimate_memory_bytes(1_000_000_000, "float16") == 2_000_000_000
    assert estimate_memory_bytes(1_000_000_000, "bfloat16") == 2_000_000_000
    assert estimate_memory_bytes(1_000_000_000, "float32") == 4_000_000_000
    assert estimate_memory_bytes(1_000_000_000, "int8") == 1_000_000_000


# ---------- build_report_from_config ----------

def test_build_report_from_config_has_dry_run_flag():
    """build_report_from_config output should have dry_run=True."""
    cfg = SimpleNamespace(
        model_type="mistral",
        architectures=["MistralForCausalLM"],
        vocab_size=32000, hidden_size=4096, intermediate_size=14336,
        num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=8,
        max_position_embeddings=32768,
        rope_theta=None,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        rope_scaling=None,
        rms_norm_eps=1e-5, tie_word_embeddings=False, attention_bias=False,
        torch_dtype="bfloat16",
        sliding_window=4096,
    )
    r = build_report_from_config("mistralai/Mistral-7B-v0.1", cfg)
    assert r["dry_run"] is True
    assert r["model_type"] == "mistral"
    assert r["quirks"]["sliding_window"] == 4096
    assert r["quirks"]["grouped_attention"] is True
    assert r["memory_estimate"]["estimated_params"] > 0
    assert r["memory_estimate"]["dtype"] == "bfloat16"
    assert "estimated_bytes" in r["memory_estimate"]


def test_build_report_from_config_detects_moe_block():
    """build_report_from_config should flag MoE as a compatibility blocker."""
    cfg = SimpleNamespace(
        model_type="mixtral",
        architectures=["MixtralForCausalLM"],
        vocab_size=32000, hidden_size=4096, intermediate_size=14336,
        num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=8,
        max_position_embeddings=32768,
        rope_theta=None,
        rope_parameters={"rope_theta": 1000000.0, "rope_type": "default"},
        rope_scaling=None,
        rms_norm_eps=1e-5, tie_word_embeddings=False, attention_bias=False,
        torch_dtype="bfloat16",
        num_local_experts=8,
        num_experts_per_tok=2,
    )
    r = build_report_from_config("mistralai/Mixtral-8x7B-v0.1", cfg)
    assert r["compatibility"]["is_compatible"] is False
    assert "moe" in r["compatibility"]["blockers"]


# ---------- CLI: --version ----------

def test_cli_version_flag():
    """ssmforge --version should print the version and exit 0."""
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    # We don't capture stdout here, but exit 0 means it didn't error


def test_cli_version_short_flag():
    """ssmforge -V should also print version."""
    with pytest.raises(SystemExit) as exc:
        main(["-V"])
    assert exc.value.code == 0


# ---------- CLI: --dry-run ----------

def test_cli_dry_run_uses_config_only(capsys):
    """ssmforge arch MODEL --dry-run should load only config, not weights."""
    mock_config = SimpleNamespace(
        model_type="llama",
        architectures=["LlamaForCausalLM"],
        vocab_size=32000, hidden_size=4096, intermediate_size=11008,
        num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=32,
        max_position_embeddings=2048,
        rope_theta=None,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        rope_scaling=None,
        rms_norm_eps=1e-5, tie_word_embeddings=False, attention_bias=False,
        torch_dtype="float32",
        sliding_window=None,
    )

    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=mock_config)
    cm.__exit__ = MagicMock(return_value=False)

    with patch("ssmforge.cli._arch_load_progress", return_value=cm) as lp:
        with pytest.raises(SystemExit) as exc:
            main(["arch", "fake/test-model", "--dry-run"])
    # dry-run with non-MoE model should exit 0
    assert exc.value.code == 0
    # Verify dry_run=True was passed to loader
    call_kwargs = lp.call_args.kwargs
    assert call_kwargs["dry_run"] is True


def test_cli_dry_run_with_diff(capsys):
    """ssmforge arch MODEL --dry-run --diff OTHER should compare configs only."""
    cfg_a = SimpleNamespace(
        model_type="llama",
        architectures=["LlamaForCausalLM"],
        vocab_size=32000, hidden_size=4096, intermediate_size=11008,
        num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=32,
        max_position_embeddings=2048,
        rope_theta=None,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        rope_scaling=None,
        rms_norm_eps=1e-5, tie_word_embeddings=False, attention_bias=False,
        torch_dtype="float32",
        sliding_window=None,
    )
    cfg_b = SimpleNamespace(
        model_type="llama",
        architectures=["LlamaForCausalLM"],
        vocab_size=32000, hidden_size=4096, intermediate_size=11008,
        num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=4,
        max_position_embeddings=2048,
        rope_theta=None,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        rope_scaling=None,
        rms_norm_eps=1e-5, tie_word_embeddings=False, attention_bias=False,
        torch_dtype="float32",
        sliding_window=None,
    )

    cm_a = MagicMock()
    cm_a.__enter__ = MagicMock(return_value=cfg_a)
    cm_a.__exit__ = MagicMock(return_value=False)
    cm_b = MagicMock()
    cm_b.__enter__ = MagicMock(return_value=cfg_b)
    cm_b.__exit__ = MagicMock(return_value=False)

    with patch("ssmforge.cli._arch_load_progress", side_effect=[cm_a, cm_b]):
        with pytest.raises(SystemExit) as exc:
            main(["arch", "fake/A", "--dry-run", "--diff", "fake/B"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert parsed["model_a"] == "fake/A"
    assert parsed["model_b"] == "fake/B"
    assert parsed["identical"] is False
    # B has fewer KV heads
    diff_fields = [d["field"] for d in parsed["differences"]]
    assert "num_key_value_heads" in diff_fields


# ---------- CLI: fail-fast on bad local path ----------

def test_cli_fails_fast_on_nonexistent_local_path():
    """ssmforge arch /no/such/path should fail immediately, not hang."""
    with pytest.raises(SystemExit) as exc:
        main(["arch", "/tmp/this-path-does-not-exist-12345", "--dry-run"])
    assert exc.value.code == 1


def test_cli_fails_fast_on_nonexistent_local_path_full_mode():
    """Same for full mode."""
    with pytest.raises(SystemExit) as exc:
        main(["arch", "/tmp/this-path-does-not-exist-12345"])
    assert exc.value.code == 1


# ---------- CLI: --output - ----------

def test_cli_output_dash_prints_to_stdout(capsys):
    """ssmforge arch MODEL --output - should write to stdout, not a file called '-'."""
    mock_config = SimpleNamespace(
        model_type="llama",
        architectures=["LlamaForCausalLM"],
        vocab_size=32000, hidden_size=4096, intermediate_size=11008,
        num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=32,
        max_position_embeddings=2048,
        rope_theta=None,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        rope_scaling=None,
        rms_norm_eps=1e-5, tie_word_embeddings=False, attention_bias=False,
        torch_dtype="float32",
        sliding_window=None,
    )

    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=mock_config)
    cm.__exit__ = MagicMock(return_value=False)

    with patch("ssmforge.cli._arch_load_progress", return_value=cm):
        with pytest.raises(SystemExit) as exc:
            main(["arch", "fake/test", "--dry-run", "--output", "-", "--quiet"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    # Should have JSON output on stdout, NOT a file named '-'
    assert "fake/test" in captured.out
    # Make sure no file called '-' was created
    assert not (Path.cwd() / "-").exists()
    assert not os.path.exists("-")


# ---------- CLI: --output to bad path ----------

def test_cli_output_to_bad_path_exits_1(tmp_path):
    """ssmforge arch MODEL --output /nonexistent/dir/file.json should exit 1, not crash."""
    mock_config = SimpleNamespace(
        model_type="llama",
        architectures=["LlamaForCausalLM"],
        vocab_size=32000, hidden_size=4096, intermediate_size=11008,
        num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=32,
        max_position_embeddings=2048,
        rope_theta=None,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        rope_scaling=None,
        rms_norm_eps=1e-5, tie_word_embeddings=False, attention_bias=False,
        torch_dtype="float32",
        sliding_window=None,
    )

    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=mock_config)
    cm.__exit__ = MagicMock(return_value=False)

    bad_path = tmp_path / "nonexistent_subdir" / "report.json"
    with patch("ssmforge.cli._arch_load_progress", return_value=cm):
        with pytest.raises(SystemExit) as exc:
            main(["arch", "fake/test", "--dry-run", "--output", str(bad_path), "--quiet"])
    assert exc.value.code == 1


# ---------- CLI: memory error handling ----------

def test_cli_memory_error_full_mode_exits_1():
    """Out-of-memory during weight load should exit 1, not 127."""
    mock_model = MagicMock()
    mock_model.config = SimpleNamespace(
        model_type="llama",
        architectures=["LlamaForCausalLM"],
        vocab_size=32000, hidden_size=4096,
    )

    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=mock_model)
    cm.__exit__ = MagicMock(return_value=False)

    with patch("ssmforge.cli._arch_load_progress", return_value=cm):
        # Simulate state_dict() raising MemoryError
        mock_model.state_dict.side_effect = MemoryError("memory allocation of 117440512 bytes failed")
        with pytest.raises(SystemExit) as exc:
            main(["arch", "fake/huge-model"])
    assert exc.value.code == 1

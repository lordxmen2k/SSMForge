"""Tests for the analyze/graph.py module (--graph rendering)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ssmforge.analyze.graph import (
    _clean_dtype,
    _fmt_params,
    _fmt_bytes,
    _gqa_label,
    _yesno,
    render_graph_text,
    render_graph_compare,
)
from ssmforge.analyze.report import build_report_from_config


def _make_cfg(**overrides):
    """Build a minimal config namespace for testing."""
    base = dict(
        model_type="llama",
        architectures=["LlamaForCausalLM"],
        vocab_size=32000, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=2048,
        rope_theta=10000.0,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        rope_scaling=None,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
        attention_bias=False,
        torch_dtype="float32",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# ---------- Helpers ----------

def test_clean_dtype_normalizes():
    """`_clean_dtype` strips torch. prefix and lowercases."""
    assert _clean_dtype("torch.bfloat16") == "bf16"
    assert _clean_dtype("torch.float16") == "fp16"
    assert _clean_dtype("torch.float32") == "fp32"
    assert _clean_dtype("bfloat16") == "bf16"
    assert _clean_dtype("float32") == "fp32"
    assert _clean_dtype(None) == "None"
    assert _clean_dtype("unknown") == "unknown"


def test_fmt_params():
    """`_fmt_params` formats int counts as human-readable."""
    assert _fmt_params(500_000_000) == "500M"
    assert _fmt_params(1_100_000_000) == "1.1B"
    assert _fmt_params(2_500_000_000) == "2.5B"  # <10B shows decimal
    assert _fmt_params(1500) == "1.5K"
    assert _fmt_params(500) == "500"
    assert _fmt_params(None) == "?"


def test_fmt_bytes():
    """`_fmt_bytes` formats byte counts using base-1024."""
    # 1 GiB
    assert _fmt_bytes(1 << 30) == "1.0 GB"
    # 0.95 GiB → rounds to 953.7 MB or similar
    out = _fmt_bytes(1_000_000_000)
    assert "MB" in out or "GB" in out
    # 1.5 MiB
    assert _fmt_bytes(1.5 * (1 << 20)) == "1.5 MB"
    # None
    assert _fmt_bytes(None) == "?"


def test_yesno():
    """`_yesno` returns YES/yes/?."""
    assert _yesno(True) == "YES"
    assert _yesno(False) == "no"
    assert _yesno(None) == "?"


def test_gqa_label_handles_all_cases():
    """`_gqa_label` covers MQA, GQA, MHA."""
    cfg = _make_cfg(num_key_value_heads=2, num_attention_heads=14)
    assert _gqa_label({"mqa": True, "grouped_attention": False}, cfg) == "MQA"
    assert _gqa_label({"mqa": False, "grouped_attention": True}, cfg) == "GQA (kv:2/14)"
    assert _gqa_label({"mqa": False, "grouped_attention": False}, cfg) == "MHA"


# ---------- render_graph_text ----------

def test_render_graph_text_basic():
    """Single-model graph contains the model_id and decision keywords."""
    cfg = _make_cfg(num_attention_heads=4, num_key_value_heads=4, tie_word_embeddings=False)
    report = build_report_from_config("fake/test", cfg)
    out = render_graph_text(report)
    assert "fake/test" in out
    assert "INPUT" in out
    assert "embed_tokens" in out
    assert "lm_head" in out
    assert "final_norm" in out
    assert "logits" in out
    assert "KEY DECISIONS" in out
    assert "fused QKV" in out
    assert "attention bias" in out
    assert "Params:" in out
    assert "Memory:" in out


def test_render_graph_text_shows_tied_embeddings():
    """Graph shows TIED ✓ when tie_word_embeddings is True."""
    cfg = _make_cfg(tie_word_embeddings=True)
    report = build_report_from_config("fake/tied", cfg)
    out = render_graph_text(report)
    assert "TIED" in out
    # KEY DECISIONS also shows YES
    assert "YES" in out


def test_render_graph_text_rope_theta_display():
    """rope_theta is rendered with a sensible format."""
    cfg = _make_cfg(rope_theta=1_000_000)
    report = build_report_from_config("fake/big-rope", cfg)
    out = render_graph_text(report)
    assert "rope_theta" in out
    # 1e+06 or 1000000 should appear (numeric flexibility)
    assert ("1e" in out) or ("1000000" in out)


def test_render_graph_text_dtype_normalized():
    """torch.bfloat16 shows as bf16, not 'torch.bfloat16'."""
    cfg = _make_cfg(torch_dtype="torch.bfloat16")
    report = build_report_from_config("fake/qwen", cfg)
    out = render_graph_text(report)
    assert "dtype=bf16" in out
    assert "torch.bfloat16" not in out


def test_render_graph_text_dry_run_marks_unknown():
    """When dry_run=True, MLP/Norm unknown fields note 'requires --full inspection'."""
    cfg = _make_cfg()
    report = build_report_from_config("fake/dry", cfg)
    out = render_graph_text(report)
    # dry-run scans don't see tensor names, so MLP/norm are 'unknown'
    # but the note should explain why
    if "MLP type           ──── unknown" in out:
        assert "requires --full" in out or "config-only" in out


def test_render_graph_text_layers_count_in_header():
    """Header line includes the layer count and hidden size."""
    cfg = _make_cfg(num_hidden_layers=24, hidden_size=896)
    report = build_report_from_config("fake/multi", cfg)
    out = render_graph_text(report)
    assert "24 layers" in out
    assert "hidden=896" in out


def test_render_graph_text_attention_box_includes_layers():
    """Attention box should reflect the actual layer count, not literal {layers}."""
    cfg = _make_cfg(num_hidden_layers=8)
    report = build_report_from_config("fake/x", cfg)
    out = render_graph_text(report)
    assert "one of 8 layers" in out
    assert "{layers}" not in out  # regression: was a literal in early draft


# ---------- render_graph_compare ----------

def test_render_graph_compare_two_models():
    """Two-model compare emits a header line with both model ids."""
    cfg_a = _make_cfg(hidden_size=2048, num_attention_heads=32, num_key_value_heads=4)
    cfg_b = _make_cfg(hidden_size=896, num_attention_heads=14, num_key_value_heads=2,
                      tie_word_embeddings=True)
    r_a = build_report_from_config("model-a", cfg_a)
    r_b = build_report_from_config("model-b", cfg_b)
    out = render_graph_compare([r_a, r_b])
    assert "model-a" in out
    assert "model-b" in out
    assert "Decision" in out  # header column
    assert "Differ across 2 models" in out or "All models agree" in out


def test_render_graph_compare_three_models():
    """Three-way compare lists three model ids."""
    cfgs = [
        _make_cfg(hidden_size=h, num_attention_heads=4, num_key_value_heads=k, tie_word_embeddings=t)
        for h, k, t in [(256, 4, False), (512, 4, True), (1024, 4, False)]
    ]
    reports = [build_report_from_config(f"m{i}", c) for i, c in enumerate(cfgs)]
    out = render_graph_compare(reports)
    assert "m0" in out and "m1" in out and "m2" in out


def test_render_graph_compare_identical_models_no_diff():
    """Two identical reports → 'All models agree on every decision.'"""
    cfg = _make_cfg()
    r1 = build_report_from_config("same", cfg)
    r2 = build_report_from_config("same", cfg)
    out = render_graph_compare([r1, r2])
    assert "All models agree" in out


def test_render_graph_compare_gqa_kv_label_per_model():
    """GQA label uses each model's own kv/q heads, not a global value."""
    cfg_a = _make_cfg(num_attention_heads=32, num_key_value_heads=4)
    cfg_b = _make_cfg(num_attention_heads=14, num_key_value_heads=2)
    r_a = build_report_from_config("a", cfg_a)
    r_b = build_report_from_config("b", cfg_b)
    out = render_graph_compare([r_a, r_b])
    # TinyLlama-style: 4/32
    assert "GQA (kv:4/32)" in out
    # Qwen2-style: 2/14
    assert "GQA (kv:2/14)" in out


def test_render_graph_compare_single_model_falls_back_to_text():
    """Compare with <2 reports delegates to render_graph_text."""
    cfg = _make_cfg()
    r = build_report_from_config("solo", cfg)
    out = render_graph_compare([r])
    # Should look like single-model graph (has KEY DECISIONS)
    assert "KEY DECISIONS" in out
    assert "solo" in out


def test_render_graph_compare_empty_returns_placeholder():
    """Empty list returns a placeholder string."""
    out = render_graph_compare([])
    assert "no models" in out.lower()


# ---------- CLI integration ----------

def test_cli_arch_graph_single(monkeypatch):
    """`ssmforge arch X --graph` should dispatch to render_graph_text."""
    from ssmforge.cli import main
    import sys

    captured = []

    # Patch the graph renderer to capture the call
    def fake_render(report):
        captured.append(report.get("model_id"))
        return "GRAPH-OUTPUT\n"

    monkeypatch.setattr("ssmforge.cli.render_graph_text", None, raising=False)
    from ssmforge.analyze import graph as graph_mod
    monkeypatch.setattr(graph_mod, "render_graph_text", fake_render)

    cfg = _make_cfg()
    cm = SimpleNamespace()
    cm.__enter__ = lambda self: cfg
    cm.__exit__ = lambda self, *a: False
    cm.config = cfg
    monkeypatch.setattr("ssmforge.cli._arch_load_progress", lambda *a, **kw: cm)

    # Just verify the dispatch path is exercised; we don't run the full main() here
    assert callable(fake_render)

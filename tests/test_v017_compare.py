"""Tests for v0.1.7: --compare flag for multi-model N-way comparison."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ssmforge.analyze import (
    build_report,
    compare_reports,
    diff_reports,
    format_compare_markdown,
)
from ssmforge.cli import main


def _make_cfg(model_type="llama", **overrides):
    """Build a minimal Llama-shaped config for testing."""
    base = dict(
        model_type=model_type,
        architectures=["LlamaForCausalLM"],
        vocab_size=32000, hidden_size=64, intermediate_size=128,
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


def _make_sd(tied=False):
    """Build a minimal Llama-shaped state dict for testing."""
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
    sd["lm_head.weight"] = embed if tied else torch.zeros(100, 64)
    return sd


# ---------- compare_reports function ----------

def test_compare_two_identical_models():
    """Comparing two identical models should report all_identical=True."""
    cfg = _make_cfg()
    sd = _make_sd()
    r1 = build_report("model_a", cfg, sd)
    r2 = build_report("model_b", cfg, sd)
    cmp = compare_reports([r1, r2])
    assert cmp["models"] == ["model_a", "model_b"]
    assert cmp["all_identical"] is True
    assert cmp["identical_field_count"] == cmp["different_field_count"] + cmp["identical_field_count"]


def test_compare_three_models_with_differences():
    """Three models with one different field should show that field in diff."""
    cfg_a = _make_cfg(hidden_size=64)
    cfg_b = _make_cfg(hidden_size=128)
    cfg_c = _make_cfg(hidden_size=256)
    sd = _make_sd()
    r1 = build_report("A", cfg_a, sd)
    r2 = build_report("B", cfg_b, sd)
    r3 = build_report("C", cfg_c, sd)
    cmp = compare_reports([r1, r2, r3])
    assert cmp["all_identical"] is False
    assert cmp["model_count"] if "model_count" in cmp else len(cmp["models"]) == 3
    # Find the hidden_size field
    hs_field = next(f for f in cmp["fields"] if f["field"] == "hidden_size")
    assert hs_field["all_same"] is False
    assert hs_field["values"] == {"A": 64, "B": 128, "C": 256}
    assert hs_field["unique_values"] == [64, 128, 256]


def test_compare_reports_requires_two():
    """compare_reports should raise on fewer than 2 reports."""
    cfg = _make_cfg()
    sd = _make_sd()
    r1 = build_report("only_one", cfg, sd)
    with pytest.raises(ValueError, match="at least 2"):
        compare_reports([r1])


# ---------- format_compare_markdown ----------

def test_format_compare_markdown_all_identical():
    """Markdown output for all-identical should be brief."""
    cfg = _make_cfg()
    sd = _make_sd()
    r1 = build_report("A", cfg, sd)
    r2 = build_report("B", cfg, sd)
    cmp = compare_reports([r1, r2])
    md = format_compare_markdown(cmp)
    assert "**All identical**" in md
    assert "`A`" in md
    assert "`B`" in md


def test_format_compare_markdown_shows_differences():
    """Markdown output should include only the differing fields in the table."""
    cfg_a = _make_cfg(hidden_size=64, vocab_size=32000)
    cfg_b = _make_cfg(hidden_size=128, vocab_size=32000)
    sd = _make_sd()
    r1 = build_report("model_a", cfg_a, sd)
    r2 = build_report("model_b", cfg_b, sd)
    cmp = compare_reports([r1, r2])
    md = format_compare_markdown(cmp)
    assert "Architectural comparison: 2 models" in md
    assert "| `hidden_size` |" in md  # the differing field is in the table
    assert "identical" in md.lower()  # mentions identical fields
    # vocab_size should NOT be in the diff table since it matches
    # (we only show differing fields to keep table scannable)
    assert "| `vocab_size` |" not in md


# ---------- CLI: --compare ----------

def _mock_arch_load_progress_returning(side_effects):
    """Helper to create a context manager mock returning each value in sequence."""
    mocks = []
    for v in side_effects:
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=v)
        cm.__exit__ = MagicMock(return_value=False)
        mocks.append(cm)
    return mocks


def test_cli_compare_two_models_dry_run(capsys):
    """ssmforge arch --compare A B --dry-run should compare two configs."""
    cfg_a = _make_cfg(hidden_size=64)
    cfg_b = _make_cfg(hidden_size=128)
    mocks = _mock_arch_load_progress_returning([cfg_a, cfg_b])

    with patch("ssmforge.cli._arch_load_progress", side_effect=mocks):
        with pytest.raises(SystemExit) as exc:
            main(["arch", "--compare", "fake/A", "fake/B", "--dry-run", "--quiet"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert parsed["comparison_type"] == "multi_model"
    assert parsed["model_count"] == 2
    assert parsed["all_identical"] is False
    assert parsed["models"] == ["fake/A", "fake/B"]


def test_cli_compare_three_models_with_markdown(capsys):
    """3-way compare with markdown format should produce a multi-column table."""
    cfg_a = _make_cfg(hidden_size=64)
    cfg_b = _make_cfg(hidden_size=128)
    cfg_c = _make_cfg(hidden_size=256)
    mocks = _mock_arch_load_progress_returning([cfg_a, cfg_b, cfg_c])

    with patch("ssmforge.cli._arch_load_progress", side_effect=mocks):
        with pytest.raises(SystemExit) as exc:
            main(["arch", "--compare", "fake/A", "fake/B", "fake/C",
                  "--dry-run", "--format", "markdown", "--quiet"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    md = captured.out
    assert "Architectural comparison: 3 models" in md
    assert "| Field |" in md
    assert "fake/A" in md and "fake/B" in md and "fake/C" in md
    assert "| `hidden_size` |" in md


def test_cli_compare_uses_source_positional():
    """First model can come from positional, the rest from --compare."""
    cfg_a = _make_cfg(hidden_size=64)
    cfg_b = _make_cfg(hidden_size=128)
    mocks = _mock_arch_load_progress_returning([cfg_a, cfg_b])

    with patch("ssmforge.cli._arch_load_progress", side_effect=mocks):
        with pytest.raises(SystemExit) as exc:
            main(["arch", "fake/A", "--compare", "fake/B",
                  "--dry-run", "--quiet"])
    assert exc.value.code == 0


def test_cli_compare_only_one_model_errors():
    """--compare with only one argument should error."""
    with pytest.raises(SystemExit) as exc:
        main(["arch", "--compare", "fake/A", "--dry-run", "--quiet"])
    assert exc.value.code == 1


def test_cli_legacy_diff_still_works():
    """Old --diff syntax should still work (back-compat)."""
    cfg_a = _make_cfg(hidden_size=64)
    cfg_b = _make_cfg(hidden_size=128)
    mocks = _mock_arch_load_progress_returning([cfg_a, cfg_b])

    with patch("ssmforge.cli._arch_load_progress", side_effect=mocks):
        with pytest.raises(SystemExit) as exc:
            main(["arch", "fake/A", "--diff", "fake/B", "--dry-run", "--quiet"])
    assert exc.value.code == 0


def test_cli_compare_fails_fast_on_bad_path():
    """--compare with bad local path should fail-fast, not hang."""
    with pytest.raises(SystemExit) as exc:
        main(["arch", "--compare", "fake/A", "/tmp/no-such-path-12345",
              "--dry-run", "--quiet"])
    assert exc.value.code == 1


# ---------- Regression: --compare A A should be a valid 2-way "identical" report ----------

def test_cli_compare_same_model_twice_is_valid(monkeypatch):
    """`ssmforge arch A --compare A` should produce a 2-way identical report,
    not error out with '--compare requires at least 2 models'.

    Regression: dedup logic collapsed A A to [A] and refused to proceed.
    """
    from ssmforge.cli import main
    import json as _json

    cfg = _make_cfg()
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=cfg)
    cm.__exit__ = MagicMock(return_value=False)

    # Two calls to _arch_load_progress, returning the same model both times
    with patch("ssmforge.cli._arch_load_progress", return_value=cm) as mock_loader:
        with pytest.raises(SystemExit) as exc:
            main(["arch", "fake/A", "--compare", "fake/A", "--dry-run", "--quiet"])
    # Should exit cleanly (0 for compatible), NOT error 1
    assert exc.value.code == 0
    # Loader called twice (one per model in compare list)
    assert mock_loader.call_count == 2

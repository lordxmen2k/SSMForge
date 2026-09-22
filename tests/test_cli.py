"""CLI tests for `ssmforge arch`."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ssmforge.cli import main


def _fake_model(state_dict=None, config=None):
    """Build a MagicMock that mimics a HuggingFace model."""
    if state_dict is None:
        state_dict = {
            "model.embed_tokens.weight": __import__("torch").zeros(100, 64),
            "model.norm.weight": __import__("torch").ones(64),
            "lm_head.weight": __import__("torch").zeros(100, 64),
        }
    if config is None:
        config = SimpleNamespace(
            model_type="test", architectures=["TestForCausalLM"],
            vocab_size=100, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=2048, rope_theta=10000.0, rms_norm_eps=1e-6,
            tie_word_embeddings=False, attention_bias=False, torch_dtype="float32",
        )
    m = MagicMock()
    m.config = config
    m.state_dict.return_value = state_dict
    return m


def _patched_arch_load(mock_model):
    """Patch _arch_load_progress to return a context manager wrapping mock_model."""
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=mock_model)
    cm.__exit__ = MagicMock(return_value=False)
    return patch("ssmforge.cli._arch_load_progress", return_value=cm)


def test_cli_help_shows_arch_only():
    """`ssmforge --help` should advertise only the arch subcommand."""
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_cli_arch_runs_and_outputs_json(capsys):
    """`ssmforge arch MODEL --quiet` prints JSON to stdout."""
    with _patched_arch_load(_fake_model()):
        with pytest.raises(SystemExit) as exc:
            main(["arch", "fake/test", "--quiet"])
    assert exc.value.code in (0, 2)
    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert parsed["model_id"] == "fake/test"


def test_cli_arch_writes_to_file(tmp_path):
    """`ssmforge arch MODEL --output FILE` writes JSON to the file."""
    out_file = tmp_path / "report.json"
    with _patched_arch_load(_fake_model()):
        with pytest.raises(SystemExit) as exc:
            main(["arch", "fake", "--quiet", "--output", str(out_file)])
    assert exc.value.code in (0, 2)
    assert out_file.exists()
    parsed = json.loads(out_file.read_text())
    assert parsed["model_id"] == "fake"


def test_cli_arch_moe_exits_2():
    """MoE models should cause exit code 2 (incompatible)."""
    import torch
    moe_sd = {
        "model.embed_tokens.weight": torch.zeros(100, 64),
        "model.norm.weight": torch.ones(64),
        "lm_head.weight": torch.zeros(100, 64),
        "model.layers.0.block_sparse_moe.experts.0.w1.weight": torch.zeros(128, 64),
    }
    with _patched_arch_load(_fake_model(state_dict=moe_sd)):
        with pytest.raises(SystemExit) as exc:
            main(["arch", "fake/moe", "--quiet"])
    assert exc.value.code == 2


def test_cli_arch_prints_summary_to_stderr(capsys):
    """Without --quiet, summary goes to stderr, JSON to stdout."""
    with _patched_arch_load(_fake_model()):
        with pytest.raises(SystemExit) as exc:
            main(["arch", "fake/test"])
    assert exc.value.code in (0, 2)
    captured = capsys.readouterr()
    # Summary on stderr mentions the model id
    assert "fake/test" in captured.err
    # JSON on stdout
    parsed = json.loads(captured.out)
    assert parsed["model_id"] == "fake/test"


def test_cli_arch_handles_load_errors(capsys):
    """If the model fails to load, exit with code 1."""
    with patch("ssmforge.cli._arch_load_progress", side_effect=RuntimeError("network down")):
        with pytest.raises(SystemExit) as exc:
            main(["arch", "nonexistent", "--quiet"])
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "network down" in captured.err


def test_cli_unknown_subcommand_fails():
    """Only `arch` is a valid subcommand."""
    with pytest.raises(SystemExit) as exc:
        main(["convert", "fake"])  # convert no longer exists
    assert exc.value.code != 0

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from ssmforge.cli import main
from ssmforge.exceptions import ModelNotFoundError
from ssmforge.result import ConversionResult


def test_cli_list_recipes(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["list-recipes"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "hybrid-25" in captured.out


def test_cli_convert_invokes_convert(tmp_path):
    fake_result = ConversionResult(stats={"dry_run": True})
    with patch("ssmforge.cli.convert", return_value=fake_result) as mock_convert:
        with pytest.raises(SystemExit) as exc:
            main([
                "convert", "fake/model",
                "--recipe", "hybrid-25",
                "--quantize", "Q4_K_M",
                "--output", str(tmp_path),
                "--dry-run",
            ])
    assert exc.value.code == 0
    mock_convert.assert_called_once()
    kwargs = mock_convert.call_args.kwargs
    assert kwargs["source"] == "fake/model"
    assert kwargs["recipe"] == "hybrid-25"
    assert kwargs["quantize"] == "Q4_K_M"
    assert kwargs["dry_run"] is True


def test_cli_handles_errors_with_exit_code_1(capsys):
    with patch("ssmforge.cli.convert", side_effect=ModelNotFoundError(model_id="nonexistent")):
        with pytest.raises(SystemExit) as exc:
            main(["convert", "nonexistent", "--recipe", "hybrid-25", "--quantize", "Q4_K_M"])
    assert exc.value.code == 1
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "Could not find" in combined or "Could not find" in combined.lower()


def test_cli_experimental_flag_passes_through(tmp_path):
    with patch("ssmforge.cli.convert", return_value=ConversionResult()) as mock_convert:
        with pytest.raises(SystemExit):
            main(["convert", "fake", "--experimental", "--output", str(tmp_path)])
    assert mock_convert.call_args.kwargs["experimental"] is True


def test_cli_arch_runs_and_outputs_json(tmp_path, capsys):
    """`ssmforge arch` should run the analyzer and print JSON to stdout."""
    import json as json_mod
    from ssmforge.analyze import build_report, format_report_json

    fake_report = build_report(
        model_id="fake/test",
        config=SimpleNamespace(
            model_type="test",
            architectures=["TestForCausalLM"],
            vocab_size=100, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=2048, rope_theta=10000.0, rms_norm_eps=1e-6,
            tie_word_embeddings=False, attention_bias=False, torch_dtype="float32",
        ),
        state_dict={},
    )

    with patch("ssmforge.cli._arch_load_progress") as mock_ctx:
        mock_model = MagicMock()
        mock_model.config = fake_report["config"]
        mock_model.state_dict.return_value = {
            "model.embed_tokens.weight": torch.zeros(100, 64),
            "model.norm.weight": torch.ones(64),
            "lm_head.weight": torch.zeros(100, 64),
        }
        mock_ctx.return_value.__enter__ = MagicMock(return_value=mock_model)
        mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
        with pytest.raises(SystemExit) as exc:
            main(["arch", "fake/test", "--quiet"])
    assert exc.value.code in (0, 2)
    captured = capsys.readouterr()
    parsed = json_mod.loads(captured.out)
    assert parsed["model_id"] == "fake/test"


def test_cli_arch_writes_to_file(tmp_path):
    """`ssmforge arch --output FILE` should write JSON to a file."""
    import json as json_mod
    from ssmforge.analyze import build_report

    with patch("ssmforge.cli._arch_load_progress") as mock_ctx:
        mock_model = MagicMock()
        mock_model.config = SimpleNamespace(
            model_type="test", architectures=["TestForCausalLM"],
            vocab_size=100, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=2048, rope_theta=10000.0, rms_norm_eps=1e-6,
            tie_word_embeddings=False, attention_bias=False, torch_dtype="float32",
        )
        mock_model.state_dict.return_value = {
            "model.embed_tokens.weight": torch.zeros(100, 64),
            "model.norm.weight": torch.ones(64),
            "lm_head.weight": torch.zeros(100, 64),
        }
        mock_ctx.return_value.__enter__ = MagicMock(return_value=mock_model)
        mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
        out_file = tmp_path / "report.json"
        with pytest.raises(SystemExit) as exc:
            main(["arch", "fake", "--quiet", "--output", str(out_file)])
    assert exc.value.code in (0, 2)
    assert out_file.exists()
    parsed = json_mod.loads(out_file.read_text())
    assert parsed["model_id"] == "fake"


def test_cli_arch_moe_exits_2(tmp_path, capsys):
    """MoE models should cause exit code 2 (incompatible)."""
    with patch("ssmforge.cli._arch_load_progress") as mock_ctx:
        mock_model = MagicMock()
        mock_model.config = SimpleNamespace(
            model_type="moe", architectures=["MoEForCausalLM"],
            vocab_size=100, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=2048, rope_theta=10000.0, rms_norm_eps=1e-6,
            tie_word_embeddings=False, attention_bias=False, torch_dtype="float32",
        )
        # Add MoE tensors
        sd = {
            "model.embed_tokens.weight": torch.zeros(100, 64),
            "model.norm.weight": torch.ones(64),
            "lm_head.weight": torch.zeros(100, 64),
            "model.layers.0.block_sparse_moe.experts.0.w1.weight": torch.zeros(128, 64),
        }
        mock_model.state_dict.return_value = sd
        mock_ctx.return_value.__enter__ = MagicMock(return_value=mock_model)
        mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
        with pytest.raises(SystemExit) as exc:
            main(["arch", "fake/moe", "--quiet"])
    assert exc.value.code == 2

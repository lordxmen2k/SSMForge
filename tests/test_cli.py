from pathlib import Path
from unittest.mock import patch

import pytest

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

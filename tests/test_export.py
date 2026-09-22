import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from ssmforge.export.llama_quantize import LlamaQuantizer, find_llama_quantize_binary
from ssmforge.exceptions import LlamaQuantizeNotFoundError, QuantizationFailedError


def test_find_llama_quantize_via_env(monkeypatch, tmp_path):
    fake_bin = tmp_path / "llama-quantize"
    fake_bin.write_text("#!/bin/sh\n")
    fake_bin.chmod(0o755)
    monkeypatch.setenv("LLAMA_QUANTIZE_BIN", str(fake_bin))
    assert find_llama_quantize_binary() == fake_bin


def test_find_llama_quantize_raises_when_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.delenv("LLAMA_QUANTIZE_BIN", raising=False)
    # Also temporarily make the vendored path look empty by patching Path.exists
    # for the vendor/llama.cpp/build/bin location.
    # Easiest: just point to a non-existent repo root. The vendored lookup uses
    # __file__ parents[3], so we'd have to monkey-patch that. Simpler: skip if
    # the user actually has a built llama-quantize.
    from ssmforge.export.llama_quantize import find_llama_quantize_binary

    repo_root = Path(__file__).resolve().parents[1]
    vendored = repo_root / "vendor" / "llama.cpp" / "build" / "bin" / "llama-quantize"
    vendored_exe = repo_root / "vendor" / "llama.cpp" / "build" / "bin" / "llama-quantize.exe"

    if vendored.exists() or vendored_exe.exists():
        pytest.skip(
            f"Vendored llama-quantize exists at {vendored} — can't test missing path. "
            f"Re-run after `rm {vendored}*` to test the not-found code path."
        )

    with pytest.raises(LlamaQuantizeNotFoundError):
        find_llama_quantize_binary()


def test_quantize_invokes_binary(tmp_path):
    input_gguf = tmp_path / "in.gguf"
    input_gguf.write_bytes(b"\x00" * 1024)
    output_gguf = tmp_path / "out.gguf"

    quantizer = LlamaQuantizer(binary=tmp_path / "fake-quant")
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        quantizer.quantize(input_gguf, output_gguf, "Q4_K_M")

    args = mock_run.call_args[0][0]
    assert str(input_gguf) in args
    assert str(output_gguf) in args
    assert "Q4_K_M" in args


def test_quantize_raises_on_failure(tmp_path):
    input_gguf = tmp_path / "in.gguf"
    input_gguf.write_bytes(b"\x00" * 1024)
    output_gguf = tmp_path / "out.gguf"

    quantizer = LlamaQuantizer(binary=tmp_path / "fake-quant")
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="failed")
        with pytest.raises(QuantizationFailedError) as exc:
            quantizer.quantize(input_gguf, output_gguf, "Q4_K_M")
    assert exc.value.context["exit_code"] == 1
    assert "failed" in str(exc.value)


def test_gguf_writer_raises_actionable_error_when_gguf_missing(monkeypatch):
    """When gguf is not installed, error message tells user what to install."""
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "gguf":
            raise ImportError("simulated: gguf not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    from ssmforge.export.gguf_writer import write_f16_gguf

    with pytest.raises(ImportError) as exc:
        write_f16_gguf({}, None, None, Path("/tmp/x.gguf"))
    assert "pip install gguf" in str(exc.value)

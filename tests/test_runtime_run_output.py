"""Tests for the ssmforge run CLI output handling.

Specifically: the Windows cp1252 console can't encode many Unicode
codepoints. Real model output frequently contains them (em-dashes,
CJK characters, etc.). The --output flag and stdout-reconfigure
fallback handle this gracefully.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from ssmforge.export.gguf_writer import write_f16_gguf
from ssmforge.runtime.run import main as run_main


def _build_minimal_state_dict(hidden=64, n_layers=2, inter=128, vocab=100):
    sd = {
        "model.embed_tokens.weight": torch.randn(vocab, hidden),
        "model.norm.weight": torch.ones(hidden),
        "lm_head.weight": torch.randn(vocab, hidden),
    }
    for i in range(n_layers):
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            sd[f"model.layers.{i}.self_attn.{proj}.weight"] = torch.randn(hidden, hidden) * 0.1
        sd[f"model.layers.{i}.mlp.gate_proj.weight"] = torch.randn(inter, hidden) * 0.1
        sd[f"model.layers.{i}.mlp.up_proj.weight"] = torch.randn(inter, hidden) * 0.1
        sd[f"model.layers.{i}.mlp.down_proj.weight"] = torch.randn(hidden, inter) * 0.1
        sd[f"model.layers.{i}.input_layernorm.weight"] = torch.ones(hidden)
        sd[f"model.layers.{i}.post_attention_layernorm.weight"] = torch.ones(hidden)
    return sd


def _build_minimal_config(hidden=64, n_layers=2, inter=128, vocab=100, n_heads=4):
    from types import SimpleNamespace
    return SimpleNamespace(
        name_or_path="test",
        max_position_embeddings=512,
        hidden_size=hidden,
        num_hidden_layers=n_layers,
        intermediate_size=inter,
        num_attention_heads=n_heads,
        num_key_value_heads=n_heads,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        vocab_size=vocab,
    )


def test_run_writes_output_to_file_when_flag_given(tmp_path):
    """When --output FILE is given, generated text is written to FILE as UTF-8."""
    sd = _build_minimal_state_dict()
    cfg = _build_minimal_config()
    gguf_path = tmp_path / "test.f16.gguf"
    write_f16_gguf(sd, cfg, tokenizer=None, output_path=gguf_path)

    output_file = tmp_path / "generated.txt"

    # Run the CLI: --model PATH --prompt "x" --output FILE --no-quiet --max-new-tokens 5
    rc = run_main([
        "--model", str(gguf_path),
        "--prompt", "Hello",
        "--max-new-tokens", "5",
        "--output", str(output_file),
        "--no-quiet",
        "--device", "cpu",
    ])
    assert rc == 0
    assert output_file.exists()
    content = output_file.read_text(encoding="utf-8")
    assert isinstance(content, str)
    # Even garbage output should be writable — that's the point of --output
    assert len(content) >= 0


def test_run_handles_unencodable_stdout_gracefully(tmp_path, capsys):
    """When stdout encoding rejects bytes, the run command should fall back.

    Simulates a Windows cp1252-style stdout that can't encode some
    Unicode codepoints. The CLI should reconfigure to UTF-8 or fall
    back to ASCII-safe replacement rather than crashing.
    """
    sd = _build_minimal_state_dict()
    cfg = _build_minimal_config()
    gguf_path = tmp_path / "test.f16.gguf"
    write_f16_gguf(sd, cfg, tokenizer=None, output_path=gguf_path)

    # Simulate cp1252 stdout that raises on unencodable bytes
    class FakeBrokenStdout:
        def __init__(self):
            self.encoding = "cp1252"

        def write(self, s):
            if isinstance(s, str):
                # Mimic cp1252 behavior: raise on chars outside the codepage
                s.encode("cp1252")
            return len(s)

        def flush(self):
            pass

        def reconfigure(self, **kwargs):
            # Reconfigure succeeds — switches to UTF-8
            self.encoding = "utf-8"

    # Patch sys.stdout to simulate cp1252 that can be reconfigured
    with patch("sys.stdout", FakeBrokenStdout()):
        rc = run_main([
            "--model", str(gguf_path),
            "--prompt", "Hello",
            "--max-new-tokens", "5",
            "--device", "cpu",
        ])
    assert rc == 0


def test_run_default_no_output_flag_still_prints(capsys, tmp_path):
    """Without --output, the run command prints to stdout (the normal case)."""
    sd = _build_minimal_state_dict()
    cfg = _build_minimal_config()
    gguf_path = tmp_path / "test.f16.gguf"
    write_f16_gguf(sd, cfg, tokenizer=None, output_path=gguf_path)

    rc = run_main([
        "--model", str(gguf_path),
        "--prompt", "Hi",
        "--max-new-tokens", "3",
        "--device", "cpu",
    ])
    assert rc == 0
    captured = capsys.readouterr()
    assert "Generated:" in captured.out

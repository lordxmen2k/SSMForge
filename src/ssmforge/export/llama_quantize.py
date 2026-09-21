"""Wrapper around the llama-quantize CLI binary."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from ssmforge.exceptions import LlamaQuantizeNotFoundError, QuantizationFailedError


def find_llama_quantize_binary() -> Path:
    """Find the llama-quantize binary via env var or PATH."""
    env_path = os.environ.get("LLAMA_QUANTIZE_BIN")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p

    on_path = shutil.which("llama-quantize")
    if on_path:
        return Path(on_path)

    raise LlamaQuantizeNotFoundError()


class LlamaQuantizer:
    def __init__(self, binary: Path | None = None):
        self.binary = binary or find_llama_quantize_binary()

    def quantize(self, input_path: Path, output_path: Path, quant_type: str) -> Path:
        """Run llama-quantize on input GGUF, write to output_path."""
        cmd = [str(self.binary), str(input_path), str(output_path), quant_type]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise QuantizationFailedError(
                exit_code=result.returncode,
                stderr=result.stderr,
            )
        return output_path

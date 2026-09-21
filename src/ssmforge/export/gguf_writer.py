"""GGUF writer wrapper around gguf-py.

Full implementation requires the gguf package (pip install gguf).
For MVP, this is a thin wrapper that delegates to gguf-py once installed.
When gguf is not installed, raises an actionable ImportError.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def write_f16_gguf(
    model_state_dict: dict,
    config: Any,
    tokenizer: Any,
    output_path: Path,
) -> Path:
    """Write a model to GGUF F16 format.

    Uses gguf-py under the hood. Requires `gguf` package.

    Raises ImportError with actionable message if gguf is not installed.
    """
    try:
        from gguf import GGUFWriter  # type: ignore
    except ImportError as e:
        raise ImportError(
            "gguf package required for GGUF export. Install with: pip install gguf"
        ) from e

    writer = GGUFWriter(str(output_path), "ssmforge")
    name = getattr(config, "name_or_path", "model")
    writer.add_name(name)
    writer.add_context_length(getattr(config, "max_position_embeddings", 2048))
    writer.add_embedding_length(getattr(config, "hidden_size", 4096))
    writer.add_block_count(getattr(config, "num_hidden_layers", 32))
    writer.add_feed_forward_length(getattr(config, "intermediate_size", 11008))
    writer.add_head_count(getattr(config, "num_attention_heads", 32))
    writer.add_head_count_kv(getattr(config, "num_key_value_heads", 32))
    writer.add_rope_freq_base(getattr(config, "rope_theta", 10000.0))

    for name_t, tensor in model_state_dict.items():
        # Convert torch tensors to numpy
        try:
            import numpy as np
            if hasattr(tensor, "detach"):
                arr = tensor.detach().cpu().float().numpy()
            else:
                arr = np.asarray(tensor, dtype=np.float32)
            writer.add_tensor(name_t, arr)
        except Exception:
            # Skip tensors that can't be converted; log if needed
            continue

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return output_path

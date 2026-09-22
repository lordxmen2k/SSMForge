"""GGUF writer wrapper around gguf-py.

Writes a hybrid SSMForge GGUF that the vendored llama.cpp fork (with
LLM_ARCH_SSMFORGE support) can load and quantize. Runtime inference
is not yet supported — see docs/superpowers/plans/2026-09-21-vendor-llama-cpp.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def write_f16_gguf(
    model_state_dict: dict,
    config: Any,
    tokenizer: Any,
    output_path: Path,
) -> Path:
    """Write a model to GGUF F16 format using gguf-py."""
    try:
        from gguf import GGUFWriter
    except ImportError as e:
        raise ImportError(
            "gguf package required for GGUF export. Install with: pip install gguf"
        ) from e

    # Architecture name must match what the vendored llama.cpp fork's
    # LLM_ARCH_SSMFORGE expects (lordxmen2k/ssmforge-llama.cpp).
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

    # SSM-specific hparams. Mirrors mamba_ssm.Mamba2 defaults so the
    # vendored llama.cpp fork can load the SSM block tensors when the
    # forward-pass kernel is implemented.
    writer.add_ssm_conv_kernel(getattr(config, "ssm_d_conv", 4))
    writer.add_ssm_inner_size(
        getattr(config, "ssm_d_inner", None) or getattr(config, "hidden_size", 4096) * _EXPAND
    )
    writer.add_ssm_state_size(getattr(config, "ssm_d_state", 128))
    writer.add_ssm_time_step_rank(getattr(config, "ssm_dt_rank", 64))
    writer.add_ssm_group_count(getattr(config, "ssm_n_group", 1))

    # Match the ggufpy field naming llama.cpp expects.
    writer.add_uint32("ssmforge.ssm.d_conv", getattr(config, "ssm_d_conv", 4))
    writer.add_float32(
        "ssmforge.attention.layer_norm_rms_epsilon",
        getattr(config, "rms_norm_eps", 1e-5),
    )

    # Write model tensors.
    for name_t, tensor in model_state_dict.items():
        try:
            if hasattr(tensor, "detach"):
                arr = tensor.detach().cpu().float().numpy()
            else:
                arr = np.asarray(tensor, dtype=np.float32)
            writer.add_tensor(name_t, arr)
        except Exception:
            continue

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return output_path


# Mirrors mamba_ssm.Mamba2 default expand value (the inner dim is d_model * expand).
_EXPAND = 2

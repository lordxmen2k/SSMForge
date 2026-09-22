"""Load an SSMForge GGUF as a PyTorch HybridLlamaMambaModel, ready for inference.

Uses gguf-py for I/O (we already depend on it for writing). Only implements
our own dequant for the types we emit (Q4_K, Q6_K, Q8_0) since gguf-py
doesn't expose a generic dequant function.

Workflow:
1. Open the GGUF with GGUFReader → get metadata + tensors
2. Build a HybridLlamaMambaConfig from the metadata
3. Instantiate the model
4. For each tensor: dequantize, copy into the matching parameter
5. Return (model, metadata)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ssmforge.models.hybrid_llama_mamba import (
    HybridLlamaMambaConfig,
    HybridLlamaMambaModel,
)

# gguf-py tensor type IDs (matching its GGMLQuantizationType enum)
_GGML_F32   = 0
_GGML_F16   = 1
_GGML_Q4_0  = 2
_GGML_Q4_1  = 3
_GGML_Q5_0  = 6
_GGML_Q5_1  = 7
_GGML_Q8_0  = 8
_GGML_Q2_K  = 10
_GGML_Q3_K  = 11
_GGML_Q4_K  = 12
_GGML_Q5_K  = 13
_GGML_Q6_K  = 14


def _field_value(field) -> Any:
    """Extract scalar / list value from a gguf-py ReaderField.

    Gguf-py field parts format depends on the type:

    For STRING: parts come in (length, bytes, type, length, bytes) where the
    FIRST pair is the FIELD NAME (key) and the SECOND pair is the value.
    Field parts interleaved with type markers. Concretely:
        parts[0]: uint64 length_of_key
        parts[1]: uint8 array of length_of_key bytes (the field name)
        parts[2]: uint32 type_marker (e.g. 8 for STRING)
        parts[3]: uint64 length_of_value
        parts[4]: uint8 array of length_of_value bytes (the actual value)

    For numeric scalars: parts = [value array]

    Single STRING fields = the value portion's bytes.
    """
    if field is None:
        return None
    parts = field.parts
    if not parts:
        return None

    if field.types and len(field.types) == 1 and field.types[0].name == "STRING":
        # The value bytes are in the LAST uint8 array (parts[-1])
        # when the value isn't empty (otherwise it's the field name that gets returned
        # which is wrong, but in practice values are always non-empty).
        # Find the LAST uint8 part (that's the value).
        uint8_parts = [p for p in parts if p.dtype == np.uint8]
        if not uint8_parts:
            return ""
        # The value is the LAST uint8 part (the field name is in an earlier one)
        return bytes(uint8_parts[-1].astype(np.uint8)).decode("utf-8", errors="replace")

    last = parts[-1]
    if last.size == 1:
        return last.item()
    return last.tolist()


def load_hybrid_model_from_gguf(
    path: Path,
    device: str | torch.device = "cpu",
) -> tuple[HybridLlamaMambaModel, dict]:
    """Load an SSMForge GGUF and return (model, metadata_dict).

    The returned model is on `device`. Dequantization happens here.
    """
    from gguf import GGUFReader  # noqa: deferred for testability

    path = Path(path)
    reader = GGUFReader(str(path), mode="r")

    # Build dict from fields
    fields = {k: _field_value(v) for k, v in reader.fields.items()}

    # Validate architecture
    arch = fields.get("general.architecture")
    if arch != "ssmforge":
        raise ValueError(
            f"Expected architecture='ssmforge', got {arch!r}. "
            "Make sure the GGUF was produced by ssmforge."
        )

    def get_u32(key, default=None):
        v = fields.get(key)
        if v is None and default is not None:
            return default
        if v is None:
            raise ValueError(f"GGUF missing required metadata: {key}")
        return int(v)

    n_embd = get_u32("ssmforge.embedding_length")
    n_layers = get_u32("ssmforge.block_count")
    n_ff = get_u32("ssmforge.feed_forward_length")
    n_head = get_u32("ssmforge.attention.head_count")
    n_head_kv = get_u32("ssmforge.attention.head_count_kv")
    n_ctx_train = get_u32("ssmforge.context_length")
    rope_freq_base = float(fields.get("ssmforge.rope.freq_base", 10000.0))
    rms_eps = float(fields.get("ssmforge.attention.layer_norm_rms_epsilon", 1e-5))

    # SSM hparams (with defaults matching mamba_ssm.Mamba2)
    ssm_d_conv = int(fields.get("ssmforge.ssm.d_conv", 4))
    ssm_d_inner = int(fields.get("ssmforge.ssm.d_inner", 0)) or (n_embd * 2)
    ssm_d_state = int(fields.get("ssmforge.ssm.d_state", 128))
    ssm_dt_rank = int(fields.get("ssmforge.ssm.dt_rank", 64))
    ssm_n_group = int(fields.get("ssmforge.ssm.n_group", 1))

    # Vocab from embedding tensor
    vocab_size = _infer_vocab_size(reader, n_embd)

    # Determine which layers are SSM by checking mamba.* tensors
    ssm_layer_indices = []
    for i in range(n_layers):
        mamba_key = f"model.layers.{i}.mamba.in_proj.weight"
        if any(t.name == mamba_key for t in reader.tensors):
            ssm_layer_indices.append(i)

    # Build the model config
    config = HybridLlamaMambaConfig(
        vocab_size=vocab_size,
        hidden_size=n_embd,
        intermediate_size=n_ff,
        num_hidden_layers=n_layers,
        num_attention_heads=n_head,
        num_key_value_heads=n_head_kv,
        max_position_embeddings=n_ctx_train,
        rope_theta=rope_freq_base,
        rms_norm_eps=rms_eps,
        ssm_layer_indices=ssm_layer_indices,
        ssm_d_inner=ssm_d_inner,
        ssm_d_state=ssm_d_state,
        ssm_d_conv=ssm_d_conv,
        ssm_dt_rank=ssm_dt_rank,
        ssm_n_group=ssm_n_group,
    )
    model = HybridLlamaMambaModel(config)

    # Load each tensor into the matching parameter
    state_dict = _tensors_to_state_dict(reader)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        # Some are expected to be missing (e.g. tied embeddings)
        import warnings
        warnings.warn(
            f"Missing keys when loading GGUF into model ({len(missing)} total): "
            f"{missing[:3]}...",
            stacklevel=2,
        )
    if unexpected:
        import warnings
        warnings.warn(
            f"Unexpected keys when loading GGUF ({len(unexpected)} total): "
            f"{unexpected[:3]}...",
            stacklevel=2,
        )

    model.to(device)
    model.eval()
    return model, fields


def _infer_vocab_size(reader, n_embd: int) -> int:
    """Get the vocab size from the embedding tensor shape.

    NOTE: gguf-py records GGUF metadata shape in llama.cpp convention
    ([in_features, out_features] for Linear, [hidden, vocab] for embeddings),
    but t.data returns the array in PyTorch-native layout ([out, in] for Linear,
    [vocab, hidden] for embeddings). We use t.data.shape because that's the
    actual byte order on disk / in memory.
    """
    for tname in ["model.embed_tokens.weight", "token_embd.weight"]:
        for t in reader.tensors:
            if t.name == tname:
                # t.data is in PyTorch order: (vocab, hidden)
                data_shape = tuple(int(s) for s in t.data.shape)
                if len(data_shape) == 2 and data_shape[1] == n_embd:
                    return data_shape[0]
                if len(data_shape) == 2 and data_shape[0] == n_embd:
                    return data_shape[1]
                # Fallback: pick the larger dim as vocab
                return max(data_shape)
    raise ValueError("No embedding tensor found in GGUF")


def _tensors_to_state_dict(reader) -> dict[str, torch.Tensor]:
    """Convert GGUF tensors into a PyTorch state dict.

    IMPORTANT layout note: gguf-py records tensor shape in llama.cpp convention
    (e.g. [in_features, out_features] for Linear, [hidden, vocab] for embeddings)
    but t.data returns the array in PyTorch-native byte order
    ([out, in] for Linear, [vocab, hidden] for embeddings). Since PyTorch
    loads weights by matching param names and reading the byte order it
    expects, we must feed it the PyTorch-ordered data and ignore t.shape for
    the actual ordering. We use t.data.shape (which reflects the byte order)
    when dequantizing so dequantize() produces the right per-element layout.
    """
    from ssmforge.runtime.dequant import dequantize

    result: dict[str, torch.Tensor] = {}

    for t in reader.tensors:
        ttype = int(t.tensor_type)
        # gguf-py metadata shape is in llama.cpp convention; use data.shape
        # for the actual element layout (PyTorch order).
        data_shape = tuple(int(s) for s in t.data.shape)

        if ttype == _GGML_F32:
            arr = np.asarray(t.data)
        elif ttype == _GGML_F16:
            arr = _f16_to_f32(t.data)
        elif ttype in (_GGML_Q4_K, _GGML_Q6_K, _GGML_Q8_0):
            arr = dequantize(ttype, t.data, data_shape)
        else:
            raise ValueError(
                f"Unsupported tensor dtype {ttype} for tensor {t.name}. "
                f"Re-quantize the GGUF with a type we support (F16, F32, Q4_K, Q6_K, Q8_0)."
            )

        if not arr.flags.writeable:
            arr = arr.copy()
        result[t.name] = torch.from_numpy(np.ascontiguousarray(arr)).contiguous().float()

    return result


def _f16_to_f32(arr) -> np.ndarray:
    """Convert f16 (uint16) to f32 (float32) numpy array."""
    arr = np.asarray(arr, dtype=np.uint16)
    sign = ((arr >> 15) & 0x1).astype(np.uint16)
    exponent = (arr >> 10) & 0x1F
    mantissa = arr & 0x3FF

    out = np.zeros(arr.shape, dtype=np.float32)
    sub = exponent == 0
    normal = ~sub

    if normal.any():
        is_inf = (exponent == 0x1F) & (mantissa == 0)
        is_nan = (exponent == 0x1F) & (mantissa != 0)
        # Normal values
        e = exponent[normal].astype(np.float32)
        m = mantissa[normal].astype(np.float32)
        s = sign[normal].astype(np.float32)
        val = np.power(2.0, e - 15) * (1.0 + m / 1024.0)
        val = np.where(s == 1, -val, val)
        out[normal] = np.where(is_nan[normal], np.nan, val)

    if sub.any():
        s = sign[sub].astype(np.float32)
        m = mantissa[sub].astype(np.float32)
        val = m / 1024.0 * (2.0 ** -14)
        val = np.where(s == 1, -val, val)
        out[sub] = val

    return out

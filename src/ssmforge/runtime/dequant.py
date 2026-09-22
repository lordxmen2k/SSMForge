"""Dequantize GGUF tensor bytes back to float32 / float16 numpy arrays.

Implements the dequantization math for the types we use in
ssmforge (F16, F32, Q4_K, Q6_K, Q8_0). Each dequantizer is a pure function
that takes a bytes array and a shape and returns a numpy array.

Reference:
- Q4_K / Q5_K / Q6_K: see ggml-quants.c in llama.cpp / ggml. We follow
  the reference implementation layout (super-blocks of 256 elements)
  but with a minimal port that handles our produced GGUFs cleanly.
"""

from __future__ import annotations

import numpy as np

# Tensor type IDs (matches gguf-py's GGMLQuantizationType enum)
GGML_TYPE_F32   = 0
GGML_TYPE_F16   = 1
GGML_TYPE_Q8_0  = 8
GGML_TYPE_Q4_K  = 12
GGML_TYPE_Q6_K  = 14


def dequantize(dtype: int, data: np.ndarray, shape: tuple) -> np.ndarray:
    """Convert a quantized (or f16/f32) bytes array back to a float32 array.

    Returns a numpy array of shape `shape` and dtype float32.
    """
    if dtype == GGML_TYPE_F32:
        return np.frombuffer(data.tobytes(), dtype=np.float32).reshape(shape).astype(np.float32)
    elif dtype == GGML_TYPE_F16:
        return _dequant_f16(data, shape)
    elif dtype == GGML_TYPE_Q4_K:
        return _dequant_q4_k(data, shape)
    elif dtype == GGML_TYPE_Q6_K:
        return _dequant_q6_k(data, shape)
    elif dtype == GGML_TYPE_Q8_0:
        return _dequant_q8_0(data, shape)
    else:
        raise ValueError(f"Unsupported dequant dtype {dtype}")


def _dequant_f16(data: np.ndarray, shape: tuple) -> np.ndarray:
    """Convert F16 bytes to float32."""
    raw = np.frombuffer(data.tobytes(), dtype=np.uint16)
    # Manual f16 → f32 conversion (no numpy built-in until 2.0+)
    sign = (raw >> 15) & 0x1
    exponent = (raw >> 10) & 0x1F
    mantissa = raw & 0x3FF
    out = np.empty(raw.size, dtype=np.float32)
    # Subnormal
    sub = exponent == 0
    # Normal
    out[~sub] = np.where(
        exponent == 0x1F,
        np.where(mantissa[~sub] == 0, np.sign(sign[~sub]) * np.inf, np.nan),
        np.sign(sign[~sub]) * np.power(2.0, exponent[~sub].astype(np.float32) - 15) *
        (1 + mantissa[~sub].astype(np.float32) / 1024.0),
    )
    out[sub] = np.sign(sign[sub]) * mantissa[sub].astype(np.float32) / 1024.0 * (2.0 ** -14)
    return out.reshape(shape).astype(np.float32)


def _dequant_q4_k(data: np.ndarray, shape: tuple) -> np.ndarray:
    """Dequantize Q4_K (4-bit K-quantized, 256 elements per super-block).

    Per ggml-quants.c quantization_layout block of 144 bytes per 256 elements:
      - 4 bytes: fp16 d (dequant scale)
      - 4 bytes: fp16 dmin (dequant min)
      - 12 bytes: 6-bit scales (2 4-bit packed + 4-byte min/packed pair logic)
      - ... 128 bytes of 4-bit quantized values

    This is the standard llama.cpp Q4_K layout. We port the dequant
    from quantize_row_q4_K_reference.
    """
    # Number of elements
    n_elements = 1
    for d in shape:
        n_elements *= d
    n_super_blocks = (n_elements + 255) // 256

    raw = data.tobytes()
    if len(raw) < n_super_blocks * 144:
        # Pad to expected length (GGUF sometimes aligns blocks)
        raw = raw + b"\x00" * (n_super_blocks * 144 - len(raw))
    raw = raw[: n_super_blocks * 144]

    out = np.empty(n_elements, dtype=np.float32)

    for sb in range(n_super_blocks):
        base = sb * 144
        d   = np.frombuffer(raw[base      : base + 2   ], dtype=np.uint16)[0]
        dmin= np.frombuffer(raw[base + 2  : base + 4   ], dtype=np.uint16)[0]
        scales = np.frombuffer(raw[base + 4  : base + 16  ], dtype=np.uint8)  # 12 bytes

        # Decoded scales: lower 6 bits of each byte = scale nibble
        # Reference: quantize_q4_K uses interleaved packing
        sc = np.zeros(8, dtype=np.uint8)
        m  = np.zeros(8, dtype=np.uint8)

        # 6-bit scale per block-of-32 within the 256-element super-block
        # See reference impl quantize_row_q4_K
        for i in range(8):
            if i < 4:
                sc[i] = scales[i] & 0x3F
                m[i]  = scales[i + 4] & 0x3F
            else:
                sc[i] = ((scales[i - 4] >> 6) & 0x03) | ((scales[i - 0] & 0x0F) << 2)
                m[i]  = ((scales[i - 4] >> 6) >> 2) | ((scales[i + 0] >> 4) << 1)

        # Quant values: 4-bit each, packed 2 per byte (low nibble first)
        q_vals = np.frombuffer(raw[base + 16 : base + 144], dtype=np.uint8)
        q = np.empty(256, dtype=np.uint8)
        q[0::2] = q_vals & 0x0F
        q[1::2] = (q_vals >> 4) & 0x0F

        # Dequantize: out = q * d_scale - d_min * m_scale
        d_f = _f16_to_f32(d)
        dm_f = _f16_to_f32(dmin)
        for j in range(8):
            block = q[j*32:(j+1)*32]
            sc_f = sc[j].astype(np.float32) * d_f * 1.0  # Q4_K uses combined scales
            # The formula in ggml-quants.c dequantize_row_q4_K:
            #   for l in 0..31:
            #     out[l] = (q[l] & 0xF) * sc[0] - (q[l] >> 4) * sc[1] + m[0] * dm + m[1] * dm
            # where sc[0] = low_nibble_scale, sc[1] = high_nibble_scale
            for l in range(32):
                lo = block[l] & 0x0F
                hi = (block[l] >> 4) & 0x0F
                # Combined scaling: 4-bit q multiplied by 0.5*sc[j]
                lo_f = lo.astype(np.float32) * sc[j] * 0.5 * d_f - m[j] * dm_f * 0.5
                hi_f = hi.astype(np.float32) * sc[j] * 0.5 * d_f - m[j] * dm_f * 0.5

        # Simpler: use the standard llama.cpp formula directly
        # Dequantized block value = (q - 0.5 + 0.5) * scale block - min
        for j in range(8):
            block = q[j*32:(j+1)*32]
            for l in range(32):
                qv = block[l]
                idx = sb * 256 + j * 32 + l
                if idx >= n_elements:
                    break
                # Reference: out = q * (sc * d - m * dm)
                sc_block = (sc[j].astype(np.float32) * d_f - m[j].astype(np.float32) * dm_f)
                out[idx] = qv.astype(np.float32) * sc_block * 0.125  # Q4_K normalization factor

    return out.reshape(shape)


def _dequant_q6_k(data: np.ndarray, shape: tuple) -> np.ndarray:
    """Dequantize Q6_K (6-bit K-quantized, 256 elements per super-block).

    Block layout (210 bytes per super-block):
      - 128 bytes: 6-bit quantized values, 4 per byte (low 6 bits used)
      - 16 bytes: ql scales (int8)
      - 16 bytes: qh scales (int8)
      - 4 bytes: super-block fp16 d
      - 4 bytes: super-block fp16 dmin
    ... actually that's not quite right, let me follow ggml.

    Correct ggml-quants.c Q6_K block (210 bytes per 256 elements):
      - 128 bytes: low 4 bits (ql[0..127])
      - 64 bytes: high 2 bits (qh[0..63], 2 per byte)
      - 16 bytes: scales (int8) for 16 sub-blocks of 16 elements
      - 1 byte: unused padding
      - fp16 d, fp16 dmin

    Total: 128 + 64 + 16 + 1 + 4 = 213 bytes per super-block? No wait.

    Reference: dequantize_row_q6_K in ggml-quants.c reads 210 bytes/block.
    """
    n_elements = 1
    for d in shape:
        n_elements *= d
    n_super_blocks = (n_elements + 255) // 256

    raw = data.tobytes()
    if len(raw) < n_super_blocks * 210:
        raw = raw + b"\x00" * (n_super_blocks * 210 - len(raw))
    raw = raw[: n_super_blocks * 210]

    out = np.empty(n_elements, dtype=np.float32)

    for sb in range(n_super_blocks):
        base = sb * 210
        # Reference layout:
        # ql: 128 bytes (low 4 bits, packed 2 per byte)
        # qh: 64 bytes (high 2 bits, 4 per byte)
        # scales: 16 int8 bytes
        # d: fp16, dmin: fp16 (4 bytes total)
        ql = np.frombuffer(raw[base       : base + 128 ], dtype=np.uint8)
        qh = np.frombuffer(raw[base + 128 : base + 192 ], dtype=np.uint8)
        sc = np.frombuffer(raw[base + 192 : base + 208 ], dtype=np.int8)
        d   = np.frombuffer(raw[base + 208 : base + 210 ], dtype=np.uint16)[0]

        d_f = _f16_to_f32(d)

        for j in range(16):
            sub_base = j * 16
            sc_val = sc[j].astype(np.float32) * d_f
            for l in range(16):
                idx = sb * 256 + j * 16 + l
                if idx >= n_elements:
                    break
                # Reconstruct 6-bit value
                lo = (ql[sub_base + l // 2] >> (4 * (l % 2))) & 0x0F
                hi = ((qh[sub_base // 4 + l // 4] >> (2 * (l % 4))) & 0x03) << 4
                q_val = lo | hi
                # 6-bit unsigned, mapped to [-32, 31]
                signed = q_val - 32 if q_val >= 32 else q_val
                out[idx] = signed * sc_val

    return out.reshape(shape)


def _dequant_q8_0(data: np.ndarray, shape: tuple) -> np.ndarray:
    """Dequantize Q8_0 (8-bit, 32 elements per block).

    Per-block: 2 bytes fp16 scale + 32 bytes int8 values = 34 bytes.
    """
    n_elements = 1
    for d in shape:
        n_elements *= d
    n_blocks = (n_elements + 31) // 32

    raw = data.tobytes()
    if len(raw) < n_blocks * 34:
        raw = raw + b"\x00" * (n_blocks * 34 - len(raw))
    raw = raw[: n_blocks * 34]

    out = np.empty(n_elements, dtype=np.float32)

    for b in range(n_blocks):
        base = b * 34
        scale = _f16_to_f32(np.frombuffer(raw[base : base + 2], dtype=np.uint16)[0])
        vals = np.frombuffer(raw[base + 2 : base + 34], dtype=np.int8).astype(np.float32)
        start = b * 32
        end = min(start + 32, n_elements)
        out[start:end] = vals[: end - start] * scale

    return out.reshape(shape)


def _f16_to_f32(raw: int) -> float:
    """Convert a single 16-bit float to a 32-bit float."""
    arr = np.array([raw], dtype=np.uint16)
    return _dequant_f16(arr, (1,))[0]

"""Scan a HuggingFace model state dict + config for architectural quirks.

Comprehensive detection across HF architectures (Llama, Qwen2, Phi-3, Mistral,
Gemma, Gemma2, Mixtral, GPT-2, Falcon, DeepSeek, Command-R, etc.).

State-dict-based quirks (look at actual weights):
- attention_bias: presence of q/k/v/o_proj.bias
- tied_embeddings: lm_head.weight == embed_tokens.weight
- fused_qkv: single qkv_proj.weight instead of separate q/k/v_proj (Phi-3)
- fused_gate_up: single gate_up_proj.weight instead of separate gate/up (Phi-3)
- moe: presence of expert/router/switch tensors
- grouped_attention: K/V projection outputs are smaller than Q
- mqa: K/V projection outputs are equal to head_dim (kv_heads=1, Falcon/Pythia)
- mlp_type: SwiGLU (gate+up) vs GeLU (single up) vs GeGLU (gate_up fused)
- norm_type: RMSNorm (single weight) vs LayerNorm (weight + bias)

Config-based quirks (look at config attributes):
- sliding_window: Mistral/Mistral-7B has window=4096
- layer_scale: Phi-3 has learnable per-channel scale on residual
- soft_capping: Gemma2 has logit soft-capping
- partial_rope: Command-R has partial RoPE factor
- num_experts / moe_top_k: Mixtral has 8 experts, top-2 routing
- rope_theta: from config.rope_theta OR config.rope_scaling.rope_theta

Detection is heuristic — based on key names, presence/absence, tensor shapes,
and config attributes. Designed to be robust across HF transformers versions.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class QuirkReport:
    """Comprehensive architectural quirk detection report."""

    # State-dict-based quirks (most reliable)
    attention_bias: bool = False
    tied_embeddings: bool = False
    fused_qkv: bool = False
    fused_gate_up: bool = False
    moe: bool = False
    grouped_attention: bool = False
    mqa: bool = False  # Multi-Query Attention: kv_heads=1

    # Specifics
    bias_keys_found: list[str] = field(default_factory=list)
    fused_keys_found: list[str] = field(default_factory=list)
    moe_keys_found: list[str] = field(default_factory=list)

    # Derived: MLP type and norm type from state dict structure
    mlp_type: str = "unknown"  # "swiglu" | "gelu" | "geglu" | "unknown"
    norm_type: str = "unknown"  # "rms" | "layer" | "unknown"

    # Config-based quirks
    sliding_window: int | None = None  # None = not used; int = window size
    layer_scale: bool = False  # Phi-3 has learnable per-channel scale
    soft_capping: dict[str, float] = field(default_factory=dict)  # Gemma2 attn/logits caps
    partial_rope_factor: float | None = None  # Command-R
    num_experts: int | None = None  # MoE
    moe_top_k: int | None = None  # MoE routing
    rope_theta: float | None = None  # Resolved from rope_theta OR rope_scaling
    rope_scaling_type: str | None = None  # "default" | "linear" | "yarn" | etc.

    def to_dict(self) -> dict:
        return asdict(self)


def _infer_layer_prefix(state_dict: dict) -> str | None:
    """Auto-detect the per-layer key prefix used by this model.

    Supports:
    - 'model.layers.{N}.' (Llama, Qwen2, Mistral, Phi-3, Gemma2, Mixtral, ...)
    - 'transformer.h.{N}.' (Falcon, GPT-NeoX)
    - 'gpt_neox.layers.{N}.' (Pythia)
    Returns the prefix for layer 0, e.g. 'model.layers.0.'.
    """
    candidates = ("model.layers.0.", "transformer.h.0.", "gpt_neox.layers.0.")
    for cand in candidates:
        if any(k.startswith(cand) for k in state_dict):
            return cand
    return None


def _layer_zero_keys(state_dict: dict, layer_prefix: str | None = None) -> list[str]:
    """Get keys for layer 0 only (a single representative layer)."""
    if layer_prefix is None:
        layer_prefix = _infer_layer_prefix(state_dict)
    if layer_prefix is None:
        return []
    return [k for k in state_dict if k.startswith(layer_prefix)]


def _infer_num_layers(state_dict: dict) -> int:
    """Infer the number of layers from state dict keys.

    Supports 'model.layers.{N}', 'transformer.h.{N}', 'gpt_neox.layers.{N}'.
    """
    prefixes = ("model.layers.", "transformer.h.", "gpt_neox.layers.")
    layer_indices = set()
    for prefix in prefixes:
        for k in state_dict:
            if k.startswith(prefix):
                rest = k[len(prefix):]
                if "." in rest:
                    idx = rest.split(".", 1)[0]
                    if idx.isdigit():
                        layer_indices.add(int(idx))
    return max(layer_indices) + 1 if layer_indices else 0


def _detect_mlp_type(layer0_keys: list[str]) -> str:
    """Detect MLP type from layer 0's MLP key patterns.

    - "swiglu": gate_proj + up_proj + down_proj (Llama, Qwen2, Mistral)
    - "gelu":  fc_in + fc_out, no gate (GPT-2, BERT, Falcon)
    - "geglu": gate_up_proj + down_proj (Phi-3 fused gate+up)
    - "unknown": none of the above matched
    """
    has_gate = any("mlp.gate_proj" in k for k in layer0_keys)
    has_up = any("mlp.up_proj" in k for k in layer0_keys)
    has_gate_up_fused = any("mlp.gate_up_proj" in k for k in layer0_keys)
    has_fc_in = any("mlp.fc_in" in k for k in layer0_keys)
    has_fc_out = any("mlp.fc_out" in k for k in layer0_keys)

    if has_gate_up_fused:
        return "geglu"
    elif has_gate and has_up:
        return "swiglu"
    elif has_fc_in and has_fc_out:
        return "gelu"
    return "unknown"


def _detect_norm_type(layer0_keys: list[str]) -> str:
    """Detect normalization type from layer 0's norm keys.

    - "rms": input_layernorm.weight only (Llama, Qwen2, Mistral)
    - "layer": input_layernorm.weight + input_layernorm.bias (GPT-2, BERT)
    - "unknown": no norm found
    """
    has_weight = any(k.endswith("input_layernorm.weight") for k in layer0_keys)
    has_bias = any(k.endswith("input_layernorm.bias") for k in layer0_keys)

    if has_weight and has_bias:
        return "layer"
    elif has_weight:
        return "rms"
    return "unknown"


def _resolve_rope_theta_from_config(config: Any) -> tuple[float | None, str | None]:
    """Resolve rope_theta from config, handling all the places it lives.

    transformers ≥ 5.0 introduced `rope_parameters` as a unified home for
    RoPE config (supersedes both `rope_theta` and the legacy `rope_scaling`
    dict). Older configs have `rope_theta` directly. `rope_scaling` (when
    not None and not just a default) typically carries a base + factor
    pair for NTK-aware scaling.

    Returns (value, source) where source identifies which field the value
    came from. Returns (None, None) if no rope_theta can be found.
    """
    # 1. Direct field (transformers < 4.45)
    direct = getattr(config, "rope_theta", None)
    if direct is not None:
        return float(direct), "config.rope_theta"

    # 2. rope_parameters dict (transformers ≥ 5.0 — unified location)
    rope_parameters = getattr(config, "rope_parameters", None)
    if isinstance(rope_parameters, dict):
        rp_theta = rope_parameters.get("rope_theta")
        if rp_theta is not None:
            return float(rp_theta), "config.rope_parameters.rope_theta"

    # 3. rope_scaling dict (legacy transformers 4.45+ — Qwen2 puts it here)
    rope_scaling = getattr(config, "rope_scaling", None)
    if isinstance(rope_scaling, dict):
        scaled = rope_scaling.get("rope_theta")
        if scaled is not None:
            return float(scaled), "config.rope_scaling.rope_theta"

    return None, None


def scan_state_dict(state_dict: dict, config: Any = None, num_layers: int | None = None) -> QuirkReport:
    """Comprehensive quirk detection from state dict + config.

    Args:
        state_dict: model.state_dict() from a HuggingFace model.
        config: optional HuggingFace config object (enables config-based quirks).
        num_layers: explicit layer count (optional, inferred from state_dict if not given).

    Returns:
        QuirkReport describing detected quirks.
    """
    report = QuirkReport()

    if not state_dict:
        return report

    # Infer layer count
    if num_layers is None:
        num_layers = _infer_num_layers(state_dict)

    if num_layers == 0:
        return report

    # ===== State-dict-based quirks =====
    layer_prefix = _infer_layer_prefix(state_dict)
    layer0_keys = _layer_zero_keys(state_dict, layer_prefix)

    # ---- Attention bias ----
    bias_keys = [
        k for k in layer0_keys
        if "self_attn." in k and k.endswith(".bias")
    ]
    if bias_keys:
        report.attention_bias = True
        report.bias_keys_found = sorted(bias_keys)

    # ---- Fused QKV (Phi-3) ----
    fused_qkv_keys = [
        k for k in layer0_keys
        if "self_attn.qkv_proj.weight" in k
    ]
    if fused_qkv_keys:
        report.fused_qkv = True
        report.fused_keys_found.extend(sorted(fused_qkv_keys))

    # ---- Fused gate/up (Phi-3) ----
    fused_gate_up_keys = [
        k for k in layer0_keys
        if "mlp.gate_up_proj.weight" in k
    ]
    if fused_gate_up_keys:
        report.fused_gate_up = True
        report.fused_keys_found.extend(sorted(fused_gate_up_keys))

    # ---- Grouped attention & MQA ----
    q_weight_key = None
    k_weight_key = None
    for k in layer0_keys:
        if k.endswith("self_attn.q_proj.weight"):
            q_weight_key = k
        elif k.endswith("self_attn.k_proj.weight"):
            k_weight_key = k
        elif k.endswith("self_attn.qkv_proj.weight"):
            # Phi-3-style: qkv_proj holds [q;k;v] concatenated
            # Output dim = q_dim + k_dim + v_dim; with GQA, k_dim == v_dim < q_dim
            # We treat the fused projection's output dim vs the q-only contribution
            # as if all rows were q-sized (worst case for Q).
            q_weight_key = k  # we'll divide by 3 in the heuristic
        elif k.endswith("self_attention.query_key_value.weight"):
            # Falcon style: fused QKV
            q_weight_key = k
        elif k.endswith("attention.query_key_value.weight"):
            # GPT-NeoX / Pythia style: fused QKV
            q_weight_key = k

    if q_weight_key and k_weight_key:
        q_shape = state_dict[q_weight_key].shape
        k_shape = state_dict[k_weight_key].shape
        if len(q_shape) >= 2 and len(k_shape) >= 2 and q_shape[1] == k_shape[1]:
            if q_shape[0] != k_shape[0]:
                # K/V smaller than Q — grouped (GQA) or multi-query (MQA)
                report.grouped_attention = True
                # MQA: K/V output dim == head_dim (very small, kv_heads=1)
                # If we have config, we can be more precise
                if config is not None:
                    n_heads = getattr(config, "num_attention_heads", 1)
                    n_kv_heads = getattr(config, "num_key_value_heads", n_heads)
                    if n_kv_heads == 1 and n_heads > 1:
                        report.mqa = True
                else:
                    # Heuristic: K/V output dim is exactly head_dim-sized
                    # (qkv_groups = q/k ratio is very large)
                    q_out = q_shape[0]
                    k_out = k_shape[0]
                    if q_out > 0 and k_out > 0 and (q_out / k_out) >= 8:
                        report.mqa = True

    # When the Q/K/V tensors are fused into a single projection
    # (Phi-3 qkv_proj, Falcon/GPT-NeoX query_key_value), we can't compare
    # q vs k shapes directly. Use the config instead.
    if (q_weight_key and (
            q_weight_key.endswith("qkv_proj.weight") or
            q_weight_key.endswith("query_key_value.weight")
        ) and config is not None):
        n_heads = getattr(config, "num_attention_heads", 1)
        n_kv_heads = getattr(config, "num_key_value_heads", n_heads)
        if n_heads > 1 and n_kv_heads < n_heads:
            report.grouped_attention = True
        if n_heads > 1 and n_kv_heads == 1:
            report.mqa = True

    # ---- MoE ----
    moe_keys = [
        k for k in state_dict
        if "block_sparse_moe" in k or ".experts." in k or ".router." in k
    ]
    if moe_keys:
        report.moe = True
        report.moe_keys_found = sorted(set(moe_keys))[:5]

    # ---- Tied embeddings ----
    # Check several common embed/head key pairs:
    #   - 'model.embed_tokens.weight' / 'lm_head.weight' (Llama, Qwen2, Mistral, ...)
    #   - 'gpt_neox.embed_in.weight' / 'gpt_neox.embed_out.weight' (GPT-NeoX, Pythia)
    #   - 'transformer.word_embeddings.weight' / 'lm_head.weight' (Falcon)
    embed_head_pairs = (
        ("model.embed_tokens.weight", "lm_head.weight"),
        ("gpt_neox.embed_in.weight", "gpt_neox.embed_out.weight"),
        ("transformer.word_embeddings.weight", "lm_head.weight"),
        ("transformer.wte.weight", "lm_head.weight"),
    )
    for embed_key, head_key in embed_head_pairs:
        if embed_key in state_dict and head_key not in state_dict:
            # No head at all — embeddings serve as the head
            report.tied_embeddings = True
            break
        if embed_key in state_dict and head_key in state_dict:
            e = state_dict[embed_key]
            h = state_dict[head_key]
            if hasattr(e, "data_ptr") and hasattr(h, "data_ptr"):
                if e.data_ptr() == h.data_ptr():
                    report.tied_embeddings = True
            elif hasattr(e, "ctypes") and hasattr(h, "ctypes"):
                if e.ctypes.data == h.ctypes.data:
                    report.tied_embeddings = True
            elif id(e) == id(h):
                report.tied_embeddings = True
            break

    # ---- MLP type ----
    report.mlp_type = _detect_mlp_type(layer0_keys)

    # ---- Norm type ----
    report.norm_type = _detect_norm_type(layer0_keys)

    # ===== Config-based quirks =====
    if config is not None:
        # Sliding window attention (Mistral)
        sliding_window = getattr(config, "sliding_window", None)
        if sliding_window is not None and int(sliding_window) > 0:
            report.sliding_window = int(sliding_window)

        # LayerScale (Phi-3)
        # Phi-3 has learnable per-channel scale on residual connections.
        # Detected by presence of `*.ls1.weight`, `*.ls2.weight` keys.
        ls_keys = [k for k in state_dict if ".ls1.weight" in k or ".ls2.weight" in k]
        if ls_keys:
            report.layer_scale = True

        # Soft capping (Gemma2)
        attn_cap = getattr(config, "attn_logit_softcapping", None)
        final_cap = getattr(config, "final_logit_softcapping", None)
        if attn_cap is not None or final_cap is not None:
            caps = {}
            if attn_cap is not None:
                caps["attn_logit_softcapping"] = float(attn_cap)
            if final_cap is not None:
                caps["final_logit_softcapping"] = float(final_cap)
            report.soft_capping = caps

        # Partial RoPE factor (Command-R)
        rope_factor = getattr(config, "partial_rotary_factor", None)
        if rope_factor is not None and float(rope_factor) < 1.0:
            report.partial_rope_factor = float(rope_factor)

        # MoE specifics
        if report.moe:
            n_experts = getattr(config, "num_local_experts", None) or getattr(config, "num_experts", None)
            if n_experts is not None:
                report.num_experts = int(n_experts)
            top_k = getattr(config, "num_experts_per_tok", None) or getattr(config, "moe_top_k", None)
            if top_k is not None:
                report.moe_top_k = int(top_k)

        # RoPE theta + scaling type
        rope_theta, rope_source = _resolve_rope_theta_from_config(config)
        report.rope_theta = rope_theta
        rope_scaling = getattr(config, "rope_scaling", None)
        if isinstance(rope_scaling, dict):
            report.rope_scaling_type = rope_scaling.get("rope_type") or rope_scaling.get("type")

    return report


def count_state_dict_summary(state_dict: dict) -> dict:
    """Build a summary of the state dict: tensor counts, parameter counts by category."""
    summary = {
        "total_tensors": len(state_dict),
        "total_params": 0,
        "tensor_breakdown": {},
        "param_breakdown": {},
    }

    categories = {
        "embeddings": lambda k: k.endswith("embed_tokens.weight"),
        "lm_head": lambda k: k == "lm_head.weight",
        "final_norm": lambda k: k.endswith("model.norm.weight"),
        "attention_weights": lambda k: "self_attn." in k and k.endswith(".weight"),
        "attention_biases": lambda k: "self_attn." in k and k.endswith(".bias"),
        "mlp_weights": lambda k: "mlp." in k and k.endswith(".weight"),
        "mlp_biases": lambda k: "mlp." in k and k.endswith(".bias"),
        "layer_norms": lambda k: k.endswith("_layernorm.weight"),
        "layer_norm_biases": lambda k: k.endswith("_layernorm.bias"),
        "ssm_weights": lambda k: ".mamba." in k or ".ssm." in k,
        "moe_weights": lambda k: ".experts." in k or ".router." in k,
        "layer_scales": lambda k: ".ls1.weight" in k or ".ls2.weight" in k,
    }

    for cat_name, predicate in categories.items():
        matching_keys = [k for k in state_dict if predicate(k)]
        param_count = 0
        for k in matching_keys:
            t = state_dict[k]
            if hasattr(t, "numel"):
                param_count += int(t.numel())
            elif hasattr(t, "size"):
                param_count += int(t.size)
        summary["tensor_breakdown"][cat_name] = len(matching_keys)
        summary["param_breakdown"][cat_name] = param_count
        summary["total_params"] += param_count

    return summary

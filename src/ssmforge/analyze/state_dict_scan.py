"""Scan a HuggingFace model state dict for architectural quirks.

Detects features that affect downstream conversion/compatibility:
- attention_bias: presence of q/k/v/o_proj.bias
- tied_embeddings: lm_head.weight == embed_tokens.weight
- fused_qkv: single qkv_proj.weight instead of separate q/k/v_proj
- fused_gate_up: single gate_up_proj.weight instead of separate gate/up_proj
- moe: presence of expert/router/switch tensors
- grouped_attention: K/V projection outputs are smaller than Q (num_kv_heads < num_attention_heads)

Detection is heuristic — based on key names, presence/absence, and tensor
shapes. Does not require the model to be loaded into memory; works on the
state dict dict.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class QuirkReport:
    """Summary of architectural quirks detected in a state dict."""

    attention_bias: bool = False
    tied_embeddings: bool = False
    fused_qkv: bool = False
    fused_gate_up: bool = False
    moe: bool = False
    grouped_attention: bool = False

    # Specifics for transparency
    bias_keys_found: list[str] = field(default_factory=list)
    fused_keys_found: list[str] = field(default_factory=list)
    moe_keys_found: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _all_keys_with_prefix(state_dict: dict, prefix: str) -> list[str]:
    return [k for k in state_dict if k.startswith(prefix)]


def _layer_zero_keys(state_dict: dict, layer_prefix: str = "model.layers.0.") -> list[str]:
    """Get keys for layer 0 only (a single representative layer)."""
    return [k for k in state_dict if k.startswith(layer_prefix)]


def scan_state_dict(state_dict: dict, num_layers: int | None = None) -> QuirkReport:
    """Analyze a state dict and return a QuirkReport.

    Args:
        state_dict: model.state_dict() from a HuggingFace model.
        num_layers: explicit layer count (optional, inferred from state_dict if not given).

    Returns:
        QuirkReport describing detected quirks.
    """
    report = QuirkReport()

    if not state_dict:
        return report

    # Infer layer count if not given
    if num_layers is None:
        layer_indices = set()
        for k in state_dict:
            if k.startswith("model.layers."):
                rest = k[len("model.layers."):]
                if "." in rest:
                    idx = rest.split(".", 1)[0]
                    if idx.isdigit():
                        layer_indices.add(int(idx))
        num_layers = max(layer_indices) + 1 if layer_indices else 0

    if num_layers == 0:
        return report

    # ---- Attention bias ----
    # Sample layer 0 only — assume consistency across layers (true for all known HF models)
    layer0_keys = _layer_zero_keys(state_dict)
    bias_keys = [
        k for k in layer0_keys
        if "self_attn." in k and k.endswith(".bias")
    ]
    if bias_keys:
        report.attention_bias = True
        report.bias_keys_found = sorted(bias_keys)

    # ---- Fused QKV (Phi-3 uses qkv_proj instead of q/k/v_proj) ----
    fused_qkv_keys = [
        k for k in layer0_keys
        if "self_attn.qkv_proj.weight" in k
    ]
    if fused_qkv_keys:
        report.fused_qkv = True
        report.fused_keys_found.extend(sorted(fused_qkv_keys))

    # ---- Fused gate/up (Phi-3 fuses gate_proj + up_proj into gate_up_proj) ----
    fused_gate_up_keys = [
        k for k in layer0_keys
        if "mlp.gate_up_proj.weight" in k
    ]
    if fused_gate_up_keys:
        report.fused_gate_up = True
        report.fused_keys_found.extend(sorted(fused_gate_up_keys))

    # ---- Grouped attention (K/V smaller than Q) ----
    # Q: shape (n_heads * head_dim, hidden)
    # K: shape (n_kv_heads * head_dim, hidden)
    # If K/V projection's output dim is smaller than Q's, it's grouped.
    q_weight_key = None
    k_weight_key = None
    for k in layer0_keys:
        if k.endswith("self_attn.q_proj.weight"):
            q_weight_key = k
        elif k.endswith("self_attn.k_proj.weight"):
            k_weight_key = k

    if q_weight_key and k_weight_key:
        q_shape = state_dict[q_weight_key].shape
        k_shape = state_dict[k_weight_key].shape
        # Both should have shape[1] == hidden_size
        if len(q_shape) >= 2 and len(k_shape) >= 2 and q_shape[1] == k_shape[1]:
            if q_shape[0] != k_shape[0]:
                report.grouped_attention = True

    # ---- MoE (presence of expert / router / switch tensors) ----
    moe_indicators = ["expert", "router", "switch", "gate_proj"]
    moe_keys = [
        k for k in state_dict
        if any(ind in k.lower() for ind in moe_indicators)
        # Filter false positives: mlp.gate_proj is standard FFN, not MoE routing
        and not (
            "mlp.gate_proj" in k and "mlp.up_proj" in state_dict
        )
    ]
    # Stronger signal: explicit MoE keys like "block_sparse_moe" or ".experts."
    moe_keys = [
        k for k in state_dict
        if "block_sparse_moe" in k or ".experts." in k or ".router." in k
    ]
    if moe_keys:
        report.moe = True
        report.moe_keys_found = sorted(set(moe_keys))[:5]  # cap for readability

    # ---- Tied embeddings (lm_head.weight == embed_tokens.weight) ----
    # Detect by: either (a) lm_head missing entirely OR (b) lm_head is identical
    # tensor to embed_tokens (same data_ptr when both are torch tensors).
    embed_key = None
    head_key = None
    for k in state_dict:
        if k == "model.embed_tokens.weight":
            embed_key = k
        elif k == "lm_head.weight":
            head_key = k

    if embed_key is not None and head_key is None:
        # lm_head missing entirely — strong signal of tied embeddings
        report.tied_embeddings = True
    elif embed_key is not None and head_key is not None:
        # Check if they're the same tensor (sharing storage)
        e = state_dict[embed_key]
        h = state_dict[head_key]
        if hasattr(e, "data_ptr") and hasattr(h, "data_ptr"):
            if e.data_ptr() == h.data_ptr():
                report.tied_embeddings = True
        # Also check by values for numpy arrays
        elif hasattr(e, "ctypes") and hasattr(h, "ctypes"):
            if e.ctypes.data == h.ctypes.data:
                report.tied_embeddings = True
        # If we can't compare storage, fall back to identity
        elif id(e) == id(h):
            report.tied_embeddings = True

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
        "ssm_weights": lambda k: ".mamba." in k or ".ssm." in k,
        "moe_weights": lambda k: ".experts." in k or ".router." in k,
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
            else:
                param_count += 0
        summary["tensor_breakdown"][cat_name] = len(matching_keys)
        summary["param_breakdown"][cat_name] = param_count
        summary["total_params"] += param_count

    return summary

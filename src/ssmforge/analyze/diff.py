"""Diff two architecture reports.

Useful for:
- Comparing a fine-tuned model against its base
- Checking what changes a LoRA / merge introduces (the merge should have
  the same architecture, different weights)
- Comparing two candidate source models before a conversion experiment
"""

from __future__ import annotations

from typing import Any


def _flatten_quirks(report: dict) -> dict[str, Any]:
    """Flatten a report's quirks + config into a comparable dict."""
    q = report.get("quirks", {})
    cfg = report.get("config", {})
    return {
        "model_type": report.get("model_type"),
        "hidden_size": cfg.get("hidden_size"),
        "num_hidden_layers": cfg.get("num_hidden_layers"),
        "num_attention_heads": cfg.get("num_attention_heads"),
        "num_key_value_heads": cfg.get("num_key_value_heads"),
        "intermediate_size": cfg.get("intermediate_size"),
        "vocab_size": cfg.get("vocab_size"),
        "max_position_embeddings": cfg.get("max_position_embeddings"),
        "rope_theta": cfg.get("rope_theta"),
        "tie_word_embeddings": cfg.get("tie_word_embeddings"),
        "attention_bias": cfg.get("attention_bias"),
        "attention_bias_in_state_dict": q.get("attention_bias"),
        "tied_embeddings_in_state_dict": q.get("tied_embeddings"),
        "fused_qkv": q.get("fused_qkv"),
        "fused_gate_up": q.get("fused_gate_up"),
        "grouped_attention": q.get("grouped_attention"),
        "mqa": q.get("mqa"),
        "moe": q.get("moe"),
        "mlp_type": q.get("mlp_type"),
        "norm_type": q.get("norm_type"),
        "sliding_window": q.get("sliding_window"),
        "layer_scale": q.get("layer_scale"),
        "partial_rope_factor": q.get("partial_rope_factor"),
        "num_experts": q.get("num_experts"),
        "moe_top_k": q.get("moe_top_k"),
        "soft_capping_attn": q.get("soft_capping", {}).get("attn_logit_softcapping"),
        "soft_capping_final": q.get("soft_capping", {}).get("final_logit_softcapping"),
    }


def diff_reports(report_a: dict, report_b: dict) -> dict:
    """Compare two architecture reports and produce a diff.

    Returns a dict with:
        - 'identical': bool — True if architectures are functionally the same
        - 'differences': list of dicts with 'field', 'a', 'b' for each difference
        - 'added_quirks': list of quirks present in B but not A
        - 'removed_quirks': list of quirks present in A but not B
    """
    a = _flatten_quirks(report_a)
    b = _flatten_quirks(report_b)

    differences = []
    for field in sorted(set(a.keys()) | set(b.keys())):
        va = a.get(field)
        vb = b.get(field)
        if va != vb:
            differences.append({"field": field, "a": va, "b": vb})

    # Quirk-specific diff
    added_quirks = []
    removed_quirks = []
    for q in ("attention_bias_in_state_dict", "tied_embeddings_in_state_dict",
              "fused_qkv", "fused_gate_up", "grouped_attention", "mqa", "moe",
              "layer_scale"):
        va = a.get(q)
        vb = b.get(q)
        if not va and vb:
            added_quirks.append(q)
        elif va and not vb:
            removed_quirks.append(q)

    return {
        "identical": len(differences) == 0,
        "differences": differences,
        "added_quirks": added_quirks,
        "removed_quirks": removed_quirks,
    }

"""Build a JSON-friendly architecture report from an HF model.

Combines:
- Config fields (vocab_size, hidden_size, etc.)
- State dict quirks (biases, fused tensors, MoE, etc.)
- Config-based quirks (sliding window, soft capping, MoE config)
- Compatibility notes (what may break when converting to a Llama-shaped target)
- Interpretive profile (human-readable summary of what makes this model special)

The output is intentionally machine-readable AND human-readable — the JSON
is pretty-printed so it can be eyeballed in a terminal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any

from ssmforge.analyze.state_dict_scan import (
    QuirkReport,
    scan_state_dict,
    count_state_dict_summary,
)


@dataclass
class CompatibilityNote:
    """A single issue or warning about how the model interacts with our tooling."""

    severity: str  # "info" | "warning" | "error"
    quirk: str     # e.g. "attention_bias"
    message: str   # human-readable explanation


# Quirks that block conversion (we don't have converters for these yet).
_BLOCKING_QUIRKS = {"moe"}


def build_compatibility_notes(quirks: QuirkReport) -> list[CompatibilityNote]:
    """Generate human-readable notes about quirks and their compatibility status."""
    notes: list[CompatibilityNote] = []

    # ---- Attention bias ----
    if quirks.attention_bias:
        notes.append(CompatibilityNote(
            severity="info",
            quirk="attention_bias",
            message=(
                "Model has bias=True on attention q/k/v projections. "
                "Our loader builds models with attention_bias=True by default, "
                "so bias terms load correctly. No action needed."
            ),
        ))
    else:
        notes.append(CompatibilityNote(
            severity="info",
            quirk="attention_bias",
            message="No attention biases detected (Llama convention).",
        ))

    # ---- Tied embeddings ----
    if quirks.tied_embeddings:
        notes.append(CompatibilityNote(
            severity="info",
            quirk="tied_embeddings",
            message=(
                "Embeddings are tied to the LM head (Qwen2, some Pythia, etc.). "
                "Our Qwen2ToHybridConverter synthesizes lm_head.weight from "
                "embed_tokens.weight when missing. Loader handles either form."
            ),
        ))

    # ---- Fused QKV / gate_up ----
    if quirks.fused_qkv:
        notes.append(CompatibilityNote(
            severity="info",
            quirk="fused_qkv",
            message=(
                "Attention Q/K/V projections are fused into a single qkv_proj.weight "
                "(Phi-3 style). Our Phi3ToHybridConverter splits this into separate "
                "q/k/v_proj tensors during conversion."
            ),
        ))

    if quirks.fused_gate_up:
        notes.append(CompatibilityNote(
            severity="info",
            quirk="fused_gate_up",
            message=(
                "MLP gate_proj and up_proj are fused into a single gate_up_proj.weight "
                "(Phi-3 style). Our Phi3ToHybridConverter splits this into separate "
                "gate/up tensors."
            ),
        ))

    # ---- MoE ----
    if quirks.moe:
        n = f" with {quirks.num_experts} experts, top-{quirks.moe_top_k} routing" \
            if quirks.num_experts else ""
        notes.append(CompatibilityNote(
            severity="error",
            quirk="moe",
            message=(
                f"Mixture-of-Experts{n} detected (experts/router tensors). "
                "SSMForge does NOT support MoE architectures — conversion will "
                "fail or silently drop MoE layers. Not currently on the roadmap."
            ),
        ))

    # ---- Grouped attention / MQA ----
    if quirks.mqa:
        notes.append(CompatibilityNote(
            severity="info",
            quirk="mqa",
            message=(
                "Multi-Query Attention (kv_heads=1). K/V cache is shared across all "
                "Q heads. Memory-efficient at inference time but limits quality. "
                "Our loader detects this from config and constructs attention layers "
                "with num_key_value_heads=1."
            ),
        ))
    elif quirks.grouped_attention:
        notes.append(CompatibilityNote(
            severity="info",
            quirk="grouped_attention",
            message=(
                "K/V projection output dim is smaller than Q's (grouped query "
                "attention, common in Qwen2 / Llama-3 / Mistral). Our loader "
                "detects num_kv_heads from GGUF metadata and constructs "
                "attention layers with the correct KV head count."
            ),
        ))

    # ---- Sliding window ----
    if quirks.sliding_window:
        notes.append(CompatibilityNote(
            severity="info",
            quirk="sliding_window",
            message=(
                f"Sliding window attention (window={quirks.sliding_window}). "
                f"Attention only attends to the last {quirks.sliding_window} tokens, "
                "reducing memory at long context. Our forward pass uses dense attention "
                "(no window), so the model will have higher memory cost but same "
                "quality. To preserve windowed attention, custom kernel required."
            ),
        ))

    # ---- LayerScale ----
    if quirks.layer_scale:
        notes.append(CompatibilityNote(
            severity="info",
            quirk="layer_scale",
            message=(
                "LayerScale present (Phi-3 style learnable per-channel scale on residual). "
                "Our LlamaDecoderLayer doesn't have LayerScale. Weights would be "
                "silently dropped during load_state_dict (strict=False). For best "
                "quality, add LayerScale support to HybridLlamaMambaModel."
            ),
        ))

    # ---- Soft capping (Gemma2) ----
    if quirks.soft_capping:
        caps = quirks.soft_capping
        notes.append(CompatibilityNote(
            severity="info",
            quirk="soft_capping",
            message=(
                f"Logit soft-capping enabled (attn={caps.get('attn_logit_softcapping')}, "
                f"final={caps.get('final_logit_softcapping')}). Gemma2 specific. "
                "Our forward pass doesn't apply soft-capping, so quality will degrade. "
                "Add soft-capping to attention + final logits to fix."
            ),
        ))

    # ---- Partial RoPE (Command-R) ----
    if quirks.partial_rope_factor:
        notes.append(CompatibilityNote(
            severity="warning",
            quirk="partial_rope",
            message=(
                f"Partial RoPE (factor={quirks.partial_rope_factor}): only part of "
                "the head dim gets rotary, the rest is left untouched. Command-R specific. "
                "Our LlamaAttention applies full RoPE. Partial weights would still load "
                "but quality would degrade. Add partial RoPE support to fix."
            ),
        ))

    # ---- MLP type / norm type ----
    if quirks.mlp_type == "gelu":
        notes.append(CompatibilityNote(
            severity="warning",
            quirk="mlp_type",
            message=(
                "MLP uses GELU activation (not SwiGLU). GPT-2 / BERT / Falcon style. "
                "Our LlamaMLP uses SwiGLU. Conversion would replace GELU with SwiGLU, "
                "changing the activation function. Quality will degrade significantly."
            ),
        ))
    elif quirks.mlp_type == "unknown":
        notes.append(CompatibilityNote(
            severity="warning",
            quirk="mlp_type",
            message=(
                "MLP structure didn't match any known pattern (SwiGLU / GELU / GeGLU). "
                "Conversion may fail silently."
            ),
        ))

    if quirks.norm_type == "layer":
        notes.append(CompatibilityNote(
            severity="warning",
            quirk="norm_type",
            message=(
                "Layer normalization (with bias), not RMSNorm. GPT-2 / BERT style. "
                "Our LlamaRMSNorm has no bias. Conversion would drop the norm bias."
            ),
        ))

    return notes


def _format_rope_theta(config: Any, quirks: QuirkReport) -> tuple[float | None, str]:
    """Resolve rope_theta from config, returning (value, source_label)."""
    # First try the quirk report (which already resolved it)
    if quirks.rope_theta is not None:
        # Determine source for the label
        direct = getattr(config, "rope_theta", None) if config else None
        if direct is not None:
            return float(direct), "config.rope_theta"
        return quirks.rope_theta, "config.rope_scaling.rope_theta"
    return None, "defaulted"


def _format_attention_bias(config: Any, quirks: QuirkReport) -> bool:
    """Resolve attention_bias: trust config if explicit, else use state_dict truth."""
    config_says = getattr(config, "attention_bias", None) if config else None
    if config_says is not None:
        return bool(config_says)
    return quirks.attention_bias


def build_report(
    model_id: str,
    config: Any,
    state_dict: dict,
    num_layers: int | None = None,
) -> dict:
    """Build the full architecture report.

    Args:
        model_id: HuggingFace model id (e.g. "Qwen/Qwen2-1.5B-Instruct")
                  or local path. Used for display only.
        config: HuggingFace config object (PretrainedConfig).
        state_dict: model.state_dict() from the HF model.
        num_layers: explicit layer count (optional).

    Returns:
        Dict that can be json.dumps'd. Always includes 'quirks',
        'compatibility', 'state_dict_summary', and 'profile' sections.
    """
    quirks = scan_state_dict(state_dict, config=config, num_layers=num_layers)
    notes = build_compatibility_notes(quirks)
    summary = count_state_dict_summary(state_dict)

    def cfg(name, default=None):
        return getattr(config, name, default)

    # Cross-check config.attention_bias against the actual state_dict
    reported_attention_bias = _format_attention_bias(config, quirks)

    # Resolve rope_theta for reporting
    rope_theta_value, rope_theta_source = _format_rope_theta(config, quirks)

    # Build the report
    report = {
        "model_id": model_id,
        "model_type": cfg("model_type", "unknown"),
        "architectures": cfg("architectures", []),
        "config": {
            "vocab_size": cfg("vocab_size"),
            "hidden_size": cfg("hidden_size"),
            "intermediate_size": cfg("intermediate_size"),
            "num_hidden_layers": cfg("num_hidden_layers"),
            "num_attention_heads": cfg("num_attention_heads"),
            "num_key_value_heads": cfg("num_key_value_heads", cfg("num_attention_heads")),
            "max_position_embeddings": cfg("max_position_embeddings"),
            "rope_theta": rope_theta_value,
            "rope_theta_source": rope_theta_source,
            "rope_scaling_type": quirks.rope_scaling_type,
            "rms_norm_eps": cfg("rms_norm_eps"),
            "tie_word_embeddings": cfg("tie_word_embeddings", False),
            "attention_bias": reported_attention_bias,
            "torch_dtype": str(cfg("torch_dtype", "unknown")),
        },
        "quirks": quirks.to_dict(),
        "state_dict_summary": summary,
        "compatibility": {
            "issues": [asdict(n) for n in notes],
            "is_compatible": all(n.severity != "error" for n in notes),
            "blockers": [n.quirk for n in notes if n.severity == "error"],
            "warnings": [n.quirk for n in notes if n.severity == "warning"],
        },
        "profile": _build_profile(quirks, cfg, model_id),
    }

    return report


def _build_profile(quirks: QuirkReport, cfg, model_id: str) -> dict:
    """Build an interpretive profile — what kind of model is this?

    This is the "interpretative" section: not just data, but a human-readable
    summary of what makes this model unique.
    """
    profile = {
        "family": _infer_family(quirks, cfg, model_id),
        "attention_type": _infer_attention_type(quirks, cfg),
        "mlp_type": quirks.mlp_type,
        "norm_type": quirks.norm_type,
    }

    # Add narrative descriptors based on detected quirks
    descriptors: list[str] = []
    if quirks.mqa:
        descriptors.append("multi-query attention")
    elif quirks.grouped_attention:
        n_q = getattr(cfg, "num_attention_heads", None)
        n_kv = getattr(cfg, "num_key_value_heads", None)
        if n_q and n_kv:
            descriptors.append(f"GQA ({n_kv} KV heads / {n_q} Q heads, {n_q // n_kv}x compression)")

    if quirks.sliding_window:
        descriptors.append(f"sliding window={quirks.sliding_window}")

    if quirks.fused_qkv:
        descriptors.append("fused QKV")
    if quirks.fused_gate_up:
        descriptors.append("fused gate+up MLP")

    if quirks.tied_embeddings:
        descriptors.append("tied embeddings")

    if quirks.attention_bias:
        descriptors.append("attention biases")

    if quirks.layer_scale:
        descriptors.append("LayerScale residual")

    if quirks.soft_capping:
        descriptors.append("soft-capped logits")

    if quirks.partial_rope_factor:
        descriptors.append(f"partial RoPE (factor={quirks.partial_rope_factor})")

    if quirks.moe:
        n = quirks.num_experts or "?"
        k = quirks.moe_top_k or "?"
        descriptors.append(f"MoE ({n} experts, top-{k})")

    profile["descriptors"] = descriptors
    return profile


def _infer_family(quirks: QuirkReport, cfg, model_id: str) -> str:
    """Infer the architecture family from quirks + model_id."""
    mid = (model_id or "").lower()
    arch = (getattr(cfg, "model_type", "") or "").lower()

    # MoE models
    if quirks.moe:
        if "mixtral" in mid:
            return "Mixtral (MoE Llama)"
        return f"MoE ({arch or 'unknown'})"

    # Fused QKV → Phi-3
    if quirks.fused_qkv:
        return "Phi-3 (fused QKV + fused gate/up)"

    # Sliding window → Mistral / Gemma2
    if quirks.sliding_window and "gemma" in arch:
        return f"Gemma2 ({arch})"
    if quirks.sliding_window:
        return "Mistral (sliding window + GQA)"

    # Soft capping → Gemma2
    if quirks.soft_capping:
        return "Gemma2 (soft-capped logits)"

    # LayerScale → Phi-2/Phi-3 (but fused_qkv already caught Phi-3)
    if quirks.layer_scale:
        return "Phi (LayerScale)"

    # Tied embeddings + attention bias → Qwen2
    if quirks.tied_embeddings and quirks.attention_bias:
        return "Qwen2 (tied embeddings + attention bias)"

    # MQA → Falcon / Pythia
    if quirks.mqa:
        return "Falcon / Pythia (MQA)"

    # Partial RoPE → Command-R
    if quirks.partial_rope_factor:
        return "Command-R (partial RoPE)"

    # GPT-2 / BERT pattern (LayerNorm + GELU)
    if quirks.norm_type == "layer" and quirks.mlp_type == "gelu":
        return "GPT-2 / BERT style (LayerNorm + GELU)"

    # Default: dense transformer
    if quirks.grouped_attention:
        return "Llama-style dense (GQA)"
    return "Llama-style dense"


def _infer_attention_type(quirks: QuirkReport, cfg) -> str:
    """Infer the attention variant: MHA, GQA, MQA, or fused."""
    if quirks.fused_qkv:
        return "fused"
    if quirks.mqa:
        return "MQA (kv_heads=1)"
    if quirks.grouped_attention:
        return "GQA"
    return "MHA"


def format_report_json(report: dict, indent: int = 2) -> str:
    """Render a report as pretty-printed JSON."""
    return json.dumps(report, indent=indent, default=str)

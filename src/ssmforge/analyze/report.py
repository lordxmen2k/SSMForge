"""Build a JSON-friendly architecture report from an HF model.

Combines:
- Config fields (vocab_size, hidden_size, etc.)
- State dict quirks (biases, fused tensors, MoE, etc.)
- Compatibility notes (what may break when converting to a Llama-shaped target)

The output is intentionally machine-readable AND human-readable — the JSON
is pretty-printed so it can be eyeballed in a terminal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any

from ssmforge.analyze.state_dict_scan import QuirkReport, scan_state_dict, count_state_dict_summary


@dataclass
class CompatibilityNote:
    """A single issue or warning about how the model interacts with our tooling."""

    severity: str  # "info" | "warning" | "error"
    quirk: str     # e.g. "attention_bias"
    message: str   # human-readable explanation


# Architectural quirks that we know how to handle, mapped to a flag indicating
# whether ssmforge's loader/converter is currently compatible.
_KNOWN_COMPATIBLE_QUIRKS = {
    "attention_bias": True,   # loader uses attention_bias=True by default
    "tied_embeddings": True,  # Qwen2ToHybridConverter ties them
    "fused_qkv": True,        # Phi3ToHybridConverter splits them
    "fused_gate_up": True,    # Phi3ToHybridConverter splits them
    "moe": False,             # no MoE converter yet
    "grouped_attention": True, # loader detects n_kv_heads from GGUF metadata
}


def build_compatibility_notes(quirks: QuirkReport) -> list[CompatibilityNote]:
    """Generate human-readable notes about quirks and their compatibility status.

    These are exactly the kinds of issues that silently broke our pure-attention
    smoke test on Qwen2-1.5B — the goal of this report is to surface them
    BEFORE running a 30-minute conversion.
    """
    notes: list[CompatibilityNote] = []

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

    if quirks.moe:
        notes.append(CompatibilityNote(
            severity="error",
            quirk="moe",
            message=(
                "Mixture-of-Experts tensors detected (experts/router). "
                "SSMForge does NOT support MoE architectures — conversion will "
                "fail or silently drop MoE layers. Not currently on the roadmap."
            ),
        ))

    if quirks.grouped_attention:
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

    return notes


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
        'compatibility', and 'state_dict_summary' sections.
    """
    quirks = scan_state_dict(state_dict, num_layers=num_layers)
    notes = build_compatibility_notes(quirks)
    summary = count_state_dict_summary(state_dict)

    # Extract config fields. Use getattr with defaults so missing fields
    # don't crash the report.
    def cfg(name, default=None):
        return getattr(config, name, default)

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
            "rope_theta": cfg("rope_theta"),
            "rms_norm_eps": cfg("rms_norm_eps"),
            "tie_word_embeddings": cfg("tie_word_embeddings", False),
            "attention_bias": cfg("attention_bias", False),
            "torch_dtype": str(cfg("torch_dtype", "unknown")),
        },
        "quirks": quirks.to_dict(),
        "state_dict_summary": summary,
        "compatibility": {
            "issues": [asdict(n) for n in notes],
            "is_compatible": all(n.severity != "error" for n in notes),
        },
    }

    return report


def format_report_json(report: dict, indent: int = 2) -> str:
    """Render a report as pretty-printed JSON."""
    return json.dumps(report, indent=indent, default=str)

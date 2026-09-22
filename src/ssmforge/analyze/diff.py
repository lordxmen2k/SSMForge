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


def compare_reports(reports: list[dict]) -> dict:
    """Compare N architecture reports side-by-side.

    Args:
        reports: list of report dicts, each with 'model_id' field.

    Returns a dict with:
        - 'models': list of model ids in the same order
        - 'fields': list of dicts {field, values, all_same, unique_values}
                    one per field that varies across the N models
        - 'all_identical': True if every field has the same value across models
        - 'identical_field_count': number of fields with identical values
        - 'different_field_count': number of fields with at least one difference

    Useful for:
    - Comparing a fine-tuned model, its base, and a competitor side by side
    - Spotting field-level inconsistencies across 3+ checkpoints of the same
      model
    - Generating a comparison table for a paper or wiki
    """
    if len(reports) < 2:
        raise ValueError(
            f"compare_reports requires at least 2 reports, got {len(reports)}"
        )

    model_ids = [r.get("model_id", f"model_{i}") for i, r in enumerate(reports)]
    flat_reports = [_flatten_quirks(r) for r in reports]

    # Collect all fields across all reports (union)
    all_fields = set()
    for fr in flat_reports:
        all_fields.update(fr.keys())

    fields = []
    identical_count = 0
    for field in sorted(all_fields):
        values = {model_ids[i]: fr.get(field) for i, fr in enumerate(flat_reports)}
        unique_values = list(dict.fromkeys(values.values()))  # preserve order
        all_same = len(unique_values) == 1
        if all_same:
            identical_count += 1
        fields.append({
            "field": field,
            "values": values,
            "all_same": all_same,
            "unique_values": unique_values,
        })

    return {
        "models": model_ids,
        "fields": fields,
        "all_identical": identical_count == len(fields),
        "identical_field_count": identical_count,
        "different_field_count": len(fields) - identical_count,
    }


def format_compare_markdown(comparison: dict) -> str:
    """Render an N-way comparison as a Markdown table."""
    models = comparison["models"]
    fields = comparison["fields"]

    lines = []
    lines.append(f"# Architectural comparison: {len(models)} models")
    lines.append("")

    if comparison["all_identical"]:
        lines.append("**All identical** — every field matches across all models.")
        lines.append("")
        lines.append(f"Models compared: {', '.join(f'`{m}`' for m in models)}")
        return "\n".join(lines)

    lines.append(
        f"**{comparison['different_field_count']} field(s) differ** "
        f"out of {len(fields)}; "
        f"{comparison['identical_field_count']} identical."
    )
    lines.append("")

    # Build header row
    header = "| Field | " + " | ".join(models) + " |"
    sep = "|-------|" + "|".join(["-------"] * len(models)) + "|"
    lines.append(header)
    lines.append(sep)

    # Body rows
    for field in fields:
        if field["all_same"]:
            continue  # skip identical rows to keep the table scannable
        vals = [str(field["values"][m]) for m in models]
        lines.append(f"| `{field['field']}` | " + " | ".join(f"`{v}`" for v in vals) + " |")

    # Section listing the identical fields for completeness
    identical_fields = [f["field"] for f in fields if f["all_same"]]
    if identical_fields:
        lines.append("")
        lines.append(f"### Identical across all models")
        lines.append("")
        lines.append(f"{len(identical_fields)} field(s) had the same value: "
                     + ", ".join(f"`{f}`" for f in identical_fields[:20]))
        if len(identical_fields) > 20:
            lines.append(f"  (and {len(identical_fields) - 20} more)")

    return "\n".join(lines)

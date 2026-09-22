"""Render an architecture report as a human-readable summary.

This is the stderr output for `ssmforge arch` — the JSON goes to stdout,
and the human-readable summary goes to stderr so the JSON can be piped
to `jq` or saved to a file cleanly.
"""

from __future__ import annotations

from typing import Any


def _humanize_param_count(n: int) -> str:
    """Convert a parameter count to human form (e.g. 1543714304 -> 1.54B)."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    elif n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.2f}K"
    return str(n)


def render_summary(report: dict) -> str:
    """Build a human-readable summary string for stderr output."""
    lines = []
    lines.append(f"\nSSMForge arch report: {report['model_id']}")

    # Profile (interpretive section)
    profile = report.get("profile", {})
    if profile:
        family = profile.get("family", "unknown")
        lines.append(f"  family:              {family}")
        if profile.get("attention_type"):
            lines.append(f"  attention_type:      {profile['attention_type']}")
        if profile.get("mlp_type") and profile["mlp_type"] != "unknown":
            lines.append(f"  mlp_type:            {profile['mlp_type']}")
        if profile.get("norm_type") and profile["norm_type"] != "unknown":
            lines.append(f"  norm_type:           {profile['norm_type']}")
        if profile.get("descriptors"):
            descriptors = ", ".join(profile["descriptors"])
            lines.append(f"  descriptors:         {descriptors}")

    lines.append("")
    lines.append("Model config:")
    cfg = report["config"]
    lines.append(f"  model_type:          {report['model_type']}")
    lines.append(f"  hidden_size:         {cfg.get('hidden_size')}")
    lines.append(f"  num_hidden_layers:   {cfg.get('num_hidden_layers')}")
    lines.append(f"  num_attention_heads: {cfg.get('num_attention_heads')}")
    lines.append(f"  num_kv_heads:        {cfg.get('num_key_value_heads')}")
    lines.append(f"  intermediate_size:   {cfg.get('intermediate_size')}")
    lines.append(f"  vocab_size:          {cfg.get('vocab_size')}")
    lines.append(f"  max_position:        {cfg.get('max_position_embeddings')}")
    if cfg.get("rope_theta"):
        rope_str = f"{cfg['rope_theta']:g}"
        rope_src = cfg.get("rope_theta_source", "")
        if rope_src and rope_src != "config.rope_theta":
            rope_str += f" (from {rope_src})"
        lines.append(f"  rope_theta:          {rope_str}")
    if cfg.get("rope_scaling_type"):
        lines.append(f"  rope_scaling_type:   {cfg['rope_scaling_type']}")
    lines.append(f"  tie_word_embeddings: {cfg.get('tie_word_embeddings')}")
    lines.append(f"  attention_bias:      {cfg.get('attention_bias')}")

    q = report["quirks"]
    lines.append("")
    lines.append("Detected quirks:")
    quirk_lines = [
        ("attention_bias", "attention bias=True"),
        ("tied_embeddings", "tied embeddings"),
        ("fused_qkv", "fused QKV"),
        ("fused_gate_up", "fused gate/up"),
        ("grouped_attention", "GQA"),
        ("mqa", "MQA (kv_heads=1)"),
        ("moe", "MoE"),
    ]
    for key, label in quirk_lines:
        marker = "✓" if q.get(key) else "✗"
        lines.append(f"  {marker} {label}")
    if q.get("mlp_type") and q["mlp_type"] != "unknown":
        lines.append(f"    mlp: {q['mlp_type']}")
    if q.get("norm_type") and q["norm_type"] != "unknown":
        lines.append(f"    norm: {q['norm_type']}")
    if q.get("sliding_window"):
        lines.append(f"    sliding_window: {q['sliding_window']}")
    if q.get("layer_scale"):
        lines.append(f"    layer_scale: True")
    if q.get("soft_capping"):
        caps = q["soft_capping"]
        cap_str = ", ".join(f"{k}={v}" for k, v in caps.items())
        lines.append(f"    soft_capping: {cap_str}")
    if q.get("partial_rope_factor"):
        lines.append(f"    partial_rope_factor: {q['partial_rope_factor']}")
    if q.get("num_experts"):
        lines.append(f"    num_experts: {q['num_experts']} (top-{q.get('moe_top_k', '?')})")

    summary = report["state_dict_summary"]
    lines.append("")
    lines.append(f"State dict: {summary['total_tensors']} tensors, "
                 f"{_humanize_param_count(summary['total_params'])} params")
    for cat, count in summary["tensor_breakdown"].items():
        if count == 0:
            continue
        param_str = _humanize_param_count(summary["param_breakdown"].get(cat, 0))
        lines.append(f"  {cat:25s} {count:>4} tensors, {param_str:>8} params")

    # Compatibility
    compat = report["compatibility"]
    lines.append("")
    if compat["is_compatible"]:
        lines.append("Compatibility: OK (no blockers)")
        if compat.get("warnings"):
            lines.append(f"  Warnings: {len(compat['warnings'])} (see issues below)")
    else:
        blockers = compat.get("blockers", [])
        lines.append(f"Compatibility: BLOCKED by: {', '.join(blockers)}")
    for issue in compat["issues"]:
        sev = issue["severity"].upper()
        lines.append(f"  [{sev}] {issue['quirk']}: {issue['message']}")
    lines.append("")
    return "\n".join(lines)

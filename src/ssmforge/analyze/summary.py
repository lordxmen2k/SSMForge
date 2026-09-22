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
    lines.append(f"  model_type:           {report['model_type']}")
    cfg = report["config"]
    lines.append(f"  hidden_size:          {cfg.get('hidden_size')}")
    lines.append(f"  num_hidden_layers:    {cfg.get('num_hidden_layers')}")
    lines.append(f"  num_attention_heads:  {cfg.get('num_attention_heads')}")
    lines.append(f"  num_kv_heads:         {cfg.get('num_key_value_heads')}")
    lines.append(f"  intermediate_size:    {cfg.get('intermediate_size')}")
    lines.append(f"  vocab_size:           {cfg.get('vocab_size')}")
    lines.append(f"  max_position:         {cfg.get('max_position_embeddings')}")
    lines.append(f"  tie_word_embeddings:  {cfg.get('tie_word_embeddings')}")
    lines.append(f"  attention_bias:       {cfg.get('attention_bias')}")

    q = report["quirks"]
    lines.append("")
    lines.append("Detected quirks:")
    lines.append(f"  attention_bias:       {q['attention_bias']}")
    lines.append(f"  tied_embeddings:      {q['tied_embeddings']}")
    lines.append(f"  fused_qkv:            {q['fused_qkv']}")
    lines.append(f"  fused_gate_up:        {q['fused_gate_up']}")
    lines.append(f"  grouped_attention:    {q['grouped_attention']}")
    lines.append(f"  moe:                  {q['moe']}")

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
    else:
        lines.append("Compatibility: BLOCKED (see issues below)")
    for issue in compat["issues"]:
        sev = issue["severity"].upper()
        lines.append(f"  [{sev}] {issue['quirk']}: {issue['message']}")
    lines.append("")
    return "\n".join(lines)

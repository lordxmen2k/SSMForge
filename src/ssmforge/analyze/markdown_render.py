"""Render an architecture report as Markdown.

Designed to be pasted into GitHub issues, READMEs, wikis, etc.
Produces a single Markdown document with sections:
- Title and metadata
- Profile (family, attention type)
- Quirk checklist
- Compatibility summary
- Tensor breakdown
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


def _check(x: bool) -> str:
    return "x" if x else " "


def _render_severity_badge(severity: str) -> str:
    """Render severity as a GitHub-flavored badge."""
    s = severity.upper()
    badges = {
        "INFO": "`INFO`",
        "WARN": "`WARN`",
        "WARNING": "`WARN`",
        "ERROR": "`ERROR`",
    }
    return badges.get(s, f"`{s}`")


def format_report_markdown(report: dict) -> str:
    """Render a full architecture report as a Markdown document."""
    md = []
    md.append(f"# `ssmforge arch` — {report['model_id']}")
    md.append("")

    # Profile
    profile = report.get("profile", {})
    if profile:
        md.append("## Profile")
        md.append("")
        family = profile.get("family", "unknown")
        md.append(f"- **Family:** `{family}`")
        if profile.get("attention_type"):
            md.append(f"- **Attention:** `{profile['attention_type']}`")
        if profile.get("mlp_type") and profile["mlp_type"] != "unknown":
            md.append(f"- **MLP:** `{profile['mlp_type']}`")
        if profile.get("norm_type") and profile["norm_type"] != "unknown":
            md.append(f"- **Norm:** `{profile['norm_type']}`")
        if profile.get("descriptors"):
            md.append("- **Descriptors:**")
            for d in profile["descriptors"]:
                md.append(f"  - {d}")
        md.append("")

    # Config
    md.append("## Model config")
    md.append("")
    cfg = report["config"]
    md.append("| Field | Value |")
    md.append("|-------|-------|")
    md.append(f"| model_type | `{report['model_type']}` |")
    md.append(f"| hidden_size | `{cfg.get('hidden_size')}` |")
    md.append(f"| num_hidden_layers | `{cfg.get('num_hidden_layers')}` |")
    md.append(f"| num_attention_heads | `{cfg.get('num_attention_heads')}` |")
    md.append(f"| num_key_value_heads | `{cfg.get('num_key_value_heads')}` |")
    md.append(f"| intermediate_size | `{cfg.get('intermediate_size')}` |")
    md.append(f"| vocab_size | `{cfg.get('vocab_size')}` |")
    md.append(f"| max_position_embeddings | `{cfg.get('max_position_embeddings')}` |")
    if cfg.get("rope_theta") is not None:
        rope_str = f"`{cfg['rope_theta']:g}`"
        src = cfg.get("rope_theta_source", "")
        if src and src != "config.rope_theta":
            rope_str += f" (from `{src}`)"
        md.append(f"| rope_theta | {rope_str} |")
    if cfg.get("rope_scaling_type"):
        md.append(f"| rope_scaling_type | `{cfg['rope_scaling_type']}` |")
    md.append(f"| tie_word_embeddings | `{cfg.get('tie_word_embeddings')}` |")
    md.append(f"| attention_bias | `{cfg.get('attention_bias')}` |")
    if cfg.get("torch_dtype"):
        md.append(f"| torch_dtype | `{cfg['torch_dtype']}` |")
    md.append("")

    # Quirks checklist
    md.append("## Detected quirks")
    md.append("")
    md.append("| Quirk | Detected |")
    md.append("|-------|----------|")
    q = report["quirks"]
    md.append(f"| attention bias=True | {_check(q.get('attention_bias'))} |")
    md.append(f"| tied embeddings | {_check(q.get('tied_embeddings'))} |")
    md.append(f"| fused QKV | {_check(q.get('fused_qkv'))} |")
    md.append(f"| fused gate/up | {_check(q.get('fused_gate_up'))} |")
    md.append(f"| grouped attention (GQA) | {_check(q.get('grouped_attention'))} |")
    md.append(f"| multi-query attention (MQA) | {_check(q.get('mqa'))} |")
    md.append(f"| Mixture-of-Experts (MoE) | {_check(q.get('moe'))} |")
    md.append("")

    extras = []
    if q.get("mlp_type") and q["mlp_type"] != "unknown":
        extras.append(f"- **MLP type:** `{q['mlp_type']}`")
    if q.get("norm_type") and q["norm_type"] != "unknown":
        extras.append(f"- **Norm type:** `{q['norm_type']}`")
    if q.get("sliding_window"):
        extras.append(f"- **Sliding window:** `{q['sliding_window']}`")
    if q.get("layer_scale"):
        extras.append("- **LayerScale residual** (Phi-3 style)")
    if q.get("soft_capping"):
        caps = q["soft_capping"]
        cap_str = ", ".join(f"`{k}`={v}" for k, v in caps.items())
        extras.append(f"- **Soft capping:** {cap_str}")
    if q.get("partial_rope_factor"):
        extras.append(f"- **Partial RoPE factor:** `{q['partial_rope_factor']}`")
    if q.get("num_experts"):
        extras.append(f"- **MoE experts:** `{q['num_experts']}`, top-`{q.get('moe_top_k', '?')}` routing")
    if extras:
        md.append("### Additional")
        md.append("")
        md.extend(extras)
        md.append("")

    # Compatibility
    compat = report["compatibility"]
    md.append("## Compatibility")
    md.append("")
    if compat["is_compatible"]:
        md.append(f"**OK** — no blockers")
    else:
        blockers = compat.get("blockers", [])
        md.append(f"**BLOCKED** by: {', '.join(f'`{b}`' for b in blockers)}")
    md.append("")
    if compat.get("warnings"):
        md.append(f"_{len(compat['warnings'])} warning(s) — see issues below_")
        md.append("")
    md.append("| Severity | Quirk | Message |")
    md.append("|----------|-------|---------|")
    for issue in compat["issues"]:
        sev = _render_severity_badge(issue["severity"])
        quirk = issue["quirk"]
        msg = issue["message"].replace("|", "\\|").replace("\n", " ")
        md.append(f"| {sev} | `{quirk}` | {msg} |")
    md.append("")

    # State dict
    summary = report["state_dict_summary"]
    md.append("## State dict")
    md.append("")
    md.append(f"**{summary['total_tensors']} tensors, {_humanize_param_count(summary['total_params'])} parameters**")
    md.append("")
    md.append("| Category | Tensors | Parameters |")
    md.append("|----------|--------:|----------:|")
    for cat, count in summary["tensor_breakdown"].items():
        if count == 0:
            continue
        param_str = _humanize_param_count(summary["param_breakdown"].get(cat, 0))
        md.append(f"| {cat} | {count} | {param_str} |")
    md.append("")

    return "\n".join(md)

"""Text-based architectural graph rendering for ``ssmforge arch --graph``.

Two output shapes are supported:

- ``render_graph_text(report)`` — single-model decision-tree style graph
  showing the flow from input → embed → layer (attn + MLP + norms) → final norm
  → lm_head → logits, with explicit decisions (fused QKV? GQA? attention_bias?
  rope_theta? MoE? tied embeddings?) marked inline.

- ``render_graph_compare(reports)`` — side-by-side decision comparison across
  N models, showing how each model answers the same architectural questions.

Both renderers are pure functions over report dicts, so they work with the
same data path as JSON / Markdown — no extra model loading is required.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_dtype(dtype: Any) -> str:
    """Normalize torch dtype string to a short, human-readable form.

    `str(torch.bfloat16)` returns `"torch.bfloat16"`. We just want `"bf16"`.
    """
    s = str(dtype).lower()
    if "bfloat16" in s or "bf16" in s:
        return "bf16"
    if "float16" in s or "fp16" in s or "half" in s:
        return "fp16"
    if "float32" in s or "fp32" in s:
        return "fp32"
    return str(dtype)


def _fmt_int(n: int | None) -> str:
    """Format an int with thousands separators; '<?' for None/negative."""
    if n is None or n < 0:
        return "?"
    return f"{n:,}"


def _fmt_params(n: int | None) -> str:
    """Format a parameter count as a human-readable size (494M, 1.1B, etc.)."""
    if n is None or n < 0:
        return "?"
    if n >= 1_000_000_000:
        v = n / 1_000_000_000
        return f"{v:.1f}B" if v < 10 else f"{v:.0f}B"
    if n >= 1_000_000:
        v = n / 1_000_000
        return f"{v:.0f}M" if v >= 10 else f"{v:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _fmt_bytes(n: int | None) -> str:
    """Format a byte count as human-readable (1.5 GB, 800 MB, etc.).

    Uses base-1024 (GiB-style) since model sizes are reported in those
    units in HF / vLLM / most ML tooling.
    """
    if n is None or n < 0:
        return "?"
    units = [
        (1 << 40, "TB"),
        (1 << 30, "GB"),
        (1 << 20, "MB"),
        (1 << 10, "KB"),
    ]
    for div, label in units:
        if n >= div:
            v = n / div
            return f"{v:.1f} {label}" if v < 10 else f"{v:.0f} {label}"
    return f"{n} B"


def _memory_alt(bytes_per_param: int, base_bytes: int, dtype_label: str) -> str:
    """Build a memory estimate line at a different dtype."""
    if base_bytes is None or base_bytes <= 0:
        return ""
    scaled = int(base_bytes * bytes_per_param / 2)  # base is bf16/fp16 = 2 bytes
    return f"{scaled / (1 << 30):.1f} GB @ {dtype_label}"


# ---------------------------------------------------------------------------
# Decision probes
# ---------------------------------------------------------------------------

def _attn_decisions(quirks: dict, config: dict) -> list[tuple[str, str, str]]:
    """Return (key, value, note) tuples for attention-related decisions."""
    out = []
    out.append((
        "fused QKV",
        "YES" if quirks.get("fused_qkv") else "NO",
        "one combined qkv_proj" if quirks.get("fused_qkv") else "separate q, k, v projections",
    ))
    gqa = quirks.get("grouped_attention", False)
    mqa = quirks.get("mqa", False)
    num_kv = config.get("num_key_value_heads")
    num_q = config.get("num_attention_heads")
    if mqa:
        out.append(("MQA", "YES", "all Q heads share 1 KV head"))
    elif gqa and num_kv and num_q:
        out.append((
            "GQA",
            f"YES (kv:{num_kv} of {num_q})",
            f"{num_kv} KV heads vs {num_q} Q heads",
        ))
    else:
        out.append(("MQA/GQA", "NO", "standard multi-head attention"))
    out.append((
        "attention bias",
        "YES" if config.get("attention_bias") else "NO",
        "Q/K/V/O projections have bias terms" if config.get("attention_bias") else "no bias terms (Llama convention)",
    ))
    rope_theta = config.get("rope_theta")
    if rope_theta:
        if rope_theta >= 1_000_000:
            theta_str = f"{rope_theta:.0e}"
        else:
            theta_str = str(rope_theta)
        out.append(("rope_theta", theta_str, "rotary position embedding base"))
    rope_scaling = config.get("rope_scaling_type")
    if rope_scaling and rope_scaling != "default":
        out.append(("rope_scaling", rope_scaling, "extended context scaling"))
    sliding = quirks.get("sliding_window")
    if sliding and sliding != "none":
        out.append(("sliding_window", str(sliding), f"attention window size: {sliding}"))
    return out


def _mlp_decisions(quirks: dict, config: dict, dry_run: bool = False) -> list[tuple[str, str, str]]:
    """Return (key, value, note) tuples for MLP-related decisions."""
    out = []
    mlp_type = quirks.get("mlp_type", "unknown")
    if mlp_type == "swiglu":
        out.append(("MLP type", "SwiGLU", "gate + up + down_proj, silu(gate) * up"))
    elif mlp_type == "geglu":
        out.append(("MLP type", "GeGLU", "gate + up + down_proj, gelu(gate) * up"))
    elif mlp_type == "gelu":
        out.append(("MLP type", "GELU", "single up_proj, gelu(x * up)"))
    elif mlp_type == "moe":
        out.append(("MLP type", "Mixture-of-Experts", f"{quirks.get('num_experts', '?')} experts, top-{quirks.get('moe_top_k', '?')}"))
    else:
        note = (
            "requires --full inspection (config-only scan can't classify)"
            if dry_run else
            "MLP structure didn't match a known pattern"
        )
        out.append(("MLP type", "unknown", note))
    out.append((
        "fused gate/up",
        "YES" if quirks.get("fused_gate_up") else "NO",
        "one combined gate_up_proj" if quirks.get("fused_gate_up") else "separate gate_proj and up_proj",
    ))
    return out


def _norm_decisions(quirks: dict, dry_run: bool = False) -> list[tuple[str, str, str]]:
    """Return (key, value, note) tuples for normalization decisions."""
    out = []
    norm = quirks.get("norm_type", "unknown")
    if norm == "unknown":
        note = (
            "requires --full inspection (config-only scan can't classify)"
            if dry_run else
            "applied before attention and MLP"
        )
        out.append(("Norm", "unknown", note))
    else:
        out.append(("Norm", norm, "applied before attention and MLP"))
    return out


def _global_decisions(config: dict, quirks: dict) -> list[tuple[str, str, str]]:
    """Return model-wide decisions that don't belong to a specific sub-layer."""
    out = []
    tied = config.get("tie_word_embeddings", False)
    out.append((
        "Tied embeddings",
        "YES" if tied else "NO",
        "lm_head shares weights with embed_tokens" if tied else "lm_head and embed_tokens are independent",
    ))
    return out


# ---------------------------------------------------------------------------
# Single-model graph
# ---------------------------------------------------------------------------

def render_graph_text(report: dict[str, Any]) -> str:
    """Render a single-model architecture decision graph as ASCII art.

    The output is the "what's the flow and what are the choices" view that
    someone evaluating a model for conversion needs. Decision points (fused
    QKV? GQA? MoE? tied embeddings?) are listed with their verdict right next
    to where they matter in the pipeline.
    """
    model_id = report.get("model_id", "<unknown>")
    config = report.get("config", {})
    quirks = report.get("quirks", {})
    mem = report.get("memory_estimate", {})
    dry_run = bool(report.get("dry_run", False))

    layers = config.get("num_hidden_layers", "?")
    hidden = config.get("hidden_size", "?")
    n_heads = config.get("num_attention_heads", "?")
    n_kv = config.get("num_key_value_heads", n_heads)
    inter = config.get("intermediate_size", "?")
    vocab = config.get("vocab_size", "?")
    dtype = _clean_dtype(config.get("torch_dtype", "unknown"))
    params = mem.get("estimated_params")
    bytes_est = mem.get("estimated_bytes")

    head_line = (
        f"{model_id}    ({layers} layers, hidden={hidden}, "
        f"heads={n_heads} kv:{n_kv}, "
        f"ffn={inter}, vocab={vocab}, dtype={dtype})"
    )

    lines: list[str] = []
    lines.append(head_line)
    lines.append("")
    lines.append("  INPUT")
    lines.append("    │")
    tied = config.get("tie_word_embeddings", False)
    if tied:
        lines.append("  embed_tokens ─── lm_head ─→ TIED ✓")
    else:
        lines.append("  embed_tokens ─── lm_head ─→ independent")
    lines.append("    │")
    lines.append("    ├─  ┌──────────────────────────────────────────────────────────┐")
    lines.append(f"    │   │ ATTENTION (one of {layers} layers)                          │")
    lines.append("    │   │")
    lines.append("    ▼   │ pre-attention norm")
    lines.append("    │   │   │")
    lines.append("    │   │   ▼")

    for key, value, note in _attn_decisions(quirks, config):
        lines.append(f"    │   │   ├─ {key:<18} ──── {value:<24}   # {note}")

    lines.append("    │   │")
    lines.append("    │   │   └─→ output projection (o_proj)")
    lines.append("    ▼   │      │")
    lines.append("    │   │      └─ (residual add)")
    lines.append("    │   │")
    lines.append("    │   │ pre-mlp norm")
    lines.append("    │   │   │")
    lines.append("    │   │   ▼")

    for key, value, note in _mlp_decisions(quirks, config, dry_run=dry_run):
        lines.append(f"    │   │   ├─ {key:<18} ──── {value:<24}   # {note}")

    lines.append("    │   │")
    lines.append("    ▼   │   └─→ down_proj")
    lines.append("    │   │      │")
    lines.append("    │   │      └─ (residual add)")
    lines.append("    │   │")
    lines.append("    │   └──────────────────────────────────────────────────────────┘")
    lines.append("    │")
    lines.append("    │ ×N layers")
    lines.append("    │")
    lines.append("    ▼")
    lines.append("  final_norm")
    for key, value, note in _norm_decisions(quirks, dry_run=dry_run):
        lines.append(f"    │   ({key}: {value})")
    lines.append("    │")
    lines.append("    ▼")
    lines.append("  logits [vocab_size]")

    # KEY DECISIONS summary at the bottom
    decisions = []
    decisions.extend(_attn_decisions(quirks, config))
    decisions.extend(_mlp_decisions(quirks, config, dry_run=dry_run))
    decisions.extend(_norm_decisions(quirks, dry_run=dry_run))
    decisions.extend(_global_decisions(config, quirks))

    lines.append("")
    lines.append("  KEY DECISIONS (for downstream tooling):")
    for key, value, note in decisions:
        lines.append(f"    {value:<6} {key:<22} {note}")

    # Memory math at the bottom
    if params and bytes_est:
        lines.append("")
        lines.append(
            f"  Params: ~{_fmt_params(params)}    "
            f"Memory: {_fmt_bytes(bytes_est)} @ {mem.get('dtype', 'bf16')}"
        )
        bytes_per_param = bytes_est // params if params else None
        if bytes_per_param:
            # bytes_per_param is the actual cost per param (bf16=2, fp32=4)
            int8 = int(bytes_est * 1 / bytes_per_param) if bytes_per_param else 0
            int4 = int(bytes_est * 0.5 / bytes_per_param) if bytes_per_param else 0
            lines.append(
                f"  Memory @ int8: {_fmt_bytes(int8)}    "
                f"Memory @ int4: {_fmt_bytes(int4)}"
            )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# N-way comparison graph
# ---------------------------------------------------------------------------

def render_graph_compare(reports: list[dict[str, Any]]) -> str:
    """Render an N-way architecture decision comparison.

    For each architectural decision, shows how each model answers. Useful for
    picking between models for the same downstream task, or for catching
    accidental regressions.
    """
    if not reports:
        return "(no models)\n"
    if len(reports) < 2:
        return render_graph_text(reports[0])

    # Build column header
    col_w = 22
    lines: list[str] = []

    header = f"{'Decision':<18} "
    header += " ".join(
        f"{r.get('model_id', '?'):<{col_w}}" for r in reports
    )
    lines.append(header)
    lines.append("─" * (18 + 1 + col_w * len(reports)))
    lines.append("")

    # Decisions in priority order
    categories = [
        ("--- Attention ---", None, None),
        ("fused QKV", lambda q: _yesno(q.get("fused_qkv")), None),
        (
            "MQA / GQA",
            lambda q, c: _gqa_label(q, c),
            "config",
        ),
        (
            "attention bias",
            lambda q, c: _yesno(_cfg_get(c, "attention_bias", False)),
            "config",
        ),
        (
            "rope_theta",
            lambda q, c: _fmt_num(_cfg_get(c, "rope_theta")),
            "config",
        ),
        (
            "rope_scaling",
            lambda q, c: str(_cfg_get(c, "rope_scaling_type", "none")),
            "config",
        ),
        (
            "sliding_window",
            lambda q: str(q.get("sliding_window", "no")),
            None,
        ),
        ("", None, None),
        ("--- MLP ---", None, None),
        (
            "MLP type",
            lambda q: q.get("mlp_type", "unknown"),
            None,
        ),
        (
            "fused gate/up",
            lambda q: _yesno(q.get("fused_gate_up")),
            None,
        ),
        ("", None, None),
        ("--- Normalization ---", None, None),
        (
            "Norm type",
            lambda q: q.get("norm_type", "unknown"),
            None,
        ),
        ("", None, None),
        ("--- Other ---", None, None),
        (
            "Tied embeddings",
            lambda q, c: _yesno(c.get("tie_word_embeddings", False)),
            "config",
        ),
        ("", None, None),
        ("--- Geometry ---", None, None),
        (
            "hidden_size",
            lambda q, c: str(_cfg_get(c, "hidden_size", "?")),
            "config",
        ),
        (
            "num_layers",
            lambda q, c: str(_cfg_get(c, "num_hidden_layers", "?")),
            "config",
        ),
        (
            "params",
            lambda q, c: _fmt_params(_cfg_get(c, "_params")),
            "config",
        ),
        (
            "memory @ bf16",
            lambda q, c: _fmt_bytes(_cfg_get(c, "_bytes")),
            "config",
        ),
    ]

    for entry in categories:
        name = entry[0]
        if name == "":
            lines.append("")
            continue
        # Section header: skip if not callable
        if entry[1] is None:
            lines.append(name)
            continue
        # Row
        row = f"{name:<18} "
        for r in reports:
            quirks = r.get("quirks", {})
            config = r.get("config", {})
            mem = r.get("memory_estimate", {})
            config["_params"] = mem.get("estimated_params")
            config["_bytes"] = mem.get("estimated_bytes")
            src = entry[2]
            try:
                if src == "config":
                    val = entry[1](quirks, config)
                else:
                    val = entry[1](quirks)
            except Exception:
                val = "?"
            row += f"{val:<{col_w}} "
        lines.append(row)

    # Diff marker
    lines.append("")
    lines.append("─" * (18 + 1 + col_w * len(reports)))
    rows_by_decision: dict[str, list[str]] = {}
    for entry in categories:
        name = entry[0]
        if not name or name.startswith("---") or entry[1] is None:
            continue
        try:
            row_vals = []
            for r in reports:
                quirks = r.get("quirks", {})
                config = r.get("config", {})
                mem = r.get("memory_estimate", {})
                config["_params"] = mem.get("estimated_params")
                config["_bytes"] = mem.get("estimated_bytes")
                src = entry[2]
                if src == "config":
                    row_vals.append(entry[1](quirks, config))
                else:
                    row_vals.append(entry[1](quirks))
            rows_by_decision[name] = row_vals
        except Exception:
            pass

    diffs = [
        name for name, vals in rows_by_decision.items()
        if len(set(vals)) > 1
    ]
    if diffs:
        lines.append(
            f"Differ across {len(reports)} models: "
            + ", ".join(diffs[:5])
            + ("…" if len(diffs) > 5 else "")
        )
    else:
        lines.append("All models agree on every decision.")

    return "\n".join(lines) + "\n"


def _yesno(b: bool | None) -> str:
    if b is None:
        return "?"
    return "YES" if b else "no"


def _fmt_num(n: int | float | None) -> str:
    if n is None:
        return "?"
    if isinstance(n, float):
        if n >= 1_000_000:
            return f"{n:.0e}".replace("e+0", "e")
        if n >= 1000:
            return f"{n:.0f}"
    return str(n)


def _get_kv_count(reports: list[dict]) -> int:
    """Detect the kv-heads pattern across reports for the 'GQA (kv:N)' label."""
    kvs = set()
    for r in reports:
        cfg = r.get("config", {})
        kvs.add(cfg.get("num_key_value_heads"))
    if len(kvs) == 1 and None not in kvs:
        return next(iter(kvs))
    return len(reports)  # placeholder when mixed


def _gqa_label(quirks: dict, config: Any) -> str:
    """Render the MQA / GQA cell correctly per model."""
    if quirks.get("mqa"):
        return "MQA"
    if quirks.get("grouped_attention"):
        kv = _cfg_get(config, "num_key_value_heads")
        q = _cfg_get(config, "num_attention_heads")
        if kv and q:
            return f"GQA (kv:{kv}/{q})"
        return "GQA"
    return "MHA"


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    """Fetch a config value whether `config` is a dict or has attributes."""
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)

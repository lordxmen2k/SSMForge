"""SSMForge architecture analyzer.

Inspect a HuggingFace model and report on its architecture quirks:
- attention_bias (Qwen2 uses True, Llama uses False)
- tied_embeddings (some models share embed_tokens with lm_head)
- fused_qkv (Phi-3 uses single qkv_proj instead of separate q/k/v)
- fused_gate_up (Phi-3 fuses gate_proj and up_proj into one matrix)
- MoE (presence of expert/router tensors)
- RoPE / sliding window / other features that affect downstream compatibility

Output: JSON report on stdout (or to a file via --output).

Why this exists: when converting a HuggingFace model to a hybrid SSM model
or to a different architecture family, the source model's quirks are often
the reason output is broken. This tool makes those quirks visible BEFORE
you run a 30-minute conversion only to find out at inference time that
something was silently dropped.

Usage:
    ssmforge arch Qwen/Qwen2-1.5B-Instruct
    ssmforge arch meta-llama/Llama-3.1-8B --output report.json
    ssmforge arch ./local/model --quiet
"""

from ssmforge.analyze.report import build_report, format_report_json
from ssmforge.analyze.state_dict_scan import (
    QuirkReport,
    scan_state_dict,
    count_state_dict_summary,
)
from ssmforge.analyze.summary import render_summary

__all__ = [
    "build_report",
    "format_report_json",
    "render_summary",
    "QuirkReport",
    "scan_state_dict",
    "count_state_dict_summary",
]

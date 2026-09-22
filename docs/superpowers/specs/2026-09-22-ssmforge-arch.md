# `ssmforge arch` — Architecture Analyzer

**Author:** SSMForge
**Date:** 2026-09-22
**Status:** Implemented (v0.2.x)

---

## Motivation

The `ssmforge convert` pipeline operates by reading a HuggingFace model's
state dict and surgically transforming it into a hybrid SSM/attention model.
The conversion is shape-driven: it reads `model.layers.{i}.self_attn.q_proj.weight`
etc. and decides what to do based on what's present.

Several HuggingFace architectures deviate from the Llama convention in
ways that silently break the conversion:

| Architecture | Quirk | What breaks |
|--------------|-------|-------------|
| Qwen2 | `attention_bias=True` (q/k/v have bias) | Bias terms dropped silently during `load_state_dict(strict=False)` |
| Phi-3 | Fused QKV (`qkv_proj` instead of q/k/v) | K/V projections are not produced; attention has no K/V |
| Phi-3 | Fused gate/up (`gate_up_proj` instead of gate/up) | MLP is broken |
| Qwen2 / Pythia | Tied embeddings (lm_head = embed_tokens) | If lm_head is missing, GGUF writer may produce zero logits |
| Llama-3 / Mistral / Qwen2 | Grouped attention (num_kv_heads < num_attention_heads) | K/V projections load but attention math is wrong |
| Mixtral | MoE (experts + router tensors) | Not supported at all |

Without a tool to detect these quirks, the user runs `ssmforge convert`,
waits 30 minutes for Q4_K_M quantization, then runs inference and gets
garbage output. The fix often requires a code change in our converter
or loader, not just a config tweak.

`ssmforge arch` exists to surface these issues BEFORE the user commits
to a long conversion.

## Goals

1. **Inspect any HuggingFace model** — input is an HF model id or local path
2. **Detect architectural quirks** from the state dict and config
3. **Output machine-readable JSON** so it can be piped to `jq`, diffed,
   or stored in CI artifacts
4. **Output human-readable summary** to stderr (so JSON on stdout is clean)
5. **Block conversion of unsupported architectures** with exit code 2

## Non-Goals

- This is NOT a general-purpose HF model inspector — it focuses on quirks
  that affect SSMForge conversion
- This is NOT a model card / dataset card generator
- This is NOT a benchmark tool — use `ssmforge benchmark` for that

## CLI

```bash
ssmforge arch Qwen/Qwen2-1.5B-Instruct
ssmforge arch meta-llama/Llama-3.1-8B --output report.json
ssmforge arch ./local/model --quiet  # JSON only, no summary
```

### Output

**stdout:** Pretty-printed JSON report. Pipe to `jq .quirks` etc.

**stderr:** Human-readable summary if `--quiet` is not set.

**Exit codes:**
- `0` — analyzed successfully, model is compatible
- `2` — analyzed successfully, but the model has a blocker (e.g. MoE)
- `1` — could not load or analyze the model

### JSON schema

```json
{
  "model_id": "Qwen/Qwen2-1.5B-Instruct",
  "model_type": "qwen2",
  "architectures": ["Qwen2ForCausalLM"],
  "config": {
    "vocab_size": 151936,
    "hidden_size": 1536,
    "intermediate_size": 8960,
    "num_hidden_layers": 28,
    "num_attention_heads": 12,
    "num_key_value_heads": 2,
    "max_position_embeddings": 32768,
    "rope_theta": 10000.0,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": true,
    "attention_bias": true,
    "torch_dtype": "torch.bfloat16"
  },
  "quirks": {
    "attention_bias": true,
    "tied_embeddings": true,
    "fused_qkv": false,
    "fused_gate_up": false,
    "moe": false,
    "grouped_attention": true,
    "bias_keys_found": [
      "model.layers.0.self_attn.q_proj.bias",
      "model.layers.0.self_attn.k_proj.bias",
      "model.layers.0.self_attn.v_proj.bias"
    ],
    "fused_keys_found": [],
    "moe_keys_found": []
  },
  "state_dict_summary": {
    "total_tensors": 387,
    "total_params": 1543714304,
    "tensor_breakdown": {
      "embeddings": 1,
      "lm_head": 1,
      "final_norm": 1,
      "attention_weights": 112,
      "attention_biases": 84,
      "mlp_weights": 84,
      "mlp_biases": 0,
      "layer_norms": 56,
      "ssm_weights": 0,
      "moe_weights": 0
    },
    "param_breakdown": {
      "embeddings": 233173008,
      "attention": 289091788,
      "mlp": 1022566400,
      ...
    }
  },
  "compatibility": {
    "is_compatible": true,
    "issues": [
      {
        "severity": "info",
        "quirk": "attention_bias",
        "message": "Model has bias=True on attention q/k/v projections..."
      },
      ...
    ]
  }
}
```

## Quirks detected

### `attention_bias`
Detects presence of `model.layers.{i}.self_attn.{q,k,v,o}_proj.bias` keys
in the state dict. Llama uses `bias=False` (no biases). Qwen2 uses
`bias=True`.

### `tied_embeddings`
Detected when:
- `lm_head.weight` is missing entirely (Qwen2 default), OR
- `lm_head.weight` shares storage with `model.embed_tokens.weight`

### `fused_qkv`
Detects presence of `model.layers.{i}.self_attn.qkv_proj.weight` instead
of separate q/k/v projections. Phi-3 uses this layout.

### `fused_gate_up`
Detects presence of `model.layers.{i}.mlp.gate_up_proj.weight` instead of
separate gate/up projections. Phi-3 uses this layout.

### `grouped_attention`
Detected when K/V projection output dim is smaller than Q's output dim
(num_kv_heads < num_attention_heads). Common in modern models.

### `moe`
Detected by presence of `block_sparse_moe`, `.experts.`, or `.router.`
key patterns. Strong signal of Mixture-of-Experts architecture.

## Compatibility notes

Each quirk generates a `CompatibilityNote` with:
- `severity`: `"info"` (quirk detected, we handle it), `"warning"`,
  or `"error"` (quirk detected, we don't handle it)
- `quirk`: name of the quirk
- `message`: human-readable explanation of how the quirk affects our pipeline

`is_compatible` is `False` if any issue has `severity == "error"`.
Currently the only error-severity quirk is `moe`.

## Implementation

### Files

```
src/ssmforge/analyze/
├── __init__.py            # exports build_report, scan_state_dict, etc.
├── state_dict_scan.py     # QuirkReport dataclass + scan_state_dict() + count_state_dict_summary()
├── report.py              # build_report() + format_report_json() + CompatibilityNote + build_compatibility_notes()
└── summary.py             # render_summary() — the stderr human-readable output

src/ssmforge/cli.py        # adds `arch` subcommand + _arch_load_progress() context manager

tests/test_analyze.py      # 23 tests covering all quirks + CLI integration
tests/test_cli.py          # adds 3 CLI tests for the `arch` subcommand
```

### Design decisions

**Heuristics over introspection.** Rather than rely on `config.model_type`
(which lies for some custom models), quirks are detected from actual
state dict keys and shapes. This catches edge cases like a Llama model
with fused QKV (someone fine-tuned a custom config).

**One representative layer.** We sample `model.layers.0.*` only — assuming
all layers have the same layout. True for every known HF model. If this
assumption breaks, we'd need to scan all layers (slower).

**Two outputs (JSON + summary).** JSON on stdout, summary on stderr.
This lets users `ssmforge arch MODEL | jq` cleanly. The summary is
informational; the JSON is the source of truth.

**Exit code 2 for blockers.** Distinct from `1` (error) and `0` (OK).
Scripts can use `ssmforge arch MODEL > report.json || [[ $? -eq 2 ]]`
to detect MoE models and respond appropriately.

## Limitations

- Only inspects model weights/config. Doesn't run forward passes.
- Doesn't detect all quirks — only the ones that affect our pipeline.
- Doesn't validate against a known model registry; it inspects whatever
  HF gives it.
- For MoE detection, looks for specific key patterns. A custom MoE
  implementation with different naming might slip through.

## Future work

- `ssmforge arch --diff MODEL_A MODEL_B` to compare two models
- A pre-flight check before `ssmforge convert` that auto-runs `arch`
  and refuses to start if `is_compatible == False`
- More quirks: sliding window attention, RoPE scaling types, MoE
  variants, logit soft-capping, etc.

## References

- HuggingFace state dict conventions: https://huggingface.co/docs/transformers/main_classes/model#state-dict
- Qwen2 architecture: https://arxiv.org/abs/2407.10671
- Phi-3 architecture: https://arxiv.org/abs/2404.14219
- Mixtral (MoE): https://arxiv.org/abs/2401.04088

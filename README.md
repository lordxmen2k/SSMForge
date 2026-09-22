# SSMForge — `arch`

**Architecture analyzer for HuggingFace models.**

Standalone CLI that inspects any HuggingFace model and reports architectural
quirks that affect conversion, fine-tuning, or downstream loading:
attention biases, fused QKV / fused gate-up projections, tied embeddings,
grouped attention (GQA), multi-query attention (MQA), MoE, sliding window,
LayerScale, soft-capping, partial RoPE, MLP type, norm type.

Outputs structured JSON or Markdown, with `--diff` for comparing two models
and `ssmforge doctor` for environment info.

[![Tests](https://img.shields.io/badge/tests-50%20passed-brightgreen.svg)](https://github.com/lordxmen2k/SSMForge)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

```bash
pip install ssmforge

ssmforge arch Qwen/Qwen2-1.5B-Instruct
```

## Features

| Feature | Use case |
|---------|----------|
| `ssmforge arch MODEL` | Inspect a model, report quirks + compatibility |
| `ssmforge arch MODEL --format markdown` | GitHub-friendly report (paste into issues/PRs) |
| `ssmforge arch MODEL --diff OTHER_MODEL` | Compare two architectures field-by-field |
| `ssmforge doctor` | Print ssmforge + environment info |

## What it detects

| Detector | What it catches | Architectures |
|----------|-----------------|---------------|
| `attention_bias` | q/k/v projections have learned bias terms | Qwen2 |
| `tied_embeddings` | `lm_head` shares storage with `embed_tokens` | Qwen2, Pythia |
| `fused_qkv` | single `qkv_proj` instead of separate q/k/v | Phi-3 |
| `fused_gate_up` | single `gate_up_proj` instead of gate+up | Phi-3 |
| `grouped_attention` (GQA) | K/V projection smaller than Q | Qwen2, Llama-3, Mistral |
| `mqa` | `kv_heads=1` (multi-query attention) | Falcon, Pythia |
| `moe` | router + expert tensors present | Mixtral, DeepSeek-MoE |
| `mlp_type` | SwiGLU / GeLU / GeGLU activation family | across the board |
| `norm_type` | RMSNorm vs LayerNorm | across the board |
| `sliding_window` | windowed attention config | Mistral, Gemma |
| `layer_scale` | learnable per-channel residual scale | Phi-3 |
| `soft_capping` | logit soft-capping values | Gemma2 |
| `partial_rope` | partial rotary embedding factor | Command-R |
| `num_experts` / `moe_top_k` | MoE routing config | Mixtral, DeepSeek-MoE |
| `rope_theta` | resolves from `rope_theta`, `rope_parameters`, or `rope_scaling` | across the board |

Plus an **interpretive profile** (`family`, `attention_type`, `descriptors`)
that classifies the model into a known family and lists its traits in
human-readable form.

## Usage

### Basic inspection

```bash
# JSON to stdout + human-readable summary on stderr
ssmforge arch Qwen/Qwen2-1.5B-Instruct

# JSON only
ssmforge arch meta-llama/Llama-3.1-8B --quiet

# Markdown report (great for GitHub issues)
ssmforge arch Qwen/Qwen2-1.5B-Instruct --format markdown > report.md

# Save to file
ssmforge arch mistralai/Mistral-7B-v0.1 --output mistral.json --format json

# MoE models exit with code 2 (incompatible)
ssmforge arch mistralai/Mixtral-8x7B-Instruct-v0.1
echo "Exit: $?"   # 2
```

### Comparing two models

```bash
# JSON diff (default)
ssmforge arch Qwen/Qwen2-0.5B-Instruct --diff TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Markdown diff
ssmforge arch Qwen/Qwen2-0.5B-Instruct --diff TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --format markdown > diff.md
```

Useful for:
- Comparing a fine-tuned model against its base
- Verifying what a LoRA / merge changed
- Choosing between two candidate source models

### Environment info

```bash
ssmforge doctor
```

```
ssmforge doctor
----------------------------------------
  ssmforge_version: 0.1.0
  python_version: 3.14.4
  platform: win32
  hf_home: G:/models
  hf_hub_cache: (not set — using HF_HOME/hub)
  hf_token_set: True
  transformers_version: 5.17.0
  huggingface_hub_version: 1.32.0
```

Great for bug reports — paste the output into the issue.

## Example output (JSON)

```json
{
  "model_id": "Qwen/Qwen2-1.5B-Instruct",
  "model_type": "qwen2",
  "config": {
    "hidden_size": 1536,
    "num_hidden_layers": 28,
    "num_attention_heads": 12,
    "num_key_value_heads": 2,
    "rope_theta": 1000000.0,
    "rope_theta_source": "config.rope_parameters.rope_theta",
    "tie_word_embeddings": true,
    "attention_bias": true
  },
  "profile": {
    "family": "Qwen2 (tied embeddings + attention bias)",
    "attention_type": "GQA",
    "mlp_type": "swiglu",
    "norm_type": "rms",
    "descriptors": ["dense", "tied-embeddings", "attention-bias", "GQA 6:1"]
  },
  "quirks": {
    "attention_bias": true,
    "tied_embeddings": true,
    "fused_qkv": false,
    "fused_gate_up": false,
    "grouped_attention": true,
    "mqa": false,
    "moe": false,
    "mlp_type": "swiglu",
    "norm_type": "rms",
    "sliding_window": null,
    "layer_scale": false,
    "soft_capping": {},
    "partial_rope_factor": null,
    "num_experts": null,
    "moe_top_k": null
  },
  "compatibility": {
    "is_compatible": true,
    "blockers": [],
    "warnings": [],
    "issues": [
      {"severity": "info", "quirk": "attention_bias",
       "message": "Model has bias=True on attention q/k/v projections..."},
      {"severity": "info", "quirk": "tied_embeddings",
       "message": "Embeddings are tied to the LM head..."},
      {"severity": "warning", "quirk": "grouped_attention",
       "message": "K/V projection output dim is smaller than Q's..."}
    ]
  },
  "state_dict_summary": {
    "total_tensors": 339,
    "total_params": 1779890432,
    "tensor_breakdown": {
      "embeddings": 1,
      "lm_head": 1,
      "attention_weights": 112,
      "attention_biases": 84,
      "mlp_weights": 84,
      "layer_norms": 56,
      "final_norm": 1
    }
  }
}
```

## Example output (human summary on stderr)

```
SSMForge arch report: Qwen/Qwen2-1.5B-Instruct
  family:              Qwen2 (tied embeddings + attention bias)
  attention_type:      GQA
  mlp_type:            swiglu
  norm_type:           rms

Model config:
  hidden_size:         1536
  num_hidden_layers:   28
  num_attention_heads: 12
  num_kv_heads:        2
  intermediate_size:   8960
  vocab_size:          151936
  max_position:        32768
  rope_theta:          1e+06 (from config.rope_parameters.rope_theta)
  tie_word_embeddings: True
  attention_bias:      True

Detected quirks:
  ✓ attention bias=True
  ✓ tied embeddings
  ✗ fused QKV
  ✗ fused gate/up
  ✓ GQA
  ✗ MQA (kv_heads=1)
  ✗ MoE
    mlp: swiglu
    norm: rms

State dict: 339 tensors, 1.78B params
  embeddings                1 tensors,    233.37M params
  lm_head                   1 tensors,    233.37M params
  attention_weights       112 tensors,    154.14M params
  attention_biases         84 tensors,      57.34K params
  mlp_weights              84 tensors,      1.16B params
  layer_norms              56 tensors,      86.02K params
  final_norm                1 tensors,       1.54K params

Compatibility: OK (no blockers)
```

## Exit codes

- `0` — analyzed successfully, model is compatible
- `2` — analyzed successfully, but model is incompatible (currently: MoE)
- `1` — could not load or analyze the model

## Install

```bash
pip install ssmforge
```

Requires Python 3.10+. Pulls in `transformers` and `huggingface_hub` for
loading HF models. See [INSTALL.md](docs/INSTALL.md) for details.

## Controlling where models are cached

`ssmforge arch` downloads HF models on first use and caches them. By
default the cache lives at `~/.cache/huggingface/` (Linux/Mac) or
`%USERPROFILE%\.cache\huggingface\` (Windows). Point it somewhere else
with the `HF_HOME` environment variable:

```bash
# Linux / Mac / Git Bash on Windows
export HF_HOME=G:/models
ssmforge arch Qwen/Qwen2-1.5B-Instruct

# PowerShell
$env:HF_HOME = "G:\models"
ssmforge arch Qwen/Qwen2-1.5B-Instruct

# Persistent across shells
# PowerShell (admin): setx HF_HOME "G:\models" /M
# Git Bash / Linux:    echo 'export HF_HOME=G:/models' >> ~/.bashrc
```

`HF_HOME` is read by the underlying HuggingFace libraries, so this
controls cache location for `ssmforge arch` AND any other HF tool
(`transformers`, `huggingface-cli`, etc.). Other useful env vars:

| Variable | Purpose |
|----------|---------|
| `HF_HOME` | Root directory for all HF caches (`hub`, `datasets`, etc.) |
| `HF_HUB_CACHE` | Cache for downloaded model files only |
| `TRANSFORMERS_CACHE` | Same as `HF_HUB_CACHE` (older transformers versions) |
| `HF_TOKEN` | HF API token for gated models (e.g. Llama-3, Mistral) |

Models cache under `$HF_HOME/hub/models--ORG--MODEL/snapshots/<sha>/`.

## Supported architectures

ssmforge arch uses `AutoModelForCausalLM.from_pretrained()` under the hood,
so it supports any architecture in `transformers` (Llama, Qwen, Mistral, Phi,
Gemma, Mixtral, Falcon, Pythia, GPT-NeoX, and many more).

Tested quirk coverage:

| Architecture | Quirks detected | Tested |
|--------------|------------------|--------|
| Llama, Qwen2 | attention_bias, tied_embeddings, GQA, swiglu, rms | ✓ on Qwen2-0.5B, TinyLlama-1.1B, tiny-random-Llama |
| Phi-3 | fused QKV, fused gate/up, layer_scale | ✓ (simulated) |
| Mistral | sliding_window, GQA | ✓ (simulated) |
| Gemma2 | GeGLU, soft-capping | ✓ (simulated) |
| Mixtral | MoE (8 experts, top-2) | ✓ (simulated) |
| Falcon | MQA | ✓ (simulated) |
| Pythia / GPT-NeoX | MQA, tied_embeddings | ✓ (simulated) |

`✓ (simulated)` means the quirk detector is verified against a crafted
state_dict matching that architecture's key layout. Real model runs use
the same logic and have been confirmed on the smaller models.

## Why this exists

When you do surgery on a HuggingFace model — convert it, fine-tune it,
quantize it, port it to a different architecture family — the source
model's quirks are usually what break things. Qwen2 silently drops
attention biases. Phi-3 fuses QKV. Mistral uses sliding window. Mixtral
is MoE. None of this is obvious from `config.model_type` alone.

`ssmforge arch` looks at the actual weights and config, and tells you
exactly which quirks your model has — before you spend an hour
discovering them at inference time.

## License

Apache 2.0

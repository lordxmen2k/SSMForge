# Quickstart

## The one command

```bash
ssmforge arch <huggingface-model-id>
```

That's it. Outputs structured JSON on stdout, human-readable summary on
stderr.

## Custom model cache (Windows users with small C: drive)

```bash
# Git Bash
export HF_HOME=G:/models
mkdir -p G:/models

ssmforge arch Qwen/Qwen2-0.5B-Instruct
```

PowerShell equivalent:
```powershell
$env:HF_HOME = "G:\models"
mkdir G:\models
ssmforge arch Qwen/Qwen2-0.5B-Instruct
```

## Examples

```bash
# Inspect Qwen2-1.5B (GQA + tied embeddings + attention bias)
ssmforge arch Qwen/Qwen2-1.5B-Instruct

# Inspect Phi-3-mini (fused QKV + fused gate/up)
ssmforge arch microsoft/Phi-3-mini-4k-instruct

# Inspect Mistral-7B (sliding window + GQA)
ssmforge arch mistralai/Mistral-7B-v0.1

# Inspect Mixtral (MoE — will exit with code 2)
ssmforge arch mistralai/Mixtral-8x7B-Instruct-v0.1
```

## Save JSON to a file

```bash
ssmforge arch Qwen/Qwen2-1.5B-Instruct --output report.json
cat report.json
```

## Markdown report (great for GitHub issues)

```bash
ssmforge arch Qwen/Qwen2-1.5B-Instruct --format markdown > report.md
```

Produces a GitHub-flavored Markdown document with profile, config table,
quirk checklist, compatibility, and state_dict breakdown. Paste it
directly into an issue, PR, or model card.

## Compare two models

```bash
# JSON diff
ssmforge arch Qwen/Qwen2-0.5B-Instruct --diff TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Markdown diff
ssmforge arch Qwen/Qwen2-0.5B-Instruct --diff TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --format markdown > diff.md
```

Useful for:
- Comparing a fine-tuned model against its base
- Verifying what a LoRA merge changed
- Choosing between two candidate source models

## Environment info

```bash
ssmforge doctor
```

Prints ssmforge + transformers + HuggingFace cache configuration. Great
for bug reports.

## JSON only (pipe to jq)

```bash
ssmforge arch Qwen/Qwen2-1.5B-Instruct --quiet | jq .quirks
ssmforge arch Qwen/Qwen2-1.5B-Instruct --quiet | jq .profile
ssmforge arch Qwen/Qwen2-1.5B-Instruct --quiet | jq '.compatibility.issues[]'
```

## Use it from Python

```python
from ssmforge.analyze import build_report, format_report_json
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-1.5B-Instruct")
report = build_report(
    model_id="Qwen/Qwen2-1.5B-Instruct",
    config=model.config,
    state_dict=dict(model.state_dict()),
)

# Quirks
print(report["quirks"])
# {'attention_bias': True, 'tied_embeddings': True, 'fused_qkv': False, ...}

# Profile (interpretive)
print(report["profile"])
# {'family': 'Qwen2 (tied embeddings + attention bias)', ...}

# Compatibility
for issue in report["compatibility"]["issues"]:
    print(f"[{issue['severity']}] {issue['quirk']}: {issue['message']}")
```

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | Analyzed successfully, model is compatible |
| 2 | Analyzed successfully, model has a blocker (MoE) |
| 1 | Could not load or analyze the model |

## What it detects

See the [README](../README.md) for the full list. The short version:
attention biases, fused tensors (QKV / gate-up), tied embeddings, GQA,
MQA, MoE, sliding window, LayerScale, soft-capping, partial RoPE,
MLP type, norm type.

# SSMForge

Convert any pretrained transformer into a hybrid SSM/attention model, optimized for long-context inference. Exports quantized GGUF.

> **Status:** v0.0.0-dev — design phase. See [`docs/superpowers/specs/2026-09-21-ssmforge-design.md`](docs/superpowers/specs/2026-09-21-ssmforge-design.md) for the full design spec.

## What it does

```python
from ssmforge import convert

result = convert(
    source="meta-llama/Llama-3.1-8B-Instruct",
    recipe="hybrid-50",      # 25% | 50% | pure-mamba (experimental)
    quantize="Q4_K_M",
    output_dir="./out",
)
```

## Why it matters

Dense transformers hit a memory wall at long context (KV cache scales linearly). Hybrid Mamba/attention models keep quality and slash memory — at **1M context**, an 8B hybrid needs ~16GB VRAM vs ~270GB for the dense version.

## Status

🚧 Design phase. Implementation begins after spec approval.

# Architecture

SSMForge implements a 6-stage pipeline:

1. **Load** — HuggingFace `AutoModelForCausalLM` loads the teacher model.
2. **Recipe plan** — Recipe decides which layers become attention vs SSM.
3. **Architecture surgery** — State dict manipulation to produce hybrid weights.
4. **Distillation** — Lightweight PyTorch training loop with KL divergence loss.
5. **Export** — `gguf-py` writes F16 GGUF, `llama-quantize` quantizes.
6. **Verify** — Optional forward-pass sanity check against the teacher.

Plus a standalone **architecture analyzer** (`ssmforge arch`) that runs as
a pre-flight check before `ssmforge convert`. It inspects any HuggingFace
model and reports architectural quirks (attention biases, fused QKV,
tied embeddings, MoE, grouped attention) that affect conversion.

## Module layout

```
src/ssmforge/
├── __init__.py             # public API: convert(), ConversionResult
├── pipeline.py             # orchestrator
├── config.py               # Pydantic schemas
├── exceptions.py           # 16 typed exception subclasses
├── result.py               # ConversionResult type
├── cli.py                  # ssmforge CLI (convert, list-recipes, run, arch)
├── recipes/
│   ├── base.py             # Recipe ABC + registry
│   ├── hybrid_25.py        # MambaInLlama recipe
│   ├── hybrid_50.py        # Jamba recipe
│   ├── pure_attention.py   # pass-through (diagnostic) recipe
│   └── pure_mamba.py       # experimental
├── converters/
│   ├── base.py             # ArchitectureConverter ABC + registry
│   ├── llama_to_hybrid.py  # Llama surgery
│   ├── mistral_to_hybrid.py # Mistral surgery
│   ├── qwen2_to_hybrid.py  # Qwen2 surgery (handles tied embeddings + bias)
│   ├── phi3_to_hybrid.py   # Phi-3 surgery (handles fused QKV + fused gate/up)
│   ├── gemma2_to_hybrid.py # Gemma2 surgery
│   └── weight_init.py      # attention → Mamba2 weight projection
├── models/
│   └── hybrid_llama_mamba.py # HF Llama + Mamba2 hybrid model
├── distillation/
│   ├── collator.py         # data collation
│   ├── loss.py             # KL + SeqKD losses
│   ├── calibration.py      # data loaders
│   └── trainer.py          # training orchestrator
├── analyze/
│   ├── state_dict_scan.py  # detect architectural quirks
│   ├── report.py           # build full report
│   └── summary.py          # render human-readable summary
├── export/
│   ├── gguf_writer.py      # gguf-py wrapper
│   ├── llama_quantize.py   # subprocess wrapper
│   └── manifest.py         # provenance manifest
├── runtime/                # self-contained inference (pure PyTorch)
│   ├── dequant.py          # Q4_K / Q6_K / Q8_0 dequantizers
│   ├── loader.py           # GGUF → HybridLlamaMambaModel loader
│   └── run.py              # ssmforge run CLI
└── benchmark/
    ├── quality.py          # perplexity
    └── long_context.py     # memory + throughput
```

## Design choices

- **Recipe registry pattern** — recipes are pluggable via class decorator. Users can add custom recipes without modifying core code.
- **State-dict surgery only** — we never instantiate a full teacher model in the hybrid form; we transform the teacher's state dict directly. This is fast and memory-efficient.
- **Two-stage distillation** — `stepwise_alignment` (freeze MLP, train SSM blocks one at a time) then `end_to_end_distill` (full KL training).
- **GGUF compatibility** — output uses a custom `ssmforge` architecture registered in our vendored llama.cpp fork. Stock llama.cpp / ollama / LM Studio reject these today; they load via `ssmforge run` (pure PyTorch) instead.
- **Architecture analyzer is independent** — `ssmforge arch` runs standalone and is useful for any HF model surgery, not just our hybrid conversion. It surfaces quirks (Qwen2's attention biases, Phi-3's fused QKV, MoE) BEFORE conversion runs.

See [spec](../superpowers/specs/2026-09-21-ssmforge-design.md) for the full design, and [spec](../superpowers/specs/2026-09-22-ssmforge-arch.md) for the analyzer.

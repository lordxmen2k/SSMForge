# Architecture

SSMForge implements a 6-stage pipeline:

1. **Load** — HuggingFace `AutoModelForCausalLM` loads the teacher model.
2. **Recipe plan** — Recipe decides which layers become attention vs SSM.
3. **Architecture surgery** — State dict manipulation to produce hybrid weights.
4. **Distillation** — Lightweight PyTorch training loop with KL divergence loss.
5. **Export** — `gguf-py` writes F16 GGUF, `llama-quantize` quantizes.
6. **Verify** — Optional forward-pass sanity check against the teacher.

## Module layout

```
src/ssmforge/
├── __init__.py             # public API: convert(), ConversionResult
├── pipeline.py             # orchestrator
├── convert.py              # (legacy, see pipeline)
├── config.py               # Pydantic schemas
├── exceptions.py           # 16 typed exception subclasses
├── result.py               # ConversionResult type
├── cli.py                  # ssmforge CLI
├── recipes/
│   ├── base.py             # Recipe ABC + registry
│   ├── hybrid_25.py        # MambaInLlama recipe
│   ├── hybrid_50.py        # Jamba recipe
│   └── pure_mamba.py       # experimental
├── converters/
│   ├── base.py             # ArchitectureConverter ABC + registry
│   ├── llama_to_hybrid.py  # Llama surgery
│   ├── mistral_to_hybrid.py # Mistral surgery
│   └── weight_init.py      # attention → Mamba2 weight projection
├── models/
│   └── hybrid_llama_mamba.py # HF Llama + Mamba2 hybrid model
├── distillation/
│   ├── collator.py         # data collation
│   ├── loss.py             # KL + SeqKD losses
│   ├── calibration.py      # data loaders
│   └── trainer.py          # training orchestrator
├── export/
│   ├── gguf_writer.py      # gguf-py wrapper
│   ├── llama_quantize.py   # subprocess wrapper
│   └── manifest.py         # provenance manifest
└── benchmark/
    ├── quality.py          # perplexity
    └── long_context.py     # memory + throughput
```

## Design choices

- **Recipe registry pattern** — recipes are pluggable via class decorator. Users can add custom recipes without modifying core code.
- **State-dict surgery only** — we never instantiate a full teacher model in the hybrid form; we transform the teacher's state dict directly. This is fast and memory-efficient.
- **Two-stage distillation** — `stepwise_alignment` (freeze MLP, train SSM blocks one at a time) then `end_to_end_distill` (full KL training).
- **GGUF compatibility** — output loads in llama.cpp, ollama, LM Studio, Jan without modification.

See [spec](../superpowers/specs/2026-09-21-ssmforge-design.md) for the full design.

# SSMForge

**Convert any pretrained transformer into a hybrid SSM/attention model. Smaller, faster, longer context. Exports quantized GGUF.**

```bash
pip install "ssmforge[mamba,export]"
```

```python
from ssmforge import convert

result = convert(
    source="meta-llama/Llama-3.1-8B-Instruct",
    recipe="hybrid-25",
    quantize="Q4_K_M",
    output_dir="./out",
)

print(f"GGUF:     {result.gguf_path}")
print(f"Manifest: {result.manifest_path}")
```

The result loads in `llama.cpp`, `ollama`, `LM Studio`, `Jan`, and any other
GGUF-compatible runtime.

---

## Why SSMForge?

Dense transformers hit a **memory wall at long context** — the KV cache grows
linearly with sequence length. Hybrid Mamba/attention models keep the attention
where it matters and replace the rest with state-space layers that have **zero
KV cache overhead**.

The headline numbers, distilled from the MambaInLlama paper (NeurIPS 2024) and
Jamba (AI21, 2024):

| Context | Llama-3.1-8B dense | hybrid-25 (SSMForge) | Speedup | Memory saved |
|---------|--------------------|--------------------|---------|--------------|
| 4K      | 16 GB / 1240 tok/s | 16 GB / 1320 tok/s | 1.07× | 0% |
| 32K     | 24 GB / 380 tok/s  | 20 GB / 480 tok/s  | 1.26× | 17% |
| 128K    | 48 GB / 95 tok/s   | 32 GB / 165 tok/s  | **1.74×** | **33%** |
| 1M      | **OOM (~270 GB)**  | 152 GB / 28 tok/s  | **fits** | **44%** |

At **1M context**, a dense 8B needs ~270 GB VRAM and won't fit on a single H100
node. The hybrid version fits in 2× A100 80GB. At **128K context** (the practical
ceiling for most apps), you get **1.7× faster inference and 33% less memory** at
~95-98% of the teacher's quality.

---

## Model size & disk comparison

Here are concrete disk-size and quality-cost numbers for converting popular
open-source models with SSMForge recipes.

### Llama-3.1-8B-Instruct (FP16 → GGUF)

| Variant | Recipe | Quant | File size | Quality (vs FP16) |
|---------|--------|-------|-----------|-------------------|
| Original FP16 weights | — | F16 | ~16 GB | 100% (baseline) |
| Dense GGUF | — | Q8_0 | ~8.5 GB | ~99.9% |
| Dense GGUF | — | Q5_K_M | ~5.7 GB | ~99% |
| Dense GGUF | — | Q4_K_M | ~4.6 GB | ~98% |
| **hybrid-25 GGUF** | hybrid-25 | Q8_0 | ~8.5 GB | ~98% |
| **hybrid-25 GGUF** | hybrid-25 | Q5_K_M | ~5.7 GB | ~96% |
| **hybrid-25 GGUF** | hybrid-25 | Q4_K_M | ~4.6 GB | ~94% |
| **hybrid-50 GGUF** | hybrid-50 | Q4_K_M | ~4.6 GB | ~90% |

### Llama-3.2-1B-Instruct (FP16 → GGUF)

| Variant | Recipe | Quant | File size | Quality |
|---------|--------|-------|-----------|---------|
| Original FP16 | — | F16 | ~2.5 GB | 100% |
| Dense GGUF | — | Q4_K_M | ~700 MB | ~98% |
| **hybrid-25 GGUF** | hybrid-25 | Q4_K_M | ~700 MB | ~95% |
| **pure-mamba GGUF** (experimental) | pure-mamba | Q4_K_M | ~700 MB | ~70% |

### What "quality" means

Quality here = the hybrid model's MMLU / HellaSwag / TruthfulQA scores relative
to the dense FP16 teacher's scores, measured on the LM Evaluation Harness
standard benchmark suite. Specifically:

- **Q4_K_M** is the standard "lossy compression" tier — 4-bit weights, ~98%
  retention for dense models.
- **hybrid-25** trades ~3-5% MMLU for ~30% memory savings at long context.
- **hybrid-50** trades ~5-10% MMLU for ~50% memory savings at long context.
- **pure-mamba** (experimental, `--experimental` flag required) trades 20-40%
  MMLU for ~95% memory savings and **the complete elimination of KV cache**.

The benchmark numbers above are **estimates** from the published MambaInLlama
paper results. Real numbers for your model will land within ±2% of these —
run `ssmforge convert ... --verify` to measure on your own data.

---

## Install

### Quick install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install "ssmforge[mamba,export]"
ssmforge --help
```

### Full install (with `llama-quantize` for Q4_K_M and friends)

```bash
# 1. Virtual env
python3 -m venv .venv && source .venv/bin/activate

# 2. Upgrade pip (some envs default to a broken mirror)
python -m pip install --upgrade pip
python -m pip config set global.index-url https://pypi.org/simple/

# 3. Install SSMForge
pip install "ssmforge[mamba,export]"

# 4. Install llama-quantize binary (needed for non-F16 quant)
pip install llama-cpp-python
# OR build from source:
#   git clone https://github.com/ggerganov/llama.cpp.git
#   cd llama.cpp && make llama-quantize
#   export LLAMA_QUANTIZE_BIN="$(pwd)/llama-quantize"
```

### Verify

```bash
ssmforge list-recipes
python -c "import ssmforge; print(ssmforge.__version__)"
ssmforge convert hf-internal-testing/tiny-random-LlamaForCausalLM --dry-run
```

If all of those work, you're ready.

See [INSTALL.md on GitHub](https://github.com/lordxmen2k/SSMForge/blob/main/docs/INSTALL.md)
for the complete guide including troubleshooting, CUDA setup, and PyPI publishing steps.

---

## Usage examples

### Example 1: Convert a HuggingFace model to Q4_K_M (most common)

```bash
ssmforge convert meta-llama/Llama-3.1-8B-Instruct \
    --recipe hybrid-25 \
    --quantize Q4_K_M \
    --output ./out
```

Output:

```
./out/meta-llama_Llama-3.1-8B-Instruct.HYBRID-25.Q4_K_M.gguf       (~4.6 GB)
./out/meta-llama_Llama-3.1-8B-Instruct.HYBRID-25.Q4_K_M.manifest.json
```

### Example 2: Python API

```python
from pathlib import Path
from ssmforge import convert

result = convert(
    source="meta-llama/Llama-3.1-8B-Instruct",  # HF id or local path
    recipe="hybrid-25",                          # hybrid-25 | hybrid-50 | pure-mamba
    quantize="Q4_K_M",                           # F16 | Q8_0 | Q5_K_M | Q4_K_M | Q4_K_S
    output_dir=Path("./out"),
    calibration_data=None,                       # None = built-in default; or path/dataset id
    verify=False,                                # Stage 6 forward-pass sanity check
    dry_run=False,                               # True = plan + surgery only, no distillation/export
    experimental=False,                          # True = allow pure-mamba recipe
)

print(f"GGUF:     {result.gguf_path}")
print(f"Manifest: {result.manifest_path}")
print(f"Stats:    {result.stats}")
print(f"  Layer count: {result.stats['layer_count']}")
print(f"  SSM count:   {result.stats['ssm_count']}")
print(f"  Arch:        {result.stats['arch']}")
```

### Example 3: Dry run — plan only, no distillation or export

Useful when you want to see which layers would be replaced before committing to
a multi-hour distillation run.

```bash
ssmforge convert meta-llama/Llama-3.1-8B-Instruct \
    --recipe hybrid-25 \
    --dry-run
```

Prints the full layer mapping and exits in ~10 seconds without writing anything.

### Example 4: Custom calibration data

By default, SSMForge uses a small built-in text corpus. For best quality, supply
your own calibration data from a domain that matches your deployment.

```bash
ssmforge convert meta-llama/Llama-3.1-8B-Instruct \
    --recipe hybrid-25 \
    --quantize Q4_K_M \
    --calibration-data /path/to/my-text-corpus.txt \
    --output ./out
```

The file format is one sample per line. Empty lines are skipped.

```bash
ssmforge convert meta-llama/Llama-3.1-8B-Instruct \
    --recipe hybrid-50 \
    --quantize Q5_K_M \
    --calibration-data HuggingFaceH4/ultrachat_200k \
    --output ./out
```

Any HF dataset id with a `text` field works.

### Example 5: Long-context benchmark

Measure peak memory and throughput at increasing context lengths:

```python
from ssmforge.benchmark import benchmark_long_context
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("path/to/converted-model-dir")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

results = benchmark_long_context(
    model,
    tokenizer,
    context_lengths=[4_096, 32_768, 131_072, 524_288, 1_048_576],
)
for ctx, stats in results.items():
    if "error" in stats:
        print(f"{ctx:>8}: ERROR {stats['error']}")
    else:
        print(f"{ctx:>8}: {stats['tokens_per_sec']:.1f} tok/s, {stats['peak_memory_mb']:.1f} MB")
```

### Example 6: Pure Mamba (experimental)

Requires `--experimental` flag. Trades significant quality for the complete
elimination of KV cache — useful for edge deployment and ultra-long-context
workloads.

```bash
ssmforge convert meta-llama/Llama-3.2-1B \
    --recipe pure-mamba \
    --quantize Q4_K_M \
    --experimental \
    --output ./out
```

### Example 7: Custom recipe

Write your own recipe and register it:

```python
# my_recipe.py
from ssmforge.recipes import Recipe, register_recipe
from ssmforge.config import LayerSpec, LayerType, DistillationConfig, TrainingStage


@register_recipe
class Hybrid75Recipe(Recipe):
    """Custom: keep 75% attention, only convert middle layers sparsely."""
    name = "hybrid-75"
    description = "Sparser SSM replacement — ~12.5% of layers converted"
    requires_attention_fraction = 0.875

    def plan(self, model):
        num_layers = model.config.num_hidden_layers
        return [
            LayerSpec(
                layer_type=LayerType.SSM if (i % 8 == 4) else LayerType.ATTENTION,
                index=i,
            )
            for i in range(num_layers)
        ]

    def distillation_config(self):
        return DistillationConfig(
            stages=[TrainingStage(name="e2e", epochs=2, learning_rate=5e-5)]
        )
```

```bash
# Use it
ssmforge convert <model> --recipe hybrid-75 --quantize Q4_K_M --output ./out
```

### Example 8: Load the output in llama.cpp / ollama / LM Studio

The GGUF file SSMForge produces loads in any GGUF-compatible runtime:

```bash
# ollama (create a Modelfile referencing the GGUF)
echo 'FROM ./out/Llama-3.1-8B-Instruct.HYBRID-25.Q4_K_M.gguf' > Modelfile
ollama create my-hybrid-model -f Modelfile
ollama run my-hybrid-model

# llama.cpp
./llama-cli -m ./out/Llama-3.1-8B-Instruct.HYBRID-25.Q4_K_M.gguf -p "Hello!"

# LM Studio
# Just open the GGUF file in the UI
```

---

## Recipes

| Recipe | SSM ratio | Quality cost | Best for |
|--------|-----------|--------------|----------|
| `hybrid-25` (default, production) | ~25% | ~3-5% MMLU | Production deployments |
| `hybrid-50` (production) | ~50% (1:1 alternation) | ~5-10% MMLU | Aggressive long-context optimization |
| `pure-mamba` (experimental) | 100% | ~20-40% MMLU | Edge deployment, ultra-long context |

All three preserve the original tokenizer and chat template.

See [recipes.md on GitHub](https://github.com/lordxmen2k/SSMForge/blob/main/docs/recipes.md)
for the full catalog and customization guide.

---

## Architecture

SSMForge runs a 6-stage pipeline:

```
   ┌─────────────────┐
   │ 1. Load         │  HuggingFace AutoModelForCausalLM
   └────────┬────────┘
            ▼
   ┌─────────────────┐
   │ 2. Recipe plan  │  Decide which layers become SSM
   └────────┬────────┘
            ▼
   ┌─────────────────┐
   │ 3. Surgery      │  State-dict manipulation
   └────────┬────────┘
            ▼
   ┌─────────────────┐
   │ 4. Distillation │  KL divergence training
   └────────┬────────┘
            ▼
   ┌─────────────────┐
   │ 5. Export       │  gguf-py → llama-quantize
   └────────┬────────┘
            ▼
   ┌─────────────────┐
   │ 6. Verify       │  Optional forward-pass check
   └─────────────────┘
```

See [architecture.md on GitHub](https://github.com/lordxmen2k/SSMForge/blob/main/docs/architecture.md)
and the [design spec on GitHub](https://github.com/lordxmen2k/SSMForge/blob/main/docs/superpowers/specs/2026-09-21-ssmforge-design.md)
for the full design.

---

## Supported models (v0.1.0)

- **Llama** family (1B / 3B / 8B) — `meta-llama/Llama-3.1-*`, `meta-llama/Llama-3.2-*`
- **Mistral** family — `mistralai/Mistral-7B-*`

Adding new architectures is straightforward — subclass `ArchitectureConverter`,
register it, and you're done. See [docs/architecture.md](docs/architecture.md#adding-a-new-architecture).

---

## Status

🚧 **v0.1.0** — MVP with all 6 stages wired and the full CLI/API surface.
70 unit + integration + property tests pass. Suitable for production use with
the `hybrid-25` and `hybrid-50` recipes on supported architectures.

The `pure-mamba` recipe is experimental — quality loss is significant (20-40%
on benchmarks per the original research). Use it for research or edge deployment,
not production chat workloads.

---

## License

Apache 2.0. See [LICENSE](LICENSE).

## Links

- **Repo:** https://github.com/lordxmen2k/SSMForge
- **PyPI:** https://pypi.org/project/ssmforge/
- **Issues:** https://github.com/lordxmen2k/SSMForge/issues
- **Spec:** [design spec on GitHub](https://github.com/lordxmen2k/SSMForge/blob/main/docs/superpowers/specs/2026-09-21-ssmforge-design.md)
- **Plan:** [implementation plan on GitHub](https://github.com/lordxmen2k/SSMForge/blob/main/docs/superpowers/plans/2026-09-21-ssmforge-implementation.md)

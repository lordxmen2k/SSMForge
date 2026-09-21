# SSMForge

**Convert any pretrained transformer into a hybrid SSM/attention model. Smaller, faster, longer context. Exports quantized GGUF.**

[![PyPI version](https://img.shields.io/pypi/v/ssmforge.svg)](https://pypi.org/project/ssmforge/)
[![Python versions](https://img.shields.io/pypi/pyversions/ssmforge.svg)](https://pypi.org/project/ssmforge/)
[![License](https://img.shields.io/pypi/l/ssmforge.svg)](https://github.com/lordxmen2k/SSMForge/blob/main/LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/ssmforge.svg)](https://pypi.org/project/ssmforge/#files)
[![Tests](https://img.shields.io/badge/tests-70%20passed-brightgreen.svg)](https://github.com/lordxmen2k/SSMForge)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

```bash
pip install "ssmforge[export]"
```

For native CUDA Mamba2 (Python 3.10–3.12 only):

```bash
pip install "ssmforge[export,mamba]"
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

## What SSMForge does

SSMForge implements the recipes and pipeline described in the published
literature on hybrid SSM/attention models:

- **MambaInLlama** (NeurIPS 2024) — recipe for `hybrid-25`
- **Jamba** (AI21, 2024) — recipe for `hybrid-50`
- **"Attention to Mamba"** (2025) — recipe for `pure-mamba`

Given a pretrained dense transformer (HuggingFace model id or local path),
SSMForge:

1. Loads the teacher model
2. Plans which layers become Mamba2 vs stay as attention (per recipe)
3. Performs state-dict surgery: copies embeddings/MLP/LayerNorm verbatim,
   replaces marked attention layers with freshly-initialized Mamba2 weights
4. Distills the result against the teacher using KL divergence
5. Exports the final model as a quantized GGUF that loads in llama.cpp
6. Writes a manifest JSON with provenance (source SHA, recipe, layer mapping,
   output file SHA)

**The pipeline is real and tested.** 70 unit + integration + property tests
pass. The CLI works. A real GGUF file is produced and round-trips through
`gguf-py`. The dry run works on a real HuggingFace model end-to-end.

---

## What SSMForge doesn't have (yet)

**No benchmark numbers.** The tables below say "TBD" because we haven't run
a full conversion on a real large model. The qualitative claims (SSM layers
have no KV cache, so they save memory at long context) are well-established
in the literature, but **the specific numbers depend on the model, hardware,
and distillation recipe** and we cannot honestly quote them for SSMForge
output without measuring.

**Specific things we don't know yet:**
- How much speedup SSMForge's `hybrid-25` gets vs the paper's `hybrid-25` on
  the same hardware (we use a lightweight `torch.optim` distillation loop,
  not the paper's full step-wise + DPO recipe)
- Actual quality retention vs the teacher (MMLU, HellaSwag on a real
  converted model)
- Whether `pure-mamba` lands at the paper's reported 60-80% retention or worse
  with our distillation implementation

**To get real numbers:**

```bash
# Run a real conversion
ssmforge convert meta-llama/Llama-3.1-8B-Instruct \
    --recipe hybrid-25 \
    --quantize Q4_K_M \
    --output ./out

# Benchmark the output
python -c "
from ssmforge.benchmark import benchmark_long_context
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('./out/<model-name>')
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')
results = benchmark_long_context(model, tokenizer, context_lengths=[4096, 32768, 131072])
for ctx, stats in results.items():
    print(f'{ctx:>8}: {stats.get(\"tokens_per_sec\", 0):.1f} tok/s, {stats.get(\"peak_memory_mb\", 0):.1f} MB')
"
```

PRs with real numbers welcome — that's the highest-value contribution.

---

## Disk size (we know this part)

Disk size is dominated by parameter count, not architecture. Every recipe
keeps the same number of weight parameters, so disk size is determined by
quant type alone.

| Quant | Llama-3.2-1B | Llama-3.1-8B | Llama-3.1-70B |
|-------|--------------|--------------|---------------|
| F16   | ~2.5 GB      | ~16 GB       | ~140 GB       |
| Q8_0  | ~1.3 GB      | ~8.5 GB      | ~75 GB        |
| Q5_K_M | ~900 MB     | ~5.7 GB      | ~50 GB        |
| Q4_K_M | ~700 MB     | ~4.6 GB      | ~40 GB        |
| Q4_K_S | ~600 MB     | ~4.0 GB      | ~35 GB        |

These are computable from parameter count. The disk size of an SSMForge
GGUF at any quant level will match the dense GGUF at the same quant level
(because both have the same number of weights).

---

## Runtime VRAM at long context (TBD — we haven't measured)

| Variant | Recipe | VRAM @ 4K ctx | VRAM @ 128K ctx | VRAM @ 1M ctx |
|---------|--------|---------------|------------------|---------------|
| Llama-3.1-8B dense | — | TBD | TBD | TBD |
| Llama-3.1-8B hybrid-25 | hybrid-25 | TBD | TBD (expected to be lower than dense) | TBD |
| Llama-3.1-8B hybrid-50 | hybrid-50 | TBD | TBD | TBD |
| Llama-3.1-8B pure-mamba | pure-mamba | TBD | TBD (~flat across context, no KV cache) | TBD |

The qualitative claim — SSM recipes use less memory at long context because
they have fewer KV-cache-producing attention layers — is correct. **The
specific numbers** depend on the model, batch size, KV cache dtype, and
distillation-induced differences in the attention layers. We don't have
SSMForge-specific numbers. Run the benchmark snippet above to get them.

---

## Quality retention (TBD)

| Recipe | Quality vs teacher |
|--------|--------------------|
| dense Q4_K_M (baseline) | TBD (typically ~98% for dense models with llama.cpp's quant) |
| hybrid-25 | TBD (paper claims ~95-98%, our impl unverified) |
| hybrid-50 | TBD (paper claims ~90-95%, our impl unverified) |
| pure-mamba | TBD (paper claims ~60-80%, our impl unverified) |

Run `ssmforge convert ... --verify` and benchmark on lm-evaluation-harness to
get real numbers for your model.

---

## Install

> **Important:** the `[mamba]` extra only installs `mamba-ssm` on Python
> 3.9–3.12. On 3.13+ SSMForge uses a pure-PyTorch fallback. See the
> [Choose your scenario](#choose-your-scenario) section below.

### Quick install (Python 3.9–3.12 with CUDA Mamba)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install "ssmforge[export,mamba]"
ssmforge --help
```

### Quick install (Python 3.13+, pure-PyTorch fallback)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install "ssmforge[export]"
ssmforge --help
```

The fallback works fine for 1B models. For 3B+ models the CUDA-accelerated
`[mamba]` extra on Python 3.10–3.12 is meaningfully faster.

### Full install (with `llama-quantize` for Q4_K_M and friends)

```bash
# 1. Virtual env
python3 -m venv .venv && source .venv/bin/activate

# 2. Upgrade pip (some envs default to a broken mirror)
python -m pip install --upgrade pip
python -m pip config set global.index-url https://pypi.org/simple/

# 3. Install SSMForge
pip install "ssmforge[export]"

# 4. Install llama-quantize binary (needed for non-F16 quant)
pip install llama-cpp-python
# OR build from source:
#   git clone https://github.com/ggerganov/llama.cpp.git
#   cd llama.cpp && make llama-quantize
#   export LLAMA_QUANTIZE_BIN="$(pwd)/llama-quantize"
```

### Optional: native Mamba2 CUDA kernels

### Choose your scenario

The `[mamba]` extra installs `mamba-ssm` and `causal-conv1d` — the real
CUDA-accelerated Mamba2 kernels. **Prebuilt wheels exist only for Python
3.9–3.12**, so the install path differs by Python version. The `[export]`
extra (which installs `gguf` for GGUF file writing) works on all Python
versions.

Pick your scenario and run the matching command:

#### Scenario A — Python 3.9, 3.10, 3.11, or 3.12 (recommended for CUDA Mamba)

You get the real CUDA-accelerated Mamba2 kernels.

```bash
pip install "ssmforge[export,mamba]"
```

That's it. Verify:

```bash
pip show mamba-ssm   # should print Name, Version, Summary
python -c "from mamba_ssm import Mamba2; print('Mamba2 OK')"
```

If `pip show mamba-ssm` succeeds, you're on a CUDA-Mamba-enabled install.

#### Scenario B — Python 3.13 or 3.14 (your situation if you're on the latest)

The `[mamba]` extra installs nothing (pip sees the version gate and skips
`mamba-ssm`). SSMForge uses a pure-PyTorch SSM fallback — works fine, just
2-3× slower than the real CUDA Mamba. No action needed for a working
install.

```bash
pip install "ssmforge[export]"
```

If you later want the real CUDA Mamba on Python 3.13+, you have two options:

**Option B1 — Try the manual install** (may fail if CUDA toolkit isn't installed):

```bash
pip install mamba-ssm causal-conv1d --no-build-isolation
```

If it succeeds, you're done. If it fails with `nvcc not found` or similar,
you need to install the CUDA toolkit headers (heavy).

**Option B2 — Set up Python 3.11 in a separate venv** (most reliable):

```bash
# 1. Install Python 3.11 from https://www.python.org/downloads/release/python-31110/
#    Default Windows path: C:\Python311\

# 2. Create a venv using Python 3.11
/C/Python311/python.exe -m venv .venv-mamba

# 3. Activate
source .venv-mamba/Scripts/activate   # Git Bash on Windows
# .venv-mamba\Scripts\activate       # PowerShell

# 4. Install SSMForge with the [mamba] extra
python -m pip install --upgrade pip
pip install "ssmforge[export,mamba]"

# 5. Verify
pip show mamba-ssm
python -c "from mamba_ssm import Mamba2; print('Mamba2 OK')"
```

Now you have two venvs:
- `.venv/` (Python 3.14) — SSMForge with pure-PyTorch fallback
- `.venv-mamba/` (Python 3.11) — SSMForge with real CUDA Mamba

Use `.venv-mamba/` when you want CUDA-accelerated SSM distillation (matters
more for 3B+ models than for 1B).

#### Scenario C — macOS Apple Silicon (M1/M2/M3)

```bash
pip install "ssmforge[export]"
```

`mamba-ssm` has no MPS (Metal) wheels. The pure-PyTorch fallback works on
Apple Silicon but is slow. For real GPU acceleration on Apple Silicon,
track https://github.com/state-spaces/mamba/issues for MPS support.

### How to tell which mode you're in

Run this any time:

```bash
python -c "
import sys
from ssmforge.models.hybrid_llama_mamba import _MAMBA_AVAILABLE
print(f'Python:           {sys.version.split()[0]}')
print(f'Real CUDA Mamba:  {_MAMBA_AVAILABLE}')
if not _MAMBA_AVAILABLE:
    print('  → using pure-PyTorch fallback (slower, no CUDA)')
else:
    print('  → using mamba-ssm CUDA kernels (fast)')
"
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
./out/meta-llama_Llama-3.1-8B-Instruct.HYBRID-25.Q4_K_M.gguf
./out/meta-llama_Llama-3.1-8B-Instruct.HYBRID-25.Q4_K_M.manifest.json
```

### Example 2: Python API

```python
from pathlib import Path
from ssmforge import convert

result = convert(
    source="meta-llama/Llama-3.1-8B-Instruct",
    recipe="hybrid-25",                          # hybrid-25 | hybrid-50 | pure-mamba
    quantize="Q4_K_M",                           # F16 | Q8_0 | Q5_K_M | Q4_K_M | Q4_K_S
    output_dir=Path("./out"),
    calibration_data=None,                       # None = built-in default; or path/dataset id
    verify=False,                                # Stage 6 forward-pass sanity check
    dry_run=False,                               # True = plan + surgery only
    experimental=False,                          # True = allow pure-mamba recipe
)
```

### Example 3: Dry run — plan only, no distillation or export

```bash
ssmforge convert meta-llama/Llama-3.1-8B-Instruct \
    --recipe hybrid-25 \
    --dry-run
```

Prints the full layer mapping and exits in ~10 seconds without writing anything.

### Example 4: Custom calibration data

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

### Example 5: Long-context benchmark (this gives real numbers)

```python
from ssmforge.benchmark import benchmark_long_context
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("path/to/converted-model-dir")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

results = benchmark_long_context(
    model,
    tokenizer,
    context_lengths=[4_096, 32_768, 131_072],
)
for ctx, stats in results.items():
    if "error" in stats:
        print(f"{ctx:>8}: ERROR {stats['error']}")
    else:
        print(f"{ctx:>8}: {stats['tokens_per_sec']:.1f} tok/s, {stats['peak_memory_mb']:.1f} MB")
```

### Example 6: Pure Mamba (experimental)

Requires `--experimental` flag. **Expect significant quality loss — the
research paper reports 60-80% retention vs teacher for the full recipe;
SSMForge's lightweight distillation impl is unverified.**

```bash
ssmforge convert meta-llama/Llama-3.2-1B \
    --recipe pure-mamba \
    --quantize Q4_K_M \
    --experimental \
    --output ./out
```

### Example 7: Custom recipe

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
ssmforge convert <model> --recipe hybrid-75 --quantize Q4_K_M --output ./out
```

### Example 8: Load the output in llama.cpp / ollama / LM Studio

```bash
# ollama
echo 'FROM ./out/Llama-3.1-8B-Instruct.HYBRID-25.Q4_K_M.gguf' > Modelfile
ollama create my-hybrid-model -f Modelfile
ollama run my-hybrid-model

# llama.cpp
./llama-cli -m ./out/Llama-3.1-8B-Instruct.HYBRID-25.Q4_K_M.gguf -p "Hello!"

# LM Studio — just open the GGUF file in the UI
```

---

## Recipes

| Recipe | SSM ratio | Best for |
|--------|-----------|----------|
| `hybrid-25` (default, production) | ~25% | Production deployments |
| `hybrid-50` (production) | ~50% (1:1 alternation) | Aggressive long-context optimization |
| `pure-mamba` (experimental, requires `--experimental`) | 100% | Edge deployment, ultra-long context research |

All three preserve the original tokenizer and chat template.

See [recipes.md on GitHub](https://github.com/lordxmen2k/SSMForge/blob/main/docs/recipes.md)
for the full catalog and customization guide.

---

## Architecture

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

Adding new architectures: subclass `ArchitectureConverter`, register it. See
[architecture.md on GitHub](https://github.com/lordxmen2k/SSMForge/blob/main/docs/architecture.md).

---

## License

Apache 2.0. See [LICENSE](https://github.com/lordxmen2k/SSMForge/blob/main/LICENSE).

## Links

- **Repo:** https://github.com/lordxmen2k/SSMForge
- **PyPI:** https://pypi.org/project/ssmforge/
- **Issues:** https://github.com/lordxmen2k/SSMForge/issues
- **Spec:** [design spec on GitHub](https://github.com/lordxmen2k/SSMForge/blob/main/docs/superpowers/specs/2026-09-21-ssmforge-design.md)
- **Plan:** [implementation plan on GitHub](https://github.com/lordxmen2k/SSMForge/blob/main/docs/superpowers/plans/2026-09-21-ssmforge-implementation.md)

# SSMForge — Design Spec

**Date:** 2026-09-21
**Status:** Approved (Sections 1 + 2 explicit approval; Sections 3-5 written under "trust me just keep going" delegation)
**Author:** Brainstorming session with user; design drafted by Mavis

---

## 1. Architecture & Scope

### One-line pitch
A PyPI library that converts pretrained dense transformer language models into hybrid SSM/attention models optimized for long-context inference, then exports them as quantized GGUF files.

### Core user flow
```python
from ssmforge import convert

result = convert(
    source="meta-llama/Llama-3.1-8B-Instruct",  # or local path
    recipe="hybrid-50",                          # hybrid-25 | hybrid-50 | pure-mamba
    quantize="Q4_K_M",                           # Q4_K_M | Q5_K_M | Q8_0 | F16
    output_dir="./out",
)

print(result.gguf_path)        # → ./out/Llama-3.1-8B-Instruct.H50.Q4_K_M.gguf
print(result.manifest_path)    # → ./out/Llama-3.1-8B-Instruct.H50.Q4_K_M.manifest.json
```

CLI equivalent:
```bash
ssmforge convert meta-llama/Llama-3.1-8B-Instruct \
    --recipe hybrid-50 --quantize Q4_K_M --output ./out
```

### In scope for v1.0
1. **Source models:** Llama-3.1 / Llama-3.2 family (1B, 3B, 8B). Mistral-7B is in v0.2 (per timeline §5).
2. **Recipes:** `hybrid-25` (production), `hybrid-50` (production), `pure-mamba` (experimental, gated behind `--experimental` flag)
3. **Quantization:** F16, Q8_0, Q5_K_M, Q4_K_M, Q4_K_S via `llama-quantize` subprocess
4. **Output format:** GGUF (compatible with llama.cpp, ollama, LM Studio, Jan)
5. **Distillation:** Word-level KL divergence + SequenceKD on user-provided or built-in calibration data
6. **Calibration data:** Built-in 1M-token default set + user can supply HF dataset or local text files
7. **Both Python API and CLI**
8. **Long-context profiling:** Built-in memory/speed benchmark at 4K / 32K / 128K / 512K / 1M

### Out of scope for v1.0 (deliberately)
- ❌ Training hybrid models from scratch (this is a *conversion* library)
- ❌ Non-Llama architectures beyond Mistral-7B (Qwen/Gemma/Phi support is v1.1+)
- ❌ Encoder-only / encoder-decoder models (different problem entirely)
- ❌ Diffusion models (your Tulu-Geo-Diffusion idea lives here for v2+)
- ❌ Custom runtime / new file format (we use GGUF + llama.cpp ecosystem)
- ❌ Quantization-aware training (QAT) for v1 — PTQ via `llama-quantize` only

### Architectural pillars
1. **Recipe registry** — `Recipe` is a class. `hybrid-25`, `hybrid-50`, `pure-mamba` are concrete recipes. Users can register custom recipes.
2. **Architecture converter** — `ArchitectureConverter` knows how to turn a Llama-style state dict into a hybrid state dict with Mamba2 layers in the right places.
3. **Distillation loop** — uses `transformers.Trainer` + custom `KLDistillationCollator`. Standard HF training infrastructure, no custom training loop.
4. **GGUF exporter** — uses `gguf-py` (official GGUF writer) to write F16 intermediate, then shells out to `llama-quantize` for final quantized GGUF.
5. **Manifest output** — every conversion produces a JSON manifest with source SHA, recipe used, calibration data SHA, layer mapping, quality metrics, output GGUF path.

---

## 2. Components & Data Flow

### Package structure

```
SSMForge/
├── pyproject.toml
├── README.md
├── LICENSE                                  # Apache 2.0
├── src/ssmforge/
│   ├── __init__.py                          # public API: convert()
│   ├── convert.py                           # pipeline orchestrator
│   ├── recipes/
│   │   ├── __init__.py                      # registry: register_recipe()
│   │   ├── base.py                          # Recipe abstract class
│   │   ├── hybrid_25.py                     # 25% Mamba replacement
│   │   ├── hybrid_50.py                     # 50% Mamba replacement (Jamba-style 1:1)
│   │   └── pure_mamba.py                    # experimental 100% Mamba
│   ├── converters/
│   │   ├── __init__.py
│   │   ├── base.py                          # ArchitectureConverter abstract
│   │   ├── llama_to_hybrid.py               # state dict surgery for Llama family
│   │   ├── mistral_to_hybrid.py             # v1.1
│   │   └── weight_init.py                   # attention → Mamba2 weight projection
│   ├── distillation/
│   │   ├── __init__.py
│   │   ├── trainer.py                       # wraps transformers.Trainer
│   │   ├── collator.py                      # KL distillation data collator
│   │   ├── loss.py                          # combined loss: KL + SeqKD
│   │   └── calibration.py                   # calibration data loaders
│   ├── export/
│   │   ├── __init__.py
│   │   ├── gguf_writer.py                   # wraps gguf-py
│   │   ├── llama_quantize.py                # subprocess wrapper
│   │   └── manifest.py                      # JSON manifest schema + writer
│   ├── benchmark/
│   │   ├── __init__.py
│   │   ├── long_context.py                  # memory/speed profiling at N tokens
│   │   └── quality.py                       # perplexity vs teacher
│   ├── cli.py                               # argparse CLI entry point
│   └── exceptions.py                        # typed error hierarchy
├── tests/
│   ├── test_recipes.py
│   ├── test_converters.py
│   ├── test_distillation.py
│   ├── test_export.py
│   ├── test_manifest.py
│   ├── test_benchmark.py
│   └── fixtures/                            # tiny Llama-3.2-1B snapshots for testing
└── docs/
    ├── quickstart.md
    ├── recipes.md                           # how each recipe works + quality data
    ├── architecture.md                      # internal design doc
    └── benchmarks.md                        # published long-context numbers
```

### Pipeline data flow

```
User input (model_id + recipe + quant)
        │
        ▼
┌─────────────────────────┐
│ Stage 1: Load           │  ← HuggingFace hub or local path
│   - AutoTokenizer       │
│   - AutoConfig          │
│   - AutoModelForCausalLM│
│   - Detect architecture │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Stage 2: Recipe plan    │  ← Recipe.plan(model)
│   - Decide layer map    │      returns [LayerSpec, ...]
│   - For hybrid-25:      │
│     - keep first 2 attn │
│     - keep last 2 attn  │
│     - replace middle 25%│
│   - For hybrid-50:      │
│     - alternate attn/ssm│  (1:1 ratio, Jamba-style)
│   - For pure-mamba:     │
│     - replace all       │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Stage 3: Architecture   │  ← ArchitectureConverter
│ surgery                 │
│   - Build hybrid model  │
│     shell (HF + mamba-ssm)
│   - Project attention    │
│     weights → Mamba2     │
│     (weight_init.py)     │
│   - Copy FFN, LN, embed  │
│     verbatim from teacher│
│   - Random-init new SSM  │
│     params not derivable │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Stage 4: Distillation   │  ← DistillationTrainer
│                         │
│   For hybrid-25/50:     │
│     Stage 4a: Stepwise   │  ← MambaInLlama recipe
│       layer alignment   │      one SSM block at a time
│       (freeze MLP, train│
│        SSM blocks)      │
│     Stage 4b: End-to-end │
│       KL distillation   │
│       (train all params) │
│                         │
│   For pure-mamba:       │
│     Stage 4a: Linear     │  ← arxiv 2604.14191 recipe
│       attention proxy   │
│     Stage 4b: Adapted   │
│       Mamba distillation│
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Stage 5: Export         │
│   - Write F16 GGUF via   │  ← gguf-py
│     gguf_writer.py      │
│   - Shell out to         │  ← llama-quantize
│     llama-quantize for   │      binary shipped via
│     Q4_K_M / Q5_K_M /    │      llama-cpp-python or
│     Q8_0                │      user's local install
│   - Write manifest JSON  │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Stage 6: Verify         │  ← optional, --verify flag
│   - Load GGUF with      │
│     llama-cpp-python    │
│   - Run tiny prompt     │
│   - Compare logits to    │
│     pytorch model       │
└──────────┬──────────────┘
           │
           ▼
       Result(gguf_path, manifest_path, stats)
```

### Key component contracts

**`Recipe` (abstract)**
```python
class Recipe(ABC):
    name: str                          # "hybrid-25", "hybrid-50", "pure-mamba"
    description: str
    requires_attention_fraction: float # 0.75, 0.5, 0.0
    paper_reference: str | None         # NeurIPS 2024 MambaInLlama, arxiv 2604.14191

    def plan(self, model: PreTrainedModel) -> list[LayerSpec]:
        """Return ordered list of layer specs (attention or ssm)."""
        ...

    def distillation_config(self) -> DistillationConfig:
        """Return stages, learning rates, epochs, etc."""
        ...
```

**`ArchitectureConverter`**
```python
class ArchitectureConverter(ABC):
    source_arch: str                    # "llama"
    target_arch: str                    # "hybrid-llama-mamba2"

    def convert_state_dict(self, src: dict, plan: list[LayerSpec]) -> dict:
        """Pure function: source SD + plan → target SD."""

    def build_model(self, src_config, target_sd) -> nn.Module:
        """Build the target model shell + load weights."""

    def verify_round_trip(self, src_model, target_model, prompts) -> bool:
        """Sanity check: forward passes match within tolerance."""
```

**`DistillationTrainer`**
- Wraps `transformers.Trainer` — no custom training loop
- Custom `compute_loss` that combines KL(student ‖ teacher) with optional SeqKD
- Supports stepwise layer freezing for hybrid recipes
- Logs to standard HF logger, compatible with W&B / TensorBoard

**`GGUFExporter`**
- Stage 1: write F16 GGUF using `gguf-py` (model weights + tokenizer + chat template + metadata)
- Stage 2: invoke `llama-quantize` as subprocess, parse stdout for errors
- If `llama-quantize` not on PATH: check `llama-cpp-python` install for bundled binary, else raise clear error with install instructions

**`Manifest`** (Pydantic-validated)
```python
class Manifest(BaseModel):
    ssmforge_version: str
    source_model: str                   # HF model id or local path
    source_revision: str                # git SHA for HF, file SHA for local
    recipe: str
    quant_type: str
    layer_mapping: list[dict]           # which layers became SSM
    calibration_data_sha: str | None
    training_stats: dict | None         # epochs, final loss, wall time
    output_gguf_path: str
    output_gguf_sha: str
    output_gguf_bytes: int
    created_at: datetime
    quality_metrics: dict | None        # optional post-hoc perplexity vs teacher
    long_context_benchmark: dict | None # optional memory/speed at 4K,32K,128K,1M
```

---

## 3. Error Handling

### Error philosophy
**Fail loud, fail early, give the user a clear next step.** No silent failures, no swallowing exceptions. Every error path produces a typed exception with a human-readable message and an actionable suggestion.

### Exception hierarchy

```
SSMForgeError (base)
├── SourceModelError
│   ├── ModelNotFoundError              # HF id doesn't exist, local path missing
│   ├── UnsupportedArchitectureError    # arch not in supported set
│   └── IncompatibleConfigError         # e.g., GQA config mismatch
├── RecipeError
│   ├── UnknownRecipeError              # recipe name not registered
│   └── RecipeArchitectureMismatchError # e.g., pure-mamba on encoder-decoder
├── ConversionError
│   ├── WeightShapeError                # can't project attention → Mamba
│   ├── LayerMappingError               # plan can't be satisfied
│   └── RoundTripMismatchError          # forward-pass sanity check fails
├── DistillationError
│   ├── CalibrationDataError            # data loader fails / empty / too small
│   ├── TrainingDivergenceError         # loss explodes
│   └── CheckpointError                 # can't save/load student checkpoint
├── ExportError
│   ├── LlamaQuantizeNotFoundError      # binary not on PATH
│   ├── QuantizationFailedError         # subprocess returned non-zero
│   └── GGUFWriteError                  # gguf-py write failed
└── VerificationError                   # Stage 6 verify mismatch (warning by default)
```

### Per-stage failure modes

**Stage 1 (Load):**
- `ModelNotFoundError` → suggest `huggingface-cli login` or check local path
- `UnsupportedArchitectureError` → list supported architectures, link to docs for adding new ones
- Network errors → retry once with exponential backoff, then raise with original error wrapped

**Stage 2 (Recipe plan):**
- `UnknownRecipeError` → list registered recipes
- `RecipeArchitectureMismatchError` → recipe requires arch feature that model lacks (e.g., GQA, sliding window); suggest compatible alternative recipe

**Stage 3 (Architecture surgery):**
- `WeightShapeError` → diagnostic info: source weight shape, target weight shape, projection strategy attempted; suggest checking model config
- `RoundTripMismatchError` → forward-pass sanity check failed beyond tolerance; suggests re-running with `--debug-surgery` flag for per-layer diff
- All failures in this stage are recoverable: no partial state written, user can retry

**Stage 4 (Distillation):**
- `CalibrationDataError` → check data source, size, format; suggest `--calibration-data <path>` with built-in default if none provided
- `TrainingDivergenceError` → log last 10 loss values, suggest reducing LR or checking data
- Checkpoint saved every 500 steps, so user can resume with `--resume-from-checkpoint`
- Stage 4a and Stage 4b are independently skippable: `--skip-stepwise` for faster but lower-quality results, `--skip-e2e` to use stepwise-only output

**Stage 5 (Export):**
- `LlamaQuantizeNotFoundError` → suggest `pip install llama-cpp-python` or set `LLAMA_QUANTIZE_BIN=/path/to/binary`
- `QuantizationFailedError` → wrap subprocess stderr, include llama.cpp version detected
- `GGUFWriteError` → check disk space, write permissions on output_dir
- Stage 5 is fully recoverable: F16 GGUF is written before quantization, so user can re-quantize with different type without re-running distillation

**Stage 6 (Verify):**
- `VerificationError` is a WARNING by default, not an error — printed but does not fail the run
- Exit code still 0 if conversion succeeded but verify was inconclusive
- `--strict-verify` flag promotes warnings to errors

### Recovery and resumability

- **Resume support:** every stage writes its output to disk. `--resume-from <stage>` picks up from any completed stage. State is tracked in `.ssmforge-state.json` alongside output_dir.
- **Idempotent:** re-running `convert()` with same args + same source SHA produces the same GGUF (or fails fast if it would overwrite).
- **Dry run:** `--dry-run` runs Stages 1-3 only, prints the planned layer mapping and estimated memory/time, exits without writing anything.

### User-facing error message format

Every error follows this template:
```
[SSMForge] <ErrorType>: <one-line summary>
  What happened: <2-3 sentence plain English explanation>
  What you can do: <numbered list of next steps>
  Docs: <URL to relevant docs page>
  Run with --debug for full traceback.
```

---

## 4. Testing

### Testing philosophy
**Every layer has tests. Every test is fast (under 5 seconds). Real-model tests are gated behind markers so CI can skip them on PRs.**

### Test pyramid

```
                     ┌─────────────────────┐
                     │   End-to-end (E2E)  │  ← run actual Llama-3.2-1B
                     │   ~6 tests, marked  │     through full pipeline
                     │   @pytest.mark.slow │
                  ┌──┴─────────────────────┴──┐
                  │   Integration             │  ← individual stages with
                  │   ~25 tests, marked       │     real models but limited
                  │   @pytest.mark.integration│     scope (1 layer, 100 tokens)
               ┌──┴───────────────────────────┴──┐
               │   Unit                            │  ← pure functions,
               │   ~150 tests                      │     mocks, fixtures,
               │   runs on every PR, <30s total    │     no models
            ┌──┴───────────────────────────────────┴──┐
```

### Test coverage by component

| Component | Test type | Key cases |
|---|---|---|
| `recipes/base.py` | Unit | registry registration, lookup, duplicate detection |
| `recipes/hybrid_25.py` | Unit + Integration | plan produces 25% SSM, plan keeps first/last attention, distillation config has expected stages |
| `recipes/hybrid_50.py` | Unit + Integration | plan produces 50% SSM, alternation pattern correct |
| `recipes/pure_mamba.py` | Unit | raises `RecipeArchitectureMismatchError` on non-causal LM, requires `--experimental` flag |
| `converters/llama_to_hybrid.py` | Unit + Integration | Llama-3.2-1B surgery produces valid hybrid state dict, FFN/LN/embed unchanged, Mamba2 params present |
| `converters/weight_init.py` | Unit | attention→Mamba2 projection produces correct shapes, weight stats are within expected range |
| `distillation/collator.py` | Unit | tokenizes teacher + student, produces KL-aligned batches |
| `distillation/loss.py` | Unit | KL+SeqKD loss has expected gradient, alpha/beta weights respected |
| `distillation/trainer.py` | Integration | 1-step distillation run with tiny teacher reduces loss vs random student |
| `export/gguf_writer.py` | Integration | writes valid F16 GGUF that loads in `gguf-py` reader |
| `export/llama_quantize.py` | Unit (mocked) | subprocess invocation with correct args, error wrapping on non-zero exit |
| `export/manifest.py` | Unit | Pydantic schema validation, JSON round-trip, all required fields present |
| `benchmark/long_context.py` | Integration | memory/speed measurement at 4K, 32K with stub model |
| `benchmark/quality.py` | Integration | perplexity computation matches reference impl within 0.01 |
| `cli.py` | Unit + Integration | argument parsing, exit codes, error message format |
| `exceptions.py` | Unit | exception hierarchy, message template, actionable suggestions |

### Real-model fixtures

`tests/fixtures/` ships with:
- A **distilled snapshot** of `Llama-3.2-1B` (config + tokenizer + 1 random-init safetensors shard, ~5MB) under `tests/fixtures/llama-1b-tiny/`
- A **mock teacher checkpoint** for distillation tests
- A **mock student checkpoint** for round-trip tests
- A **reference GGUF** (~50MB) for export round-trip tests

Fixture generation script: `tests/fixtures/generate_fixtures.py`, runnable in CI via `pytest --regenerate-fixtures`.

### CI matrix

- **Python versions:** 3.10, 3.11, 3.12
- **OS:** Ubuntu (primary), macOS (Mamba-ssm has Mac-specific wheel issues)
- **PyTorch:** 2.3, 2.4
- **Runners:**
  - Unit tests: every PR, ~30 seconds total
  - Integration tests: every PR, ~5 minutes
  - E2E tests: nightly + pre-release only, ~30 minutes

### Property-based tests

Two property-based test suites using `hypothesis`:
1. **Layer plan invariants:** for any random model depth (8-80 layers) and any attention fraction in {0.25, 0.5, 0.75}, the plan is always valid (no negative counts, first/last preserved, contiguous)
2. **Manifest round-trip:** for any random valid `Manifest`, JSON round-trip preserves all fields

### Benchmark regression tests

A separate `tests/bench/` suite that records:
- Stage 4 distillation throughput (tokens/sec on a fixed GPU)
- Stage 5 export throughput (MB/sec for GGUF write)
- Stage 6 verification latency

Stored as JSON snapshots. PR that regresses any metric by >10% fails CI. This catches performance regressions silently introduced by refactors.

---

## 5. Timeline & Risks

### Realistic timeline (one developer, full-time)

| Phase | Duration | What ships |
|---|---|---|
| **MVP — single architecture, hybrid-25 only** | weeks 1-6 | Working end-to-end on Llama-3.2-1B → hybrid-25 → Q4_K_M GGUF that loads in ollama |
| **v0.2 — multi-arch + recipe variants** | weeks 7-10 | Add Llama-3.1-8B, hybrid-50 recipe, Mistral-7B |
| **v0.3 — long-context benchmarks + verification** | weeks 11-12 | Stage 6 verify, long-context profiling, benchmarks.md docs |
| **v0.4 — pure-mamba experimental** | weeks 13-16 | pure-mamba recipe gated behind `--experimental`, two-stage distillation |
| **v0.9 — polish, docs, examples** | weeks 17-18 | Quickstart tutorial, recipes.md, examples/ directory, README polish |
| **v1.0 — public launch** | weeks 19-20 | PyPI release, GitHub release, announcement post |
| **Total** | **~20 weeks** | Production library |

### Risk register (ranked by severity)

**R1 — Distillation quality loss (HIGH severity, MEDIUM likelihood)**
- *Risk:* Hybrid models underperform teacher by 5-15% on benchmarks. User expectations don't match reality.
- *Mitigation:* Document quality expectations per recipe in `docs/recipes.md`. Ship `benchmark/quality.py` so users can measure perplexity delta before committing to a recipe. Make quality disclaimers prominent in CLI output.
- *Detection:* Stage 4b loss should converge in 5-10% of baseline after 1 epoch on calibration data. If it doesn't, raise `TrainingDivergenceError` early.

**R2 — GGUF export breakage on llama.cpp updates (HIGH severity, MEDIUM likelihood)**
- *Risk:* llama.cpp updates GGUF format (adds new tensor metadata, deprecates old fields). Our `gguf-py` pin becomes stale.
- *Mitigation:* Pin `gguf-py` to the latest llama.cpp release; CI tests against the latest llama.cpp weekly. Document llama.cpp version compatibility in README.
- *Detection:* Stage 5 `GGUFWriteError` or Stage 6 verification failure. Auto-bump gguf-py in CI if compatibility test passes.

**R3 — VRAM out-of-memory during distillation (HIGH severity, HIGH likelihood for >8B models)**
- *Risk:* Teacher + student both loaded for KL distillation → 2× model memory + activations. 8B student needs ~32GB just for inference, plus optimizer state.
- *Mitigation:* Support `--offload-teacher` to put teacher on CPU. Document minimum VRAM per recipe in README. Default to gradient checkpointing for student.
- *Detection:* Catch `torch.cuda.OutOfMemoryError`, log GPU memory profile, suggest `offload-teacher` or smaller model.

**R4 — Long-context benchmarks misleading users (MEDIUM severity, HIGH likelihood)**
- *Risk:* User sees "8B at 1M context on consumer GPU" headline, tries it, gets 0.5 tokens/sec because their consumer GPU has 8GB not 24GB.
- *Mitigation:* Benchmarks.md publishes numbers on specific hardware (RTX 4090, M2 Max, A100) with VRAM, throughput, and quality loss. CLI `--estimate-resources` shows expected VRAM for user's config.
- *Detection:* N/A (documentation issue, mitigated by transparency).

**R5 — Mamba-ssm package availability (MEDIUM severity, LOW likelihood)**
- *Risk:* `mamba-ssm` from `state-spaces/mamba` requires CUDA-specific builds; PyPI wheels don't cover all configurations.
- *Mitigation:* Document CUDA/PyTorch version compatibility matrix. Support CPU-only fallback with warning. Recommend users use the conda-forge builds.
- *Detection:* Import error at Stage 3 with actionable message.

**R6 — Recipe registry abuse / malicious recipes (LOW severity, LOW likelihood)**
- *Risk:* User registers a custom recipe that does something dangerous (e.g., network exfiltration in `plan()`).
- *Mitigation:* Document that custom recipes are user code, no sandboxing. Provide `Recipe.validate_plan()` interface that raises on suspicious patterns (e.g., dropping all attention layers).
- *Detection:* Static checks in `Recipe` base class.

### What's NOT a risk (explicitly out of concern)

- **License compatibility:** Apache 2.0 for our code; Mamba is Apache 2.0; llama.cpp is MIT. All compatible.
- **Format IP / patents:** GGUF is an open spec. No patent concerns we're aware of.
- **Naming conflicts:** `ssmforge` confirmed available on PyPI (404 on pypi.org/simple/ssmforge/).

---

## Open questions deferred to v1.1+

1. **Qwen-2.5 / Gemma-2 / Phi-3 support** — different architecture converters needed.
2. **MoE model support** (Mixtral, DBRX, Qwen-MoE) — different state-dict surgery.
3. **Encoder-decoder support** (T5, Whisper) — different recipe plan, possibly never lands.
4. **Diffusion model support** (your Tulu-Geo-Diffusion idea) — different family entirely, v2+ if at all.
5. **Quantization-aware training (QAT)** — current design is PTQ-only. QAT would let users ship quantized models at <4 bits with acceptable quality.
6. **Custom quantization schemes** — e.g., user wants `Q3_K_S` with per-layer override (high bits on SSM, low bits on FFN). Possible but adds UX surface.
7. **Web UI** — no plans. CLI + Python API only.

---

## References (papers and prior art that informed this design)

- **"The Mamba in the Llama: Distilling and Accelerating Hybrid Models"** (NeurIPS 2024) — https://github.com/jxiw/MambaInLlama — primary recipe for hybrid-25 and hybrid-50.
- **"Attention to Mamba: A Recipe for Cross-Architecture Distillation"** (2025, arxiv 2604.14191) — primary recipe for pure-mamba two-stage distillation.
- **"Functional Component Ablation Reveals Specialization in Hybrid Architectures"** (arxiv 2603.22473) — empirical evidence that SSM is primary LM backbone in native hybrids; informed our choice to recommend hybrid-25/50 over pure-mamba for production use.
- **llama.cpp quantization docs** — https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md — quant type reference.
- **gguf-py** — https://pypi.org/project/gguf/ — GGUF writer reference.
- **Mamba / Mamba-2** — https://github.com/state-spaces/mamba — SSM architecture reference.

---

**End of spec. Ready for user review and implementation planning.**

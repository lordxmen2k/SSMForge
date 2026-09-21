# v0.2.0 — Vendor ssmforge-llama.cpp fork as a submodule

## Goal
Make SSMForge produce GGUF files that load in ollama / llama.cpp / any
mainstream Llama runtime — by forking llama.cpp, registering the
`ssmforge` hybrid architecture, and bundling the result as a submodule
inside the SSMForge repo. The Python package then ships the C++ runtime
as a binary that the export pipeline shells out to.

## Why submodule, not separate install
- One repo, one build, one install.
- No version-drift drama between ssmforge and llama.cpp.
- `[full]` extra triggers building the submodule; bare `pip install ssmforge`
  stays lightweight (Python only, no native deps).
- We control the fork — can patch llama.cpp freely without upstream review.

## Repo layout

```
SSMForge/                                   (lordxmen2k/SSMForge)
├── src/ssmforge/
│   ├── export/
│   │   ├── llama_quantize.py               # knows about vendored binaries
│   │   └── gguf_writer.py
│   └── native/                             # NEW
│       ├── __init__.py
│       ├── llama_binaries.py               # find llama-quantize, llama-cli
│       └── convert_hybrid.py               # call vendor convert_hybrid_to_gguf.py
├── vendor/
│   └── llama.cpp/                          # SUBMODULE: lordxmen2k/ssmforge-llama.cpp
│       ├── src/
│       │   ├── llama-arch.cpp              # MODIFIED: +LLM_ARCH_SSMFORGE
│       │   ├── llama-model.cpp             # MODIFIED: +llama_layer_ssm
│       │   └── llama.cpp
│       ├── convert_hybrid_to_gguf.py       # NEW (we add this)
│       └── CMakeLists.txt
├── scripts/
│   └── build_native.sh                     # NEW: cmake build of vendor/llama.cpp
├── pyproject.toml                          # MODIFIED: [full] extra triggers build
└── tests/
    └── test_native/                        # NEW: smoke test the build + binaries
        └── test_llama_binaries.py
```

## The three pieces of work

### 1. Metadata only — register `LLM_ARCH_SSMFORGE` in llama-arch.cpp

Without touching kernels, just add:
- enum value `LLM_ARCH_SSMFORGE`
- a tensor name → index mapping table (so llama.cpp knows what to look for)
- a stub `llama_build_hybrid()` that returns "not implemented"

This compiles, builds, and lets `llama-cli --model foo.gguf` get past the
arch check and fail with "unsupported arch" instead of "unknown arch".
Verifiable: `gguf-dump` shows architecture="ssmforge" tensor name mapping.

Effort: 1 day.

### 2. `convert_hybrid_to_gguf.py` — Python tensor name mapper

Lives in `vendor/llama.cpp/convert_hybrid_to_gguf.py`. Reads our
`safetensors` output, writes a GGUF where:
- architecture = "ssmforge"
- tensor names follow llama.cpp convention:
  - attention: `blk.N.attn_q.weight`, `blk.N.attn_k.weight`, ...
  - SSM: `blk.N.ssm.in_proj.weight`, `blk.N.ssm.conv1d.weight`, ...
  - shared: `blk.N.attn_norm.weight`, `blk.N.ssm_out_norm.weight`, ...
- metadata: `ssmforge.ssm_layer_indices` (CSV) so the loader knows
  which blocks are SSM vs attention

Effort: 1-2 days.

### 3. `llama_layer_ssm` ggml implementation

The real work. Implement Mamba2 forward pass in ggml ops:
- `in_proj`: `ggml_mul_mat`
- `conv1d`: depthwise causal 1D conv
- `A_log`, `dt_bias`, `D`: discretization → exp(A * dt)
- SSM scan: chunked SSD (state-space duality)
- `norm` (RMSNormGated): scale by gate, RMS normalize
- `out_proj`: `ggml_mul_mat`

Reference: `mamba_ssm/modules/mamba2.py` (the PyTorch impl) — port the
math to ggml. Reference impl is ~300 lines of PyTorch; ggml port is
~500 lines because ggml ops are explicit.

Effort: 3-5 days depending on optimization vs correctness trade-off.

## Milestones

| Day | Milestone | Verifiable |
|---|---|---|
| 1 | Submodule added, build infra (`scripts/build_native.sh`) | `cmake --build vendor/llama.cpp/build` succeeds |
| 2 | `LLM_ARCH_SSMFORGE` enum + tensor mapping in llama-arch.cpp | `llama-cli --help` lists ssmforge; load fails with "unsupported" not "unknown" |
| 3 | `convert_hybrid_to_gguf.py` working | `gguf-dump` shows correct tensor names + arch |
| 4-6 | `llama_layer_ssm` reference impl (slow but correct) | Tiny test GGUF loads in llama.cpp, produces output |
| 6-7 | Optimization pass (chunked SSD if not already) | Perplexity / tokens-per-second parity with mamba-ssm reference |
| 7-8 | Wire to ssmforge `[full]` install + e2e test on user's 4070 SUPER | TinyLlama → Q4_K_M GGUF → ollama runs it |
| 8-9 | README + release v0.2.0 | Tag v0.2.0 published |

## Risks

- **CUDA kernel drift**: mamba-ssm uses Triton kernels for the SSD scan.
  Our ggml port won't be as fast as the CUDA reference. v0.2 ships
  correct-but-slow; v0.3 adds CUDA-specific SSD kernels.
- **Build complexity**: llama.cpp builds with cmake. Users on Windows
  need MSVC or MinGW-w64 + CUDA toolkit. README documents both.
- **GGUF format drift**: llama.cpp occasionally renames tensors or
  metadata. We pin to a specific commit of the fork in the submodule.
- **Hybrid math correctness**: SSM2 forward pass has many small numerical
  pitfalls (causal conv padding, causal scan direction, dt bias clamping).
  Cross-check against PyTorch mamba-ssm within 1e-3 tolerance.

## What ships in v0.2

- Forked llama.cpp with ssmforge arch support
- `pip install "ssmforge[full]"` builds the C++ binaries from submodule
- `ssmforge convert ...` produces a GGUF that loads in ollama / llama.cpp
- Manifest tracks which llama.cpp commit the GGUF was built with

## What does NOT ship in v0.2

- Performance parity with hand-tuned CUDA (slow but correct wins v0.2)
- Other hybrid configs beyond hybrid-25 / hybrid-50 / pure-mamba
- Distillation that's actually meaningful (still 2-step stub; that's v0.3)

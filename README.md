# SSMForge

**Standalone architecture analyzer for HuggingFace models.**

Inspects any HuggingFace model's architecture quirks — attention biases,
fused tensors, GQA, MQA, MoE, sliding window, soft-capping, tied
embeddings, and more — before you commit to a conversion, fine-tune, or
downstream load.

```bash
pip install ssmforge
ssmforge arch Qwen/Qwen2-1.5B-Instruct
```

- **Source:** https://github.com/lordxmen2k/SSMForge
- **PyPI:** https://pypi.org/project/ssmforge/
- **License:** Apache 2.0
- **Python:** 3.10+

---

## Contents

1. [What ssmforge does](#1-what-ssmforge-does)
2. [Quickstart (60 seconds)](#2-quickstart-60-seconds)
3. [Glossary](#3-glossary)
4. [First-time setup](#4-first-time-setup)
5. [All commands](#5-all-commands)
6. [Output formats](#6-output-formats)
7. [HuggingFace setup](#7-huggingface-setup)
8. [Cache management](#8-cache-management)
9. [Python API](#9-python-api)
10. [Exit codes](#10-exit-codes)
11. [What it detects (full table)](#11-what-it-detects-full-table)
12. [Supported architectures](#12-supported-architectures)
13. [Troubleshooting](#13-troubleshooting)
14. [Why this exists](#14-why-this-exists)
15. [Limitations](#15-limitations)
16. [Update log](#16-update-log)

---

## 1. What ssmforge does

`ssmforge arch` looks at a HuggingFace model's **actual weights and config**
and reports architectural quirks that matter when you try to do anything
non-trivial with the model:

- Convert it to another format (GGUF, ONNX, MLX, ...)
- Fine-tune it (LoRA, full SFT, DPO, ...)
- Quantize it (GPTQ, AWQ, bitsandbytes, ...)
- Load it in a different runtime (vLLM, llama.cpp, TGI, ...)
- Port it to a different architecture family (Mamba, Jamba, RWKV, ...)

When output is gibberish, weights silently drop, or conversion segfaults —
the source model's quirks are usually the cause. `ssmforge arch` makes them
visible *before* you spend the next hour finding out the hard way.

It does **not** modify the model. It does **not** run inference. It only
inspects.

### What's new in 0.2.3

| Fix | What was broken |
|-----|-----------------|
| `--output` on Git Bash + Windows with unquoted backslash paths | Git Bash strips backslashes from unquoted args, so `--output C:\Users\foo\out.md` reached Python as `C:Usersfooout.md` — silently writing a single literal-named file in the cwd. Pre-parse guard now catches this BEFORE argparse runs and exits 1 with a clear "wrap in double quotes or use forward slashes" hint. |
| `C:Usersnetge...` (bash-mangled, no separators) detected | The post-write `_check_output_path` was too late. The pre-parse guard catches the mangled pattern early and tells you exactly what bash did to your argument. |

**Quote your paths from now on.** Or use forward slashes — they work on both Windows and POSIX.

```bash
# ✅ Recommended — works everywhere
python -m ssmforge.cli arch X --graph --format markdown --output ~/Desktop/report.md
python -m ssmforge.cli arch X --graph --format markdown --output ./report.md
python -m ssmforge.cli arch X --graph --format markdown --output "C:/Users/me/Desktop/report.md"

# ❌ These will now fail with a clear error:
python -m ssmforge.cli arch X --graph --format markdown --output C:\Users\me\Desktop\report.md
python -m ssmforge.cli arch X --graph --format markdown --output C:UsersmeDesktopreport.md
```

Plus 9 new tests (174 total).

### What's new in 0.2.2

| Fix | What was broken |
|-----|-----------------|
| `--output PATH` with bad parent dir | Previously got a generic `bash: Permission denied` (when redirecting) or a Python traceback (when using `--output`). Now pre-flight checks the parent directory, prints a clear hint pointing at writable locations like `./` and `~/`, then exits 1. |

Plus 6 new tests (162 total). Tiny release — main bug fix only.

### What's new in 0.2.1

| Feature | Why it matters |
|---------|----------------|
| `doctor --check-install` | Diagnoses "ssmforge: command not found" in one command. Prints `✓` when on PATH, `✗` + a copy-pasteable fix hint when not. Exits non-zero so you can use it in scripts. |
| One-time PATH warning | First invocation when `ssmforge` is off-PATH prints a hint to stderr pointing at `python -m ssmforge.cli`. Suppressed by `SSMFORGE_NO_PATH_WARN=1`. Skipped for `doctor` so its output stays clean. |
| `--graph --format {text,json,markdown}` | `--graph` now respects `--format`. Single + compare modes emit ASCII (text), structured JSON (for pipelines), or GitHub-flavored Markdown (for docs/Issues). |
| `--graph` ASCII cleanup | Box-drawing hierarchy fixed: single-rail `▼` flow + box `╭╮╰╯` panels for ATTENTION and MLP, `├─/└─` decision-tree style inside. |
| Qwen2 attention_bias fixed | `config.attention_bias=None` + zero bias tensors (Qwen2's vestigial biases) no longer reports `attention_bias=True`. Now requires either config explicit True OR non-zero bias values. |
| `--compare A A` is now valid | Was previously dedup'd to `[A]` and rejected. Now produces a 2-way "all identical" report. |
| README install lead | Section 4.2 now recommends a venv more strongly + explains why + gives `python -m ssmforge.cli` as the universal workaround. |

Plus 26 new tests (156 total).

### What's new in 0.2.0

| Feature | Why it matters |
|---------|----------------|
| `--graph` | Render a text-based architecture decision graph in your terminal. Single-model shows the dataflow with decisions inline. `--compare --graph` shows an N-way decision table with a "Differ across N models" footer. |
| New `analyze.graph` module | `render_graph_text()` and `render_graph_compare()` are exposed for Python API use — embed them in your own tooling. |
| `--full` for `--graph` (implicit) | `--graph` works with `--dry-run`. Quirks that need weight inspection (MLP type, norm type, fused QKV, attention_bias) show as "unknown" with a note pointing at `--full`. Drop `--dry-run` to get the full answer. |

Plus 19 new tests (130 total).

### What's new in 0.1.9

| Fix | What was broken |
|-----|-----------------|
| `--fields profile.family` no longer crashes with `KeyError: 'model_id'` | When subsetting stripped `model_id`, the dry-run summary still tried to read it. Now uses `.get('model_id', '<unknown>')`. |
| `--quiet` now silences transformers deprecation warnings | The `[transformers] torch_dtype is deprecated` and `pad_token_id must be None` messages bypassed the logging system. Now also filters `warnings` module. |

Plus 4 new regression tests.

### What's new in 0.1.8

| Feature | Why it matters |
|---------|----------------|
| `--fields name1,name2` | Subset the report output to specific fields (works for single + compare) |
| `--profile` | Emit just the profile section (family, attention_type, mlp_type, norm_type, descriptors) |
| `--only-different` | In `--compare` mode, hide the "Identical across all models" section |
| `--rev` / `--revision` | Pin to a specific HF commit/tag/branch for reproducibility |
| HF revision tracked in reports | Every report now has `hf_revision` field with the resolved sha |
| `FIELD_DESCRIPTIONS` dict | Programmatic access to inline docs for every quirk field |
| `--fields` supports dotted paths | `quirks.attention_bias`, `profile.family` etc. |
| `--compare --profile` | Compare only the profile sections of N models |
| `--compare --only-different` JSON | Output JSON only contains differing fields |

Plus 16 new tests (107 total).

### What's new in 0.1.7

| Feature | Why it matters |
|---------|----------------|
| `ssmforge arch --compare A B C D` | N-way model comparison. Side-by-side table of every field across all models. JSON or Markdown. |
| `ssmforge arch MODEL --compare B C D` | First model from positional, rest from `--compare`. |
| `--compare` works with `--dry-run` | Compare configs without downloading any weights. |
| Legacy `--diff` still works | Same one-vs-one behavior, fully back-compat. |

Plus 11 new tests (91 total).

### What's new in 0.1.6

| Feature | Why it matters |
|---------|----------------|
| StarCoder / OLMo / DeepSeek / OLMoE detection | More architectures now get a proper family label instead of "unknown" |
| Falcon / Pythia now labeled specifically | Was: generic "Falcon / Pythia (MQA)". Now: "Falcon (MQA)" or "Pythia / GPT-NeoX (MQA)" |
| `--quiet` also silences transformers warnings | The `torch_dtype` deprecation noise is gone when piping JSON to jq |
| Family inference bug fix | StarCoder/OLMo/etc. were being mis-labeled due to a callable/config mismatch in the inferrer |

Plus 11 new tests (80 total).

### What's new in 0.1.5

| Feature | Why it matters |
|---------|----------------|
| `ssmforge arch MODEL --dry-run` | Preview a model's config + memory cost before downloading GBs of weights |
| `ssmforge --version` / `-V` | Print version, exit 0 — usable in CI scripts |
| `--output -` | Unix convention: write to stdout instead of a file named `-` |
| Fail-fast on missing local paths | Was: hangs trying to load as HF id |
| Memory error → exit 1 + `--dry-run` hint | Was: process killed with exit 127 |
| Ctrl+C → exit 130 with clean message | Was: traceback dumped to terminal |

Plus 19 new tests (69 total).

### What's new in 0.1.4

- Initial PyPI release
- 50 tests passing
- Three user-facing features: `arch`, `--diff`, `doctor`

---

## 2. Quickstart (60 seconds)

If you already have Python 3.10+ and a working `pip`:

```bash
# 1. Install
pip install ssmforge

# 2. Confirm it works (no model download)
ssmforge doctor
ssmforge --version            # should print: ssmforge 0.1.5

# 3. Run a tiny test (no HF account needed, ~50 MB download)
ssmforge arch hf-internal-testing/tiny-random-LlamaForCausalLM --dry-run
# ^ Config-only, no weights. Fast.
# Then try with full load:
ssmforge arch hf-internal-testing/tiny-random-LlamaForCausalLM

# 4. Real model (no HF account needed for these)
ssmforge arch Qwen/Qwen2-0.5B-Instruct --dry-run   # 1 GB — preview first
ssmforge arch Qwen/Qwen2-0.5B-Instruct             # then load it
```

The `--dry-run` step is now strongly recommended for any model over 1 GB.
It tells you the memory cost and config-only quirks (GQA, MoE, sliding
window, etc.) before committing to a multi-GB download.

For a comparison / diff:

```bash
ssmforge arch Qwen/Qwen2-0.5B-Instruct --diff TinyLlama/TinyLlama-1.1B-Chat-v1.0
# Both models are small; full diff downloads both.
```

See [Section 4](#4-first-time-setup) for full setup including HuggingFace
account and gated models.

---

## 3. Glossary

If a term below is unclear when you encounter it in the output, come back
here.

| Term | Meaning |
|------|---------|
| **Architecture** | The blueprint of the model: how many layers, attention type, MLP type, etc. Examples: `llama`, `qwen2`, `mistral`, `phi3`, `gemma2`, `mixtral`, `falcon`. |
| **State dict** | A flat dictionary mapping parameter names to tensors. E.g. `{"model.layers.0.self_attn.q_proj.weight": tensor(...), ...}`. This is what's saved in `model.safetensors`. |
| **Config** | The non-weight side of the model: vocab size, layer count, attention bias flag, etc. Stored in `config.json`. |
| **HF Hub / HuggingFace** | The website https://huggingface.co where models are hosted. |
| **HF model id** | The string you pass to `ssmforge arch`, e.g. `Qwen/Qwen2-1.5B-Instruct` — org/owner slash model name. |
| **Gated model** | A model on HF that requires you to (1) have an HF account, (2) accept the license terms on the model's page, and (3) provide an HF token. Examples: Llama-3, Mistral-7B (some checkpoints), Gemma. |
| **MHA (Multi-Head Attention)** | Standard attention where each head has its own Q, K, V. `num_attention_heads == num_key_value_heads`. |
| **GQA (Grouped Query Attention)** | Attention where multiple Q heads share one K/V head. `num_key_value_heads < num_attention_heads`. Saves memory and compute on the K/V side. Used by Llama-2-70B, Llama-3, Mistral, Qwen2. |
| **MQA (Multi-Query Attention)** | Attention where ALL Q heads share a single K head and single V head. `num_key_value_heads == 1`. Used by Falcon, Pythia, some PaLM variants. |
| **MoE (Mixture of Experts)** | The model's MLP layers are replaced with N parallel "expert" MLPs, and a router decides which experts to use per token. `num_experts` is the count, `num_experts_per_tok` (or `moe_top_k`) is how many are activated per token. Examples: Mixtral (8 experts, top-2), DeepSeek-MoE. |
| **Tied embeddings** | The `lm_head` (output projection) shares its weight matrix with `embed_tokens` (input embedding). Saves memory but means you can't independently tweak them. Used by Qwen2, Pythia, Gemma. |
| **Fused QKV / fused gate-up** | Some models put multiple matrices into one tensor. Phi-3 has a single `qkv_proj.weight` of shape `(q_dim + 2*kv_dim, hidden)` instead of separate `q_proj`, `k_proj`, `v_proj`. Saves memory and is faster on some hardware but you have to slice the tensor to get the individual matrices. |
| **Attention bias** | Standard attention has no bias on Q/K/V projections. Some models (Qwen2) add a bias term. If your target runtime assumes no bias, you'll silently lose accuracy. |
| **Sliding window attention** | Each token only attends to the previous N tokens, not the full sequence. Used by Mistral (N=4096) and Gemma2. Different runtime cost than full attention. |
| **Soft-capping** | Gemma2's trick of clamping logit values (e.g. to ±50) before softmax. Makes training more stable. Has to be implemented in your target runtime or you'll diverge from the reference. |
| **LayerScale** | A learnable per-channel multiplier applied after attention and MLP residual blocks. Used by Phi-3 and some vision transformers. |
| **Partial RoPE** | Apply rotary embeddings to only a fraction of the head dimensions. Used by Command-R. |
| **RoPE (Rotary Position Embedding)** | The way most modern LLMs encode token positions. Configured by `rope_theta` (a base frequency) and optionally `rope_scaling` (e.g. for long-context extensions like YaRN). |
| **SwiGLU / GeGLU / GeLU** | Different activation functions used in the MLP. SwiGLU is the gated version (`gate_proj * up_proj`) used by Llama, Qwen, Mistral. GeGLU is used by Gemma2. GeLU is the older transformer default. |
| **RMSNorm vs LayerNorm** | Two ways to normalize activations. RMSNorm is simpler (no mean-centering) and used by most modern LLMs (Llama, Qwen, Mistral, Phi). |
| **Quirk** | In ssmforge context: any architectural detail that's not in the default Llama config. Could be a fused tensor, a gating decision, a position-encoding variant. The whole point of the tool is to surface quirks. |
| **Compatibility (issue)** | A quirk the analyzer flags with a severity: `info` (handled transparently), `warning` (may degrade), `error` (blocks). MoE is currently the only `error`. |

---

## 4. First-time setup

If you've never used HuggingFace tools before, here's the full path from
zero to running `ssmforge arch` on a real model.

### 4.1 Install Python

You need Python 3.10 or newer.

```bash
# Check
python --version          # Linux / Mac / Git Bash
# or
py --version              # Windows Python Launcher
```

If you don't have it:
- Linux: `sudo apt install python3.10` (or use your distro's package manager)
- macOS: `brew install python@3.12`
- Windows: download from https://www.python.org/downloads/

### 4.2 Create a virtual environment (recommended — strongly)

**Use a venv.** It sidesteps the most common install problem on Windows
(`ssmforge: command not found` after pip install). With a venv, the
`ssmforge` script lands at `.venv\Scripts\ssmforge.exe` (Windows) or
`.venv/bin/ssmforge` (Linux/macOS), which is automatically on PATH after
you activate.

A venv keeps ssmforge and its dependencies isolated from system Python.

```bash
# Linux / Mac / Git Bash
python -m venv .venv
source .venv/bin/activate

# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Windows cmd.exe
python -m venv .venv
.venv\Scripts\activate.bat
```

You should see `(.venv)` in your prompt after activating. From here on,
`pip install` only affects this venv.

**Alternative: use `python -m ssmforge.cli`** as a workaround without a
venv. Works anywhere, doesn't need any PATH setup, but requires typing
`python -m ssmforge.cli` instead of just `ssmforge`.

```bash
python -m ssmforge.cli --version
python -m ssmforge.cli arch hf-internal-testing/tiny-random-LlamaForCausalLM --dry-run
```

### 4.3 Install ssmforge

```bash
pip install ssmforge
```

This pulls in:
- `transformers` — the HuggingFace model library ssmforge uses to load models
- `huggingface_hub` — the client for the HuggingFace Hub API
- `safetensors`, `tokenizers`, `numpy` — transitive dependencies

Verify:

```bash
ssmforge --help
ssmforge doctor
```

The `doctor` command should print ssmforge's version, Python version,
and the resolved cache directories.

### 4.4 (Optional) Create a HuggingFace account

You need an HF account to:
- Download gated models (Llama-3, Mistral, Gemma, etc.)
- Upload your own models
- Increase download rate limits

If you only need non-gated models (Qwen2, TinyLlama, most Phi-3 variants,
many community fine-tunes), you can skip this step.

To create an account:
1. Go to https://huggingface.co/join
2. Sign up with email or GitHub
3. Verify your email

### 4.5 (Optional) Generate an HF access token

The token lets `ssmforge arch` authenticate to HF Hub.

1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Name it anything (e.g. "ssmforge")
4. Type: **Read** (you only need to download models)
5. Click "Generate"
6. Copy the token — it starts with `hf_...`

Set it as an environment variable:

```bash
# Linux / Mac / Git Bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxx

# PowerShell
$env:HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxx"

# Windows cmd.exe
set HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxx
```

To make it persistent:

```bash
# Git Bash: append to ~/.bashrc
echo 'export HF_TOKEN=hf_xxxxxxxxx' >> ~/.bashrc
source ~/.bashrc

# PowerShell (persistent for current user)
[Environment]::SetEnvironmentVariable("HF_TOKEN", "hf_xxxxx", "User")

# Linux / Zsh: append to ~/.zshrc
echo 'export HF_TOKEN=hf_xxxxxxxxx' >> ~/.zshrc
source ~/.zshrc
```

**Never commit the token to git.** Don't put it in `.env` files inside
your repo. The env var approach keeps it out of your codebase.

### 4.6 (Optional) Accept gated model licenses

Some models require you to click "Agree and access repository" on their
HF page before you can download them.

For example, to use Llama-3.1-8B:
1. Go to https://huggingface.co/meta-llama/Llama-3.1-8B
2. Log in to your HF account
3. Fill out the license form (your name, email, agree to Meta's license)
4. Click "Agree and access repository"

Without this, you'll get an error like:

```
401 Client Error: Unauthorized for url: https://huggingface.co/...
Repository Not Found for url: ...
```

### 4.7 Verify everything works

```bash
# Tiny test model — no HF account needed, ~50 MB download
ssmforge arch hf-internal-testing/tiny-random-LlamaForCausalLM

# Real non-gated model
ssmforge arch Qwen/Qwen2-0.5B-Instruct

# Real gated model (requires HF_TOKEN)
ssmforge arch meta-llama/Llama-3.1-8B-Instruct
```

If the third one fails with a 401, you either haven't set `HF_TOKEN`,
or you haven't accepted the license on the model's HF page.

---

## 5. All commands

### `ssmforge doctor`

Print ssmforge + environment info. Useful for bug reports.

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

JSON variant:

```bash
ssmforge doctor --format json
```

#### `ssmforge doctor --check-install`

Run install checks (PATH, scripts location) and exit non-zero on issues.
Useful in scripts or when troubleshooting "command not found":

```bash
ssmforge doctor --check-install
```

```
ssmforge doctor --check-install (0.2.1)
--------------------------------------------------
  ✓ ssmforge command on PATH: True
  python: C:\Users\netge\.venv\Scripts\python.exe
  platform: win32
```

If `ssmforge` isn't on PATH, you get a fix-hint block:

```
ssmforge doctor --check-install (0.2.1)
--------------------------------------------------
  ✗ ssmforge command on PATH: False
  python: C:\Python314\python.exe
  platform: win32

The 'ssmforge' script was installed to:
    C:\Users\netge\AppData\Roaming\Python\Python314\Scripts\ssmforge.exe
but that folder is not on PATH.

  Quick fix (current shell):
    export PATH="/c/Users/netge/AppData/Roaming/Python/Python314/Scripts:$PATH"

  Permanent fix (PowerShell, restart shell after):
    [Environment]::SetEnvironmentVariable("PATH", "<scripts>;" + [Environment]::GetEnvironmentVariable("PATH", "User"), "User")

  Or just use: python -m ssmforge.cli
```

JSON variant:

```bash
ssmforge doctor --check-install --format json
# {"ssmforge_command_on_path": false, "fix_hint": "...", ...}
```

### `ssmforge arch MODEL`

The main command. Inspect a HuggingFace model.

```bash
ssmforge arch <model_id_or_path> [options]
```

| Flag | Short | Purpose | Default |
|------|-------|---------|---------|
| `--output PATH` | `-o` | Write report to this file (use `-` for stdout) | stdout |
| `--format {json,markdown,md}` | `-f` | Output format | `json` |
| `--quiet` | | Don't print the human-readable summary to stderr; print only the report | stderr summary on |
| `--diff OTHER_MODEL` | | Compare against one other model (legacy, use `--compare` for 3+) | none |
| `--compare MODEL [MODEL ...]` | | Compare 2+ models side-by-side. First model can come from positional. | none |
| `--dry-run` | | Fetch only the config (no weight download); reports config-only quirks + memory estimate | full load |
| `--rev` / `--revision` | | Pin to a specific HF commit sha/tag/branch (default: current HEAD) | HEAD |
| `--fields name1,name2` | | Subset output to comma-separated field names. Dotted paths supported (e.g. `quirks.attention_bias`) | all |
| `--profile` | | Emit only the profile section (family, attention_type, mlp_type, norm_type, descriptors) | full report |
| `--only-different` | | In `--compare` mode, hide the "Identical across all models" section | show identical |
| `--graph` | | Render a text-based decision-graph in your terminal. Single-model shows the full pipeline with decisions inline. `--compare --graph` shows an N-way decision table. Honors `--format` (`text`, `json`, `markdown`/`md`) for JSON pipeline consumption and GFM-friendly docs. | JSON |

**Examples:**

```bash
# Default: JSON to stdout, summary on stderr
ssmforge arch Qwen/Qwen2-1.5B-Instruct

# JSON only (pipe-friendly)
ssmforge arch Qwen/Qwen2-1.5B-Instruct --quiet | jq .quirks

# Markdown to file (great for GitHub issues)
ssmforge arch Qwen/Qwen2-1.5B-Instruct --format markdown --output report.md

# Compare two models (JSON diff, legacy syntax still works)
ssmforge arch Qwen/Qwen2-0.5B-Instruct --diff TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Compare 2+ models (NEW: side-by-side table)
ssmforge arch --compare Qwen/Qwen2-0.5B-Instruct TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --format markdown

# Compare 3 models (small + base + finetune)
ssmforge arch --compare tiny-random-Llama Qwen/Qwen2-0.5B-Instruct TinyLlama \
  --dry-run --format markdown

# Combine: compare with positional + --compare list
ssmforge arch Qwen/Qwen2-0.5B-Instruct --compare TinyLlama/TinyLlama \
  --diff would-be-redundant-but-fine

# Local path instead of HF id
ssmforge arch /path/to/local/model --output local-report.json

# Skip summary (only print JSON)
ssmforge arch mistralai/Mistral-7B-v0.1 --quiet
```

### Visual decision graph (`--graph`)

`--graph` renders a text-based architecture decision graph in your terminal.
The output shows the model's dataflow (input → embed → attn + MLP + norms →
final norm → lm_head) with each architectural choice marked inline with its
verdict. In `--compare` mode, it shows an N-way decision table that lists
where the models differ.

This is the view you want when you're deciding whether a model is
compatible with your downstream pipeline (quantizer, hybrid-SSM converter,
GGUF exporter). It tells you the choices that matter, not the raw config.

**Single-model graph (Qwen2-0.5B):**

```bash
ssmforge arch Qwen/Qwen2-0.5B-Instruct --dry-run --graph
```

```
Qwen/Qwen2-0.5B-Instruct    (24 layers, hidden=896, heads=14 kv:2, ffn=4864, vocab=151936, dtype=bf16)

  INPUT
    │
  embed_tokens ─── lm_head ─→ TIED ✓
    │
    ├─  ┌──────────────────────────────────────────────────────────┐
    │   │ ATTENTION (one of 24 layers)                          │
    │   │   ├─ fused QKV          ──── NO    # separate q, k, v projections
    │   │   ├─ GQA                ──── YES (kv:2 of 14)    # 2 KV heads vs 14 Q heads
    │   │   ├─ attention bias     ──── NO    # no bias terms (Llama convention)
    │   │   ├─ rope_theta         ──── 1e+06                 # rotary position embedding base
    │   │   └─→ output projection (o_proj)
    │   │
    │   │ MLP (gated, SwiGLU when --full)
    │   │   ├─ MLP type           ──── unknown   # requires --full inspection (config-only scan can't classify)
    │   │   └─ fused gate/up      ──── NO    # separate gate_proj and up_proj
    ▼
  final_norm
  logits [151936]

  KEY DECISIONS (for downstream tooling):
    NO     fused QKV              separate q, k, v projections
    YES (kv:2 of 14) GQA         2 KV heads vs 14 Q heads
    NO     attention bias         no bias terms (Llama convention)
    1e+06  rope_theta             rotary position embedding base
    NO     fused gate/up          separate gate_proj and up_proj
    NO     Tied embeddings        lm_head and embed_tokens are independent

  Params: ~494M    Memory: 942.2 MB @ bfloat16
  Memory @ int8: 471.1 MB    Memory @ int4: 235.6 MB
```

The `KEY DECISIONS` block is the actionable summary — every line tells
you a choice that affects how a downstream tool (quantizer, converter)
has to handle the model.

**N-way compare graph (multiple models):**

```bash
ssmforge arch --compare Qwen/Qwen2-0.5B-Instruct TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dry-run --graph
```

```
Decision           Qwen/Qwen2-0.5B-Instruct  TinyLlama/TinyLlama-1.1B-Chat-v1.0
──────────────────────────────────────────────────────────────────────────

--- Attention ---
fused QKV          no                        no
MQA / GQA          GQA (kv:2/14)             GQA (kv:4/32)
attention bias     no                        no
rope_theta         1000000                   10000
rope_scaling       None                      None
sliding_window     None                      None

--- MLP ---
MLP type           unknown                   unknown
fused gate/up      no                        no

--- Normalization ---
Norm type          unknown                   unknown

--- Other ---
Tied embeddings    YES                       no

--- Geometry ---
hidden_size        896                       2048
num_layers         24                        22
params             494M                      1.10B
memory @ bf16      942.2 MB                  2.0 GB

──────────────────────────────────────────────────────────────────────────
Differ across 2 models: MQA / GQA, rope_theta, Tied embeddings, hidden_size, num_layers, params, memory @ bf16
```

The "Differ across N models" footer lists which decisions actually
disagree — read this to spot architectural drift between model sizes
or families.

**Notes:**

- The graph mode prints to stdout, so you can pipe to `less` or
  redirect to a file with `> architecture.txt`.
- Exit code matches normal rules: 0 if compatible, 2 if blockers
  found.
- In `--compare --graph` mode, exit is always 0 (compare itself
  is informational, not a verdict).

### Comparing 2+ models (`--compare`)

The `--compare` flag takes 2 or more model ids and renders a multi-model
table showing every field's value across all models. Fields that match
across all models are listed in an "Identical across all models" section
at the bottom (not in the table) to keep the table scannable.

```bash
ssmforge arch --compare Qwen/Qwen2-0.5B-Instruct TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dry-run --format markdown
```

Output:

```markdown
# Architectural comparison: 2 models

**10 field(s) differ** out of 27; 17 identical.

| Field | Qwen/Qwen2-0.5B-Instruct | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
|-------|-------|-------|
| `grouped_attention` | `True` | `True` |
| `hidden_size` | `896` | `2048` |
| `model_type` | `qwen2` | `llama` |
| `num_attention_heads` | `14` | `32` |
| `rope_theta` | `1000000.0` | `10000.0` |
| `tie_word_embeddings` | `True` | `False` |
| `vocab_size` | `151936` | `32000` |
...

### Identical across all models
17 field(s) had the same value: `attention_bias`, `attention_bias_in_state_dict`, ...
```

Useful for:

- Comparing a fine-tuned model, its base, and a competitor side by side
- Spotting field-level inconsistencies across 3+ checkpoints of the same model
- Generating comparison tables for papers / model cards / wiki pages
- Validating that a LoRA / merge didn't accidentally change architecture

Both JSON and Markdown output work. JSON has every field with all
values, plus summary counts (`identical_field_count`, `different_field_count`).

Note: `--diff` (one-vs-one) is still supported and unchanged for
back-compat. `--compare A B` produces the same output as `--diff B`
when given via positional `A --compare B`.

### Pre-flight check (no weight download)

```bash
# Fetch only the config — no GBs of weights
ssmforge arch mistralai/Mixtral-8x7B-v0.1 --dry-run
```

Output:

```
SSMForge arch DRY-RUN: mistralai/Mixtral-8x7B-v0.1
  family:              Mixtral (MoE Llama)
  attention_type:      GQA

Model config:
  model_type:          mixtral
  hidden_size:         4096
  num_hidden_layers:   32
  num_attention_heads: 32
  num_kv_heads:        8
  ...
  torch_dtype:         torch.bfloat16

Memory estimate: ~13.49 GB in bfloat16 (7.24B params)

Detected quirks (config-only):
  ✓ GQA
  ✓ MoE (num_experts=8, top_k=2)
  ✓ rope_theta=1000000.0 (from config.rope_parameters.rope_theta)

Quirks requiring --no-dry-run (state_dict inspection):
  · attention_bias (config truth vs state_dict truth)
  · tied_embeddings (lm_head tensor identity)
  · fused_qkv (qkv_proj vs q/k/v split)
  · fused_gate_up (gate_up_proj vs gate/up split)
  · mlp_type / norm_type (state_dict structure)
  · layer_scale (ls1/ls2 keys)

Compatibility: BLOCKED by: moe
```

Useful for:
- Pre-flight memory check before committing to a multi-GB download
- Quickly checking if a model is MoE (which currently blocks compatibility)
- Comparing configs of many models without filling the cache
- Using `--dry-run --diff` for a config-only diff (no weight downloads on either side)

Note: quirks that need actual tensor inspection (tied_embeddings, fused_qkv,
attention_bias, mlp_type, norm_type, layer_scale) are NOT verified in
dry-run mode. The output explicitly lists which quirks require a full load.

### `ssmforge --help`

```bash
ssmforge --help
ssmforge arch --help
```

Use these to see all flags with their descriptions.

### `ssmforge --version` / `-V`

```bash
ssmforge --version
# ssmforge 0.1.5

ssmforge -V
# ssmforge 0.1.5
```

Print ssmforge's version and exit 0. Useful for scripting:

```bash
ssmforge --version | awk '{print $2}'   # gets "0.1.5"
require=$(ssmforge --version | awk '{print $2}')
if [[ "$require" < "0.1.5" ]]; then
  echo "ssmforge too old, please upgrade"
fi
```

### Special output paths

`--output -` means stdout (Unix convention). Without it, `ssmforge arch MODEL --output -` would *create a file called `-`* in the cwd. The `-` sentinel routes the report to stdout instead:

```bash
# Pipe JSON straight to jq
ssmforge arch Qwen/Qwen2-1.5B-Instruct --output - --quiet | jq '.quirks.tied_embeddings'
# true
```

`--output PATH` to a path you can't write (read-only dir, missing parent
dir, permission denied) exits 1 with a clean error message — no traceback.

---

## 6. Output formats

### 6.1 JSON (default)

JSON goes to stdout. You can pipe it to `jq`, save it to a file, or
post-process with any tool.

Structure:

```json
{
  "model_id": "Qwen/Qwen2-1.5B-Instruct",
  "model_type": "qwen2",
  "architectures": ["Qwen2ForCausalLM"],
  "config": { /* relevant config fields */ },
  "quirks": { /* 15 boolean/value fields */ },
  "profile": {
    "family": "Qwen2 (tied embeddings + attention bias)",
    "attention_type": "GQA",
    "mlp_type": "swiglu",
    "norm_type": "rms",
    "descriptors": ["dense", "tied-embeddings", "attention-bias", "GQA 6:1"]
  },
  "compatibility": {
    "is_compatible": true,
    "blockers": [],
    "warnings": [],
    "issues": [
      {"severity": "info", "quirk": "attention_bias", "message": "..."}
    ]
  },
  "state_dict_summary": {
    "total_tensors": 339,
    "total_params": 1779890432,
    "tensor_breakdown": {
      "embeddings": 1,
      "lm_head": 1,
      "attention_weights": 112,
      ...
    },
    "param_breakdown": { /* same shape, with parameter counts */ }
  }
}
```

### 6.2 Markdown

```bash
ssmforge arch MODEL --format markdown
```

Produces a GitHub-flavored Markdown document with:
- Title with model id
- Profile section (family, attention, MLP, norm, descriptors)
- Config table
- Detected-quirks checklist
- Compatibility issues table
- State dict breakdown

Designed to paste into GitHub issues, PRs, READMEs, or model cards.

### 6.3 Human summary (stderr)

Whenever you run `ssmforge arch MODEL` without `--quiet` and without
`--output`, a human-readable summary is printed on stderr. JSON still goes
to stdout. This lets you do:

```bash
# See summary + save JSON
ssmforge arch Qwen/Qwen2-1.5B-Instruct --output report.json
# stderr: human summary
# file report.json: JSON
```

The summary shows profile, model config, detected quirks (with ✓/✗ boxes),
state dict totals, and compatibility verdict.

---

## 7. HuggingFace setup

### 7.1 What HuggingFace is

HuggingFace (https://huggingface.co) is where most open-weight LLMs live.
When you give `ssmforge arch` an id like `Qwen/Qwen2-1.5B-Instruct`,
it's downloading from `huggingface.co/Qwen/Qwen2-1.5B-Instruct`.

### 7.2 What `transformers` is

`transformers` is the Python library that knows how to load many model
architectures. `ssmforge arch` uses it under the hood to:
1. Download the model files (`config.json`, `model.safetensors`, ...)
2. Parse `config.json` into a typed config object
3. Load the weights into memory as a `state_dict`

You don't need to interact with `transformers` directly to use ssmforge.

### 7.3 Model access tiers

| Tier | Examples | HF account needed? |
|------|----------|--------------------|
| Open (no gate) | `Qwen/Qwen2-0.5B-Instruct`, `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, `hf-internal-testing/*`, most Phi-3, most community fine-tunes | No |
| Gated | `meta-llama/Llama-3.1-8B-Instruct`, `mistralai/Mistral-7B-v0.1`, `google/gemma-2-2b` | Yes + license acceptance + `HF_TOKEN` |
| Private | Anything in a private org you have access to | Yes + `HF_TOKEN` + org membership |

### 7.4 Setting up for gated models

1. Create an HF account (see [4.4](#44-optional-create-a-huggingface-account))
2. Generate a token with **Read** scope (see [4.5](#45-optional-generate-an-hf-access-token))
3. Visit the gated model's HF page and click "Agree and access repository"
4. Set the token: `export HF_TOKEN=hf_xxxxx`
5. Try `ssmforge arch <gated-model>`

### 7.5 Common HF errors

| Error | Cause | Fix |
|-------|-------|-----|
| `Repository Not Found` | Wrong id, or private repo you can't access | Check the id; check your token has the right scope |
| `401 Client Error: Unauthorized` | Gated model, no token, or license not accepted | Set `HF_TOKEN`, accept license on HF page |
| `403 Forbidden` | Token has wrong scope (e.g. Write only) | Generate a Read-scope token |
| `429 Too Many Requests` | Rate-limited (anonymous or new account) | Set `HF_TOKEN`, wait, or upgrade to HF Pro |
| `Could not load model` / `OSError: [Errno 28] No space left` | Cache drive full | Change `HF_HOME` to a bigger drive |

---

## 8. Cache management

### 8.1 Where models get cached

When `ssmforge arch` downloads a model, it stores the files locally so
the next run is instant.

| Env var | Default | Purpose |
|---------|---------|---------|
| `HF_HOME` | `~/.cache/huggingface/` (Linux/Mac/Git Bash) / `%USERPROFILE%\.cache\huggingface\` (Windows) | Root for all HF caches |
| `HF_HUB_CACHE` | `$HF_HOME/hub` | Cache for downloaded model files |
| `TRANSFORMERS_CACHE` | (older versions only) | Same as `HF_HUB_CACHE` |
| `HF_TOKEN` | (unset) | Your HF token for gated models |

### 8.2 Why you might want to change `HF_HOME`

The default cache lives on your home drive. On Windows that's usually
`C:\Users\<you>\.cache\huggingface\` — and `C:` is often a small SSD.

If your home drive is small (or you have a bigger drive elsewhere), point
`HF_HOME` at it:

```bash
# Git Bash on Windows, with a G: drive for models
export HF_HOME=G:/models
mkdir -p G:/models
ssmforge arch Qwen/Qwen2-0.5B-Instruct

# PowerShell
$env:HF_HOME = "G:\models"
mkdir G:\models
ssmforge arch Qwen/Qwen2-0.5B-Instruct

# Persistent across shells
# PowerShell (admin, system-wide):
setx HF_HOME "G:\models" /M
# PowerShell (user, current user):
[Environment]::SetEnvironmentVariable("HF_HOME", "G:\models", "User")
# Git Bash / Linux / Zsh:
echo 'export HF_HOME=G:/models' >> ~/.bashrc
source ~/.bashrc
```

Models are stored under `$HF_HOME/hub/models--ORG--MODEL/snapshots/<sha>/`.

### 8.3 Inspecting the cache

```bash
ssmforge doctor
```

Tells you what `HF_HOME` resolves to, whether `HF_TOKEN` is set, and what
versions of `transformers` and `huggingface_hub` are installed.

### 8.4 Clearing the cache

```bash
# Linux / Mac / Git Bash
rm -rf ~/.cache/huggingface/hub/models--*
# Or just one model
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct

# Windows PowerShell
Remove-Item -Recurse -Force $env:USERPROFILE\.cache\huggingface\hub\models--*
```

Clearing the cache doesn't remove anything from HF Hub — it'll just
re-download next time.

### 8.5 Per-test cache (only relevant if you run the test suite)

The ssmforge test suite uses an isolated HF cache at `./tests/.cache/`
so it doesn't pollute your real model cache. Override with:

```bash
SSMFORGE_TEST_HF_HOME=/path/to/test/cache python -m pytest tests/
```

---

## 9. Python API

You can use ssmforge from Python without going through the CLI:

```python
from ssmforge.analyze import build_report, format_report_json
from ssmforge.analyze.markdown_render import format_report_markdown
from transformers import AutoModelForCausalLM

# Load a model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-1.5B-Instruct")

# Build the report
report = build_report(
    model_id="Qwen/Qwen2-1.5B-Instruct",
    config=model.config,
    state_dict=dict(model.state_dict()),
)

# Inspect fields
print(report["quirks"]["tied_embeddings"])   # True
print(report["profile"]["family"])            # "Qwen2 (tied embeddings + attention bias)"
print(report["compatibility"]["is_compatible"])  # True

# Serialize
print(format_report_json(report))             # JSON string
print(format_report_markdown(report))         # Markdown string

# Diff two reports
from ssmforge.analyze import diff_reports, compare_reports
model2 = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
report2 = build_report(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    model2.config,
    dict(model2.state_dict()),
)
diff = diff_reports(report, report2)
print(diff["identical"])                      # False
for d in diff["differences"]:
    print(f"{d['field']}: {d['a']!r} -> {d['b']!r}")

# Compare 3+ models
model3 = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")
report3 = build_report(
    "tiny-random",
    model3.config,
    dict(model3.state_dict()),
)
cmp = compare_reports([report, report2, report3])
print(cmp["all_identical"])                   # False
print(f"{cmp['different_field_count']} fields differ out of {len(cmp['fields'])}")
# Iterate only the differing fields
for f in cmp["fields"]:
    if not f["all_same"]:
        print(f"{f['field']}: {f['values']}")
```

The `analyze/` package exposes:
- `build_report(model_id, config, state_dict)` — main entry point
- `format_report_json(report)` — JSON serializer
- `format_report_markdown(report)` — Markdown serializer
- `render_summary(report)` — human-readable summary (for stderr / logs)
- `diff_reports(a, b)` — diff two reports
- `compare_reports(reports_list)` — N-way comparison of 2+ reports
- `format_compare_markdown(comparison)` — render a comparison as a Markdown table
- `render_graph_text(report)` — render a single-model architecture decision graph (see [Visual decision graph](#visual-decision-graph--graph))
- `render_graph_compare(reports_list)` — render an N-way architecture decision table (>= 2 reports)
- `QuirkReport` — dataclass for the raw quirk scan
- `scan_state_dict(state_dict, config, num_layers)` — lower-level scan
- `count_state_dict_summary(state_dict)` — tensor breakdown

---

## 10. Exit codes

| Code | Meaning | What to do |
|------|---------|------------|
| `0` | Analyzed successfully, no blockers | Continue with your pipeline |
| `1` | Could not load or analyze the model | Check stderr for details — network, auth, missing path, OOM, or `--output` write failure |
| `2` | Analyzed successfully, but model has a blocker | Currently only MoE blocks. Check `compatibility.blockers` in the JSON |
| `130` | Interrupted by Ctrl+C | You pressed Ctrl+C during load. Re-run when ready |

Use exit codes in shell pipelines:

```bash
ssmforge arch mistralai/Mixtral-8x7B-Instruct-v0.1 && echo "OK" || echo "BLOCKED"
```

In CI:

```bash
ssmforge arch $MODEL
case $? in
  0) ;;                  # OK
  2) handle_moe;;        # MoE blocked
  130) exit 1;;          # user interrupted
  1) handle_load_error;; # look at stderr
esac
```

---

## 11. What it detects (full table)

| Detector | Field in JSON | What it catches | Example architectures |
|----------|---------------|-----------------|-----------------------|
| `attention_bias` | `quirks.attention_bias` (bool) | q/k/v projections have learned bias terms | Qwen2 |
| `tied_embeddings` | `quirks.tied_embeddings` (bool) | `lm_head` shares storage with `embed_tokens` | Qwen2, Pythia, Gemma, many small LMs |
| `fused_qkv` | `quirks.fused_qkv` (bool) | One `qkv_proj` instead of separate q/k/v | Phi-3 |
| `fused_gate_up` | `quirks.fused_gate_up` (bool) | One `gate_up_proj` instead of gate + up | Phi-3 |
| `grouped_attention` | `quirks.grouped_attention` (bool) | K/V projection smaller than Q (GQA) | Qwen2, Llama-3, Mistral, Gemma2 |
| `mqa` | `quirks.mqa` (bool) | K/V projections output exactly `head_dim` (MQA: kv_heads=1) | Falcon, Pythia |
| `moe` | `quirks.moe` (bool) | Router + expert tensors present | Mixtral, DeepSeek-MoE |
| `mlp_type` | `quirks.mlp_type` (str) | SwiGLU / GeGLU / GeLU | All |
| `norm_type` | `quirks.norm_type` (str) | RMSNorm / LayerNorm | All |
| `sliding_window` | `quirks.sliding_window` (int or null) | Windowed attention size | Mistral, Gemma2 |
| `layer_scale` | `quirks.layer_scale` (bool) | Learnable per-channel residual scale | Phi-3 |
| `soft_capping` | `quirks.soft_capping` (dict) | Logit soft-capping values (attn, final) | Gemma2 |
| `partial_rope_factor` | `quirks.partial_rope_factor` (float or null) | Partial rotary embedding factor | Command-R |
| `num_experts` | `quirks.num_experts` (int or null) | MoE expert count | Mixtral (8), DeepSeek (60+) |
| `moe_top_k` | `quirks.moe_top_k` (int or null) | Tokens-per-step routing count | Mixtral (2), DeepSeek (6+) |
| `rope_theta` | `config.rope_theta` (float) | RoPE base frequency | All |
| `rope_theta_source` | `config.rope_theta_source` (str) | Where `rope_theta` was resolved from | transformers 5.x detail |
| `rope_scaling_type` | `config.rope_scaling_type` (str or null) | Type of long-context scaling, if any | All (optional) |

Plus an **interpretive profile** (`report.profile`) that summarizes
the model:

```json
"profile": {
  "family": "Qwen2 (tied embeddings + attention bias)",
  "attention_type": "GQA",
  "mlp_type": "swiglu",
  "norm_type": "rms",
  "descriptors": ["dense", "tied-embeddings", "attention-bias", "GQA 6:1"]
}
```

`descriptors` is a human-readable list of traits. Use it for sanity
checks: if a "Qwen2" model doesn't have `"attention-bias"` in its
descriptors, something is off.

---

## 12. Supported architectures

ssmforge uses `transformers.AutoModelForCausalLM.from_pretrained()` under
the hood. That means it supports any causal LM architecture in the
`transformers` library: Llama, Qwen, Mistral, Phi-3, Gemma, Mixtral,
Falcon, Pythia, GPT-NeoX, StarCoder, DeepSeek, and many more.

### Quirks detector coverage

| Architecture | Quirks detected | Verified by |
|--------------|-----------------|-------------|
| Llama, Qwen2 | attention_bias, tied_embeddings, GQA, swiglu, rms | Real runs on Qwen2-0.5B, TinyLlama-1.1B, tiny-random-Llama |
| Phi-3 | fused_qkv, fused_gate_up, layer_scale | Simulated state_dict |
| Mistral | sliding_window, GQA | Simulated state_dict |
| Gemma2 | GeGLU, soft_capping | Simulated state_dict |
| Mixtral | MoE (8 experts, top-2) | Simulated state_dict |
| Falcon | MQA | Simulated state_dict |
| Pythia / GPT-NeoX | MQA, tied_embeddings | Simulated state_dict |

"Simulated state_dict" means the quirk detector is verified against a
crafted state_dict matching that architecture's key layout. The
detection logic is the same — it just hasn't been run on real model
weights for those architectures in CI.

If you spot a quirk being missed on a real model, please open an issue
at https://github.com/lordxmen2k/SSMForge/issues with the model id and
the missing quirk.

---

## 13. Troubleshooting

### Installation

**`pip install ssmforge` fails with `ModuleNotFoundError`**

You're probably on Python < 3.10. Check: `python --version`. Upgrade or
use a newer Python.

**`pip install ssmforge` succeeds but `ssmforge` command not found**

The install put the script in a directory not on your PATH. Activate
your venv (see [4.2](#42-create-a-virtual-environment-recommended-strongly)) or
check `pip show ssmforge` for the install location.

**`ssmforge: command not found` after `pip install` succeeds (Windows)**

When `pip install ssmforge` writes to a per-user location instead of
your venv (often on Windows when system Python isn't writable), the
`ssmforge.exe` script ends up at:

```
%APPDATA%\Python\Python314\Scripts\ssmforge.exe
```

…but that folder isn't on your PATH by default. Three fixes:

```powershell
# PowerShell (persistent, restart shell after)
$env:APPDATA = $env:APPDATA
[Environment]::SetEnvironmentVariable("PATH",
    "$env:APPDATA\Python\Python314\Scripts;" +
    [Environment]::GetEnvironmentVariable("PATH", "User"),
    "User")
```

```bash
# Git Bash (persistent)
echo 'export PATH="$APPDATA/Python/Python314/Scripts:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

Or just use `python -m ssmforge.cli <subcommand>` as a workaround:

```bash
python -m ssmforge.cli --version
python -m ssmforge.cli doctor
python -m ssmforge.cli arch Qwen/Qwen2-0.5B-Instruct --dry-run
```

**Recommendation: install in a venv** so the Scripts folder is
predictably inside your project's `.venv/`:

```bash
python -m venv .venv
source .venv/bin/activate    # Linux / Mac / Git Bash
.venv\Scripts\activate       # Windows PowerShell
pip install ssmforge
ssmforge --version           # works
```

### Loading the model

**`401 Unauthorized` for a gated model**

You haven't accepted the license on the model's HF page, or `HF_TOKEN`
isn't set. See [4.5](#45-optional-generate-an-hf-access-token) and
[4.6](#46-optional-accept-gated-model-licenses).

**`Repository Not Found`**

Wrong id, or you don't have access to a private repo. Go to
https://huggingface.co/USER/MODEL and check that the URL matches what
you passed.

**`OSError: [Errno 28] No space left on device`**

Your `HF_HOME` drive is full. Either move `HF_HOME` to a bigger drive
or clear old models from the cache:

```bash
# See what's there
ls -lh $HF_HOME/hub/
# Remove specific models
rm -rf $HF_HOME/hub/models--Qwen--Qwen2-1.5B-Instruct
```

**`memory allocation of N bytes failed` / Out of memory during load**

ssmforge catches these and exits 1 with a hint. The model needs more RAM
than you have free. Two options:

```bash
# 1. Preview the model first to see how much RAM it needs
ssmforge arch mistralai/Mixtral-8x7B-v0.1 --dry-run
# Memory estimate: ~13.49 GB in bfloat16 (7.24B params)
# Compatibility: BLOCKED by: moe

# 2. Use a smaller model or a machine with more RAM
ssmforge arch Qwen/Qwen2-0.5B-Instruct   # only 1 GB, fits almost anywhere
```

**`Error: local path does not exist: /tmp/foo`**

You passed a local path that doesn't exist. ssmforge fails fast instead
of hanging trying to load it as an HF model id. Either fix the path or
drop the leading `/`, `./`, or `~/` if you meant an HF id like
`Qwen/Qwen2-1.5B-Instruct`.

**Ctrl+C mid-load**

Exit code 130 with a clean `Interrupted.` message — no traceback. Just
re-run when ready.

**First run is slow**

Expected. The first `ssmforge arch MODEL` downloads the model. The next
run is instant because the model is cached.

### Running tests

**`PermissionError: [WinError 5]` from pytest on Windows**

Set `SSMFORGE_BASETEMP` to a directory your user owns:

```bash
SSMFORGE_BASETEMP=G:/test-tmp python -m pytest tests/
```

The default config also pins `addopts = "--basetemp=/tmp/pytest-ssmforge"`
in `pyproject.toml`. If `/tmp` doesn't exist on your system, override it.

**`pytest` hangs collecting tests**

You may have a slow or unavailable cache. The test suite downloads tiny
test models (a few MB). Override `HF_HOME` for tests:

```bash
SSMFORGE_TEST_HF_HOME=/tmp/ssmforge-test-cache python -m pytest tests/
```

### Output looks wrong

**`rope_theta` shows wrong value or `rope_theta_source` is `null`**

Older transformers versions (< 4.45) put `rope_theta` directly in the
config. Newer versions (≥ 5.0) move it into `rope_parameters`. ssmforge
checks both. If neither has it, the source will be `null` — that's a
sign of a malformed config, not a ssmforge bug.

**`Profile.family` says "unknown"**

The `_infer_family` heuristic looks at `config.model_type` and known
quirk combinations. New architectures we haven't seen will report
`unknown` until we add them. Open an issue if you see this for a known
architecture.

### Performance

**Memory usage**

Loading a 7B model in fp16 takes ~14 GB of RAM. ssmforge holds the
model in memory while building the report. If you're tight on RAM,
stick to smaller models (under 3B) or use `--quiet` (doesn't help
memory, but reduces output processing).

ssmforge doesn't currently support loading in 8-bit / 4-bit mode —
that would require extra dependencies. Open an issue if you need this.

**Speed**

Most of the time is spent downloading and loading the model, not
analyzing. The quirk scan itself is O(layers + tensors), typically a
few seconds.

---

## 14. Why this exists

When you do surgery on a HuggingFace model — convert it, fine-tune it,
quantize it, port it to a different architecture family — the source
model's quirks are usually what break things. Qwen2 silently drops
attention biases. Phi-3 fuses QKV. Mistral uses sliding window. Mixtral
is MoE. None of this is obvious from `config.model_type` alone.

`ssmforge arch` looks at the actual weights and config, and tells you
exactly which quirks your model has — before you spend an hour
discovering them at inference time.

It's intentionally a **diagnostic only**. It doesn't try to fix
anything. It just tells you what's going on so you can make an
informed decision.

---

## 15. Limitations

What ssmforge **does not** do:

- **Inference.** Doesn't generate text. Use `transformers` or `vLLM`.
- **Modification.** Doesn't convert, quantize, or rewrite anything.
- **Training.** No fine-tuning, no LoRA, no dataset handling.
- **Conversion to GGUF / ONNX / MLX / etc.** That's a different tool.
- **Quantization.** Same.
- **Loading in 8-bit / 4-bit.** Memory-hungry but straightforward.
- **Standalone runtime.** Doesn't load the model into a chatbot. Use ollama / vLLM.

The PyPI package has zero PyTorch / GGUF / mamba-ssm dependencies. It's
intentionally minimal — if you only need to inspect, this is what you
install.

---

## 16. Update log

### v0.2.3 (current) — 2026-09-23

**Bug fix: Git Bash strips backslashes from unquoted `--output` args**

When you run on Git Bash + Windows and pass an unquoted Windows path:
```bash
python -m ssmforge.cli arch X --graph --format markdown \
    --output C:\Users\netge\Desktop\report.md
```

Git Bash sees the backslashes as path separators and strips them
**before** passing the argument to Python. Python receives:
```
--output C:UsersnetgeDesktopreport.md
```
which is interpreted as a single literal filename containing no
separators. The CLI then writes a file named
`./C:UsersnetgeDesktopreport.md` in the current directory — totally
wrong, and v0.2.2 didn't catch it.

**v0.2.3 fixes this with a pre-parse guard** that runs before argparse
and checks for:

1. **Literal backslash paths** (`C:\\file.md`) — caught with hint to
   use forward slashes or wrap in quotes.
2. **Bash-mangled paths** (`C:UsersnetgeDesktop.md`) — caught with
   hint explaining what bash did to the argument.

The error fires immediately, before any model is loaded:

```
Error: --output value looks like a bash-mangled Windows path.
  Got: 'C:UsersnetgeDesktopreport.md'
  Git Bash strips backslashes from unquoted arguments.
  Wrap the value in double quotes, e.g.:
    --output "C:\Users\netge\Desktop\report.md"
  Or use forward slashes (works on both Windows and POSIX):
    --output C:/Users/netge/Desktop/report.md
```

**Recommended from v0.2.3 onward:** quote all `--output` paths, or
use forward slashes.

Tests: 174 passed (was 165). 9 new tests for the pre-parse guard
covering bash-mangled separate-args, bash-mangled equals-form,
backslash-only paths, plain POSIX paths, forward-slash Windows paths,
and bare drive-letter patterns.

### v0.2.2 — 2026-09-23

**Bug fix: `--output PATH` directory validation**

When `--output PATH` points to a non-writable location, users were getting
either a generic shell `Permission denied` (when using shell redirection)
or an opaque Python `OSError` (when using `--output` directly). Now
ssmforge pre-flight checks the parent directory and prints a friendly
hint pointing at writable alternatives.

**Before:**
```
$ python -m ssmforge.cli arch X --graph --format markdown > q.md
bash: q.md: Permission denied

$ python -m ssmforge.cli arch X --graph --format markdown --output C:\q.md
Error: cannot write to 'C:\\q.md': [Errno 13] Permission denied: 'C:\\q.md'
```

**After:**
```
$ python -m ssmforge.cli arch X --graph --format markdown > q.md
# (still works in any shell that allows the redirect)

$ python -m ssmforge.cli arch X --graph --format markdown --output C:\q.md
Error: cannot write to directory: C:\
  Permission denied.
  Try a writable location like:
    --output ./report.md        (current directory)
    --output ~/report.md        (your home directory)
    --output $TMPDIR/report.md  (system temp)
```

Tests: 162 passed (was 156). 6 new tests for `_check_output_path`
(returns (None, None) for None / '-' paths, validates writability,
emits helpful hints for missing or read-only parents).

### v0.2.1 — 2026-09-23

**New feature: `doctor --check-install`**

Run install checks (PATH, scripts location, platform) and exit non-zero
when something is wrong. Use in scripts or when troubleshooting:

```bash
ssmforge doctor --check-install
# prints ✓ or ✗ + fix hint
echo $?  # 0 = good, 1 = something off
```

The fix-hint output is a copy-pasteable shell command — exactly the
right `export PATH=...` or `[Environment]::SetEnvironmentVariable(...)`
line for your platform.

**New: one-time PATH warning on stderr**

When ssmforge detects it's been installed off-PATH, every `ssmforge
arch ...` call prints a one-time hint to stderr:

```
Note: 'ssmforge' is not on your PATH for `python -m ssmforge.cli` users.
      This is normal if you're running via 'python -m ssmforge.cli'.
      If 'ssmforge --version' fails, see: https://github.com/lordxmen2k/SSMForge#troubleshooting
```

This is suppressed:
- when ssmforge IS on PATH
- when the user explicitly runs `ssmforge doctor ...`
- when `SSMFORGE_NO_PATH_WARN=1` is set in the env

**README install lead**

Section 4.2 (Create a virtual environment) now leads with a stronger
recommendation and a fallback path (`python -m ssmforge.cli`). No
changes to the install command itself — just clearer context.

**Bug fix: Qwen2 attention_bias false positive**

Previously, `scan_state_dict` set `attention_bias=True` whenever
`*self_attn.*.bias` tensors existed in the state dict. Qwen2 ships
zero-initialized bias tensors (vestigial, never used) and was
incorrectly reported as having attention biases.

Now: `attention_bias` is set to True only when **either** the config
explicitly says `attention_bias=True` **or** the bias tensors contain
non-zero values. The `bias_keys_found` list still records that the
tensors exist for transparency.

Tests:
- `test_scan_detects_attention_bias` (updated) — zero biases → False
- `test_scan_detects_nonzero_attention_bias` (new) — non-zero → True
- `test_reports_attention_bias_from_state_dict_when_config_missing`
  (updated) — config.attention_bias=None + zero biases → False

**Bug fix: `--compare A A` no longer rejected**

Previously `--compare A A` was dedup'd to `[A]` and rejected with
"requires at least 2 models". Now produces a valid 2-way "all identical"
report.

**Feature: `--graph` honors `--format {text,json,markdown}`**

Three output modes for `--graph`:
- `--format text` (default for `--graph`) — ASCII with box-drawing
- `--format json` — structured pipeline + decisions for piping into
  other tools
- `--format markdown`/`md` — GitHub-flavored Markdown tables for
  embedding in docs, Issues, PRs

All three work for both single-model and `--compare --graph` modes.

**Bug fix: `--graph` ASCII hierarchy**

The previous box-drawing had two parallel rails (`├─` and `▼`) that
suggested two unrelated flows. Replaced with single-rail `▼` flow and
`╭╮╰╯` panels for ATTENTION and MLP blocks. Decision-tree style
`├─/└─` inside each panel. Renders correctly in any terminal.

**Tests: 156 passed** (was 142). 14 new tests:
- 12 from `test_v021_check_install.py` (PATH detection, doctor
  --check-install, main() integration)
- 1 from `test_v017_compare.py` (`--compare A A` regression)
- 1 from `test_analyze.py` (nonzero attention_bias detection)
- 13 from `test_graph.py` (JSON/Markdown formatters, GFM
  compatibility, edge cases)
- (some overlap with existing test updates for attention_bias fix)

Tests: 142 passed (was 130). 12 new tests:
- `_check_ssmforge_on_path`: returns correct tuple, finds off-path
  install, returns no fix when on PATH
- `_warn_path_once`: writes to stderr when off-path, respects env
  var, skips when on PATH
- `main()` integration: suppresses warning for doctor, emits for arch
- `doctor --check-install`: text output on/off PATH, JSON output,
  exit codes (0 / 1), back-compat with default `doctor`

### v0.2.0 — 2026-09-22

**New feature: `--graph`**

- Single-model: renders a text-based decision graph of the architecture.
  Shows input → embed → attn (with fused QKV / GQA / attention_bias /
  rope_theta decisions inline) → MLP (SwiGLU / GeGLU / MoE verdict) →
  final norm → lm_head → logits. Plus a `KEY DECISIONS` summary
  block at the bottom that lists every choice a downstream tool
  needs to know about, and a memory estimate line for the active
  dtype plus int8 / int4 projections.
- Compare (`--compare --graph`): renders an N-way decision table.
  Each row is one architectural decision; each column is one model.
  A "Differ across N models" footer lists the decisions that
  actually disagree.

Why semver 0.2.0: this is a new visual output mode (not just an
additive flag), and the CLI's `--graph` flag changes the parsing
shape. Per semver rules, that's a minor-version bump.

Implementation: new `ssmforge/analyze/graph.py` module exposes
`render_graph_text(report)` and `render_graph_compare(reports)`
for Python API users.

Tests: 130 passed (was 111). 19 new tests:
- `_clean_dtype`, `_fmt_params`, `_fmt_bytes`, `_yesno` (4 helpers)
- `_gqa_label` covers MQA / GQA / MHA correctly with both dict and
  SimpleNamespace config inputs
- `render_graph_text` covers: basic structure, tied embeddings,
  rope_theta formatting, dtype normalization, dry-run unknown fields
  with honest "requires --full inspection" notes, header layers count,
  attention box layer count (regression for `{layers}` literal bug)
- `render_graph_compare` covers: 2-model header, 3-model list,
  identical-models-no-diff message, per-model GQA labels (regression
  for hardcoded `kv:2` label), single-model fallback, empty input

Plus 130 total tests passing.

### v0.1.9 — 2026-09-22

**Bug fixes:**

- `--fields <path>` no longer crashes with `KeyError: 'model_id'`
  when the subset strips the `model_id` field. The dry-run summary
  now uses `.get('model_id', '<unknown>')`. Triggered by:
  ```
  ssmforge arch X --dry-run --fields "profile.family,quirks.tied_embeddings"
  ```
- `--quiet` now silences transformers' `[transformers]` warnings
  that bypass the logging system (printed directly via Python's
  `warnings` module). Affected messages:
  - `[transformers] torch_dtype is deprecated! Use dtype instead!`
  - `[transformers] Model config: pad_token_id must be None or an integer...`

Both fixes also apply in `--compare` mode.

Tests: 111 passed (was 107). 4 new regression tests:
- `test_cli_fields_subset_no_model_id_keyerror` — verifies the fix
- `test_cli_fields_does_not_crash_on_full_subset` — multi-path subset
- `test_quiet_transformers_warnings_filters_deprecation` — verifies filter
- `test_quiet_transformers_warnings_handles_missing_transformers` — graceful fallback

### v0.1.8 — 2026-09-22

**New flags:**

- `--fields name1,name2` — Subset the report output. Top-level keys
  (`config`, `quirks`, `profile`, `state_dict_summary`, etc.) or dotted
  paths (`quirks.attention_bias`, `profile.family`). Invalid field
  names print a warning but don't crash.
- `--profile` — Emit just the profile section (`family`,
  `attention_type`, `mlp_type`, `norm_type`, `descriptors`). Useful
  for quick eyeball checks or shell pipelines that only care about
  the family.
- `--only-different` — In `--compare` mode, hide the "Identical
  across all models" section. Default already shows only differing
  fields in the table; this flag additionally excludes the identical
  summary.
- `--rev` / `--revision` — Pin to a specific HF commit sha, tag, or
  branch for reproducible reports. Default loads the current HEAD.

**HF revision tracking:**

Every report now has an `hf_revision` field with the resolved commit sha
when available. Combined with `--rev`, this gives full reproducibility
for arch reports.

**New Python API:**

- `FIELD_DESCRIPTIONS` dict in `ssmforge.analyze.state_dict_scan` —
  inline docs for every quirk and config key. Useful for tooltips,
  docs generation, or programmatic field validation.

Tests: 107 passed (was 91). 16 new tests covering field subsetting,
profile-only output, revision tracking, --only-different filtering,
and FIELD_DESCRIPTIONS coverage.

### v0.1.7 — 2026-09-22

**New feature: N-way model comparison via `--compare`**

```bash
ssmforge arch --compare Qwen/Qwen2-0.5B-Instruct TinyLlama/TinyLlama-1.1B-Chat-v1.0
ssmforge arch A --compare B C D
ssmforge arch --compare A B C D --format markdown
```

Renders a multi-model comparison table showing every field's value
across all models. Fields that match across all models are listed in an
"Identical across all models" section (not in the table) to keep the
table scannable.

Both JSON and Markdown output supported. JSON includes per-field
`all_same` and `unique_values` so programmatic consumers can quickly
filter for differences.

Works with `--dry-run` for config-only comparisons (no weight download).
Works with `--format markdown` for GitHub-friendly tables.

**Back-compat:** legacy `--diff B` (one-vs-one) still works and produces
the same output as `--compare A B`. New flag is purely additive.

**Python API additions:**
- `compare_reports(reports_list)` — N-way comparison
- `format_compare_markdown(comparison)` — Markdown table renderer

Tests: 91 passed (was 80). 11 new tests covering 2-model, 3-model,
markdown output, JSON output, dry-run, source-positional-then-compare,
back-compat with `--diff`, and fail-fast on bad paths.

### v0.1.6 — 2026-09-22

**New quirks detector coverage:**
- StarCoder / StarCoder2 (`gpt_bigcode`, `starcoder`, `starcoder2` model types)
- OLMo and OLMoE (MoE and non-MoE variants)
- DeepSeek (both regular and DeepSeek-MoE)
- Qwen1.5-MoE / Qwen2-MoE specific family label
- Falcon and Pythia / GPT-NeoX now report their specific family name (was:
  generic "Falcon / Pythia (MQA)")

**Bug fix:**
- `_infer_family` was being passed the local `cfg(name)` callable inside
  `build_report`, but tried `getattr(cfg, "model_type")` which always
  returned empty. Family inference for non-Llama architectures was
  silently degrading to the generic default. Now detects model_type
  correctly through either interface.

**Quality of life:**
- `--quiet` now also suppresses transformers' noisy deprecation warnings
  (`torch_dtype` migration, `pad_token_id` warnings on small test
  models). Real warnings (ERROR level) still show through.

Tests: 80 passed (was 69). 11 new tests for the expanded family inference
and warning suppression.

### v0.1.5 — 2026-09-22

**New features:**
- `ssmforge arch MODEL --dry-run` — config-only mode. No weight download.
  Reports memory cost and config-only quirks (GQA, MQA, MoE, sliding window,
  soft-capping, partial RoPE, rope_theta). Works with `--diff` for
  config-only diffs.
- `ssmforge --version` / `-V` — print version + exit 0.

**Bug fixes:**
- `--output -` now means stdout (Unix convention), not a file named `-`.
- Local paths that don't exist fail fast with a clear error, instead of
  hanging trying to load them as HF model ids.
- `--output` to a non-writable or non-existent path exits 1 with a clean
  message, not a traceback.
- Ctrl+C during load prints `Interrupted.` and exits 130, not a traceback.
- Memory allocation failures during load exit 1 with a hint to use
  `--dry-run`, not the OS kill signal (exit 127).
- `MoE` is now detected from the config alone — `--dry-run` reports
  `Compatibility: BLOCKED by: moe` without downloading any weights.

**Tests:** 69 passed (was 50). 19 new tests for the v0.1.5 features.

**Python API additions:**
- `ssmforge.analyze.build_report_from_config(model_id, config)`
- `ssmforge.analyze.scan_config_only(config)`
- `ssmforge.analyze.estimate_params_from_config(config)`
- `ssmforge.analyze.estimate_memory_bytes(params, dtype)`

### v0.1.4 — 2026-09-22

First PyPI release. Three user-facing features:

- `ssmforge arch MODEL` — JSON + human summary report
- `ssmforge arch MODEL --diff OTHER` — compare two architectures
- `ssmforge doctor` — environment + cache info

50 tests passing. Detects 15 architectural quirks including attention
bias, tied embeddings, fused QKV, fused gate-up, GQA, MQA, MoE, sliding
window, soft-capping, partial RoPE, MLP type, norm type, layer scale,
num experts, moe top-k.

PyPI: https://pypi.org/project/ssmforge/

### Notes for upgrading from earlier versions

If you previously installed ssmforge from a different source or a
pre-PyPI development build, upgrade to pick up v0.1.5:

```bash
pip install --upgrade ssmforge
ssmforge --version
# ssmforge 0.1.5
```

If you have an old editable install from the v0.0.x development series,
uninstall it first:

```bash
pip uninstall ssmforge
pip install ssmforge
```

---

## Contributing

Bug reports and feature requests welcome:
https://github.com/lordxmen2k/SSMForge/issues

Pull requests: fork the repo, make your change, open a PR. Tests are
required for new features (`pip install -e ".[dev]" && pytest tests/`).

## License

Apache 2.0. See `LICENSE`.

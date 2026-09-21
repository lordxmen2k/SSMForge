# SSMForge — Full Installation Guide

This document walks you through **everything you need** to install SSMForge, run
its CLI, and publish it to PyPI from your own machine. No GitHub Actions, no CI —
just you, a Python venv, and `twine`.

> **TL;DR for the impatient** — see the [Quick install](#quick-install) section.
> **Stuck on something?** — jump to [Troubleshooting](#troubleshooting).

---

## System requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python    | 3.10    | 3.11        |
| RAM       | 8 GB    | 16 GB       |
| Disk      | 5 GB    | 20 GB (for HF model cache + GGUF artifacts) |
| OS        | Linux, macOS, Windows | Linux + NVIDIA GPU for distillation |
| Network   | HTTPS access to `pypi.org` and `huggingface.co` | same |

---

## Quick install

> **Important:** the `[mamba]` extra only installs `mamba-ssm` on Python
> 3.9–3.12 (where prebuilt wheels exist). On 3.13+ SSMForge uses a
> pure-PyTorch fallback.

For users who just want to **run** SSMForge (not develop it):

```bash
# 1. Create a virtual environment (strongly recommended)
python3 -m venv .venv
source .venv/bin/activate           # Linux/macOS
# .venv\Scripts\activate           # Windows

# 2. Upgrade pip and configure the official PyPI index
#    (Some environments default to a broken mirror that times out.)
python -m pip install --upgrade pip
python -m pip config set global.index-url https://pypi.org/simple/

# 3. Install SSMForge
#    Python 3.9–3.12: gets real CUDA Mamba
pip install "ssmforge[export,mamba]"
#    Python 3.13+: gets pure-PyTorch fallback (works fine, just slower)
pip install "ssmforge[export]"
```

After install, verify everything works:

```bash
ssmforge --help
ssmforge list-recipes
python -c "import ssmforge; print(ssmforge.__version__)"
```

You should see `hybrid-25`, `hybrid-50`, and `pure-mamba` in the recipes list.

---

## Detailed install — every step explained

### Step 1 — Python

Verify your Python version:

```bash
python3 --version
```

Must be **3.10 or higher**. If your system Python is older, install a newer one:

- **macOS:** `brew install python@3.11`
- **Ubuntu/Debian:** `sudo apt install python3.11 python3.11-venv`
- **Windows:** download from [python.org](https://www.python.org/downloads/) (check "Add Python to PATH")

### Step 2 — Virtual environment

A venv keeps SSMForge's dependencies isolated from your system Python. Skip this
only if you know what you're doing (and use `--break-system-packages`).

```bash
python3 -m venv .venv
source .venv/bin/activate            # Linux/macOS
# .venv\Scripts\activate             # Windows PowerShell
```

Your prompt should now show `(.venv)`.

### Step 3 — pip and PyPI index

Some sandboxes and CI environments default to a regional mirror that can be
flaky or outright broken. Force pip to use the official index:

```bash
python -m pip install --upgrade pip
python -m pip config set global.index-url https://pypi.org/simple/
```

Verify pip can reach PyPI:

```bash
pip install --dry-run --quiet "setuptools>=68" && echo OK
```

### Step 4 — Install SSMForge

SSMForge has three optional dependency groups:

| Extra    | Installs                                  | When you need it |
|----------|-------------------------------------------|------------------|
| `mamba`  | `mamba-ssm`, `causal-conv1d`              | Real CUDA Mamba2 layers (vs the pure-PyTorch fallback) |
| `export` | `gguf` (gguf-py)                          | Writing GGUF files for llama.cpp |
| `dev`    | `pytest`, `pytest-cov`, `hypothesis`, `ruff` | Running the test suite |

Common combinations:

```bash
# Production use — GGUF export (recommended)
pip install "ssmforge[export]"

# Production + CUDA Mamba (Python 3.9–3.12 only)
pip install "ssmforge[export,mamba]"

# Development — includes test deps
pip install "ssmforge[export,dev]"

# Minimal — no GGUF export (Stage 5 will error)
pip install ssmforge
```

**Note on the `[mamba]` extra:** `mamba-ssm` ships prebuilt wheels only for
Python 3.9–3.12. The `[mamba]` extra in `pyproject.toml` is gated with
`python_version < "3.13"` markers so it installs cleanly on Python 3.13+ by
skipping the package gracefully instead of failing. On 3.13+, SSMForge uses a
pure-PyTorch SSM fallback (slower, no CUDA acceleration) without you needing
to do anything special.

### Optional: native Mamba2 CUDA kernels

**SSMForge does NOT include `[mamba]` in its install extras anymore** (as of
v0.1.1). The `mamba-ssm` package only ships prebuilt wheels for Python 3.9–
3.12, so forcing it into the install broke Python 3.13+ users. SSMForge
works without it using a pure-PyTorch SSM fallback.

If you're on Python 3.9, 3.10, 3.11, or 3.12 and want the real CUDA kernels:

```bash
pip install mamba-ssm causal-conv1d --no-build-isolation
```

If that command fails on your system, skip it — the fallback works.

### Step 5 — Install `llama-quantize` (for non-F16 quantization)

`llama-quantize` is a C++ binary that ships with **llama.cpp**. SSMForge shells
out to it for Q4_K_M / Q5_K_M / Q8_0 / etc. If you only ever use `--quantize F16`,
skip this section.

**Option A — install via llama-cpp-python (recommended):**

```bash
pip install llama-cpp-python
# The `llama-quantize` binary is bundled in this wheel
which llama-quantize   # should print a path
```

**Option B — build our vendored llama.cpp fork (with ssmforge arch support):**

The vendored llama.cpp fork at `vendor/llama.cpp/` (lordxmen2k/ssmforge-llama.cpp)
is registered as a git submodule. It has the `LLM_ARCH_SSMFORGE` arch metadata so
its `llama-quantize` will accept ssmforge-generated GGUFs without the "unknown
architecture" error:

```bash
# Initialize the submodule if you haven't already
git submodule update --init --recursive

# Build just the tools we need (CPU-only by default; add --cuda for NVIDIA)
bash scripts/build_vendor.sh           # CPU-only
bash scripts/build_vendor.sh --cuda    # NVIDIA GPU (CUDA)
bash scripts/build_vendor.sh --metal   # Apple Silicon

# The binary lives at vendor/llama.cpp/build/bin/llama-quantize
# SSMForge finds it automatically. No env var needed.
ls vendor/llama.cpp/build/bin/llama-quantize
```

**Option C — build upstream llama.cpp from source (no ssmforge arch):**

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
make llama-quantize -j$(nproc)
# Binary will be at ./build/bin/llama-quantize
export LLAMA_QUANTIZE_BIN="$(pwd)/build/bin/llama-quantize"
echo 'export LLAMA_QUANTIZE_BIN="'"$(pwd)"'/build/bin/llama-quantize"' >> ~/.bashrc
```

SSMForge finds the binary in this order:

1. `$LLAMA_QUANTIZE_BIN` environment variable
2. `vendor/llama.cpp/build/bin/llama-quantize` (vendored fork's build)
3. `llama-quantize` on `$PATH`

If none is found, you'll see an actionable `LlamaQuantizeNotFoundError` when you
try to quantize to anything other than F16.

### Step 6 — (Optional) Mamba CUDA acceleration

`mamba-ssm` provides the real Mamba2 CUDA kernels. Without it, SSMForge falls
back to a slower pure-PyTorch implementation (still functional, just slow).

The pip wheels for `mamba-ssm` are CUDA-specific. Pick the right one for your GPU:

| GPU / CUDA | Install command |
|------------|-----------------|
| NVIDIA Ampere+ (RTX 30/40/A100) on CUDA 12.x | `pip install "ssmforge[mamba]"` (auto-picks) |
| NVIDIA Hopper (H100) on CUDA 12.x | `pip install "ssmforge[mamba]"` |
| Older NVIDIA (pre-Ampere) on CUDA 11.x | `pip install torch==2.3.*+cu118 mamba-ssm==2.0.* causal-conv1d==1.4.*` |
| CPU only (no GPU) | `pip install ssmforge` (skip `[mamba]`) |
| macOS Apple Silicon | `pip install ssmforge` (mamba-ssm has no MPS wheel yet) |

If `pip install "ssmforge[mamba]"` fails because no compatible wheel exists for
your CUDA/Python combination, install without `[mamba]` and the fallback will
be used automatically.

### Step 7 — Verify the install

```bash
# CLI present
ssmforge --help

# Recipes registered
ssmforge list-recipes
# Expected output:
#   Registered recipes:
#     - hybrid-25
#     - hybrid-50
#     - pure-mamba

# Importable
python -c "import ssmforge; print(ssmforge.__version__)"

# Real end-to-end dry run (downloads ~50MB HF model)
ssmforge convert hf-internal-testing/tiny-random-LlamaForCausalLM --dry-run
```

If all of the above work, you're ready to convert real models.

---

## Running SSMForge

### Dry run (Stages 1-3 only — no distillation, no export)

```bash
ssmforge convert meta-llama/Llama-3.2-1B \
    --recipe hybrid-25 \
    --dry-run
```

This loads the model, runs the recipe plan, performs state-dict surgery, and
prints the layer mapping + stats. No GPU required. Takes ~10 seconds.

### Full conversion (Stages 1-5)

```bash
ssmforge convert meta-llama/Llama-3.1-8B-Instruct \
    --recipe hybrid-25 \
    --quantize Q4_K_M \
    --output ./out
```

Output:

```
./out/meta-llama_Llama-3.1-8B-Instruct.HYBRID-25.Q4_K_M.gguf       (the model)
./out/meta-llama_Llama-3.1-8B-Instruct.HYBRID-25.Q4_K_M.manifest.json (provenance)
```

### Python API

```python
from ssmforge import convert
from pathlib import Path

result = convert(
    source="meta-llama/Llama-3.2-1B",
    recipe="hybrid-50",
    quantize="Q4_K_M",
    output_dir=Path("./out"),
    calibration_data=None,        # uses built-in default
    verify=False,                  # skip Stage 6 (slow)
    dry_run=False,                 # full run
    experimental=False,            # required=True for pure-mamba
)

print(f"GGUF:      {result.gguf_path}")
print(f"Manifest:  {result.manifest_path}")
print(f"Stats:     {result.stats}")
```

---

## Publishing to PyPI from your machine

This section is for you (the developer) — not for end users. Run all commands
in this section from the project root.

### Step 1 — PyPI account

If you don't have one, register at https://pypi.org/account/register/.

### Step 2 — API token

Generate an API token at https://pypi.org/manage/account/token/.

- Scope: **Entire account** (first time) or **Project: ssmforge** (after first upload)
- Copy the token immediately — PyPI only shows it once.

### Step 3 — Store the token securely

**Don't paste the token in chat or commit it.** Use one of these:

```bash
# Option A: keyring (recommended)
pip install keyring
keyring set https://upload.pypi.org/legacy/ your-pypi-username
# Paste the token (including the pypi- prefix) when prompted

# Option B: ~/.pypirc (file-based, chmod 600)
cat > ~/.pypirc <<'EOF'
[distutils]
index-servers = pypi

[pypi]
username = __token__
password = pypi-YOUR_TOKEN_HERE
EOF
chmod 600 ~/.pypirc
```

### Step 4 — Bump the version

Edit `pyproject.toml`:

```toml
[project]
name = "ssmforge"
version = "0.1.0"     # ← bump this
```

And `src/ssmforge/__init__.py`:

```python
__version__ = "0.1.0"  # ← keep in sync
```

### Step 5 — Build the distribution

```bash
# Install build tools (once)
pip install --upgrade build twine

# Clean any prior builds
rm -rf dist/ build/ *.egg-info src/*.egg-info

# Build sdist + wheel
python -m build
```

Expected output:

```
dist/
├── ssmforge-0.1.0-py3-none-any.whl
└── ssmforge-0.1.0.tar.gz
```

### Step 6 — Verify the build

```bash
twine check dist/*
```

Should report `PASSED` for both files. If it warns about long description,
that's fine — we have a README but no PyPI long_description configured.

### Step 7 — Upload to PyPI

```bash
twine upload dist/*
```

When prompted:

- **Username:** `__token__`
- **Password:** your PyPI token (the `pypi-...` string)

Or, if you used `~/.pypirc`, twine will pick up the credentials automatically.

### Step 8 — Verify on PyPI

Visit https://pypi.org/project/ssmforge/ — your version should be live.

Install from PyPI in a fresh venv to confirm:

```bash
python3 -m venv /tmp/verify
source /tmp/verify/bin/activate
pip install ssmforge
ssmforge --help
ssmforge list-recipes
```

### Optional — Test on TestPyPI first

If you want to test the upload without committing to real PyPI:

```bash
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ ssmforge
```

(Note: real PyPI and TestPyPI have separate user accounts.)

---

## Reproducing this build on a fresh machine

If you cloned the repo and want to build everything from source:

```bash
git clone https://github.com/lordxmen2k/SSMForge.git
cd SSMForge

# Create venv
python3 -m venv .venv
source .venv/bin/activate

# Install build deps + package in editable mode + dev tools
pip install --upgrade pip build twine
pip install --index-url https://pypi.org/simple/ -e ".[dev,mamba,export]"

# Run the test suite
pytest -v

# Build the distribution
python -m build

# (Optional) upload to PyPI — uses your ~/.pypirc credentials
twine upload dist/*
```

---

## Troubleshooting

### `pip install ssmforge` hangs or times out

The default pip index in your environment is broken. Fix:

```bash
pip config set global.index-url https://pypi.org/simple/
pip install ssmforge
```

### `hatchling` build error / `prepare_metadata_for_build_editable` AttributeError

This sandbox had this exact issue. SSMForge switched to `setuptools` (commit
`41d55e4`) to avoid it. If you see this with another project, the fix is to
change `build-backend` in `pyproject.toml` from `hatchling.build` to
`setuptools.build_meta` and add `[tool.setuptools.packages.find]`.

### `llama-quantize: command not found` when uploading Q4_K_M

SSMForge shells out to the binary. Install it:

```bash
pip install llama-cpp-python
# OR
git clone https://github.com/ggerganov/llama.cpp.git && cd llama.cpp && make llama-quantize
export LLAMA_QUANTIZE_BIN="/path/to/llama-quantize"
```

### `ImportError: mamba-ssm package required`

**Note:** SSMForge does NOT raise this error. It has a pure-PyTorch SSM fallback
that activates automatically when `mamba-ssm` isn't installed. You will see a
pure-PyTorch SSM layer (slower, no CUDA acceleration) — the package runs fine.

If you want the real CUDA Mamba2 (Python 3.9–3.12 only):

```bash
pip install "ssmforge[export,mamba]"
# Or manually:
pip install mamba-ssm causal-conv1d --no-build-isolation
```

If you only have Python 3.13+ available, the real CUDA Mamba is not currently
installable. Use the fallback (default behavior) or set up Python 3.11/3.12 in
a separate venv.

### `OSError: fake is not a local folder and is not a valid model identifier`

The `--source` argument is wrong. Use one of:

- An HF model id: `meta-llama/Llama-3.1-8B-Instruct`
- A local path: `/path/to/model/` (must contain `config.json`)
- A tiny test model: `hf-internal-testing/tiny-random-LlamaForCausalLM`

### Tests pass individually but fail when run together

Some tests rely on import order. Run them with the project's `pytest.ini_options`
settings:

```bash
pytest -v
```

If using `python -m pytest`, the `pyproject.toml` config is auto-discovered from
the project root. If you see "collected 0 items", you're probably in the wrong
directory.

### `git push` rejected with "Personal Access Token does not have workflow scope"

You're trying to push a GitHub Actions workflow file (`.github/workflows/*.yml`)
with a PAT that doesn't have `workflow` scope. Either:

1. Remove the workflow file before pushing
2. Generate a new PAT with the `workflow` scope at https://github.com/settings/tokens

SSMForge's repo was kept clean of workflow files specifically to avoid this.

### `from ssmforge import convert` returns the module, not the function

This happens when you have a `convert.py` file at the package root — Python
resolves `ssmforge.convert` to the submodule, shadowing the function in
`__init__.py`. SSMForge avoided this by naming the orchestrator `pipeline.py`.
If you're writing your own library, name orchestrator files something other
than the function name.

---

## Where to get help

- **Docs:** [docs/quickstart.md](quickstart.md), [docs/recipes.md](recipes.md),
  [docs/architecture.md](architecture.md), [docs/benchmarks.md](benchmarks.md)
- **Spec:** [docs/superpowers/specs/2026-09-21-ssmforge-design.md](superpowers/specs/2026-09-21-ssmforge-design.md)
- **Issues:** https://github.com/lordxmen2k/SSMForge/issues
- **PyPI:** https://pypi.org/project/ssmforge/

# Install

## Quick install

```bash
pip install ssmforge
```

That's it. Pulls in `transformers` and `huggingface_hub` for loading HF models.

## Requirements

- Python 3.10+
- ~50MB for the package itself
- A few hundred MB to a few GB of disk for cached models (the analyzer downloads the model on first use)

## Verify

```bash
ssmforge --help
ssmforge arch --help

# Run on a tiny model (downloads ~50MB on first run)
ssmforge arch hf-internal-testing/tiny-random-LlamaForCausalLM
```

You should see a JSON report on stdout and a human-readable summary on stderr.

## Controlling where models are cached

By default, HuggingFace caches downloaded models under
`~/.cache/huggingface/` (Linux/Mac/Git Bash) or
`%USERPROFILE%\.cache\huggingface\` (Windows). On a system with limited
space on the home drive (e.g. C: on Windows), point the cache elsewhere
with the `HF_HOME` environment variable:

```bash
# Git Bash on Windows
export HF_HOME=G:/models
mkdir -p G:/models
ssmforge arch Qwen/Qwen2-1.5B-Instruct

# PowerShell
$env:HF_HOME = "G:\models"
ssmforge arch Qwen/Qwen2-1.5B-Instruct

# Make it permanent:
#   PowerShell (admin):  setx HF_HOME "G:\models" /M
#   Git Bash / Linux:    echo 'export HF_HOME=G:/models' >> ~/.bashrc
#   Zsh:                  echo 'export HF_HOME=G:/models' >> ~/.zshrc
```

Models are cached under `$HF_HOME/hub/models--ORG--MODEL/snapshots/<sha>/`.

Other useful env vars:

| Variable | Purpose |
|----------|---------|
| `HF_HOME` | Root directory for all HF caches (`hub`, `datasets`, etc.) |
| `HF_HUB_CACHE` | Cache for downloaded model files only (overrides `HF_HOME/hub`) |
| `TRANSFORMERS_CACHE` | Same as `HF_HUB_CACHE` (older transformers versions) |
| `HF_TOKEN` | HF API token for gated models (e.g. Llama-3, Mistral) |

## Running tests

```bash
pip install -e ".[dev]"
python -m pytest tests/
```

Tests use a separate HF cache at `./tests/.cache/` so they don't pollute
your real cache (often on a different drive). Override with:

```bash
SSMFORGE_TEST_HF_HOME=G:/test-cache python -m pytest tests/
```

If you hit Windows `PermissionError` on `%TEMP%`, the test config pins
`basetemp` to `/tmp/pytest-ssmforge`. Override with:

```bash
SSMFORGE_BASETEMP=G:/test-tmp python -m pytest tests/
```

## Troubleshooting

### `ModuleNotFoundError: transformers`

```bash
pip install transformers huggingface_hub
```

### Slow first run

The first time you `ssmforge arch` a model, transformers downloads the
weights (default cache: `~/.cache/huggingface/`). Set `HF_HOME` to control
the cache location. Subsequent runs are instant.

### Exit code 2

The model has a quirk that blocks downstream use (currently: MoE). The
report will list which quirk(s) in the `compatibility.issues` section.

### Permission errors on Windows

If you see `PermissionError: [WinError 5]` from pytest, set
`SSMFORGE_BASETEMP` to a directory your user owns:

```bash
SSMFORGE_BASETEMP=G:/test-tmp python -m pytest tests/
```

## License

Apache 2.0

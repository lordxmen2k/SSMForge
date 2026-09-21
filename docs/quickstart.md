# Quickstart

## Install

```bash
pip install ssmforge[mamba,export]
```

Optional extras:
- `mamba` — installs `mamba-ssm` and `causal-conv1d` for native Mamba2 layers (falls back to a pure-PyTorch placeholder if not installed)
- `export` — installs `gguf` (gguf-py) for GGUF writing

## Convert a model

```python
from ssmforge import convert

result = convert(
    source="meta-llama/Llama-3.2-1B",
    recipe="hybrid-25",
    quantize="Q4_K_M",
    output_dir="./out",
)

print(f"GGUF: {result.gguf_path}")
print(f"Stats: {result.stats}")
```

## CLI

```bash
ssmforge convert meta-llama/Llama-3.2-1B \
    --recipe hybrid-25 \
    --quantize Q4_K_M \
    --output ./out

ssmforge list-recipes
```

For dry runs (plan + surgery only, no distillation or export):

```bash
ssmforge convert meta-llama/Llama-3.2-1B --dry-run
```

## Recipes

- `hybrid-25` (production, default): ~25% Mamba2 replacement, ~95-98% teacher quality
- `hybrid-50` (production): 1:1 alternation, ~90-95% teacher quality
- `pure-mamba` (experimental, requires `--experimental`): 100% Mamba2, ~60-80% quality

## What you get

After a successful conversion, a GGUF file lands at `./out/<model>.H<recipe>.Q4_K_M.gguf` (F16 if you specified F16). Load it with:

- `ollama run <path-to-gguf>`
- `./llama-cli -m <path-to-gguf>`
- LM Studio: open the file directly

## Status

🚧 **v0.1.0-dev (MVP phase)** — Stages 1-3 (load, plan, surgery) implemented. Distillation + export + verify coming in subsequent phases.

## Next steps

- See `docs/recipes.md` for the recipe catalog and quality expectations
- See `docs/architecture.md` for internal design
- See `docs/superpowers/specs/2026-09-21-ssmforge-design.md` for the full design spec

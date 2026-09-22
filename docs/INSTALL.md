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

## Troubleshooting

### `ModuleNotFoundError: transformers`

```bash
pip install transformers huggingface_hub
```

### Slow first run

The first time you `ssmforge arch` a model, transformers downloads the
weights to `~/.cache/huggingface/`. Subsequent runs are instant.

### Exit code 2

The model has a quirk that blocks downstream use (currently: MoE). The
report will list which quirk(s) in the `compatibility.issues` section.

## License

Apache 2.0

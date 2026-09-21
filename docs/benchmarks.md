# Benchmarks

Long-context memory and speed measurements for SSMForge-produced hybrid models.

## Methodology

Each model is loaded with `transformers.AutoModelForCausalLM` and benchmarked at
context lengths [4K, 32K, 128K] tokens. We measure:
- Peak GPU memory (via `torch.cuda.max_memory_allocated()`)
- Forward-pass latency in tokens/sec

Hardware target: NVIDIA A100 80GB (or consumer GPU like RTX 4090 for smaller models).

## How to run

```python
from ssmforge.benchmark import benchmark_long_context
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("path/to/model")
tokenizer = AutoTokenizer.from_pretrained("model-id")

results = benchmark_long_context(model, tokenizer, context_lengths=[4096, 32768, 131072])
for ctx, stats in results.items():
    print(f"{ctx:>8}: {stats.get('tokens_per_sec', 'N/A')} tok/s, {stats.get('peak_memory_mb', 0):.1f} MB")
```

## Estimated reference numbers

The numbers below are **estimated reference values** based on the Mamba/SSM literature
and Jamba benchmarks. Real numbers will be filled in once full distillation runs complete.

### Llama-3.1-8B-Instruct — Dense baseline (FP16)

| Context | Memory (GB) | Speed (tok/s) |
|---------|-------------|---------------|
| 4K      | 16.2        | 1240          |
| 32K     | 24.1        | 380           |
| 128K    | 48.3        | 95            |

### Llama-3.1-8B-Instruct — hybrid-25 (SSMForge, estimated)

| Context | Memory (GB) | Speed (tok/s) | Quality (vs teacher) |
|---------|-------------|---------------|----------------------|
| 4K      | 16.0        | 1320          | -2% MMLU             |
| 32K     | 20.4        | 480           | -3% MMLU             |
| 128K    | 32.1        | 165           | -4% MMLU             |

## Headline

At **1M context** (extrapolated):
- Dense baseline: OOM on 80GB GPU (needs ~270GB)
- hybrid-25: fits on 2× A100 (80GB), ~28 tok/s
- Quality cost: 6-8% MMLU vs teacher

At **128K context** (practical long-context ceiling):
- Dense: 48 GB, 95 tok/s
- hybrid-25: 32 GB, 165 tok/s
- **1.7× faster, 33% less memory**

## Status

These numbers are placeholders pending real benchmark runs with the released v0.3+.
Run `python examples/quickstart.py` followed by `benchmark_long_context(...)` to fill in real numbers for your hardware.

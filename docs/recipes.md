# Recipes

SSMForge ships with three recipes. All preserve the original tokenizer and chat template.

## hybrid-25 (production, default)

**What it does:** Replaces ~25% of middle attention layers with Mamba2. First 2 and last 2 attention layers preserved as anchors.

**Quality:** ~95-98% of teacher on standard benchmarks.

**Best for:** Production deployments where quality matters more than maximum long-context savings.

**Reference:** [MambaInLlama (NeurIPS 2024)](https://arxiv.org/abs/2408.15237)

## hybrid-50 (production)

**What it does:** Alternating attention and Mamba2 layers 1:1.

**Quality:** ~90-95% of teacher on standard benchmarks.

**Best for:** Aggressive long-context optimization, when you can afford the quality loss.

**Reference:** [Jamba (AI21)](https://arxiv.org/abs/2403.19887)

## pure-mamba (experimental)

**What it does:** 100% Mamba2 conversion via two-stage distillation (linear attention proxy → adapted Mamba).

**Quality:** ~60-80% of teacher on standard benchmarks. Significant degradation.

**Best for:** Research, memory-constrained edge deployment, ultra-long-context workloads where any attention cost is too high.

**Requires:** `--experimental` flag.

**Reference:** [Attention to Mamba (2025)](https://arxiv.org/abs/2604.14191)

## Adding custom recipes

```python
from ssmforge.recipes import Recipe, register_recipe
from ssmforge.config import LayerSpec, LayerType

@register_recipe
class MyCustomRecipe(Recipe):
    name = "my-custom"
    description = "..."
    requires_attention_fraction = 0.4

    def plan(self, model):
        # Decide layer mapping based on your heuristic
        ...

    def distillation_config(self):
        # Configure training stages
        ...
```

See `examples/custom_recipe.py` for a complete example.

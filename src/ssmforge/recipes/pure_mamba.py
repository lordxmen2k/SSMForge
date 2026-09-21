"""pure-mamba recipe: 100% Mamba2 with two-stage distillation.

EXPERIMENTAL. Quality loss is significant (20-40% on benchmarks).
Requires --experimental flag at CLI.

Two stages:
1. Linear attention proxy distillation (Transformer → linearized attention)
2. Adapted Mamba distillation (linearized attention → Mamba2)
"""

from __future__ import annotations

from typing import Any

from ssmforge.config import LayerSpec, LayerType, DistillationConfig, TrainingStage
from ssmforge.recipes.base import Recipe, register_recipe


@register_recipe
class PureMambaRecipe(Recipe):
    name = "pure-mamba"
    description = "EXPERIMENTAL: 100% Mamba2 conversion. Significant quality loss expected."
    requires_attention_fraction = 0.0
    paper_reference = "https://arxiv.org/abs/2604.14191 (Attention to Mamba, 2025)"

    def plan(self, model: Any) -> list[LayerSpec]:
        num_layers = 32
        if model is not None and hasattr(model, "config"):
            num_layers = model.config.num_hidden_layers
        return [LayerSpec(layer_type=LayerType.SSM, index=i, freeze_mlp=False) for i in range(num_layers)]

    def distillation_config(self) -> DistillationConfig:
        return DistillationConfig(
            stages=[
                TrainingStage(
                    name="linear_attention_proxy",
                    epochs=3,
                    learning_rate=1e-5,
                    batch_size=1,
                    gradient_accumulation_steps=8,
                    freeze_mlp=True,
                    stepwise=False,
                ),
                TrainingStage(
                    name="adapted_mamba_distill",
                    epochs=8,
                    learning_rate=1e-5,
                    batch_size=1,
                    gradient_accumulation_steps=8,
                    freeze_mlp=False,
                    stepwise=False,
                ),
            ],
            kl_weight=0.5,
            seqkd_weight=0.5,
            max_seq_length=4096,
            warmup_steps=500,
        )

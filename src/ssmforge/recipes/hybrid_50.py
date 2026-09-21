"""hybrid-50 recipe: 1:1 alternation of attention and Mamba2 (Jamba-style).

Based on Jamba (AI21, 2024): alternating attention and Mamba layers in a 1:1 ratio.
"""

from __future__ import annotations

from typing import Any

from ssmforge.config import LayerSpec, LayerType, DistillationConfig, TrainingStage
from ssmforge.recipes.base import Recipe, register_recipe


@register_recipe
class Hybrid50Recipe(Recipe):
    name = "hybrid-50"
    description = "Alternate attention and Mamba2 layers 1:1. Higher SSM ratio than hybrid-25."
    requires_attention_fraction = 0.5
    paper_reference = "https://arxiv.org/abs/2403.19887 (Jamba, AI21)"

    def plan(self, model: Any) -> list[LayerSpec]:
        num_layers = 32
        if model is not None and hasattr(model, "config"):
            num_layers = model.config.num_hidden_layers

        specs = []
        for i in range(num_layers):
            layer_type = LayerType.ATTENTION if i % 2 == 0 else LayerType.SSM
            specs.append(LayerSpec(layer_type=layer_type, index=i, freeze_mlp=True))
        return specs

    def distillation_config(self) -> DistillationConfig:
        return DistillationConfig(
            stages=[
                TrainingStage(
                    name="stepwise_alignment",
                    epochs=1,
                    learning_rate=1e-5,
                    batch_size=1,
                    gradient_accumulation_steps=8,
                    freeze_mlp=True,
                    stepwise=True,
                ),
                TrainingStage(
                    name="end_to_end_distill",
                    epochs=5,
                    learning_rate=3e-5,
                    batch_size=1,
                    gradient_accumulation_steps=8,
                    freeze_mlp=False,
                    stepwise=False,
                ),
            ],
            kl_weight=0.7,
            seqkd_weight=0.3,
            max_seq_length=2048,
            warmup_steps=200,
        )

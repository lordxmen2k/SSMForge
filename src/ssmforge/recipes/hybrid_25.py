"""hybrid-25 recipe: 25% of middle attention layers replaced with Mamba2.

Based on "The Mamba in the Llama" (NeurIPS 2024). Keeps first 2 and last 2
attention layers, replaces middle layers such that ~25% of total layers are SSM.
"""

from __future__ import annotations

from typing import Any

from ssmforge.config import LayerSpec, LayerType, DistillationConfig, TrainingStage
from ssmforge.recipes.base import Recipe, register_recipe


@register_recipe
class Hybrid25Recipe(Recipe):
    name = "hybrid-25"
    description = "Replace ~25% of middle attention layers with Mamba2 blocks. Recommended for most use cases."
    requires_attention_fraction = 0.75
    paper_reference = "https://arxiv.org/abs/2408.15237 (MambaInLlama, NeurIPS 2024)"

    def plan(self, model: Any) -> list[LayerSpec]:
        """Keep first 2 + last 2 attention layers. Convert middle layers with ~25% SSM ratio."""
        num_layers = 32
        if model is not None and hasattr(model, "config"):
            num_layers = model.config.num_hidden_layers

        specs: list[LayerSpec] = []
        for i in range(num_layers):
            if i < 2 or i >= num_layers - 2:
                specs.append(LayerSpec(layer_type=LayerType.ATTENTION, index=i, freeze_mlp=True))
            else:
                middle_idx = i - 2
                if middle_idx % 4 == 0:
                    specs.append(LayerSpec(layer_type=LayerType.SSM, index=i, freeze_mlp=True))
                else:
                    specs.append(LayerSpec(layer_type=LayerType.ATTENTION, index=i, freeze_mlp=True))
        return specs

    def distillation_config(self) -> DistillationConfig:
        """Two-stage: stepwise layer alignment + end-to-end KL distillation."""
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
                    epochs=3,
                    learning_rate=5e-5,
                    batch_size=1,
                    gradient_accumulation_steps=8,
                    freeze_mlp=False,
                    stepwise=False,
                ),
            ],
            kl_weight=0.7,
            seqkd_weight=0.3,
            max_seq_length=2048,
            warmup_steps=100,
        )

"""pure-attention recipe: pass through all layers as attention, no SSM substitution.

This is the baseline recipe for verifying the conversion pipeline is
non-destructive. It does NOT replace any layers with SSM blocks, so the
resulting model is architecturally identical to the source transformer
(just routed through our F16 GGUF writer + llama-quantize + loader).

Use case: smoke test for the conversion infrastructure. If a pure-attention
recipe produces coherent output from the source model, then any degradation
in the hybrid-* recipes is due to untrained SSM layers, not bugs in
the conversion pipeline.

Usage:
    ssmforge convert Qwen/Qwen2-1.5B-Instruct \\
        --recipe pure-attention \\
        --quantize Q4_K_M \\
        --no-distill \\
        --output ./out-qwen2-pure-attn

WARNING: experimental — this recipe exists to verify infrastructure, not
to be useful in itself. There is no SSM in the output.
"""

from __future__ import annotations

from typing import Any

from ssmforge.config import LayerSpec, LayerType, DistillationConfig, TrainingStage
from ssmforge.recipes.base import Recipe, register_recipe


@register_recipe
class PureAttentionRecipe(Recipe):
    name = "pure-attention"
    description = (
        "Pass-through recipe: keeps all layers as attention. Use as a baseline "
        "to verify the conversion pipeline is non-destructive. No SSM substitution."
    )
    # All layers stay as attention, so we require 100% attention
    requires_attention_fraction = 1.0
    paper_reference = None  # No paper — this is an SSMForge diagnostic recipe

    def plan(self, model: Any) -> list[LayerSpec]:
        """All layers stay as attention — no SSM substitution."""
        num_layers = 32
        if model is not None and hasattr(model, "config"):
            num_layers = model.config.num_hidden_layers

        specs: list[LayerSpec] = []
        for i in range(num_layers):
            specs.append(LayerSpec(layer_type=LayerType.ATTENTION, index=i, freeze_mlp=True))
        return specs

    def distillation_config(self) -> DistillationConfig:
        """No-op distillation config.

        Pure-attention has no SSM layers to distill, so distillation would
        be wasted compute. The pipeline should skip Stage 4 when ssm_count
        is 0 (see pipeline.py); this config exists only to satisfy the
        abstract base class.
        """
        return DistillationConfig(
            stages=[
                TrainingStage(
                    name="no_op",
                    epochs=0,
                    learning_rate=0.0,
                    batch_size=1,
                    gradient_accumulation_steps=1,
                    freeze_mlp=True,
                    stepwise=False,
                ),
            ],
            kl_weight=0.0,
            seqkd_weight=0.0,
            max_seq_length=2048,
            warmup_steps=0,
        )

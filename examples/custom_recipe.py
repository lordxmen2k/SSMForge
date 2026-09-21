"""Example: register a custom recipe."""

from ssmforge.recipes import Recipe, register_recipe
from ssmforge.config import LayerSpec, LayerType, DistillationConfig, TrainingStage


@register_recipe
class Hybrid75Recipe(Recipe):
    """Custom: keep 75% attention, only convert middle layers sparsely."""
    name = "hybrid-75"
    description = "Sparser SSM replacement — only ~12.5% of layers converted"
    requires_attention_fraction = 0.875

    def plan(self, model):
        num_layers = model.config.num_hidden_layers
        return [
            LayerSpec(
                layer_type=LayerType.SSM if (i % 8 == 4) else LayerType.ATTENTION,
                index=i,
            )
            for i in range(num_layers)
        ]

    def distillation_config(self):
        return DistillationConfig(
            stages=[TrainingStage(name="e2e", epochs=2, learning_rate=5e-5)]
        )


if __name__ == "__main__":
    from ssmforge.recipes import list_recipes
    from ssmforge import convert

    print(f"Registered recipes: {list_recipes()}")
    print("Now you can use: convert(source='...', recipe='hybrid-75', ...)")

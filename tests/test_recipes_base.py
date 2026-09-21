import pytest
from ssmforge.config import LayerSpec, LayerType, DistillationConfig, TrainingStage
from ssmforge.recipes.base import (
    Recipe,
    register_recipe,
    get_recipe,
    _clear_registry,
    RecipeRegistryError,
)
from ssmforge.exceptions import UnknownRecipeError


class _StubRecipe(Recipe):
    name = "stub-test"
    description = "for tests"
    requires_attention_fraction = 0.5
    paper_reference = None

    def plan(self, model):
        return [LayerSpec(layer_type=LayerType.SSM if i % 2 else LayerType.ATTENTION) for i in range(4)]

    def distillation_config(self):
        return DistillationConfig(stages=[TrainingStage(name="e2e", epochs=1, learning_rate=1e-5)])


@pytest.fixture(autouse=True)
def _clean_registry():
    _clear_registry()
    yield
    _clear_registry()


def test_register_and_get_recipe():
    register_recipe(_StubRecipe)
    recipe = get_recipe("stub-test")
    assert recipe.name == "stub-test"
    assert recipe.requires_attention_fraction == 0.5


def test_get_unknown_recipe_raises():
    with pytest.raises(UnknownRecipeError) as exc:
        get_recipe("nonexistent")
    assert "nonexistent" in str(exc.value)


def test_register_duplicate_raises():
    register_recipe(_StubRecipe)
    with pytest.raises(RecipeRegistryError):
        register_recipe(_StubRecipe)


def test_plan_returns_layer_specs():
    register_recipe(_StubRecipe)
    recipe = get_recipe("stub-test")
    plan = recipe.plan(None)
    assert len(plan) == 4
    assert plan[0].layer_type == LayerType.ATTENTION
    assert plan[1].layer_type == LayerType.SSM


def test_distillation_config_has_stages():
    register_recipe(_StubRecipe)
    recipe = get_recipe("stub-test")
    cfg = recipe.distillation_config()
    assert len(cfg.stages) == 1
    assert cfg.stages[0].epochs == 1
    assert cfg.stages[0].learning_rate == 1e-5

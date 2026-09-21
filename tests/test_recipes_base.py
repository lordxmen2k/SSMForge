"""Tests for Recipe ABC + registry.

These tests use a unique stub recipe and don't touch the global registry
of real recipes. The `_clear_registry` helper exists for test isolation but
is not called here — recipes are class-level decorators and clearing breaks
other test files that depend on built-in recipes.
"""

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
    name = "stub-test-base-isolated"
    description = "for tests"
    requires_attention_fraction = 0.5
    paper_reference = None

    def plan(self, model):
        return [LayerSpec(layer_type=LayerType.SSM if i % 2 else LayerType.ATTENTION) for i in range(4)]

    def distillation_config(self):
        return DistillationConfig(stages=[TrainingStage(name="e2e", epochs=1, learning_rate=1e-5)])


def test_register_and_get_recipe():
    register_recipe(_StubRecipe)
    recipe = get_recipe("stub-test-base-isolated")
    assert recipe.name == "stub-test-base-isolated"
    assert recipe.requires_attention_fraction == 0.5


def test_get_unknown_recipe_raises():
    with pytest.raises(UnknownRecipeError) as exc:
        get_recipe("nonexistent-this-will-never-be-registered")
    assert "nonexistent" in str(exc.value)


def test_register_duplicate_raises():
    # Use a fresh, never-registered stub name
    class _StubDup(Recipe):
        name = "stub-dup-test"
        description = "dup test"
        requires_attention_fraction = 0.5

        def plan(self, model):
            return []

        def distillation_config(self):
            return DistillationConfig(stages=[TrainingStage(name="x", epochs=1)])

    register_recipe(_StubDup)
    with pytest.raises(RecipeRegistryError):
        register_recipe(_StubDup)


def test_plan_returns_layer_specs():
    recipe = get_recipe("stub-test-base-isolated")
    plan = recipe.plan(None)
    assert len(plan) == 4
    assert plan[0].layer_type == LayerType.ATTENTION
    assert plan[1].layer_type == LayerType.SSM


def test_distillation_config_has_stages():
    recipe = get_recipe("stub-test-base-isolated")
    cfg = recipe.distillation_config()
    assert len(cfg.stages) == 1
    assert cfg.stages[0].epochs == 1
    assert cfg.stages[0].learning_rate == 1e-5

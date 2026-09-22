"""Tests for the pure-attention recipe (baseline / non-destructive pipeline check)."""

import pytest
from ssmforge.recipes import get_recipe, list_recipes
from ssmforge.config import LayerType

# Importing the module registers the recipe (decorator runs at import time).
import ssmforge.recipes.pure_attention  # noqa: F401, E402


def test_pure_attention_is_registered():
    assert "pure-attention" in list_recipes()


def test_pure_attention_attention_fraction():
    recipe = get_recipe("pure-attention")
    assert recipe.requires_attention_fraction == 1.0


def test_pure_attention_plan_has_no_ssm_layers():
    """Every layer in the plan should be attention (no SSM substitution)."""
    recipe = get_recipe("pure-attention")
    plan = recipe.plan(model=None)
    assert len(plan) > 0
    for spec in plan:
        assert spec.layer_type == LayerType.ATTENTION, \
            f"pure-attention must not have SSM layers, got {spec.layer_type} at index {spec.index}"


def test_pure_attention_plan_respects_model_layer_count():
    """When given a model with N layers, plan should have N specs."""
    from types import SimpleNamespace
    fake_model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=42))
    recipe = get_recipe("pure-attention")
    plan = recipe.plan(fake_model)
    assert len(plan) == 42


def test_pure_attention_distillation_config_is_no_op():
    """Pure-attention has no SSM layers, so distillation should be a no-op."""
    recipe = get_recipe("pure-attention")
    cfg = recipe.distillation_config()
    assert len(cfg.stages) == 1
    assert cfg.stages[0].epochs == 0
    assert cfg.kl_weight == 0.0
    assert cfg.seqkd_weight == 0.0


def test_pure_attention_plan_indices_match_layer_order():
    """Spec.index should equal the position in the plan (0, 1, 2, ...)."""
    recipe = get_recipe("pure-attention")
    plan = recipe.plan(model=None)
    for i, spec in enumerate(plan):
        assert spec.index == i, f"spec at position {i} has index {spec.index}"

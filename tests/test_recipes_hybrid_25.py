import pytest
from ssmforge.recipes import get_recipe, list_recipes
from ssmforge.config import LayerType

# Importing the module registers the recipe (decorator runs at import time).
import ssmforge.recipes.hybrid_25  # noqa: F401, E402


def test_hybrid_25_is_registered():
    assert "hybrid-25" in list_recipes()


def test_hybrid_25_attention_fraction():
    recipe = get_recipe("hybrid-25")
    assert recipe.requires_attention_fraction == 0.75


def test_hybrid_25_plan_keeps_first_and_last():
    recipe = get_recipe("hybrid-25")
    plan = recipe.plan(model=None)
    for i in [0, 1, -2, -1]:
        assert plan[i].layer_type == LayerType.ATTENTION, f"layer {i} should be attention"


def test_hybrid_25_plan_replaces_middle_quarter():
    recipe = get_recipe("hybrid-25")
    plan = recipe.plan(model=None)
    ssm_count = sum(1 for spec in plan if spec.layer_type == LayerType.SSM)
    attn_count = sum(1 for spec in plan if spec.layer_type == LayerType.ATTENTION)
    assert ssm_count > 0
    assert attn_count == len(plan) - ssm_count
    assert 0.20 <= ssm_count / len(plan) <= 0.30


def test_hybrid_25_distillation_config_has_stepwise():
    recipe = get_recipe("hybrid-25")
    cfg = recipe.distillation_config()
    assert len(cfg.stages) >= 2
    stepwise = [s for s in cfg.stages if s.stepwise]
    assert len(stepwise) >= 1

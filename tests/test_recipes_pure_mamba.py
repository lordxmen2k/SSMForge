from ssmforge.recipes import get_recipe, list_recipes
from ssmforge.config import LayerType

import ssmforge.recipes.pure_mamba  # noqa: F401, E402  registers on import


def test_pure_mamba_is_registered():
    assert "pure-mamba" in list_recipes()


def test_pure_mamba_attention_fraction():
    recipe = get_recipe("pure-mamba")
    assert recipe.requires_attention_fraction == 0.0


def test_pure_mamba_plan_all_ssm():
    recipe = get_recipe("pure-mamba")
    plan = recipe.plan(model=None)
    assert all(spec.layer_type == LayerType.SSM for spec in plan)


def test_pure_mamba_has_two_stage_distillation():
    recipe = get_recipe("pure-mamba")
    cfg = recipe.distillation_config()
    stage_names = [s.name for s in cfg.stages]
    assert any("linear" in n for n in stage_names)
    assert any("mamba" in n for n in stage_names)

from ssmforge.recipes import get_recipe, list_recipes
from ssmforge.config import LayerType

# Import to trigger decorator registration
import ssmforge.recipes.hybrid_50  # noqa: F401, E402


def test_hybrid_50_is_registered():
    assert "hybrid-50" in list_recipes()


def test_hybrid_50_attention_fraction():
    recipe = get_recipe("hybrid-50")
    assert recipe.requires_attention_fraction == 0.5


def test_hybrid_50_plan_alternates():
    recipe = get_recipe("hybrid-50")
    plan = recipe.plan(model=None)
    for i, spec in enumerate(plan):
        expected = LayerType.ATTENTION if i % 2 == 0 else LayerType.SSM
        assert spec.layer_type == expected, f"layer {i}: expected {expected}, got {spec.layer_type}"


def test_hybrid_50_50_percent_ssm():
    recipe = get_recipe("hybrid-50")
    plan = recipe.plan(model=None)
    ssm_count = sum(1 for s in plan if s.layer_type == LayerType.SSM)
    assert abs(ssm_count / len(plan) - 0.5) < 0.05


def test_hybrid_50_distillation_has_e2e():
    recipe = get_recipe("hybrid-50")
    cfg = recipe.distillation_config()
    stage_names = [s.name for s in cfg.stages]
    assert any("end_to_end" in n for n in stage_names)

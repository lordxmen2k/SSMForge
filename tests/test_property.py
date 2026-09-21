"""Property-based tests using hypothesis.

These tests verify invariants that should hold for any valid input,
catching edge cases that example-based tests miss.
"""

import pytest
from hypothesis import given, strategies as st

from ssmforge.config import LayerSpec, LayerType
from ssmforge.export.manifest import Manifest


# Skip if hypothesis isn't installed (it's in dev extras)
hypothesis = pytest.importorskip("hypothesis")


@pytest.fixture(autouse=True)
def _load_recipes():
    import ssmforge.recipes.hybrid_25  # noqa: F401
    import ssmforge.recipes.hybrid_50  # noqa: F401
    import ssmforge.recipes.pure_mamba  # noqa: F401


@given(st.integers(min_value=16, max_value=80))
def test_hybrid_25_plan_invariants(num_layers):
    """For any depth 16-80, the hybrid-25 plan satisfies invariants.

    Small depths (<16) excluded: anchor layers (first/last 2) eat too much
    of the model for the SSM ratio to converge.
    """
    from ssmforge.recipes.hybrid_25 import Hybrid25Recipe

    class FakeModel:
        class config:
            pass

    m = FakeModel()
    m.config.num_hidden_layers = num_layers

    recipe = Hybrid25Recipe()
    plan = recipe.plan(m)
    assert len(plan) == num_layers
    for i in [0, 1, -2, -1]:
        assert plan[i].layer_type == LayerType.ATTENTION
    ssm_count = sum(1 for s in plan if s.layer_type == LayerType.SSM)
    middle_count = max(num_layers - 4, 0)
    if middle_count > 0:
        ratio = ssm_count / middle_count
        # Allow 25% target ± 0.05 to absorb small-layer off-by-one
        assert 0.20 <= ratio <= 0.35


@given(st.integers(min_value=10, max_value=80))
def test_hybrid_50_plan_is_50_percent(num_layers):
    """For any depth 10-80, hybrid-50 keeps ~50% SSM ratio (±2 layers)."""
    from ssmforge.recipes.hybrid_50 import Hybrid50Recipe

    class FakeModel:
        class config:
            pass

    m = FakeModel()
    m.config.num_hidden_layers = num_layers

    recipe = Hybrid50Recipe()
    plan = recipe.plan(m)
    assert len(plan) == num_layers
    ssm_count = sum(1 for s in plan if s.layer_type == LayerType.SSM)
    # Allow up to 2-layer tolerance for off-by-one in alternation
    assert abs(ssm_count - num_layers / 2) <= 2


@given(
    st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=32),
            st.sampled_from([LayerType.ATTENTION, LayerType.SSM]),
        ),
        min_size=1,
        max_size=32,
    )
)
def test_layer_spec_round_trip(layers):
    """LayerSpec JSON round-trip via Pydantic preserves fields."""
    import json
    from datetime import datetime

    spec_dicts = [
        {"index": idx, "layer_type": lt.value}
        for idx, lt in layers
    ]
    m = Manifest(
        ssmforge_version="0.1.0",
        source_model="test",
        source_revision="abc",
        recipe="hybrid-25",
        quant_type="F16",
        layer_mapping=spec_dicts,
        calibration_data_sha=None,
        training_stats=None,
        output_gguf_path="/tmp/test.gguf",
        output_gguf_sha="deadbeef",
        output_gguf_bytes=1024,
        created_at=datetime.now(),
    )
    json_str = m.model_dump_json()
    loaded = Manifest.model_validate_json(json_str)
    assert loaded.layer_mapping == m.layer_mapping

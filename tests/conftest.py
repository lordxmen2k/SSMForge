import pytest


@pytest.fixture
def small_model_id() -> str:
    """Smallest reasonable HF causal LM for tests."""
    return "meta-llama/Llama-3.2-1B"


@pytest.fixture
def tiny_model_id() -> str:
    """Tiny random HF model (~50MB, no auth) for fast integration tests."""
    return "hf-internal-testing/tiny-random-LlamaForCausalLM"

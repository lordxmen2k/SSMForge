import pytest

pytestmark = pytest.mark.integration


def test_compute_perplexity_returns_finite_number():
    from unittest.mock import MagicMock
    from ssmforge.benchmark.quality import compute_perplexity

    model = MagicMock()
    model.eval = MagicMock()
    model.return_value = MagicMock(loss=MagicMock(item=MagicMock(return_value=0.5)))
    tokenizer = MagicMock()
    tokenizer.return_value = MagicMock(input_ids=MagicMock(shape=[1, 16]))

    ppl = compute_perplexity(model, tokenizer, ["hello world", "this is a test"], max_length=16)
    assert ppl > 0
    assert ppl < 1e6


def test_benchmark_long_context_runs_on_mock():
    from unittest.mock import MagicMock
    import torch

    from ssmforge.benchmark.long_context import benchmark_long_context

    model = MagicMock()
    tokenizer = MagicMock()
    tokenizer.vocab_size = 1000

    results = benchmark_long_context(model, tokenizer, context_lengths=[64, 128])
    assert 64 in results
    assert 128 in results
    for ctx, stats in results.items():
        if "error" not in stats:
            assert "elapsed_sec" in stats
            assert "tokens_per_sec" in stats

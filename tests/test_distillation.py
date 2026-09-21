import pytest
import torch

from ssmforge.distillation.loss import compute_kl_loss, compute_seqkd_loss
from ssmforge.distillation.collator import KLDistillationCollator


def test_kl_loss_zero_when_logits_match():
    logits = torch.randn(2, 10, 100)
    loss = compute_kl_loss(logits, logits)
    assert loss.item() < 1e-6


def test_kl_loss_positive_when_logits_differ():
    # Make teacher concentrate on different tokens than student
    student = torch.zeros(2, 10, 100)
    student[:, :, 0] = 10.0  # student favors token 0
    teacher = torch.zeros(2, 10, 100)
    teacher[:, :, 1] = 10.0  # teacher favors token 1
    loss = compute_kl_loss(student, teacher)
    assert loss.item() > 0


def test_kl_loss_gradient_flows():
    student = torch.randn(2, 10, 100, requires_grad=True)
    teacher = torch.randn(2, 10, 100)
    loss = compute_kl_loss(student, teacher)
    loss.backward()
    assert student.grad is not None
    assert student.grad.abs().sum() > 0


def test_seqkd_loss_uses_teacher_argmax():
    student = torch.randn(2, 10, 100, requires_grad=True)
    teacher = torch.randn(2, 10, 100)
    labels = teacher.argmax(dim=-1)
    loss = compute_seqkd_loss(student, labels)
    loss.backward()
    assert student.grad is not None


def test_kl_weight_alpha_changes_loss():
    student = torch.randn(2, 10, 100)
    teacher = torch.randn(2, 10, 100)
    loss_full_kl = compute_kl_loss(student, teacher, alpha=1.0)
    loss_half_kl = compute_kl_loss(student, teacher, alpha=0.5)
    assert not torch.isclose(loss_full_kl, loss_half_kl)


def test_collator_tokenizes_input():
    """Mock tokenizer for fast unit test (no HF download)."""
    class MockTokenizer:
        def __call__(self, texts, padding, truncation, max_length, return_tensors):
            n = len(texts)
            ids = torch.randint(0, 100, (n, max_length))
            return {"input_ids": ids}

    collator = KLDistillationCollator(tokenizer=MockTokenizer(), max_length=16)
    batch = collator(["hello world", "this is a test"])
    assert "input_ids" in batch
    assert batch["input_ids"].shape == (2, 16)
    assert "labels" in batch
    assert torch.equal(batch["labels"], batch["input_ids"])

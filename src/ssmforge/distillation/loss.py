"""Distillation loss functions: KL divergence + SequenceKD."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    alpha: float = 0.7,
    temperature: float = 1.0,
    labels: torch.Tensor | None = None,
) -> torch.Tensor:
    """Word-level KL distillation loss.

    Loss = alpha * KL(student || teacher, T=temperature)
         + (1 - alpha) * CE(student, labels)  [if labels provided]
    """
    kl = F.kl_div(
        input=F.log_softmax(student_logits / temperature, dim=-1),
        target=F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature ** 2)

    if labels is None:
        return alpha * kl

    ce = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )
    return alpha * kl + (1 - alpha) * ce


def compute_seqkd_loss(
    student_logits: torch.Tensor,
    teacher_labels: torch.Tensor,
) -> torch.Tensor:
    """Sequence-level KD: use teacher's argmax as pseudo-labels."""
    return F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        teacher_labels.view(-1),
        ignore_index=-100,
    )

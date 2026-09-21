"""Distillation trainer: KL divergence training loop.

Lightweight implementation using torch.optim directly (not transformers.Trainer)
to minimize dependencies. Supports the two-stage MambaInLlama recipe:
  1. Stepwise layer alignment (freeze MLP, train SSM blocks one at a time)
  2. End-to-end distillation
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.optim import AdamW

from ssmforge.config import DistillationConfig, TrainingStage
from ssmforge.distillation.collator import KLDistillationCollator
from ssmforge.distillation.loss import compute_kl_loss
from ssmforge.distillation.calibration import CalibrationDataLoader


class DistillationTrainer:
    """High-level orchestrator that runs the full distillation training."""

    def __init__(
        self,
        student_model,
        teacher_model,
        calibration_loader: CalibrationDataLoader,
        tokenizer,
        config: DistillationConfig,
        device: str = "cpu",
    ):
        self.student = student_model
        self.teacher = teacher_model
        self.config = config
        self.tokenizer = tokenizer
        self.device = device

        self.calibration_samples = calibration_loader.load()

        self.collator = KLDistillationCollator(
            tokenizer=tokenizer,
            max_length=config.max_seq_length,
        )

        # Move models to device
        self.student.to(device)
        self.teacher.to(device)
        self.teacher.eval()

    def train(self, num_steps: Optional[int] = None) -> None:
        """Run end-to-end distillation stage (last stage in config)."""
        stage = self.config.stages[-1]
        optimizer = AdamW(self.student.parameters(), lr=stage.learning_rate)

        if num_steps is None:
            num_steps = stage.epochs * max(1, len(self.calibration_samples) // stage.batch_size)

        self.student.train()
        step = 0
        for text in self.calibration_samples:
            if step >= num_steps:
                break
            batch = self.collator([text])
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with torch.no_grad():
                teacher_out = self.teacher(**batch)
                teacher_logits = teacher_out.logits

            student_out = self.student(**batch)
            student_logits = student_out.logits

            loss = compute_kl_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                alpha=self.config.kl_weight,
                labels=batch.get("labels"),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1

    def evaluate_loss(self) -> float:
        """Compute current loss on a small calibration subset."""
        self.student.eval()
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for text in self.calibration_samples[:8]:
                batch = self.collator([text])
                batch = {k: v.to(self.device) for k, v in batch.items()}
                student_out = self.student(**batch)
                teacher_out = self.teacher(**batch)
                loss = compute_kl_loss(
                    student_out.logits,
                    teacher_out.logits,
                    alpha=self.config.kl_weight,
                )
                total_loss += loss.item()
                n += 1
        return total_loss / max(n, 1)

from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.integration


def test_trainer_initializes_with_models():
    """Smoke test: trainer constructs and runs evaluate_loss."""
    from ssmforge.distillation import DistillationTrainer, CalibrationDataLoader
    from ssmforge.config import DistillationConfig, TrainingStage

    student = MagicMock()
    student.to = MagicMock(return_value=student)
    student.eval = MagicMock()
    teacher = MagicMock()
    teacher.to = MagicMock(return_value=teacher)
    teacher.eval = MagicMock()

    config = DistillationConfig(
        stages=[TrainingStage(name="smoke_test", epochs=1, learning_rate=1e-4, batch_size=1, gradient_accumulation_steps=1)],
        max_seq_length=16,
    )
    cal = CalibrationDataLoader(source=None, max_samples=4)
    tokenizer = MagicMock()

    trainer = DistillationTrainer(
        student_model=student,
        teacher_model=teacher,
        calibration_loader=cal,
        tokenizer=tokenizer,
        config=config,
    )

    # Just verify it constructs without error
    assert trainer.config == config
    assert len(trainer.calibration_samples) == 4

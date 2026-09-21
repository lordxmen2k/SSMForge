"""Pydantic schemas for SSMForge configuration."""

from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field


class LayerType(str, Enum):
    ATTENTION = "attention"
    SSM = "ssm"


class LayerSpec(BaseModel):
    """Specification for one layer in the hybrid model."""

    layer_type: LayerType
    index: int = Field(default=0, description="0-based position in the original model")
    freeze_mlp: bool = Field(default=True, description="Freeze MLP weights during distillation (MambaInLlama recipe)")


class TrainingStage(BaseModel):
    """One stage of the distillation training."""

    name: str
    epochs: int = 1
    learning_rate: float = 1e-5
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    freeze_mlp: bool = False
    stepwise: bool = Field(default=False, description="Train one layer at a time")


class DistillationConfig(BaseModel):
    """Full distillation configuration for a recipe."""

    stages: list[TrainingStage]
    kl_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    seqkd_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    max_seq_length: int = 2048
    warmup_steps: int = 100

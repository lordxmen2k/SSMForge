"""Architecture converter base + registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ssmforge.config import LayerSpec


class ArchitectureConverter(ABC):
    """Converts a source model's state dict to a hybrid state dict per a recipe plan."""

    source_arch: str
    target_arch: str

    @abstractmethod
    def convert_state_dict(self, src: dict, plan: list[LayerSpec]) -> dict:
        """Pure function: source SD + plan → target SD.

        For MVP: copies embeddings/MLP/norm/head verbatim, replaces attention
        layers marked as SSM with placeholder Mamba2 tensors.
        """
        ...

    def build_model(self, src_config: Any, target_sd: dict) -> Any:
        """Build the target model shell + load weights. Raises NotImplementedError in MVP."""
        raise NotImplementedError("Model construction requires mamba-ssm; see Task 6")

    def verify_round_trip(self, src_model: Any, target_model: Any, prompts: list[str]) -> bool:
        """Forward-pass sanity check. Raises NotImplementedError in MVP."""
        raise NotImplementedError("Verification requires full model construction; see Task 6")


class ArchitectureConverterRegistry:
    _registry: dict[str, type[ArchitectureConverter]] = {}

    @classmethod
    def register(cls, arch: str, converter_cls: type[ArchitectureConverter]) -> None:
        cls._registry[arch] = converter_cls

    @classmethod
    def get(cls, arch: str) -> ArchitectureConverter:
        from ssmforge.exceptions import UnsupportedArchitectureError

        if arch not in cls._registry:
            raise UnsupportedArchitectureError(
                arch=arch,
                version="0.1.0",
                supported=cls.list_supported(),
            )
        return cls._registry[arch]()

    @classmethod
    def list_supported(cls) -> list[str]:
        return sorted(cls._registry.keys())

"""Recipe abstract base class and global registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ssmforge.config import LayerSpec, DistillationConfig


class RecipeRegistryError(Exception):
    """Raised when registering a duplicate recipe name."""


_REGISTRY: dict[str, type[Recipe]] = {}


def register_recipe(cls: type[Recipe]) -> type[Recipe]:
    """Register a recipe class. Can be used as a decorator."""
    if cls.name in _REGISTRY:
        raise RecipeRegistryError(f"Recipe '{cls.name}' is already registered")
    _REGISTRY[cls.name] = cls
    return cls


def get_recipe(name: str) -> Recipe:
    """Look up a registered recipe by name."""
    from ssmforge.exceptions import UnknownRecipeError

    if name not in _REGISTRY:
        raise UnknownRecipeError(recipe=name, registered=sorted(_REGISTRY.keys()))
    return _REGISTRY[name]()


def list_recipes() -> list[str]:
    """Return names of all registered recipes."""
    return sorted(_REGISTRY.keys())


def _clear_registry() -> None:
    """Clear all registered recipes. For tests only."""
    _REGISTRY.clear()


class Recipe(ABC):
    """Abstract base for SSMForge conversion recipes.

    A recipe decides:
    - which layers become attention vs SSM (plan())
    - how distillation is configured (distillation_config())
    """

    name: str
    description: str
    requires_attention_fraction: float
    paper_reference: str | None = None

    @abstractmethod
    def plan(self, model: Any) -> list[LayerSpec]:
        """Return ordered layer specs for this model."""
        ...

    @abstractmethod
    def distillation_config(self) -> DistillationConfig:
        """Return distillation training configuration."""
        ...

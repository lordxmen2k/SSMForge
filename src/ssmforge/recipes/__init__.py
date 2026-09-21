"""SSMForge recipes.

Importing this package registers all built-in recipes via decorator side effects.
"""

from ssmforge.recipes.base import (
    Recipe,
    register_recipe,
    get_recipe,
    list_recipes,
    RecipeRegistryError,
)

# Import built-in recipes to trigger their registration.
from ssmforge.recipes import hybrid_25  # noqa: F401, E402

__all__ = [
    "Recipe",
    "register_recipe",
    "get_recipe",
    "list_recipes",
    "RecipeRegistryError",
]

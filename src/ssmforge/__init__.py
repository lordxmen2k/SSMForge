"""SSMForge: convert pretrained transformers to hybrid SSM/attention models."""

__version__ = "0.1.0"

from ssmforge.pipeline import convert
from ssmforge.result import ConversionResult
from ssmforge import recipes  # registers recipes on import
from ssmforge.recipes import list_recipes

__all__ = ["convert", "ConversionResult", "list_recipes", "__version__"]

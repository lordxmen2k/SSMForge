from ssmforge.converters.base import (
    ArchitectureConverter,
    ArchitectureConverterRegistry,
)
from ssmforge.converters.llama_to_hybrid import LlamaToHybridConverter

# Auto-register built-in converters
ArchitectureConverterRegistry.register("llama", LlamaToHybridConverter)

__all__ = ["ArchitectureConverter", "ArchitectureConverterRegistry", "LlamaToHybridConverter"]

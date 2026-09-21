from ssmforge.converters.base import (
    ArchitectureConverter,
    ArchitectureConverterRegistry,
)
from ssmforge.converters.llama_to_hybrid import LlamaToHybridConverter
from ssmforge.converters.mistral_to_hybrid import MistralToHybridConverter
from ssmforge.converters.qwen2_to_hybrid import Qwen2ToHybridConverter
from ssmforge.converters.phi3_to_hybrid import Phi3ToHybridConverter

# Trigger registration via direct import (registers on import via module-level code)
import ssmforge.converters.llama_to_hybrid  # noqa: F401
import ssmforge.converters.mistral_to_hybrid  # noqa: F401
import ssmforge.converters.qwen2_to_hybrid  # noqa: F401
import ssmforge.converters.phi3_to_hybrid  # noqa: F401

__all__ = [
    "ArchitectureConverter",
    "ArchitectureConverterRegistry",
    "LlamaToHybridConverter",
    "MistralToHybridConverter",
    "Qwen2ToHybridConverter",
    "Phi3ToHybridConverter",
]

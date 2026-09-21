"""State-dict surgery for Mistral → hybrid Mistral+Mamba2.

Mistral uses sliding window attention by default. We preserve this in the
attention layers that survive, and replace others with Mamba2.
"""

from __future__ import annotations

from ssmforge.config import LayerSpec
from ssmforge.converters.base import ArchitectureConverter
from ssmforge.converters.llama_to_hybrid import LlamaToHybridConverter


class MistralToHybridConverter(ArchitectureConverter):
    source_arch = "mistral"
    target_arch = "hybrid-mistral-mamba2"

    def convert_state_dict(self, src: dict, plan: list[LayerSpec]) -> dict:
        llama_converter = LlamaToHybridConverter()
        return llama_converter.convert_state_dict(src, plan)


# Auto-register on import
from ssmforge.converters.base import ArchitectureConverterRegistry  # noqa: E402

ArchitectureConverterRegistry.register("mistral", MistralToHybridConverter)

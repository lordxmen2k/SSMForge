"""State-dict surgery for Gemma2 → hybrid Llama+Mamba2.

Gemma2 (google/gemma-2-2b, gemma-2-9b, gemma-2-27b) has the SAME state-dict
layout as Llama:

- q_proj, k_proj, v_proj, o_proj (separate, not fused)
- gate_proj, up_proj, down_proj (Llama-style SwiGLU shape)
- input_layernorm, post_attention_layernorm

The differences from Llama that matter for inference (but NOT for surgery):

1. **GeLU** activation (not SiLU/SwiGLU) — irrelevant for surgery since
   we copy weights verbatim and don't apply the activation.
2. **Different normalization order** (post-norm vs pre-norm in some
   Gemma2 variants) — also irrelevant for surgery.
3. **Sliding-window attention on alternating layers** — same shape
   contract as full attention; we'll keep the per-layer q/k/v intact.

So Gemma2 = Llama converter alias. We add a separate class for clarity
in the registry and future-proofing if we ever need to specialize.

Caveat: this assumes the source model's attention has bias-free
q/k/v/o projections (Gemma2 default). For models with bias=True
(rare), the bias tensors would need to be filtered out.
"""

from __future__ import annotations

from ssmforge.config import LayerSpec
from ssmforge.converters.base import ArchitectureConverter
from ssmforge.converters.llama_to_hybrid import LlamaToHybridConverter


class Gemma2ToHybridConverter(ArchitectureConverter):
    """Converts Gemma2 state dict → hybrid Llama+Mamba2 state dict.

    Gemma2 has identical state-dict layout to Llama, so we delegate
    the entire surgery to LlamaToHybridConverter.
    """

    source_arch = "gemma2"
    target_arch = "hybrid-llama-mamba2"

    def convert_state_dict(self, src: dict, plan: list[LayerSpec]) -> dict:
        # Gemma2 may ship with bias tensors on attention projections
        # (default is bias=False, but newer variants may differ). Strip
        # them so the Llama target doesn't reject unknown keys.
        cleaned = {
            k: v for k, v in src.items()
            if not k.endswith(".q_proj.bias")
            and not k.endswith(".k_proj.bias")
            and not k.endswith(".v_proj.bias")
            and not k.endswith(".o_proj.bias")
        }
        llama_converter = LlamaToHybridConverter()
        return llama_converter.convert_state_dict(cleaned, plan)


# Auto-register on import
from ssmforge.converters.base import ArchitectureConverterRegistry  # noqa: E402

ArchitectureConverterRegistry.register("gemma2", Gemma2ToHybridConverter)


__all__ = ["Gemma2ToHybridConverter"]

"""State-dict surgery for Qwen2 → hybrid Qwen2+Mamba2.

Qwen2 is architecturally very close to Llama — same RMSNorm placement,
same attention/MLP shape conventions. Differences from Llama:

1. **Tied embeddings by default** — many Qwen2 checkpoints don't ship
   a separate `lm_head.weight` (it shares the embedding matrix).
   We tie them in the target dict if lm_head is missing.
2. **No biases on attention projections** — Qwen2 uses bias=False for
   q_proj/k_proj/v_proj (different from some older Llama variants).
3. **Same tie_word_embeddings for MLP** — Qwen2's Q, K, V, O projections
   have identical layout to Llama.

We reuse the LlamaToHybridConverter's per-layer logic by reading the
source state dict and adapting the few differences. Where possible,
we use the existing Llama logic to avoid duplicating surgery code.
"""

from __future__ import annotations

from typing import Any

from ssmforge.config import LayerSpec
from ssmforge.converters.base import ArchitectureConverter
from ssmforge.converters.llama_to_hybrid import LlamaToHybridConverter


class Qwen2ToHybridConverter(ArchitectureConverter):
    """Converts a Qwen2 state dict to a hybrid Llama+Mamba2 state dict.

    The hybrid model class (HybridLlamaMambaModel) is built on top of
    the Llama stack, so Qwen2 must be converted into Llama-keyed form.
    """

    source_arch = "qwen2"
    target_arch = "hybrid-llama-mamba2"  # same target model as Llama

    def convert_state_dict(self, src: dict, plan: list[LayerSpec]) -> dict:
        # Step 1: rename Qwen2 source keys to Llama-style keys.
        # Qwen2 state_dict uses the SAME names as Llama for q/k/v/o,
        # MLP, embeddings, norms. Only the LM head may be missing.
        llama_sd = self._adapt_source_keys(src)

        # Step 2: delegate the actual surgery to the Llama converter.
        # It's already battle-tested on real Llama state dicts and
        # handles all layer types (attention / SSM) plus the SSM init.
        llama_converter = LlamaToHybridConverter()
        return llama_converter.convert_state_dict(llama_sd, plan)

    def _adapt_source_keys(self, src: dict) -> dict:
        """Adapt Qwen2 → Llama key conventions.

        Qwen2 differences we handle:
        - Missing lm_head.weight when tie_word_embeddings=True:
          copy embed_tokens.weight to lm_head.weight
        - Strip any Qwen2-only keys (none today, but future-proof)
        """
        adapted = dict(src)

        # _hidden_size is metadata; copy through
        if "_hidden_size" not in adapted:
            # Try to infer from embed_tokens shape
            for k, v in adapted.items():
                if k == "model.embed_tokens.weight" and hasattr(v, "shape"):
                    adapted["_hidden_size"] = v.shape[1]
                    break

        # lm_head tie
        if (
            "lm_head.weight" not in adapted
            and "model.embed_tokens.weight" in adapted
        ):
            adapted["lm_head.weight"] = adapted["model.embed_tokens.weight"]

        return adapted


# Auto-register on import
from ssmforge.converters.base import ArchitectureConverterRegistry  # noqa: E402

ArchitectureConverterRegistry.register("qwen2", Qwen2ToHybridConverter)


__all__ = ["Qwen2ToHybridConverter"]

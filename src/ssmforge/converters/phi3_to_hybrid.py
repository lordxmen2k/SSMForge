"""State-dict surgery for Phi-3 → hybrid Llama+Mamba2.

Phi-3 (microsoft/Phi-3-mini, Phi-3-medium) differs from Llama in three
ways that affect surgery:

1. **Fused QKV projection** — instead of separate q_proj/k_proj/v_proj,
   Phi-3 has a single `qkv_proj` of shape
   `(num_heads*head_dim + 2*num_kv_heads*head_dim, hidden_size)`.
   Slice the first axis along the output dim:
     - First num_heads*head_dim rows → q_proj
     - Next num_kv_heads*head_dim rows → k_proj
     - Last num_kv_heads*head_dim rows → v_proj

2. **Fused gate+up projection** — instead of gate_proj/up_proj,
   Phi-3 has `gate_up_proj` of shape `(2*intermediate_size, hidden_size)`.
   Slice:
     - First intermediate_size rows → gate_proj
     - Last intermediate_size rows → up_proj

3. **q/k/v linear bias=False** (same as Qwen2).

After slicing, the per-layer layout matches Llama exactly, so we delegate
to LlamaToHybridConverter's per-layer surgery.
"""

from __future__ import annotations

import torch

from ssmforge.config import LayerSpec
from ssmforge.converters.base import ArchitectureConverter
from ssmforge.converters.llama_to_hybrid import LlamaToHybridConverter


class Phi3ToHybridConverter(ArchitectureConverter):
    """Converts Phi-3 state dict → hybrid Llama+Mamba2 state dict."""

    source_arch = "phi3"
    target_arch = "hybrid-llama-mamba2"

    def convert_state_dict(self, src: dict, plan: list[LayerSpec]) -> dict:
        hidden_size = src.get("_hidden_size")
        if hidden_size is None:
            # Infer from embed_tokens
            emb = src.get("model.embed_tokens.weight")
            if isinstance(emb, torch.Tensor) and emb.ndim == 2:
                hidden_size = emb.shape[1]

        # Phi-3 needs config to slice fused projections. Pull from src metadata
        # or infer from shapes.
        num_heads = src.get("_num_attention_heads", 32)
        num_kv_heads = src.get("_num_key_value_heads", num_heads)
        head_dim = src.get("_head_dim", hidden_size // num_heads if hidden_size else 128)
        intermediate_size = src.get("_intermediate_size")

        # Build the Llama-shaped source state dict by slicing fused weights
        llama_sd = self._slice_fused_projections(
            src,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
        )

        # Delegate to LlamaToHybridConverter for the actual surgery
        llama_converter = LlamaToHybridConverter()
        return llama_converter.convert_state_dict(llama_sd, plan)

    def _slice_fused_projections(
        self,
        src: dict,
        hidden_size: int | None,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int | None,
    ) -> dict:
        """Slice Phi-3 fused projections into separate Llama-shaped tensors."""
        out: dict = dict(src)

        for key, value in list(src.items()):
            if not isinstance(value, torch.Tensor):
                continue

            # Fused QKV: model.layers.{i}.self_attn.qkv_proj.weight
            if key.endswith(".self_attn.qkv_proj.weight"):
                layer_prefix = key[: -len("qkv_proj.weight")]
                q_size = num_heads * head_dim
                kv_size = num_kv_heads * head_dim
                if value.shape[0] != q_size + 2 * kv_size:
                    # Unrecognized shape — leave it for downstream to complain
                    continue
                q_w = value[:q_size].contiguous()
                k_w = value[q_size : q_size + kv_size].contiguous()
                v_w = value[q_size + kv_size :].contiguous()
                out[f"{layer_prefix}q_proj.weight"] = q_w
                out[f"{layer_prefix}k_proj.weight"] = k_w
                out[f"{layer_prefix}v_proj.weight"] = v_w
                # Drop the original fused tensor — the model only knows the
                # separate Q/K/V projections
                out.pop(key, None)

            # Fused gate+up: model.layers.{i}.mlp.gate_up_proj.weight
            elif key.endswith(".mlp.gate_up_proj.weight"):
                layer_prefix = key[: -len("gate_up_proj.weight")]
                if intermediate_size is None:
                    # Infer: gate_up_proj has 2*intermediate_size rows
                    intermediate_size = value.shape[0] // 2
                gate_w = value[:intermediate_size].contiguous()
                up_w = value[intermediate_size : 2 * intermediate_size].contiguous()
                out[f"{layer_prefix}gate_proj.weight"] = gate_w
                out[f"{layer_prefix}up_proj.weight"] = up_w
                # Drop the original fused tensor — same reasoning as above
                out.pop(key, None)

        # Tied embeddings: Phi-3-mini ships without lm_head.weight when
        # tie_word_embeddings=True. Synthesize it from embed_tokens.
        if (
            "lm_head.weight" not in out
            and "model.embed_tokens.weight" in out
        ):
            out["lm_head.weight"] = out["model.embed_tokens.weight"]

        # Copy metadata through
        if hidden_size is not None:
            out["_hidden_size"] = hidden_size

        return out


# Auto-register on import
from ssmforge.converters.base import ArchitectureConverterRegistry  # noqa: E402

ArchitectureConverterRegistry.register("phi3", Phi3ToHybridConverter)


__all__ = ["Phi3ToHybridConverter"]

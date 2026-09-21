"""State-dict surgery for Llama → hybrid Llama+Mamba2.

For MVP: produces a target state dict where:
- Embeddings, MLP, LayerNorm, output head are copied verbatim
- Attention layers marked as SSM in the plan are replaced with placeholder
  Mamba2 tensors (random init of correct shape, ready for distillation)
"""

from __future__ import annotations

from ssmforge.config import LayerSpec, LayerType
from ssmforge.converters.base import ArchitectureConverter, ArchitectureConverterRegistry
from ssmforge.converters.weight_init import init_mamba2_from_attention


class LlamaToHybridConverter(ArchitectureConverter):
    source_arch = "llama"
    target_arch = "hybrid-llama-mamba2"

    def convert_state_dict(self, src: dict, plan: list[LayerSpec]) -> dict:
        target: dict = {}

        for key, value in src.items():
            if "layers." not in key and not key.startswith("_"):
                target[key] = value

        hidden_size = src.get("_hidden_size", 2048)

        for spec in plan:
            layer_idx = spec.index
            if spec.layer_type == LayerType.ATTENTION:
                for key, value in src.items():
                    if key.startswith(f"model.layers.{layer_idx}."):
                        target[key] = value
            else:  # SSM
                # Copy MLP + LN verbatim
                for key, value in src.items():
                    if (
                        key.startswith(f"model.layers.{layer_idx}.")
                        and (
                            "mlp." in key
                            or "post_attention_layernorm" in key
                            or "input_layernorm" in key
                        )
                    ):
                        target[key] = value
                # Initialize Mamba2 weights from attention weights
                attention_keys = [
                    k for k in src.keys()
                    if k.startswith(f"model.layers.{layer_idx}.self_attn.")
                ]
                attention_sd = {k.split("self_attn.")[-1]: src[k] for k in attention_keys}
                mamba_sd = init_mamba2_from_attention(attention_sd, hidden_size=hidden_size)
                for mk, mv in mamba_sd.items():
                    target[f"model.layers.{layer_idx}.mamba.{mk}"] = mv

        return target


# Auto-register on import
ArchitectureConverterRegistry.register("llama", LlamaToHybridConverter)

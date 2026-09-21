"""Hybrid Llama + Mamba2 model.

For MVP, this uses HuggingFace LlamaAttention layers and a custom Mamba2 wrapper.
Full Mamba2 support requires `mamba-ssm` package (install with `pip install ssmforge[mamba]`).
When mamba-ssm is not installed, a pure-PyTorch fallback SSM layer is used.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM

try:
    from mamba_ssm import Mamba2
    _MAMBA_AVAILABLE = True
except ImportError:
    _MAMBA_AVAILABLE = False


class HybridLlamaMambaConfig(LlamaConfig):
    """Llama config extended with Mamba2 layer specification."""

    model_type = "hybrid_llama_mamba"

    def __init__(
        self,
        ssm_layer_indices: Optional[list[int]] = None,
        ssm_expand: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ssm_layer_indices = ssm_layer_indices or []
        self.ssm_expand = ssm_expand


class _FallbackMambaLayer(nn.Module):
    """Pure-PyTorch SSM-like layer used when mamba-ssm is unavailable."""

    def __init__(self, hidden_size: int, expand: int = 2):
        super().__init__()
        d_inner = hidden_size * expand
        self.in_proj = nn.Linear(hidden_size, d_inner * 2, bias=True)
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=4, groups=d_inner, padding=3)
        self.x_proj = nn.Linear(d_inner, d_inner // 2, bias=False)
        self.dt_proj = nn.Linear(d_inner // 2, d_inner // 2, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.empty(d_inner // 2).uniform_(1.0, 16.0)))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, hidden_size, bias=True)
        self.d_inner = d_inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, H) → simple linear projection (placeholder for full SSM)
        z = self.in_proj(x)  # (B, L, 2*d_inner)
        return self.out_proj(z[:, :, : self.d_inner])


class HybridMamba2Layer(nn.Module):
    """Mamba2 SSM layer wrapped to be drop-in compatible with LlamaDecoderLayer.

    Uses mamba-ssm if available; falls back to a pure-PyTorch placeholder otherwise.
    """

    def __init__(self, hidden_size: int, expand: int = 2):
        super().__init__()
        if _MAMBA_AVAILABLE:
            self.mamba = Mamba2(
                d_model=hidden_size,
                d_state=128,
                d_conv=4,
                expand=expand,
            )
        else:
            self.mamba = _FallbackMambaLayer(hidden_size, expand=expand)
        self.input_layernorm = nn.LayerNorm(hidden_size, eps=1e-5)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size, eps=1e-5)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.mamba(hidden_states)
        hidden_states = residual + hidden_states
        # No-op FFN for parity with LlamaDecoderLayer; real FFN is separate
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class HybridLlamaMambaModel(LlamaForCausalLM):
    """Llama model with Mamba2 layers substituted at specified indices."""

    config_class = HybridLlamaMambaConfig

    def __init__(self, config: HybridLlamaMambaConfig):
        super().__init__(config)
        self._replace_ssm_layers()

    def _replace_ssm_layers(self):
        new_layers = nn.ModuleList()
        for i, layer in enumerate(self.model.layers):
            if i in self.config.ssm_layer_indices:
                new_layers.append(
                    HybridMamba2Layer(
                        hidden_size=self.config.hidden_size,
                        expand=self.config.ssm_expand,
                    )
                )
            else:
                new_layers.append(layer)
        self.model.layers = new_layers


__all__ = [
    "HybridLlamaMambaConfig",
    "HybridLlamaMambaModel",
    "HybridMamba2Layer",
]

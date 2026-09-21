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
    """Pure-PyTorch SSM-like layer used when mamba-ssm is unavailable.

    Mirrors the state-dict shape contract of mamba_ssm.Mamba2 with default
    args so that weights produced by the converter can be loaded either way.
    Not a real SSM — placeholder for distillation to refine.
    """

    # Defaults copied from mamba_ssm.Mamba2 to keep shape parity
    _EXPAND = 2
    _HEADDIM = 64
    _D_STATE = 128
    _NGROUPS = 1
    _D_CONV = 4

    def __init__(self, hidden_size: int, expand: int = 2):
        super().__init__()
        d_model = hidden_size
        d_inner = d_model * self._EXPAND
        nheads = d_inner // self._HEADDIM
        conv_dim = d_inner + 2 * self._NGROUPS * self._D_STATE
        in_proj_dim = 2 * d_inner + 2 * self._NGROUPS * self._D_STATE + nheads

        self.in_proj = nn.Linear(d_model, in_proj_dim, bias=False)
        self.conv1d = nn.Conv1d(
            conv_dim, conv_dim, kernel_size=self._D_CONV,
            groups=conv_dim, padding=self._D_CONV - 1, bias=True,
        )
        self.dt_bias = nn.Parameter(torch.zeros(nheads))
        # A_log and D registered as buffers (matching mamba_ssm.Mamba2)
        self.register_buffer("A_log", torch.log(torch.empty(nheads).uniform_(1.0, 16.0)))
        self.register_buffer("D", torch.ones(nheads))
        # RMSNormGated — simplified to a plain RMSNorm for the placeholder
        self.norm = nn.RMSNorm(d_inner) if hasattr(nn, "RMSNorm") else _SimpleRMSNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        self.d_inner = d_inner
        self.nheads = nheads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Placeholder behavior — let the converter-baked weights pass through
        # a structurally similar path. Distillation will shape these into real
        # SSM behavior.
        y = self.in_proj(x)  # (B, L, in_proj_dim)
        # Don't actually invoke the conv/ssm path; produce something whose
        # grad will flow back through in_proj and out_proj (the only params
        # distillation materially tunes in the first few steps anyway).
        return self.out_proj(y[:, :, : self.d_inner])


class _SimpleRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class HybridMamba2Layer(nn.Module):
    """Mamba2 SSM layer wrapped to be drop-in compatible with LlamaDecoderLayer.

    Uses mamba-ssm if available; falls back to a pure-PyTorch placeholder otherwise.
    Keeps the FFN (MLP) from LlamaDecoderLayer so converted MLP weights load.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, expand: int = 2):
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
        # LlamaDecoderLayer uses LlamaRMSNorm, not nn.LayerNorm — match that
        # exactly so converted weights load with no missing keys.
        from transformers.models.llama.modeling_llama import LlamaRMSNorm
        self.input_layernorm = LlamaRMSNorm(hidden_size, eps=1e-5)
        self.post_attention_layernorm = LlamaRMSNorm(hidden_size, eps=1e-5)
        # Reuse Llama's standard MLP. Importing here to avoid circular import at module load.
        # intermediate_size is taken from the source model — not hardcoded — so
        # non-default ratios (e.g. TinyLlama 5632/2048 = 2.75) work.
        from transformers.models.llama.modeling_llama import LlamaMLP
        self.mlp = LlamaMLP(_FakeLlamaConfig(hidden_size, intermediate_size))

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        # SSM block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.mamba(hidden_states)
        hidden_states = residual + hidden_states
        # FFN block (real Llama MLP — weights load from the source)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class _FakeLlamaConfig:
    """Minimal duck-typed config for LlamaMLP that supplies the attrs the
    transformers LlamaMLP constructor reads.
    """
    def __init__(self, hidden_size: int, intermediate_size: int):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size  # take from source, not hardcoded
        self.hidden_act = "silu"
        self.mlp_bias = False  # Llama default


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
                        intermediate_size=self.config.intermediate_size,
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

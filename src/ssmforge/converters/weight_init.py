"""Initialize Mamba2 weights from attention weights.

This writes state-dict keys that match `mamba_ssm.Mamba2` (default args).
Reference: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba2.py

Default Mamba2 parameters used here:
- expand = 2
- headdim = 64
- d_state = 128
- ngroups = 1
- bias = False
- conv_bias = True
- rmsnorm = True

For these defaults:
- d_inner = d_model * expand
- nheads = d_inner // headdim
- conv_dim = d_inner + 2 * ngroups * d_state

State-dict keys produced:
- in_proj.weight:     (2*d_inner + 2*ngroups*d_state + nheads, d_model)
- in_proj.bias:       (2*d_inner + 2*ngroups*d_state + nheads,)  (when bias=True; ours is False)
- conv1d.weight:      (conv_dim, 1, d_conv=4)
- conv1d.bias:        (conv_dim,)
- dt_bias:            (nheads,)
- A_log:              (nheads,)
- D:                  (nheads,)
- norm.weight:        (d_inner,)            # RMSNormGated
- out_proj.weight:    (d_model, d_inner)
- out_proj.bias:      (d_model,)            (when bias=True; ours is False)
"""

from __future__ import annotations

import math

import torch

# Match mamba_ssm.Mamba2 defaults
_EXPAND = 2
_HEADDIM = 64
_D_STATE = 128
_NGROUPS = 1
_D_CONV = 4
_BIAS = False
_CONV_BIAS = True


def init_mamba2_from_attention(attention_sd: dict, hidden_size: int) -> dict:
    """Convert attention weights into an initial Mamba2 state-dict that matches
    mamba_ssm.Mamba2 with default args.

    Args:
        attention_sd: attention submodule state dict (q_proj, k_proj, v_proj, o_proj)
        hidden_size: model hidden dimension (d_model)

    Returns:
        Mamba2-compatible submodule state dict
    """
    d_inner = hidden_size * _EXPAND
    nheads = d_inner // _HEADDIM
    conv_dim = d_inner + 2 * _NGROUPS * _D_STATE
    in_proj_dim = 2 * d_inner + 2 * _NGROUPS * _D_STATE + nheads

    # in_proj: Linear(d_model, in_proj_dim, bias=False)
    # Initialize with small random values (will be overwritten by distillation)
    in_proj_w = torch.empty(in_proj_dim, hidden_size)
    torch.nn.init.normal_(in_proj_w, mean=0.0, std=0.02)

    # conv1d: Conv1d(conv_dim, conv_dim, 4, groups=conv_dim, bias=True, padding=3)
    conv1d_w = torch.empty(conv_dim, 1, _D_CONV)
    torch.nn.init.kaiming_uniform_(conv1d_w, a=math.sqrt(5))
    conv1d_b = torch.zeros(conv_dim)

    # dt_bias: (nheads,). Init to small positive (after softplus it becomes a reasonable scale).
    dt_bias = torch.zeros(nheads)

    # A_log: (nheads,). Init to log of values in (1, 16) like the reference impl.
    A_log = torch.log(torch.empty(nheads).uniform_(1.0, 16.0))

    # D: (nheads,). Init to ones.
    D = torch.ones(nheads)

    # norm.weight (RMSNormGated): (d_inner,). Init to ones.
    norm_w = torch.ones(d_inner)

    # out_proj: Linear(d_inner, d_model, bias=False).
    # Initialize from attention's o_proj by tiling columns (hidden_size → d_inner)
    o_proj_w = attention_sd.get("o_proj.weight")
    if o_proj_w is None or not isinstance(o_proj_w, torch.Tensor):
        out_proj_w = torch.empty(hidden_size, d_inner)
        torch.nn.init.xavier_uniform_(out_proj_w)
    elif o_proj_w.shape == (hidden_size, hidden_size):
        # Repeat each column `expand` times along dim 1, scale by 1/sqrt(expand)
        out_proj_w = o_proj_w.repeat(1, _EXPAND).contiguous() / math.sqrt(_EXPAND)
    elif o_proj_w.shape == (hidden_size, d_inner):
        out_proj_w = o_proj_w.contiguous()
    else:
        out_proj_w = torch.empty(hidden_size, d_inner)
        torch.nn.init.xavier_uniform_(out_proj_w)

    return {
        "in_proj.weight": in_proj_w.contiguous(),
        "conv1d.weight": conv1d_w.contiguous(),
        "conv1d.bias": conv1d_b.contiguous(),
        "dt_bias": dt_bias.contiguous(),
        "A_log": A_log.contiguous(),
        "D": D.contiguous(),
        "norm.weight": norm_w.contiguous(),
        "out_proj.weight": out_proj_w,
    }

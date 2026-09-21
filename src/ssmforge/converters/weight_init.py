"""Initialize Mamba2 weights from attention weights.

Approach (MambaInLlama recipe):
- Reuse attention's o_proj → Mamba2's out_proj
- Initialize in_proj, conv1d, x_proj, dt_bias, A_log, D with small random values
"""

from __future__ import annotations

import math

import torch


def init_mamba2_from_attention(attention_sd: dict, hidden_size: int) -> dict:
    """Convert attention weights to Mamba2 initial weights.

    Args:
        attention_sd: attention submodule state dict (q_proj, k_proj, v_proj, o_proj)
        hidden_size: model hidden dimension

    Returns:
        Mamba2 submodule state dict (in_proj, conv1d, x_proj, dt_bias, A_log, D, out_proj)
    """
    out_proj_weight = attention_sd.get("o_proj.weight")
    if out_proj_weight is None:
        out_proj_weight = torch.empty(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(out_proj_weight)

    expand = 2
    d_inner = hidden_size * expand

    in_proj = torch.empty(d_inner * 2, hidden_size)
    torch.nn.init.xavier_uniform_(in_proj)

    conv1d_weight = torch.empty(d_inner, 1, 4)
    torch.nn.init.kaiming_uniform_(conv1d_weight, a=math.sqrt(5))

    x_proj_weight = torch.empty(d_inner // 2, d_inner)
    torch.nn.init.xavier_uniform_(x_proj_weight)

    dt_bias = torch.zeros(d_inner // 2)
    A_log = torch.log(torch.empty(d_inner // 2).uniform_(1.0, 16.0))
    D = torch.ones(d_inner)

    return {
        "in_proj.weight": in_proj,
        "conv1d.weight": conv1d_weight,
        "x_proj.weight": x_proj_weight,
        "dt_bias": dt_bias,
        "A_log": A_log,
        "D": D,
        "out_proj.weight": out_proj_weight,
    }

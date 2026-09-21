"""Tests that Mamba2 weights are correctly shaped for mamba_ssm.Mamba2
with default args. Reference shapes derived from:
https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba2.py
"""

import torch

from ssmforge.converters.weight_init import init_mamba2_from_attention

# Match Mamba2 defaults
_EXPAND = 2
_HEADDIM = 64
_D_STATE = 128
_NGROUPS = 1
_D_CONV = 4


def _expected_shapes(hidden_size: int) -> dict:
    d_inner = hidden_size * _EXPAND
    nheads = d_inner // _HEADDIM
    conv_dim = d_inner + 2 * _NGROUPS * _D_STATE
    in_proj_dim = 2 * d_inner + 2 * _NGROUPS * _D_STATE + nheads
    return {
        "in_proj.weight": (in_proj_dim, hidden_size),
        "conv1d.weight": (conv_dim, 1, _D_CONV),
        "conv1d.bias": (conv_dim,),
        "dt_bias": (nheads,),
        "A_log": (nheads,),
        "D": (nheads,),
        "norm.weight": (d_inner,),
        "out_proj.weight": (hidden_size, d_inner),
    }


def test_mamba2_weights_have_correct_shapes():
    """All keys in the returned dict must match mamba_ssm.Mamba2 state_dict shape."""
    hidden_size = 2048
    expected = _expected_shapes(hidden_size)
    attention_sd = {
        "q_proj.weight": torch.randn(hidden_size, hidden_size),
        "k_proj.weight": torch.randn(hidden_size, hidden_size),
        "v_proj.weight": torch.randn(hidden_size, hidden_size),
        "o_proj.weight": torch.randn(hidden_size, hidden_size),
    }
    sd = init_mamba2_from_attention(attention_sd, hidden_size=hidden_size)
    for key, want in expected.items():
        assert key in sd, f"missing key: {key}"
        assert sd[key].shape == want, f"{key}: got {tuple(sd[key].shape)}, want {want}"


def test_mamba2_weights_only_have_required_keys():
    """Returned dict must contain only the standard Mamba2 keys (no extras)."""
    hidden_size = 2048
    expected = set(_expected_shapes(hidden_size).keys())
    attention_sd = {"o_proj.weight": torch.randn(hidden_size, hidden_size)}
    sd = init_mamba2_from_attention(attention_sd, hidden_size=hidden_size)
    assert set(sd.keys()) == expected, f"got keys {set(sd.keys())}, want {expected}"


def test_mamba2_out_proj_shape_scales_with_hidden_size():
    """Shape contract holds across model sizes."""
    for hidden_size in [256, 512, 2048, 4096]:
        attention_sd = {"o_proj.weight": torch.randn(hidden_size, hidden_size)}
        sd = init_mamba2_from_attention(attention_sd, hidden_size=hidden_size)
        d_inner = hidden_size * _EXPAND
        assert sd["out_proj.weight"].shape == (hidden_size, d_inner)


def test_mamba2_handles_missing_o_proj():
    """No o_proj → fresh xavier init with correct shape."""
    attention_sd = {}
    sd = init_mamba2_from_attention(attention_sd, hidden_size=2048)
    assert sd["out_proj.weight"].shape == (2048, 4096)


def test_mamba2_handles_string_o_proj_sentinel():
    """Test sentinels (string) shouldn't crash; treat as fresh init."""
    attention_sd = {"o_proj.weight": "sentinel"}  # matches placeholder style
    sd = init_mamba2_from_attention(attention_sd, hidden_size=2048)
    assert sd["out_proj.weight"].shape == (2048, 4096)


def test_mamba2_initialized_parameters_are_finite():
    """All returned tensors must be finite (no NaN or Inf from init)."""
    attention_sd = {"o_proj.weight": torch.randn(2048, 2048)}
    sd = init_mamba2_from_attention(attention_sd, hidden_size=2048)
    for key, t in sd.items():
        assert torch.isfinite(t).all(), f"{key} has NaN or Inf"

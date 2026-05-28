"""Spec 13 §108 path-equivalence tests for Conv2d INT16 fixed-point.

ImageNet/MobileNet-V2 layers like ``features.18.0`` (1x1 conv, ``C_in=320``,
``C_out=1280``) showed INT16 vs FIXED_SCALE_QDQ cosine ~= 0.10. Layer-by-layer
diagnosis isolated the gap to the *boundary* quantization grid, not the
``Conv2dInt16Kernel`` math itself. These tests pin both sides:

1. :func:`test_conv2d_int16_kernel_matches_float_reference` -- INT16 kernel vs
   an integer-equivalent ``F.conv2d`` float reference (kernel math sanity).
2. :func:`test_conv2d_int16_matches_full_fixed_scale_qdq_path` -- INT16 kernel
   vs the full FIXED_SCALE_QDQ path (QDQ inputs/weights via the same fixed
   scale grid, ``F.conv2d``, then quantize via the same ``(m, rshift)``).
   Together they prove that spec §108 (≥ 0.999 cosine) is *reachable* when
   both modes share the fixed-scale grid; the network-level cosine gap then
   maps directly to the choice of boundary grid in :mod:`boundary_quantize`.

All tests use S8 (matching the actual MobileNet-V2 sim) to avoid bit-pattern
surprises in the ``int16`` storage slot for unsigned grids.
"""

from __future__ import annotations

from typing import Tuple

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import aimet_torch.fixed_point.kernels  # noqa: F401
from aimet_torch.fixed_point import (
    Int16QuantizedTensor,
    OutputEncoding,
    get_fixed_kernel,
    quantize_multiplier,
)
from aimet_torch.fixed_point.encoding import FixedScaleEncoding
from aimet_torch.fixed_point.fixed_scale_qdq import (
    quantize_dequantize_from_float_encoding,
    quantize_with_fixed_scale,
)

# Match the actual MobileNet-V2 v2 sim default quantizer (S8 symmetric).
S8_QMIN, S8_QMAX = -128, 127


def _quantize_per_tensor_sym_s8(x: torch.Tensor) -> Tuple[Int16QuantizedTensor, float]:
    scale = max(abs(float(x.min().item())), abs(float(x.max().item()))) / S8_QMAX
    if scale == 0:
        scale = 1.0
    q = torch.round(x / scale).clamp(S8_QMIN, S8_QMAX).to(torch.int16)
    return (
        Int16QuantizedTensor(
            int_repr=q,
            scale=torch.tensor(scale, dtype=torch.float32),
            zero_point=torch.tensor(0, dtype=torch.int32),
            qmin=S8_QMIN,
            qmax=S8_QMAX,
        ),
        scale,
    )


def _quantize_per_channel_sym_s8(w: torch.Tensor) -> Tuple[Int16QuantizedTensor, torch.Tensor]:
    abs_max = w.abs().amax(dim=tuple(range(1, w.dim())))
    abs_max = torch.where(abs_max == 0, torch.ones_like(abs_max), abs_max)
    scale = abs_max / S8_QMAX
    scale_b = scale.view(-1, *([1] * (w.dim() - 1)))
    q = torch.round(w / scale_b).clamp(S8_QMIN, S8_QMAX).to(torch.int16)
    return (
        Int16QuantizedTensor(
            int_repr=q,
            scale=scale.to(torch.float32),
            zero_point=torch.zeros_like(scale, dtype=torch.int32),
            qmin=S8_QMIN,
            qmax=S8_QMAX,
            axis=0,
        ),
        scale,
    )


def _build_layer_output_encoding(
    y_scale: float,
    real_m_per_channel: torch.Tensor,
) -> OutputEncoding:
    """``OutputEncoding`` mirroring v2 adapter (per-channel ``(M, rshift)``)."""

    multiplier, rshift = quantize_multiplier(real_m_per_channel)
    out_channels = real_m_per_channel.numel()
    return OutputEncoding(
        scale=torch.tensor(y_scale, dtype=torch.float32),
        zero_point=torch.tensor(0, dtype=torch.int32),
        qmin=S8_QMIN,
        qmax=S8_QMAX,
        multiplier=multiplier.view(1, out_channels, 1, 1),
        rshift=rshift.view(1, out_channels, 1, 1),
        axis=None,
    )


def _per_channel_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    af = a.to(torch.float32).transpose(0, 1).reshape(a.shape[1], -1)
    bf = b.to(torch.float32).transpose(0, 1).reshape(b.shape[1], -1)
    return (af * bf).sum(dim=1) / (af.norm(dim=1) * bf.norm(dim=1)).clamp_min(1e-30)


@pytest.mark.parametrize(
    "n, c_in, c_out, h, kernel_size",
    [
        # Mirror features.18.0 (1x1 conv, 320 -> 1280, 7x7) at smaller scale.
        (1, 64, 128, 7, 1),
        # Mirror features.0.0 (3x3 stem, stride 2).
        (1, 3, 16, 32, 3),
    ],
)
def test_conv2d_int16_kernel_matches_float_reference(n, c_in, c_out, h, kernel_size):
    """The INT16 kernel must equal an integer-equivalent ``F.conv2d`` reference.

    Both paths share the same ``(M, rshift)``; divergence isolates a bug in the
    kernel's ``unfold + matmul + requant`` path.
    """

    from aimet_torch.fixed_point.requantize import requantize_int

    torch.manual_seed(0)
    x_float = torch.randn(n, c_in, h, h)
    w_float = torch.randn(c_out, c_in, kernel_size, kernel_size) * 0.1
    b_float = torch.randn(c_out) * 0.05

    x_int, x_scale = _quantize_per_tensor_sym_s8(x_float)
    w_int, w_scale = _quantize_per_channel_sym_s8(w_float)

    padding = kernel_size // 2
    y_ref_float = F.conv2d(x_float, w_float, bias=b_float, padding=padding)
    y_scale = max(abs(float(y_ref_float.min().item())), abs(float(y_ref_float.max().item()))) / S8_QMAX
    real_m = (x_scale * w_scale) / y_scale
    out_enc = _build_layer_output_encoding(y_scale, real_m)
    bias_int32 = torch.round(b_float / (x_scale * w_scale)).to(torch.int32)

    kernel = get_fixed_kernel(nn.Conv2d)
    kernel_out = kernel(
        [x_int],
        {"weight": w_int, "bias": bias_int32},
        out_enc,
        {"stride": 1, "padding": padding, "dilation": 1, "groups": 1},
    )

    x_centered = x_int.int_repr.to(torch.float32)
    w_centered = w_int.int_repr.to(torch.float32)
    acc = F.conv2d(x_centered, w_centered, bias=None, stride=1, padding=padding).to(torch.int32)
    acc = acc + bias_int32.view(1, -1, 1, 1).to(torch.int32)
    ref_int_repr = requantize_int(
        acc,
        out_enc.multiplier,
        out_enc.rshift,
        out_enc.zero_point,
        out_enc.qmin,
        out_enc.qmax,
    ).to(torch.int16)

    per_ch_cos = _per_channel_cosine(kernel_out.int_repr, ref_int_repr)
    worst_cos = float(per_ch_cos.min().item())
    max_abs_err = int(
        (kernel_out.int_repr.to(torch.int32) - ref_int_repr.to(torch.int32)).abs().max().item()
    )
    assert worst_cos >= 0.999, (
        f"Conv2dInt16Kernel math diverges from float reference: cos={worst_cos:.6f}, "
        f"max_abs_err={max_abs_err}."
    )
    assert max_abs_err <= 1, f"Max int-LSB error = {max_abs_err}; expected <= 1."


@pytest.mark.parametrize(
    "n, c_in, c_out, h",
    [
        (1, 64, 128, 7),
        (1, 320, 256, 7),  # close to features.18.0 fan-in
    ],
)
def test_conv2d_int16_matches_full_fixed_scale_qdq_path(n, c_in, c_out, h):
    """spec 13 §108 reachability proof.

    Both modes use the **same** fixed-scale grid for input/weight Q/DQ and the
    same per-channel output multiplier. INT16 kernel and FIXED_SCALE_QDQ
    (QDQ -> F.conv2d -> quantize_with_fixed_scale) must then agree to within
    1 int-LSB at 0.999 cosine -- proving the spec floor is reachable in
    isolation. If the network-level cosine drops below this, the gap is owned
    by the choice of boundary grid (``boundary_quantize``), not by the kernel.
    """

    torch.manual_seed(42)
    x_float = torch.randn(n, c_in, h, h)
    w_float = torch.randn(c_out, c_in, 1, 1) * 0.1
    b_float = torch.randn(c_out) * 0.05

    x_int, x_scale = _quantize_per_tensor_sym_s8(x_float)
    w_int, w_scale = _quantize_per_channel_sym_s8(w_float)

    y_ref_float = F.conv2d(x_float, w_float, bias=b_float)
    y_scale = max(abs(float(y_ref_float.min().item())), abs(float(y_ref_float.max().item()))) / S8_QMAX
    real_m = (x_scale * w_scale) / y_scale
    out_enc = _build_layer_output_encoding(y_scale, real_m)
    bias_int32 = torch.round(b_float / (x_scale * w_scale)).to(torch.int32)

    kernel = get_fixed_kernel(nn.Conv2d)
    kernel_out = kernel(
        [x_int],
        {"weight": w_int, "bias": bias_int32},
        out_enc,
        {"stride": 1, "padding": 0, "dilation": 1, "groups": 1},
    )

    # FIXED_SCALE_QDQ path: Q/DQ via the same fixed-scale grid, then quantize y.
    x_qdq = quantize_dequantize_from_float_encoding(
        x_float,
        torch.tensor(x_scale, dtype=torch.float32),
        torch.tensor(0.0),
        S8_QMIN,
        S8_QMAX,
    ).float()
    w_qdq = quantize_dequantize_from_float_encoding(
        w_float,
        w_scale.view(c_out, 1, 1, 1).float(),
        torch.zeros_like(w_scale).view(c_out, 1, 1, 1),
        S8_QMIN,
        S8_QMAX,
    ).float()
    y_floatconv = F.conv2d(x_qdq, w_qdq, bias=b_float)
    My, Ry = quantize_multiplier(torch.tensor(y_scale))
    fixed_y = FixedScaleEncoding(
        m_int16=My,
        rshift=Ry,
        zero_point=torch.tensor(0, dtype=torch.int32),
        qmin=S8_QMIN,
        qmax=S8_QMAX,
        axis=None,
    )
    fs_int = quantize_with_fixed_scale(y_floatconv, fixed_y).to(torch.int32)

    per_ch_cos = _per_channel_cosine(kernel_out.int_repr, fs_int)
    worst_cos = float(per_ch_cos.min().item())
    max_abs_err = int(
        (kernel_out.int_repr.to(torch.int32) - fs_int).abs().max().item()
    )
    assert worst_cos >= 0.999, (
        f"INT16 kernel vs full FIXED_SCALE_QDQ path diverge: cos={worst_cos:.6f}, "
        f"max_abs_err={max_abs_err}. Spec 13 §108 cannot be reached without "
        f"aligning the boundary grid; see boundary_quantize.should_use_fixed_scale_boundary."
    )
    assert max_abs_err <= 2, f"Max int-LSB error = {max_abs_err}; expected <= 2."

# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
"""Q/DQ with encoding scale ``m_int16 / 2**rshift`` (``fixed_scale_qdq`` execution mode)."""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from aimet_torch.fixed_point.encoding import FixedScaleEncoding
from aimet_torch.fixed_point.offline.scale_fixed import fixed_scale_encoding_from_tensors
from aimet_torch.fixed_point.rounding import RoundingMode


def _offset_tensor(
    encoding: FixedScaleEncoding,
    *,
    device,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return ``-encoding.zero_point`` on ``device`` with high-precision default.

    ``dtype`` defaults to ``torch.float64`` so integer zero_point values are
    preserved exactly even when the surrounding tensor is bf16/fp16 (the 7-bit
    mantissa silently rounds zp outside ``[-2**8, 2**8]``). Callers may pass a
    different dtype; downstream autograd will auto-cast grads back to leaf dtype.
    """
    return (-encoding.zero_point).to(device=device, dtype=dtype)


def _round_tensor(tensor: torch.Tensor, rounding_mode: RoundingMode) -> torch.Tensor:
    if rounding_mode == RoundingMode.TRUNCATE:
        return torch.trunc(tensor)
    if rounding_mode == RoundingMode.HALF_AWAY_FROM_ZERO:
        # ``torch.round`` is banker's rounding (half-to-even); implement
        # half-away-from-zero explicitly to honor the documented spec.
        return torch.sign(tensor) * torch.floor(torch.abs(tensor) + 0.5)
    return torch.round(tensor)


def _broadcast_fixed_scale(
    tensor: torch.Tensor,
    m_int16: torch.Tensor,
    rshift: torch.Tensor,
    offset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    m = m_int16.to(device=tensor.device)
    r = rshift.to(device=tensor.device)
    off = offset.to(device=tensor.device)
    return torch.broadcast_tensors(tensor, m, r, off)


def _compute_q_pre_clamp(
    tensor: torch.Tensor,
    m_int16: torch.Tensor,
    rshift: torch.Tensor,
    offset_f64: torch.Tensor,
    rounding_mode: RoundingMode,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Integer-grid candidate ``q = round(x * 2**r / m) - off`` in float64, before clamp.

    Positions with ``m == 0`` collapse to ``off`` so the dequant result is 0 and the
    intermediate division never produces NaN/Inf.

    Returns ``(q_pre, m_b_f64, scale_pow_f64, off_b_f64, zero_m_mask)`` — all
    broadcasted to ``tensor.shape`` in float64. Callers needing only ``q_pre``
    can unpack with ``q, *_ = _compute_q_pre_clamp(...)``; the Q+D autograd
    forward reuses the rest to avoid re-broadcasting / recomputing ``2**r``.
    """
    x, m, r, off = _broadcast_fixed_scale(tensor, m_int16, rshift, offset_f64)
    x64 = x.to(torch.float64)
    m64 = m.to(torch.float64)
    r64 = r.to(torch.float64)
    off64 = off.to(torch.float64)
    zero_m = m64 == 0
    safe_m = torch.where(zero_m, torch.ones_like(m64), m64)
    scale_pow = torch.pow(2.0, r64)
    q = _round_tensor(x64 * scale_pow / safe_m, rounding_mode) - off64
    q_pre = torch.where(zero_m, off64, q)
    return q_pre, m64, scale_pow, off64, zero_m


def _sum_to_shape(grad: torch.Tensor, target_shape: Sequence[int]) -> torch.Tensor:
    """Reduce ``grad`` back to ``target_shape`` via summation along broadcast dims."""
    if tuple(grad.shape) == tuple(target_shape):
        return grad
    while grad.dim() > len(target_shape):
        grad = grad.sum(dim=0)
    for i, size in enumerate(target_shape):
        if size == 1 and grad.shape[i] != 1:
            grad = grad.sum(dim=i, keepdim=True)
    return grad


def quantize_with_fixed_scale(
    tensor: torch.Tensor,
    encoding: FixedScaleEncoding,
    rounding_mode: RoundingMode = RoundingMode.HALF_TO_EVEN,
) -> torch.Tensor:
    """Quantize float tensor using ``m_int16`` and ``rshift`` (no float scale in multiply path)."""

    offset = _offset_tensor(encoding, device=tensor.device, dtype=torch.float64)
    q, *_ = _compute_q_pre_clamp(
        tensor, encoding.m_int16, encoding.rshift, offset, rounding_mode
    )
    return q.clamp(encoding.qmin, encoding.qmax)


def dequantize_with_fixed_scale(
    tensor: torch.Tensor,
    encoding: FixedScaleEncoding,
) -> torch.Tensor:
    """Dequantize integer-grid tensor using ``m_int16`` and ``rshift``."""

    offset = _offset_tensor(encoding, device=tensor.device, dtype=torch.float64)
    q, m, r, off = _broadcast_fixed_scale(
        tensor, encoding.m_int16, encoding.rshift, offset
    )
    q64 = q.to(torch.float64)
    m64 = m.to(torch.float64)
    r64 = r.to(torch.float64)
    off64 = off.to(torch.float64)
    scale_pow = torch.pow(2.0, r64)
    zero_m = m64 == 0
    out = (q64 + off64) * m64 / scale_pow
    return torch.where(zero_m, torch.zeros_like(out), out).to(tensor.dtype)


class _FixedScaleQuantDequantFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        m_int16: torch.Tensor,
        rshift: torch.Tensor,
        offset: torch.Tensor,
        qmin: int,
        qmax: int,
        rounding_mode: RoundingMode,
    ):
        # ``offset`` follows the ``-zero_point`` convention. Keep it continuous so a
        # learnable offset retains precision; rounding to int32 zp would discretize
        # the forward while leaving the backward formula continuous (mismatch).
        off64 = offset.to(torch.float64)
        q_pre, m64, scale_pow, off64_b, zero_m = _compute_q_pre_clamp(
            tensor, m_int16, rshift, off64, rounding_mode
        )

        need_mask = tensor.requires_grad or offset.requires_grad
        mask = (q_pre >= qmin) & (q_pre <= qmax) if need_mask else None

        q = q_pre.clamp(qmin, qmax)
        out = (q + off64_b) * m64 / scale_pow
        out = torch.where(zero_m, torch.zeros_like(out), out).to(tensor.dtype)

        ctx.qmin = qmin
        ctx.qmax = qmax
        ctx.tensor_requires_grad = tensor.requires_grad
        ctx.offset_requires_grad = offset.requires_grad
        ctx.tensor_shape = tuple(tensor.shape)
        ctx.offset_shape = tuple(offset.shape)
        ctx.save_for_backward(m_int16, rshift, mask)
        return out

    @staticmethod
    def backward(ctx, grad):
        m_int16, rshift, mask = ctx.saved_tensors
        tensor_grad = None
        offset_grad = None

        if mask is not None:
            if ctx.tensor_requires_grad:
                g = grad * mask
                tensor_grad = _sum_to_shape(g, ctx.tensor_shape)
            if ctx.offset_requires_grad:
                m64 = m_int16.to(torch.float64)
                r64 = rshift.to(torch.float64)
                eff_scale = (m64 / torch.pow(2.0, r64)).to(grad.dtype)
                g = grad * (~mask) * eff_scale
                offset_grad = _sum_to_shape(g, ctx.offset_shape)

        return tensor_grad, None, None, offset_grad, None, None, None


def quantize_dequantize_with_fixed_scale(
    tensor: torch.Tensor,
    encoding: FixedScaleEncoding,
    rounding_mode: RoundingMode = RoundingMode.HALF_TO_EVEN,
) -> torch.Tensor:
    """Differentiable Q/DQ with fixed ``(m_int16, rshift)`` scale (STE on tensor)."""

    # ``encoding.zero_point`` is a frozen integer tensor with no grad, so the
    # offset derived from it is non-learnable on this entry point. Use float64
    # to preserve large integer zp exactly regardless of ``tensor.dtype``.
    offset = _offset_tensor(encoding, device=tensor.device).detach()
    return _FixedScaleQuantDequantFunc.apply(
        tensor,
        encoding.m_int16.detach(),
        encoding.rshift.detach(),
        offset,
        encoding.qmin,
        encoding.qmax,
        rounding_mode,
    )


def quantize_dequantize_from_float_encoding(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor,
    qmin: int,
    qmax: int,
    *,
    m_int16: Optional[torch.Tensor] = None,
    rshift: Optional[torch.Tensor] = None,
    rounding_mode: RoundingMode = RoundingMode.HALF_TO_EVEN,
) -> torch.Tensor:
    """Build :class:`FixedScaleEncoding` from float ``scale``/``offset`` and run Q/DQ.

    ``offset`` follows the v2 affine ``-zero_point`` convention (same as
    :class:`aimet_torch.v2.quantization.affine.encoding.AffineEncoding.offset`):
    the integer-grid value is ``round(x / scale) - offset``, equivalent to
    ``round(x / scale) + zero_point``. Passing ``+zero_point`` here flips the
    quantization grid and yields silently wrong results.
    """

    encoding = fixed_scale_encoding_from_tensors(
        scale=scale,
        offset=offset,
        qmin=qmin,
        qmax=qmax,
        m_int16=m_int16,
        rshift=rshift,
    )
    # Keep caller's offset dtype (typically fp32) instead of downcasting to
    # tensor.dtype; autograd will cast backward grad back to the leaf's dtype.
    # This avoids silent zp precision loss when ``tensor`` is bf16/fp16.
    offset_live = offset.to(device=tensor.device)
    return _FixedScaleQuantDequantFunc.apply(
        tensor,
        encoding.m_int16.detach(),
        encoding.rshift.detach(),
        offset_live,
        qmin,
        qmax,
        rounding_mode,
    )

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
"""Offline bias quantization for INT16 fixed-point kernels."""

from typing import Optional

import torch

from aimet_torch.fixed_point.requantize import INT32_QMAX, INT32_QMIN

INT16_QMIN = -32768
INT16_QMAX = 32767


def _align_acc_scale_to_bias(
    acc_scale: torch.Tensor, bias_float: torch.Tensor
) -> torch.Tensor:
    """Reshape ``acc_scale`` so element-wise division with ``bias_float`` is well-defined.

    Real per-channel weight encodings carry trailing reduction dims
    (``(out_channels, 1)`` for Linear, ``(out_channels, 1, 1, 1)`` for Conv),
    while ``bias_float`` is 1-D ``(out_channels,)``. Naïve broadcasting then
    produces a 2-D / 4-D bias tensor instead of the intended per-channel
    division. This helper folds ``acc_scale`` to the bias shape exactly when
    element counts match, accepts a true scalar as-is, and refuses any other
    shape so silent broadcasting bugs surface early.
    """

    if acc_scale.numel() == 1:
        return acc_scale.reshape(())
    if acc_scale.numel() == bias_float.numel():
        return acc_scale.reshape(bias_float.shape)
    raise ValueError(
        "acc_scale shape "
        f"{tuple(acc_scale.shape)} is not broadcastable per-channel onto "
        f"bias shape {tuple(bias_float.shape)}; expected scalar or "
        f"{bias_float.numel()} elements."
    )


def quantize_bias_int(
    bias_float: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    *,
    bits: int = 32,
    saturate: Optional[bool] = None,
) -> torch.Tensor:
    """Quantize a floating bias against the accumulator scale ``S_x * S_w``.

    This implementation stores bias at the accumulator scale by folding any
    bias-scale ratio offline into the integer bias term used by the hardware
    accumulator. Spec 04_01 derives the requantize identity in terms of a
    distinct ``S_b``; we choose to absorb ``S_b`` into the offline bias so
    the runtime path needs no separate bias scale, but this is an
    implementation choice rather than a spec-mandated equation.

    ``bits`` selects the storage container:

    * ``bits=32`` → ``torch.int32`` (legacy/simulator default; matches the
      MAC accumulator width and never overflows on realistic models).
      Default ``saturate`` is ``True`` to keep INT16 QAT stable when float
      bias drifts under SGD while encodings stay frozen.
    * ``bits=16`` → ``torch.int16`` (spec 04_01 hardware canonical bias).
      Default ``saturate`` is ``False``: the int16 range is small and silent
      clamping would change model semantics under explicit_config without
      surfacing the misconfiguration. Pass ``saturate=True`` only when the
      caller has accepted that out-of-range bias entries are clipped.

    ``acc_scale = S_x * S_w`` is reshaped to match ``bias_float`` exactly,
    so per-channel weight encodings (``(out_channels, 1[, 1, 1])``) divide
    a 1-D bias correctly instead of broadcasting into an extra trailing dim.
    """

    if bits not in (16, 32):
        raise ValueError(f"bits must be 16 or 32; got {bits}.")
    if not bias_float.is_floating_point():
        raise TypeError(f"bias_float must be floating point; got {bias_float.dtype}.")
    if not torch.all(torch.isfinite(bias_float)):
        raise ValueError("bias_float must be finite (no NaN/Inf entries).")
    if not (
        torch.all(torch.isfinite(x_scale))
        and torch.all(torch.isfinite(w_scale))
    ):
        raise ValueError("x_scale and w_scale must be finite (no NaN/Inf).")
    if torch.any(x_scale <= 0) or torch.any(w_scale <= 0):
        raise ValueError("x_scale and w_scale must be strictly positive.")

    if saturate is None:
        saturate = bits == 32

    acc_scale = x_scale.to(torch.float64) * w_scale.to(torch.float64)
    acc_scale = _align_acc_scale_to_bias(acc_scale, bias_float)
    bias_int64 = torch.round(bias_float.to(torch.float64) / acc_scale).to(torch.int64)

    if bits == 32:
        qmin, qmax, dtype = INT32_QMIN, INT32_QMAX, torch.int32
    else:
        qmin, qmax, dtype = INT16_QMIN, INT16_QMAX, torch.int16

    if saturate:
        bias_int64 = bias_int64.clamp(qmin, qmax)
    elif torch.any(bias_int64 < qmin) or torch.any(bias_int64 > qmax):
        raise ValueError(f"Quantized bias exceeds int{bits} range.")

    return bias_int64.to(dtype)


def quantize_bias_int32(
    bias_float: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    *,
    saturate: bool = True,
) -> torch.Tensor:
    """Backward-compat alias for ``quantize_bias_int(..., bits=32)``."""

    return quantize_bias_int(
        bias_float, x_scale, w_scale, bits=32, saturate=saturate
    )

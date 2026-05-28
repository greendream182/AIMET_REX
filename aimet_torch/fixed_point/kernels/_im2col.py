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
"""Pure-integer ``im2col`` helper for fixed-point Conv / AvgPool kernels.

PyTorch's ``F.unfold`` lacks integer CPU/CUDA kernels on most builds. The G3
fixed-point path forbids intermediate float tensors (ADR-002), so kernels
that need an im2col-style rearrangement use :func:`im2col_int` instead. The
implementation is a pure-integer composition of ``F.pad`` and
``Tensor.unfold``, which both support integer dtypes.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def im2col_int(
    x: torch.Tensor,
    kernel_size: Tuple[int, int],
    *,
    dilation: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    stride: Tuple[int, int] = (1, 1),
    pad_value: int = 0,
) -> torch.Tensor:
    """Pure-integer ``im2col`` matching ``F.unfold`` semantics.

    Args:
        x: 4-D integer tensor ``(N, C, H, W)``.
        kernel_size: ``(Kh, Kw)``.
        dilation: ``(Dh, Dw)``.
        padding: ``(Ph, Pw)``.
        stride: ``(Sh, Sw)``.
        pad_value: scalar pad value (defaults to ``0``; callers wanting
            zero-point-aware padding can pre-add zp before calling).

    Returns:
        ``(N, C * Kh * Kw, L)`` with ``L = oH * oW`` patches per batch, exactly
        matching ``F.unfold`` ordering so downstream ``matmul`` patterns work.
    """

    if x.dim() != 4:
        raise ValueError(f"im2col_int expects (N,C,H,W); got {tuple(x.shape)}.")
    kh, kw = int(kernel_size[0]), int(kernel_size[1])
    sh, sw = int(stride[0]), int(stride[1])
    dh, dw = int(dilation[0]), int(dilation[1])
    ph, pw = int(padding[0]), int(padding[1])

    # F.pad order is (left, right, top, bottom) for last two dims.
    if ph or pw:
        x = F.pad(x, [pw, pw, ph, ph], value=pad_value)

    # Effective kernel extent after dilation.
    eff_h = (kh - 1) * dh + 1
    eff_w = (kw - 1) * dw + 1

    n, c, padded_h, padded_w = x.shape
    out_h = (padded_h - eff_h) // sh + 1
    out_w = (padded_w - eff_w) // sw + 1

    # ``unfold(dim, size, step)`` materializes overlapping windows. For dilation
    # we unfold with the dilated extent first, then slice every dilation-th
    # element to recover the dilated tap positions.
    windows = x.unfold(-2, eff_h, sh).unfold(-2, eff_w, sw)
    # windows shape: (N, C, out_h, out_w, eff_h, eff_w)
    if dh != 1:
        windows = windows[..., ::dh, :]
    if dw != 1:
        windows = windows[..., :, ::dw]
    # Now: (N, C, out_h, out_w, kh, kw)
    # Match F.unfold's column order: each column is C*kh*kw values for one
    # patch; columns iterate over out_h * out_w in row-major.
    windows = windows.permute(0, 1, 4, 5, 2, 3).contiguous()
    # (N, C, kh, kw, out_h, out_w) -> reshape to (N, C*kh*kw, out_h*out_w)
    return windows.view(n, c * kh * kw, out_h * out_w)

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
"""STE utilities for INT16 fixed-point QAT simulation."""

import torch

from aimet_torch.fixed_point.requantize import INT16_QMAX, INT16_QMIN


class FakeQuantInt16STE(torch.autograd.Function):
    """Fake-quantize to INT16 grid with straight-through estimator backward."""

    @staticmethod
    def forward(ctx, tensor, scale, zero_point, qmin=INT16_QMIN, qmax=INT16_QMAX):
        if not tensor.is_floating_point():
            raise TypeError(f"tensor must be floating point; got {tensor.dtype}.")
        if torch.any(scale == 0):
            raise ValueError("scale must be non-zero.")

        scale = scale.to(device=tensor.device, dtype=tensor.dtype)
        zero_point = zero_point.to(device=tensor.device, dtype=tensor.dtype)
        q = torch.round(tensor / scale + zero_point)
        keep_mask = (qmin <= q) & (q <= qmax)
        q = torch.clamp(q, qmin, qmax)
        output = (q - zero_point) * scale
        ctx.save_for_backward(keep_mask)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (keep_mask,) = ctx.saved_tensors
        grad_tensor = grad_output * keep_mask.to(dtype=grad_output.dtype)
        return grad_tensor, None, None, None, None


def fake_quantize_int16_qat(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    qmin: int = INT16_QMIN,
    qmax: int = INT16_QMAX,
) -> torch.Tensor:
    """Apply INT16 fake quantization with STE backward."""

    return FakeQuantInt16STE.apply(tensor, scale, zero_point, qmin, qmax)

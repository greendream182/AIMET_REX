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

import torch

from aimet_torch.fixed_point.requantize import INT32_QMAX, INT32_QMIN


def quantize_bias_int32(
    bias_float: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
) -> torch.Tensor:
    """Quantize floating bias to int32 accumulator scale."""

    if not bias_float.is_floating_point():
        raise TypeError(f"bias_float must be floating point; got {bias_float.dtype}.")
    if torch.any(x_scale == 0) or torch.any(w_scale == 0):
        raise ValueError("x_scale and w_scale must be non-zero.")

    acc_scale = x_scale.to(torch.float64) * w_scale.to(torch.float64)
    bias_int64 = torch.round(bias_float.to(torch.float64) / acc_scale).to(torch.int64)

    if torch.any(bias_int64 < INT32_QMIN) or torch.any(bias_int64 > INT32_QMAX):
        raise ValueError("Quantized bias exceeds int32 range.")

    return bias_int64.to(torch.int32)

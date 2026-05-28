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
"""INT16 fixed-point Softmax (stable max-subtract + PWL exp + integer normalize)."""

from __future__ import annotations

import torch

from aimet_torch.fixed_point.encoding import InputEncoding, OutputEncoding
from aimet_torch.fixed_point.metrics.thresholds import PWL_HARDWARE_NUM_SEGMENTS
from aimet_torch.fixed_point.offline.lut_gen import generate_pwl_lut
from aimet_torch.fixed_point.requantize import SIM_TENSOR_DTYPE, saturate_sim_tensor
from aimet_torch.fixed_point.tensor import Int16QuantizedTensor

# Fixed-point fraction bits for softmax probability before mapping to output q-grid.
_SOFTMAX_PROB_FRAC_BITS = 15


def _integer_softmax_normalize(
    exp_q: torch.Tensor,
    exp_sum: torch.Tensor,
    *,
    qmin: int,
    qmax: int,
    zero_point: torch.Tensor,
) -> torch.Tensor:
    """Map nonnegative integer ``exp_q`` to output q-grid via ``exp_q / sum(exp_q)`` in integer."""
    denom = exp_sum.to(torch.int64).clamp(min=1)
    numer = exp_q.to(torch.int64) << _SOFTMAX_PROB_FRAC_BITS
    prob_fixed = (numer + denom // 2) // denom

    zp = int(zero_point.reshape(-1)[0].item())
    span = int(qmax - qmin)
    if span <= 0:
        raise ValueError("output qmax must exceed qmin for softmax.")

    q = (prob_fixed * span) >> _SOFTMAX_PROB_FRAC_BITS
    q = q + zp
    return saturate_sim_tensor(q, qmin, qmax)


def softmax_int16_pwl(
    x: Int16QuantizedTensor,
    *,
    dim: int,
    output_encoding: OutputEncoding,
) -> Int16QuantizedTensor:
    """Stable softmax on the INT16 grid using PWL ``exp`` and integer normalization."""
    if x.int_repr.dim() == 0:
        raise ValueError("softmax_int16_pwl expects at least 1-D input.")

    dim = int(dim)
    if dim < 0:
        dim = x.int_repr.dim() + dim

    device = x.int_repr.device
    scale = x.scale.to(device=device, dtype=torch.float32).reshape(-1)[0]
    zp = int(x.zero_point.reshape(-1)[0].item())

    centered = x.int_repr.to(torch.int32) - zp
    max_centered = centered.amax(dim=dim, keepdim=True)
    shifted = centered - max_centered

    shift_min = int(shifted.min().item())
    in_enc = InputEncoding(
        scale=scale.reshape(1),
        zero_point=torch.zeros(1, device=device, dtype=torch.int32),
        qmin=max(shift_min, -32768),
        qmax=0,
    )
    exp_scale = torch.tensor(1.0 / max(int(output_encoding.qmax), 1), device=device)
    exp_enc = OutputEncoding(
        scale=exp_scale.reshape(1),
        zero_point=torch.zeros(1, device=device, dtype=torch.int32),
        qmin=0,
        qmax=int(output_encoding.qmax),
    )
    pwl_lut = generate_pwl_lut(
        torch.exp,
        in_enc,
        exp_enc,
        num_segments=PWL_HARDWARE_NUM_SEGMENTS,
    )

    from aimet_torch.fixed_point.kernels.lut import evaluate_pwl_lut_int16

    shifted_q = torch.clamp(shifted, in_enc.qmin, in_enc.qmax).to(SIM_TENSOR_DTYPE)
    exp_q = evaluate_pwl_lut_int16(shifted_q, pwl_lut).to(torch.int32)
    exp_sum = exp_q.sum(dim=dim, keepdim=True)

    q = _integer_softmax_normalize(
        exp_q,
        exp_sum,
        qmin=int(output_encoding.qmin),
        qmax=int(output_encoding.qmax),
        zero_point=output_encoding.zero_point.to(device=device, dtype=torch.int32),
    )

    return Int16QuantizedTensor(
        int_repr=q,
        scale=output_encoding.scale.to(device=device),
        zero_point=output_encoding.zero_point.to(device=device, dtype=torch.int32),
        qmin=output_encoding.qmin,
        qmax=output_encoding.qmax,
        axis=output_encoding.axis,
    )

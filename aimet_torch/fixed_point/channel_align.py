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
"""Per-channel activation grid alignment for weighted INT16 ops (Conv / Linear).

Conv MAC + single output ``M/rshift`` assumes one input scale. When activations
use per-channel ``axis=1`` grids (MRNN ``freq_downs`` sparse spectrum), each
input channel must be requantized to a unified scale before the standard
``im2col → int32 MAC → requantize`` path (spec 04_01 per-tensor input contract).
"""

from __future__ import annotations

from typing import Optional

import torch

from aimet_torch.fixed_point.offline.multiplier import quantize_multiplier
from aimet_torch.fixed_point.requantize import (
    hw_ref_mode_enabled,
    requantize_int,
    saturate_sim_tensor,
)
from aimet_torch.fixed_point.rounding import RoundingMode
from aimet_torch.fixed_point.tensor import Int16QuantizedTensor, align_stat_rank


def is_per_channel_activation(tensor: Int16QuantizedTensor) -> bool:
    """Return True when ``tensor`` carries a per-channel scale on ``axis``."""

    return tensor.axis is not None and tensor.scale.numel() > 1


def _collapse_zero_point(tensor: Int16QuantizedTensor, device: torch.device) -> torch.Tensor:
    zp = tensor.zero_point.to(device=device, dtype=torch.int32)
    flat = zp.reshape(-1)
    if flat.numel() <= 1:
        return flat.reshape(()).clone() if flat.numel() == 1 else torch.zeros((), dtype=torch.int32, device=device)
    if torch.all(flat == flat[0]):
        return flat[0].reshape(()).clone()
    return torch.zeros((), dtype=torch.int32, device=device)


def align_per_channel_activation_for_conv_input(
    tensor: Int16QuantizedTensor,
    *,
    unified_scale: Optional[torch.Tensor] = None,
) -> Int16QuantizedTensor:
    """Align per-channel activation to a scalar grid for Conv/Linear MAC.

    Uses ``max(S_x[c])`` as the unified scale so outlier channels stay in range.
    Symmetric per-channel ``zero_point=0`` (MRNN path) is preserved at ``Z=0``.
    """

    if not is_per_channel_activation(tensor):
        return tensor

    device = tensor.int_repr.device
    scale = tensor.scale.to(device=device, dtype=torch.float32)
    scale_ch = scale.reshape(-1)
    if unified_scale is None:
        unified = scale_ch.max()
    else:
        unified = unified_scale.to(device=device, dtype=torch.float32).reshape(()).clone()

    unified_zp = _collapse_zero_point(tensor, device)

    if torch.allclose(scale_ch, unified.expand_as(scale_ch)):
        return Int16QuantizedTensor(
            int_repr=tensor.int_repr,
            scale=unified.clone(),
            zero_point=unified_zp,
            qmin=tensor.qmin,
            qmax=tensor.qmax,
            axis=None,
        )

    ratio = (scale / unified).detach()
    multiplier, rshift = quantize_multiplier(ratio)
    multiplier_b = align_stat_rank(
        multiplier.to(device=device, dtype=torch.uint16), tensor.int_repr
    )
    rshift_b = align_stat_rank(rshift.to(device=device, dtype=torch.int8), tensor.int_repr)
    rounding = (
        RoundingMode.HALF_UP if hw_ref_mode_enabled() else RoundingMode.HALF_TO_EVEN
    )

    centered_qmin = int(tensor.qmin) - int(unified_zp.item())
    centered_qmax = int(tensor.qmax) - int(unified_zp.item())
    aligned_centered = requantize_int(
        tensor.centered_int32(),
        multiplier_b,
        rshift_b,
        torch.zeros((), dtype=torch.int32, device=device),
        centered_qmin,
        centered_qmax,
        rounding_mode=rounding,
    )
    int_repr = saturate_sim_tensor(
        aligned_centered + unified_zp.to(torch.int64),
        tensor.qmin,
        tensor.qmax,
    )
    return Int16QuantizedTensor(
        int_repr=int_repr,
        scale=unified.clone(),
        zero_point=unified_zp,
        qmin=tensor.qmin,
        qmax=tensor.qmax,
        axis=None,
    )

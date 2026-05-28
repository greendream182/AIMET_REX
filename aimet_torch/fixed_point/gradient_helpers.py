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
"""Identity-gradient helpers for QuantGRU black-box INT16 boundaries (QAT protocol)."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Union

import torch

from aimet_torch.fixed_point.requantize import SIM_TENSOR_DTYPE, saturate_sim_tensor
from aimet_torch.fixed_point.tensor import Int16QuantizedTensor
from aimet_torch.v2.quantization.tensor import QuantizedTensorBase

__all__ = [
    "is_quantized_activation",
    "stop_grad_dequantize",
    "wrap_int_tensor_with_meta",
]


def is_quantized_activation(data: Any) -> bool:
    """Return True if ``data`` is an AIMET quantized activation carrier."""

    return isinstance(data, (Int16QuantizedTensor, QuantizedTensorBase))


def stop_grad_dequantize(
    data: Union[Int16QuantizedTensor, QuantizedTensorBase],
) -> torch.Tensor:
    """Dequantize to fp32 with identity backward (no boundary STE)."""

    if isinstance(data, Int16QuantizedTensor):
        fp = data.to_float(torch.float32)
    else:
        fp = data.dequantize()
    return fp + (fp - fp.detach())


def _q_bounds(bitwidth: int, is_symmetric: bool) -> tuple[int, int]:
    if is_symmetric:
        qmax = (1 << (bitwidth - 1)) - 1
        return -qmax, qmax
    return 0, (1 << bitwidth) - 1


def wrap_int_tensor_with_meta(
    int_tensor: torch.Tensor,
    meta: Mapping[str, Any],
) -> Int16QuantizedTensor:
    """Attach ``(scale, zp)`` metadata to an already-quantized integer tensor."""

    bitwidth = int(meta.get("bitwidth", 16))
    is_symmetric = bool(meta.get("is_symmetric", True))
    qmin, qmax = _q_bounds(bitwidth, is_symmetric)

    device = int_tensor.device
    scale = torch.as_tensor(meta["scale"], dtype=torch.float32, device=device)
    zero_point = torch.as_tensor(meta["zp"], dtype=torch.int32, device=device)

    int_repr = int_tensor.to(SIM_TENSOR_DTYPE)
    if int_repr.dtype != SIM_TENSOR_DTYPE:
        int_repr = int_repr.to(SIM_TENSOR_DTYPE)

    return Int16QuantizedTensor(
        int_repr=int_repr,
        scale=scale,
        zero_point=zero_point,
        qmin=qmin,
        qmax=qmax,
    )


def requantize_fp_to_int(
    fp_tensor: torch.Tensor,
    meta: Mapping[str, Any],
) -> torch.Tensor:
    """Round fp32 activations back to integer grid (stub ``forward_quantized`` helper)."""

    bitwidth = int(meta.get("bitwidth", 16))
    is_symmetric = bool(meta.get("is_symmetric", True))
    qmin, qmax = _q_bounds(bitwidth, is_symmetric)

    device = fp_tensor.device
    scale = torch.as_tensor(meta["scale"], dtype=torch.float32, device=device)
    zero_point = torch.as_tensor(meta["zp"], dtype=torch.int32, device=device)

    carrier = Int16QuantizedTensor.from_float(fp_tensor, scale=scale, zero_point=zero_point)
    return saturate_sim_tensor(carrier.int_repr, qmin, qmax)

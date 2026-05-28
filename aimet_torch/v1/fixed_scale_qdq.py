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
"""Thin v1 adapter: ``fixed_scale_qdq`` on StaticGrid / STE quantize-dequantize paths."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from aimet_common.defs import QuantizationDataType
from aimet_torch.fixed_point import ExecutionMode, get_quant_execution_mode
from aimet_torch.fixed_point.fixed_scale_qdq import quantize_dequantize_from_float_encoding
from aimet_torch.v2.quantization.affine.backends import _derive_qmin_qmax

if TYPE_CHECKING:
    from aimet_torch.v1.tensor_quantizer import TensorQuantizer


def is_v1_fixed_scale_qdq_mode() -> bool:
    return get_quant_execution_mode() == ExecutionMode.FIXED_SCALE_QDQ


def qmin_qmax_for_v1_quantizer(tensor_quantizer: "TensorQuantizer") -> Tuple[int, int]:
    """Match v1 int grid used by :func:`fixed_point_dispatch._v1_quantizer_to_affine_encoding`."""

    signed = bool(getattr(tensor_quantizer, "use_symmetric_encodings", True))
    return _derive_qmin_qmax(bitwidth=tensor_quantizer.bitwidth, signed=signed)


def _stitch_per_channel_scale_offset(
    ref_shape: torch.Size,
    enc_list: List,
    ch_axis: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if ref_shape[ch_axis] != len(enc_list):
        raise RuntimeError(
            "Per-channel encoding count does not match tensor shape on channel axis."
        )
    deltas = torch.tensor(
        [float(e.delta) for e in enc_list], dtype=torch.float32, device=device
    )
    offsets = torch.tensor(
        [float(e.offset) for e in enc_list], dtype=torch.float32, device=device
    )
    shape = [1] * len(ref_shape)
    shape[ch_axis] = len(enc_list)
    return deltas.reshape(shape), offsets.reshape(shape)


def scale_offset_from_static_grid_encoding(
    tensor: torch.Tensor,
    tensor_quantizer: "TensorQuantizer",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build broadcastable ``scale`` / ``offset`` from v1 TfEncoding(s)."""

    # pylint: disable=import-outside-toplevel
    from aimet_torch.v1.tensor_quantizer import (
        StaticGridPerChannelQuantizer,
        StaticGridPerTensorQuantizer,
    )

    enc = tensor_quantizer.encoding
    if enc is None:
        raise RuntimeError(
            "fixed_scale_qdq requires initialized v1 quantizer encodings; "
            "run compute_encodings and freeze first."
        )

    device = tensor.device
    if isinstance(tensor_quantizer, StaticGridPerChannelQuantizer):
        if not isinstance(enc, list) or not enc:
            raise RuntimeError("Per-channel quantizer requires a list of encodings.")
        return _stitch_per_channel_scale_offset(
            tensor.shape, enc, tensor_quantizer.channel_axis, device
        )
    if isinstance(tensor_quantizer, StaticGridPerTensorQuantizer):
        scale = torch.tensor(float(enc.delta), dtype=torch.float32, device=device)
        offset = torch.tensor(float(enc.offset), dtype=torch.float32, device=device)
        return scale, offset

    raise TypeError(
        f"fixed_scale_qdq v1 adapter supports StaticGrid quantizers only; "
        f"got {type(tensor_quantizer).__name__}."
    )


def try_fixed_scale_quantize_dequantize(
    tensor: torch.Tensor,
    tensor_quantizer: "TensorQuantizer",
    *,
    scale: Optional[torch.Tensor] = None,
    offset: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Return Q/DQ with ``(M,r)`` when mode is ``fixed_scale_qdq``, else ``None``."""

    if not is_v1_fixed_scale_qdq_mode():
        return None
    if not tensor_quantizer.enabled or tensor_quantizer.bitwidth == 32:
        return None
    if tensor_quantizer.data_type != QuantizationDataType.int:
        return None

    if scale is None or offset is None:
        scale, offset = scale_offset_from_static_grid_encoding(tensor, tensor_quantizer)

    qmin, qmax = qmin_qmax_for_v1_quantizer(tensor_quantizer)
    work = tensor
    if not work.is_floating_point():
        work = work.float()
    elif work.dtype not in (torch.float32, torch.float64):
        work = work.float()

    out = quantize_dequantize_from_float_encoding(
        work, scale, offset, qmin, qmax
    )
    return out.to(tensor.dtype if tensor.is_floating_point() else torch.float32)

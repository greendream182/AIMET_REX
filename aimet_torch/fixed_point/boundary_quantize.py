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
"""G3 boundary Q/DQ aligned with ``fixed_scale_qdq`` (M, rshift) when enabled."""

from __future__ import annotations

import os
from typing import Optional

import torch

from aimet_torch.fixed_point.encoding import FixedScaleEncoding
from aimet_torch.fixed_point.fixed_scale_qdq import quantize_with_fixed_scale
from aimet_torch.fixed_point.offline.scale_fixed import (
    _FIXED_SCALE_CACHE_ATTR,
    get_or_create_fixed_scale_encoding,
)
from aimet_torch.fixed_point.offline.scale_fixed import fixed_scale_float_scale
from aimet_torch.fixed_point.tensor import Int16QuantizedTensor


def int16_boundary_use_m_r() -> bool:
    """Whether G3 boundary quantize uses ``(m_int16, rshift)`` instead of float scale."""

    return os.environ.get("AIMET_RX_INT16_BOUNDARY_USE_M_R", "0").lower() in (
        "1",
        "true",
        "yes",
    )


def has_fixed_scale_cache(encoding) -> bool:
    return getattr(encoding, _FIXED_SCALE_CACHE_ATTR, None) is not None


def should_use_fixed_scale_boundary(encoding) -> bool:
    """Use (M,r) boundary Q when env is on or encodings were pre-converted."""

    # pylint: disable=import-outside-toplevel
    from aimet_torch.fixed_point.execution_mode import (
        ExecutionMode,
        get_quant_execution_mode,
    )

    if get_quant_execution_mode() is ExecutionMode.INT16_FIXED_EVAL:
        return True
    if not int16_boundary_use_m_r():
        return has_fixed_scale_cache(encoding)
    return True


def quantize_boundary_from_affine(
    tensor: torch.Tensor,
    encoding,
) -> Int16QuantizedTensor:
    """Quantize activations/weights at G3 boundaries (eval / qat_sim dispatch)."""

    if should_use_fixed_scale_boundary(encoding):
        fixed = (
            getattr(encoding, _FIXED_SCALE_CACHE_ATTR, None)
            or get_or_create_fixed_scale_encoding(encoding)
        )
        return Int16QuantizedTensor.from_fixed_scale_encoding(tensor, fixed)
    return Int16QuantizedTensor.from_affine_encoding(tensor, encoding)

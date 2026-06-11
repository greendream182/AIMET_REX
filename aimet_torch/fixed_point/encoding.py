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
"""Fixed-point encoding containers."""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class InputEncoding:
    """Integer tensor encoding metadata."""

    scale: torch.Tensor
    zero_point: torch.Tensor
    qmin: int
    qmax: int
    axis: Optional[int] = None


@dataclass(frozen=True)
class OutputEncoding(InputEncoding):
    """Output encoding with runtime integer requantization parameters.

    ``bias_bits`` selects the storage bit-width of Conv/Linear bias and is
    one of {16, 32} (default 32). The MAC accumulator stays in int32; with
    ``bias_bits=16`` the bias tensor is int16 and is up-cast to int32 at
    add-time. Spec 04_01 lists ``i16`` as the canonical hardware bias dtype;
    ``i32`` is the legacy/simulator default kept for backward compatibility.
    """

    multiplier: Optional[torch.Tensor] = None
    rshift: Optional[torch.Tensor] = None
    bias_bits: int = 32


@dataclass(frozen=True)
class FixedScaleEncoding:
    """Encoding for ``fixed_scale_qdq``: scale = m_int16 / 2**rshift."""

    m_int16: torch.Tensor
    rshift: torch.Tensor
    zero_point: torch.Tensor
    qmin: int
    qmax: int
    axis: Optional[int] = None
    scale_fp_legacy: Optional[torch.Tensor] = None

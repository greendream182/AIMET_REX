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
    """Output encoding with runtime integer requantization parameters."""

    multiplier: Optional[torch.Tensor] = None
    rshift: Optional[torch.Tensor] = None


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

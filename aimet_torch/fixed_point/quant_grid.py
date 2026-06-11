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
"""Standard quantization grids for fixed-point simulation (G3).

``FixedPointSimTensor`` is not INT16-only: ``qmin``/``qmax`` follow the AIMET
quantizer. This module names common grids (i8/u8/i16/u16/i32/u32) for tests and
offline tooling.

Storage: ``int_repr`` uses :data:`~aimet_torch.fixed_point.requantize.SIM_TENSOR_DTYPE`
(``torch.int32``). Codes must lie in ``[INT32_QMIN, INT32_QMAX]``; full **u32**
(``0 … 2**32-1``) needs a future int64 container (ADR-013).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from aimet_torch.fixed_point.requantize import INT32_QMAX, INT32_QMIN

MAX_SEMANTIC_LEVELS = 1 << 32


@dataclass(frozen=True)
class QuantGridSpec:
    """One symmetric or asymmetric integer quantization grid."""

    name: str
    qmin: int
    qmax: int
    default_zero_point: int = 0
    signed: bool = True

    def __post_init__(self) -> None:
        if self.qmin > self.qmax:
            raise ValueError(f"qmin ({self.qmin}) must be <= qmax ({self.qmax}).")
        levels = self.num_levels
        if levels <= 0 or levels > MAX_SEMANTIC_LEVELS:
            raise ValueError(
                f"grid {self.name}: invalid level count {levels} (max {MAX_SEMANTIC_LEVELS})."
            )

    @property
    def num_levels(self) -> int:
        return int(self.qmax) - int(self.qmin) + 1

    @property
    def fits_sim_int32_container(self) -> bool:
        """True when every code in ``[qmin, qmax]`` fits ``torch.int32`` storage."""

        return self.qmin >= INT32_QMIN and self.qmax <= INT32_QMAX


# Standard dtype names used in mixed-precision configs.
GRID_I8 = QuantGridSpec("i8", -128, 127, default_zero_point=0, signed=True)
GRID_U8 = QuantGridSpec("u8", 0, 255, default_zero_point=0, signed=False)
GRID_I16 = QuantGridSpec("i16", -32768, 32767, default_zero_point=0, signed=True)
GRID_U16 = QuantGridSpec("u16", 0, 65535, default_zero_point=0, signed=False)
GRID_I32 = QuantGridSpec(
    "i32", -2147483648, 2147483647, default_zero_point=0, signed=True
)
GRID_U32 = QuantGridSpec(
    "u32", 0, (1 << 32) - 1, default_zero_point=0, signed=False
)

STANDARD_QUANT_GRIDS: Tuple[QuantGridSpec, ...] = (
    GRID_I8,
    GRID_U8,
    GRID_I16,
    GRID_U16,
    GRID_I32,
    GRID_U32,
)

# Grids runnable on the current int32 sim-tensor container (excludes full u32).
SIM_INT32_QUANT_GRIDS: Tuple[QuantGridSpec, ...] = tuple(
    g for g in STANDARD_QUANT_GRIDS if g.fits_sim_int32_container
)

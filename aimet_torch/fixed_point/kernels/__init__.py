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
"""Fixed-point kernel registrations."""

import warnings

# Core kernels (no torchvision / custom op dependency).
from aimet_torch.fixed_point.kernels import conv_linear  # noqa: F401
from aimet_torch.fixed_point.kernels import clz_lut  # noqa: F401
from aimet_torch.fixed_point.kernels import lut  # noqa: F401
from aimet_torch.fixed_point.kernels import norm  # noqa: F401

# Optional kernels: pull in AIMET ``custom`` nn modules, which require torchvision.
try:
    from aimet_torch.fixed_point.kernels import eltwise  # noqa: F401
    from aimet_torch.fixed_point.kernels import pool  # noqa: F401
    from aimet_torch.fixed_point.kernels import shape_ops  # noqa: F401
except ImportError as exc:
    warnings.warn(
        "INT16 kernels eltwise/pool/shape_ops were not registered "
        f"(missing optional dependency): {exc}",
        UserWarning,
        stacklevel=1,
    )

__all__ = ["clz_lut", "conv_linear", "lut", "norm", "eltwise", "pool", "shape_ops"]

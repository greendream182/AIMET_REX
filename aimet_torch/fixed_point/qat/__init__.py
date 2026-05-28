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
"""QAT helpers for fixed-point simulation."""

from aimet_torch.fixed_point.qat.ste import FakeQuantInt16STE, fake_quantize_int16_qat

__all__ = ["FakeQuantInt16STE", "fake_quantize_int16_qat"]

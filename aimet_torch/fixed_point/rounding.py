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
"""Rounding policies for fixed-point requantization."""

from enum import Enum


class RoundingMode(str, Enum):
    """Supported integer rounding modes."""

    HALF_TO_EVEN = "half_to_even"
    HALF_AWAY_FROM_ZERO = "half_away_from_zero"
    HALF_UP = "half_up"
    TRUNCATE = "truncate"

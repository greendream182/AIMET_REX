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
"""Fixed-point / QuantGRU adapter errors."""

from __future__ import annotations

__all__ = [
    "IncompatibleAdapterVersionError",
    "QuantGRUFlagLockedError",
]


class IncompatibleAdapterVersionError(RuntimeError):
    """Raised when ``QuantGRU.aimet_capabilities()`` reports an unsupported adapter version."""


class QuantGRUFlagLockedError(RuntimeError):
    """Raised when user tries to mutate AIMET-managed QuantGRU flags directly."""

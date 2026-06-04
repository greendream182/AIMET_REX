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
"""Thread-local INT16 carrier map for ``int16_fixed_qat_sim`` super-group handoff."""

from __future__ import annotations

import contextvars
from typing import Any, Optional

import torch

from aimet_torch.fixed_point.tensor import Int16QuantizedTensor

_int16_carrier_map: contextvars.ContextVar[dict[int, Int16QuantizedTensor]] = (
    contextvars.ContextVar("aimet_rx_int16_carrier_map", default=None)
)


def publish_int16_carrier(float_tensor: torch.Tensor, carrier: Int16QuantizedTensor) -> None:
    """Associate ``carrier`` with ``float_tensor`` for the next consumer op in qat_sim."""

    mapping = _int16_carrier_map.get()
    if mapping is None:
        mapping = {}
        _int16_carrier_map.set(mapping)
    mapping[id(float_tensor)] = carrier


def consume_int16_carrier(float_tensor: torch.Tensor) -> Optional[Int16QuantizedTensor]:
    """Pop the INT16 carrier published for ``float_tensor``, if any."""

    mapping = _int16_carrier_map.get()
    if not mapping:
        return None
    return mapping.pop(id(float_tensor), None)


def maybe_int16_carrier(data: Any) -> Optional[Int16QuantizedTensor]:
    """Return an :class:`Int16QuantizedTensor` carried by ``data`` or passed through directly."""

    if isinstance(data, Int16QuantizedTensor):
        return data
    if isinstance(data, torch.Tensor) and data.is_floating_point():
        return consume_int16_carrier(data)
    return None


def clear_int16_carriers() -> None:
    """Drop any unconsumed QAT carriers from the current execution context."""

    mapping = _int16_carrier_map.get()
    if mapping:
        mapping.clear()

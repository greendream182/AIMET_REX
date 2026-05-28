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
"""Thread-local flags for INT16 eval debug paths (``to_float`` under profiler)."""

import threading
from contextlib import contextmanager
from typing import Iterator

_LOCAL = threading.local()


def int16_eval_debug_float_allowed() -> bool:
    """Return True when :class:`FixedPointProfiler` (or tests) allow ``to_float`` in eval."""

    return bool(getattr(_LOCAL, "allow_debug_float", False))


def set_int16_eval_debug_float_allowed(allowed: bool) -> None:
    """Enable/disable permissive ``to_float`` in :class:`~aimet_torch.fixed_point.execution_mode.ExecutionMode.INT16_FIXED_EVAL`."""

    _LOCAL.allow_debug_float = allowed


@contextmanager
def int16_eval_allow_debug_float() -> Iterator[None]:
    """Temporarily allow :meth:`Int16QuantizedTensor.to_float` in INT16 eval (tests, ad-hoc debug)."""

    previous = getattr(_LOCAL, "allow_debug_float", False)
    set_int16_eval_debug_float_allowed(True)
    try:
        yield
    finally:
        set_int16_eval_debug_float_allowed(previous)

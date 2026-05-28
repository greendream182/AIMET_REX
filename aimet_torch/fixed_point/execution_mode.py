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
"""Execution mode controls for AIMET RX quantization experiments."""

import os
import threading
from contextlib import contextmanager
from enum import Enum
from typing import Iterator, Union


class ExecutionMode(str, Enum):
    """Supported quantization execution modes."""

    FP32_QDQ = "fp32_qdq"
    FP16_QDQ = "fp16_qdq"
    FIXED_SCALE_QDQ = "fixed_scale_qdq"
    INT16_FIXED_EVAL = "int16_fixed_eval"
    INT16_FIXED_QAT_SIM = "int16_fixed_qat_sim"


ModeLike = Union[ExecutionMode, str]

_ENV_VAR = "AIMET_RX_QUANT_EXECUTION_MODE"
_LOCK = threading.RLock()
_STATE = threading.local()


def _parse_execution_mode(mode: ModeLike) -> ExecutionMode:
    if isinstance(mode, ExecutionMode):
        return mode

    if isinstance(mode, str):
        try:
            return ExecutionMode(mode)
        except ValueError as exc:
            valid_modes = ", ".join(item.value for item in ExecutionMode)
            raise ValueError(
                f"Unsupported quant execution mode: {mode!r}. "
                f"Valid modes are: {valid_modes}."
            ) from exc

    raise TypeError(
        "Quant execution mode must be an ExecutionMode or str; "
        f"got {type(mode).__name__}."
    )


def _get_initial_mode() -> ExecutionMode:
    env_mode = os.environ.get(_ENV_VAR)
    if env_mode:
        return _parse_execution_mode(env_mode)
    return ExecutionMode.FP32_QDQ


def set_quant_execution_mode(mode: ModeLike) -> None:
    """Set the current process-local quantization execution mode."""

    parsed_mode = _parse_execution_mode(mode)
    with _LOCK:
        _STATE.current_mode = parsed_mode


def get_quant_execution_mode() -> ExecutionMode:
    """Return the current quantization execution mode."""

    with _LOCK:
        current_mode = getattr(_STATE, "current_mode", None)
        if current_mode is None:
            current_mode = _get_initial_mode()
            _STATE.current_mode = current_mode
        return current_mode


@contextmanager
def quant_execution_mode(mode: ModeLike) -> Iterator[ExecutionMode]:
    """Temporarily switch quantization execution mode within a context."""

    previous_mode = get_quant_execution_mode()
    set_quant_execution_mode(mode)
    try:
        yield get_quant_execution_mode()
    finally:
        set_quant_execution_mode(previous_mode)

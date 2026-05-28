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
"""Shared fixtures for fixed-point tests.

Full ``tests/fixed_point/`` coverage expects a normal AIMET stack, including at least:
``torch``, ``numpy``, ``onnx``, ``onnxscript``, ``torchvision``, ``packaging``, ``pytest``,
and built/installed ``aimet_common`` (libpymo). Core tests such as ``test_execution_mode``
and ``test_requantize`` run with only ``torch`` + ``numpy`` after lazy-loading in
``aimet_torch.__init__``.

When ``quant_gru`` is not installed via ``pip``, sibling checkout
``../quant-gru-pytorch/pytorch`` (or ``QUANT_GRU_PYTORCH_DIR``) is prepended to
``sys.path`` so CUDA wrapper tests can run without manual ``PYTHONPATH``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


def _maybe_prepend_quant_gru_pytorch() -> None:
    """Make local quant-gru-pytorch checkout importable before test collection."""
    try:
        import quant_gru  # noqa: F401
        return
    except ImportError:
        pass

    env_dir = os.environ.get("QUANT_GRU_PYTORCH_DIR")
    candidates = []
    if env_dir:
        candidates.append(Path(env_dir))
    fixed_point_dir = Path(__file__).resolve().parent
    candidates.extend(
        [
            fixed_point_dir.parent.parent.parent / "quant-gru-pytorch" / "pytorch",
            fixed_point_dir.parent.parent / "quant-gru-pytorch" / "pytorch",
        ]
    )

    for candidate in candidates:
        if not candidate.is_dir():
            continue
        if not (candidate / "quant_gru.py").is_file():
            continue
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)
        return


_maybe_prepend_quant_gru_pytorch()


@pytest.fixture(autouse=True)
def reset_quant_execution_mode():
    """Each test starts and ends in FP32_QDQ so mode never leaks across cases or files."""

    try:
        from aimet_torch.fixed_point import ExecutionMode, set_quant_execution_mode
    except ImportError:
        yield
        return

    set_quant_execution_mode(ExecutionMode.FP32_QDQ)
    yield
    set_quant_execution_mode(ExecutionMode.FP32_QDQ)

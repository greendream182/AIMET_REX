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
"""Shared helpers for fixed-point tests (spec 13)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import torch

from aimet_torch.fixed_point import FixedPointSimTensor, Int16QuantizedTensor, InputEncoding, OutputEncoding
from aimet_torch.fixed_point.requantize import SIM_TENSOR_DTYPE

DATA_DIR = Path(__file__).resolve().parent / "data"


def load_golden_npz(name: str) -> Dict[str, np.ndarray]:
    """Load ``tests/fixed_point/data/<name>.npz``."""

    path = DATA_DIR / f"{name}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Golden data not found: {path}")
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def load_golden_meta(name: str) -> Dict[str, Any]:
    """Optional JSON metadata alongside a golden npz."""

    path = DATA_DIR / f"{name}.json"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def assert_int_tensor_equal(
    actual: Union[torch.Tensor, np.ndarray],
    expected: Union[torch.Tensor, np.ndarray],
    *,
    msg: str = "",
) -> None:
    """Exact integer equality (no atol)."""

    a = torch.as_tensor(actual)
    e = torch.as_tensor(expected)
    if a.shape != e.shape:
        raise AssertionError(f"{msg} shape {tuple(a.shape)} != {tuple(e.shape)}")
    if not torch.equal(a, e):
        raise AssertionError(f"{msg} values differ: {a.tolist()} != {e.tolist()}")


def build_int16_tensor(
    values,
    *,
    scale: float = 1.0,
    zero_point: int = 0,
) -> FixedPointSimTensor:
    int_repr = torch.tensor(values, dtype=SIM_TENSOR_DTYPE)
    return FixedPointSimTensor(
        int_repr=int_repr,
        scale=torch.tensor(scale, dtype=torch.float32),
        zero_point=torch.tensor(zero_point, dtype=torch.int32),
    )


def build_output_encoding(
    *,
    multiplier: int = 32767,
    rshift: int = 15,
    scale: float = 1.0,
    zero_point: int = 0,
) -> OutputEncoding:
    return OutputEncoding(
        scale=torch.tensor(scale, dtype=torch.float32),
        zero_point=torch.tensor(zero_point, dtype=torch.int32),
        qmin=-32768,
        qmax=32767,
        multiplier=torch.tensor(multiplier, dtype=torch.uint16),
        rshift=torch.tensor(rshift, dtype=torch.int8),
    )


def build_input_encoding(
    *,
    scale: float = 1.0,
    zero_point: int = 0,
) -> InputEncoding:
    return InputEncoding(
        scale=torch.tensor(scale, dtype=torch.float32),
        zero_point=torch.tensor(zero_point, dtype=torch.int32),
        qmin=-32768,
        qmax=32767,
    )

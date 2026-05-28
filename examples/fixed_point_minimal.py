#!/usr/bin/env python3
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
"""Minimal AIMET RX fixed-point execution-mode example.

This is intentionally smaller than ``quick_start.py``: it starts from an already
quantized AIMET v2 module with initialized encodings, then shows how to compare:

* ``fp32_qdq``: normal AIMET Q/DQ reference path
* ``fixed_scale_qdq``: G2 path, Q/DQ scale represented as ``(M_int16, rshift)``
* ``int16_fixed_eval``: G3 path, integer carrier plus fixed-point kernels

For production PTQ/QAT flows, compute encodings with ``QuantizationSimModel`` and
call ``ensure_output_quantizers_for_int16_eval(sim)`` before calibration.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn as nn

import aimet_torch.fixed_point.kernels  # noqa: F401 - register fixed kernels
from aimet_torch.fixed_point import (
    ExecutionMode,
    FixedPointSimTensor,
    quant_execution_mode,
)
from aimet_torch.fixed_point.metrics import int16_eval_allow_debug_float
from aimet_torch.v2.nn import QuantizedLinear
from aimet_torch.v2.quantization.affine import Quantize


def _to_float(output):
    """Return a float tensor from either AIMET QTensor or fixed-point carrier."""

    if isinstance(output, FixedPointSimTensor):
        return output.to_float()
    if hasattr(output, "dequantize"):
        return output.dequantize()
    return output


def build_quantized_linear() -> QuantizedLinear:
    """Create a tiny quantized layer with initialized encodings."""

    layer = QuantizedLinear(4, 3, bias=True)
    layer.input_quantizers[0] = Quantize((), 8, symmetric=True)
    layer.param_quantizers["weight"] = Quantize((3, 1), 8, symmetric=True)
    layer.output_quantizers[0] = Quantize((), 8, symmetric=True)

    layer.input_quantizers[0].min = nn.Parameter(torch.tensor(-2.0))
    layer.input_quantizers[0].max = nn.Parameter(torch.tensor(2.0))
    layer.param_quantizers["weight"].min = nn.Parameter(torch.full((3, 1), -0.5))
    layer.param_quantizers["weight"].max = nn.Parameter(torch.full((3, 1), 0.5))
    layer.output_quantizers[0].min = nn.Parameter(torch.tensor(-1.0))
    layer.output_quantizers[0].max = nn.Parameter(torch.tensor(1.0))

    nn.init.constant_(layer.weight, 0.1)
    nn.init.constant_(layer.bias, 0.05)
    return layer.eval()


def main() -> int:
    torch.manual_seed(0)

    model = build_quantized_linear()
    x = torch.tensor([[0.5, -0.25, 0.125, 0.0]], dtype=torch.float32)

    with torch.no_grad(), quant_execution_mode(ExecutionMode.FP32_QDQ):
        y_fp32_qdq = _to_float(model(x))

    with torch.no_grad(), quant_execution_mode(ExecutionMode.FIXED_SCALE_QDQ):
        y_fixed_scale = _to_float(model(x))

    with torch.no_grad(), quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y_int16 = model(x)
        with int16_eval_allow_debug_float():
            y_int16_float = _to_float(y_int16)

    print("input:", x)
    print("fp32_qdq:", y_fp32_qdq)
    print("fixed_scale_qdq:", y_fixed_scale)
    print("int16_fixed_eval dequantized:", y_int16_float)
    print(
        "max |fixed_scale_qdq - fp32_qdq|:",
        torch.max(torch.abs(y_fixed_scale - y_fp32_qdq)).item(),
    )
    print(
        "max |int16_fixed_eval - fp32_qdq|:",
        torch.max(torch.abs(y_int16_float - y_fp32_qdq)).item(),
    )

    if isinstance(y_int16, FixedPointSimTensor):
        print("int carrier dtype:", y_int16.int_repr.dtype)
        print("int carrier values:", y_int16.int_repr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

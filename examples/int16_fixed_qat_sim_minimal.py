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
"""Minimal INT16 fixed-point QAT simulation example.

``INT16_FIXED_QAT_SIM`` runs the forward pass through the fixed-point simulation
path and uses a surrogate gradient path for backpropagation. This example is a
tiny training-loop skeleton only; it does not replace INT16 eval, hardware
correlation, or a production QAT recipe for a real model.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

import aimet_torch.fixed_point.kernels  # noqa: F401 - register fixed kernels
from aimet_torch.fixed_point import ExecutionMode, FixedPointSimTensor, quant_execution_mode
from aimet_torch.fixed_point.metrics import int16_eval_allow_debug_float
from aimet_torch.v2.nn import QuantizedLinear
from aimet_torch.v2.quantization.affine import Quantize


def _to_float(output):
    if isinstance(output, FixedPointSimTensor):
        with int16_eval_allow_debug_float():
            return output.to_float()
    if hasattr(output, "dequantize"):
        return output.dequantize()
    return output


def build_quantized_student() -> QuantizedLinear:
    """Create a tiny initialized quantized student layer."""

    layer = QuantizedLinear(4, 3, bias=True)
    layer.input_quantizers[0] = Quantize((), 8, symmetric=True)
    layer.param_quantizers["weight"] = Quantize((3, 1), 8, symmetric=True)
    layer.output_quantizers[0] = Quantize((), 8, symmetric=True)

    layer.input_quantizers[0].min = nn.Parameter(torch.tensor(-2.0))
    layer.input_quantizers[0].max = nn.Parameter(torch.tensor(2.0))
    layer.param_quantizers["weight"].min = nn.Parameter(torch.full((3, 1), -0.5))
    layer.param_quantizers["weight"].max = nn.Parameter(torch.full((3, 1), 0.5))
    layer.output_quantizers[0].min = nn.Parameter(torch.tensor(-2.0))
    layer.output_quantizers[0].max = nn.Parameter(torch.tensor(2.0))

    nn.init.constant_(layer.weight, 0.05)
    nn.init.constant_(layer.bias, 0.0)
    return layer


def build_float_teacher() -> nn.Linear:
    """Create a small float teacher used only to generate toy targets."""

    teacher = nn.Linear(4, 3, bias=True)
    with torch.no_grad():
        teacher.weight.copy_(
            torch.tensor(
                [
                    [0.30, -0.10, 0.20, 0.05],
                    [-0.20, 0.25, 0.10, -0.15],
                    [0.10, 0.05, -0.30, 0.20],
                ],
                dtype=torch.float32,
            )
        )
        teacher.bias.copy_(torch.tensor([0.05, -0.02, 0.03], dtype=torch.float32))
    return teacher.eval()


def main() -> int:
    torch.manual_seed(7)

    student = build_quantized_student()
    teacher = build_float_teacher()
    optimizer = torch.optim.SGD(student.parameters(), lr=0.25)

    x = torch.tensor(
        [
            [0.5, -0.25, 0.125, 0.0],
            [-0.4, 0.2, 0.1, 0.3],
            [0.25, 0.1, -0.2, -0.1],
            [0.0, -0.5, 0.35, 0.2],
        ],
        dtype=torch.float32,
    )
    with torch.no_grad():
        target = teacher(x)

    student.train()
    losses: list[float] = []
    for step in range(8):
        optimizer.zero_grad(set_to_none=True)
        with quant_execution_mode(ExecutionMode.INT16_FIXED_QAT_SIM):
            pred = student(x)
        loss = F.mse_loss(pred.float(), target)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        print(f"step={step:02d} loss={losses[-1]:.6f}")

    student.eval()
    with torch.no_grad(), quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        eval_output = student(x)

    print("initial loss:", f"{losses[0]:.6f}")
    print("final loss:", f"{losses[-1]:.6f}")
    print("int16 eval output:", _to_float(eval_output))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

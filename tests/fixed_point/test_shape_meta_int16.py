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
"""Tests for shape-meta scalar bypass in INT16 dispatch (``shape_meta.py``)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from aimet_torch._base.nn.modules import custom
from aimet_torch.fixed_point.shape_meta import (
    is_intentionally_unquantized_module,
    is_shape_meta_only_quantized_module,
    is_shape_meta_scalar,
    try_dispatch_shape_meta_op,
)
from aimet_torch.fixed_point.tensor import FixedPointSimTensor, Int16QuantizedTensor
from aimet_torch.fixed_point.requantize import SIM_TENSOR_DTYPE


def _carrier() -> FixedPointSimTensor:
    return FixedPointSimTensor(
        int_repr=torch.zeros(2, 3, dtype=SIM_TENSOR_DTYPE),
        scale=torch.tensor(1.0),
        zero_point=torch.tensor(0, dtype=torch.int32),
    )


@pytest.mark.parametrize(
    "value,expected",
    [
        (2, True),
        (-1, True),
        (torch.tensor(4, dtype=torch.int64), True),
        (torch.tensor([2, 3], dtype=torch.int64), True),
        (torch.tensor(1.0), False),
        (_carrier(), False),
    ],
)
def test_is_shape_meta_scalar(value, expected):
    assert is_shape_meta_scalar(value) is expected


def test_try_dispatch_multiply_python_ints():
    ok, out = try_dispatch_shape_meta_op(custom.Multiply, 2, 3)
    assert ok is True
    assert out == 6


def test_try_dispatch_multiply_int64_tensors():
    ok, out = try_dispatch_shape_meta_op(
        custom.Multiply,
        torch.tensor(2, dtype=torch.int64),
        torch.tensor(3, dtype=torch.int64),
    )
    assert ok is True
    assert int(out.item()) == 6


def test_try_dispatch_rejects_activation_operands():
    ok, out = try_dispatch_shape_meta_op(custom.Multiply, _carrier(), 2)
    assert ok is False
    assert out is None


def test_try_dispatch_add_subtract_divide():
    assert try_dispatch_shape_meta_op(custom.Add, 5, 7) == (True, 12)
    assert try_dispatch_shape_meta_op(custom.Subtract, 10, 3) == (True, 7)
    assert try_dispatch_shape_meta_op(custom.FloorDivide, 10, 3) == (True, 3)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for QuantizedMultiply integration",
)
def test_dispatch_int16_fixed_multiply_shape_scalars_without_oq():
    """dispatch_int16_fixed must bypass oq for shape-meta Multiply(b, f)."""
    pytest.importorskip("onnxscript")
    from aimet_torch.fixed_point import ExecutionMode, quant_execution_mode
    from aimet_torch.v2.nn.modules.custom import QuantizedMultiply
    from aimet_torch.v2.quantization.affine.fixed_point.adapter import (
        dispatch_int16_fixed,
    )

    mul = QuantizedMultiply().cuda()
    # Deliberately leave output quantizer None — meta bypass must not require it.
    assert mul.output_quantizers[0] is None

    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        out = dispatch_int16_fixed(mul, 2, 3)
    assert out == 6

    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        out_t = dispatch_int16_fixed(
            mul,
            torch.tensor(2, device="cuda", dtype=torch.int64),
            torch.tensor(3, device="cuda", dtype=torch.int64),
        )
    assert int(out_t.item()) == 6


def test_is_shape_meta_only_quantized_module_detects_layout_mul():
    pytest.importorskip("onnxscript")
    from aimet_torch.v2.nn.modules.custom import QuantizedMultiply

    mul = QuantizedMultiply()
    assert mul.output_quantizers[0] is None
    assert is_shape_meta_only_quantized_module(mul) is True


def test_is_intentionally_unquantized_module():
    pytest.importorskip("onnxscript")
    from aimet_torch.v2.nn.modules.custom import QuantizedMultiply

    mul = QuantizedMultiply()
    mul.input_quantizers[0] = None
    mul.output_quantizers[0] = None
    assert is_intentionally_unquantized_module(mul) is True


def test_diagnose_skips_shape_meta_uninitialized_encoding():
    pytest.importorskip("onnxscript")
    from aimet_torch.fixed_point.diagnose import diagnose_int16_readiness
    from aimet_torch.v2.nn.modules.custom import QuantizedMultiply
    from aimet_torch.v2.quantization.affine import Quantize

    mul = QuantizedMultiply()
    mul.output_quantizers[0] = Quantize((), 16, symmetric=True)
    assert not mul.output_quantizers[0].is_initialized()
    assert is_shape_meta_only_quantized_module(mul) is True

    class _FakeSim:
        model = torch.nn.Module()
        model.mul = mul

    report = diagnose_int16_readiness(_FakeSim())
    assert report["uninitialized_encoding"] == []

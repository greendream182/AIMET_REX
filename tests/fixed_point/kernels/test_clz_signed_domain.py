"""CLZ kernels: signed-domain semantics for power_2 and reciprocal."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import aimet_torch.fixed_point.kernels  # noqa: F401
from aimet_torch.fixed_point import ExecutionMode, quant_execution_mode
from aimet_torch.fixed_point.metrics import assert_int16_vs_fp32_reference
from aimet_torch.fixed_point.offline.clz_gen import resolve_abc_lut_root
from aimet_torch.v2.nn.modules.custom import QuantizedReciprocal, QuantizedSquare
from aimet_torch.v2.quantization.affine import Quantize

pytestmark = pytest.mark.skipif(
    resolve_abc_lut_root() is None, reason="abc_lut-shuai not found"
)


def _init_positive_unary(m: nn.Module, *, in_max: float = 4.0, out_max: float = 4.0) -> None:
    m.input_quantizers[0] = Quantize((), 16, symmetric=True)
    m.output_quantizers[0] = Quantize((), 16, symmetric=True)
    m.input_quantizers[0].min = nn.Parameter(torch.tensor(-in_max))
    m.input_quantizers[0].max = nn.Parameter(torch.tensor(in_max))
    m.output_quantizers[0].min = nn.Parameter(torch.tensor(0.0))
    m.output_quantizers[0].max = nn.Parameter(torch.tensor(out_max))


def test_square_clz_handles_negative_input_as_x_squared():
    m = QuantizedSquare()
    _init_positive_unary(m)
    x = torch.tensor([[-2.0, -1.0, 0.5, 2.0]], dtype=torch.float32)

    with quant_execution_mode(ExecutionMode.FP32_QDQ):
        y_fp = m(x)
        y_fp = y_fp.dequantize() if hasattr(y_fp, "dequantize") else y_fp

    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y_int = m(x)

    assert_int16_vs_fp32_reference(y_int, y_fp, max_lsb=8192.0, min_cosine=0.98)


def test_reciprocal_clz_negative_input_sign_not_saturated_to_max():
    m = QuantizedReciprocal()
    _init_positive_unary(m, out_max=4.0)
    m.input_quantizers[0].min = nn.Parameter(torch.tensor(-2.0))
    m.input_quantizers[0].max = nn.Parameter(torch.tensor(2.0))
    x = torch.tensor([[-1.0, -0.5, 0.5, 1.0]], dtype=torch.float32)

    with quant_execution_mode(ExecutionMode.FP32_QDQ):
        y_fp = m(x)
        y_fp = y_fp.dequantize() if hasattr(y_fp, "dequantize") else y_fp

    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y_int = m(x)

    y_i = y_int.to_float()
    assert y_i[0, 0] < 0.0
    assert y_i[0, 1] < 0.0
    assert y_i[0, 2] > 0.0
    assert_int16_vs_fp32_reference(y_int, y_fp, max_lsb=16384.0, min_cosine=0.95)

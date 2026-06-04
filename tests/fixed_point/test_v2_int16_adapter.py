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

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxscript")
import torch.nn as nn  # noqa: E402

import aimet_torch.fixed_point.kernels  # noqa: F401, E402
from aimet_torch.fixed_point import (  # noqa: E402
    ExecutionMode,
    Int16QuantizedTensor,
    quant_execution_mode,
)
from aimet_torch.fixed_point.metrics import assert_int16_vs_fp32_reference  # noqa: E402
from aimet_torch.fixed_point.offline.clz_gen import resolve_abc_lut_root  # noqa: E402
from aimet_torch.v2.nn import (  # noqa: E402
    QuantizedConv1d,
    QuantizedConv2d,
    QuantizedDropout,
    QuantizedLinear,
    QuantizedSoftmax,
)
from aimet_torch.v2.nn.modules.custom import (  # noqa: E402
    QuantizedCos,
    QuantizedExponential,
    QuantizedLog,
    QuantizedReciprocal,
    QuantizedReshape,
    QuantizedRSqrt,
    QuantizedSin,
    QuantizedSqrt,
    QuantizedSquare,
)
from aimet_torch.v2.quantization.affine import Quantize  # noqa: E402


def _init_linear_quantizers(m: QuantizedLinear) -> None:
    m.input_quantizers[0] = Quantize((), 8, symmetric=True)
    m.param_quantizers["weight"] = Quantize((m.out_features, 1), 8, symmetric=True)
    m.output_quantizers[0] = Quantize((), 8, symmetric=True)
    m.input_quantizers[0].min = nn.Parameter(torch.tensor(-4.0))
    m.input_quantizers[0].max = nn.Parameter(torch.tensor(4.0))
    m.param_quantizers["weight"].min = nn.Parameter(torch.full((m.out_features, 1), -0.5))
    m.param_quantizers["weight"].max = nn.Parameter(torch.full((m.out_features, 1), 0.5))
    m.output_quantizers[0].min = nn.Parameter(torch.tensor(-2.0))
    m.output_quantizers[0].max = nn.Parameter(torch.tensor(2.0))


def _init_linear_quantizers_bw(m: QuantizedLinear, weight_bw: int, act_bw: int) -> None:
    m.input_quantizers[0] = Quantize((), act_bw, symmetric=True)
    m.param_quantizers["weight"] = Quantize((m.out_features, 1), weight_bw, symmetric=True)
    m.output_quantizers[0] = Quantize((), act_bw, symmetric=True)
    m.input_quantizers[0].min = nn.Parameter(torch.tensor(-2.0))
    m.input_quantizers[0].max = nn.Parameter(torch.tensor(2.0))
    m.param_quantizers["weight"].min = nn.Parameter(torch.full((m.out_features, 1), -0.5))
    m.param_quantizers["weight"].max = nn.Parameter(torch.full((m.out_features, 1), 0.5))
    m.output_quantizers[0].min = nn.Parameter(torch.tensor(-2.0))
    m.output_quantizers[0].max = nn.Parameter(torch.tensor(2.0))


def _init_conv2d_quantizers(m: QuantizedConv2d) -> None:
    oc = m.out_channels
    m.input_quantizers[0] = Quantize((), 8, symmetric=True)
    m.param_quantizers["weight"] = Quantize((oc, 1, 1, 1), 8, symmetric=True)
    m.output_quantizers[0] = Quantize((), 8, symmetric=True)
    m.input_quantizers[0].min = nn.Parameter(torch.tensor(-2.0))
    m.input_quantizers[0].max = nn.Parameter(torch.tensor(2.0))
    m.param_quantizers["weight"].min = nn.Parameter(torch.full((oc, 1, 1, 1), -0.3))
    m.param_quantizers["weight"].max = nn.Parameter(torch.full((oc, 1, 1, 1), 0.3))
    m.output_quantizers[0].min = nn.Parameter(torch.tensor(-4.0))
    m.output_quantizers[0].max = nn.Parameter(torch.tensor(4.0))


def _init_conv1d_quantizers(m: QuantizedConv1d) -> None:
    oc = m.out_channels
    m.input_quantizers[0] = Quantize((), 8, symmetric=True)
    m.param_quantizers["weight"] = Quantize((oc, 1, 1), 8, symmetric=True)
    m.output_quantizers[0] = Quantize((), 8, symmetric=True)
    m.input_quantizers[0].min = nn.Parameter(torch.tensor(-2.0))
    m.input_quantizers[0].max = nn.Parameter(torch.tensor(2.0))
    m.param_quantizers["weight"].min = nn.Parameter(torch.full((oc, 1, 1), -0.35))
    m.param_quantizers["weight"].max = nn.Parameter(torch.full((oc, 1, 1), 0.35))
    m.output_quantizers[0].min = nn.Parameter(torch.tensor(-3.0))
    m.output_quantizers[0].max = nn.Parameter(torch.tensor(3.0))


def test_quantized_linear_int16_fixed_dispatch():
    m = QuantizedLinear(3, 2, bias=True)
    _init_linear_quantizers(m)
    nn.init.constant_(m.weight, 0.1)
    nn.init.constant_(m.bias, 0.05)

    x = torch.tensor([[0.5, -0.25, 0.0]], dtype=torch.float32)

    with quant_execution_mode(ExecutionMode.FP32_QDQ):
        y_q = m(x)
        y_fp = y_q.dequantize() if hasattr(y_q, "dequantize") else y_q

    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y_int = m(x)

    assert isinstance(y_int, Int16QuantizedTensor)
    assert y_int.to_float().shape == y_fp.shape
    assert_int16_vs_fp32_reference(y_int, y_fp)


@pytest.mark.parametrize(
    "weight_bw,act_bw",
    [
        pytest.param(4, 8, id="W4A8"),
        pytest.param(8, 8, id="W8A8"),
        pytest.param(4, 16, id="W4A16"),
        pytest.param(16, 16, id="W16A16"),
    ],
)
def test_quantized_linear_int16_fixed_mixed_bitwidth_dispatch(weight_bw: int, act_bw: int):
    m = QuantizedLinear(4, 3, bias=True)
    _init_linear_quantizers_bw(m, weight_bw, act_bw)
    with torch.no_grad():
        m.weight.copy_(
            torch.tensor(
                [
                    [0.20, -0.10, 0.05, 0.15],
                    [-0.25, 0.10, 0.20, -0.05],
                    [0.05, 0.30, -0.15, 0.10],
                ],
                dtype=torch.float32,
            )
        )
        m.bias.copy_(torch.tensor([0.01, -0.02, 0.03], dtype=torch.float32))

    expected_w_qmin = -(2 ** (weight_bw - 1))
    expected_w_qmax = 2 ** (weight_bw - 1) - 1
    expected_a_qmin = -(2 ** (act_bw - 1))
    expected_a_qmax = 2 ** (act_bw - 1) - 1

    x = torch.tensor(
        [
            [0.5, -0.25, 0.0, 0.75],
            [-0.5, 0.1, 0.3, -0.2],
        ],
        dtype=torch.float32,
    )

    w_enc = m.param_quantizers["weight"].get_encodings()
    x_enc = m.input_quantizers[0].get_encodings()
    y_enc = m.output_quantizers[0].get_encodings()
    assert (w_enc.qmin, w_enc.qmax) == (expected_w_qmin, expected_w_qmax)
    assert (x_enc.qmin, x_enc.qmax) == (expected_a_qmin, expected_a_qmax)
    assert (y_enc.qmin, y_enc.qmax) == (expected_a_qmin, expected_a_qmax)

    with quant_execution_mode(ExecutionMode.FP32_QDQ):
        y_q = m(x)
        y_fp = y_q.dequantize() if hasattr(y_q, "dequantize") else y_q

    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y_int = m(x)

    assert isinstance(y_int, Int16QuantizedTensor)
    assert y_int.int_repr.dtype is torch.int32
    assert (y_int.qmin, y_int.qmax) == (expected_a_qmin, expected_a_qmax)
    assert torch.all(y_int.int_repr >= expected_a_qmin)
    assert torch.all(y_int.int_repr <= expected_a_qmax)
    assert y_int.to_float().shape == y_fp.shape
    assert_int16_vs_fp32_reference(y_int, y_fp)


def test_quantized_linear_no_bias_int16_fixed_dispatch():
    m = QuantizedLinear(4, 2, bias=False)
    _init_linear_quantizers(m)
    nn.init.constant_(m.weight, 0.15)

    x = torch.tensor([[0.1, -0.2, 0.3, 0.0]], dtype=torch.float32)

    with quant_execution_mode(ExecutionMode.FP32_QDQ):
        y_q = m(x)
        y_fp = y_q.dequantize() if hasattr(y_q, "dequantize") else y_q

    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y_int = m(x)

    assert isinstance(y_int, Int16QuantizedTensor)
    assert y_int.to_float().shape == y_fp.shape
    assert_int16_vs_fp32_reference(y_int, y_fp)


def test_quantized_conv2d_int16_fixed_dispatch():
    m = QuantizedConv2d(1, 2, kernel_size=2, stride=1, padding=0, bias=True)
    _init_conv2d_quantizers(m)
    nn.init.constant_(m.weight, 0.2)
    nn.init.constant_(m.bias, 0.01)

    x = torch.tensor([[[[0.5, -0.25, 0.1], [0.0, 0.3, -0.2], [0.1, 0.1, 0.0]]]])

    with quant_execution_mode(ExecutionMode.FP32_QDQ):
        y_q = m(x)
        y_fp = y_q.dequantize() if hasattr(y_q, "dequantize") else y_q

    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y_int = m(x)

    assert isinstance(y_int, Int16QuantizedTensor)
    assert y_int.to_float().shape == y_fp.shape
    assert_int16_vs_fp32_reference(y_int, y_fp)


def test_quantized_conv1d_int16_fixed_dispatch():
    m = QuantizedConv1d(1, 2, kernel_size=2, stride=1, padding=0, bias=True)
    _init_conv1d_quantizers(m)
    nn.init.constant_(m.weight, 0.25)
    nn.init.constant_(m.bias, 0.02)

    x = torch.tensor([[[0.4, -0.1, 0.2, 0.0, 0.15]]])

    with quant_execution_mode(ExecutionMode.FP32_QDQ):
        y_q = m(x)
        y_fp = y_q.dequantize() if hasattr(y_q, "dequantize") else y_q

    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y_int = m(x)

    assert isinstance(y_int, Int16QuantizedTensor)
    assert y_int.to_float().shape == y_fp.shape
    assert_int16_vs_fp32_reference(y_int, y_fp)


def test_quantized_conv2d_groups_int16_fixed_dispatch():
    m = QuantizedConv2d(2, 2, kernel_size=2, stride=1, padding=0, groups=2, bias=False)
    _init_conv2d_quantizers(m)
    nn.init.constant_(m.weight, 0.18)

    x = torch.tensor(
        [
            [
                [[0.2, -0.1], [0.0, 0.3]],
                [[-0.15, 0.1], [0.05, -0.2]],
            ]
        ]
    )

    with quant_execution_mode(ExecutionMode.FP32_QDQ):
        y_q = m(x)
        y_fp = y_q.dequantize() if hasattr(y_q, "dequantize") else y_q

    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y_int = m(x)

    assert isinstance(y_int, Int16QuantizedTensor)
    assert y_int.to_float().shape == y_fp.shape
    assert_int16_vs_fp32_reference(y_int, y_fp)


def test_quantized_softmax_int16_reference_kernel():
    m = QuantizedSoftmax(dim=-1)
    m.input_quantizers[0] = Quantize((), 8, symmetric=True)
    m.output_quantizers[0] = Quantize((), 8, symmetric=True)
    m.input_quantizers[0].min = nn.Parameter(torch.tensor(-4.0))
    m.input_quantizers[0].max = nn.Parameter(torch.tensor(4.0))
    m.output_quantizers[0].min = nn.Parameter(torch.tensor(0.0))
    m.output_quantizers[0].max = nn.Parameter(torch.tensor(1.0))

    x = torch.tensor([[1.0, 2.0, 0.5, -1.0], [0.0, 0.1, -0.2, 0.3]], dtype=torch.float32)

    with quant_execution_mode(ExecutionMode.FP32_QDQ):
        y_q = m(x)
        y_fp = y_q.dequantize() if hasattr(y_q, "dequantize") else y_q

    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y_int = m(x)

    assert isinstance(y_int, Int16QuantizedTensor)
    assert_int16_vs_fp32_reference(y_int, y_fp)


def _init_periodic_quantizers(m) -> None:
    m.input_quantizers[0] = Quantize((), 16, symmetric=True)
    m.output_quantizers[0] = Quantize((), 16, symmetric=True)
    m.input_quantizers[0].min = nn.Parameter(torch.tensor(-3.15))
    m.input_quantizers[0].max = nn.Parameter(torch.tensor(3.15))
    m.output_quantizers[0].min = nn.Parameter(torch.tensor(-1.0))
    m.output_quantizers[0].max = nn.Parameter(torch.tensor(1.0))


def _init_sqrt_quantizers(m) -> None:
    m.input_quantizers[0] = Quantize((), 16, symmetric=True)
    m.output_quantizers[0] = Quantize((), 16, symmetric=True)
    m.input_quantizers[0].min = nn.Parameter(torch.tensor(0.05))
    m.input_quantizers[0].max = nn.Parameter(torch.tensor(4.0))
    m.output_quantizers[0].min = nn.Parameter(torch.tensor(0.2))
    m.output_quantizers[0].max = nn.Parameter(torch.tensor(2.0))


def test_quantized_sin_int16_fixed_dispatch():
    m = QuantizedSin()
    _init_periodic_quantizers(m)
    x = torch.tensor([[-1.0, 0.0, 1.5, -2.5]], dtype=torch.float32)

    with quant_execution_mode(ExecutionMode.FP32_QDQ):
        y_q = m(x)
        y_fp = y_q.dequantize() if hasattr(y_q, "dequantize") else y_q

    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y_int = m(x)

    assert isinstance(y_int, Int16QuantizedTensor)
    assert_int16_vs_fp32_reference(y_int, y_fp, max_lsb=512.0, min_cosine=0.999)


def test_quantized_cos_int16_fixed_dispatch():
    m = QuantizedCos()
    _init_periodic_quantizers(m)
    x = torch.tensor([[0.5, -1.2, 2.0, -0.3]], dtype=torch.float32)

    with quant_execution_mode(ExecutionMode.FP32_QDQ):
        y_q = m(x)
        y_fp = y_q.dequantize() if hasattr(y_q, "dequantize") else y_q

    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y_int = m(x)

    assert isinstance(y_int, Int16QuantizedTensor)
    assert_int16_vs_fp32_reference(y_int, y_fp, max_lsb=512.0, min_cosine=0.999)


@pytest.mark.skipif(resolve_abc_lut_root() is None, reason="abc_lut-shuai not found")
def test_quantized_sqrt_int16_clz_dispatch():
    m = QuantizedSqrt()
    _init_sqrt_quantizers(m)
    x = torch.tensor([[0.25, 1.0, 2.25, 3.5]], dtype=torch.float32)

    with quant_execution_mode(ExecutionMode.FP32_QDQ):
        y_q = m(x)
        y_fp = y_q.dequantize() if hasattr(y_q, "dequantize") else y_q

    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y_int = m(x)

    assert isinstance(y_int, Int16QuantizedTensor)
    assert_int16_vs_fp32_reference(y_int, y_fp, max_lsb=2048.0, min_cosine=0.999)


def test_quantized_log_int16_fixed_dispatch():
    m = QuantizedLog()
    _init_sqrt_quantizers(m)
    m.input_quantizers[0].min = nn.Parameter(torch.tensor(0.1))
    x = torch.tensor([[0.5, 1.0, 2.0]], dtype=torch.float32)

    with quant_execution_mode(ExecutionMode.FP32_QDQ):
        y_q = m(x)
        y_fp = y_q.dequantize() if hasattr(y_q, "dequantize") else y_q

    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y_int = m(x)

    assert isinstance(y_int, Int16QuantizedTensor)
    assert_int16_vs_fp32_reference(y_int, y_fp, max_lsb=2048.0, min_cosine=0.999)


@pytest.mark.skipif(resolve_abc_lut_root() is None, reason="abc_lut-shuai not found")
def test_quantized_rsqrt_int16_clz_dispatch():
    m = QuantizedRSqrt()
    _init_sqrt_quantizers(m)
    x = torch.tensor([[0.25, 1.0, 4.0]], dtype=torch.float32)

    with quant_execution_mode(ExecutionMode.FP32_QDQ):
        y_q = m(x)
        y_fp = y_q.dequantize() if hasattr(y_q, "dequantize") else y_q

    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y_int = m(x)

    assert isinstance(y_int, Int16QuantizedTensor)
    assert_int16_vs_fp32_reference(y_int, y_fp, max_lsb=2048.0, min_cosine=0.999)


@pytest.mark.skipif(resolve_abc_lut_root() is None, reason="abc_lut-shuai not found")
def test_quantized_reciprocal_int16_clz_dispatch():
    m = QuantizedReciprocal()
    m.input_quantizers[0] = Quantize((), 16, symmetric=True)
    m.output_quantizers[0] = Quantize((), 16, symmetric=True)
    m.input_quantizers[0].min = nn.Parameter(torch.tensor(0.2))
    m.input_quantizers[0].max = nn.Parameter(torch.tensor(2.0))
    m.output_quantizers[0].min = nn.Parameter(torch.tensor(0.2))
    m.output_quantizers[0].max = nn.Parameter(torch.tensor(3.0))
    x = torch.tensor([[0.5, 1.0, 1.5]], dtype=torch.float32)

    with quant_execution_mode(ExecutionMode.FP32_QDQ):
        y_q = m(x)
        y_fp = y_q.dequantize() if hasattr(y_q, "dequantize") else y_q

    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y_int = m(x)

    assert isinstance(y_int, Int16QuantizedTensor)
    assert_int16_vs_fp32_reference(y_int, y_fp, max_lsb=16384.0, min_cosine=0.98)


@pytest.mark.skipif(resolve_abc_lut_root() is None, reason="abc_lut-shuai not found")
def test_quantized_square_int16_clz_dispatch():
    m = QuantizedSquare()
    _init_sqrt_quantizers(m)
    x = torch.tensor([[0.25, 1.0, 1.5]], dtype=torch.float32)

    with quant_execution_mode(ExecutionMode.FP32_QDQ):
        y_q = m(x)
        y_fp = y_q.dequantize() if hasattr(y_q, "dequantize") else y_q

    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y_int = m(x)

    assert isinstance(y_int, Int16QuantizedTensor)
    assert_int16_vs_fp32_reference(y_int, y_fp, max_lsb=8192.0, min_cosine=0.98)


@pytest.mark.skipif(resolve_abc_lut_root() is None, reason="abc_lut-shuai not found")
def test_quantized_square_8bit_int16_clz_aligns_op_grid_to_lut_grid():
    """MRNN uses W8A8 Square; CLZ LUT is fit on int16 grids and needs §3.0 input align."""

    m = QuantizedSquare()
    m.input_quantizers[0] = Quantize((), 8, symmetric=True)
    m.output_quantizers[0] = Quantize((), 8, symmetric=True)
    m.input_quantizers[0].min = nn.Parameter(torch.tensor(-2.0))
    m.input_quantizers[0].max = nn.Parameter(torch.tensor(2.0))
    m.output_quantizers[0].min = nn.Parameter(torch.tensor(0.0))
    m.output_quantizers[0].max = nn.Parameter(torch.tensor(4.0))
    x = torch.randn(4, 16, 8, dtype=torch.float32) * 0.5

    with quant_execution_mode(ExecutionMode.FP32_QDQ):
        m(x)

    with quant_execution_mode(ExecutionMode.FP32_QDQ):
        y_fp = m(x).dequantize()

    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y_int = m(x)

    assert isinstance(y_int, Int16QuantizedTensor)
    assert int((y_int.int_repr != 0).sum().item()) > 0
    assert_int16_vs_fp32_reference(y_int, y_fp, max_lsb=8192.0, min_cosine=0.98)


def test_quantized_exponential_int16_fixed_dispatch():
    m = QuantizedExponential()
    _init_periodic_quantizers(m)
    m.input_quantizers[0].min = nn.Parameter(torch.tensor(-2.0))
    m.input_quantizers[0].max = nn.Parameter(torch.tensor(2.0))
    m.output_quantizers[0].min = nn.Parameter(torch.tensor(0.0))
    m.output_quantizers[0].max = nn.Parameter(torch.tensor(4.0))
    x = torch.tensor([[-1.0, 0.0, 0.5]], dtype=torch.float32)

    with quant_execution_mode(ExecutionMode.FP32_QDQ):
        y_q = m(x)
        y_fp = y_q.dequantize() if hasattr(y_q, "dequantize") else y_q

    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y_int = m(x)

    assert isinstance(y_int, Int16QuantizedTensor)
    assert_int16_vs_fp32_reference(y_int, y_fp, max_lsb=2048.0, min_cosine=0.999)


def test_quantized_dropout_int16_requantizes_between_grids():
    m = QuantizedDropout(p=0.0).eval()
    m.input_quantizers[0] = Quantize((), 8, symmetric=True)
    m.output_quantizers[0] = Quantize((), 8, symmetric=True)
    m.input_quantizers[0].min = nn.Parameter(torch.tensor(-4.0))
    m.input_quantizers[0].max = nn.Parameter(torch.tensor(4.0))
    m.output_quantizers[0].min = nn.Parameter(torch.tensor(-1.0))
    m.output_quantizers[0].max = nn.Parameter(torch.tensor(1.0))
    x = torch.tensor([[-0.75, 0.0, 0.5, 0.9]], dtype=torch.float32)

    with quant_execution_mode(ExecutionMode.FP32_QDQ):
        y_q = m(x)
        y_fp = y_q.dequantize() if hasattr(y_q, "dequantize") else y_q

    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y_int = m(x)

    assert isinstance(y_int, Int16QuantizedTensor)
    assert_int16_vs_fp32_reference(y_int, y_fp, max_lsb=1.0, min_cosine=0.999)


def test_quantized_reshape_int16_fixed_dispatch_with_shape_tensor():
    m = QuantizedReshape()
    m.input_quantizers[0] = Quantize((), 8, symmetric=True)
    m.output_quantizers[0] = Quantize((), 8, symmetric=True)
    m.input_quantizers[0].min = nn.Parameter(torch.tensor(-2.0))
    m.input_quantizers[0].max = nn.Parameter(torch.tensor(2.0))
    m.output_quantizers[0].min = nn.Parameter(torch.tensor(-2.0))
    m.output_quantizers[0].max = nn.Parameter(torch.tensor(2.0))
    x = torch.tensor([[0.5, -0.25, 0.0], [0.75, -0.5, 0.25]], dtype=torch.float32)
    shape = torch.tensor([3, 2], dtype=torch.int64)

    with quant_execution_mode(ExecutionMode.FP32_QDQ):
        y_q = m(x, shape)
        y_fp = y_q.dequantize() if hasattr(y_q, "dequantize") else y_q

    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y_int = m(x, shape)

    assert isinstance(y_int, Int16QuantizedTensor)
    assert y_int.int_repr.shape == (3, 2)
    assert_int16_vs_fp32_reference(y_int, y_fp)


def test_int16_fixed_unsupported_module_raises_instead_of_float_fallback():
    m = QuantizedDropout(p=0.0)
    x = torch.tensor([1.0], dtype=torch.float32)

    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        with pytest.raises(RuntimeError, match="not implemented"):
            m(x)

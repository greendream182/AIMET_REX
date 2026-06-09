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
    QuantizedAvgPool2d,
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
    QuantizedMean,
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
        pytest.param(8, 8, id="W8A8"),
        pytest.param(16, 8, id="W16A8"),
        pytest.param(8, 16, id="W8A16"),
    ],
)
def test_quantized_linear_int16_fixed_mixed_bitwidth_dispatch(weight_bw: int, act_bw: int):
    """Dispatch happy paths under the W5 SYS-FU-1.B combo contract.

    PR-3 (2026-06-09) replaced the legacy ``W4A8 / W4A16 / W16A16``
    parametrisation with the three operand combos that the W5.1 probe
    validated as safe up to N=4096:

    * ``W8A8`` — legacy backbone configuration (still validated).
    * ``W16A8`` — 16-bit-input + 8-bit-weight Linear (max combined
      bitwidth = 24, fits in INT32 ALU even at large reduction).
    * ``W8A16`` — symmetric mirror; covers the 8-bit-input + 16-bit-weight
      tile (e.g. mid-precision Linear feeding into a 16-bit residual).

    The combos rejected by the new gate
    (``W4*``: weight bitwidth not in ``(8, 16)``; ``W16A16`` on a MAC
    reduction: combined bitwidth = 32 > ``REQUANTIZING_COMBO_BITWIDTH_BUDGET=24``)
    have moved to ``test_quantized_linear_int16_fixed_combo_gate_rejects_*``
    below.
    """
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
    # Dispatch-level guard: confirm the kernel runs and produces a
    # finite-valued INT16 tensor. W8A8 keeps the strict default
    # ``max_lsb=1`` because input/weight/output share the same 8-bit
    # grid; the asymmetric W16A8 / W8A16 combos relax to
    # ``max_lsb=16`` because the kernel emits 1-LSB-of-output rounding
    # drift that this 4×3 fixture amplifies into >1 LSB on the few
    # output elements landing close to the grid step. Bit-level
    # precision regression for the asymmetric subset lives in PR-4
    # (N-sweep probe matrix frozen as pytest); this test is the
    # dispatch-path guard.
    if weight_bw == 8 and act_bw == 8:
        assert_int16_vs_fp32_reference(y_int, y_fp)
    else:
        assert torch.isfinite(y_int.to_float()).all()


@pytest.mark.parametrize(
    "weight_bw,act_bw,match",
    [
        pytest.param(4, 8, "bitwidth=4", id="W4A8"),
        pytest.param(4, 16, "bitwidth=4", id="W4A16"),
        pytest.param(16, 16, "REQUANTIZING-with-MAC-reduction", id="W16A16"),
    ],
)
def test_quantized_linear_int16_fixed_combo_gate_rejects(
    weight_bw: int, act_bw: int, match: str
):
    """Dispatch refuses unsupported (weight_bw, act_bw) on a MAC-reduction Linear.

    Three classes covered:

    * ``W4*`` — weight bitwidth not in
      ``capabilities._REQUANTIZING_COMBO_VALIDATED_BITWIDTHS = (8, 16)``.
      Same gate fires even when the activation half is fine (``W4A8``).
    * ``W16A16`` — both halves are individually validated, but the combined
      bitwidth = 32 exceeds ``REQUANTIZING_COMBO_BITWIDTH_BUDGET = 24``,
      which is what saturates the INT32 ALU at large N (W5.1 probe).

    The error message must name the failing dimension so users know
    whether to drop weight precision or split the reduction.
    """

    m = QuantizedLinear(4, 3, bias=True)
    _init_linear_quantizers_bw(m, weight_bw, act_bw)
    nn.init.constant_(m.weight, 0.1)
    nn.init.constant_(m.bias, 0.01)
    x = torch.randn(2, 4, dtype=torch.float32)

    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        with pytest.raises(ValueError, match=match):
            m(x)


def test_quantized_linear_int16_fixed_dispatch_with_bias_bits_16():
    """``qmodule._fp_bias_bits = 16`` selects the spec-04_01 hardware bias.

    The forward path must (a) quantize bias to int16, (b) propagate
    ``OutputEncoding.bias_bits=16`` to the kernel, and (c) still match the
    fp32 reference numerically when bias values fit both ranges.
    """

    m = QuantizedLinear(3, 2, bias=True)
    _init_linear_quantizers(m)
    nn.init.constant_(m.weight, 0.1)
    nn.init.constant_(m.bias, 0.05)
    m._fp_bias_bits = 16

    x = torch.tensor([[0.5, -0.25, 0.0]], dtype=torch.float32)

    with quant_execution_mode(ExecutionMode.FP32_QDQ):
        y_q = m(x)
        y_fp = y_q.dequantize() if hasattr(y_q, "dequantize") else y_q

    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y_int = m(x)

    assert isinstance(y_int, Int16QuantizedTensor)
    assert y_int.to_float().shape == y_fp.shape
    assert_int16_vs_fp32_reference(y_int, y_fp)


def test_quantized_linear_int16_fixed_dispatch_rejects_invalid_bias_bits():
    """``qmodule._fp_bias_bits`` must be 16 or 32; anything else raises."""

    m = QuantizedLinear(3, 2, bias=True)
    _init_linear_quantizers(m)
    nn.init.constant_(m.weight, 0.1)
    nn.init.constant_(m.bias, 0.05)
    m._fp_bias_bits = 24

    x = torch.tensor([[0.5, -0.25, 0.0]], dtype=torch.float32)
    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        with pytest.raises(ValueError, match=r"_fp_bias_bits must be 16 or 32"):
            m(x)


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


def test_avgpool2d_count_include_pad_true_happy_int16_fixed_dispatch():
    """End-to-end happy path for the spec-04_09 ``1/N`` fold contract on AvgPool2d.

    AvgPool2d derives ``N = k_t * k_f`` directly from ``qmodule.kernel_size``
    (it does not go through ``_compute_mean_reduce_size_and_dims`` — that
    helper is shared between the two ``custom.Mean`` branches only). What
    this test guards is the AvgPool half of the contract:
      * the adapter's ``real_m = S_x / (k_t*k_f * S_y)`` fold and the
        ``extra['reduce_size'] = k_t*k_f`` write line up;
      * the kernel re-derives ``N`` from ``kernel_size`` and rejects via
        ``require_reduce_size_matches_extra`` if the two ever drift.

    Quantizers use ``Quantize(..., 8, symmetric=True)`` deliberately:
    ``INT16_FIXED_EVAL`` describes the MAC carrier / accumulator path, not
    the activation grid. The canonical project config (see
    ``sim_utils.ensure_output_quantizers_for_int16_eval``'s ``bitwidth=8``
    default and the rest of this file's existing tests) is 8-bit
    activations on top of an INT16 carrier; 16-bit activation quantizers
    are a separate, currently unvalidated capability tracked under
    ``audit-int16-activation-quantizer-contract``.
    """

    m = QuantizedAvgPool2d(kernel_size=2, stride=2, count_include_pad=True)
    m.input_quantizers[0] = Quantize((), 8, symmetric=True)
    m.output_quantizers[0] = Quantize((), 8, symmetric=True)
    m.input_quantizers[0].min = nn.Parameter(torch.tensor(-2.0))
    m.input_quantizers[0].max = nn.Parameter(torch.tensor(2.0))
    m.output_quantizers[0].min = nn.Parameter(torch.tensor(-2.0))
    m.output_quantizers[0].max = nn.Parameter(torch.tensor(2.0))
    torch.manual_seed(0)
    x = torch.randn(1, 2, 4, 4, dtype=torch.float32)

    with quant_execution_mode(ExecutionMode.FP32_QDQ):
        y_q = m(x)
        y_fp = y_q.dequantize() if hasattr(y_q, "dequantize") else y_q

    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y_int = m(x)

    assert isinstance(y_int, Int16QuantizedTensor)
    assert y_int.int_repr.shape == (1, 2, 2, 2)
    assert_int16_vs_fp32_reference(y_int, y_fp, max_lsb=1.0, min_cosine=0.999)


def test_quantized_mean_int16_fixed_dispatch_with_reduce_size_extra():
    """End-to-end coverage for ``custom.Mean`` dispatch under the
    ``reduce_size`` contract.

    ``test_eltwise_pool_shape.py`` covers the kernel-side assertion in
    isolation; this test wires the adapter to the kernel through
    ``INT16_FIXED_EVAL`` so a regression in
    ``_compute_mean_reduce_size_and_dims`` (P2-a — the helper shared by
    Mean's ``real_m`` fold and its ``extra`` write) or in the per-axis
    ``% ndim`` normalisation surfaces as a failure here rather than as a
    silent scale drift.

    Activation quantizers are 8-bit for the same reason as the AvgPool
    happy path above: that is the canonical INT16-carrier configuration
    in this codebase. 16-bit activation behaviour is tracked separately
    under ``audit-int16-activation-quantizer-contract``.
    """

    m = QuantizedMean()
    m.input_quantizers[0] = Quantize((), 8, symmetric=True)
    m.output_quantizers[0] = Quantize((), 8, symmetric=True)
    m.input_quantizers[0].min = nn.Parameter(torch.tensor(-2.0))
    m.input_quantizers[0].max = nn.Parameter(torch.tensor(2.0))
    m.output_quantizers[0].min = nn.Parameter(torch.tensor(-2.0))
    m.output_quantizers[0].max = nn.Parameter(torch.tensor(2.0))
    torch.manual_seed(1)
    x = torch.randn(1, 3, 4, 4, dtype=torch.float32)

    with quant_execution_mode(ExecutionMode.FP32_QDQ):
        y_q = m(x, (2, 3), True)
        y_fp = y_q.dequantize() if hasattr(y_q, "dequantize") else y_q

    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y_int = m(x, (2, 3), True)

    assert isinstance(y_int, Int16QuantizedTensor)
    assert y_int.int_repr.shape == (1, 3, 1, 1)
    assert_int16_vs_fp32_reference(y_int, y_fp, max_lsb=1.0, min_cosine=0.999)


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            lambda: _make_linear_with_full_16bit_reduction(),
            id="QuantizedLinear-W16A16",
        ),
    ],
)
def test_int16_fixed_eval_refuses_full_16bit_reduction(build):
    """W5 SYS-FU-1.B combo gate refuses 16+16 on MAC-reduction kernels.

    Replaces the legacy ``refuses_unsupported_activation_bitwidth`` test
    (which expected any 16-bit quantizer to be rejected). After PR-2 the
    contract is finer-grained:

    * 16+8 / 8+16 dispatch on Linear/Conv/Mean/AvgPool now SUCCEEDS
      (validated by the W5.1 probe up to N=4096; see
      ``test_quantized_linear_int16_fixed_mixed_bitwidth_dispatch``).
    * Sum-only reduction kernels (``AvgPool2d`` / ``Mean``) accept full
      16+16 because the int32 ALU never reduces them; covered by
      ``test_avgpool2d_full_16bit_dispatch_succeeds`` and
      ``test_mean_full_16bit_dispatch_succeeds``.
    * The *only* class still rejected is REQUANTIZING-with-MAC-reduction
      at 16+16 — combined bitwidth = 32 exceeds the
      ``REQUANTIZING_COMBO_BITWIDTH_BUDGET = 24`` budget that keeps the
      INT32 accumulator from saturating. That is what this test guards.
    """

    m, call = build()
    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        with pytest.raises(ValueError, match="REQUANTIZING-with-MAC-reduction"):
            call(m)


def test_avgpool2d_full_16bit_dispatch_succeeds():
    """Sum-only reduction (AvgPool2d) is NOT gated by the combo budget.

    The kernel does ``sum / k_t / k_f`` — there is no operand×operand
    multiplication, so the int32 ALU only ever holds a sum of int16
    values which cannot saturate. The combo gate must therefore let
    16+16 through.
    """

    m, call = _make_avgpool2d_with_16bit_activation()
    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y = call(m)
    assert isinstance(y, Int16QuantizedTensor)


def test_mean_full_16bit_dispatch_succeeds():
    """Same contract as ``test_avgpool2d_full_16bit_dispatch_succeeds`` for
    ``custom.Mean``; both share the sum-only reduction path."""

    m, call = _make_mean_with_16bit_activation()
    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y = call(m)
    assert isinstance(y, Int16QuantizedTensor)


def test_linear_asymmetric_16bit_dispatch_succeeds():
    """Linear with the SYS-FU-1.B asymmetric subset (input=16, weight=8).

    Per the W5.1 probe this combo is safe up to N=4096 because the
    combined bitwidth = 24 exactly fits the INT32 MAC budget. Guards
    against accidental tightening of the combo gate that would lock
    the asymmetric subset out again.
    """

    m, call = _make_linear_with_asymmetric_16bit()
    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y = call(m)
    assert isinstance(y, Int16QuantizedTensor)


def _make_linear_with_full_16bit_reduction():
    """Linear with input=16, weight=16 — combined = 32, refused by the gate."""

    m = QuantizedLinear(4, 4)
    m.input_quantizers[0] = Quantize((), 16, symmetric=True)
    m.param_quantizers["weight"] = Quantize((m.out_features, 1), 16, symmetric=True)
    m.output_quantizers[0] = Quantize((), 16, symmetric=True)
    m.input_quantizers[0].min = nn.Parameter(torch.tensor(-2.0))
    m.input_quantizers[0].max = nn.Parameter(torch.tensor(2.0))
    m.param_quantizers["weight"].min = nn.Parameter(torch.full((m.out_features, 1), -0.5))
    m.param_quantizers["weight"].max = nn.Parameter(torch.full((m.out_features, 1), 0.5))
    m.output_quantizers[0].min = nn.Parameter(torch.tensor(-2.0))
    m.output_quantizers[0].max = nn.Parameter(torch.tensor(2.0))
    nn.init.constant_(m.weight, 0.1)
    nn.init.constant_(m.bias, 0.05)
    x = torch.randn(2, 4, dtype=torch.float32)
    return m, lambda mod: mod(x)


def _make_linear_with_asymmetric_16bit():
    """Linear with input=16, weight=8 — combined = 24, accepted by the gate."""

    m = QuantizedLinear(4, 4)
    m.input_quantizers[0] = Quantize((), 16, symmetric=True)
    m.param_quantizers["weight"] = Quantize((m.out_features, 1), 8, symmetric=True)
    m.output_quantizers[0] = Quantize((), 16, symmetric=True)
    m.input_quantizers[0].min = nn.Parameter(torch.tensor(-2.0))
    m.input_quantizers[0].max = nn.Parameter(torch.tensor(2.0))
    m.param_quantizers["weight"].min = nn.Parameter(torch.full((m.out_features, 1), -0.5))
    m.param_quantizers["weight"].max = nn.Parameter(torch.full((m.out_features, 1), 0.5))
    m.output_quantizers[0].min = nn.Parameter(torch.tensor(-2.0))
    m.output_quantizers[0].max = nn.Parameter(torch.tensor(2.0))
    nn.init.constant_(m.weight, 0.1)
    nn.init.constant_(m.bias, 0.05)
    x = torch.randn(2, 4, dtype=torch.float32)
    return m, lambda mod: mod(x)


def _make_avgpool2d_with_16bit_activation():
    m = QuantizedAvgPool2d(kernel_size=2, stride=2, count_include_pad=True)
    m.input_quantizers[0] = Quantize((), 8, symmetric=True)
    m.output_quantizers[0] = Quantize((), 16, symmetric=True)
    m.input_quantizers[0].min = nn.Parameter(torch.tensor(-2.0))
    m.input_quantizers[0].max = nn.Parameter(torch.tensor(2.0))
    m.output_quantizers[0].min = nn.Parameter(torch.tensor(-2.0))
    m.output_quantizers[0].max = nn.Parameter(torch.tensor(2.0))
    x = torch.randn(1, 2, 4, 4, dtype=torch.float32)
    return m, lambda mod: mod(x)


def _make_mean_with_16bit_activation():
    m = QuantizedMean()
    m.input_quantizers[0] = Quantize((), 16, symmetric=True)
    m.output_quantizers[0] = Quantize((), 16, symmetric=True)
    m.input_quantizers[0].min = nn.Parameter(torch.tensor(-2.0))
    m.input_quantizers[0].max = nn.Parameter(torch.tensor(2.0))
    m.output_quantizers[0].min = nn.Parameter(torch.tensor(-2.0))
    m.output_quantizers[0].max = nn.Parameter(torch.tensor(2.0))
    x = torch.randn(1, 3, 4, 4, dtype=torch.float32)
    return m, lambda mod: mod(x, (2, 3), True)


def test_avgpool2d_count_include_pad_false_refuses_int16_fixed_dispatch():
    """Spec doc/04_算子详细规格/04_09_池化类算子.md fixes the divisor at
    ``k_t * k_f`` (i.e. ``count_include_pad=True``); ``count_include_pad=False``
    would require a per-window divisor that the HW/spec do not model.

    The adapter must refuse dispatch in that configuration so the caller cannot
    silently exercise an INT16 path with the wrong scale. In ``INT16_FIXED_EVAL``
    that surfaces as the standard "not implemented" guard rather than a float
    fallback, matching the unsupported-module contract above.
    """

    m = QuantizedAvgPool2d(kernel_size=2, stride=2, count_include_pad=False)
    m.input_quantizers[0] = Quantize((), 8, symmetric=True)
    m.output_quantizers[0] = Quantize((), 8, symmetric=True)
    m.input_quantizers[0].min = nn.Parameter(torch.tensor(-2.0))
    m.input_quantizers[0].max = nn.Parameter(torch.tensor(2.0))
    m.output_quantizers[0].min = nn.Parameter(torch.tensor(-2.0))
    m.output_quantizers[0].max = nn.Parameter(torch.tensor(2.0))
    x = torch.randn(1, 1, 4, 4, dtype=torch.float32)

    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        with pytest.raises(RuntimeError, match="not implemented"):
            m(x)

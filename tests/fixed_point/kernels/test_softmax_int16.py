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
from aimet_torch.fixed_point import ExecutionMode, quant_execution_mode  # noqa: E402
from aimet_torch.fixed_point.kernels.softmax import _integer_softmax_normalize, softmax_int16_pwl  # noqa: E402
from aimet_torch.fixed_point.metrics import assert_int16_vs_fp32_reference  # noqa: E402
from aimet_torch.fixed_point.tensor import Int16QuantizedTensor  # noqa: E402
from aimet_torch.v2.nn import QuantizedSoftmax  # noqa: E402
from aimet_torch.v2.quantization.affine import Quantize  # noqa: E402


def _init_softmax_quantizers(m: QuantizedSoftmax) -> None:
    m.input_quantizers[0] = Quantize((), 8, symmetric=True)
    m.output_quantizers[0] = Quantize((), 8, symmetric=True)
    m.input_quantizers[0].min = nn.Parameter(torch.tensor(-4.0))
    m.input_quantizers[0].max = nn.Parameter(torch.tensor(4.0))
    m.output_quantizers[0].min = nn.Parameter(torch.tensor(0.0))
    m.output_quantizers[0].max = nn.Parameter(torch.tensor(1.0))


def test_integer_softmax_normalize_sums_to_one_on_grid():
    exp_q = torch.tensor([[1000, 2000, 1000]], dtype=torch.int32)
    exp_sum = exp_q.sum(dim=-1, keepdim=True)
    q = _integer_softmax_normalize(
        exp_q,
        exp_sum,
        qmin=0,
        qmax=255,
        zero_point=torch.zeros(1, dtype=torch.int32),
    )
    probs = q.float() / 255.0
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=0.02)


def test_quantized_softmax_int16_pwl_kernel():
    m = QuantizedSoftmax(dim=-1)
    _init_softmax_quantizers(m)
    x = torch.tensor([[1.0, 2.0, 0.5, -1.0], [0.0, 0.1, -0.2, 0.3]], dtype=torch.float32)

    with quant_execution_mode(ExecutionMode.FP32_QDQ):
        y_q = m(x)
        y_fp = y_q.dequantize() if hasattr(y_q, "dequantize") else y_q

    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y_int = m(x)

    assert isinstance(y_int, Int16QuantizedTensor)
    assert_int16_vs_fp32_reference(y_int, y_fp, min_cosine=0.999)

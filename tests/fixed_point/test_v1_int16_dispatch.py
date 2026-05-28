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
pytest.importorskip("bokeh")
try:
    import aimet_common  # noqa: F401
except ImportError:
    pytest.skip("aimet_common is not installed", allow_module_level=True)

import torch.nn as nn  # noqa: E402

import aimet_torch.fixed_point.kernels  # noqa: F401, E402
from aimet_common.defs import QuantScheme, QuantizationDataType  # noqa: E402
from aimet_torch.fixed_point.metrics import assert_int16_vs_fp32_reference  # noqa: E402
from aimet_torch.fixed_point import (  # noqa: E402
    ExecutionMode,
    Int16QuantizedTensor,
    quant_execution_mode,
)
from aimet_torch.v1.qc_quantize_op import (  # noqa: E402
    QcQuantizeOpMode,
    StaticGridQuantWrapper,
)
from aimet_torch.v1.utils import create_encoding_from_dict  # noqa: E402

try:
    _V1_CPP_PROBE = StaticGridQuantWrapper(
        nn.Linear(1, 1, bias=False),
        weight_bw=8,
        activation_bw=8,
        round_mode="nearest",
        quant_scheme=QuantScheme.post_training_tf,
        is_output_quantized=True,
        is_symmetric=True,
        num_inputs=1,
        num_outputs=1,
        data_type=QuantizationDataType.int,
    )
    del _V1_CPP_PROBE
except RuntimeError as _exc:
    if "AimetTensorQuantizer" in str(_exc) or "Unable to initialize" in str(_exc):
        pytest.skip(
            f"v1 tensor quantizer C++ extension unavailable: {_exc}",
            allow_module_level=True,
        )
    raise


def _symmetric_tf_encoding(delta: float, offset: float, min_val: float, max_val: float):
    return create_encoding_from_dict(
        {
            "bitwidth": 8,
            "min": min_val,
            "max": max_val,
            "scale": delta,
            "offset": offset,
            "is_symmetric": "True",
        }
    )


def test_v1_static_grid_wrapper_int16_matches_qdq():
    linear = nn.Linear(3, 2, bias=True)
    wrap = StaticGridQuantWrapper(
        linear,
        weight_bw=8,
        activation_bw=8,
        round_mode="nearest",
        quant_scheme=QuantScheme.post_training_tf,
        is_output_quantized=True,
        is_symmetric=True,
        num_inputs=1,
        num_outputs=1,
        data_type=QuantizationDataType.int,
    )
    wrap.enable_input_quantizers(True)
    enc_in = _symmetric_tf_encoding(0.04, 0.0, -4.0, 4.0)
    enc_w = _symmetric_tf_encoding(0.02, 0.0, -0.5, 0.5)
    enc_out = _symmetric_tf_encoding(0.05, 0.0, -3.0, 3.0)
    wrap.input_quantizers[0].encoding = enc_in
    wrap.param_quantizers["weight"].encoding = enc_w
    wrap.output_quantizers[0].encoding = enc_out
    wrap.set_mode(QcQuantizeOpMode.ACTIVE)

    nn.init.constant_(linear.weight, 0.12)
    nn.init.constant_(linear.bias, 0.03)
    x = torch.tensor([[0.4, -0.2, 0.1]], dtype=torch.float32)

    with quant_execution_mode(ExecutionMode.FP32_QDQ):
        y_fp = wrap(x)

    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y_int = wrap(x)

    assert isinstance(y_int, Int16QuantizedTensor)
    assert y_int.to_float().shape == y_fp.shape
    assert_int16_vs_fp32_reference(y_int, y_fp)


def test_v1_int16_eval_does_not_use_fixed_path_in_analysis_mode():
    """INT16 fast path is only wired for ACTIVE; ANALYSIS must keep stats collection."""

    linear = nn.Linear(3, 2, bias=False)
    wrap = StaticGridQuantWrapper(
        linear,
        weight_bw=8,
        activation_bw=8,
        round_mode="nearest",
        quant_scheme=QuantScheme.post_training_tf,
        is_output_quantized=True,
        is_symmetric=True,
        num_inputs=1,
        num_outputs=1,
        data_type=QuantizationDataType.int,
    )
    wrap.enable_input_quantizers(True)
    wrap.input_quantizers[0].encoding = _symmetric_tf_encoding(0.04, 0.0, -4.0, 4.0)
    wrap.param_quantizers["weight"].encoding = _symmetric_tf_encoding(0.02, 0.0, -0.5, 0.5)
    wrap.output_quantizers[0].encoding = _symmetric_tf_encoding(0.05, 0.0, -3.0, 3.0)
    wrap.set_mode(QcQuantizeOpMode.ANALYSIS)
    nn.init.constant_(linear.weight, 0.1)
    x = torch.tensor([[0.1, 0.2, -0.1]], dtype=torch.float32)

    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y = wrap(x)

    assert y.shape == (1, 2)

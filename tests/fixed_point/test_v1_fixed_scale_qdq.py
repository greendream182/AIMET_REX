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
"""v1 StaticGrid wrapper: ``fixed_scale_qdq`` vs ``fp32_qdq``."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("bokeh")
try:
    import aimet_common  # noqa: F401
except ImportError:
    pytest.skip("aimet_common is not installed", allow_module_level=True)

import torch.nn as nn  # noqa: E402

from aimet_common.defs import QuantScheme, QuantizationDataType  # noqa: E402
from aimet_torch.fixed_point import ExecutionMode, quant_execution_mode  # noqa: E402
from aimet_torch.fixed_point.metrics import compute_pair_metrics  # noqa: E402
from aimet_torch.v1.qc_quantize_op import QcQuantizeOpMode, StaticGridQuantWrapper  # noqa: E402
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


def _build_active_wrapper():
    linear = nn.Linear(4, 3, bias=True)
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
    wrap.set_mode(QcQuantizeOpMode.ACTIVE)
    nn.init.uniform_(linear.weight, -0.2, 0.2)
    nn.init.uniform_(linear.bias, -0.05, 0.05)
    return wrap


def test_v1_static_grid_fixed_scale_matches_fp32_qdq():
    wrap = _build_active_wrapper()
    x = torch.randn(2, 4)

    with torch.no_grad():
        with quant_execution_mode(ExecutionMode.FP32_QDQ):
            y_fp32 = wrap(x)
        with quant_execution_mode(ExecutionMode.FIXED_SCALE_QDQ):
            y_fix = wrap(x)

    metrics = compute_pair_metrics(y_fp32, y_fix)
    assert metrics["cosine_similarity"] >= 0.9998, metrics
    assert (y_fp32.argmax(-1) == y_fix.argmax(-1)).all()


def test_v1_fixed_scale_unchanged_in_fp32_mode():
    wrap = _build_active_wrapper()
    x = torch.randn(1, 4)

    with torch.no_grad():
        with quant_execution_mode(ExecutionMode.FP32_QDQ):
            y1 = wrap(x)
            y2 = wrap(x)

    assert torch.equal(y1, y2)

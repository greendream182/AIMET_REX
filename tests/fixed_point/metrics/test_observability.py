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

import json
import os
from tempfile import NamedTemporaryFile

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from aimet_torch.fixed_point import (  # noqa: E402
    ExecutionMode,
    InputEncoding,
    OutputEncoding,
    fixed_point_tensor_bundle,
    generate_pwl_lut,
    get_quant_execution_mode,
    input_encoding_from_dict,
    input_encoding_to_dict,
    output_encoding_from_dict,
    output_encoding_to_dict,
    quant_execution_mode,
)
from aimet_torch.fixed_point.metrics import FixedPointProfiler, compare_modes  # noqa: E402
from aimet_torch.fixed_point.offline.lut_gen import pwl_lut_to_json_dict  # noqa: E402
from aimet_torch.fixed_point.tensor import Int16QuantizedTensor  # noqa: E402


class _ReturnsInt16(nn.Module):
    """Minimal module so :class:`FixedPointProfiler` sees an :class:`Int16QuantizedTensor` (no v2 stack)."""

    def forward(self, x: torch.Tensor) -> Int16QuantizedTensor:
        return Int16QuantizedTensor.from_float(
            torch.tanh(x),
            scale=torch.tensor(0.02, device=x.device, dtype=torch.float32),
            zero_point=torch.zeros((), dtype=torch.int32, device=x.device),
        )


class _DualModeLinearish(nn.Module):
    """FP32 path returns float; INT16 path returns a fixed-point carrier (smoke compare_modes)."""

    def forward(self, x: torch.Tensor):
        y = x * 1.1 + 0.01
        if get_quant_execution_mode() is ExecutionMode.INT16_FIXED_EVAL:
            return Int16QuantizedTensor.from_float(
                y,
                scale=torch.tensor(0.05, device=x.device, dtype=torch.float32),
                zero_point=torch.zeros((), dtype=torch.int32, device=x.device),
            )
        return y


def test_output_encoding_json_roundtrip():
    enc = OutputEncoding(
        scale=torch.tensor([0.02, 0.03], dtype=torch.float32),
        zero_point=torch.tensor([1, 2], dtype=torch.int32),
        qmin=-32768,
        qmax=32767,
        multiplier=torch.tensor([1200, 1300], dtype=torch.int16),
        rshift=torch.tensor([14, 15], dtype=torch.int8),
    )
    data = output_encoding_to_dict(enc)
    restored = output_encoding_from_dict(data)
    assert torch.allclose(restored.scale, enc.scale)
    assert torch.equal(restored.zero_point, enc.zero_point)
    assert torch.equal(restored.multiplier, enc.multiplier)
    assert torch.equal(restored.rshift, enc.rshift)
    round_input = input_encoding_from_dict(input_encoding_to_dict(enc))
    assert torch.allclose(round_input.scale, enc.scale)


def test_fixed_point_bundle_includes_pwl_sidecar():
    out = OutputEncoding(
        scale=torch.tensor(0.05),
        zero_point=torch.tensor(0, dtype=torch.int32),
        qmin=-32768,
        qmax=32767,
        multiplier=torch.tensor(1000, dtype=torch.int16),
        rshift=torch.tensor(12, dtype=torch.int8),
    )
    ienc = InputEncoding(
        scale=torch.tensor(0.04),
        zero_point=torch.tensor(0, dtype=torch.int32),
        qmin=-32768,
        qmax=32767,
    )
    pwl = generate_pwl_lut(torch.sigmoid, ienc, out, num_segments=4, samples_per_segment=8)
    sidecar = pwl_lut_to_json_dict(pwl, func_name="sigmoid", input_encoding=ienc, output_encoding=out)
    bundle = fixed_point_tensor_bundle(layer_name="act_0", output_encoding=out, pwl_json=sidecar)
    assert bundle["layer"] == "act_0"
    assert bundle["pwl"]["sigmoid"]["num_segments"] == 4
    re_out = output_encoding_from_dict(bundle["output_encoding"])
    assert torch.allclose(re_out.scale, out.scale)


def test_to_float_warns_in_int16_eval_without_debug_allow():
    q = Int16QuantizedTensor(
        int_repr=torch.zeros(1, dtype=torch.int16),
        scale=torch.ones(1, dtype=torch.float32),
        zero_point=torch.zeros(1, dtype=torch.int32),
    )
    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        with pytest.warns(UserWarning, match="FixedPointProfiler"):
            q.to_float()


def test_to_float_strict_env_raises():
    q = Int16QuantizedTensor(
        int_repr=torch.zeros(1, dtype=torch.int16),
        scale=torch.ones(1, dtype=torch.float32),
        zero_point=torch.zeros(1, dtype=torch.int32),
    )
    old = os.environ.get("AIMET_RX_INT16_STRICT_TO_FLOAT")
    try:
        os.environ["AIMET_RX_INT16_STRICT_TO_FLOAT"] = "1"
        with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
            with pytest.raises(RuntimeError, match="without FixedPointProfiler"):
                q.to_float()
    finally:
        if old is None:
            os.environ.pop("AIMET_RX_INT16_STRICT_TO_FLOAT", None)
        else:
            os.environ["AIMET_RX_INT16_STRICT_TO_FLOAT"] = old


def test_fixed_point_profiler_stats_and_json():
    net = _ReturnsInt16()
    x = torch.tensor([[0.1, -0.1]], dtype=torch.float32)

    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        prof = FixedPointProfiler(net)
        with prof:
            net(x)
        stats = prof.to_dict()

    assert "<root>" in stats
    assert "saturation_ratio" in stats["<root>"]

    with NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        path = tmp.name
    try:
        with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
            prof = FixedPointProfiler(net)
            with prof:
                net(x)
            prof.to_json(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert "<root>" in data
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def test_profiler_reports_multiplier_for_quantized_linear():
    pytest.importorskip("onnxscript")
    from aimet_torch.v2.nn import QuantizedLinear  # noqa: E402
    from aimet_torch.v2.quantization.affine import Quantize  # noqa: E402

    m = QuantizedLinear(2, 2)
    m.input_quantizers[0] = Quantize((), 8, symmetric=True)
    m.param_quantizers["weight"] = Quantize((2, 1), 8, symmetric=True)
    m.output_quantizers[0] = Quantize((), 8, symmetric=True)
    m.input_quantizers[0].min = nn.Parameter(torch.tensor(-2.0))
    m.input_quantizers[0].max = nn.Parameter(torch.tensor(2.0))
    m.param_quantizers["weight"].min = nn.Parameter(torch.full((2, 1), -0.5))
    m.param_quantizers["weight"].max = nn.Parameter(torch.full((2, 1), 0.5))
    m.output_quantizers[0].min = nn.Parameter(torch.tensor(-2.0))
    m.output_quantizers[0].max = nn.Parameter(torch.tensor(2.0))
    nn.init.constant_(m.weight, 0.1)
    nn.init.constant_(m.bias, 0.0)

    import aimet_torch.fixed_point.kernels  # noqa: F401
    from aimet_torch.fixed_point.metrics import FixedPointProfiler  # noqa: E402

    x = torch.tensor([[0.1, -0.1]], dtype=torch.float32)
    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        with FixedPointProfiler(nn.Sequential(m)) as prof:
            m(x)
        stats = prof.to_dict()
    layer_key = next(k for k in stats if k != "<root>" or len(stats) == 1)
    if "1" in stats:
        layer_key = "1"
    assert stats[layer_key]["multiplier"] != 0 or stats[layer_key]["rshift"] != 0


def test_compare_modes_fp32_vs_int16_smoke():
    model = _DualModeLinearish()
    x = torch.tensor([[0.5, -0.2, 0.1]], dtype=torch.float32)
    out = compare_modes(
        model,
        x,
        [ExecutionMode.FP32_QDQ, ExecutionMode.INT16_FIXED_EVAL],
        metrics=("max_abs_error", "max_error_lsb", "cosine_similarity"),
    )
    key = "fp32_qdq_vs_int16_fixed_eval"
    assert key in out["pairwise"]


def test_compare_modes_default_list_includes_fixed_scale():
    from aimet_torch.fixed_point.metrics import DEFAULT_COMPARE_MODES

    assert ExecutionMode.FIXED_SCALE_QDQ in DEFAULT_COMPARE_MODES
    out = compare_modes(
        _DualModeLinearish(),
        torch.tensor([[0.5, -0.2, 0.1]], dtype=torch.float32),
        metrics=("cosine_similarity",),
    )
    assert "fp32_qdq_vs_fixed_scale_qdq" in out["pairwise"]
    assert out["pairwise"]["fp32_qdq_vs_fixed_scale_qdq"]["cosine_similarity"] <= 1.0


def test_compare_modes_per_layer_and_csv(tmp_path):
    from aimet_torch.fixed_point.metrics.compare import write_per_layer_csv

    model = _DualModeLinearish()
    x = torch.tensor([[0.5, -0.2, 0.1]], dtype=torch.float32)
    out = compare_modes(
        model,
        x,
        [ExecutionMode.FP32_QDQ, ExecutionMode.INT16_FIXED_EVAL],
        metrics=("cosine_similarity", "max_abs_error"),
        include_per_layer=True,
    )
    assert "per_layer" in out
    assert len(out["per_layer"]) >= 1
    csv_path = tmp_path / "per_layer.csv"
    write_per_layer_csv(out, csv_path)
    text = csv_path.read_text(encoding="utf-8")
    assert "layer" in text
    assert "cosine_similarity" in text

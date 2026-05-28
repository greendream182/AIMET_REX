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
"""Phase B: INT16 sidecar export / load / runtime consistency."""

import json
import os
from tempfile import TemporaryDirectory

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxscript")
import torch.nn as nn  # noqa: E402

import aimet_torch.fixed_point.kernels  # noqa: F401, E402
from aimet_torch.fixed_point.export import (  # noqa: E402
    INT16_SIDECAR_VERSION,
    attach_int16_sidecar_to_model,
    attach_onnx_name_hints,
    build_int16_sidecar_document,
    build_onnx_name_hints_for_layers,
    compare_sidecar_with_model,
    default_int16_sidecar_path,
    export_int16_sidecar_json,
    layers_from_sidecar,
    load_int16_sidecar_json,
)
from aimet_torch.fixed_point import ExecutionMode, quant_execution_mode  # noqa: E402
from aimet_torch.fixed_point.offline.clz_gen import resolve_abc_lut_root  # noqa: E402
from aimet_torch.v2.nn import QuantizedConv2d, QuantizedLinear, QuantizedSigmoid  # noqa: E402
from aimet_torch.v2.nn.modules.custom import (  # noqa: E402
    QuantizedReciprocal,
    QuantizedSin,
    QuantizedSqrt,
    QuantizedSquare,
)
from aimet_torch.v2.quantization.affine import Quantize  # noqa: E402


def _init_linear(m: QuantizedLinear, in_range=2.0, weight_range=0.5, out_range=2.0):
    m.input_quantizers[0] = Quantize((), 8, symmetric=True)
    m.param_quantizers["weight"] = Quantize((m.out_features, 1), 8, symmetric=True)
    m.output_quantizers[0] = Quantize((), 8, symmetric=True)
    m.input_quantizers[0].min = nn.Parameter(torch.tensor(-in_range))
    m.input_quantizers[0].max = nn.Parameter(torch.tensor(in_range))
    m.param_quantizers["weight"].min = nn.Parameter(
        torch.full((m.out_features, 1), -weight_range)
    )
    m.param_quantizers["weight"].max = nn.Parameter(
        torch.full((m.out_features, 1), weight_range)
    )
    m.output_quantizers[0].min = nn.Parameter(torch.tensor(-out_range))
    m.output_quantizers[0].max = nn.Parameter(torch.tensor(out_range))


def _init_unary(m: nn.Module, in_range=2.0, out_min=-2.0, out_max=2.0):
    m.input_quantizers[0] = Quantize((), 8, symmetric=True)
    m.output_quantizers[0] = Quantize((), 8, symmetric=True)
    m.input_quantizers[0].min = nn.Parameter(torch.tensor(-in_range))
    m.input_quantizers[0].max = nn.Parameter(torch.tensor(in_range))
    m.output_quantizers[0].min = nn.Parameter(torch.tensor(out_min))
    m.output_quantizers[0].max = nn.Parameter(torch.tensor(out_max))


def test_export_sidecar_linear_has_multiplier_rshift():
    model = nn.Sequential(QuantizedLinear(3, 2))
    _init_linear(model[0])
    nn.init.constant_(model[0].weight, 0.1)
    nn.init.constant_(model[0].bias, 0.0)

    doc = build_int16_sidecar_document(model)
    assert doc["version"] == INT16_SIDECAR_VERSION
    assert doc["layer_count"] == 1
    layer = doc["layers"]["0"]
    enc = layer["output_encoding"]
    assert "multiplier" in enc
    assert "rshift" in enc


def test_compare_sidecar_ok_when_lut_attached_from_file():
    model = nn.Sequential(
        QuantizedLinear(3, 3),
        QuantizedSigmoid(),
    )
    _init_linear(model[0], in_range=2.0, weight_range=0.5, out_range=4.0)
    _init_unary(model[1], in_range=4.0, out_min=0.0, out_max=1.0)
    nn.init.constant_(model[0].weight, 0.15)

    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "m.int16.json")
        export_int16_sidecar_json(model, path)
        loaded = load_int16_sidecar_json(path)

    attach_int16_sidecar_to_model(model, loaded)
    report = compare_sidecar_with_model(model, loaded)
    assert report["ok"], report


def test_export_load_compare_roundtrip():
    model = nn.Sequential(
        QuantizedLinear(3, 3),
        QuantizedSigmoid(),
    )
    _init_linear(model[0], in_range=2.0, weight_range=0.5, out_range=4.0)
    _init_unary(model[1], in_range=4.0, out_min=0.0, out_max=1.0)
    nn.init.constant_(model[0].weight, 0.15)
    nn.init.constant_(model[0].bias, 0.0)

    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.int16.json")
        export_int16_sidecar_json(model, path)
        loaded = load_int16_sidecar_json(path)

    report = compare_sidecar_with_model(model, loaded)
    assert report["ok"], report
    assert "pwl" in loaded["layers"]["1"]
    pwl_block = loaded["layers"]["1"]["pwl"]
    fn_block = next(iter(pwl_block.values()))
    assert "quality" in fn_block
    quality = fn_block["quality"]
    assert quality["status"] == "PASS"
    assert set(quality["metrics"]) >= {
        "max_lsb",
        "p99_lsb",
        "p999_lsb",
        "rmse_lsb",
        "cosine_similarity",
    }
    assert quality["limits"]["min_cosine_similarity"] > 0.0


def test_layers_from_sidecar_rebuilds_output_encoding():
    model = nn.Sequential(QuantizedLinear(2, 2))
    _init_linear(model[0])
    doc = build_int16_sidecar_document(model)
    encodings = layers_from_sidecar(doc)
    assert "0" in encodings
    assert encodings["0"].multiplier is not None
    assert encodings["0"].rshift is not None


def test_onnx_name_hints_from_aimet_encodings(tmp_path):
    enc_path = tmp_path / "m.encodings"
    enc_path.write_text(
        json.dumps(
            {
                "version": "2.0.0",
                "encodings": [
                    {"name": "0.input", "bitwidth": 8},
                    {"name": "0.output", "bitwidth": 8},
                    {"name": "1.output", "bitwidth": 8},
                ],
            }
        ),
        encoding="utf-8",
    )
    hints = build_onnx_name_hints_for_layers(["0", "1"], str(enc_path))
    assert "0.output" in hints["0"]
    assert "1.output" in hints["1"]

    model = nn.Sequential(QuantizedLinear(2, 2))
    _init_linear(model[0])
    doc = build_int16_sidecar_document(model, aimet_encoding_path=str(enc_path))
    assert doc["onnx_name_hints_attached"] is True
    assert "onnx_tensor_names" in doc["layers"]["0"]


def test_default_int16_sidecar_path():
    assert default_int16_sidecar_path("/tmp/out", "net").endswith("net.int16.json")


def test_sidecar_matches_int16_forward_output_scale():
    """Exported output scale must match carrier scale after INT16 eval forward."""

    from aimet_torch.fixed_point import Int16QuantizedTensor  # noqa: E402

    model = QuantizedLinear(3, 2)
    _init_linear(model)
    nn.init.constant_(model.weight, 0.12)
    nn.init.constant_(model.bias, 0.01)
    x = torch.tensor([[0.3, -0.1, 0.2]], dtype=torch.float32)

    doc = build_int16_sidecar_document(nn.Sequential(model))
    sidecar_scale = doc["layers"]["0"]["output_encoding"]["scale"]

    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y = model(x)
    assert isinstance(y, Int16QuantizedTensor)
    runtime_scale = float(y.scale.reshape(-1)[0].item())
    if isinstance(sidecar_scale, list):
        assert abs(sidecar_scale[0] - runtime_scale) < 1e-4
    else:
        assert abs(sidecar_scale - runtime_scale) < 1e-4


def test_export_sidecar_sin_includes_pwl_and_phase_fold():
    model = nn.Sequential(QuantizedSin())
    _init_unary(model[0], in_range=3.15, out_min=-1.0, out_max=1.0)
    doc = build_int16_sidecar_document(model)
    layer = doc["layers"]["0"]
    assert layer["op"] == "Sin"
    assert layer.get("phase_fold") == "sin"
    assert "pwl" in layer
    sin_block = layer["pwl"]["sin"]
    assert sin_block.get("phase_fold") == "sin"
    assert "pwl_input_encoding" in sin_block
    assert sin_block["num_segments"] == 16


@pytest.mark.skipif(resolve_abc_lut_root() is None, reason="abc_lut-shuai not found")
def test_export_sidecar_sqrt_includes_clz_lut():
    model = nn.Sequential(QuantizedSqrt())
    _init_unary(model[0], in_range=4.0, out_min=0.2, out_max=2.0)
    model[0].input_quantizers[0].min = nn.Parameter(torch.tensor(0.05))
    doc = build_int16_sidecar_document(model)
    layer = doc["layers"]["0"]
    assert layer["op"] == "Sqrt"
    assert "clz" in layer
    assert "sqrt" in layer["clz"]
    assert "segments" in layer["clz"]["sqrt"]


@pytest.mark.skipif(resolve_abc_lut_root() is None, reason="abc_lut-shuai not found")
def test_export_sidecar_reciprocal_includes_clz_lut():
    model = nn.Sequential(QuantizedReciprocal())
    model[0].input_quantizers[0] = Quantize((), 16, symmetric=True)
    model[0].output_quantizers[0] = Quantize((), 16, symmetric=True)
    model[0].input_quantizers[0].min = nn.Parameter(torch.tensor(0.2))
    model[0].input_quantizers[0].max = nn.Parameter(torch.tensor(2.0))
    model[0].output_quantizers[0].min = nn.Parameter(torch.tensor(0.2))
    model[0].output_quantizers[0].max = nn.Parameter(torch.tensor(3.0))
    model(torch.tensor([[0.5, 1.0]], dtype=torch.float32))
    doc = build_int16_sidecar_document(model)
    layer = doc["layers"]["0"]
    assert layer["op"] == "Reciprocal"
    assert "clz" in layer
    assert "reciprocal" in layer["clz"]
    assert "segments" in layer["clz"]["reciprocal"]
    out_scale = float(layer["clz"]["reciprocal"]["quantization"]["output"]["scale"])
    assert out_scale < 1.0
    assert "export_metrics" in layer["clz"]["reciprocal"]
    assert layer["clz"]["reciprocal"]["export_metrics"]["clz_x_min"] == pytest.approx(0.2)


@pytest.mark.skipif(resolve_abc_lut_root() is None, reason="abc_lut-shuai not found")
def test_export_sidecar_square_includes_clz_power2():
    model = nn.Sequential(QuantizedSquare())
    model[0].input_quantizers[0] = Quantize((), 16, symmetric=True)
    model[0].output_quantizers[0] = Quantize((), 16, symmetric=True)
    model[0].input_quantizers[0].min = nn.Parameter(torch.tensor(0.05))
    model[0].input_quantizers[0].max = nn.Parameter(torch.tensor(4.0))
    model[0].output_quantizers[0].min = nn.Parameter(torch.tensor(0.2))
    model[0].output_quantizers[0].max = nn.Parameter(torch.tensor(2.0))
    model(torch.tensor([[0.25, 1.0]], dtype=torch.float32))
    doc = build_int16_sidecar_document(model)
    layer = doc["layers"]["0"]
    assert layer["op"] == "Square"
    assert "clz" in layer
    assert "power_2" in layer["clz"]
    assert "segments" in layer["clz"]["power_2"]

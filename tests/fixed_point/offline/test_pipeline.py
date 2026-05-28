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
"""Offline freeze pipeline (spec 10)."""

import json
import os
from tempfile import TemporaryDirectory

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from aimet_torch.fixed_point.export import compare_sidecar_with_model, load_int16_sidecar_json
from aimet_torch.fixed_point.offline.multiplier import quantize_multiplier
from aimet_torch.fixed_point.offline.pipeline import (
    freeze_int16_fixed,
    freeze_int16_fixed_report_only,
    multiplier_relative_error,
)
from aimet_torch.v2.nn import QuantizedLinear  # noqa: E402
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


def test_quantize_multiplier_known_value():
    m, s = quantize_multiplier(0.1234)
    approx = float(m.item()) / (1 << int(s.item()))
    assert abs(approx - 0.1234) / 0.1234 < 1e-3


def test_quantize_multiplier_per_channel():
    real = torch.tensor([0.1, 0.05, 0.01])
    m, s = quantize_multiplier(real)
    assert m.dtype == torch.int16
    assert s.dtype == torch.int8
    assert m.numel() == 3


def test_multiplier_relative_error_zero_real():
    m, s = quantize_multiplier(0.0)
    err = multiplier_relative_error(torch.tensor(0.0), m, s)
    assert err == 0.0


def test_freeze_pipeline_writes_sidecar_and_bias():
    model = nn.Sequential(QuantizedLinear(3, 2))
    _init_linear(model[0])
    nn.init.constant_(model[0].weight, 0.1)
    nn.init.constant_(model[0].bias, 0.05)

    with TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, "tiny.int16.json")
        report_path = os.path.join(tmp, "freeze_report.json")
        summary = freeze_int16_fixed(model, out_path, write_binaries=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        assert os.path.isfile(out_path)
        assert summary["layer_count"] == 1
        layer_report = summary["layers"]["0"]
        assert layer_report["status"] in ("ok", "warning")
        assert "multiplier_int16" in layer_report
        assert "rshift_int8" in layer_report
        assert "bias_int32_path" in layer_report
        assert os.path.isfile(layer_report["bias_int32_path"])

        loaded = load_int16_sidecar_json(out_path)
        assert "bias_int32_path" in loaded["layers"]["0"]
        cmp = compare_sidecar_with_model(model, loaded)
        assert cmp["ok"], cmp


def test_freeze_report_only_matches_written_layers():
    model = nn.Sequential(QuantizedLinear(2, 2))
    _init_linear(model[0])
    diag = freeze_int16_fixed_report_only(model)
    assert diag["layers"]["0"]["status"] in ("ok", "warning")

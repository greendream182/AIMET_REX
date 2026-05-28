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
"""Smoke tests for :func:`per_layer_chained_cosine`.

The function is the chained-accumulation sibling of
:func:`per_layer_isolated_cosine`; tests here mirror that style on the
same two-layer toy network so they stay CPU-only and fast.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

import aimet_torch.fixed_point.kernels  # noqa: F401, E402  pylint: disable=unused-import
from aimet_torch.fixed_point import ExecutionMode  # noqa: E402
from aimet_torch.fixed_point.metrics import per_layer_chained_cosine  # noqa: E402
from aimet_torch.v2.nn import QuantizedConv2d, QuantizedDropout  # noqa: E402
from aimet_torch.v2.quantization.affine import Quantize  # noqa: E402


def _init_conv2d_quantizers(
    m: QuantizedConv2d,
    *,
    in_range: float = 2.0,
    out_range: float = 4.0,
    w_range: float = 0.3,
) -> None:
    oc = m.out_channels
    m.input_quantizers[0] = Quantize((), 8, symmetric=True)
    m.param_quantizers["weight"] = Quantize((oc, 1, 1, 1), 8, symmetric=True)
    m.output_quantizers[0] = Quantize((), 8, symmetric=True)
    m.input_quantizers[0].min = nn.Parameter(torch.tensor(-in_range))
    m.input_quantizers[0].max = nn.Parameter(torch.tensor(in_range))
    m.param_quantizers["weight"].min = nn.Parameter(torch.full((oc, 1, 1, 1), -w_range))
    m.param_quantizers["weight"].max = nn.Parameter(torch.full((oc, 1, 1, 1), w_range))
    m.output_quantizers[0].min = nn.Parameter(torch.tensor(-out_range))
    m.output_quantizers[0].max = nn.Parameter(torch.tensor(out_range))


def _build_two_layer_sim() -> nn.Sequential:
    torch.manual_seed(0)
    conv1 = QuantizedConv2d(3, 4, kernel_size=3, padding=1, bias=True)
    conv2 = QuantizedConv2d(4, 2, kernel_size=3, padding=1, bias=True)
    _init_conv2d_quantizers(conv1, out_range=4.0)
    _init_conv2d_quantizers(conv2, in_range=4.0, out_range=4.0)
    nn.init.normal_(conv1.weight, std=0.1)
    nn.init.normal_(conv2.weight, std=0.1)
    nn.init.zeros_(conv1.bias)
    nn.init.zeros_(conv2.bias)
    return nn.Sequential(conv1, conv2)


def test_chained_self_compare_returns_unit_cosine():
    """``ref==cand`` ⇒ every layer cosine == 1.0 (and Tier-1 == zero noise).

    Tightest canary for hook/clone plumbing: any drift on cached outputs
    would knock the cosine below 1.
    """

    model = _build_two_layer_sim()
    x = torch.randn(1, 3, 8, 8)

    rows = per_layer_chained_cosine(
        model,
        x,
        ref_mode=ExecutionMode.FP32_QDQ,
        cand_mode=ExecutionMode.FP32_QDQ,
    )

    assert len(rows) == 2
    for row in rows:
        assert row["cosine"] == pytest.approx(1.0, abs=1e-6)
        assert row["rmse"] == pytest.approx(0.0, abs=1e-6)
        assert math.isinf(row["sqnr_db"]) or row["sqnr_db"] > 100


def test_chained_row_schema_includes_tier1_metrics():
    """Row keys form a stable contract for dashboards / CI parsers."""

    model = _build_two_layer_sim()
    x = torch.randn(1, 3, 8, 8)

    rows = per_layer_chained_cosine(
        model,
        x,
        ref_mode=ExecutionMode.FP32_QDQ,
        cand_mode=ExecutionMode.FIXED_SCALE_QDQ,
    )

    required = {
        "module",
        "cosine",
        "shape",
        "ref_rms",
        "max_abs_err",
        "norm_max_err",
        "rmse",
        "sqnr_db",
        "p99_abs_err",
    }
    for row in rows:
        missing = required - row.keys()
        assert not missing, f"missing keys in row: {missing} (row={row})"


def test_chained_sorted_ascending_and_top_k():
    model = _build_two_layer_sim()
    x = torch.randn(1, 3, 8, 8)

    rows = per_layer_chained_cosine(
        model,
        x,
        ref_mode=ExecutionMode.FP32_QDQ,
        cand_mode=ExecutionMode.FIXED_SCALE_QDQ,
    )

    cosines = [r["cosine"] for r in rows]
    assert cosines == sorted(cosines)

    truncated = per_layer_chained_cosine(
        model,
        x,
        ref_mode=ExecutionMode.FP32_QDQ,
        cand_mode=ExecutionMode.FIXED_SCALE_QDQ,
        top_k=1,
    )
    assert len(truncated) == 1
    assert truncated[0]["cosine"] == cosines[0]


def test_chained_dequantizes_quantized_tensor_outputs():
    """Chained hooks must compare dequantized values, not Tensor-subclass payloads."""

    torch.manual_seed(3)
    conv = QuantizedConv2d(3, 4, kernel_size=3, padding=1, bias=True)
    dropout = QuantizedDropout(p=0.0).eval()
    _init_conv2d_quantizers(conv, out_range=4.0)
    dropout.output_quantizers[0] = Quantize((), 8, symmetric=True)
    dropout.output_quantizers[0].min = nn.Parameter(torch.tensor(-4.0))
    dropout.output_quantizers[0].max = nn.Parameter(torch.tensor(4.0))
    nn.init.normal_(conv.weight, std=0.1)
    nn.init.zeros_(conv.bias)
    model = nn.Sequential(conv, dropout)
    x = torch.randn(1, 3, 8, 8)

    rows = per_layer_chained_cosine(
        model,
        x,
        ref_mode=ExecutionMode.FP32_QDQ,
        cand_mode=ExecutionMode.FP32_QDQ,
        module_filter=lambda name, _module: name == "1",
    )

    assert len(rows) == 1
    assert rows[0]["cosine"] == pytest.approx(1.0, abs=1e-6), rows
    assert rows[0]["rmse"] == pytest.approx(0.0, abs=1e-6), rows


def test_chained_handles_inplace_relu_downstream():
    """Inplace ReLU between probed layers must not corrupt the captured outputs.

    Without the defensive ``.clone()`` in the chained hook, the cached
    conv output would silently morph into the post-ReLU values and the
    self-compare cosine would dip below 1.
    """

    torch.manual_seed(1)
    conv = QuantizedConv2d(3, 4, kernel_size=3, padding=1, bias=True)
    _init_conv2d_quantizers(conv, out_range=4.0)
    nn.init.normal_(conv.weight, std=0.1)
    nn.init.zeros_(conv.bias)
    model = nn.Sequential(conv, nn.ReLU6(inplace=True))
    x = torch.randn(1, 3, 8, 8)

    rows = per_layer_chained_cosine(
        model,
        x,
        ref_mode=ExecutionMode.FP32_QDQ,
        cand_mode=ExecutionMode.FP32_QDQ,
    )

    assert len(rows) == 1
    assert rows[0]["cosine"] == pytest.approx(1.0, abs=1e-6)


def test_chained_p99_abs_err_on_large_activation():
    """MobileNet-scale activations must not trip ``torch.quantile`` size limits."""

    from aimet_torch.fixed_point.metrics.accuracy import p99_abs_error

    huge = torch.randn(16, 96, 56, 56)
    assert p99_abs_error(huge) >= 0.0

    model = _build_two_layer_sim()
    rows = per_layer_chained_cosine(
        model,
        torch.randn(16, 3, 56, 56),
        ref_mode=ExecutionMode.FP32_QDQ,
        cand_mode=ExecutionMode.FIXED_SCALE_QDQ,
        top_k=1,
    )
    assert rows and "p99_abs_err" in rows[0]

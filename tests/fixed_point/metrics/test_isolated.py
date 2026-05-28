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
"""Smoke tests for :func:`per_layer_isolated_cosine`.

These tests target the *plumbing*: hooks fire on every quantized leaf,
inputs are cached cleanly (no in-place corruption), the candidate module is
re-invoked under the requested mode, and the row order is descending by
quantization noise. They run on a tiny Conv→Conv toy network so they stay
CPU-only and well under one second.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

import aimet_torch.fixed_point.kernels  # noqa: F401, E402  pylint: disable=unused-import
from aimet_torch.fixed_point import ExecutionMode  # noqa: E402
from aimet_torch.fixed_point.metrics import per_layer_isolated_cosine  # noqa: E402
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
    """Two stacked ``QuantizedConv2d`` modules with realistic, narrow ranges."""

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


def test_isolated_self_compare_returns_unit_cosine():
    """Reference vs reference must always yield cosine == 1.0 per layer.

    This is the canary that pins down hook plumbing & input caching: any
    bug that lets the cached input drift (e.g. in-place mutation, dtype
    mismatch) would knock the cosine below 1.
    """

    model = _build_two_layer_sim()
    x = torch.randn(1, 3, 8, 8)

    rows = per_layer_isolated_cosine(
        model,
        x,
        cand_mode=ExecutionMode.FP32_QDQ,
        ref_mode=ExecutionMode.FP32_QDQ,
    )

    assert len(rows) == 2, f"expected one row per Conv2d, got {rows}"
    for row in rows:
        assert row["isolated_cosine"] == pytest.approx(1.0, abs=1e-6), row


def test_isolated_fs_vs_fp32_keeps_high_cosine():
    """FIXED_SCALE_QDQ vs FP32_QDQ isolated cosine should stay near 1.

    A single 8-bit quantization grid (no upstream accumulation) cannot
    reasonably push the per-layer cosine below ~0.99 for friendly ranges,
    so this also acts as a regression guard against future kernel drifts.
    """

    model = _build_two_layer_sim()
    x = torch.randn(1, 3, 8, 8)

    rows = per_layer_isolated_cosine(
        model,
        x,
        cand_mode=ExecutionMode.FIXED_SCALE_QDQ,
        ref_mode=ExecutionMode.FP32_QDQ,
    )

    assert len(rows) == 2, rows
    cosines = [row["isolated_cosine"] for row in rows]
    assert cosines == sorted(cosines), f"rows must be sorted ascending: {cosines}"
    assert min(cosines) >= 0.99, f"isolated cosine drift: {rows}"


def test_isolated_top_k_truncates_rows():
    model = _build_two_layer_sim()
    x = torch.randn(1, 3, 8, 8)

    rows = per_layer_isolated_cosine(
        model,
        x,
        cand_mode=ExecutionMode.FP32_QDQ,
        ref_mode=ExecutionMode.FP32_QDQ,
        top_k=1,
    )

    assert len(rows) == 1


def test_isolated_module_filter_respected():
    """A custom filter must shrink the probe set to matching modules only."""

    model = _build_two_layer_sim()
    x = torch.randn(1, 3, 8, 8)

    def only_first_conv(name: str, module: nn.Module) -> bool:
        return name == "0" and isinstance(module, QuantizedConv2d)

    rows = per_layer_isolated_cosine(
        model,
        x,
        cand_mode=ExecutionMode.FP32_QDQ,
        ref_mode=ExecutionMode.FP32_QDQ,
        module_filter=only_first_conv,
    )

    assert len(rows) == 1
    assert rows[0]["module"] == "0"


def test_isolated_row_schema_includes_tier1_metrics():
    """Every row must carry the Tier-1 metric set (cosine + sqnr + rmse + p99).

    Locks the row schema so dashboards / CI parsers stay stable.
    """

    import math

    model = _build_two_layer_sim()
    x = torch.randn(1, 3, 8, 8)

    rows = per_layer_isolated_cosine(
        model,
        x,
        cand_mode=ExecutionMode.FP32_QDQ,
        ref_mode=ExecutionMode.FP32_QDQ,
    )
    assert rows, "expected at least one row"
    required = {
        "module",
        "isolated_cosine",
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
        # self-compare ⇒ noise == 0 ⇒ rmse 0, sqnr +inf, p99 0.
        assert row["rmse"] == pytest.approx(0.0, abs=1e-6)
        assert math.isinf(row["sqnr_db"]) or row["sqnr_db"] > 100
        assert row["p99_abs_err"] == pytest.approx(0.0, abs=1e-6)


def test_isolated_int16_covers_all_layers_via_carrier():
    """INT16 cand_mode reaches every layer via the carrier pass.

    Without the predecessor-carrier hook this test would only see the
    first conv (whose own input quantizer is initialized); with it, both
    convs must be probed and produce a high cosine.
    """

    model = _build_two_layer_sim()
    x = torch.randn(1, 3, 8, 8)

    rows = per_layer_isolated_cosine(
        model,
        x,
        cand_mode=ExecutionMode.INT16_FIXED_EVAL,
        ref_mode=ExecutionMode.FP32_QDQ,
    )

    assert len(rows) == 2, f"INT16 isolated should cover both convs: {rows}"
    cosines = [row["isolated_cosine"] for row in rows]
    assert min(cosines) >= 0.99, f"INT16 isolated cosine drift: {rows}"


def test_isolated_dequantizes_quantized_tensor_inputs():
    """Inputs cached from FP32_QDQ may be QuantizedTensorBase Tensor subclasses.

    MobileNet ``classifier.0`` (Dropout) exposed this: treating the subclass as
    a plain tensor cached integer-like quantized payloads and produced a false
    low isolated cosine.
    """

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

    rows = per_layer_isolated_cosine(
        model,
        x,
        cand_mode=ExecutionMode.INT16_FIXED_EVAL,
        ref_mode=ExecutionMode.FP32_QDQ,
        module_filter=lambda name, _module: name == "1",
    )

    assert len(rows) == 1
    assert rows[0]["isolated_cosine"] == pytest.approx(1.0, abs=1e-6), rows
    assert rows[0]["rmse"] == pytest.approx(0.0, abs=1e-6), rows


def test_isolated_handles_inplace_relu_downstream():
    """Inplace ReLU after the probed conv must not corrupt the cache.

    Models like MobileNet V2 chain ``Conv2d → ReLU6(inplace=True)``; without
    the defensive ``.clone()`` in the hook, the cached "output" of the conv
    would silently morph into the post-ReLU values before we compute cosine.
    """

    torch.manual_seed(1)
    conv = QuantizedConv2d(3, 4, kernel_size=3, padding=1, bias=True)
    _init_conv2d_quantizers(conv, out_range=4.0)
    nn.init.normal_(conv.weight, std=0.1)
    nn.init.zeros_(conv.bias)
    model = nn.Sequential(conv, nn.ReLU6(inplace=True))
    x = torch.randn(1, 3, 8, 8)

    rows = per_layer_isolated_cosine(
        model,
        x,
        cand_mode=ExecutionMode.FP32_QDQ,
        ref_mode=ExecutionMode.FP32_QDQ,
    )

    assert len(rows) == 1
    assert rows[0]["isolated_cosine"] == pytest.approx(1.0, abs=1e-6), rows[0]

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
"""Smoke tests for the model-agnostic logits / classification metrics."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

import aimet_torch.fixed_point.kernels  # noqa: F401, E402  pylint: disable=unused-import
from aimet_torch.fixed_point import ExecutionMode  # noqa: E402
from aimet_torch.fixed_point.metrics import (  # noqa: E402
    DEFAULT_VS_FP32_MODES,
    dequantize_logits,
    logits_cosine,
    mean_logits_cosine_on_loader,
    mean_logits_cosine_vs_fp32,
    top1_accuracy,
    top1_drop,
    top_k_accuracy,
    top_k_prediction_agreement,
)
from aimet_torch.v2.nn import QuantizedLinear  # noqa: E402
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


def _build_tiny_classifier(num_classes: int = 4) -> nn.Module:
    torch.manual_seed(0)
    layer = QuantizedLinear(6, num_classes, bias=True)
    _init_linear_quantizers(layer)
    nn.init.normal_(layer.weight, std=0.2)
    nn.init.zeros_(layer.bias)
    return layer


def _build_loader(num_classes: int = 4, n: int = 12) -> DataLoader:
    torch.manual_seed(42)
    x = torch.randn(n, 6)
    y = torch.randint(0, num_classes, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=4)


def test_dequantize_logits_passes_plain_tensor_through():
    t = torch.tensor([1.0, 2.0])
    out = dequantize_logits(t)
    assert torch.equal(out, t)


def test_logits_cosine_self_compare_is_one():
    model = _build_tiny_classifier()
    x = torch.randn(2, 6)
    val = logits_cosine(
        model,
        x,
        ref_mode=ExecutionMode.FP32_QDQ,
        cand_mode=ExecutionMode.FP32_QDQ,
    )
    assert val == pytest.approx(1.0, abs=1e-6)


def test_mean_logits_cosine_on_loader_bounded_and_finite():
    model = _build_tiny_classifier()
    loader = _build_loader()
    val = mean_logits_cosine_on_loader(
        model,
        loader,
        ref_mode=ExecutionMode.FP32_QDQ,
        cand_mode=ExecutionMode.FIXED_SCALE_QDQ,
        max_batches=2,
    )
    assert 0.0 <= val <= 1.0 + 1e-6
    assert val > 0.9, f"unexpected drift on FS vs FP32: {val}"


def test_mean_logits_cosine_vs_fp32_returns_value_keys():
    """Dict keys must equal ``ExecutionMode.value`` strings for JSON stability."""

    model = _build_tiny_classifier()
    loader = _build_loader()
    out = mean_logits_cosine_vs_fp32(model, loader, max_batches=1)
    expected_keys = {mode.value for mode in DEFAULT_VS_FP32_MODES}
    assert set(out) == expected_keys
    for v in out.values():
        assert 0.0 <= v <= 1.0 + 1e-6


def test_top1_accuracy_and_drop_self_consistent():
    model = _build_tiny_classifier()
    loader = _build_loader()
    fp32_acc = top1_accuracy(model, loader, ExecutionMode.FP32_QDQ)
    fs_acc = top1_accuracy(model, loader, ExecutionMode.FIXED_SCALE_QDQ)
    assert 0.0 <= fp32_acc <= 1.0
    assert 0.0 <= fs_acc <= 1.0
    drop = top1_drop(fp32_acc, fs_acc)
    assert drop >= 0.0
    assert drop == pytest.approx(max(0.0, fp32_acc - fs_acc), abs=1e-9)


def test_top_k_accuracy_monotone_in_k():
    """top-k accuracy is non-decreasing in k for the same loader / mode."""

    model = _build_tiny_classifier(num_classes=4)
    loader = _build_loader(num_classes=4)
    k1 = top_k_accuracy(model, loader, ExecutionMode.FP32_QDQ, k=1)
    k2 = top_k_accuracy(model, loader, ExecutionMode.FP32_QDQ, k=2)
    k4 = top_k_accuracy(model, loader, ExecutionMode.FP32_QDQ, k=4)
    assert k1 <= k2 + 1e-9 <= k4 + 1e-9
    assert k4 == pytest.approx(1.0, abs=1e-9), "top-k with k=#classes must be 1.0"


def test_top_k_prediction_agreement_self_compare_is_one():
    model = _build_tiny_classifier()
    loader = _build_loader()
    val = top_k_prediction_agreement(
        model,
        loader,
        ref_mode=ExecutionMode.FP32_QDQ,
        cand_mode=ExecutionMode.FP32_QDQ,
        k=1,
    )
    assert val == pytest.approx(1.0, abs=1e-9)


def test_top1_accuracy_rejects_invalid_k():
    model = _build_tiny_classifier()
    loader = _build_loader()
    with pytest.raises(ValueError):
        top_k_accuracy(model, loader, ExecutionMode.FP32_QDQ, k=0)


def test_imagenet_eval_thin_wrappers_match_metrics_core():
    """Ensure ImageNet's backward-compat shims call into the new core."""

    from aimet_torch.fixed_point.e2e.imagenet_eval import (
        cosine_vs_fp32_on_loader,
        logits_cosine_between_modes,
        logits_cosine_on_loader,
        top1_accuracy as legacy_top1,
        top1_drop as legacy_top1_drop,
    )

    model = _build_tiny_classifier()
    loader = _build_loader()

    # ``logits_cosine_on_loader`` ≡ mean_logits_cosine_on_loader(ref=FP32, cand=…)
    legacy = logits_cosine_on_loader(
        model, loader, ExecutionMode.FIXED_SCALE_QDQ, max_batches=1
    )
    direct = mean_logits_cosine_on_loader(
        model,
        loader,
        ref_mode=ExecutionMode.FP32_QDQ,
        cand_mode=ExecutionMode.FIXED_SCALE_QDQ,
        max_batches=1,
    )
    assert legacy == pytest.approx(direct, abs=1e-9)

    # ``logits_cosine_between_modes`` accepts arbitrary (ref, cand) pairs.
    pair = logits_cosine_between_modes(
        model,
        loader,
        ExecutionMode.FIXED_SCALE_QDQ,
        ExecutionMode.INT16_FIXED_EVAL,
        max_batches=1,
    )
    assert 0.0 <= pair <= 1.0 + 1e-6

    # ``cosine_vs_fp32_on_loader`` keeps the historical key shape.
    summary = cosine_vs_fp32_on_loader(model, loader, max_batches=1)
    assert set(summary) == {mode.value for mode in DEFAULT_VS_FP32_MODES}

    # top1 / top1_drop shims agree with the new public API.
    assert legacy_top1(model, loader, ExecutionMode.FP32_QDQ) == pytest.approx(
        top1_accuracy(model, loader, ExecutionMode.FP32_QDQ), abs=1e-9
    )
    assert legacy_top1_drop(0.8, 0.7) == pytest.approx(0.1, abs=1e-9)

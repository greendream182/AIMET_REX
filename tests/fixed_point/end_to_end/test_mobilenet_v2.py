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
"""End-to-end smoke tests for a real CNN: ``MobileNet V2`` (mock).

These tests exercise the full pipeline that production code follows:

    raw nn.Module
      └─ prepare_model (replace functional `+`, `mean` with modules)
      └─ fold_all_batch_norms
      └─ QuantizationSimModel(default 8b)
      └─ ensure_output_quantizers_for_int16_eval (materialize missing oq)
      └─ compute_encodings
      └─ forward in {FP32_QDQ, INT16_FIXED_EVAL, FP16_QDQ}

The tests guard against regressions in:
  * AvgPool int unfold dtype handling
  * QuantizedMean INT16 kernel (global-average-pool path)
  * FP16_QDQ parameter dtype propagation (bias/weight cast)
  * INT16 dispatch refusing to run without an output quantizer everywhere

Cosine tests print metrics on pass; run with ``pytest -s`` or ``pytest -rP`` to see them.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxscript")
pytest.importorskip("torchvision")  # examples.mobilenet has no direct dep, but v2 stack does
import torch.nn.functional as F  # noqa: E402

import aimet_torch.fixed_point.kernels  # noqa: F401, E402
from aimet_torch.fixed_point import (  # noqa: E402
    ExecutionMode,
    Int16QuantizedTensor,
    ensure_output_quantizers_for_int16_eval,
    iter_missing_output_quantizers,
    quant_execution_mode,
    set_quant_execution_mode,
)
from aimet_torch.fixed_point.metrics import (  # noqa: E402
    compute_pair_metrics,
    int16_eval_allow_debug_float,
)


INPUT_SIZE = 64
CALIB_BATCH = 2
CALIB_ITERS = 4
EVAL_BATCH = 2


@pytest.fixture(scope="module")
def calibrated_sim():
    """Build, prepare, fold, calibrate a tiny MobileNet V2 once per test module."""

    from .mobilenet_v2_helpers import build_calibrated_sim, build_prepared_mobilenet_v2

    set_quant_execution_mode(ExecutionMode.FP32_QDQ)
    model, dummy = build_prepared_mobilenet_v2()
    bundle = build_calibrated_sim(model, dummy)

    return {
        "sim": bundle.sim,
        "n_missing_before_patch": bundle.n_oq_patched,
        "n_patched": bundle.n_oq_patched,
    }


def _forward_dequantized(sim, x, mode):
    with quant_execution_mode(mode):
        y = sim.model(x)
    if hasattr(y, "dequantize"):
        y = y.dequantize()
    return y


def _report_mobilenet_cosine(case: str, cosine: float, *, min_cosine: float, **extras: float) -> None:
    """Print metrics on pass; use ``pytest -s`` or ``pytest -rP`` to see stdout."""

    parts = [f"[mobilenet_v2] {case}: cosine={cosine:.9f} (min {min_cosine})"]
    for key, value in extras.items():
        parts.append(f"{key}={value:.6g}")
    print(", ".join(parts), flush=True)


def test_mobilenet_v2_int16_dispatchable_after_oq_patch(calibrated_sim):
    """After ``ensure_output_quantizers_for_int16_eval`` the sim must have no missing oq."""

    sim = calibrated_sim["sim"]
    assert calibrated_sim["n_missing_before_patch"] >= 1, (
        "Expected the default super-group config to leave at least one "
        "intermediate Conv without an output quantizer; if this changed, "
        "this test still passes after the patch but we should review the helper."
    )
    leftover = list(iter_missing_output_quantizers(sim))
    assert leftover == [], (
        "ensure_output_quantizers_for_int16_eval did not patch every dispatchable "
        f"module: {leftover}"
    )


def test_mobilenet_v2_runs_in_all_three_modes(calibrated_sim):
    """Forward must succeed in FP32_QDQ, FP16_QDQ, and INT16_FIXED_EVAL."""

    sim = calibrated_sim["sim"]
    torch.manual_seed(7)
    x = torch.randn(EVAL_BATCH, 3, INPUT_SIZE, INPUT_SIZE)

    with torch.no_grad():
        y_fp32 = _forward_dequantized(sim, x, ExecutionMode.FP32_QDQ)
        y_fp16 = _forward_dequantized(sim, x, ExecutionMode.FP16_QDQ)
        with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
            y_int = sim.model(x)

    assert y_fp32.shape == (EVAL_BATCH, 10)
    assert y_fp32.dtype == torch.float32
    assert y_fp16.shape == (EVAL_BATCH, 10)
    assert y_fp16.dtype == torch.float16
    assert isinstance(y_int, Int16QuantizedTensor), (
        f"Expected Int16QuantizedTensor from INT16_FIXED_EVAL; got {type(y_int).__name__}"
    )
    assert y_int.int_repr.shape == (EVAL_BATCH, 10)


def test_mobilenet_v2_int16_vs_fp32_cosine_meets_minimum(calibrated_sim):
    """INT16 output stays close to the FP32_QDQ reference on a real CNN.

    With a tiny MobileNet V2 (10 classes, 64x64 input), 16-segment PWLs and an
    8-bit output grid the cosine vs FP32_QDQ is typically >=0.998 on random
    inputs. We use a slightly relaxed bound here (0.99) because:
      * the network has 6 PWL-driven layers (ReLU is exact, but tanh/gelu
        appear in alternative configs); per-fn PWL error compounds across
        depth more strongly than in a single-op test
      * uncalibrated dropout makes the final logits sensitive to small drifts
    """

    sim = calibrated_sim["sim"]
    torch.manual_seed(13)
    x = torch.randn(EVAL_BATCH, 3, INPUT_SIZE, INPUT_SIZE)

    with torch.no_grad():
        y_ref = _forward_dequantized(sim, x, ExecutionMode.FP32_QDQ)
        with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
            y_int = sim.model(x)

    assert isinstance(y_int, Int16QuantizedTensor)
    with int16_eval_allow_debug_float():
        cand = y_int.to_float()
    metrics = compute_pair_metrics(
        y_ref,
        cand,
        scale=y_int.scale,
        zero_point=y_int.zero_point,
        qmin=y_int.qmin,
        qmax=y_int.qmax,
        candidate_int_repr=y_int.int_repr,
    )
    min_cos = 0.99
    _report_mobilenet_cosine(
        "INT16 vs FP32_QDQ",
        metrics["cosine_similarity"],
        min_cosine=min_cos,
        max_lsb=metrics["max_error_lsb"],
        max_abs=metrics["max_abs_error"],
        rmse=metrics["rmse"],
    )
    assert metrics["cosine_similarity"] >= min_cos, (
        f"MobileNet V2 INT16 vs FP32_QDQ cosine={metrics['cosine_similarity']:.6f} "
        f"max_lsb={metrics['max_error_lsb']:.2f}"
    )


def test_mobilenet_v2_fixed_scale_vs_fp32_cosine_meets_minimum(calibrated_sim):
    """fixed_scale_qdq logits stay close to fp32_qdq on calibrated MobileNet V2."""

    sim = calibrated_sim["sim"]
    torch.manual_seed(17)
    x = torch.randn(EVAL_BATCH, 3, INPUT_SIZE, INPUT_SIZE)

    with torch.no_grad():
        y_ref = _forward_dequantized(sim, x, ExecutionMode.FP32_QDQ)
        y_fix = _forward_dequantized(sim, x, ExecutionMode.FIXED_SCALE_QDQ)

    metrics = compute_pair_metrics(y_ref, y_fix)
    min_cos = 0.9998
    top1_match = (y_ref.argmax(-1) == y_fix.argmax(-1)).float().mean().item()
    _report_mobilenet_cosine(
        "fixed_scale_qdq vs FP32_QDQ",
        metrics["cosine_similarity"],
        min_cosine=min_cos,
        max_abs=metrics["max_abs_error"],
        rmse=metrics["rmse"],
        top1_match=top1_match,
    )
    # (M,r) scale approximation; typical cosine ~0.99985 on this mock net.
    assert metrics["cosine_similarity"] >= min_cos, (
        f"fixed_scale vs fp32 cosine={metrics['cosine_similarity']:.6f}"
    )
    assert top1_match == 1.0


def test_mobilenet_v2_fp16_vs_fp32_cosine_meets_minimum(calibrated_sim):
    """FP16 QDQ output stays close to FP32 QDQ on a real CNN."""

    sim = calibrated_sim["sim"]
    torch.manual_seed(21)
    x = torch.randn(EVAL_BATCH, 3, INPUT_SIZE, INPUT_SIZE)

    with torch.no_grad():
        y_ref = _forward_dequantized(sim, x, ExecutionMode.FP32_QDQ)
        y_fp16 = _forward_dequantized(sim, x, ExecutionMode.FP16_QDQ)

    cos = F.cosine_similarity(
        y_ref.flatten().unsqueeze(0),
        y_fp16.float().flatten().unsqueeze(0),
    ).item()
    min_cos = 0.999
    metrics = compute_pair_metrics(y_ref, y_fp16.float())
    _report_mobilenet_cosine(
        "FP16 vs FP32_QDQ",
        cos,
        min_cosine=min_cos,
        max_abs=metrics["max_abs_error"],
        rmse=metrics["rmse"],
    )
    assert cos >= min_cos, f"MobileNet V2 FP16 vs FP32_QDQ cosine={cos:.6f}"


def test_mobilenet_v2_sidecar_export_matches_online_int_repr(calibrated_sim):
    """Exported ``*.int16.json`` must replay the same INT16 logits as online LUT generation."""

    import os
    from tempfile import TemporaryDirectory

    from aimet_torch.fixed_point.export import (
        attach_int16_sidecar_to_model,
        detach_int16_sidecar_from_model,
        export_int16_sidecar_json,
    )

    sim = calibrated_sim["sim"]
    detach_int16_sidecar_from_model(sim.model)

    torch.manual_seed(11)
    x = torch.randn(EVAL_BATCH, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
            y_online = sim.model(x)

    with TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "mobilenet.int16.json")
        doc = export_int16_sidecar_json(sim.model, path)
        assert doc["layer_count"] >= 1

        detach_int16_sidecar_from_model(sim.model)
        attach_int16_sidecar_to_model(sim.model, path)
        with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
            y_sidecar = sim.model(x)
        detach_int16_sidecar_from_model(sim.model)

    assert isinstance(y_online, Int16QuantizedTensor)
    assert isinstance(y_sidecar, Int16QuantizedTensor)
    torch.testing.assert_close(
        y_online.int_repr.to(torch.int32),
        y_sidecar.int_repr.to(torch.int32),
        rtol=0,
        atol=0,
    )

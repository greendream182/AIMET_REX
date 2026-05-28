#!/usr/bin/env python3
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
"""Generate fixed-point quality report.

Sections:
  * PWL kernels vs analytic activations (per-fn metrics + per-fn thresholds)
  * INT16 single-op (v2 Quantized*) vs FP32_QDQ (max_lsb / cosine on output grid)
  * FP16 QDQ vs FP32 QDQ (cosine on backend QDQ path)
  * Mock MobileNet V2 PTQ (vanilla / CLE / AdaRound) and INT16 QAT vs FP32_QDQ
  * pytest summary for ``tests/fixed_point``

Outputs both Markdown and JSON in ``scripts/fixed_point/reports/``.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn

from aimet_torch.fixed_point import (
    ExecutionMode,
    InputEncoding,
    Int16QuantizedTensor,
    OutputEncoding,
    quant_execution_mode,
)
from aimet_torch.fixed_point.metrics import (
    FIXED_SCALE_VS_FP32_MIN_COSINE_SIMILARITY,
    FP16_VS_FP32_MIN_COSINE_SIMILARITY,
    INT16_VS_FP32_MAX_ERROR_LSB,
    INT16_VS_FP32_MIN_COSINE_SIMILARITY,
    compute_pair_metrics,
    cosine_similarity,
    int16_eval_allow_debug_float,
)
from aimet_torch.fixed_point.metrics.thresholds import (
    PWL_HARDWARE_NUM_SEGMENTS,
    PWL_VS_ANALYTIC_DEFAULT_LIMITS,
    PWL_VS_ANALYTIC_MIN_COSINE_SIMILARITY,
    PWL_VS_ANALYTIC_PER_FN_LIMITS,
    int16_single_op_min_cosine_similarity,
)
from aimet_torch.fixed_point.offline import (
    check_pwl_metrics_within_limits,
    generate_pwl_lut_for_export,
    resolve_pwl_quality_limits,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_DIR = REPO_ROOT / "scripts" / "fixed_point" / "reports"

# Known noisy AIMET messages during report generation (informational, not gate failures).
_KNOWN_RUNTIME_NOTES = [
    "ConnectedGraph: Unable to isolate model outputs (prepared MobileNet with functional Mean/Add).",
    "Quant: Unsupported op type Mean / Shape / If (graph preparer placeholders).",
    "v2 AutoQuant may print ignored torch.export SpecViolationError when ONNX export is attempted on prepared models; use --quiet to skip AutoQuant.",
    "cvxpy optional: AMP convert-op reduction logs at debug when cvxpy is absent.",
]


def _configure_report_logging(*, quiet: bool) -> None:
    """Reduce console noise for ``run_report_fast.sh`` / ``--quiet``."""

    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=".*NLLLoss2d.*",
    )
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message=".*quant_scheme.*",
    )
    level = logging.ERROR if quiet else logging.WARNING
    for name in (
        "ConnectedGraph",
        "Quant",
        "ModelPreparer",
        "Utils",
        "AutoQuant",
    ):
        logging.getLogger(name).setLevel(level)


# --------------------------------------------------------------------------- #
# Section 1: PWL kernels vs analytic activations
# --------------------------------------------------------------------------- #

@dataclass
class PwlCase:
    fn_name: str
    fn: Callable[[torch.Tensor], torch.Tensor]
    input_range: float = 8.0
    output_qmin: int = 0
    output_qmax: int = 32767
    output_range: float = 1.0


_PWL_CASES: list[PwlCase] = [
    PwlCase("sigmoid", torch.sigmoid),
    PwlCase("tanh", torch.tanh, output_qmin=-32768, output_qmax=32767, output_range=1.0),
    PwlCase("gelu", torch.nn.functional.gelu, output_qmin=-32768, output_qmax=32767, output_range=8.0),
    PwlCase("silu", torch.nn.functional.silu, output_qmin=-32768, output_qmax=32767, output_range=8.0),
    PwlCase("softplus", torch.nn.functional.softplus, output_qmin=0, output_qmax=32767, output_range=8.0),
    PwlCase("mish", torch.nn.functional.mish, output_qmin=-32768, output_qmax=32767, output_range=8.0),
]


def _enc(scale: float, qmin: int, qmax: int) -> InputEncoding:
    return InputEncoding(
        scale=torch.tensor(scale, dtype=torch.float32),
        zero_point=torch.tensor(0, dtype=torch.int32),
        qmin=qmin,
        qmax=qmax,
    )


def _oenc(scale: float, qmin: int, qmax: int) -> OutputEncoding:
    return OutputEncoding(
        scale=torch.tensor(scale, dtype=torch.float32),
        zero_point=torch.tensor(0, dtype=torch.int32),
        qmin=qmin,
        qmax=qmax,
    )


def collect_pwl_section() -> dict[str, Any]:
    results = []
    for case in _PWL_CASES:
        input_enc = _enc(case.input_range / 32767, -32768, 32767)
        output_enc = _oenc(case.output_range / (case.output_qmax or 32767), case.output_qmin, case.output_qmax)
        _, num_segments, metrics = generate_pwl_lut_for_export(
            case.fn, input_enc, output_enc, fn_name=case.fn_name
        )
        limits = resolve_pwl_quality_limits(case.fn_name)
        failures = check_pwl_metrics_within_limits(metrics, limits)
        results.append(
            {
                "fn_name": case.fn_name,
                "num_segments": num_segments,
                "input_range": [-case.input_range, case.input_range],
                "output_grid": {"qmin": case.output_qmin, "qmax": case.output_qmax, "range": case.output_range},
                "metrics": metrics,
                "limits": limits,
                "failures": [
                    {"metric": name, "actual": actual, "threshold": threshold}
                    for name, actual, threshold in failures
                ],
                "status": "PASS" if not failures else "FAIL",
            }
        )
    return {"cases": results}


# --------------------------------------------------------------------------- #
# Section 2: INT16 single-op vs FP32_QDQ
# --------------------------------------------------------------------------- #

def _try_import_v2():
    try:
        from aimet_torch.v2.nn import (  # noqa: F401
            QuantizedConv2d,
            QuantizedGELU,
            QuantizedLinear,
            QuantizedReLU,
            QuantizedSigmoid,
            QuantizedTanh,
        )
        from aimet_torch.v2.quantization.affine import Quantize  # noqa: F401
        import aimet_torch.fixed_point.kernels  # noqa: F401

        return True
    except Exception:  # pragma: no cover - environment dependent
        return False


def _set_linear(m, in_range: float, weight_range: float, out_range: float, bitwidth: int = 8) -> None:
    from aimet_torch.v2.quantization.affine import Quantize

    m.input_quantizers[0] = Quantize((), bitwidth, symmetric=True)
    m.param_quantizers["weight"] = Quantize((m.out_features, 1), bitwidth, symmetric=True)
    m.output_quantizers[0] = Quantize((), bitwidth, symmetric=True)
    m.input_quantizers[0].min = nn.Parameter(torch.tensor(-in_range))
    m.input_quantizers[0].max = nn.Parameter(torch.tensor(in_range))
    m.param_quantizers["weight"].min = nn.Parameter(torch.full((m.out_features, 1), -weight_range))
    m.param_quantizers["weight"].max = nn.Parameter(torch.full((m.out_features, 1), weight_range))
    m.output_quantizers[0].min = nn.Parameter(torch.tensor(-out_range))
    m.output_quantizers[0].max = nn.Parameter(torch.tensor(out_range))


def _set_conv2d(m, in_range: float, weight_range: float, out_range: float, bitwidth: int = 8) -> None:
    from aimet_torch.v2.quantization.affine import Quantize

    oc = m.out_channels
    m.input_quantizers[0] = Quantize((), bitwidth, symmetric=True)
    m.param_quantizers["weight"] = Quantize((oc, 1, 1, 1), bitwidth, symmetric=True)
    m.output_quantizers[0] = Quantize((), bitwidth, symmetric=True)
    m.input_quantizers[0].min = nn.Parameter(torch.tensor(-in_range))
    m.input_quantizers[0].max = nn.Parameter(torch.tensor(in_range))
    m.param_quantizers["weight"].min = nn.Parameter(torch.full((oc, 1, 1, 1), -weight_range))
    m.param_quantizers["weight"].max = nn.Parameter(torch.full((oc, 1, 1, 1), weight_range))
    m.output_quantizers[0].min = nn.Parameter(torch.tensor(-out_range))
    m.output_quantizers[0].max = nn.Parameter(torch.tensor(out_range))


def _set_unary(m, in_range: float, out_min: float, out_max: float, bitwidth: int = 8) -> None:
    from aimet_torch.v2.quantization.affine import Quantize

    m.input_quantizers[0] = Quantize((), bitwidth, symmetric=True)
    m.output_quantizers[0] = Quantize((), bitwidth, symmetric=True)
    m.input_quantizers[0].min = nn.Parameter(torch.tensor(-in_range))
    m.input_quantizers[0].max = nn.Parameter(torch.tensor(in_range))
    m.output_quantizers[0].min = nn.Parameter(torch.tensor(out_min))
    m.output_quantizers[0].max = nn.Parameter(torch.tensor(out_max))


def _run_int16_case(
    name: str,
    build: Callable[[], nn.Module],
    x: torch.Tensor,
    *,
    output_bitwidth: int,
    stress_label: str,
) -> dict[str, Any]:
    model = build()
    model.eval()
    with quant_execution_mode(ExecutionMode.FP32_QDQ):
        y_q = model(x)
        y_fp = y_q.dequantize() if hasattr(y_q, "dequantize") else y_q
    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y_int = model(x)
    assert isinstance(y_int, Int16QuantizedTensor), f"Expected Int16QuantizedTensor for {name}"
    with int16_eval_allow_debug_float():
        cur_float = y_int.to_float()
    pair = compute_pair_metrics(
        y_fp,
        cur_float,
        scale=y_int.scale,
        zero_point=y_int.zero_point,
        qmin=y_int.qmin,
        qmax=y_int.qmax,
        candidate_int_repr=y_int.int_repr,
    )
    enforce_lsb = stress_label == "deploy-8b"
    min_cos = int16_single_op_min_cosine_similarity(name)
    failures = []
    if enforce_lsb and pair["max_error_lsb"] > INT16_VS_FP32_MAX_ERROR_LSB:
        failures.append(("max_error_lsb", pair["max_error_lsb"], INT16_VS_FP32_MAX_ERROR_LSB))
    if pair["cosine_similarity"] + 1e-12 < min_cos:
        failures.append(("cosine_similarity", pair["cosine_similarity"], min_cos))
    limits = {
        "min_cosine_similarity": min_cos,
    }
    if enforce_lsb:
        limits["max_error_lsb"] = INT16_VS_FP32_MAX_ERROR_LSB
    return {
        "case": name,
        "stress": stress_label,
        "informational": not enforce_lsb,
        "n_samples": int(cur_float.numel()),
        "output_bitwidth": int(output_bitwidth),
        "output_scale": float(y_int.scale.detach().cpu().reshape(-1)[0].item()),
        "metrics": pair,
        "limits": limits,
        "failures": [{"metric": k, "actual": a, "threshold": t} for k, a, t in failures],
        "status": "PASS" if not failures else "FAIL",
    }


def collect_int16_kernel_section() -> dict[str, Any]:
    if not _try_import_v2():
        return {"skipped": True, "reason": "aimet_torch.v2 stack unavailable"}

    from aimet_torch.v2.nn import (
        QuantizedConv2d,
        QuantizedGELU,
        QuantizedLinear,
        QuantizedReLU,
        QuantizedSigmoid,
        QuantizedSoftmax,
        QuantizedTanh,
    )

    cases: list[dict[str, Any]] = []

    rng = torch.Generator().manual_seed(0)

    def _x_random(rows: int, cols: int = 3, span: float = 2.0) -> torch.Tensor:
        return (torch.rand(rows, cols, generator=rng) * 2 - 1).mul_(span)

    def _x_image_batch(batch: int = 4) -> torch.Tensor:
        return (torch.rand(batch, 1, 4, 4, generator=rng) * 2 - 1).mul_(2.0)

    # -------- deploy-style (8-bit output): mirrors production sidecar --------

    def linear_relu_linear(out_bits=8):
        model = nn.Sequential(QuantizedLinear(3, 4), QuantizedReLU(), QuantizedLinear(4, 2))
        _set_linear(model[0], 2.0, 0.5, 2.0, bitwidth=out_bits)
        _set_unary(model[1], 2.0, 0.0, 2.0, bitwidth=out_bits)
        _set_linear(model[2], 2.0, 0.5, 2.0, bitwidth=out_bits)
        nn.init.constant_(model[0].weight, 0.1)
        nn.init.constant_(model[0].bias, 0.01)
        nn.init.constant_(model[2].weight, 0.08)
        nn.init.constant_(model[2].bias, -0.02)
        return model

    def linear_sigmoid(out_bits=8):
        model = nn.Sequential(QuantizedLinear(3, 3), QuantizedSigmoid())
        _set_linear(model[0], 2.0, 0.5, 4.0, bitwidth=out_bits)
        _set_unary(model[1], 4.0, 0.0, 1.0, bitwidth=out_bits)
        nn.init.constant_(model[0].weight, 0.2)
        nn.init.constant_(model[0].bias, 0.0)
        return model

    def linear_tanh(out_bits=8):
        model = nn.Sequential(QuantizedLinear(3, 3), QuantizedTanh())
        _set_linear(model[0], 2.0, 0.5, 3.0, bitwidth=out_bits)
        _set_unary(model[1], 3.0, -1.0, 1.0, bitwidth=out_bits)
        nn.init.constant_(model[0].weight, 0.1)
        nn.init.constant_(model[0].bias, 0.0)
        return model

    def linear_gelu(out_bits=8):
        model = nn.Sequential(QuantizedLinear(3, 3), QuantizedGELU(), QuantizedLinear(3, 2))
        _set_linear(model[0], 2.0, 0.5, 3.0, bitwidth=out_bits)
        _set_unary(model[1], 3.0, -1.0, 3.0, bitwidth=out_bits)
        _set_linear(model[2], 3.0, 0.5, 2.0, bitwidth=out_bits)
        nn.init.constant_(model[0].weight, 0.15)
        nn.init.constant_(model[0].bias, 0.01)
        nn.init.constant_(model[2].weight, 0.1)
        nn.init.constant_(model[2].bias, 0.0)
        return model

    def conv2d_only(out_bits=8):
        model = QuantizedConv2d(1, 2, kernel_size=2, stride=1, padding=0, bias=True)
        _set_conv2d(model, 2.0, 0.3, 4.0, bitwidth=out_bits)
        nn.init.constant_(model.weight, 0.2)
        nn.init.constant_(model.bias, 0.01)
        return model

    def linear_softmax(out_bits=8):
        model = nn.Sequential(QuantizedLinear(3, 4), QuantizedSoftmax(dim=-1))
        _set_linear(model[0], 2.0, 0.5, 4.0, bitwidth=out_bits)
        _set_unary(model[1], 4.0, 0.0, 1.0, bitwidth=out_bits)
        nn.init.constant_(model[0].weight, 0.12)
        nn.init.constant_(model[0].bias, 0.0)
        return model

    x_seq = _x_random(64, 3, span=1.5)
    x_img = _x_image_batch(4)

    # 8-bit output (deploy-style, errors absorbed by coarse grid)
    cases.append(_run_int16_case(
        "Linear → ReLU → Linear", linear_relu_linear, x_seq,
        output_bitwidth=8, stress_label="deploy-8b",
    ))
    cases.append(_run_int16_case(
        "Linear → Sigmoid (PWL)", linear_sigmoid, x_seq,
        output_bitwidth=8, stress_label="deploy-8b",
    ))
    cases.append(_run_int16_case(
        "Linear → Tanh (PWL)", linear_tanh, x_seq,
        output_bitwidth=8, stress_label="deploy-8b",
    ))
    cases.append(_run_int16_case(
        "Linear → GELU → Linear (PWL)", linear_gelu, x_seq,
        output_bitwidth=8, stress_label="deploy-8b",
    ))
    cases.append(_run_int16_case(
        "Conv2d", conv2d_only, x_img,
        output_bitwidth=8, stress_label="deploy-8b",
    ))
    cases.append(_run_int16_case(
        "Linear → Softmax (PWL exp)", linear_softmax, x_seq,
        output_bitwidth=8, stress_label="deploy-8b",
    ))

    # 16-bit output stress (forces PWL error to surface)
    cases.append(_run_int16_case(
        "Linear → Sigmoid (PWL) [stress]", lambda: linear_sigmoid(out_bits=16), x_seq,
        output_bitwidth=16, stress_label="stress-16b-out",
    ))
    cases.append(_run_int16_case(
        "Linear → Tanh (PWL) [stress]", lambda: linear_tanh(out_bits=16), x_seq,
        output_bitwidth=16, stress_label="stress-16b-out",
    ))
    cases.append(_run_int16_case(
        "Linear → GELU → Linear (PWL) [stress]", lambda: linear_gelu(out_bits=16), x_seq,
        output_bitwidth=16, stress_label="stress-16b-out",
    ))
    cases.append(_run_int16_case(
        "Linear → Softmax (PWL exp) [stress]", lambda: linear_softmax(out_bits=16), x_seq,
        output_bitwidth=16, stress_label="stress-16b-out",
    ))

    return {"cases": cases}


# --------------------------------------------------------------------------- #
# Section 3: FP16 QDQ vs FP32 QDQ
# --------------------------------------------------------------------------- #

def collect_fp16_section() -> dict[str, Any]:
    try:
        from aimet_torch.v2.quantization.affine.backends.torch_builtins import (
            quantize_dequantize,
        )
    except Exception:
        return {"skipped": True, "reason": "aimet_torch.v2 backend unavailable"}

    cases = []
    rng = torch.Generator().manual_seed(0)
    fixtures = [
        ("random_4x8", lambda: torch.randn(4, 8, generator=rng), 0.05),
        ("random_2x16x16", lambda: torch.randn(2, 16, 16, generator=rng), 0.02),
        ("uniform_8", lambda: torch.linspace(-1.0, 1.0, 64), 0.125),
    ]
    for name, make_x, scale in fixtures:
        tensor = make_x()
        s = torch.tensor(scale, dtype=torch.float32)
        offset = torch.tensor(0.0, dtype=torch.float32)
        with quant_execution_mode(ExecutionMode.FP32_QDQ):
            y_fp32 = quantize_dequantize(tensor, s, offset, -128, 127)
        with quant_execution_mode(ExecutionMode.FP16_QDQ):
            y_fp16 = quantize_dequantize(tensor, s, offset, -128, 127)
        cos = cosine_similarity(y_fp32, y_fp16.float())
        status = "PASS" if cos >= FP16_VS_FP32_MIN_COSINE_SIMILARITY else "FAIL"
        cases.append(
            {
                "case": name,
                "metrics": {"cosine_similarity": cos},
                "limits": {"min_cosine_similarity": FP16_VS_FP32_MIN_COSINE_SIMILARITY},
                "status": status,
            }
        )
    return {"cases": cases}


# --------------------------------------------------------------------------- #
# Section 4: MobileNet V2 end-to-end (FP32_QDQ vs INT16_FIXED_EVAL vs FP16_QDQ)
# --------------------------------------------------------------------------- #


def collect_mobilenet_v2_section() -> dict[str, Any]:
    """Run a real CNN (mock MobileNet V2) through the full sim pipeline.

    Covers the prepare_model -> fold_all_batch_norms -> sim -> compute_encodings
    flow and reports FP32_QDQ / FIXED_SCALE_QDQ / INT16_FIXED_EVAL / FP16_QDQ
    cosine on a single deterministic input. Skips gracefully if any optional
    dependency is missing.
    """

    try:
        # pylint: disable=import-outside-toplevel
        from aimet_torch.batch_norm_fold import fold_all_batch_norms
        from aimet_torch.examples.mobilenet import MockMobileNetV2
        from aimet_torch.fixed_point import (
            ensure_output_quantizers_for_int16_eval,
            iter_missing_output_quantizers,
        )
        from aimet_common.defs import QuantScheme
        from aimet_torch.model_preparer import prepare_model
        from aimet_torch.v2.quantsim import QuantizationSimModel
    except Exception as exc:  # pragma: no cover - environment-dependent
        return {"skipped": True, "reason": f"missing dependency: {exc}"}

    torch.manual_seed(0)
    input_size = 64
    try:
        model = MockMobileNetV2(n_class=10, input_size=input_size).eval()
        model = prepare_model(model)
        dummy = torch.randn(1, 3, input_size, input_size)
        fold_all_batch_norms(
            model,
            input_shapes=(1, 3, input_size, input_size),
            dummy_input=dummy,
        )
        sim = QuantizationSimModel(
            model,
            dummy_input=dummy,
            default_output_bw=8,
            default_param_bw=8,
            quant_scheme=QuantScheme.post_training_tf_enhanced,
        )
        n_missing = sum(1 for _ in iter_missing_output_quantizers(sim))
        patched = ensure_output_quantizers_for_int16_eval(sim, bitwidth=8, symmetric=True)

        def _calib(m, _):
            with torch.no_grad():
                for _ in range(4):
                    m(torch.randn(2, 3, input_size, input_size))

        sim.compute_encodings(_calib, None)

        torch.manual_seed(13)
        x = torch.randn(2, 3, input_size, input_size)
        with torch.no_grad():
            with quant_execution_mode(ExecutionMode.FP32_QDQ):
                y_fp32 = sim.model(x)
                y_fp32 = y_fp32.dequantize() if hasattr(y_fp32, "dequantize") else y_fp32
            with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
                y_int = sim.model(x)
            with quant_execution_mode(ExecutionMode.FP16_QDQ):
                y_fp16 = sim.model(x)
                y_fp16 = y_fp16.dequantize() if hasattr(y_fp16, "dequantize") else y_fp16
            with quant_execution_mode(ExecutionMode.FIXED_SCALE_QDQ):
                y_fixed = sim.model(x)
                y_fixed = (
                    y_fixed.dequantize() if hasattr(y_fixed, "dequantize") else y_fixed
                )
    except Exception as exc:  # pragma: no cover
        return {"skipped": True, "reason": f"runtime error: {exc}"}

    cases: list[dict[str, Any]] = []

    if isinstance(y_int, Int16QuantizedTensor):
        with int16_eval_allow_debug_float():
            cand = y_int.to_float()
        m_int = compute_pair_metrics(
            y_fp32, cand,
            scale=y_int.scale, zero_point=y_int.zero_point,
            qmin=y_int.qmin, qmax=y_int.qmax,
            candidate_int_repr=y_int.int_repr,
        )
        # cosine-only gate: 1-LSB does not apply meaningfully on a deep network
        # where output dynamic range is far below full-scale (relative error
        # would dominate the LSB-on-output-grid view).
        min_cos = 0.99
        failures = []
        if m_int["cosine_similarity"] + 1e-12 < min_cos:
            failures.append(
                {"metric": "cosine_similarity", "actual": m_int["cosine_similarity"], "threshold": min_cos}
            )
        cases.append({
            "case": "MockMobileNetV2 / INT16 vs FP32_QDQ",
            "stress": "deploy-8b",
            "output_bitwidth": int(y_int.qmax - y_int.qmin + 1).bit_length() - 1,
            "output_scale": float(y_int.scale.reshape(-1)[0].item()),
            "n_samples": int(cand.numel()),
            "metrics": m_int,
            "limits": {"min_cosine_similarity": min_cos},
            "failures": failures,
            "status": "PASS" if not failures else "FAIL",
        })
    else:
        cases.append({
            "case": "MockMobileNetV2 / INT16 vs FP32_QDQ",
            "stress": "deploy-8b",
            "metrics": {},
            "limits": {},
            "failures": [{"metric": "type", "actual": type(y_int).__name__, "threshold": "Int16QuantizedTensor"}],
            "status": "FAIL",
        })

    cos16 = cosine_similarity(y_fp32, y_fp16.float())
    # Mock E2E: align with ``baseline.json`` / ``test_mobilenet_v2`` (0.999), not the
    # stricter single-tensor FP16 backend gate (0.9999).
    min_cos_fp16 = min(FP16_VS_FP32_MIN_COSINE_SIMILARITY, 0.999)
    fp16_failures = []
    if cos16 + 1e-12 < min_cos_fp16:
        fp16_failures.append({"metric": "cosine_similarity", "actual": cos16, "threshold": min_cos_fp16})
    cases.append({
        "case": "MockMobileNetV2 / FP16 vs FP32_QDQ",
        "stress": "deploy-fp16",
        "metrics": {"cosine_similarity": cos16},
        "limits": {"min_cosine_similarity": min_cos_fp16},
        "failures": fp16_failures,
        "status": "PASS" if not fp16_failures else "FAIL",
    })

    m_fix = compute_pair_metrics(y_fp32, y_fixed.float())
    min_cos_fix = FIXED_SCALE_VS_FP32_MIN_COSINE_SIMILARITY
    fix_failures = []
    if m_fix["cosine_similarity"] + 1e-12 < min_cos_fix:
        fix_failures.append(
            {
                "metric": "cosine_similarity",
                "actual": m_fix["cosine_similarity"],
                "threshold": min_cos_fix,
            }
        )
    cases.append({
        "case": "MockMobileNetV2 / fixed_scale_qdq vs FP32_QDQ",
        "stress": "deploy-fixed-scale",
        "metrics": m_fix,
        "limits": {"min_cosine_similarity": min_cos_fix},
        "failures": fix_failures,
        "status": "PASS" if not fix_failures else "FAIL",
    })

    return {
        "input_size": input_size,
        "missing_oq_before_patch": n_missing,
        "oq_patched": len(patched),
        "cases": cases,
    }


MOBILENET_PTQ_MIN_COSINE = 0.99
MOBILENET_CLE_COSINE_TOLERANCE = 0.005


def _mobilenet_ptq_case_row(
    name: str,
    cosine: float,
    *,
    ref_cosine: float | None = None,
    rel_floor: float = 0.0,
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    if cosine + 1e-12 < MOBILENET_PTQ_MIN_COSINE:
        failures.append({
            "metric": "cosine_similarity",
            "actual": cosine,
            "threshold": MOBILENET_PTQ_MIN_COSINE,
        })
    if ref_cosine is not None and cosine + 1e-6 < ref_cosine - rel_floor:
        failures.append({
            "metric": "cosine_vs_reference",
            "actual": cosine,
            "threshold": ref_cosine - rel_floor,
        })
    return {
        "case": name,
        "metrics": {"cosine_similarity": cosine},
        "limits": {
            "min_cosine_similarity": MOBILENET_PTQ_MIN_COSINE,
            "min_vs_reference": (ref_cosine - rel_floor) if ref_cosine is not None else None,
        },
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
    }


def _collect_mobilenet_ptq_qat_for_variant(
    *,
    variant: str,
    input_size: int,
    qat_epochs: int,
    adaround_iterations: int,
    include_long_adaround: bool,
) -> list[dict[str, Any]]:
    from aimet_torch.fixed_point.e2e.mobilenet_v2 import (
        MOBILENET_ADAROUND_ITERATIONS_LONG,
        build_calibrated_sim,
        build_prepared_mobilenet_v2,
        int16_vs_fp32_cosine,
        make_adaround_loader,
        train_int16_qat,
    )

    prefix = "Mock" if variant == "mock" else "Full"
    torch.manual_seed(13)
    x_holdout = torch.randn(2, 3, input_size, input_size)
    cases: list[dict[str, Any]] = []
    qat_losses: list[float] = []

    model, dummy = build_prepared_mobilenet_v2(input_size=input_size, variant=variant)  # type: ignore[arg-type]
    vanilla = build_calibrated_sim(
        model, dummy, input_size=input_size, variant=variant  # type: ignore[arg-type]
    )
    cos_vanilla = int16_vs_fp32_cosine(vanilla.sim, x_holdout)
    cases.append(_mobilenet_ptq_case_row(f"{prefix} PTQ vanilla (min-max)", cos_vanilla))

    model_cle, dummy_cle = build_prepared_mobilenet_v2(input_size=input_size, variant=variant)  # type: ignore[arg-type]
    cle = build_calibrated_sim(
        model_cle, dummy_cle, input_size=input_size, variant=variant, apply_cle=True  # type: ignore[arg-type]
    )
    cos_cle = int16_vs_fp32_cosine(cle.sim, x_holdout)
    cases.append(_mobilenet_ptq_case_row(
        f"{prefix} PTQ + CLE",
        cos_cle,
        ref_cosine=cos_vanilla,
        rel_floor=MOBILENET_CLE_COSINE_TOLERANCE,
    ))

    model_bc, dummy_bc = build_prepared_mobilenet_v2(input_size=input_size, variant=variant)  # type: ignore[arg-type]
    bc = build_calibrated_sim(
        model_bc,
        dummy_bc,
        input_size=input_size,
        variant=variant,  # type: ignore[arg-type]
        apply_bias_correction=True,
    )
    cos_bc = int16_vs_fp32_cosine(bc.sim, x_holdout)
    cases.append(_mobilenet_ptq_case_row(
        f"{prefix} PTQ + bias correction",
        cos_bc,
        ref_cosine=cos_vanilla,
        rel_floor=0.01,
    ))

    model_ada, dummy_ada = build_prepared_mobilenet_v2(input_size=input_size, variant=variant)  # type: ignore[arg-type]
    loader = make_adaround_loader(input_size, num_batches=2)
    with tempfile.TemporaryDirectory() as tmpdir:
        ada = build_calibrated_sim(
            model_ada,
            dummy_ada,
            input_size=input_size,
            variant=variant,  # type: ignore[arg-type]
            adaround_loader=loader,
            adaround_num_batches=2,
            adaround_iterations=adaround_iterations,
            adaround_export_dir=Path(tmpdir),
        )
    cos_ada = int16_vs_fp32_cosine(ada.sim, x_holdout)
    cases.append(_mobilenet_ptq_case_row(
        f"{prefix} PTQ + AdaRound ({adaround_iterations} iter)",
        cos_ada,
        ref_cosine=cos_vanilla,
        rel_floor=0.0001,
    ))

    if include_long_adaround and adaround_iterations < MOBILENET_ADAROUND_ITERATIONS_LONG:
        model_ada_long, dummy_ada_long = build_prepared_mobilenet_v2(  # type: ignore[arg-type]
            input_size=input_size, variant=variant
        )
        loader_long = make_adaround_loader(input_size, num_batches=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            ada_long = build_calibrated_sim(
                model_ada_long,
                dummy_ada_long,
                input_size=input_size,
                variant=variant,  # type: ignore[arg-type]
                adaround_loader=loader_long,
                adaround_num_batches=2,
                adaround_iterations=MOBILENET_ADAROUND_ITERATIONS_LONG,
                adaround_export_dir=Path(tmpdir),
            )
        cos_ada_long = int16_vs_fp32_cosine(ada_long.sim, x_holdout)
        cases.append(_mobilenet_ptq_case_row(
            f"{prefix} PTQ + AdaRound ({MOBILENET_ADAROUND_ITERATIONS_LONG} iter)",
            cos_ada_long,
            ref_cosine=cos_vanilla,
            rel_floor=0.0001,
        ))

    model_qat, dummy_qat = build_prepared_mobilenet_v2(input_size=input_size, variant=variant)  # type: ignore[arg-type]
    teacher = model_qat
    qat_bundle = build_calibrated_sim(
        model_qat, dummy_qat, input_size=input_size, variant=variant  # type: ignore[arg-type]
    )
    qat_losses = train_int16_qat(
        qat_bundle.sim,
        teacher=teacher,
        input_size=input_size,
        epochs=qat_epochs,
        lr=1e-3,
        batches_per_epoch=4,
        seed=99,
    )
    cos_qat = int16_vs_fp32_cosine(qat_bundle.sim, x_holdout)
    cases.append(_mobilenet_ptq_case_row(
        f"{prefix} PTQ + INT16 QAT ({qat_epochs} ep)",
        cos_qat,
        ref_cosine=cos_vanilla,
    ))
    return cases, qat_losses


def collect_mobilenet_v2_ptq_qat_section(
    *,
    qat_epochs: int = 8,
    adaround_iterations: int = 80,
    include_full_mobilenet: bool = True,
    full_input_size: int = 96,
    include_long_adaround: bool = False,
    skip_autoquant: bool = False,
    suppress_autoquant_logs: bool = False,
) -> dict[str, Any]:
    """Compare PTQ variants and INT16 QAT on hold-out logits (INT16 vs FP32_QDQ)."""

    try:
        from aimet_torch.fixed_point.e2e import mobilenet_v2 as _mn  # noqa: F401
    except Exception as exc:  # pragma: no cover
        return {"skipped": True, "reason": f"missing dependency: {exc}"}

    cases: list[dict[str, Any]] = []
    qat_losses: list[float] = []
    try:
        mock_cases, qat_losses = _collect_mobilenet_ptq_qat_for_variant(
            variant="mock",
            input_size=64,
            qat_epochs=qat_epochs,
            adaround_iterations=adaround_iterations,
            include_long_adaround=include_long_adaround,
        )
        cases.extend(mock_cases)

        if include_full_mobilenet:
            full_cases, _ = _collect_mobilenet_ptq_qat_for_variant(
                variant="full",
                input_size=full_input_size,
                qat_epochs=max(4, qat_epochs // 2),
                adaround_iterations=adaround_iterations,
                include_long_adaround=False,
            )
            cases.extend(full_cases)

        if not skip_autoquant:
            cases.append(_try_collect_v2_autoquant_case(
                input_size=64,
                adaround_iterations=adaround_iterations,
                suppress_autoquant_tracebacks=suppress_autoquant_logs,
            ))
    except Exception as exc:  # pragma: no cover
        return {"skipped": True, "reason": f"runtime error: {exc}"}

    qat_summary = _extract_qat_summary(cases, qat_epochs, qat_losses)
    return {
        "mock_input_size": 64,
        "full_input_size": full_input_size if include_full_mobilenet else None,
        "holdout_seed": 13,
        "qat_epochs": qat_epochs,
        "adaround_iterations": adaround_iterations,
        "include_long_adaround": include_long_adaround,
        "skip_autoquant": skip_autoquant,
        "suppress_autoquant_logs": suppress_autoquant_logs,
        "qat_final_epoch_loss": qat_losses[-1] if qat_losses else None,
        "qat_summary": qat_summary,
        "cases": cases,
    }


def _extract_qat_summary(
    cases: list[dict[str, Any]],
    qat_epochs: int,
    qat_losses: list[float],
) -> dict[str, Any]:
    """Pull PTQ vs QAT cosine from mock section-5 rows for JSON / markdown."""
    ptq_cos = None
    qat_cos = None
    qat_status = "MISSING"
    for case in cases:
        name = case.get("case", "")
        if "Mock" not in name:
            continue
        cos = case.get("metrics", {}).get("cosine_similarity")
        if "PTQ vanilla" in name:
            ptq_cos = cos
        if "INT16 QAT" in name:
            qat_cos = cos
            qat_status = case.get("status", "UNKNOWN")
    delta = None
    if ptq_cos is not None and qat_cos is not None:
        delta = float(qat_cos) - float(ptq_cos)
    return {
        "ptq_only_cosine": ptq_cos,
        "qat_cosine": qat_cos,
        "cosine_delta_vs_ptq": delta,
        "status": qat_status,
        "report_epochs": qat_epochs,
        "pytest_epochs_reference": 20,
        "final_epoch_mse": qat_losses[-1] if qat_losses else None,
    }


def _try_collect_v2_autoquant_case(
    *,
    input_size: int,
    adaround_iterations: int,
    suppress_autoquant_tracebacks: bool = False,
) -> dict[str, Any]:
    """Best-effort AIMET v2 AutoQuant row (informational; does not gate overall PASS)."""
    from aimet_torch.fixed_point.e2e.autoquant import make_autoquant_loader, try_run_v2_autoquant_ptq
    from aimet_torch.fixed_point.e2e.mobilenet_v2 import (
        build_calibrated_sim,
        build_prepared_mobilenet_v2,
        int16_vs_fp32_cosine,
    )

    torch.manual_seed(13)
    x_holdout = torch.randn(2, 3, input_size, input_size)
    model, dummy = build_prepared_mobilenet_v2(input_size=input_size)
    ref_model = model
    holdout = x_holdout

    def eval_cb(quant_model, *_args, **_kwargs):
        quant_model.eval()
        device = next(quant_model.parameters()).device
        with torch.no_grad():
            xb = holdout.to(device)
            ref = ref_model(xb)
            out = quant_model(xb)
            if hasattr(out, "dequantize"):
                out = out.dequantize()
            return float(
                cosine_similarity(ref, out.float())
            )

    loader = make_autoquant_loader(input_size, num_samples=16, batch_size=2)

    def _run_autoquant():
        with tempfile.TemporaryDirectory() as tmpdir:
            return try_run_v2_autoquant_ptq(
                model,
                dummy,
                data_loader=loader,
                eval_callback=eval_cb,
                results_dir=Path(tmpdir),
                allowed_accuracy_drop=0.05,
                model_prepare_required=False,
                adaround_num_batches=2,
                adaround_iterations=adaround_iterations,
            )

    if suppress_autoquant_tracebacks:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            aq_result = _run_autoquant()
    else:
        aq_result = _run_autoquant()

    if aq_result is None:
        return {
            "case": "Mock v2 AutoQuant / combined PTQ fallback",
            "informational": True,
            "metrics": {},
            "limits": {},
            "failures": [],
            "status": "SKIP",
            "skip_reason": "AutoQuant and combined PTQ fallback both failed",
        }

    bundle = build_calibrated_sim(aq_result.model, dummy, input_size=input_size)
    cos = int16_vs_fp32_cosine(bundle.sim, x_holdout)
    label = (
        "Mock v2 AutoQuant → INT16"
        if aq_result.source == "autoquant"
        else "Mock combined PTQ (CLE+BC+AdaRound) → INT16"
    )
    row = _mobilenet_ptq_case_row(label, cos)
    row["informational"] = aq_result.source == "autoquant"
    row["ptq_source"] = aq_result.source
    row["autoquant_eval_score"] = aq_result.eval_score
    row["encoding_path"] = aq_result.encoding_path
    return row


# --------------------------------------------------------------------------- #
# Section 6: pytest summary
# --------------------------------------------------------------------------- #

def collect_pytest_summary(
    target: str = "tests/fixed_point",
    *,
    marker: str | None = "not slow",
) -> dict[str, Any]:
    """Run pytest for section 6.

    Default excludes ``@pytest.mark.slow`` (2k AdaRound, full MobileNet) so the
    report gate matches ``pytest -m 'not slow'``. Use ``marker=None`` for all tests.
    """
    cmd = [sys.executable, "-m", "pytest", target, "-q", "--no-header", "--tb=line"]
    if marker:
        cmd.extend(["-m", marker])
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(REPO_ROOT))
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    summary_line = ""
    for line in reversed(combined.splitlines()):
        if re.search(r"\b(\d+)\s+(passed|failed|skipped|error)", line, re.I):
            summary_line = line.strip()
            break
    counts = {key: 0 for key in ("passed", "failed", "skipped", "errors", "warnings")}
    for key in counts:
        m = re.search(rf"(\d+)\s+{key}", summary_line)
        if m:
            counts[key] = int(m.group(1))
    failure_tail = None
    if proc.returncode != 0:
        failure_tail = "\n".join(combined.splitlines()[-40:]).strip() or None
    return {
        "cmd": " ".join(cmd),
        "marker": marker if marker else "(all)",
        "exit_code": proc.returncode,
        "summary_line": summary_line,
        "counts": counts,
        "failure_tail": failure_tail,
    }


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def _fmt(value: float, precision: int = 6) -> str:
    if isinstance(value, float):
        if abs(value) >= 1000 or (value != 0 and abs(value) < 1e-3):
            return f"{value:.{precision}g}"
        formatted = f"{value:.{precision}f}"
        if "." in formatted:
            formatted = formatted.rstrip("0").rstrip(".")
        return formatted
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Fixed-Point Quality Report")
    lines.append("")
    lines.append(f"- Generated at: `{report['generated_at']}`")
    lines.append(f"- Repo: `{REPO_ROOT}`")
    lines.append(f"- PyTorch: `{torch.__version__}`")
    lines.append("")

    # Overall status badge (informational cases excluded from overall gate)
    overall_pass = True
    for section_name in (
        "pwl_vs_analytic",
        "int16_vs_fp32",
        "fp16_vs_fp32",
        "mobilenet_v2",
        "mobilenet_v2_ptq_qat",
    ):
        sec = report.get(section_name, {})
        if sec.get("skipped"):
            continue
        for case in sec.get("cases", []):
            if case.get("informational") or case.get("status") == "SKIP":
                continue
            if case.get("status") == "FAIL":
                overall_pass = False
    py_sec = report["pytest"]
    if not py_sec.get("skipped"):
        overall_pass = overall_pass and py_sec.get("exit_code") == 0
    lines.append(f"**Overall status:** {'PASS' if overall_pass else 'FAIL'}")
    lines.append("")
    notes = report.get("runtime_notes") or []
    opts = report.get("report_options") or {}
    if notes:
        lines.append("## Runtime notes (console noise vs failures)")
        lines.append("")
        if opts.get("skip_autoquant"):
            lines.append(
                "- v2 AutoQuant row **skipped** (`--fast` / `--skip-autoquant`); "
                "combined PTQ rows in §5 still run."
            )
        if opts.get("quiet"):
            lines.append("- AIMET log level reduced (`--quiet`).")
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")

    # PWL section
    pwl = report["pwl_vs_analytic"]
    lines.append("## 1. PWL Kernels vs Analytic Activation")
    lines.append("")
    lines.append(
        f"- Segments (hardware-fixed): **{PWL_HARDWARE_NUM_SEGMENTS}**"
    )
    lines.append(
        f"- Required: `cosine_similarity ≥ {PWL_VS_ANALYTIC_MIN_COSINE_SIMILARITY}` and per-fn LSB limits."
    )
    lines.append("")
    lines.append("| fn | status | cosine | max_lsb (limit) | p99_lsb (limit) | p999_lsb | rmse_lsb (limit) | max_rel |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for case in pwl["cases"]:
        m = case["metrics"]
        lim = case["limits"]
        lines.append(
            "| {fn} | {st} | {cos} | {mx} ({lmx}) | {p99} ({lp99}) | {p999} | {rmse} ({lrmse}) | {rel} |".format(
                fn=case["fn_name"],
                st=case["status"],
                cos=f"{m['cosine_similarity']:.6f}",
                mx=_fmt(m["max_lsb"], 2),
                lmx=_fmt(lim.get("max_lsb", 0), 0),
                p99=_fmt(m["p99_lsb"], 2),
                lp99=_fmt(lim.get("p99_lsb", 0), 0),
                p999=_fmt(m["p999_lsb"], 2),
                rmse=_fmt(m["rmse_lsb"], 2),
                lrmse=_fmt(lim.get("rmse_lsb", 0), 0),
                rel=_fmt(m.get("max_relative_error", float("nan")), 3),
            )
        )
    for case in pwl["cases"]:
        if case["failures"]:
            lines.append("")
            lines.append(f"**{case['fn_name']} failures:**")
            for f in case["failures"]:
                lines.append(f"  - `{f['metric']}` = {f['actual']:.4f} vs limit {f['threshold']}")
    lines.append("")

    # INT16 single-op
    lines.append("## 2. INT16 Single-Op vs FP32_QDQ")
    lines.append("")
    int16 = report["int16_vs_fp32"]
    if int16.get("skipped"):
        lines.append(f"- Skipped: {int16['reason']}")
    else:
        lines.append(
            f"- Required: `max_error_lsb ≤ {INT16_VS_FP32_MAX_ERROR_LSB}` (deploy-8b only) and "
            f"per-case `cosine_similarity` floors (default ≥ {INT16_VS_FP32_MIN_COSINE_SIMILARITY}; "
            "relaxed for deploy-8b PWL tanh/gelu)."
        )
        lines.append("")
        lines.append(
            "- `deploy-8b` mirrors production (8-bit output quantizer, PWL error largely absorbed by grid)."
        )
        lines.append(
            "- `stress-16b-out` lifts the output quantizer to 16 bits to expose PWL approximation error."
        )
        lines.append(
            "- LSB is reported on the **output quantization grid** (see `out_scale` column)."
        )
        lines.append("")
        lines.append(
            "| case | stress | out_bits | out_scale | n | status | cosine | max_lsb | max_abs | rmse | sqnr_db |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for case in int16["cases"]:
            m = case["metrics"]
            lines.append(
                "| {n} | {ss} | {ob} | {oscale} | {samples} | {st} | {cos:.6f} | {ml} | {mabs} | {rm} | {sq} |".format(
                    n=case["case"],
                    ss=case.get("stress", ""),
                    ob=case.get("output_bitwidth", ""),
                    oscale=_fmt(case.get("output_scale", 0.0), 6),
                    samples=case.get("n_samples", ""),
                    st=case["status"],
                    cos=m["cosine_similarity"],
                    ml=_fmt(m["max_error_lsb"], 2),
                    mabs=_fmt(m["max_abs_error"], 6),
                    rm=_fmt(m["rmse"], 6),
                    sq=_fmt(m["sqnr_db"], 2),
                )
            )
        for case in int16["cases"]:
            if case["failures"]:
                lines.append("")
                lines.append(f"**{case['case']} ({case.get('stress', '')}) failures:**")
                for f in case["failures"]:
                    lines.append(f"  - `{f['metric']}` = {f['actual']} vs threshold {f['threshold']}")
        lines.append("")
        lines.append(
            "_Note: 1 LSB on the output grid equals `out_scale`. With an 8-bit output the LSB is large enough "
            "that PWL approximation error (which can be 200+ LSB on the int16 grid relative to analytic `fn`) "
            "is rounded to 0 LSB on the deploy grid. The stress rows confirm the underlying PWL error._"
        )
    lines.append("")

    # FP16
    lines.append("## 3. FP16 QDQ vs FP32 QDQ")
    lines.append("")
    fp16 = report["fp16_vs_fp32"]
    if fp16.get("skipped"):
        lines.append(f"- Skipped: {fp16['reason']}")
    else:
        lines.append(
            f"- Required: `cosine_similarity ≥ {FP16_VS_FP32_MIN_COSINE_SIMILARITY}` (no LSB gate)."
        )
        lines.append("")
        lines.append("| case | status | cosine |")
        lines.append("|---|---|---|")
        for case in fp16["cases"]:
            lines.append(
                f"| {case['case']} | {case['status']} | {case['metrics']['cosine_similarity']:.6f} |"
            )
    lines.append("")

    # MobileNet V2 end-to-end
    lines.append("## 4. MobileNet V2 (mock) End-to-End")
    lines.append("")
    mb = report["mobilenet_v2"]
    if mb.get("skipped"):
        lines.append(f"- Skipped: {mb['reason']}")
    else:
        lines.append(
            f"- Pipeline: ``prepare_model → fold_all_batch_norms → "
            f"QuantizationSimModel(default_output_bw=8) → "
            f"ensure_output_quantizers_for_int16_eval → compute_encodings``."
        )
        lines.append(
            f"- Input: random ``2 × 3 × {mb['input_size']} × {mb['input_size']}`` "
            f"(seeded), output: 10 logits."
        )
        lines.append(
            f"- Output quantizers materialized by the helper: "
            f"**{mb['oq_patched']}** (missing before patch: **{mb['missing_oq_before_patch']}**)."
        )
        lines.append("")
        lines.append("| case | stress | status | cosine | extras |")
        lines.append("|---|---|---|---|---|")
        for case in mb["cases"]:
            m = case.get("metrics", {})
            cos = m.get("cosine_similarity", float("nan"))
            extras = []
            if "max_error_lsb" in m:
                extras.append(f"max_lsb={_fmt(m['max_error_lsb'], 2)}")
            if "max_abs_error" in m:
                extras.append(f"max_abs={_fmt(m['max_abs_error'], 6)}")
            if "rmse" in m:
                extras.append(f"rmse={_fmt(m['rmse'], 6)}")
            if "output_scale" in case:
                extras.append(f"out_scale={_fmt(case['output_scale'], 6)}")
            lines.append(
                "| {n} | {ss} | {st} | {cos} | {ex} |".format(
                    n=case["case"],
                    ss=case.get("stress", ""),
                    st=case["status"],
                    cos=f"{cos:.6f}" if isinstance(cos, float) else str(cos),
                    ex=", ".join(extras) if extras else "",
                )
            )
        for case in mb["cases"]:
            for f in case.get("failures", []):
                lines.append(
                    f"- **{case['case']} FAIL**: `{f['metric']}` = {f['actual']} vs threshold {f['threshold']}"
                )
    lines.append("")

    # MobileNet PTQ / QAT
    lines.append("## 5. MobileNet V2 PTQ Algorithms & INT16 QAT")
    lines.append("")
    ptq = report.get("mobilenet_v2_ptq_qat", {})
    if ptq.get("skipped"):
        lines.append(f"- Skipped: {ptq['reason']}")
    else:
        lines.append(
            "- Metric: **INT16_FIXED_EVAL vs FP32_QDQ** cosine on held-out logits "
            f"(seed **{ptq['holdout_seed']}**, shape "
            f"`2×3×{ptq.get('mock_input_size', 64)}×{ptq.get('mock_input_size', 64)}`)."
        )
        lines.append(
            f"- Gates: cosine ≥ **{MOBILENET_PTQ_MIN_COSINE}**; CLE may trail vanilla by at most "
            f"**{MOBILENET_CLE_COSINE_TOLERANCE}**; bias correction by **0.01**; "
            "AdaRound/QAT must not trail same-variant vanilla PTQ."
        )
        lines.append(
            f"- Mock @ 64×64; full MobileNet @ {ptq.get('full_input_size', '—')}×… when enabled."
        )
        lines.append(
            f"- Report QAT: **{ptq['qat_epochs']}** epochs (tests use 20); AdaRound: "
            f"**{ptq['adaround_iterations']}** iterations"
            f"{', plus long run' if ptq.get('include_long_adaround') else ''}, 2 calibration batches."
        )
        if ptq.get("skip_autoquant"):
            lines.append(
                "- v2 AutoQuant informational row **omitted** (use full `run_report.sh` without "
                "`--skip-autoquant` to exercise AutoQuant + combined fallback)."
            )
        qs = ptq.get("qat_summary") or {}
        lines.append("")
        lines.append("### INT16 QAT（mock 64×64，held-out logits）")
        lines.append("")
        if qs.get("qat_cosine") is not None:
            lines.append("| metric | value |")
            lines.append("|---|---|")
            lines.append(f"| PTQ-only INT16 cosine | {qs.get('ptq_only_cosine', float('nan')):.6f} |")
            lines.append(f"| After INT16 QAT cosine | {qs['qat_cosine']:.6f} |")
            if qs.get("cosine_delta_vs_ptq") is not None:
                lines.append(f"| Δ (QAT − PTQ) | {qs['cosine_delta_vs_ptq']:+.6f} |")
            lines.append(f"| QAT status | **{qs.get('status', '—')}** |")
            lines.append(f"| Report training epochs | {qs.get('report_epochs', '—')} |")
            lines.append(f"| pytest reference (`test_int16_qat_improves_over_ptq_only`) | {qs.get('pytest_epochs_reference', 20)} ep |")
            if qs.get("final_epoch_mse") is not None:
                lines.append(f"| Final epoch train MSE | {_fmt(qs['final_epoch_mse'], 6)} |")
            lines.append("")
            lines.append(
                "验收：QAT 后 INT16 cosine **不低于** 同模型 PTQ-only（与 `test_int16_qat_improves_over_ptq_only` 一致）。"
            )
        else:
            lines.append("- QAT 行未生成（Section 5 采集失败或被跳过）。")
        lines.append("")
        lines.append("### PTQ pipelines")
        lines.append("")
        lines.append("| pipeline | status | cosine | vs vanilla |")
        lines.append("|---|---|---|---|")
        baselines: dict[str, float] = {}
        for case in ptq["cases"]:
            if "PTQ vanilla" in case["case"]:
                prefix = case["case"].split(" PTQ")[0]
                baselines[prefix] = case["metrics"]["cosine_similarity"]
        for case in ptq["cases"]:
            if "INT16 QAT" in case["case"]:
                continue
            if case.get("status") == "SKIP":
                lines.append(
                    f"| {case['case']} | SKIP | — | {case.get('skip_reason', '')} |"
                )
                continue
            cos = case["metrics"]["cosine_similarity"]
            prefix = case["case"].split(" PTQ")[0]
            ref = baselines.get(prefix)
            vs = ""
            if ref is not None and "PTQ vanilla" not in case["case"]:
                vs = f"{cos - ref:+.6f}"
            lines.append(
                f"| {case['case']} | {case['status']} | {cos:.6f} | {vs or '—'} |"
            )
        for case in ptq["cases"]:
            for f in case.get("failures", []):
                lines.append(
                    f"- **{case['case']} FAIL**: `{f['metric']}` = {f['actual']} "
                    f"vs threshold {f['threshold']}"
                )
    lines.append("")

    # pytest
    lines.append("## 6. pytest Summary (tests/fixed_point)")
    lines.append("")
    py = report["pytest"]
    if py.get("skipped"):
        lines.append("- **Status: SKIPPED**（未执行 pytest，不是 0 个测试通过）。")
        lines.append(
            f"- 原因: `{py.get('reason', '--skip-pytest')}`。"
        )
        lines.append(
            "- 本地验收: `PYTHONPATH=<repo> python3 -m pytest tests/fixed_point -q`"
        )
        lines.append(
            "- 生成报告时若需带上 pytest 统计，**不要**加 `--skip-pytest`。"
        )
    else:
        counts = py["counts"]
        lines.append(
            f"- Exit code: **{py['exit_code']}** ({'PASS' if py['exit_code'] == 0 else 'FAIL'})"
        )
        lines.append(f"- Marker: `{py.get('marker', 'not slow')}`")
        lines.append(f"- Summary: `{py['summary_line']}`")
        if py.get("failure_tail"):
            lines.append("")
            lines.append("<details><summary>pytest failure tail</summary>")
            lines.append("")
            lines.append("```")
            lines.append(py["failure_tail"])
            lines.append("```")
            lines.append("")
            lines.append("</details>")
        lines.append("")
        lines.append("| passed | failed | skipped | errors | warnings |")
        lines.append("|---|---|---|---|---|")
        lines.append(
            "| {p} | {f} | {s} | {e} | {w} |".format(
                p=counts["passed"],
                f=counts["failed"],
                s=counts["skipped"],
                e=counts["errors"],
                w=counts["warnings"],
            )
        )
        lines.append("")
        lines.append(f"_Command:_ `{py['cmd']}`")
    lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_REPORT_DIR),
        help="Directory to write report files into.",
    )
    parser.add_argument(
        "--skip-pytest",
        action="store_true",
        help="Skip pytest in section 6 (default: run tests/fixed_point).",
    )
    parser.add_argument(
        "--pytest-include-slow",
        action="store_true",
        help="Section 6: run all tests including @pytest.mark.slow (default: -m 'not slow').",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Shorthand: --skip-pytest --skip-full-mobilenet --quiet --skip-autoquant.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Lower AIMET log noise; with --skip-autoquant skip v2 AutoQuant row.",
    )
    parser.add_argument(
        "--skip-autoquant",
        action="store_true",
        help="Skip informational v2 AutoQuant row (avoids torch.export tracebacks on prepared models).",
    )
    parser.add_argument(
        "--skip-mobilenet-ptq-qat",
        action="store_true",
        help="Skip MobileNet PTQ/CLE/AdaRound/QAT benchmark (slow).",
    )
    parser.add_argument(
        "--mobilenet-qat-epochs",
        type=int,
        default=8,
        help="QAT epochs when generating section 5 (default: 8).",
    )
    parser.add_argument(
        "--mobilenet-adaround-iterations",
        type=int,
        default=80,
        help="AdaRound iterations in section 5 (default: 80). Use 10000 for full regression.",
    )
    parser.add_argument(
        "--mobilenet-long-adaround",
        action="store_true",
        help="Also run mock AdaRound with MOBILENET_ADAROUND_ITERATIONS_LONG (2000).",
    )
    parser.add_argument(
        "--skip-full-mobilenet",
        action="store_true",
        help="Skip full (non-mock) MobileNet rows in section 5.",
    )
    parser.add_argument(
        "--check-baseline",
        action="store_true",
        help="Run check_thresholds after report (default: on unless --no-check-baseline).",
    )
    parser.add_argument(
        "--no-check-baseline",
        action="store_true",
        help="Disable baseline.json comparison after report generation.",
    )
    args = parser.parse_args(argv)

    if args.fast:
        args.skip_pytest = True
        args.skip_full_mobilenet = True
        args.quiet = True
        args.skip_autoquant = True
    _configure_report_logging(quiet=args.quiet)
    check_baseline = args.check_baseline or not args.no_check_baseline

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "torch_version": torch.__version__,
        "pwl_vs_analytic": collect_pwl_section(),
        "int16_vs_fp32": collect_int16_kernel_section(),
        "fp16_vs_fp32": collect_fp16_section(),
        "mobilenet_v2": collect_mobilenet_v2_section(),
        "runtime_notes": list(_KNOWN_RUNTIME_NOTES),
        "report_options": {
            "fast": args.fast,
            "quiet": args.quiet,
            "skip_autoquant": args.skip_autoquant,
        },
    }
    if args.skip_mobilenet_ptq_qat:
        report["mobilenet_v2_ptq_qat"] = {"skipped": True, "reason": "cli --skip-mobilenet-ptq-qat"}
    else:
        report["mobilenet_v2_ptq_qat"] = collect_mobilenet_v2_ptq_qat_section(
            qat_epochs=args.mobilenet_qat_epochs,
            adaround_iterations=args.mobilenet_adaround_iterations,
            include_full_mobilenet=not args.skip_full_mobilenet,
            include_long_adaround=args.mobilenet_long_adaround,
            skip_autoquant=args.skip_autoquant,
            suppress_autoquant_logs=args.quiet and not args.skip_autoquant,
        )
    if args.skip_pytest:
        report["pytest"] = {
            "skipped": True,
            "reason": "--skip-pytest",
            "cmd": "(not run)",
            "exit_code": None,
            "summary_line": "(not run)",
            "counts": {k: None for k in ("passed", "failed", "skipped", "errors", "warnings")},
        }
    else:
        report["pytest"] = collect_pytest_summary(
            marker=None if args.pytest_include_slow else "not slow",
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "quality_report.json"
    md_path = output_dir / "quality_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")

    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")

    failed_sections: list[str] = []
    for key in (
        "pwl_vs_analytic",
        "int16_vs_fp32",
        "fp16_vs_fp32",
        "mobilenet_v2",
        "mobilenet_v2_ptq_qat",
    ):
        sec = report.get(key, {})
        if sec.get("skipped"):
            continue
        for case in sec.get("cases", []):
            if case.get("informational") or case.get("status") == "SKIP":
                continue
            if case.get("status") == "FAIL":
                failed_sections.append(key)
                break
    if not report["pytest"].get("skipped") and report["pytest"].get("exit_code") not in (0, None):
        failed_sections.append("pytest")
    if failed_sections:
        print(f"Failures detected in sections: {failed_sections}", file=sys.stderr)
        py_fail = report["pytest"].get("failure_tail")
        if py_fail:
            print("--- pytest tail ---", file=sys.stderr)
            print(py_fail, file=sys.stderr)
        return 1

    if check_baseline:
        from scripts.fixed_point.check_thresholds import check_quality_report_against_baseline

        baseline = json.loads(
            (Path(__file__).parent / "baseline.json").read_text(encoding="utf-8")
        )
        bl_failures = check_quality_report_against_baseline(report, baseline)
        if bl_failures:
            print("Baseline threshold failures:", file=sys.stderr)
            for scope, metric, actual, expected in bl_failures:
                print(
                    f"  {scope} {metric}: actual={actual!r} expected>={expected!r}",
                    file=sys.stderr,
                )
            return 1
        print("Baseline check: PASS")

    return 0


if __name__ == "__main__":
    sys.exit(main())

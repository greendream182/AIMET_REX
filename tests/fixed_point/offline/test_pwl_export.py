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

from aimet_torch.fixed_point import InputEncoding, OutputEncoding
from aimet_torch.fixed_point.metrics.thresholds import (
    PWL_HARDWARE_NUM_SEGMENTS,
    PWL_VS_ANALYTIC_MIN_COSINE_SIMILARITY,
    PWL_VS_ANALYTIC_PER_FN_LIMITS,
)
from aimet_torch.fixed_point.offline.lut_gen import (
    PwlLutAccuracyError,
    assert_pwl_metrics_within_limits,
    check_pwl_metrics_within_limits,
    generate_pwl_lut,
    generate_pwl_lut_for_export,
    measure_pwl_lut_max_error_lsb,
    measure_pwl_lut_metrics,
    resolve_pwl_quality_limits,
)


def _enc(scale, qmin, qmax, zp=0):
    return InputEncoding(
        scale=torch.tensor(scale, dtype=torch.float32),
        zero_point=torch.tensor(zp, dtype=torch.int32),
        qmin=qmin,
        qmax=qmax,
    )


def _out_enc(scale, qmin, qmax, zp=0):
    return OutputEncoding(
        scale=torch.tensor(scale, dtype=torch.float32),
        zero_point=torch.tensor(zp, dtype=torch.int32),
        qmin=qmin,
        qmax=qmax,
    )


_SIGMOID_IN = _enc(8.0 / 32767, -32768, 32767)
_SIGMOID_OUT = _out_enc(1.0 / 32767, 0, 32767)


def test_generate_pwl_lut_for_export_uses_hardware_segment_count():
    pwl, num_segments, metrics = generate_pwl_lut_for_export(
        torch.sigmoid, _SIGMOID_IN, _SIGMOID_OUT
    )

    assert num_segments == PWL_HARDWARE_NUM_SEGMENTS
    assert len(pwl["q_b"]) == PWL_HARDWARE_NUM_SEGMENTS
    assert set(metrics) >= {"max_lsb", "p99_lsb", "p999_lsb", "rmse_lsb", "cosine_similarity"}


def test_generate_pwl_lut_for_export_rejects_non_hardware_segment_count():
    with pytest.raises(ValueError, match="hardware"):
        generate_pwl_lut_for_export(
            torch.sigmoid, _SIGMOID_IN, _SIGMOID_OUT, num_segments=32
        )


def test_generate_pwl_lut_for_export_strict_quality_uses_per_fn_limits_passes():
    pwl, _, metrics = generate_pwl_lut_for_export(
        torch.sigmoid, _SIGMOID_IN, _SIGMOID_OUT, enforce_quality=True
    )
    assert pwl["q_b"].numel() == PWL_HARDWARE_NUM_SEGMENTS
    limits = PWL_VS_ANALYTIC_PER_FN_LIMITS["sigmoid"]
    assert metrics["max_lsb"] <= limits["max_lsb"]
    assert metrics["cosine_similarity"] >= PWL_VS_ANALYTIC_MIN_COSINE_SIMILARITY


def test_generate_pwl_lut_for_export_strict_quality_raises_on_tight_override():
    with pytest.raises(PwlLutAccuracyError, match="max_lsb"):
        generate_pwl_lut_for_export(
            torch.sigmoid,
            _SIGMOID_IN,
            _SIGMOID_OUT,
            enforce_quality=True,
            quality_limits={"max_lsb": 1.0, "min_cosine_similarity": 0.5},
        )


def test_measure_pwl_lut_max_error_lsb_matches_measure_metrics_max():
    pwl = generate_pwl_lut(torch.sigmoid, _SIGMOID_IN, _SIGMOID_OUT, num_segments=16)
    scalar = measure_pwl_lut_max_error_lsb(
        torch.sigmoid, _SIGMOID_IN, _SIGMOID_OUT, pwl, num_samples=512
    )
    metrics = measure_pwl_lut_metrics(
        torch.sigmoid, _SIGMOID_IN, _SIGMOID_OUT, pwl, num_samples=512
    )
    assert scalar == pytest.approx(metrics["max_lsb"])


def test_resolve_pwl_quality_limits_falls_back_to_default():
    limits = resolve_pwl_quality_limits("unknown_activation")
    assert limits["max_lsb"] > 1000
    assert limits["min_cosine_similarity"] == PWL_VS_ANALYTIC_MIN_COSINE_SIMILARITY


def test_check_pwl_metrics_within_limits_flags_low_cosine_and_high_lsb():
    metrics = {"max_lsb": 5000.0, "cosine_similarity": 0.5}
    failures = check_pwl_metrics_within_limits(
        metrics, {"max_lsb": 100.0, "min_cosine_similarity": 0.99}
    )
    names = {name for name, _, _ in failures}
    assert names == {"max_lsb", "cosine_similarity"}


@pytest.mark.parametrize(
    "fn_name,fn,output_qmin",
    [
        ("sigmoid", torch.sigmoid, 0),
        ("tanh", torch.tanh, -32768),
        ("gelu", torch.nn.functional.gelu, -32768),
        ("silu", torch.nn.functional.silu, -32768),
    ],
)
def test_pwl_per_fn_metrics_within_published_limits(fn_name, fn, output_qmin):
    input_encoding = _enc(8.0 / 32767, -32768, 32767)
    output_encoding = _out_enc(
        8.0 / 32767 if fn_name in {"gelu", "silu"} else 1.0 / 32767,
        output_qmin,
        32767,
    )

    _, _, metrics = generate_pwl_lut_for_export(
        fn, input_encoding, output_encoding, enforce_quality=False, fn_name=fn_name
    )
    assert_pwl_metrics_within_limits(metrics, fn_name=fn_name, label=f"PWL[{fn_name}]")
    assert metrics["cosine_similarity"] >= PWL_VS_ANALYTIC_MIN_COSINE_SIMILARITY

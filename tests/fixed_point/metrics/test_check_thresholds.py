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
from pathlib import Path

from scripts.fixed_point.check_thresholds import (
    check_against_baseline,
    check_quality_report_against_baseline,
)


def test_check_against_baseline_passes_when_within_limits():
    baseline = {
        "metrics": {
            "fp32_qdq_vs_int16_fixed_eval": {
                "cosine_similarity_min": 0.9998,
                "max_error_lsb_max": 1.0,
            }
        }
    }
    report = {
        "pairwise": {
            "fp32_qdq_vs_int16_fixed_eval": {
                "cosine_similarity": 0.99995,
                "max_error_lsb": 0.5,
            }
        }
    }
    assert check_against_baseline(report, baseline) == []


def test_check_against_baseline_detects_cosine_failure():
    baseline = {
        "metrics": {
            "fp32_qdq_vs_fp16_qdq": {"cosine_similarity_min": 0.9999},
        }
    }
    report = {
        "global": {
            "fp32_qdq_vs_fp16_qdq": {"cosine_similarity": 0.99},
        }
    }
    failures = check_against_baseline(report, baseline)
    assert failures[0][1] == "cosine_similarity"


def test_check_quality_report_mobilenet_and_softmax():
    baseline = {
        "quality_report": {
            "mobilenet_v2": {"int16_vs_fp32_qdq_cosine_min": 0.99},
            "int16_single_op": {
                "Linear → Softmax (PWL exp)": {"min_cosine": 0.999},
            },
        }
    }
    report = {
        "mobilenet_v2": {
            "cases": [
                {
                    "case": "Mock / INT16 vs FP32_QDQ",
                    "metrics": {"cosine_similarity": 0.995},
                }
            ]
        },
        "int16_vs_fp32": {
            "cases": [
                {
                    "case": "Linear → Softmax (PWL exp)",
                    "stress": "deploy-8b",
                    "metrics": {"cosine_similarity": 0.9995, "max_error_lsb": 1.0},
                }
            ]
        },
    }
    assert check_quality_report_against_baseline(report, baseline) == []


def test_check_quality_report_qat_summary():
    baseline = {
        "quality_report": {
            "mobilenet_v2_ptq_qat": {
                "mock_qat_cosine_min": 0.99,
                "mock_qat_status": "PASS",
            },
        }
    }
    report = {
        "mobilenet_v2_ptq_qat": {
            "qat_summary": {
                "ptq_only_cosine": 0.995,
                "qat_cosine": 0.996,
                "status": "PASS",
            },
            "cases": [],
        }
    }
    assert check_quality_report_against_baseline(report, baseline) == []


def test_check_against_baseline_fixed_scale_pair():
    baseline = {
        "metrics": {
            "fp32_qdq_vs_fixed_scale_qdq": {"cosine_similarity_min": 0.9998},
        }
    }
    report = {
        "pairwise": {
            "fp32_qdq_vs_fixed_scale_qdq": {"cosine_similarity": 0.99985},
        }
    }
    assert check_against_baseline(report, baseline) == []


def test_check_quality_report_mobilenet_fixed_scale():
    baseline = {
        "quality_report": {
            "mobilenet_v2": {"fixed_scale_vs_fp32_qdq_cosine_min": 0.9998},
        }
    }
    report = {
        "mobilenet_v2": {
            "cases": [
                {
                    "case": "MockMobileNetV2 / fixed_scale_qdq vs FP32_QDQ",
                    "metrics": {"cosine_similarity": 0.9999},
                }
            ]
        },
    }
    assert check_quality_report_against_baseline(report, baseline) == []


def test_baseline_json_is_valid():
    path = Path(__file__).resolve().parents[3] / "scripts" / "fixed_point" / "baseline.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "fp32_qdq_vs_int16_fixed_eval" in data["metrics"]
    assert "fp32_qdq_vs_fixed_scale_qdq" in data["metrics"]
    assert "quality_report" in data
    assert "fixed_scale_vs_fp32_qdq_cosine_min" in data["quality_report"]["mobilenet_v2"]

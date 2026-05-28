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
"""Compare metrics against ``baseline.json`` (compare_modes or quality_report)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

Failure = Tuple[str, str, Any, Any]


def _load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def check_against_baseline(report: Dict[str, Any], baseline: Dict[str, Any]) -> List[Failure]:
    """Validate a compare_modes JSON report (pairwise / global layout)."""

    failures: List[Failure] = []
    global_metrics = report.get("global", report.get("pairwise", {}))
    expected_metrics = baseline.get("metrics", {})

    for pair_key, expected in expected_metrics.items():
        actual = global_metrics.get(pair_key, {})
        if not actual:
            failures.append((pair_key, "missing", None, expected))
            continue
        min_cos = expected.get("cosine_similarity_min")
        if min_cos is not None:
            value = actual.get("cosine_similarity")
            if value is None or value < min_cos:
                failures.append((pair_key, "cosine_similarity", value, min_cos))
        max_lsb = expected.get("max_error_lsb_max")
        if max_lsb is not None:
            value = actual.get("max_error_lsb", actual.get("max_error_lsb_float"))
            if value is None or value > max_lsb:
                failures.append((pair_key, "max_error_lsb", value, max_lsb))
    return failures


def check_quality_report_against_baseline(
    report: Dict[str, Any],
    baseline: Dict[str, Any],
) -> List[Failure]:
    """Validate ``quality_report.json`` produced by ``run_quality_report.py``."""

    failures: List[Failure] = []
    qr = baseline.get("quality_report", {})

    mn_limits = qr.get("mobilenet_v2", {})
    mn_sec = report.get("mobilenet_v2", {})
    if not mn_sec.get("skipped") and mn_limits:
        for case in mn_sec.get("cases", []):
            cos = case.get("metrics", {}).get("cosine_similarity")
            if cos is None:
                continue
            if "INT16 vs FP32" in case.get("case", ""):
                floor = mn_limits.get("int16_vs_fp32_qdq_cosine_min", 0.99)
                if cos < floor:
                    failures.append(("mobilenet_v2/int16", "cosine_similarity", cos, floor))
            if "FP16 vs FP32" in case.get("case", ""):
                floor = mn_limits.get("fp16_vs_fp32_qdq_cosine_min", 0.999)
                if cos < floor:
                    failures.append(("mobilenet_v2/fp16", "cosine_similarity", cos, floor))
            if "fixed_scale" in case.get("case", "").lower() and "fp32" in case.get("case", "").lower():
                floor = mn_limits.get("fixed_scale_vs_fp32_qdq_cosine_min", 0.9998)
                if cos < floor:
                    failures.append(("mobilenet_v2/fixed_scale", "cosine_similarity", cos, floor))

    ptq_limits = qr.get("mobilenet_v2_ptq_qat", {})
    ptq_sec = report.get("mobilenet_v2_ptq_qat", {})
    if not ptq_sec.get("skipped") and ptq_limits:
        qs = ptq_sec.get("qat_summary") or {}
        qat_cos = qs.get("qat_cosine")
        qat_floor = ptq_limits.get("mock_qat_cosine_min", 0.99)
        if qat_cos is not None and qat_cos < qat_floor:
            failures.append(("qat/mock", "cosine_similarity", qat_cos, qat_floor))
        expected_status = ptq_limits.get("mock_qat_status")
        if expected_status and qs.get("status") != expected_status:
            failures.append(("qat/mock", "status", qs.get("status"), expected_status))
        ptq_only = qs.get("ptq_only_cosine")
        if (
            ptq_only is not None
            and qat_cos is not None
            and qat_cos + 1e-6 < ptq_only
        ):
            failures.append(("qat/mock", "cosine_vs_ptq", qat_cos, ptq_only))

        for case in ptq_sec.get("cases", []):
            if case.get("status") in ("SKIP", "FAIL"):
                continue
            cos = case.get("metrics", {}).get("cosine_similarity")
            name = case.get("case", "")
            if "PTQ vanilla" in name:
                floor = ptq_limits.get("mock_vanilla_cosine_min", 0.99)
                if cos is not None and cos < floor:
                    failures.append(("ptq/vanilla", "cosine_similarity", cos, floor))
            if "combined PTQ" in name:
                floor = ptq_limits.get("mock_combined_fallback_cosine_min", 0.99)
                if cos is not None and cos < floor:
                    failures.append(("ptq/combined", "cosine_similarity", cos, floor))

    single_limits = qr.get("int16_single_op", {})
    int16_sec = report.get("int16_vs_fp32", {})
    if not int16_sec.get("skipped") and single_limits:
        for case in int16_sec.get("cases", []):
            name = case.get("case", "")
            if name not in single_limits:
                continue
            expected = single_limits[name]
            cos = case.get("metrics", {}).get("cosine_similarity")
            min_cos = expected.get("min_cosine")
            if min_cos is not None and cos is not None and cos < min_cos:
                failures.append((f"int16_single_op/{name}", "cosine_similarity", cos, min_cos))
            max_lsb_lim = expected.get("max_error_lsb")
            if max_lsb_lim is not None and case.get("stress") == "deploy-8b":
                max_lsb = case.get("metrics", {}).get("max_error_lsb")
                if max_lsb is not None and max_lsb > max_lsb_lim:
                    failures.append((f"int16_single_op/{name}", "max_error_lsb", max_lsb, max_lsb_lim))

    return failures


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "report",
        nargs="?",
        help="JSON report path (compare_modes or quality_report.json)",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="baseline.json path (default: alongside this script)",
    )
    parser.add_argument(
        "--quality-report",
        action="store_true",
        help="Treat ``report`` as quality_report.json (uses quality_report section of baseline)",
    )
    parser.add_argument(
        "--from-quality-report",
        metavar="PATH",
        help="Shorthand: ``PATH`` is quality_report.json (implies --quality-report)",
    )
    args = parser.parse_args(argv)

    report_path = args.from_quality_report or args.report
    if report_path is None:
        parser.error("report path required (or use --from-quality-report PATH)")
    if args.from_quality_report:
        args.quality_report = True

    baseline_path = args.baseline
    if baseline_path is None:
        baseline_path = str(Path(__file__).with_name("baseline.json"))

    report = _load_json(report_path)
    baseline = _load_json(baseline_path)

    if args.quality_report or "pwl_vs_analytic" in report:
        failures = check_quality_report_against_baseline(report, baseline)
    else:
        failures = check_against_baseline(report, baseline)

    if not failures:
        print("All metrics within baseline thresholds.")
        return 0

    print("Threshold failures:")
    for scope, metric, actual, expected in failures:
        print(f"  {scope} {metric}: actual={actual!r} expected>={expected!r}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

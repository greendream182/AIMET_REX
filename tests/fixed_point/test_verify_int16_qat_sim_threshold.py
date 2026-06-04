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
"""Threshold-decision tests for ``verify_int16_qat_sim`` PASS/FAIL semantics.

Covers the pure decision helper ``_decide_verify_int16_qat_sim_pass`` only —
no MRNN graph / CUDA / QAT training is required, so these run anywhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

# Import the script as a module without executing argparse / dataset paths.
import importlib

# ``int16_whole_graph_vs_float_native`` pulls torch + heavy deps at import
# time; if those are missing we skip these threshold tests rather than fail.
try:
    int16_whole_graph_vs_float_native = importlib.import_module(
        "int16_whole_graph_vs_float_native"
    )
except Exception as exc:  # noqa: BLE001
    pytest.skip(
        f"int16_whole_graph_vs_float_native import unavailable: {exc}",
        allow_module_level=True,
    )

_decide = int16_whole_graph_vs_float_native._decide_verify_int16_qat_sim_pass


def _base_results(top1: float | str = 0.93) -> dict:
    return {
        "readiness": "OK",
        "backward": "OK",
        "qat_steps": "OK",
        "int16_fixed_eval_top1": top1,
    }


def test_pass_when_all_ok_and_top1_above_threshold():
    passed, pass_reasons, fail_reasons, delta = _decide(
        _base_results(0.93), min_top1=0.90, max_delta_pp=None, float_native_top1=None,
    )
    assert passed is True
    assert fail_reasons == []
    assert delta is None
    assert any("readiness" in r for r in pass_reasons)


def test_fail_when_top1_below_threshold():
    passed, _pass, fail_reasons, _delta = _decide(
        _base_results(0.88), min_top1=0.90, max_delta_pp=None, float_native_top1=None,
    )
    assert passed is False
    assert any("top1=88.00%" in r and "<min_top1=90.00%" in r for r in fail_reasons)


def test_fail_when_readiness_blocked():
    results = _base_results(0.95)
    results["readiness"] = "FAILED: missing_output_quantizer=...]"
    passed, _pass, fail_reasons, _delta = _decide(
        results, min_top1=0.90, max_delta_pp=None, float_native_top1=None,
    )
    assert passed is False
    assert any(r.startswith("readiness=") for r in fail_reasons)


def test_fail_when_backward_or_qat_steps_failed():
    for key in ("backward", "qat_steps"):
        results = _base_results(0.95)
        results[key] = "FAILED: boom"
        passed, _pass, fail_reasons, _delta = _decide(
            results, min_top1=0.90, max_delta_pp=None, float_native_top1=None,
        )
        assert passed is False
        assert any(r.startswith(f"{key}=") for r in fail_reasons)


def test_fail_when_eval_top1_not_float():
    results = _base_results("FAILED: CUDA OOM")
    passed, _pass, fail_reasons, delta = _decide(
        results, min_top1=0.90, max_delta_pp=None, float_native_top1=None,
    )
    assert passed is False
    assert delta is None
    assert any("int16_fixed_eval_top1" in r for r in fail_reasons)


def test_delta_pp_enforced_when_provided():
    # baseline 0.9556, eval 0.94 → delta = -1.56 pp
    # max_delta_pp = 1.0 → fail; max_delta_pp = 2.0 → pass
    res_fail = _decide(
        _base_results(0.94),
        min_top1=0.90,
        max_delta_pp=1.0,
        float_native_top1=0.9556,
    )
    assert res_fail[0] is False
    assert any("delta_pp" in r and "<-1.00" in r for r in res_fail[2])
    assert res_fail[3] == pytest.approx(-1.56, abs=0.05)

    res_pass = _decide(
        _base_results(0.94),
        min_top1=0.90,
        max_delta_pp=2.0,
        float_native_top1=0.9556,
    )
    assert res_pass[0] is True
    assert res_pass[2] == []


def test_delta_pp_ignored_when_baseline_missing():
    # max_delta_pp given but no float_native_top1: delta cannot be evaluated;
    # PASS still depends only on the other criteria.
    passed, pass_reasons, fail_reasons, delta = _decide(
        _base_results(0.91),
        min_top1=0.90,
        max_delta_pp=0.5,
        float_native_top1=None,
    )
    assert passed is True
    assert delta is None
    assert not any("delta_pp" in r for r in pass_reasons + fail_reasons)

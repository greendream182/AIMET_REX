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
"""Local unit tests for QuantGRU adapter helpers (no quant_gru required).

These tests cover pure mapping logic that is exercised on every wrapper
forward / compute_encodings entry/exit. Without these the only coverage was
``@requires_quant_gru_cuda`` tests that skip locally — meaning a regression in
``ExecutionMode`` enum handling or FIXED_SCALE_QDQ aliasing would only surface
in CI.
"""

from __future__ import annotations

import pytest

from aimet_torch.fixed_point import ExecutionMode
from aimet_torch.fixed_point.quantgru_adapter import (
    _SUPPORTED_MODES,
    resolve_quantgru_mode_str,
)


def test_resolve_aliases_fixed_scale_qdq_to_fp32_qdq():
    """plan §0.2.1: FIXED_SCALE_QDQ collapses to fp32_qdq for QuantGRU."""
    assert resolve_quantgru_mode_str(ExecutionMode.FIXED_SCALE_QDQ) == "fp32_qdq"


@pytest.mark.parametrize(
    "mode,expected",
    [
        (ExecutionMode.FP32_QDQ, "fp32_qdq"),
        (ExecutionMode.FP16_QDQ, "fp16_qdq"),
        (ExecutionMode.INT16_FIXED_EVAL, "int16_fixed_eval"),
        (ExecutionMode.INT16_FIXED_QAT_SIM, "int16_fixed_qat_sim"),
    ],
)
def test_resolve_passes_through_other_modes(mode: ExecutionMode, expected: str):
    """All non-FIXED_SCALE_QDQ ExecutionMode values map 1:1 to .value."""
    assert resolve_quantgru_mode_str(mode) == expected
    assert resolve_quantgru_mode_str(mode) == mode.value


def test_resolved_modes_are_all_in_contract_v1_supported_modes():
    """Resolved string must always be acceptable to QuantGRU's aimet_configure.

    This is the property that makes the alias safe: every ExecutionMode, after
    resolution, lands in QuantGRU contract v1 ``_SUPPORTED_MODES``. If a future
    ExecutionMode is added without a corresponding alias, this test fails.
    """
    for mode in ExecutionMode:
        resolved = resolve_quantgru_mode_str(mode)
        assert resolved in _SUPPORTED_MODES, (
            f"ExecutionMode.{mode.name} resolves to {resolved!r} which is "
            f"not in QuantGRU contract v1 supported_modes={_SUPPORTED_MODES}. "
            "Add an alias in resolve_quantgru_mode_str() or extend the contract."
        )


def test_no_execution_mode_named_fp32():
    """Guard: ExecutionMode has no 'FP32' enum value.

    Earlier wrapper had ``if mode == ExecutionMode.FP32: ...`` which raises
    AttributeError at runtime. We removed it; this guard prevents anyone from
    re-introducing the bug under the assumption that such an enum exists.
    """
    enum_names = {m.name for m in ExecutionMode}
    assert "FP32" not in enum_names, (
        "ExecutionMode unexpectedly grew an FP32 member. If you intend to add "
        "it, also extend resolve_quantgru_mode_str() and update the wrapper "
        "forward routing in aimet_torch/v2/nn/modules/custom.py."
    )
    assert enum_names == {
        "FP32_QDQ",
        "FP16_QDQ",
        "FIXED_SCALE_QDQ",
        "INT16_FIXED_EVAL",
        "INT16_FIXED_QAT_SIM",
    }

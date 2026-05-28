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
"""Smoke tests for the reusable per-layer report renderer."""

from __future__ import annotations

from aimet_torch.fixed_point.metrics import (
    per_layer_table_to_markdown,
    render_per_layer_table,
)


_SAMPLE_ROW = {
    "module": "features.0",
    "cosine": 0.987654,
    "shape": (8, 16, 32, 32),
    "ref_rms": 1.0,
    "max_abs_err": 0.5,
    "norm_max_err": 0.5,
    "rmse": 0.123456,
    "sqnr_db": 12.34,
    "p99_abs_err": 0.444,
}


def test_render_per_layer_table_writes_header_and_row():
    captured: list[str] = []
    render_per_layer_table([_SAMPLE_ROW], cosine_key="cosine", printer=captured.append)

    assert len(captured) >= 2, captured
    # Header line carries the dynamic cosine key + Tier-1 columns.
    assert "| module | cosine | sqnr_dB | rmse | p99_abs_err | shape |" in captured[0]
    assert captured[1].startswith("|---")
    # Row contains the rendered values + escaped module name.
    row_line = captured[2]
    assert "`features.0`" in row_line
    assert "0.987654" in row_line
    assert "12.34" in row_line
    assert "8×16×32×32" in row_line


def test_per_layer_table_to_markdown_returns_string():
    md = per_layer_table_to_markdown(
        [_SAMPLE_ROW, _SAMPLE_ROW],
        cosine_key="isolated_cosine",
    )
    lines = md.splitlines()
    # Header + separator + 2 data rows.
    assert len(lines) == 4
    assert "isolated_cosine" in lines[0]


def test_render_per_layer_table_handles_inf_sqnr_and_missing_keys():
    captured: list[str] = []
    row = {
        "module": "noisy",
        "cosine": 1.0,
        "shape": (4,),
        "sqnr_db": float("inf"),
        # rmse / p99_abs_err intentionally missing — renderer should fall
        # back to nan formatting without raising.
    }
    render_per_layer_table([row], printer=captured.append)
    assert "+inf" in captured[2]
    assert "noisy" in captured[2]

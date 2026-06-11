"""Unit tests for M/2^n scale snapping (M_Po2)."""

from __future__ import annotations

import math

import pytest

from aimet_torch.m_po2_quantization import snap_scale_to_m_po2


@pytest.mark.parametrize(
    "scale",
    [0.00390625, 0.0078125, 0.0001, 0.5, 1.0 / 65536.0],
)
def test_snap_scale_to_m_po2_recovers_grid(scale: float):
    snapped, m, r = snap_scale_to_m_po2(scale)
    assert m >= 0
    assert r >= 0
    approx = m / float(1 << r)
    assert math.isclose(snapped, approx, rel_tol=0.0, abs_tol=1e-15)
    rel_err = abs(scale - snapped) / max(abs(scale), 1e-30)
    assert rel_err < 0.01 or scale == snapped

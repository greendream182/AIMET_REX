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
"""Tests for ``diagnose_int16_readiness`` (QuantGRU INT16 plan Phase 3)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxscript")
import torch.nn as nn  # noqa: E402

import aimet_torch.fixed_point.kernels  # noqa: F401, E402
from aimet_torch.fixed_point.diagnose import (  # noqa: E402
    diagnose_int16_readiness,
    is_int16_ready,
)
from aimet_torch.fixed_point.sim_utils import ensure_output_quantizers_for_int16_eval  # noqa: E402


def _has_quant_gru() -> bool:
    try:
        import quant_gru  # noqa: F401

        return True
    except ImportError:
        return False


requires_quant_gru_cuda = pytest.mark.skipif(
    not (_has_quant_gru() and torch.cuda.is_available()),
    reason="quant_gru + CUDA required",
)


class TinyConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 3, padding=1)
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = x.mean(dim=(2, 3))
        return self.fc(x)


@pytest.fixture(scope="module")
def gru_sim():
    if not (_has_quant_gru() and torch.cuda.is_available()):
        pytest.skip("quant_gru + CUDA required")

    from aimet_torch import model_preparer
    from aimet_torch.v2.quantsim import quantsim
    from quant_gru import QuantGRU

    device = torch.device("cuda")
    model = nn.Sequential(
        QuantGRU(8, 8, batch_first=True),
    ).to(device).eval()
    dummy = torch.randn(2, 5, 8, device=device)

    prepared = model_preparer.prepare_model(model)
    sim = quantsim.QuantizationSimModel(
        prepared,
        dummy_input=dummy,
        quant_scheme="tf",
        default_output_bw=8,
        default_param_bw=8,
    )
    ensure_output_quantizers_for_int16_eval(sim)
    sim.model.to(device).eval()

    def _calib(m):
        with torch.no_grad():
            for _ in range(3):
                m(torch.randn_like(dummy))

    sim.compute_encodings(_calib)
    return sim


def test_diagnose_report_keys():
    sim = SimpleNamespace(model=nn.Sequential(nn.Identity()))
    report = diagnose_int16_readiness(sim)
    assert set(report.keys()) == {
        "missing_output_quantizer",
        "uninitialized_encoding",
        "uncalibrated_quantgru",
        "missing_fixed_kernel",
    }


@requires_quant_gru_cuda
def test_calibrated_gru_sim_has_no_uncalibrated_quantgru(gru_sim):
    report = diagnose_int16_readiness(gru_sim)
    assert report["uncalibrated_quantgru"] == []


@requires_quant_gru_cuda
def test_is_int16_ready_after_calibration(gru_sim):
    assert is_int16_ready(gru_sim)

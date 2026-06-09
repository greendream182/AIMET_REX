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
        "unsupported_activation_bitwidth",
        "blackbox_native_ops",
    }


def test_diagnose_flags_unsupported_activation_bitwidth_blocks_readiness():
    """A 16+16 MAC-reduction Linear violates the W5 SYS-FU-1.B combo gate
    (combined bitwidth = 32 > ``REQUANTIZING_COMBO_BITWIDTH_BUDGET = 24``).
    The diagnose pass must surface it as an explicit readiness blocker so
    callers fix the config before forward, rather than getting a silent
    INT32-saturation drift at large reduction depth.

    PR-3 (2026-06-09) replaced the legacy "any 16-bit activation" fixture
    with a full 16+16 reduction. The asymmetric subset (16+8 or 8+16) is
    now validated by the combo gate and does NOT block readiness — that
    expected-pass case is covered by
    ``test_diagnose_accepts_asymmetric_16bit_combo`` below.
    """

    # pylint: disable=import-outside-toplevel
    from aimet_torch.v2.nn import QuantizedLinear
    from aimet_torch.v2.quantization.affine import Quantize

    m = QuantizedLinear(4, 4)
    m.input_quantizers[0] = Quantize((), 16, symmetric=True)
    m.param_quantizers["weight"] = Quantize((m.out_features, 1), 16, symmetric=True)
    m.output_quantizers[0] = Quantize((), 16, symmetric=True)
    m.input_quantizers[0].min = nn.Parameter(torch.tensor(-2.0))
    m.input_quantizers[0].max = nn.Parameter(torch.tensor(2.0))
    m.param_quantizers["weight"].min = nn.Parameter(torch.full((m.out_features, 1), -0.5))
    m.param_quantizers["weight"].max = nn.Parameter(torch.full((m.out_features, 1), 0.5))
    m.output_quantizers[0].min = nn.Parameter(torch.tensor(-2.0))
    m.output_quantizers[0].max = nn.Parameter(torch.tensor(2.0))

    sim = SimpleNamespace(model=nn.Sequential(m))
    report = diagnose_int16_readiness(sim)

    assert report["unsupported_activation_bitwidth"], (
        "16+16 MAC-reduction must surface in the readiness report"
    )
    label, cls_name, bw = report["unsupported_activation_bitwidth"][0]
    assert cls_name == "QuantizedLinear"
    assert bw == 16
    assert not is_int16_ready(sim)


def test_diagnose_accepts_asymmetric_16bit_combo():
    """16+8 / 8+16 combos validated by W5.1 must NOT block readiness.

    Guards against a regression of the combo gate's lower bound: if
    someone tightens ``_REQUANTIZING_COMBO_VALIDATED_BITWIDTHS`` back to
    ``(8,)`` or shrinks ``REQUANTIZING_COMBO_BITWIDTH_BUDGET`` below 24,
    this test fails fast.
    """

    # pylint: disable=import-outside-toplevel
    from aimet_torch.v2.nn import QuantizedLinear
    from aimet_torch.v2.quantization.affine import Quantize

    m = QuantizedLinear(4, 4)
    m.input_quantizers[0] = Quantize((), 16, symmetric=True)
    m.param_quantizers["weight"] = Quantize((m.out_features, 1), 8, symmetric=True)
    m.output_quantizers[0] = Quantize((), 8, symmetric=True)
    m.input_quantizers[0].min = nn.Parameter(torch.tensor(-2.0))
    m.input_quantizers[0].max = nn.Parameter(torch.tensor(2.0))
    m.param_quantizers["weight"].min = nn.Parameter(torch.full((m.out_features, 1), -0.5))
    m.param_quantizers["weight"].max = nn.Parameter(torch.full((m.out_features, 1), 0.5))
    m.output_quantizers[0].min = nn.Parameter(torch.tensor(-2.0))
    m.output_quantizers[0].max = nn.Parameter(torch.tensor(2.0))

    sim = SimpleNamespace(model=nn.Sequential(m))
    report = diagnose_int16_readiness(sim)

    assert report["unsupported_activation_bitwidth"] == [], (
        "16+8 (asymmetric, combined=24) must NOT block readiness "
        f"under the W5 SYS-FU-1.B combo gate; got {report['unsupported_activation_bitwidth']!r}"
    )


@requires_quant_gru_cuda
def test_calibrated_gru_sim_has_no_uncalibrated_quantgru(gru_sim):
    report = diagnose_int16_readiness(gru_sim)
    assert report["uncalibrated_quantgru"] == []


@requires_quant_gru_cuda
def test_is_int16_ready_after_calibration(gru_sim):
    assert is_int16_ready(gru_sim)

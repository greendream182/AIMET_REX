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
"""R3 前端分段：BandConverter (MatMul) + Conv INT16 smoke（无 STFT / librosa 依赖）。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxscript")
import torch.nn as nn  # noqa: E402

import aimet_torch.fixed_point.kernels  # noqa: F401, E402
from aimet_torch.fixed_point import (  # noqa: E402
    ExecutionMode,
    diagnose_int16_readiness,
    ensure_output_quantizers_for_int16_eval,
    is_int16_ready,
    quant_execution_mode,
)
from aimet_torch.fixed_point.registry import get_fixed_kernel  # noqa: E402
from aimet_torch._base.nn.modules import custom  # noqa: E402


def _has_quant_gru() -> bool:
    try:
        import quant_gru  # noqa: F401

        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not (_has_quant_gru() and torch.cuda.is_available()),
    reason="BandConverter segment requires quant_gru + CUDA (same env as MRNN e2e)",
)


@pytest.fixture(scope="module")
def band_matrix_dir(tmp_path_factory):
    """Minimal ERB band matrices for BandConverter."""
    band_num, freq_bins = 32, 64
    tmp_path = tmp_path_factory.mktemp("band_matrices")
    torch.save(
        torch.randn(freq_bins, band_num),
        tmp_path / f"to_band_matrix_erb_{band_num}_{freq_bins}.pt",
    )
    torch.save(
        torch.randn(band_num, freq_bins),
        tmp_path / f"inv_to_band_matrix_erb_{band_num}_{freq_bins}.pt",
    )
    return str(tmp_path), band_num, freq_bins


def _import_band_converter():
    examples = Path(__file__).resolve().parents[2] / "examples"
    if str(examples) not in sys.path:
        sys.path.insert(0, str(examples))
    from common.fft2band import BandConverter

    return BandConverter


def _build_calibrated_sim(model: nn.Module, dummy: torch.Tensor):
    from aimet_torch import model_preparer
    from aimet_torch.v2.quantsim import quantsim

    prepared = model_preparer.prepare_model(model)
    sim = quantsim.QuantizationSimModel(
        prepared,
        dummy_input=dummy,
        quant_scheme="tf",
        default_output_bw=8,
        default_param_bw=8,
    )
    ensure_output_quantizers_for_int16_eval(sim)
    sim.model.to(dummy.device).eval()

    def _calib(m: nn.Module) -> None:
        with torch.no_grad():
            for _ in range(4):
                m(torch.randn_like(dummy))

    sim.compute_encodings(_calib)
    return sim


class BandConvSegment(nn.Module):
    """BandConverter → Conv2d，模拟 MRNN 前端 mag→band→conv 子图。"""

    def __init__(self, matrix_dir: str, band_num: int, freq_bins: int):
        super().__init__()
        BandConverter = _import_band_converter()
        self.fft2band = BandConverter(
            band_num=band_num, freq_bins=freq_bins, matrix_dir=matrix_dir
        )
        self.conv = nn.Conv2d(1, 8, kernel_size=3, padding=1)

    def forward(self, mag: torch.Tensor) -> torch.Tensor:
        # mag: (B, T, F) — 跳过 STFT / Hypot，直接喂幅度谱
        band = self.fft2band(mag)
        band = band.unsqueeze(1)
        return self.conv(band)


def test_matmul_int16_kernel_registered():
    """R3 前置：MatMul INT16 kernel 已注册。"""
    get_fixed_kernel(custom.MatMul)


@pytest.fixture(scope="module")
def band_segment_sim(band_matrix_dir):
    device = torch.device("cuda")
    matrix_dir, band_num, freq_bins = band_matrix_dir
    model = BandConvSegment(matrix_dir, band_num, freq_bins).to(device).eval()
    dummy = torch.randn(2, 16, freq_bins, device=device)
    return {"sim": _build_calibrated_sim(model, dummy), "dummy": dummy}


def test_band_segment_diagnose_ready(band_segment_sim):
    report = diagnose_int16_readiness(band_segment_sim["sim"])
    assert report["uncalibrated_quantgru"] == []
    assert report["missing_fixed_kernel"] == [], report
    assert is_int16_ready(band_segment_sim["sim"]), report


def test_band_segment_int16_forward(band_segment_sim):
    sim = band_segment_sim["sim"]
    x = band_segment_sim["dummy"]
    with torch.no_grad(), quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y = sim.model(x)
    assert y.shape[0] == x.shape[0]
    assert y.dtype in (torch.int32, torch.float32)

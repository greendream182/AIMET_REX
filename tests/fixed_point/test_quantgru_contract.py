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
"""AIMET-side bidirectional CI conformance for QuantGRU contract v1 (plan §4.4)."""

from __future__ import annotations

import inspect

import pytest

torch = pytest.importorskip("torch")

try:
    from quant_gru import QuantGRU
except ImportError as exc:
    pytest.skip(f"quant_gru not installed: {exc}", allow_module_level=True)


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="QuantGRU CUDA path required",
)

_EXPECTED_FLAGS = ("use_quantization", "calibrating", "export_mode", "export_format")


def _calibrate_minmax(gru: QuantGRU, x: torch.Tensor) -> None:
    gru.calibrating = True
    gru.use_quantization = False
    with torch.no_grad():
        gru(x)
    gru.calibrating = False
    gru.finalize_calibration(verbose=False)
    gru.use_quantization = True


@pytest.fixture
def calibrated_gru():
    gru = QuantGRU(8, 8, batch_first=True).cuda()
    x = torch.randn(2, 5, 8, device="cuda")
    _calibrate_minmax(gru, x)
    return gru, x


def test_forward_signature_stable():
    sig = inspect.signature(QuantGRU.forward)
    assert "input" in sig.parameters
    assert "hx" in sig.parameters
    assert len(sig.parameters) >= 2


def test_public_flags_exist_with_defaults():
    gru = QuantGRU(4, 4, batch_first=True)
    for flag in _EXPECTED_FLAGS:
        assert hasattr(gru, flag)
    assert gru.use_quantization is False
    assert gru.calibrating is False
    assert gru.export_mode is False
    assert gru.export_format in ("float", "qdq")


def test_aimet_capabilities_major_one():
    gru = QuantGRU(4, 4, batch_first=True)
    caps = gru.aimet_capabilities()
    major, minor = map(int, caps["adapter_version"].split("."))
    assert major == 1
    assert minor >= 0
    assert caps["forward_io_dtype"] == "float32"
    assert caps["forward_io_device"] == "cuda"
    assert caps["supports_forward_quantized"] is True
    for mode in ("fp32", "fp32_qdq", "int16_fixed_eval", "int16_fixed_qat_sim", "calibrating"):
        assert mode in caps["supported_modes"]


@requires_cuda
def test_forward_io_dtype_device(calibrated_gru):
    gru, x = calibrated_gru
    out, hn = gru(x)
    assert out.dtype == torch.float32
    assert hn.dtype == torch.float32
    assert out.is_cuda and hn.is_cuda


def test_get_io_quant_meta_uncalibrated_raises():
    gru = QuantGRU(4, 4, batch_first=True)
    with pytest.raises(RuntimeError, match="not calibrated"):
        gru.get_io_quant_meta()


@requires_cuda
def test_get_io_quant_meta_schema(calibrated_gru):
    gru, _ = calibrated_gru
    meta = gru.get_io_quant_meta()
    assert set(meta.keys()) == {"input", "output", "hidden"}
    for key in meta:
        entry = meta[key]
        assert {"scale", "zp", "bitwidth", "is_symmetric"} <= set(entry.keys())
        assert entry["scale"] > 0


@requires_cuda
def test_forward_quantized_returns_integer_tensors(calibrated_gru):
    gru, x = calibrated_gru
    out_q, hn_q = gru.forward_quantized(x)
    assert out_q.dtype in (torch.int16, torch.int32)
    assert hn_q.dtype in (torch.int16, torch.int32)
    assert out_q.shape == x.shape


@requires_cuda
def test_forward_quantized_bit_exact_with_deploy_kernel(calibrated_gru):
    """AIMET INT16 边界：forward_quantized dequant 后与 forward fp32 一致。"""
    gru, x = calibrated_gru
    meta = gru.get_io_quant_meta()
    fp_out, fp_hn = gru(x)
    out_q, hn_q = gru.forward_quantized(x)

    scale_o = meta["output"]["scale"]
    zp_o = meta["output"]["zp"]
    scale_h = meta["hidden"]["scale"]
    zp_h = meta["hidden"]["zp"]

    recon_out = (out_q.to(torch.float32) - zp_o) * scale_o
    recon_hn = (hn_q.to(torch.float32) - zp_h) * scale_h
    assert torch.allclose(recon_out, fp_out, rtol=0, atol=1e-5)
    assert torch.allclose(recon_hn, fp_hn, rtol=0, atol=1e-5)


@requires_cuda
def test_aimet_configure_all_supported_modes():
    for mode in QuantGRU._AIMET_SUPPORTED_MODES:
        gru = QuantGRU(4, 4, batch_first=True)
        gru.aimet_configure(mode)

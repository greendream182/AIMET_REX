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
"""PWL grid alignment and effective-scale offline fit (ADR-015 / §3.0)."""

import torch

from aimet_torch.fixed_point import InputEncoding, generate_pwl_lut
from aimet_torch.fixed_point.offline.lut_gen import (
    _effective_scale_fp64,
    align_op_quant_grid_to_lut_quant_grid,
    encodings_share_quant_grid,
)
from aimet_torch.fixed_point.offline.scale_fixed import quantize_scale_to_m_rshift


def _enc(scale: float, zp: int = 0) -> InputEncoding:
    return InputEncoding(
        scale=torch.tensor(scale, dtype=torch.float32),
        zero_point=torch.tensor(zp, dtype=torch.int32),
        qmin=-32768,
        qmax=32767,
    )


def test_encodings_share_quant_grid_same_scale():
    enc = _enc(0.02)
    assert encodings_share_quant_grid(enc, enc)


def test_align_op_grid_changes_q_when_scales_differ():
    op_enc = _enc(0.04, zp=0)
    lut_enc = _enc(0.02, zp=0)
    q_op = torch.tensor([100, 200], dtype=torch.int32)

    q_lut = align_op_quant_grid_to_lut_quant_grid(q_op, op_enc, lut_enc)

    assert not torch.equal(q_op, q_lut)


def test_effective_scale_differs_from_calibration_float():
    scale = torch.tensor(0.03, dtype=torch.float32)
    device = scale.device
    eff = _effective_scale_fp64(scale, device).item()
    assert abs(eff - scale.item()) > 1e-9


def test_generate_pwl_lut_uses_effective_scale_not_raw_float(monkeypatch):
    """Offline fit with discrete scale must change q_b vs naive float-scale fit."""
    import aimet_torch.fixed_point.offline.lut_gen as lut_gen_mod

    in_enc = _enc(0.05)
    out_enc = InputEncoding(
        scale=torch.tensor(0.02, dtype=torch.float32),
        zero_point=torch.tensor(0, dtype=torch.int32),
        qmin=-32768,
        qmax=32767,
    )

    pwl_discrete = generate_pwl_lut(torch.tanh, in_enc, out_enc, num_segments=4)

    def _raw_scale(scale, device):
        return scale.to(device=device, dtype=torch.float64)

    monkeypatch.setattr(lut_gen_mod, "_effective_scale_fp64", _raw_scale)
    pwl_float = generate_pwl_lut(torch.tanh, in_enc, out_enc, num_segments=4)

    assert not torch.equal(pwl_discrete["q_b"], pwl_float["q_b"])

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
"""Per-channel activation align + Conv2d INT16 dispatch."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

import aimet_torch.fixed_point.kernels  # noqa: F401, E402

from aimet_torch.fixed_point.channel_align import (  # noqa: E402
    align_per_channel_activation_for_conv_input,
    is_per_channel_activation,
)
from aimet_torch.fixed_point.metrics.flags import int16_eval_allow_debug_float  # noqa: E402
from aimet_torch.fixed_point.requantize import saturate_sim_tensor  # noqa: E402
from aimet_torch.fixed_point.tensor import Int16QuantizedTensor  # noqa: E402
from aimet_torch.fixed_point import ExecutionMode, quant_execution_mode  # noqa: E402


def _sqnr_db(ref: torch.Tensor, cand: torch.Tensor) -> float:
    diff = ref.reshape(-1).float() - cand.reshape(-1).float()
    sig = ref.reshape(-1).float().pow(2).mean().item()
    noise = diff.pow(2).mean().item()
    if noise <= 0 or sig <= 0:
        return float("inf")
    return float(10.0 * math.log10(sig / noise))


def _sim_per_channel(x: torch.Tensor, *, qmax: int = 127) -> Int16QuantizedTensor:
    c = x.shape[1]
    scales = torch.empty(c, device=x.device, dtype=torch.float32)
    for ch in range(c):
        amax = float(x[:, ch].abs().max().item())
        scales[ch] = max(amax, 1e-6) / qmax
    scale_b = scales.view(1, c, 1, 1)
    q = torch.round(x / scale_b)
    int_repr = saturate_sim_tensor(q, -qmax, qmax)
    return Int16QuantizedTensor(
        int_repr=int_repr,
        scale=scale_b,
        zero_point=torch.zeros(1, c, 1, 1, device=x.device, dtype=torch.int32),
        qmin=-qmax,
        qmax=qmax,
        axis=1,
    )


def test_is_per_channel_activation():
    x = torch.randn(2, 4, 3, 3)
    pc = _sim_per_channel(x)
    assert is_per_channel_activation(pc)
    aligned = align_per_channel_activation_for_conv_input(pc)
    assert not is_per_channel_activation(aligned)
    assert aligned.scale.numel() == 1


def test_align_preserves_dequant_approx():
    torch.manual_seed(0)
    x = torch.randn(1, 8, 4, 4)
    pc = _sim_per_channel(x)
    with int16_eval_allow_debug_float():
        ref = pc.to_float()
        aligned = align_per_channel_activation_for_conv_input(pc)
        got = aligned.to_float()
    assert _sqnr_db(ref, got) > 40.0


class _TinyConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(8, 4, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        return self.conv(x)


def test_conv2d_dispatch_accepts_per_channel_input():
    """Per-channel input must not crash requantize_int shape mismatch."""

    from aimet_torch.v2.quantsim import QuantizationSimModel

    torch.manual_seed(1)
    model = _TinyConv()
    dummy = torch.randn(1, 8, 4, 4)
    sim = QuantizationSimModel(
        model,
        dummy_input=dummy,
        default_output_bw=8,
        default_param_bw=8,
    )
    sim.model.eval()
    import aimet_torch.v2 as aimet

    with torch.no_grad(), aimet.nn.compute_encodings(sim.model):
        sim.model(dummy)

    qconv = sim.model.conv
    x_fp = torch.randn(2, 8, 4, 4)
    x_pc = _sim_per_channel(x_fp.to(next(qconv.parameters()).device))

    with int16_eval_allow_debug_float(), quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        out_fp = qconv(x_fp)
        out = qconv(x_pc)
    with int16_eval_allow_debug_float():
        y_fp = out_fp.to_float() if hasattr(out_fp, "to_float") else out_fp
        y_int = out.to_float() if hasattr(out, "to_float") else out
    assert _sqnr_db(y_fp, y_int) > 15.0

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
"""Unit tests for v1 ``fixed_scale_qdq`` helpers (no libpymo / wrapper required)."""

from types import SimpleNamespace

import pytest
import torch

from aimet_common.defs import QuantizationDataType
from aimet_torch.fixed_point import ExecutionMode, quant_execution_mode
from aimet_torch.v2.quantization.affine.backends.torch_builtins import quantize_dequantize
from aimet_torch.v1 import fixed_scale_qdq as v1_fs
from aimet_torch.v1.fixed_scale_qdq import (
    _stitch_per_channel_scale_offset,
    is_v1_fixed_scale_qdq_mode,
    try_fixed_scale_quantize_dequantize,
)


def test_is_v1_fixed_scale_mode():
    with quant_execution_mode(ExecutionMode.FP32_QDQ):
        assert not is_v1_fixed_scale_qdq_mode()
    with quant_execution_mode(ExecutionMode.FIXED_SCALE_QDQ):
        assert is_v1_fixed_scale_qdq_mode()


def test_stitch_per_channel_scale_offset():
    enc_list = [
        SimpleNamespace(delta=0.1, offset=0.0),
        SimpleNamespace(delta=0.2, offset=1.0),
    ]
    scale, offset = _stitch_per_channel_scale_offset(
        torch.Size([2, 2, 3]), enc_list, ch_axis=1, device=torch.device("cpu")
    )
    assert scale.shape == (1, 2, 1)
    assert float(scale[0, 0, 0]) == pytest.approx(0.1)
    assert float(scale[0, 1, 0]) == pytest.approx(0.2)
    assert float(offset[0, 1, 0]) == pytest.approx(1.0)


def test_try_fixed_scale_uses_m_r_path(monkeypatch):
    tensor = torch.tensor([0.12, -0.37, 0.51], dtype=torch.float32)
    scale = torch.tensor(0.125, dtype=torch.float32)
    offset = torch.tensor(0.0, dtype=torch.float32)

    monkeypatch.setattr(
        v1_fs,
        "scale_offset_from_static_grid_encoding",
        lambda _t, _q: (scale, offset),
    )

    tq = SimpleNamespace(
        enabled=True,
        bitwidth=8,
        data_type=QuantizationDataType.int,
        use_symmetric_encodings=True,
    )

    with quant_execution_mode(ExecutionMode.FP32_QDQ):
        y_fp32 = quantize_dequantize(tensor, scale, offset, -128, 127)
    with quant_execution_mode(ExecutionMode.FIXED_SCALE_QDQ):
        y_fix = try_fixed_scale_quantize_dequantize(tensor, tq)

    assert y_fix is not None
    assert torch.allclose(y_fp32, y_fix, atol=1e-4)


def test_try_fixed_scale_returns_none_in_fp32_mode(monkeypatch):
    monkeypatch.setattr(
        v1_fs,
        "scale_offset_from_static_grid_encoding",
        lambda _t, _q: (torch.tensor(0.1), torch.tensor(0.0)),
    )
    tq = SimpleNamespace(
        enabled=True,
        bitwidth=8,
        data_type=QuantizationDataType.int,
        use_symmetric_encodings=True,
    )
    with quant_execution_mode(ExecutionMode.FP32_QDQ):
        assert try_fixed_scale_quantize_dequantize(torch.randn(3), tq) is None

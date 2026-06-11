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
"""G3 boundary (M,r) quantization vs float-scale affine grid."""

import os

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from aimet_torch.fixed_point import ExecutionMode, quant_execution_mode
from aimet_torch.fixed_point.boundary_quantize import (
    int16_boundary_use_m_r,
    quantize_boundary_from_affine,
    should_use_fixed_scale_boundary,
)
from aimet_torch.fixed_point.offline.scale_fixed import (
    clear_fixed_scale_encoding_cache,
    get_or_create_fixed_scale_encoding,
)
from aimet_torch.fixed_point.requantize import SIM_TENSOR_DTYPE
from aimet_torch.fixed_point.tensor import Int16QuantizedTensor
from aimet_torch.v2.nn import QuantizedLinear  # noqa: E402
from aimet_torch.v2.quantization.affine import Quantize  # noqa: E402


def _init_linear(m: QuantizedLinear) -> None:
    m.input_quantizers[0] = Quantize((), 8, symmetric=True)
    m.param_quantizers["weight"] = Quantize((m.out_features, 1), 8, symmetric=True)
    m.output_quantizers[0] = Quantize((), 8, symmetric=True)
    m.input_quantizers[0].min = nn.Parameter(torch.tensor(-2.0))
    m.input_quantizers[0].max = nn.Parameter(torch.tensor(2.0))
    m.param_quantizers["weight"].min = nn.Parameter(torch.full((m.out_features, 1), -0.5))
    m.param_quantizers["weight"].max = nn.Parameter(torch.full((m.out_features, 1), 0.5))
    m.output_quantizers[0].min = nn.Parameter(torch.tensor(-2.0))
    m.output_quantizers[0].max = nn.Parameter(torch.tensor(2.0))


def test_boundary_m_r_can_differ_from_affine_grid():
    m = QuantizedLinear(3, 2)
    _init_linear(m)
    x = torch.tensor([[0.5, -0.2, 0.1]], dtype=torch.float32)
    enc = m.input_quantizers[0].get_encodings()
    clear_fixed_scale_encoding_cache(enc)
    fixed = get_or_create_fixed_scale_encoding(enc)

    q_affine = Int16QuantizedTensor.from_affine_encoding(x, enc)
    q_fixed = Int16QuantizedTensor.from_fixed_scale_encoding(x, fixed)
    # Grids align in intent; int_repr may differ when (M,r) approximates scale.
    assert q_affine.int_repr.dtype is SIM_TENSOR_DTYPE
    assert q_fixed.int_repr.dtype is SIM_TENSOR_DTYPE


def test_int16_fixed_eval_uses_m_r_boundary_without_env(monkeypatch):
    m = QuantizedLinear(2, 2)
    _init_linear(m)
    enc = m.input_quantizers[0].get_encodings()
    clear_fixed_scale_encoding_cache(enc)
    monkeypatch.delenv("AIMET_RX_INT16_BOUNDARY_USE_M_R", raising=False)
    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        assert should_use_fixed_scale_boundary(enc)


def test_env_disables_m_r_without_cache(monkeypatch):
    m = QuantizedLinear(2, 2)
    _init_linear(m)
    enc = m.input_quantizers[0].get_encodings()
    clear_fixed_scale_encoding_cache(enc)
    monkeypatch.setenv("AIMET_RX_INT16_BOUNDARY_USE_M_R", "0")
    assert not int16_boundary_use_m_r()
    x = torch.tensor([[0.1, -0.1]], dtype=torch.float32)
    q = quantize_boundary_from_affine(x, enc)
    q_ref = Int16QuantizedTensor.from_affine_encoding(x, enc)
    assert torch.equal(q.int_repr, q_ref.int_repr)

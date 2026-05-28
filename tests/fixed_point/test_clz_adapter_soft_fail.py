"""Adapter returns None when CLZ LUT cannot be generated (no abc tree)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import aimet_torch.fixed_point.kernels  # noqa: F401
import aimet_torch.fixed_point.offline.clz_gen as clz_gen
from aimet_torch.fixed_point import ExecutionMode, quant_execution_mode
from aimet_torch.fixed_point.offline.clz_gen import ClzLutGenerationError
from aimet_torch.v2.nn.modules.custom import QuantizedSqrt
from aimet_torch.v2.quantization.affine import Quantize
from aimet_torch.v2.quantization.affine.fixed_point import adapter as fp_adapter


def test_dispatch_returns_none_when_clz_lut_missing(monkeypatch):
    m = QuantizedSqrt()
    m.input_quantizers[0] = Quantize((), 16, symmetric=True)
    m.output_quantizers[0] = Quantize((), 16, symmetric=True)
    m.input_quantizers[0].min = nn.Parameter(torch.tensor(0.1))
    m.input_quantizers[0].max = nn.Parameter(torch.tensor(4.0))
    m.output_quantizers[0].min = nn.Parameter(torch.tensor(0.0))
    m.output_quantizers[0].max = nn.Parameter(torch.tensor(2.0))
    m(torch.tensor([[1.0]]))

    monkeypatch.setattr(clz_gen, "resolve_abc_lut_root", lambda explicit=None: None)
    monkeypatch.delenv("AIMET_RX_REQUIRE_CLZ_LUT", raising=False)

    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        out = fp_adapter.dispatch_int16_fixed(m, torch.tensor([[1.0]]))
    assert out is None


def test_dispatch_raises_when_require_clz_lut(monkeypatch):
    m = QuantizedSqrt()
    m.input_quantizers[0] = Quantize((), 16, symmetric=True)
    m.output_quantizers[0] = Quantize((), 16, symmetric=True)
    m.input_quantizers[0].min = nn.Parameter(torch.tensor(0.1))
    m.input_quantizers[0].max = nn.Parameter(torch.tensor(4.0))
    m.output_quantizers[0].min = nn.Parameter(torch.tensor(0.0))
    m.output_quantizers[0].max = nn.Parameter(torch.tensor(2.0))
    m(torch.tensor([[1.0]]))

    monkeypatch.setattr(clz_gen, "resolve_abc_lut_root", lambda explicit=None: None)
    monkeypatch.setenv("AIMET_RX_REQUIRE_CLZ_LUT", "1")

    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        with pytest.raises(ClzLutGenerationError):
            fp_adapter.dispatch_int16_fixed(m, torch.tensor([[1.0]]))

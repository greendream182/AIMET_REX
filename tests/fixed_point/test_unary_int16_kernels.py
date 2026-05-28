# -*- mode: python -*-
"""Smoke tests for Abs/Sign reference INT16 kernels (PowerCompress decompose path)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from aimet_torch.fixed_point.encoding import OutputEncoding
from aimet_torch.fixed_point.kernels.eltwise import AbsInt16Kernel, SignInt16Kernel
from aimet_torch.fixed_point.requantize import SIM_TENSOR_DTYPE
from aimet_torch.fixed_point.tensor import Int16QuantizedTensor


def _carrier(values: torch.Tensor, scale: float = 0.1) -> Int16QuantizedTensor:
    q = torch.round(values / scale).to(SIM_TENSOR_DTYPE)
    return Int16QuantizedTensor(
        int_repr=q,
        scale=torch.tensor(scale),
        zero_point=torch.tensor(0, dtype=torch.int32),
    )


def _out_enc(scale: float = 0.1) -> OutputEncoding:
    return OutputEncoding(
        scale=torch.tensor(scale),
        zero_point=torch.tensor(0, dtype=torch.int32),
        qmin=-128,
        qmax=127,
    )


def test_abs_int16_kernel_reference():
    x = _carrier(torch.tensor([-0.4, 0.0, 0.3]))
    out = AbsInt16Kernel()([x], {}, _out_enc(), {})
    expected = torch.abs(torch.tensor([-0.4, 0.0, 0.3]))
    got = (out.int_repr.to(torch.float32) * out.scale).tolist()
    assert got == pytest.approx(expected.tolist(), abs=0.15)


def test_sign_int16_kernel_reference():
    x = _carrier(torch.tensor([-0.4, 0.0, 0.3]))
    out = SignInt16Kernel()([x], {}, _out_enc(), {})
    got = (out.int_repr.to(torch.float32) * out.scale).tolist()
    assert got[0] < 0
    assert abs(got[1]) < 0.2
    assert got[2] > 0

# -*- mode: python -*-
"""Smoke tests for Abs/Sign INT16 kernels (PowerCompress decompose path)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import aimet_torch.fixed_point.kernels  # noqa: F401
from aimet_torch.fixed_point.encoding import InputEncoding, OutputEncoding
from aimet_torch.fixed_point.kernels.eltwise import SignInt16Kernel
from aimet_torch.fixed_point.kernels.lut import AbsInt16Kernel
from aimet_torch.fixed_point.offline.lut_gen import generate_pwl_lut_for_export
from aimet_torch.fixed_point.requantize import SIM_TENSOR_DTYPE
from aimet_torch.fixed_point.tensor import Int16QuantizedTensor


def _carrier(values: torch.Tensor, scale: float = 0.1) -> Int16QuantizedTensor:
    q = torch.round(values / scale).to(SIM_TENSOR_DTYPE)
    return Int16QuantizedTensor(
        int_repr=q,
        scale=torch.tensor(scale),
        zero_point=torch.tensor(0, dtype=torch.int32),
    )


def _in_enc(scale: float = 0.1) -> InputEncoding:
    return InputEncoding(
        scale=torch.tensor(scale),
        zero_point=torch.tensor(0, dtype=torch.int32),
        qmin=-128,
        qmax=127,
    )


def _out_enc(scale: float = 0.1) -> OutputEncoding:
    return OutputEncoding(
        scale=torch.tensor(scale),
        zero_point=torch.tensor(0, dtype=torch.int32),
        qmin=-128,
        qmax=127,
    )


def test_abs_int16_kernel_pwl():
    in_enc = _in_enc()
    out_enc = _out_enc()
    pwl, _, _ = generate_pwl_lut_for_export(torch.abs, in_enc, out_enc, fn_name="abs")
    x = _carrier(torch.tensor([-0.4, 0.0, 0.3]))
    out = AbsInt16Kernel()([x], {}, out_enc, {"pwl_lut": pwl})
    expected = torch.abs(torch.tensor([-0.4, 0.0, 0.3]))
    got = (out.int_repr.to(torch.float32) * out.scale).tolist()
    assert got == pytest.approx(expected.tolist(), abs=0.15)


def test_sign_int16_kernel_exact():
    x = _carrier(torch.tensor([-0.4, 0.0, 0.3]))
    out = SignInt16Kernel()([x], {}, _out_enc(), {})
    got = (out.int_repr.to(torch.float32) * out.scale).tolist()
    assert got[0] < 0
    assert abs(got[1]) < 0.2
    assert got[2] > 0

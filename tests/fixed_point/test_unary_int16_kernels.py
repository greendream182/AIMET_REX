# -*- mode: python -*-
"""Smoke tests for Abs/Sign INT16 kernels (PowerCompress decompose path).

``AbsInt16Kernel`` migrated from ``kernels/lut.py`` (PWL) to
``kernels/eltwise.py`` (integer-abs path per spec 04_03 §4.3.5) in Layer B1.
The smoke test no longer drives the PWL path; full coverage of the new
kernel lives in
``tests/fixed_point/kernels/test_relu_int16_precision.py::test_abs_*``.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import aimet_torch.fixed_point.kernels  # noqa: F401
from aimet_torch.fixed_point.encoding import InputEncoding, OutputEncoding
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


def test_abs_int16_kernel_integer_abs():
    """Layer B1 spec 04_03 §4.3.5 integer-abs path: |q_x − Z_x|."""
    out_enc = _out_enc()
    x = _carrier(torch.tensor([-0.4, 0.0, 0.3]))
    out = AbsInt16Kernel()([x], {}, out_enc, {})
    expected = torch.abs(torch.tensor([-0.4, 0.0, 0.3]))
    got = (out.int_repr.to(torch.float32) * out.scale).tolist()
    # integer-abs same-grid is bit-exact; compare with a small floor for the
    # 1-LSB rounding of the carrier.
    assert got == pytest.approx(expected.tolist(), abs=0.15)


def test_sign_int16_kernel_exact():
    x = _carrier(torch.tensor([-0.4, 0.0, 0.3]))
    out = SignInt16Kernel()([x], {}, _out_enc(), {"sign_float_ref": True})
    got = (out.int_repr.to(torch.float32) * out.scale).tolist()
    assert got[0] < 0
    assert abs(got[1]) < 0.2
    assert got[2] > 0

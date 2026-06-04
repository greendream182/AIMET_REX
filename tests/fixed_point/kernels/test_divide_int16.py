# -*- mode: python -*-
"""INT16 Divide kernel (BN normalize path)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import aimet_torch.fixed_point.kernels  # noqa: F401
from aimet_torch._base.nn.modules import custom
from aimet_torch.fixed_point import (
    InputEncoding,
    Int16QuantizedTensor,
    OutputEncoding,
    get_fixed_kernel,
)
from aimet_torch.fixed_point.kernels.eltwise import DivideInt16Kernel
from aimet_torch.fixed_point.metrics.accuracy import quantize_float_to_grid
from aimet_torch.fixed_point.requantize import SIM_TENSOR_DTYPE


def _enc(
    scale: float,
    *,
    device: torch.device,
    qmin: int = -32768,
    qmax: int = 32767,
) -> InputEncoding:
    return InputEncoding(
        scale=torch.tensor(scale, device=device, dtype=torch.float32),
        zero_point=torch.tensor(0, device=device, dtype=torch.int32),
        qmin=qmin,
        qmax=qmax,
    )


def _out_enc(scale: float, *, device: torch.device) -> OutputEncoding:
    return OutputEncoding(
        scale=torch.tensor(scale, device=device, dtype=torch.float32),
        zero_point=torch.tensor(0, device=device, dtype=torch.int32),
        qmin=-32768,
        qmax=32767,
    )


def _carrier(
    values: torch.Tensor,
    enc: InputEncoding,
) -> Int16QuantizedTensor:
    q = torch.round(values / enc.scale).to(SIM_TENSOR_DTYPE)
    q = torch.clamp(q, enc.qmin, enc.qmax)
    return Int16QuantizedTensor(
        int_repr=q,
        scale=enc.scale,
        zero_point=enc.zero_point,
        qmin=enc.qmin,
        qmax=enc.qmax,
    )


def _float_divide_reference(
    num: Int16QuantizedTensor,
    den: Int16QuantizedTensor,
    out_enc: OutputEncoding,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    n_f = (num.int_repr.to(torch.float32) - num.zero_point) * num.scale
    d_f = (den.int_repr.to(torch.float32) - den.zero_point) * den.scale
    out_f = n_f / torch.where(
        d_f.abs() < eps,
        eps * d_f.sign().clamp(min=1.0),
        d_f,
    )
    return quantize_float_to_grid(
        out_f,
        out_enc.scale,
        out_enc.zero_point,
        out_enc.qmin,
        out_enc.qmax,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_divide_int16_matches_float_grid(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    dev = torch.device(device)
    s_num, s_den, s_out = 0.05, 0.02, 0.1
    in_n = _enc(s_num, device=dev)
    in_d = _enc(s_den, device=dev)
    out_enc = _out_enc(s_out, device=dev)

    num_v = torch.tensor([-0.2, 0.0, 0.15, 0.4], device=dev)
    den_v = torch.tensor([0.08, 0.01, 0.12, 0.25], device=dev)
    num = _carrier(num_v, in_n)
    den = _carrier(den_v, in_d)

    y = DivideInt16Kernel()([num, den], {}, out_enc, {"eps": 1e-12})
    y_ref = _float_divide_reference(num, den, out_enc)
    err = (y.int_repr.to(torch.int32) - y_ref.to(torch.int32)).abs().max().item()
    assert err <= 2


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_divide_int16_small_positive_denominator(device):
    """BN-like std path: denominator well above ``eps`` in float and int grids."""

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    dev = torch.device(device)
    in_n = _enc(0.1, device=dev, qmin=-128, qmax=127)
    in_d = _enc(0.1, device=dev, qmin=0, qmax=127)
    out_enc = _out_enc(0.1, device=dev)

    num = _carrier(torch.tensor([0.3], device=dev), in_n)
    den = _carrier(torch.tensor([0.1], device=dev), in_d)
    y = DivideInt16Kernel()([num, den], {}, out_enc, {})
    y_ref = _float_divide_reference(num, den, out_enc)
    err = (y.int_repr.to(torch.int32) - y_ref.to(torch.int32)).abs().max().item()
    assert err <= 2


def test_divide_registered_kernel():
    dev = torch.device("cpu")
    in_n = _enc(0.1, device=dev, qmin=-128, qmax=127)
    in_d = _enc(0.2, device=dev, qmin=0, qmax=127)
    out_enc = _out_enc(0.1, device=dev)
    num = _carrier(torch.tensor([0.2], device=dev), in_n)
    den = _carrier(torch.tensor([0.4], device=dev), in_d)
    y = get_fixed_kernel(custom.Divide)([num, den], {}, out_enc, {})
    assert y.int_repr.numel() == 1

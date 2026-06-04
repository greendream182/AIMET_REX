# -*- mode: python -*-
"""INT16 Abs (PWL) and Sign (integer exact) for R3 frontend."""

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
from aimet_torch.fixed_point.kernels.eltwise import SignInt16Kernel
from aimet_torch.fixed_point.kernels.lut import AbsInt16Kernel
from aimet_torch.fixed_point.metrics.accuracy import quantize_float_to_grid
from aimet_torch.fixed_point.metrics.thresholds import (
    PWL_HARDWARE_NUM_SEGMENTS,
    PWL_VS_ANALYTIC_MIN_COSINE_SIMILARITY,
    PWL_VS_ANALYTIC_PER_FN_LIMITS,
)
from aimet_torch.fixed_point.offline.lut_gen import generate_pwl_lut_for_export
from aimet_torch.fixed_point.requantize import SIM_TENSOR_DTYPE


def _input_enc(
    device: torch.device,
    *,
    scale: float = 0.1,
    qmin: int = -128,
    qmax: int = 127,
) -> InputEncoding:
    return InputEncoding(
        scale=torch.tensor(scale, device=device, dtype=torch.float32),
        zero_point=torch.tensor(0, device=device, dtype=torch.int32),
        qmin=qmin,
        qmax=qmax,
    )


def _output_enc(
    device: torch.device,
    *,
    scale: float = 0.1,
    qmin: int = -128,
    qmax: int = 127,
) -> OutputEncoding:
    return OutputEncoding(
        scale=torch.tensor(scale, device=device, dtype=torch.float32),
        zero_point=torch.tensor(0, device=device, dtype=torch.int32),
        qmin=qmin,
        qmax=qmax,
    )


def _carrier(
    values: torch.Tensor,
    *,
    device: torch.device,
    scale: float = 0.1,
) -> Int16QuantizedTensor:
    q = torch.round(values.to(device) / scale).to(SIM_TENSOR_DTYPE)
    return Int16QuantizedTensor(
        int_repr=q,
        scale=torch.tensor(scale, device=device),
        zero_point=torch.tensor(0, device=device, dtype=torch.int32),
        qmin=-128,
        qmax=127,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_abs_pwl_export_quality(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    dev = torch.device(device)
    in_enc = _input_enc(dev)
    out_enc = _output_enc(dev)

    pwl, num_segments, metrics = generate_pwl_lut_for_export(
        torch.abs,
        in_enc,
        out_enc,
        enforce_quality=True,
        fn_name="abs",
    )
    assert num_segments == PWL_HARDWARE_NUM_SEGMENTS
    limits = PWL_VS_ANALYTIC_PER_FN_LIMITS["abs"]
    assert metrics["max_lsb"] <= limits["max_lsb"]
    assert metrics["p99_lsb"] <= limits["p99_lsb"]
    assert metrics["rmse_lsb"] <= limits["rmse_lsb"]
    assert metrics["cosine_similarity"] >= PWL_VS_ANALYTIC_MIN_COSINE_SIMILARITY

    x = _carrier(torch.tensor([-0.4, 0.0, 0.3]), device=dev)
    y = AbsInt16Kernel()([x], {}, out_enc, {"pwl_lut": pwl})
    x_f = (x.int_repr.to(torch.float32) - x.zero_point) * x.scale
    y_ref = quantize_float_to_grid(
        torch.abs(x_f),
        out_enc.scale,
        out_enc.zero_point,
        out_enc.qmin,
        out_enc.qmax,
    )
    torch.testing.assert_close(y.int_repr.to(torch.int32), y_ref.to(torch.int32))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sign_int16_exact_matches_grid(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    dev = torch.device(device)
    out_enc = _output_enc(dev)
    x = _carrier(torch.tensor([-0.4, 0.0, 0.3]), device=dev)
    y = SignInt16Kernel()([x], {}, out_enc, {})
    x_f = (x.int_repr.to(torch.float32) - x.zero_point) * x.scale
    y_ref = quantize_float_to_grid(
        torch.sign(x_f),
        out_enc.scale,
        out_enc.zero_point,
        out_enc.qmin,
        out_enc.qmax,
    )
    torch.testing.assert_close(y.int_repr.to(torch.int32), y_ref.to(torch.int32))


def test_abs_registered_via_get_fixed_kernel():
    dev = torch.device("cpu")
    in_enc = _input_enc(dev)
    out_enc = _output_enc(dev)
    pwl_abs, _, _ = generate_pwl_lut_for_export(
        torch.abs, in_enc, out_enc, fn_name="abs"
    )
    x = Int16QuantizedTensor(
        int_repr=torch.tensor([-4, 0, 3], dtype=torch.int16),
        scale=in_enc.scale,
        zero_point=in_enc.zero_point,
        qmin=-128,
        qmax=127,
    )
    y = get_fixed_kernel(custom.Abs)([x], {}, out_enc, {"pwl_lut": pwl_abs})
    assert y.int_repr.tolist() == [4, 0, 3]

# -*- mode: python -*-
"""INT16 Sign (integer-exact) tests for R3 frontend.

Historical filename ``test_abs_sign_pwl_int16.py`` is retained for git
history; the Abs portion has moved to
``test_relu_int16_precision.py::test_abs_{same,cross}_scale_random_fp32_per_grid``
following the migration of ``custom.Abs`` from a 16-segment PWL LUT to
the spec ``04_03 §4.3.5`` integer-abs path (see
``kernels/eltwise.py::AbsInt16Kernel``).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import aimet_torch.fixed_point.kernels  # noqa: F401
from aimet_torch.fixed_point import (
    InputEncoding,
    Int16QuantizedTensor,
    OutputEncoding,
)
from aimet_torch.fixed_point.kernels.eltwise import SignInt16Kernel
from aimet_torch.fixed_point.metrics.accuracy import quantize_float_to_grid
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


def test_sign_int16_centered_matches_grid_in_eval():
    """Default ``INT16_FIXED_EVAL`` path: integer compare on centered values."""

    from aimet_torch.fixed_point import ExecutionMode, quant_execution_mode

    dev = torch.device("cpu")
    out_enc = _output_enc(dev)
    x = _carrier(torch.tensor([-0.4, 0.0, 0.3]), device=dev)
    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y = SignInt16Kernel()([x], {}, out_enc, {})
    x_f = (x.int_repr.to(torch.int32) - x.zero_point) * x.scale
    centered_sign = (x_f > 0).to(torch.int32) - (x_f < 0).to(torch.int32)
    y_ref = quantize_float_to_grid(
        centered_sign.to(torch.float32) * out_enc.scale.reshape(1),
        out_enc.scale,
        out_enc.zero_point,
        out_enc.qmin,
        out_enc.qmax,
    )
    torch.testing.assert_close(y.int_repr.to(torch.int32), y_ref.to(torch.int32))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sign_int16_float_ref_matches_grid(device):
    """Optional float reference (``sign_float_ref`` / ``AIMET_RX_SIGN_FLOAT_REF``)."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    dev = torch.device(device)
    out_enc = _output_enc(dev)
    x = _carrier(torch.tensor([-0.4, 0.0, 0.3]), device=dev)
    y = SignInt16Kernel()([x], {}, out_enc, {"sign_float_ref": True})
    x_f = (x.int_repr.to(torch.float32) - x.zero_point) * x.scale
    y_ref = quantize_float_to_grid(
        torch.sign(x_f),
        out_enc.scale,
        out_enc.zero_point,
        out_enc.qmin,
        out_enc.qmax,
    )
    torch.testing.assert_close(y.int_repr.to(torch.int32), y_ref.to(torch.int32))



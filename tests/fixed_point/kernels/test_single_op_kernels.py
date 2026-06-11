import torch
import torch.nn as nn

import aimet_torch.fixed_point.kernels  # noqa: F401
from aimet_torch._base.nn.modules import custom
from aimet_torch.fixed_point import Int16QuantizedTensor, OutputEncoding, get_fixed_kernel
from aimet_torch.fixed_point.metrics.accuracy import (
    cosine_similarity,
    max_error_lsb_float,
)
from aimet_torch.fixed_point.metrics.flags import int16_eval_allow_debug_float
from aimet_torch.fixed_point.quant_grid import GRID_I16
from aimet_torch.fixed_point.requantize import SIM_TENSOR_DTYPE, saturate_sim_tensor


def _int16_tensor(values, scale=1.0, zero_point=0):
    return Int16QuantizedTensor(
        int_repr=torch.tensor(values, dtype=torch.int16),
        scale=torch.tensor(scale, dtype=torch.float32),
        zero_point=torch.tensor(zero_point, dtype=torch.int32),
    )


def _output_encoding(multiplier=32767, rshift=15, scale=1.0, zero_point=0):
    return OutputEncoding(
        scale=torch.tensor(scale, dtype=torch.float32),
        zero_point=torch.tensor(zero_point, dtype=torch.int32),
        qmin=-32768,
        qmax=32767,
        multiplier=torch.tensor(multiplier, dtype=torch.uint16),
        rshift=torch.tensor(rshift, dtype=torch.int8),
    )


def test_conv2d_single_op_int16_kernel():
    x = _int16_tensor([[[[1, 2], [3, 4]]]])
    weight = _int16_tensor([[[[1, 0], [0, 1]]]])

    output = get_fixed_kernel(nn.Conv2d)(
        [x],
        {"weight": weight},
        _output_encoding(),
        {"stride": 1, "padding": 0, "dilation": 1, "groups": 1},
    )

    assert output.int_repr.dtype is SIM_TENSOR_DTYPE
    assert output.int_repr.tolist() == [[[[5]]]]


def test_linear_single_op_int16_kernel():
    x = _int16_tensor([[1, 2, 3]])
    weight = _int16_tensor([[1, 1, 1], [1, 0, -1]])
    bias = torch.tensor([1, -1], dtype=torch.int32)

    output = get_fixed_kernel(nn.Linear)(
        [x],
        {"weight": weight, "bias": bias},
        _output_encoding(),
        {},
    )

    assert output.int_repr.dtype is SIM_TENSOR_DTYPE
    assert output.int_repr.tolist() == [[7, -3]]


def test_matmul_single_op_int16_kernel():
    lhs = _int16_tensor([[1, 2, 3], [4, 5, 6]])
    rhs = _int16_tensor([[1, 0], [0, 1], [1, -1]])

    output = get_fixed_kernel(custom.MatMul)(
        [lhs, rhs],
        {},
        _output_encoding(),
        {},
    )

    assert output.int_repr.dtype is SIM_TENSOR_DTYPE
    assert output.int_repr.tolist() == [[4, -1], [10, -1]]


def test_add_single_op_int16_kernel():
    out_enc = _output_encoding(scale=0.25)
    a = torch.tensor([2.5, -1.0, 2.0], dtype=torch.float32)
    b = torch.tensor([0.75, 1.25, -2.5], dtype=torch.float32)
    scale = 0.25
    zp = torch.tensor(0, dtype=torch.int32)
    scale_t = torch.tensor(scale, dtype=torch.float32)
    lhs = Int16QuantizedTensor(
        int_repr=saturate_sim_tensor(torch.round(a / scale), GRID_I16.qmin, GRID_I16.qmax),
        scale=scale_t,
        zero_point=zp,
        qmin=GRID_I16.qmin,
        qmax=GRID_I16.qmax,
    )
    rhs = Int16QuantizedTensor(
        int_repr=saturate_sim_tensor(torch.round(b / scale), GRID_I16.qmin, GRID_I16.qmax),
        scale=scale_t,
        zero_point=zp,
        qmin=GRID_I16.qmin,
        qmax=GRID_I16.qmax,
    )

    output = get_fixed_kernel(custom.Add)([lhs, rhs], {}, out_enc, {})

    assert output.int_repr.dtype is SIM_TENSOR_DTYPE
    assert output.int_repr.tolist() == [13, 1, -2]
    ref = a + b
    with int16_eval_allow_debug_float():
        got = output.to_float()
    assert cosine_similarity(ref, got) > 0.9999
    assert (
        max_error_lsb_float(
            ref, got, output.scale, output.zero_point, output.qmin, output.qmax
        )
        < 1.0
    )

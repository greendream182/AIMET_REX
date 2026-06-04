import torch
import torch.nn as nn

import aimet_torch.fixed_point.kernels  # noqa: F401
from aimet_torch.fixed_point import (
    Int16QuantizedTensor,
    OutputEncoding,
    get_fixed_kernel,
)
from aimet_torch.fixed_point.requantize import SIM_TENSOR_DTYPE


def _tensor(values):
    return torch.tensor(values, dtype=torch.int16)


def _int16_tensor(values, scale=1.0, zero_point=0):
    return Int16QuantizedTensor(
        int_repr=_tensor(values),
        scale=torch.tensor(scale, dtype=torch.float32),
        zero_point=torch.tensor(zero_point, dtype=torch.int32),
    )


def _output_encoding(multiplier=32767, rshift=15, scale=1.0, zero_point=0):
    return OutputEncoding(
        scale=torch.tensor(scale, dtype=torch.float32),
        zero_point=torch.tensor(zero_point, dtype=torch.int32),
        qmin=-32768,
        qmax=32767,
        multiplier=torch.tensor(multiplier, dtype=torch.int16),
        rshift=torch.tensor(rshift, dtype=torch.int8),
    )


def test_linear_int16_kernel_reference_case():
    x = _int16_tensor([[1, 2, 3]])
    weight = _int16_tensor([[1, 1, 1], [1, 0, -1]])
    bias = torch.tensor([1, -1], dtype=torch.int32)
    output_encoding = _output_encoding()

    output = get_fixed_kernel(nn.Linear)(
        [x],
        {"weight": weight, "bias": bias},
        output_encoding,
        {},
    )

    assert output.int_repr.dtype is SIM_TENSOR_DTYPE
    assert output.int_repr.tolist() == [[7, -3]]


def test_conv2d_int16_kernel_reference_case():
    x = _int16_tensor([[[[1, 2], [3, 4]]]])
    weight = _int16_tensor([[[[1, 0], [0, 1]]]])
    output_encoding = _output_encoding()

    output = get_fixed_kernel(nn.Conv2d)(
        [x],
        {"weight": weight},
        output_encoding,
        {"stride": 1, "padding": 0, "dilation": 1, "groups": 1},
    )

    assert output.int_repr.dtype is SIM_TENSOR_DTYPE
    assert output.int_repr.tolist() == [[[[5]]]]


def test_linear_int32_accumulator_saturates_by_default(monkeypatch):
    """int32 MAC saturation is now the default (HW-faithful); opt-out (=0) wraps."""
    x = _int16_tensor([[32_767, 32_767, 32_767]])
    weight = _int16_tensor([[32_767, 32_767, 32_767]])
    output_encoding = _output_encoding(multiplier=1, rshift=0, scale=1.0)

    monkeypatch.delenv("AIMET_RX_HW_REF", raising=False)
    monkeypatch.delenv("AIMET_RX_ACC_INT32_SAT", raising=False)
    out_default = get_fixed_kernel(nn.Linear)(
        [x], {"weight": weight}, output_encoding, {}
    )

    monkeypatch.setenv("AIMET_RX_ACC_INT32_SAT", "0")
    out_optout = get_fixed_kernel(nn.Linear)(
        [x], {"weight": weight}, output_encoding, {}
    )

    # Default now clamps to int32 max; opting out reverts to silent wrap.
    assert out_default.int_repr.item() == 32_767
    assert out_optout.int_repr.item() != out_default.int_repr.item()


def test_conv2d_int16_kernel_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    x = _int16_tensor([[[[1, 2], [3, 4]]]]).to("cuda")
    weight = _int16_tensor([[[[1, 0], [0, 1]]]]).to("cuda")
    output_encoding = _output_encoding()
    for key in (output_encoding.scale, output_encoding.zero_point,
                output_encoding.multiplier, output_encoding.rshift):
        if isinstance(key, torch.Tensor):
            key.data = key.data.to("cuda")

    output = get_fixed_kernel(nn.Conv2d)(
        [x],
        {"weight": weight},
        output_encoding,
        {"stride": 1, "padding": 0, "dilation": 1, "groups": 1},
    )

    assert output.int_repr.is_cuda
    assert output.int_repr.dtype is SIM_TENSOR_DTYPE
    assert output.int_repr.tolist() == [[[[5]]]]


def test_linear_int16_kernel_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    x = _int16_tensor([[1, 2, 3]]).to("cuda")
    weight = _int16_tensor([[1, 1, 1], [1, 0, -1]]).to("cuda")
    bias = torch.tensor([1, -1], dtype=torch.int32, device="cuda")
    output_encoding = _output_encoding()
    for key in (output_encoding.scale, output_encoding.zero_point,
                output_encoding.multiplier, output_encoding.rshift):
        if isinstance(key, torch.Tensor):
            key.data = key.data.to("cuda")

    output = get_fixed_kernel(nn.Linear)(
        [x],
        {"weight": weight, "bias": bias},
        output_encoding,
        {},
    )

    assert output.int_repr.is_cuda
    assert output.int_repr.tolist() == [[7, -3]]
    # Input 1x1x2x2x2, kernel 1x1x1x2x2, stride 1, no padding
    x = _int16_tensor([[[[[1, 2], [3, 4]]]]])
    weight = _int16_tensor([[[[[1, 0], [0, 1]]]]])
    output_encoding = _output_encoding()

    output = get_fixed_kernel(nn.Conv3d)(
        [x],
        {"weight": weight},
        output_encoding,
        {"stride": 1, "padding": 0, "dilation": 1, "groups": 1},
    )

    assert output.int_repr.dtype is SIM_TENSOR_DTYPE
    assert output.int_repr.tolist() == [[[[[5]]]]]

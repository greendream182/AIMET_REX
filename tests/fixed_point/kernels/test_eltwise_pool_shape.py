import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
import torch.nn as nn
from aimet_torch.fixed_point import (
    Int16QuantizedTensor,
    OutputEncoding,
    get_fixed_kernel,
)
from aimet_torch._base.nn.modules import custom


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
        multiplier=torch.tensor(multiplier, dtype=torch.int16),
        rshift=torch.tensor(rshift, dtype=torch.int8),
    )


def test_relu_int16_kernel_clamps_to_zero_point():
    x = _int16_tensor([-2, 0, 3], zero_point=0)
    output = get_fixed_kernel(nn.ReLU)([x], {}, _output_encoding(), {})

    assert output.int_repr.tolist() == [0, 0, 3]


def test_maxpool2d_int16_kernel():
    x = _int16_tensor([[[[1, 2], [3, 4]]]])
    output = get_fixed_kernel(nn.MaxPool2d)(
        [x],
        {},
        _output_encoding(),
        {"kernel_size": 2, "stride": 2, "padding": 0},
    )

    assert output.int_repr.tolist() == [[[[4]]]]


def test_flatten_int16_kernel():
    x = _int16_tensor([[[1, 2], [3, 4]]])
    output = get_fixed_kernel(nn.Flatten)(
        [x],
        {},
        _output_encoding(),
        {"start_dim": 1, "end_dim": -1},
    )

    assert output.int_repr.tolist() == [[1, 2, 3, 4]]


def test_flatten_int16_kernel_rejects_changed_encoding():
    x = _int16_tensor([[[1, 2], [3, 4]]], scale=1.0, zero_point=0)
    with pytest.raises(ValueError, match="output scale"):
        get_fixed_kernel(nn.Flatten)(
            [x],
            {},
            _output_encoding(scale=0.5, zero_point=0),
            {"start_dim": 1, "end_dim": -1},
        )


def test_add_int16_kernel_aligns_input_scales_to_output():
    x = _int16_tensor([10], scale=0.5, zero_point=0)
    y = _int16_tensor([10], scale=0.25, zero_point=0)
    output = get_fixed_kernel(custom.Add)(
        [x, y],
        {},
        _output_encoding(multiplier=32767, rshift=15, scale=0.25, zero_point=0),
        {},
    )

    assert output.int_repr.tolist() == [30]

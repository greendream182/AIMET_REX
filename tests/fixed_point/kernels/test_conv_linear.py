import pytest
import torch
import torch.nn as nn

import aimet_torch.fixed_point.kernels  # noqa: F401
from aimet_torch.fixed_point import (
    ExecutionMode,
    Int16QuantizedTensor,
    OutputEncoding,
    get_fixed_kernel,
    quant_execution_mode,
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
        multiplier=torch.tensor(multiplier, dtype=torch.uint16),
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


def test_conv3d_int16_kernel_reference_case():
    x = _int16_tensor([[[[[1, 2], [3, 4]]]]])
    weight = _int16_tensor([[[[[1, 0], [0, 1]]]]])
    output_encoding = _output_encoding()

    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        output = get_fixed_kernel(nn.Conv3d)(
            [x],
            {"weight": weight},
            output_encoding,
            {"stride": 1, "padding": 0, "dilation": 1, "groups": 1},
        )

    assert output.int_repr.dtype is SIM_TENSOR_DTYPE
    assert output.int_repr.tolist() == [[[[[5]]]]]


def test_linear_bias_bits_16_accepts_int16_bias_and_matches_int32_path():
    """``bias_bits=16`` accepts an int16 bias tensor; numerics match the int32 path
    when bias values fit both ranges (spec 04_01 explicit_config)."""

    x = _int16_tensor([[1, 2, 3]])
    weight = _int16_tensor([[1, 1, 1], [1, 0, -1]])
    bias_int32 = torch.tensor([1, -1], dtype=torch.int32)
    bias_int16 = torch.tensor([1, -1], dtype=torch.int16)
    enc_default = _output_encoding()
    enc_bias16 = OutputEncoding(
        scale=enc_default.scale,
        zero_point=enc_default.zero_point,
        qmin=enc_default.qmin,
        qmax=enc_default.qmax,
        multiplier=enc_default.multiplier,
        rshift=enc_default.rshift,
        bias_bits=16,
    )
    assert enc_default.bias_bits == 32

    out_b32 = get_fixed_kernel(nn.Linear)(
        [x], {"weight": weight, "bias": bias_int32}, enc_default, {}
    )
    out_b16 = get_fixed_kernel(nn.Linear)(
        [x], {"weight": weight, "bias": bias_int16}, enc_bias16, {}
    )

    assert out_b32.int_repr.tolist() == out_b16.int_repr.tolist() == [[7, -3]]


def test_linear_bias_bits_mismatch_dtype_raises():
    """``bias_bits=32`` rejects an int16 bias tensor; ``bias_bits=16`` rejects int32."""

    x = _int16_tensor([[1, 2, 3]])
    weight = _int16_tensor([[1, 1, 1], [1, 0, -1]])
    enc_b32 = _output_encoding()
    enc_b16 = OutputEncoding(
        scale=enc_b32.scale,
        zero_point=enc_b32.zero_point,
        qmin=enc_b32.qmin,
        qmax=enc_b32.qmax,
        multiplier=enc_b32.multiplier,
        rshift=enc_b32.rshift,
        bias_bits=16,
    )
    bias_int16 = torch.tensor([1, -1], dtype=torch.int16)
    bias_int32 = torch.tensor([1, -1], dtype=torch.int32)

    with pytest.raises(TypeError, match=r"bias_bits=32 requires dtype torch\.int32"):
        get_fixed_kernel(nn.Linear)(
            [x], {"weight": weight, "bias": bias_int16}, enc_b32, {}
        )
    with pytest.raises(TypeError, match=r"bias_bits=16 requires dtype torch\.int16"):
        get_fixed_kernel(nn.Linear)(
            [x], {"weight": weight, "bias": bias_int32}, enc_b16, {}
        )


def test_linear_bias_bits_invalid_value_raises():
    """Only ``bias_bits in {16, 32}`` is allowed."""

    x = _int16_tensor([[1, 2, 3]])
    weight = _int16_tensor([[1, 1, 1], [1, 0, -1]])
    enc_default = _output_encoding()
    enc_bad = OutputEncoding(
        scale=enc_default.scale,
        zero_point=enc_default.zero_point,
        qmin=enc_default.qmin,
        qmax=enc_default.qmax,
        multiplier=enc_default.multiplier,
        rshift=enc_default.rshift,
        bias_bits=8,
    )
    with pytest.raises(ValueError, match=r"bias_bits must be 16 or 32"):
        get_fixed_kernel(nn.Linear)(
            [x], {"weight": weight}, enc_bad, {}
        )


@pytest.mark.parametrize(
    "module_type, weight_shape, extra",
    [
        (nn.Linear, [[1, 1, 1], [1, 0, -1]], {}),
        (
            nn.Conv2d,
            [[[[1, 0], [0, 1]]]],
            {"stride": 1, "padding": 0, "dilation": 1, "groups": 1},
        ),
        (
            nn.Conv1d,
            [[[1, 0]]],
            {"stride": 1, "padding": 0, "dilation": 1, "groups": 1},
        ),
        (
            nn.Conv3d,
            [[[[[1, 0], [0, 1]]]]],
            {"stride": 1, "padding": 0, "dilation": 1, "groups": 1},
        ),
    ],
)
def test_conv_linear_reject_nonzero_weight_zero_point(module_type, weight_shape, extra):
    """Spec 04_01 / 04_02 require ``Z_w = 0``; non-zero must raise ValueError.

    The HW MAC path expands ``sum_i q_w_i * (q_x_i - Z_x)`` with no ``Z_w``
    term, so a non-zero weight zero_point would silently desync the simulator
    from hardware. Lock that in with an explicit failure here.
    """

    weight = _int16_tensor(weight_shape, zero_point=1)

    if module_type is nn.Linear:
        x = _int16_tensor([[1, 2, 3]])
    elif module_type is nn.Conv1d:
        x = _int16_tensor([[[1, 2]]])
    elif module_type is nn.Conv2d:
        x = _int16_tensor([[[[1, 2], [3, 4]]]])
    else:
        x = _int16_tensor([[[[[1, 2], [3, 4]]]]])
    output_encoding = _output_encoding()

    with pytest.raises(ValueError, match=r"weight zero_point must be 0"):
        get_fixed_kernel(module_type)(
            [x], {"weight": weight}, output_encoding, extra
        )

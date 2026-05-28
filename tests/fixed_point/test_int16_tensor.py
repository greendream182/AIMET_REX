import pytest
import torch

from aimet_torch.fixed_point import FixedPointSimTensor, Int16QuantizedTensor
from aimet_torch.fixed_point.requantize import SIM_TENSOR_DTYPE


def test_int16_tensor_from_float_per_tensor():
    tensor = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)
    scale = torch.tensor(0.01, dtype=torch.float32)
    zero_point = torch.tensor(0, dtype=torch.int32)

    quantized = FixedPointSimTensor.from_float(tensor, scale, zero_point)

    assert quantized.int_repr.dtype is SIM_TENSOR_DTYPE
    assert quantized.int_repr.tolist() == [0, 50, 100]


def test_int16_tensor_to_float_debug():
    quantized = FixedPointSimTensor(
        int_repr=torch.tensor([0, 50, 100], dtype=SIM_TENSOR_DTYPE),
        scale=torch.tensor(0.01, dtype=torch.float32),
        zero_point=torch.tensor(0, dtype=torch.int32),
    )

    tensor = quantized.to_float()

    assert torch.allclose(tensor, torch.tensor([0.0, 0.5, 1.0]))


def test_legacy_int16_repr_is_normalized_to_int32():
    quantized = Int16QuantizedTensor(
        int_repr=torch.tensor([0, 50, 100], dtype=torch.int16),
        scale=torch.tensor(0.01, dtype=torch.float32),
        zero_point=torch.tensor(0, dtype=torch.int32),
    )

    assert quantized.int_repr.dtype is SIM_TENSOR_DTYPE
    assert quantized.int_repr.tolist() == [0, 50, 100]


def test_sim_tensor_rejects_non_sim_dtype_repr():
    with pytest.raises(TypeError):
        FixedPointSimTensor(
            int_repr=torch.tensor([0.0, 1.0], dtype=torch.float32),
            scale=torch.tensor(0.01),
            zero_point=torch.tensor(0, dtype=torch.int32),
        )


def test_sim_tensor_layout_introspection_and_permute():
    """FixedPointSimTensor must expose layout attrs for FX-traced INT16 graphs."""
    carrier = FixedPointSimTensor(
        int_repr=torch.arange(24, dtype=SIM_TENSOR_DTYPE).reshape(2, 3, 4),
        scale=torch.tensor(1.0, dtype=torch.float32),
        zero_point=torch.tensor(0, dtype=torch.int32),
    )
    assert carrier.shape == (2, 3, 4)
    assert carrier.ndim == 3
    assert carrier.size() == (2, 3, 4)
    assert carrier.size(1) == 3

    permuted = carrier.permute(2, 0, 1)
    assert permuted.shape == (4, 2, 3)
    assert permuted.int_repr.shape == (4, 2, 3)

    viewed = carrier.view(-1, 4)
    assert viewed.shape == (6, 4)


def test_kernels_emit_sim_tensor_dtype():
    """PR-3 dtype guard: every registered fixed kernel must emit SIM_TENSOR_DTYPE.

    PR-2 added an automatic int16 -> int32 promotion in ``__post_init__`` so the
    PR-2 patch could ship without breaking kernels. PR-3 then migrated every
    kernel to emit ``SIM_TENSOR_DTYPE`` directly. This test forbids regression
    by sampling a small kernel from each module and confirming the carrier
    dtype is already int32 *before* any normalization could trigger.
    """

    import torch.nn as nn

    import aimet_torch.fixed_point.kernels  # noqa: F401
    from aimet_torch._base.nn.modules import custom
    from aimet_torch.fixed_point import OutputEncoding, get_fixed_kernel

    def _carrier(values):
        return FixedPointSimTensor(
            int_repr=torch.tensor(values, dtype=SIM_TENSOR_DTYPE),
            scale=torch.tensor(1.0, dtype=torch.float32),
            zero_point=torch.tensor(0, dtype=torch.int32),
        )

    def _out_enc(scale=1.0):
        return OutputEncoding(
            scale=torch.tensor(scale, dtype=torch.float32),
            zero_point=torch.tensor(0, dtype=torch.int32),
            qmin=-32768,
            qmax=32767,
            multiplier=torch.tensor(32767, dtype=torch.int16),
            rshift=torch.tensor(15, dtype=torch.int8),
        )

    # ReLU (eltwise non-binary path)
    relu_out = get_fixed_kernel(nn.ReLU)(
        [_carrier([-2, -1, 0, 1, 2])], {}, _out_enc(), {}
    )
    assert relu_out.int_repr.dtype is SIM_TENSOR_DTYPE

    # Add (eltwise binary aligned path)
    add_out = get_fixed_kernel(custom.Add)(
        [_carrier([1, 2, 3]), _carrier([1, 2, 3])], {}, _out_enc(), {}
    )
    assert add_out.int_repr.dtype is SIM_TENSOR_DTYPE

    # MaxPool2d (pool path)
    pool_carrier = FixedPointSimTensor(
        int_repr=torch.arange(16, dtype=SIM_TENSOR_DTYPE).reshape(1, 1, 4, 4),
        scale=torch.tensor(1.0, dtype=torch.float32),
        zero_point=torch.tensor(0, dtype=torch.int32),
    )
    pool_out = get_fixed_kernel(nn.MaxPool2d)(
        [pool_carrier],
        {},
        _out_enc(),
        {"kernel_size": 2, "stride": 2},
    )
    assert pool_out.int_repr.dtype is SIM_TENSOR_DTYPE

    # Identity (shape_ops path)
    id_out = get_fixed_kernel(nn.Identity)([_carrier([1, 2, 3])], {}, _out_enc(), {})
    assert id_out.int_repr.dtype is SIM_TENSOR_DTYPE

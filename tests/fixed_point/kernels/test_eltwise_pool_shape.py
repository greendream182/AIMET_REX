import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
import torch.nn as nn

import aimet_torch.fixed_point.kernels  # noqa: F401 — register eltwise/pool/shape_ops

from aimet_torch.fixed_point import (
    Int16QuantizedTensor,
    OutputEncoding,
    get_fixed_kernel,
)
from aimet_torch.fixed_point.metrics import assert_int16_vs_fp32_reference
from aimet_torch.fixed_point.metrics.thresholds import (
    ADD_INT16_VS_FLOAT_MAX_ERROR_LSB,
    ADD_INT16_VS_FLOAT_MIN_COSINE_SIMILARITY,
    ADD_INT16_VS_FLOAT_STRESS_MIN_COSINE_SIMILARITY,
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
        multiplier=torch.tensor(multiplier, dtype=torch.uint16),
        rshift=torch.tensor(rshift, dtype=torch.int8),
    )


def test_relu_int16_kernel_clamps_to_zero_point():
    x = _int16_tensor([-2, 0, 3], zero_point=0)
    output = get_fixed_kernel(nn.ReLU)([x], {}, _output_encoding(), {})

    assert output.int_repr.tolist() == [0, 0, 3]


def _add_vs_float_dequant_sum(main, residual, out_enc):
    ref = main.to_float() + residual.to_float()
    output = get_fixed_kernel(custom.Add)([main, residual], {}, out_enc, {})
    return output, ref


def test_add_int16_vs_float_dequant_sum_standard_gate():
    """Path A：``ref = dequant(A)+dequant(B)``（kernel 等价）。单算子验收见 ``test_add_ideal_float_reference``。"""

    cases = [
        ("same_scale", [10, -4, 8], [3, 5, -10], 0.25, 0.25, 0.25, 0),
        ("scale_align", [10], [10], 0.5, 0.25, 0.25, 0),
        ("symmetric_random", [-50, 0, 100, 200], [30, -80, 10, -5], 0.1, 0.2, 0.15, 0),
    ]
    for _name, main_v, res_v, s_main, s_res, s_out, zp in cases:
        out_enc = _output_encoding(scale=s_out, zero_point=zp)
        main = _int16_tensor(main_v, scale=s_main, zero_point=0)
        residual = _int16_tensor(res_v, scale=s_res, zero_point=0)
        output, ref = _add_vs_float_dequant_sum(main, residual, out_enc)
        assert_int16_vs_fp32_reference(
            output,
            ref,
            max_lsb=ADD_INT16_VS_FLOAT_MAX_ERROR_LSB,
            min_cosine=ADD_INT16_VS_FLOAT_MIN_COSINE_SIMILARITY,
            label=f"Add vs float dequant sum ({_name})",
        )


def test_add_align_preserves_negative_centered_values_for_asymmetric_output():
    """Multi-scale uint8 residual: LSB ≤ 1; cosine regression floor (may be < 0.9999)."""

    out_scale = 0.00022886419901624322
    out_zp = 134
    out_enc = OutputEncoding(
        scale=torch.tensor(out_scale, dtype=torch.float32),
        zero_point=torch.tensor(out_zp, dtype=torch.int32),
        qmin=0,
        qmax=255,
        multiplier=torch.tensor(32767, dtype=torch.uint16),
        rshift=torch.tensor(15, dtype=torch.int8),
    )
    main = _int16_tensor(
        [[120, 80, 60]],
        scale=0.00024029603810049593,
        zero_point=0,
    )
    residual = _int16_tensor(
        [[-20, -40, 10]],
        scale=1.0082941116706934e-05,
        zero_point=0,
    )
    output, ref = _add_vs_float_dequant_sum(main, residual, out_enc)
    assert_int16_vs_fp32_reference(
        output,
        ref,
        max_lsb=ADD_INT16_VS_FLOAT_MAX_ERROR_LSB,
        min_cosine=ADD_INT16_VS_FLOAT_STRESS_MIN_COSINE_SIMILARITY,
        label="Add vs float dequant sum (mobilenet residual stress)",
    )


def test_hardtanh_int16_clamps_on_quant_grid():
    x = _int16_tensor([-10, 0, 50], scale=1.0, zero_point=0)
    out_enc = _output_encoding(scale=1.0, zero_point=0)
    output = get_fixed_kernel(nn.Hardtanh)(
        [x],
        {},
        out_enc,
        {"min": -1.0, "max": 2.0, "min_int": -1, "max_int": 2},
    )

    assert output.int_repr.tolist() == [-1, 0, 2]


def test_maxpool2d_int16_kernel():
    x = _int16_tensor([[[[1, 2], [3, 4]]]])
    output = get_fixed_kernel(nn.MaxPool2d)(
        [x],
        {},
        _output_encoding(),
        {"kernel_size": 2, "stride": 2, "padding": 0},
    )

    assert output.int_repr.tolist() == [[[[4]]]]


def test_maxpool2d_int16_kernel_rejects_scale_mismatch():
    """Spec doc/04_算子详细规格/04_09 §Max-pooling: comparator-only HW; refuse
    to silently rewrap int payload under a different output scale."""
    x = _int16_tensor([[[[1, 2], [3, 4]]]], scale=1.0, zero_point=0)
    with pytest.raises(ValueError, match="MaxPool2d input/output scale"):
        get_fixed_kernel(nn.MaxPool2d)(
            [x],
            {},
            _output_encoding(scale=0.5, zero_point=0),
            {"kernel_size": 2, "stride": 2, "padding": 0},
        )


def test_maxpool2d_int16_kernel_rejects_zero_point_mismatch():
    x = _int16_tensor([[[[1, 2], [3, 4]]]], scale=1.0, zero_point=0)
    with pytest.raises(ValueError, match="MaxPool2d input/output zero_point"):
        get_fixed_kernel(nn.MaxPool2d)(
            [x],
            {},
            _output_encoding(scale=1.0, zero_point=5),
            {"kernel_size": 2, "stride": 2, "padding": 0},
        )


def test_maxpool2d_int16_kernel_rejects_qmin_qmax_mismatch():
    x = _int16_tensor([[[[1, 2], [3, 4]]]], scale=1.0, zero_point=0)
    out_enc = OutputEncoding(
        scale=torch.tensor(1.0, dtype=torch.float32),
        zero_point=torch.tensor(0, dtype=torch.int32),
        qmin=0,
        qmax=255,
        multiplier=torch.tensor(32767, dtype=torch.uint16),
        rshift=torch.tensor(15, dtype=torch.int8),
    )
    with pytest.raises(ValueError, match="MaxPool2d input/output qmin/qmax"):
        get_fixed_kernel(nn.MaxPool2d)(
            [x],
            {},
            out_enc,
            {"kernel_size": 2, "stride": 2, "padding": 0},
        )


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


# -----------------------------------------------------------------------------
# Kernel-contract guards (see aimet_torch/fixed_point/kernels/_contracts.py).
# These pin the shared validators wired up under the ``add-kernel-contracts``
# task so that future drift between kernel categories is caught at unit-test
# time, not at silent-numerical-corruption time.
# -----------------------------------------------------------------------------


def _output_encoding_no_requantize(scale=1.0, zero_point=0):
    return OutputEncoding(
        scale=torch.tensor(scale, dtype=torch.float32),
        zero_point=torch.tensor(zero_point, dtype=torch.int32),
        qmin=-32768,
        qmax=32767,
        multiplier=None,
        rshift=None,
    )


def test_avgpool2d_int16_kernel_rejects_missing_multiplier():
    """AvgPool2d is requantizing per spec — no multiplier means broken contract."""

    x = _int16_tensor([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]])
    with pytest.raises(ValueError, match="AvgPool2d output encoding must provide multiplier"):
        get_fixed_kernel(nn.AvgPool2d)(
            [x],
            {},
            _output_encoding_no_requantize(),
            {"kernel_size": 2, "stride": 2, "padding": 0, "reduce_size": 4},
        )


def test_avgpool2d_int16_kernel_rejects_missing_reduce_size():
    """Spec 04_09 ``1/N`` fold contract: extra['reduce_size'] is mandatory."""

    x = _int16_tensor([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]])
    with pytest.raises(ValueError, match="AvgPool2d extra\\['reduce_size'\\] is required"):
        get_fixed_kernel(nn.AvgPool2d)(
            [x],
            {},
            _output_encoding(),
            {"kernel_size": 2, "stride": 2, "padding": 0},
        )


def test_avgpool2d_int16_kernel_rejects_mismatched_reduce_size():
    """``reduce_size`` from adapter must equal kernel-derived ``k_t*k_f``;
    a stale value would mean the offline ``M/rshift`` carries a different
    1/N than what the kernel runtime would compute, silently mis-scaling."""

    x = _int16_tensor([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]])
    with pytest.raises(ValueError, match="AvgPool2d extra\\['reduce_size'\\]=9 disagrees"):
        get_fixed_kernel(nn.AvgPool2d)(
            [x],
            {},
            _output_encoding(),
            {"kernel_size": 2, "stride": 2, "padding": 0, "reduce_size": 9},
        )


def test_avgpool2d_int16_kernel_happy_path_with_reduce_size():
    """Happy path: ``reduce_size`` matches ``k_t*k_f`` and forward succeeds."""

    x = _int16_tensor([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]])
    out = get_fixed_kernel(nn.AvgPool2d)(
        [x],
        {},
        _output_encoding(multiplier=32767, rshift=15, scale=1.0, zero_point=0),
        {"kernel_size": 2, "stride": 2, "padding": 0, "reduce_size": 4},
    )
    assert out.int_repr.shape == (1, 1, 2, 2)


def test_mean_int16_kernel_rejects_missing_reduce_size():
    x = _int16_tensor([[[[1, 2], [3, 4]]]])
    with pytest.raises(ValueError, match="extra\\['reduce_size'\\] is required"):
        get_fixed_kernel(custom.Mean)(
            [x],
            {},
            _output_encoding(),
            {"dim": (2, 3), "keepdim": True},
        )


def test_mean_int16_kernel_rejects_mismatched_reduce_size():
    x = _int16_tensor([[[[1, 2], [3, 4]]]])
    with pytest.raises(ValueError, match="extra\\['reduce_size'\\]=7 disagrees"):
        get_fixed_kernel(custom.Mean)(
            [x],
            {},
            _output_encoding(),
            {"dim": (2, 3), "keepdim": True, "reduce_size": 7},
        )


def test_maxpool2d_int16_kernel_rejects_oversize_kernel_in_hw_ref_mode(monkeypatch):
    """Spec doc/04_算子详细规格/04_09: MaxPool kt,kf <= 3."""

    monkeypatch.setenv("AIMET_RX_HW_REF", "1")
    x = _int16_tensor(torch.zeros(1, 1, 8, 8, dtype=torch.int16).tolist())
    with pytest.raises(ValueError, match="MaxPool2d kernel kt,kf must be <= 3"):
        get_fixed_kernel(nn.MaxPool2d)(
            [x],
            {},
            _output_encoding(),
            {"kernel_size": 4, "stride": 4, "padding": 0},
        )


def test_avgpool2d_int16_kernel_rejects_unallowed_kernel_in_hw_ref_mode(monkeypatch):
    """Spec doc/04_算子详细规格/04_09: AvgPool kernel ∈ {(2,2),(4,4),(4,2),(2,4)}."""

    monkeypatch.setenv("AIMET_RX_HW_REF", "1")
    x = _int16_tensor(torch.zeros(1, 1, 4, 4, dtype=torch.int16).tolist())
    with pytest.raises(ValueError, match="AvgPool2d kernel must be one of"):
        get_fixed_kernel(nn.AvgPool2d)(
            [x],
            {},
            _output_encoding(),
            {"kernel_size": 3, "stride": 1, "padding": 0},
        )


def test_pool2d_int16_kernel_rejects_oversize_padding_in_hw_ref_mode(monkeypatch):
    monkeypatch.setenv("AIMET_RX_HW_REF", "1")
    x = _int16_tensor(torch.zeros(1, 1, 4, 4, dtype=torch.int16).tolist())
    with pytest.raises(ValueError, match="MaxPool2d padding must be <= 3"):
        get_fixed_kernel(nn.MaxPool2d)(
            [x],
            {},
            _output_encoding(),
            {"kernel_size": 2, "stride": 1, "padding": 4},
        )


def test_pool2d_size_limits_off_by_default():
    """Without ``AIMET_RX_HW_REF=1`` the software reference accepts any size,
    so existing model tests (7×7 GAP, kernel_size=3 pooling, ...) keep working.
    """

    x = _int16_tensor(torch.zeros(1, 1, 7, 7, dtype=torch.int16).tolist())
    # kernel_size=3 is illegal in HW-strict but legal in lenient mode.
    out = get_fixed_kernel(nn.MaxPool2d)(
        [x],
        {},
        _output_encoding(),
        {"kernel_size": 3, "stride": 1, "padding": 0},
    )
    assert out.int_repr.shape == (1, 1, 5, 5)


def test_adaptive_avgpool2d_int16_kernel_rejects_non_unit_output():
    x = _int16_tensor([[[[1, 2], [3, 4]]]])
    with pytest.raises(ValueError, match="AdaptiveAvgPool2d.*output_size=\\(1,1\\)"):
        get_fixed_kernel(custom.AdaptiveAvgPool2d)(
            [x],
            {},
            _output_encoding(),
            {"output_size": (2, 2), "dim": (2, 3), "keepdim": True},
        )


def test_reshape_int16_kernel_rejects_changed_encoding():
    """Layout-only kernels share the same ``require_same_grid_encoding`` guard
    as MaxPool — pin the contract so future refactors don't silently relax it.
    """

    x = _int16_tensor([[[1, 2], [3, 4]]], scale=1.0, zero_point=0)
    with pytest.raises(ValueError, match="Reshape input/output scale"):
        get_fixed_kernel(custom.Reshape)(
            [x],
            {},
            _output_encoding(scale=0.5, zero_point=0),
            {"shape": [1, 4]},
        )


def test_permute_int16_kernel_rejects_changed_encoding():
    x = _int16_tensor([[[1, 2], [3, 4]]], scale=1.0, zero_point=0)
    with pytest.raises(ValueError, match="Permute input/output zero_point"):
        get_fixed_kernel(custom.Permute)(
            [x],
            {},
            _output_encoding(scale=1.0, zero_point=5),
            {"dims": [0, 2, 1]},
        )

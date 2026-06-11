# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
"""Golden npz regression (spec 13)."""

import pytest

torch = pytest.importorskip("torch")

from aimet_torch.fixed_point import requantize_int  # noqa: E402
from aimet_torch.fixed_point.requantize import SIM_TENSOR_DTYPE  # noqa: E402
from ._helpers import (  # noqa: E402
    assert_int_tensor_equal,
    load_golden_meta,
    load_golden_npz,
)

# Byte-stream parity guard for ADR-013 (int32 sim-tensor container):
#   - Carriers / requantize outputs now live in ``torch.int32`` containers.
#   - Committed goldens still hold the historical int16 byte-stream so any
#     downstream consumer (sidecar / ONNX / Ada200 Runtime) sees the same
#     bytes as before. The .to(torch.int16) cast below mimics that
#     serialization step; if it ever diverges, byte-stream equivalence is
#     broken and PR-4 must be rolled back per plan.


@pytest.mark.golden
def test_requantize_basic_golden_npz():
    golden = load_golden_npz("requantize_basic")
    meta = load_golden_meta("requantize_basic")
    assert meta.get("expected_dtype") == "int16"

    out = requantize_int(
        torch.tensor(golden["acc"]),
        torch.tensor(golden["multiplier"]),
        torch.tensor(golden["rshift"]),
        torch.tensor(golden["zero_point"]),
    )
    assert out.dtype is SIM_TENSOR_DTYPE
    # Byte-stream parity: int32 container narrowed to int16 must match committed bytes.
    assert_int_tensor_equal(out.to(torch.int16), golden["expected"], msg="requantize_basic")


@pytest.mark.golden
def test_conv2d_basic_golden_npz():
    import torch.nn as nn

    import aimet_torch.fixed_point.kernels  # noqa: F401
    from aimet_torch.fixed_point import Int16QuantizedTensor, OutputEncoding, get_fixed_kernel

    golden = load_golden_npz("conv2d_basic")
    x = Int16QuantizedTensor(
        int_repr=torch.tensor(golden["x_int"]),
        scale=torch.tensor(1.0, dtype=torch.float32),
        zero_point=torch.tensor(0, dtype=torch.int32),
    )
    weight = Int16QuantizedTensor(
        int_repr=torch.tensor(golden["weight_int"]),
        scale=torch.tensor(1.0, dtype=torch.float32),
        zero_point=torch.tensor(0, dtype=torch.int32),
    )
    output_encoding = OutputEncoding(
        scale=torch.tensor(1.0, dtype=torch.float32),
        zero_point=torch.tensor(golden["zero_point"]),
        qmin=-32768,
        qmax=32767,
        multiplier=torch.tensor(golden["multiplier"]),
        rshift=torch.tensor(golden["rshift"]),
    )
    out = get_fixed_kernel(nn.Conv2d)(
        [x],
        {"weight": weight},
        output_encoding,
        {"stride": 1, "padding": 0, "dilation": 1, "groups": 1},
    )
    assert out.int_repr.dtype is SIM_TENSOR_DTYPE
    assert_int_tensor_equal(
        out.int_repr.to(torch.int16), golden["expected"], msg="conv2d_basic"
    )


@pytest.mark.golden
def test_round_shift_basic_golden_npz():
    from aimet_torch.fixed_point import RoundingMode, round_shift

    golden = load_golden_npz("round_shift_basic")
    out = round_shift(
        torch.tensor(golden["x"]),
        torch.tensor(golden["rshift"]),
        RoundingMode.HALF_AWAY_FROM_ZERO,
    )
    assert_int_tensor_equal(out, golden["expected"], msg="round_shift_basic")


# ---- A-class single-op goldens (expand-golden-tests) ------------------------
# Each golden pins an in-place byte-stream snapshot of a kernel under the
# encoding/contract surface settled by the prior tasks (Subitems A/B/C +
# enforce-bitwidth-ceiling). Drift surfaces as an integer-equality miss
# rather than a float tolerance ratchet, so a future refactor that
# changes saturation timing, multiplier folding, or zp routing must
# acknowledge the regression here before landing.


@pytest.mark.golden
def test_linear_basic_golden_npz():
    """Linear (3->2, with bias) byte-stream lock under
    ``_requantize_output``. The path also exercises the int32-acc
    saturation contract added in S1."""

    import torch.nn as nn

    import aimet_torch.fixed_point.kernels  # noqa: F401
    from aimet_torch.fixed_point import (
        Int16QuantizedTensor,
        OutputEncoding,
        get_fixed_kernel,
    )

    golden = load_golden_npz("linear_basic")
    x = Int16QuantizedTensor(
        int_repr=torch.tensor(golden["x_int"]),
        scale=torch.tensor(1.0, dtype=torch.float32),
        zero_point=torch.tensor(0, dtype=torch.int32),
    )
    weight = Int16QuantizedTensor(
        int_repr=torch.tensor(golden["weight_int"]),
        scale=torch.tensor(1.0, dtype=torch.float32),
        zero_point=torch.tensor(0, dtype=torch.int32),
    )
    bias = torch.tensor(golden["bias"])
    output_encoding = OutputEncoding(
        scale=torch.tensor(1.0, dtype=torch.float32),
        zero_point=torch.tensor(golden["zero_point"]),
        qmin=-32768,
        qmax=32767,
        multiplier=torch.tensor(golden["multiplier"]),
        rshift=torch.tensor(golden["rshift"]),
    )
    out = get_fixed_kernel(nn.Linear)(
        [x],
        {"weight": weight, "bias": bias},
        output_encoding,
        {},
    )
    assert out.int_repr.dtype is SIM_TENSOR_DTYPE
    assert_int_tensor_equal(
        out.int_repr.to(torch.int16), golden["expected"], msg="linear_basic"
    )


@pytest.mark.golden
def test_avgpool2d_basic_golden_npz():
    """AvgPool2d 4x4→2x2 byte-stream lock. Pins the 1/N-folded
    multiplier+rshift contract and the int32 saturation gate at the
    requantize boundary (``require_int32_saturated_accumulator``)."""

    import torch.nn as nn

    import aimet_torch.fixed_point.kernels  # noqa: F401
    from aimet_torch.fixed_point import (
        Int16QuantizedTensor,
        OutputEncoding,
        get_fixed_kernel,
    )

    golden = load_golden_npz("avgpool2d_basic")
    x = Int16QuantizedTensor(
        int_repr=torch.tensor(golden["x_int"]),
        scale=torch.tensor(1.0, dtype=torch.float32),
        zero_point=torch.tensor(0, dtype=torch.int32),
    )
    output_encoding = OutputEncoding(
        scale=torch.tensor(1.0, dtype=torch.float32),
        zero_point=torch.tensor(golden["zero_point"]),
        qmin=-32768,
        qmax=32767,
        multiplier=torch.tensor(golden["multiplier"]),
        rshift=torch.tensor(golden["rshift"]),
    )
    out = get_fixed_kernel(nn.AvgPool2d)(
        [x],
        {},
        output_encoding,
        {
            "kernel_size": (2, 2),
            "stride": (2, 2),
            "padding": (0, 0),
            "count_include_pad": True,
            "reduce_size": 4,
        },
    )
    assert out.int_repr.dtype is SIM_TENSOR_DTYPE
    assert_int_tensor_equal(
        out.int_repr.to(torch.int16), golden["expected"], msg="avgpool2d_basic"
    )


@pytest.mark.golden
def test_mean_basic_golden_npz():
    """``custom.Mean`` over (2,3) keepdim=True with N=16. Pins the
    reduce_size contract from Subitem C plus the int32 saturation gate."""

    import aimet_torch.fixed_point.kernels  # noqa: F401
    from aimet_torch._base.nn.modules import custom
    from aimet_torch.fixed_point import (
        Int16QuantizedTensor,
        OutputEncoding,
        get_fixed_kernel,
    )

    golden = load_golden_npz("mean_basic")
    x = Int16QuantizedTensor(
        int_repr=torch.tensor(golden["x_int"]),
        scale=torch.tensor(1.0, dtype=torch.float32),
        zero_point=torch.tensor(0, dtype=torch.int32),
    )
    output_encoding = OutputEncoding(
        scale=torch.tensor(1.0, dtype=torch.float32),
        zero_point=torch.tensor(golden["zero_point"]),
        qmin=-32768,
        qmax=32767,
        multiplier=torch.tensor(golden["multiplier"]),
        rshift=torch.tensor(golden["rshift"]),
    )
    out = get_fixed_kernel(custom.Mean)(
        [x],
        {},
        output_encoding,
        {"dim": (2, 3), "keepdim": True, "reduce_size": 16},
    )
    assert out.int_repr.dtype is SIM_TENSOR_DTYPE
    assert_int_tensor_equal(
        out.int_repr.to(torch.int16), golden["expected"], msg="mean_basic"
    )


@pytest.mark.golden
def test_add_basic_golden_npz():
    """``custom.Add`` same-grid scale-aligned add. Pins the
    ``int32_add_sat`` + zero-point add-back path the eltwise
    ``_BinaryAlignedKernel`` exercises."""

    import aimet_torch.fixed_point.kernels  # noqa: F401
    from aimet_torch._base.nn.modules import custom
    from aimet_torch.fixed_point import (
        Int16QuantizedTensor,
        OutputEncoding,
        get_fixed_kernel,
    )

    golden = load_golden_npz("add_basic")
    a = Int16QuantizedTensor(
        int_repr=torch.tensor(golden["a_int"]),
        scale=torch.tensor(1.0, dtype=torch.float32),
        zero_point=torch.tensor(0, dtype=torch.int32),
    )
    b = Int16QuantizedTensor(
        int_repr=torch.tensor(golden["b_int"]),
        scale=torch.tensor(1.0, dtype=torch.float32),
        zero_point=torch.tensor(0, dtype=torch.int32),
    )
    output_encoding = OutputEncoding(
        scale=torch.tensor(1.0, dtype=torch.float32),
        zero_point=torch.tensor(golden["zero_point"]),
        qmin=-32768,
        qmax=32767,
    )
    out = get_fixed_kernel(custom.Add)([a, b], {}, output_encoding, {})
    assert out.int_repr.dtype is SIM_TENSOR_DTYPE
    assert_int_tensor_equal(
        out.int_repr.to(torch.int16), golden["expected"], msg="add_basic"
    )

#!/usr/bin/env python3
"""Regenerate committed golden vectors under tests/fixed_point/data/."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from aimet_torch.fixed_point import requantize_int

OUT = Path(__file__).resolve().parent


def _write_requantize_basic() -> None:
    acc = np.array([100, -100, 32766], dtype=np.int32)
    multiplier = np.array(16384, dtype=np.uint16)
    rshift = np.array(15, dtype=np.int8)
    y_zp = np.array(0, dtype=np.int32)

    out = requantize_int(
        torch.tensor(acc),
        torch.tensor(multiplier),
        torch.tensor(rshift),
        torch.tensor(y_zp),
    )

    np.savez(
        OUT / "requantize_basic.npz",
        acc=acc,
        multiplier=multiplier,
        rshift=rshift,
        zero_point=y_zp,
        expected=out.numpy(),
    )
    meta = {
        "description": (
            "Half-to-even requantize; matches "
            "test_requantize.test_requantize_int_basic_half_to_even"
        ),
        "expected_dtype": "int16",
    }
    (OUT / "requantize_basic.json").write_text(
        json.dumps(meta, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_conv2d_basic() -> None:
    import torch.nn as nn

    import aimet_torch.fixed_point.kernels  # noqa: F401
    from aimet_torch.fixed_point import Int16QuantizedTensor, OutputEncoding, get_fixed_kernel

    x = Int16QuantizedTensor(
        int_repr=torch.tensor([[[[1, 2], [3, 4]]]], dtype=torch.int16),
        scale=torch.tensor(1.0, dtype=torch.float32),
        zero_point=torch.tensor(0, dtype=torch.int32),
    )
    weight = Int16QuantizedTensor(
        int_repr=torch.tensor([[[[1, 0], [0, 1]]]], dtype=torch.int16),
        scale=torch.tensor(1.0, dtype=torch.float32),
        zero_point=torch.tensor(0, dtype=torch.int32),
    )
    output_encoding = OutputEncoding(
        scale=torch.tensor(1.0, dtype=torch.float32),
        zero_point=torch.tensor(0, dtype=torch.int32),
        qmin=-32768,
        qmax=32767,
        multiplier=torch.tensor(32767, dtype=torch.uint16),
        rshift=torch.tensor(15, dtype=torch.int8),
    )
    out = get_fixed_kernel(nn.Conv2d)(
        [x],
        {"weight": weight},
        output_encoding,
        {"stride": 1, "padding": 0, "dilation": 1, "groups": 1},
    )

    np.savez(
        OUT / "conv2d_basic.npz",
        x_int=x.int_repr.numpy(),
        weight_int=weight.int_repr.numpy(),
        multiplier=np.array(32767, dtype=np.uint16),
        rshift=np.array(15, dtype=np.int8),
        zero_point=np.array(0, dtype=np.int32),
        expected=out.int_repr.numpy(),
    )
    (OUT / "conv2d_basic.json").write_text(
        json.dumps(
            {
                "description": "Matches test_conv_linear.test_conv2d_int16_kernel_reference_case",
                "expected_dtype": "int16",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_round_shift_basic() -> None:
    from aimet_torch.fixed_point import RoundingMode, round_shift

    x = np.array([3, -3, 5, -5], dtype=np.int64)
    rshift = np.array(1, dtype=np.int8)
    out = round_shift(torch.tensor(x), torch.tensor(rshift), RoundingMode.HALF_AWAY_FROM_ZERO)
    np.savez(OUT / "round_shift_basic.npz", x=x, rshift=rshift, expected=out.numpy())
    (OUT / "round_shift_basic.json").write_text(
        json.dumps(
            {
                "description": "Matches test_requantize.test_round_shift_half_away_from_zero_signed",
                "rounding": "half_away_from_zero",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_linear_basic() -> None:
    """Linear ``out = round(x @ Wᵀ + b) >> rshift`` byte-stream golden.

    Shape: x = [1, 3], W = [2, 3], b = [2]. Hand-picked so the int32
    accumulator is well within INT32 ALU width and the result lands
    inside [INT16_QMIN, INT16_QMAX] without saturation, so the golden
    pins the multiplier+rshift+zp pipeline under nominal conditions.
    Mirrors the kernel hot path the new
    ``require_int32_saturated_accumulator`` contract guards.
    """

    import torch.nn as nn

    import aimet_torch.fixed_point.kernels  # noqa: F401
    from aimet_torch.fixed_point import (
        Int16QuantizedTensor,
        OutputEncoding,
        get_fixed_kernel,
    )

    x = Int16QuantizedTensor(
        int_repr=torch.tensor([[10, -20, 30]], dtype=torch.int16),
        scale=torch.tensor(1.0, dtype=torch.float32),
        zero_point=torch.tensor(0, dtype=torch.int32),
    )
    weight = Int16QuantizedTensor(
        int_repr=torch.tensor([[1, 2, 3], [-1, 0, 1]], dtype=torch.int16),
        scale=torch.tensor(1.0, dtype=torch.float32),
        zero_point=torch.tensor(0, dtype=torch.int32),
    )
    bias = torch.tensor([5, -5], dtype=torch.int32)
    output_encoding = OutputEncoding(
        scale=torch.tensor(1.0, dtype=torch.float32),
        zero_point=torch.tensor(0, dtype=torch.int32),
        qmin=-32768,
        qmax=32767,
        multiplier=torch.tensor(32767, dtype=torch.uint16),
        rshift=torch.tensor(15, dtype=torch.int8),
    )
    out = get_fixed_kernel(nn.Linear)(
        [x],
        {"weight": weight, "bias": bias},
        output_encoding,
        {},
    )

    np.savez(
        OUT / "linear_basic.npz",
        x_int=x.int_repr.numpy(),
        weight_int=weight.int_repr.numpy(),
        bias=bias.numpy(),
        multiplier=np.array(32767, dtype=np.uint16),
        rshift=np.array(15, dtype=np.int8),
        zero_point=np.array(0, dtype=np.int32),
        expected=out.int_repr.numpy(),
    )
    (OUT / "linear_basic.json").write_text(
        json.dumps(
            {
                "description": (
                    "Linear (3->2, with bias) byte-stream golden. "
                    "Pins requantize-int32 contract under "
                    "``_requantize_output``."
                ),
                "expected_dtype": "int16",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_avgpool2d_basic() -> None:
    """AvgPool2d byte-stream golden.

    Shape: x = [1, 1, 4, 4], kernel=2, stride=2 → out = [1, 1, 2, 2].
    Validates the ``int32_sum_sat → saturate_mac_accumulator → contract``
    pipeline (1/N folded into multiplier=⌊2^16/4⌋=16384, rshift=15) plus
    ``require_pool2d_operand_limits``. Pinned with ``count_include_pad=True``
    (the only supported path; ``False`` is rejected at adapter level
    per the count-include-pad guard).
    """

    import torch.nn as nn

    import aimet_torch.fixed_point.kernels  # noqa: F401
    from aimet_torch.fixed_point import (
        Int16QuantizedTensor,
        OutputEncoding,
        get_fixed_kernel,
    )

    x_int = np.array(
        [[[[1, 2, 3, 4],
           [5, 6, 7, 8],
           [9, 10, 11, 12],
           [13, 14, 15, 16]]]],
        dtype=np.int16,
    )
    x = Int16QuantizedTensor(
        int_repr=torch.tensor(x_int),
        scale=torch.tensor(1.0, dtype=torch.float32),
        zero_point=torch.tensor(0, dtype=torch.int32),
    )
    multiplier = np.array(16384, dtype=np.uint16)
    rshift = np.array(15, dtype=np.int8)
    output_encoding = OutputEncoding(
        scale=torch.tensor(1.0, dtype=torch.float32),
        zero_point=torch.tensor(0, dtype=torch.int32),
        qmin=-32768,
        qmax=32767,
        multiplier=torch.tensor(multiplier),
        rshift=torch.tensor(rshift),
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

    np.savez(
        OUT / "avgpool2d_basic.npz",
        x_int=x_int,
        multiplier=multiplier,
        rshift=rshift,
        zero_point=np.array(0, dtype=np.int32),
        expected=out.int_repr.numpy(),
    )
    (OUT / "avgpool2d_basic.json").write_text(
        json.dumps(
            {
                "description": (
                    "AvgPool2d 4x4→2x2 (kernel=2,stride=2). Pins "
                    "1/N-folded multiplier+rshift contract and the "
                    "INT32 saturation gate at the requantize boundary."
                ),
                "expected_dtype": "int16",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_mean_basic() -> None:
    """``custom.Mean`` byte-stream golden over a single axis.

    Shape: x = [1, 3, 4, 4] reduced over (2, 3) with keepdim=True → out =
    [1, 3, 1, 1]. Validates ``require_reduce_size_matches_extra`` (N=16
    folded into the output multiplier) plus the same int32 saturation
    contract as AvgPool2d.
    """

    import aimet_torch.fixed_point.kernels  # noqa: F401
    from aimet_torch._base.nn.modules import custom
    from aimet_torch.fixed_point import (
        Int16QuantizedTensor,
        OutputEncoding,
        get_fixed_kernel,
    )

    rng = np.random.default_rng(42)
    x_int = rng.integers(-50, 50, size=(1, 3, 4, 4), dtype=np.int16)
    x = Int16QuantizedTensor(
        int_repr=torch.tensor(x_int),
        scale=torch.tensor(1.0, dtype=torch.float32),
        zero_point=torch.tensor(0, dtype=torch.int32),
    )
    multiplier = np.array(4096, dtype=np.uint16)
    rshift = np.array(15, dtype=np.int8)
    output_encoding = OutputEncoding(
        scale=torch.tensor(1.0, dtype=torch.float32),
        zero_point=torch.tensor(0, dtype=torch.int32),
        qmin=-32768,
        qmax=32767,
        multiplier=torch.tensor(multiplier),
        rshift=torch.tensor(rshift),
    )
    out = get_fixed_kernel(custom.Mean)(
        [x],
        {},
        output_encoding,
        {"dim": (2, 3), "keepdim": True, "reduce_size": 16},
    )

    np.savez(
        OUT / "mean_basic.npz",
        x_int=x_int,
        multiplier=multiplier,
        rshift=rshift,
        zero_point=np.array(0, dtype=np.int32),
        expected=out.int_repr.numpy(),
    )
    (OUT / "mean_basic.json").write_text(
        json.dumps(
            {
                "description": (
                    "custom.Mean over (2,3) keepdim=True, N=16. Pins "
                    "reduce_size contract + 1/N-folded multiplier."
                ),
                "expected_dtype": "int16",
                "reduce_size": 16,
                "rng_seed": 42,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_add_basic() -> None:
    """``custom.Add`` scale-aligned byte-stream golden.

    Lefthand and righthand inputs share the same per-tensor encoding as
    the output, so ``align_centered_int32_to_output`` is a no-op centred
    sum — pins the saturated ``int32_add_sat`` path and zero-point
    add-back.
    """

    import aimet_torch.fixed_point.kernels  # noqa: F401
    from aimet_torch._base.nn.modules import custom
    from aimet_torch.fixed_point import (
        Int16QuantizedTensor,
        OutputEncoding,
        get_fixed_kernel,
    )

    a_int = np.array([[10, -20, 30, 40]], dtype=np.int16)
    b_int = np.array([[1, 2, -3, 4]], dtype=np.int16)
    a = Int16QuantizedTensor(
        int_repr=torch.tensor(a_int),
        scale=torch.tensor(1.0, dtype=torch.float32),
        zero_point=torch.tensor(0, dtype=torch.int32),
    )
    b = Int16QuantizedTensor(
        int_repr=torch.tensor(b_int),
        scale=torch.tensor(1.0, dtype=torch.float32),
        zero_point=torch.tensor(0, dtype=torch.int32),
    )
    output_encoding = OutputEncoding(
        scale=torch.tensor(1.0, dtype=torch.float32),
        zero_point=torch.tensor(0, dtype=torch.int32),
        qmin=-32768,
        qmax=32767,
    )
    out = get_fixed_kernel(custom.Add)([a, b], {}, output_encoding, {})

    np.savez(
        OUT / "add_basic.npz",
        a_int=a_int,
        b_int=b_int,
        zero_point=np.array(0, dtype=np.int32),
        expected=out.int_repr.numpy(),
    )
    (OUT / "add_basic.json").write_text(
        json.dumps(
            {
                "description": (
                    "custom.Add same-grid scale-aligned add. Pins "
                    "int32_add_sat + zp add-back path."
                ),
                "expected_dtype": "int16",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    _write_requantize_basic()
    _write_conv2d_basic()
    _write_round_shift_basic()
    _write_linear_basic()
    _write_avgpool2d_basic()
    _write_mean_basic()
    _write_add_basic()
    print(f"Wrote goldens under {OUT}")


if __name__ == "__main__":
    main()

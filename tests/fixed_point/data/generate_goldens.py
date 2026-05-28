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
    multiplier = np.array(16384, dtype=np.int16)
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
        multiplier=torch.tensor(32767, dtype=torch.int16),
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
        multiplier=np.array(32767, dtype=np.int16),
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


def main() -> None:
    _write_requantize_basic()
    _write_conv2d_basic()
    _write_round_shift_basic()
    print(f"Wrote goldens under {OUT}")


if __name__ == "__main__":
    main()

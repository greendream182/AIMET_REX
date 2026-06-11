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
"""P6 byte-stream-identity ops precision vs ideal float32 reference.

Sister harness to ``test_relu_int16_precision.py`` (P5). Same unified
gates (cosine > 0.9999, max_error_lsb_float < 1.0).

Hot paths under test (``aimet_torch/fixed_point/kernels/shape_ops.py``):

- ``Identity / Flatten / Reshape / Permute`` (``SAME_GRID_VALUE``):
  pure ``int_repr`` view/rewrap; output encoding must match input grid.
  No cross-scale path — kernel raises if encodings differ.
- ``Dropout`` (``SAME_GRID_OR_REQUANT``): eval no-op in value space,
  same-grid → direct rewrap; cross-grid → centered requantize.
- ``Pad`` (``SAME_GRID_OR_REQUANT``): align to output grid via
  ``_requantize_identity_output`` then ``F.pad`` with the quantized fill.

Spec ``doc/04_算子详细规格/04_11_其他数据操作算子.md`` (shape ops) +
``04_13_特殊激活与常量算子.md`` (Pad). For all of these the contract
is "byte-stream identity in value space" — quantization-induced error
collapses to either 0 (same-grid view) or one multiplier-fold (cross-grid
Dropout/Pad).

Coverage:

- Grids: signed i8 / i16
- SAME_GRID_VALUE ops: same-scale only (kernel contract rejects mismatch)
- Dropout / Pad: same-scale + cross-scale
"""

from __future__ import annotations

import math
from typing import Any, Dict

import pytest

torch = pytest.importorskip("torch")

import aimet_torch.fixed_point.kernels  # noqa: E402, F401

from torch import nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from aimet_torch._base.nn.modules import custom  # noqa: E402
from aimet_torch.fixed_point import (  # noqa: E402
    Int16QuantizedTensor,
    OutputEncoding,
    get_fixed_kernel,
)
from aimet_torch.fixed_point.metrics.accuracy import (  # noqa: E402
    cosine_similarity,
    max_error_lsb_float,
)
from aimet_torch.fixed_point.metrics.flags import (  # noqa: E402
    int16_eval_allow_debug_float,
)
from aimet_torch.fixed_point.offline.multiplier import quantize_multiplier  # noqa: E402
from aimet_torch.fixed_point.quant_grid import (  # noqa: E402
    QuantGridSpec,
    SIM_INT32_QUANT_GRIDS,
)
from aimet_torch.fixed_point.requantize import saturate_sim_tensor  # noqa: E402

_MIN_COSINE = 0.9999
_MAX_FLOAT_LSB = 1.0

_NUM_RANDOM_TRIALS = 32
_CROSS_SCALE_LOG_RATIO_SPAN = 0.05

_GRIDS: tuple[QuantGridSpec, ...] = tuple(
    g for g in SIM_INT32_QUANT_GRIDS if g.signed and g.name in {"i8", "i16"}
)
_GRID_PARAMS = tuple(pytest.param(g, id=g.name) for g in _GRIDS)


def _quantize(
    tensor: torch.Tensor,
    *,
    scale: float,
    grid: QuantGridSpec,
    zero_point: int,
) -> Int16QuantizedTensor:
    scale_t = torch.tensor(scale, dtype=torch.float32)
    zp_t = torch.tensor(zero_point, dtype=torch.int32)
    q = torch.round(tensor.to(torch.float32) / scale_t + zp_t.to(torch.float32))
    return Int16QuantizedTensor(
        int_repr=saturate_sim_tensor(q, grid.qmin, grid.qmax),
        scale=scale_t,
        zero_point=zp_t,
        qmin=grid.qmin,
        qmax=grid.qmax,
    )


def _output_encoding(
    *,
    scale_in: float,
    scale_out: float,
    grid: QuantGridSpec,
    zero_point: int,
    with_multiplier: bool,
) -> OutputEncoding:
    """Build OutputEncoding.

    When ``with_multiplier=False`` (same-grid contract), multiplier/rshift
    are None — the kernel takes the byte-stream-identity branch. When
    ``with_multiplier=True``, real_m = S_x / S_y is folded.
    """
    if with_multiplier:
        real_m = scale_in / scale_out
        multiplier, rshift = quantize_multiplier(
            torch.tensor(real_m, dtype=torch.float64)
        )
    else:
        multiplier = None
        rshift = None
    return OutputEncoding(
        scale=torch.tensor(scale_out, dtype=torch.float32),
        zero_point=torch.tensor(zero_point, dtype=torch.int32),
        qmin=grid.qmin,
        qmax=grid.qmax,
        multiplier=multiplier,
        rshift=rshift,
    )


def _random_fp32_input(
    grid: QuantGridSpec,
    gen: torch.Generator,
    *,
    shape: tuple[int, ...],
    same_scale: bool,
) -> tuple[torch.Tensor, float, float]:
    base = math.exp(
        torch.rand(1, generator=gen, dtype=torch.float32).item() * 2.5 - 3.5
    )
    if same_scale:
        scale_in = base
        scale_out = base
    else:
        log_y = (
            torch.rand(1, generator=gen, dtype=torch.float32).item() - 0.5
        ) * _CROSS_SCALE_LOG_RATIO_SPAN
        scale_in = base
        scale_out = base * math.exp(log_y)

    code_limit = max(8, grid.qmax // 2)
    qx = torch.randint(
        -code_limit, code_limit + 1, shape,
        generator=gen, dtype=torch.int32,
    )
    x = qx.to(torch.float32) * scale_in
    return x, scale_in, scale_out


def _assert_strict_gates(
    output: Int16QuantizedTensor,
    ref: torch.Tensor,
    *,
    label: str,
) -> None:
    assert ref.dtype == torch.float32
    with int16_eval_allow_debug_float():
        candidate = output.to_float().to(torch.float32)
    cos = cosine_similarity(ref.flatten(), candidate.flatten())
    if cos <= _MIN_COSINE:
        raise AssertionError(
            f"{label}: cosine_similarity {cos:.8f} <= {_MIN_COSINE} (required >)."
        )
    float_lsb = max_error_lsb_float(
        ref.flatten(), candidate.flatten(),
        output.scale, output.zero_point, output.qmin, output.qmax,
    )
    if float_lsb >= _MAX_FLOAT_LSB:
        raise AssertionError(
            f"{label}: max_error_lsb_float {float_lsb:.6f} >= {_MAX_FLOAT_LSB} "
            "(required float error < 1 * scale_out)."
        )


# ---------- SAME_GRID_VALUE: Identity / Flatten / Reshape / Permute ----------


def _run_same_grid_value(
    grid: QuantGridSpec,
    module_cls,
    *,
    shape: tuple[int, ...],
    extra_builder,
    ref_builder,
    seed_base: int,
) -> None:
    """SAME_GRID_VALUE ops: only same-scale (kernel rejects mismatch)."""
    for trial in range(_NUM_RANDOM_TRIALS):
        gen = torch.Generator().manual_seed(
            seed_base + trial * 9973 + hash(grid.name) % 10000
        )
        x, scale_in, scale_out = _random_fp32_input(
            grid, gen, shape=shape, same_scale=True,
        )
        x_q = _quantize(
            x, scale=scale_in, grid=grid, zero_point=grid.default_zero_point,
        )
        out_enc = _output_encoding(
            scale_in=scale_in, scale_out=scale_out,
            grid=grid, zero_point=grid.default_zero_point,
            with_multiplier=False,
        )
        extra = extra_builder()
        ref = ref_builder(x.to(torch.float32))
        output = get_fixed_kernel(module_cls)([x_q], {}, out_enc, extra)
        _assert_strict_gates(
            output, ref,
            label=f"{module_cls.__name__} same-scale {grid.name} trial={trial}",
        )


@pytest.mark.parametrize("grid", _GRID_PARAMS)
def test_identity_same_scale_random_fp32_per_grid(grid: QuantGridSpec):
    _run_same_grid_value(
        grid, nn.Identity,
        shape=(2, 8, 16),
        extra_builder=lambda: {},
        ref_builder=lambda x: x,
        seed_base=710_000,
    )


@pytest.mark.parametrize("grid", _GRID_PARAMS)
def test_flatten_same_scale_random_fp32_per_grid(grid: QuantGridSpec):
    _run_same_grid_value(
        grid, nn.Flatten,
        shape=(2, 4, 8, 8),
        extra_builder=lambda: {"start_dim": 1, "end_dim": -1},
        ref_builder=lambda x: torch.flatten(x, start_dim=1, end_dim=-1),
        seed_base=720_000,
    )


@pytest.mark.parametrize("grid", _GRID_PARAMS)
def test_reshape_same_scale_random_fp32_per_grid(grid: QuantGridSpec):
    _run_same_grid_value(
        grid, custom.Reshape,
        shape=(2, 4, 16),
        extra_builder=lambda: {"shape": (2, 64)},
        ref_builder=lambda x: torch.reshape(x, (2, 64)),
        seed_base=730_000,
    )


@pytest.mark.parametrize("grid", _GRID_PARAMS)
def test_permute_same_scale_random_fp32_per_grid(grid: QuantGridSpec):
    dims = (0, 2, 1, 3)
    _run_same_grid_value(
        grid, custom.Permute,
        shape=(2, 4, 8, 4),
        extra_builder=lambda: {"dims": dims},
        ref_builder=lambda x: torch.permute(x, dims),
        seed_base=740_000,
    )


# ---------- SAME_GRID_OR_REQUANT: Dropout / Pad ----------


def _run_dropout(
    grid: QuantGridSpec,
    *,
    same_scale: bool,
    seed_base: int,
) -> None:
    for trial in range(_NUM_RANDOM_TRIALS):
        gen = torch.Generator().manual_seed(
            seed_base + trial * 9973 + hash(grid.name) % 10000
        )
        x, scale_in, scale_out = _random_fp32_input(
            grid, gen, shape=(2, 8, 16), same_scale=same_scale,
        )
        x_q = _quantize(
            x, scale=scale_in, grid=grid, zero_point=grid.default_zero_point,
        )
        out_enc = _output_encoding(
            scale_in=scale_in, scale_out=scale_out,
            grid=grid, zero_point=grid.default_zero_point,
            with_multiplier=not same_scale,
        )
        ref = x.to(torch.float32)
        output = get_fixed_kernel(nn.Dropout)([x_q], {}, out_enc, {})
        mode = "same-scale" if same_scale else "cross-scale"
        _assert_strict_gates(
            output, ref,
            label=f"Dropout {mode} {grid.name} trial={trial}",
        )


@pytest.mark.parametrize("grid", _GRID_PARAMS)
def test_dropout_same_scale_random_fp32_per_grid(grid: QuantGridSpec):
    _run_dropout(grid, same_scale=True, seed_base=750_000)


@pytest.mark.parametrize("grid", _GRID_PARAMS)
def test_dropout_cross_scale_random_fp32_per_grid(grid: QuantGridSpec):
    _run_dropout(grid, same_scale=False, seed_base=760_000)


def _run_pad(
    grid: QuantGridSpec,
    *,
    same_scale: bool,
    seed_base: int,
    pad_mode: str = "constant",
    pad: tuple[int, ...] = (1, 2, 3, 1),
    pad_value_float: float = 0.0,
) -> None:
    extra: Dict[str, Any] = {"pad": pad, "mode": pad_mode}
    if pad_mode == "constant":
        extra["value"] = pad_value_float
    for trial in range(_NUM_RANDOM_TRIALS):
        gen = torch.Generator().manual_seed(
            seed_base + trial * 9973 + hash(grid.name) % 10000
        )
        x, scale_in, scale_out = _random_fp32_input(
            grid, gen, shape=(1, 4, 8, 8), same_scale=same_scale,
        )
        x_q = _quantize(
            x, scale=scale_in, grid=grid, zero_point=grid.default_zero_point,
        )
        out_enc = _output_encoding(
            scale_in=scale_in, scale_out=scale_out,
            grid=grid, zero_point=grid.default_zero_point,
            with_multiplier=not same_scale,
        )
        if pad_mode == "constant":
            ref = F.pad(
                x.to(torch.float32), pad,
                mode="constant", value=pad_value_float,
            )
        else:
            ref = F.pad(x.to(torch.float32), pad, mode=pad_mode)
        output = get_fixed_kernel(custom.Pad)([x_q], {}, out_enc, extra)
        scale_label = "same-scale" if same_scale else "cross-scale"
        _assert_strict_gates(
            output, ref,
            label=f"Pad {pad_mode} {scale_label} {grid.name} trial={trial}",
        )


@pytest.mark.parametrize("grid", _GRID_PARAMS)
def test_pad_same_scale_random_fp32_per_grid(grid: QuantGridSpec):
    _run_pad(grid, same_scale=True, seed_base=770_000)


@pytest.mark.parametrize("grid", _GRID_PARAMS)
def test_pad_cross_scale_random_fp32_per_grid(grid: QuantGridSpec):
    _run_pad(grid, same_scale=False, seed_base=780_000)


# Layer C4: spec 04_11 §4.11.1 explicitly lists ``constant/reflect/replicate``.
# Reflect/replicate are byte-stream position copies (no arithmetic), so the
# fixed-point path is bit-exact under same-scale and bound by the existing
# requantize gate under cross-scale. Use a smaller symmetric pad on the last
# two spatial dims so reflect's "pad < dim" requirement is satisfied for
# every grid.
_PAD_REFL_REPL_PAD = (1, 1, 1, 1)


@pytest.mark.parametrize("grid", _GRID_PARAMS)
def test_pad_replicate_same_scale_random_fp32_per_grid(grid: QuantGridSpec):
    _run_pad(
        grid, same_scale=True, seed_base=771_000,
        pad_mode="replicate", pad=_PAD_REFL_REPL_PAD,
    )


@pytest.mark.parametrize("grid", _GRID_PARAMS)
def test_pad_replicate_cross_scale_random_fp32_per_grid(grid: QuantGridSpec):
    _run_pad(
        grid, same_scale=False, seed_base=781_000,
        pad_mode="replicate", pad=_PAD_REFL_REPL_PAD,
    )


@pytest.mark.parametrize("grid", _GRID_PARAMS)
def test_pad_reflect_same_scale_random_fp32_per_grid(grid: QuantGridSpec):
    _run_pad(
        grid, same_scale=True, seed_base=772_000,
        pad_mode="reflect", pad=_PAD_REFL_REPL_PAD,
    )


@pytest.mark.parametrize("grid", _GRID_PARAMS)
def test_pad_reflect_cross_scale_random_fp32_per_grid(grid: QuantGridSpec):
    _run_pad(
        grid, same_scale=False, seed_base=782_000,
        pad_mode="reflect", pad=_PAD_REFL_REPL_PAD,
    )


# ---------- SAME_GRID_OR_REQUANT: Concat (multi-branch align-then-cat) ------


def _run_concat(
    grid: QuantGridSpec,
    *,
    same_scale: bool,
    seed_base: int,
) -> None:
    """``custom.Concat`` aligns each branch to the output grid via
    ``align_centered_int32_to_output`` and then ``torch.cat``-s the int_repr.

    With ``same_scale=True`` all branches and output share scale → expect
    byte-stream identity (lsb_max=0). With ``same_scale=False`` each branch
    sits on a (slightly) different grid → one requantize per branch.
    """
    axis = 1  # channels-dim
    for trial in range(_NUM_RANDOM_TRIALS):
        gen = torch.Generator().manual_seed(
            seed_base + trial * 9973 + hash(grid.name) % 10000
        )
        # Three branches with their own input scales sharing the same output.
        base = math.exp(
            torch.rand(1, generator=gen, dtype=torch.float32).item() * 2.5 - 3.5
        )
        if same_scale:
            scale_a = scale_b = scale_c = base
        else:
            log_ratio_a = (
                torch.rand(1, generator=gen, dtype=torch.float32).item() - 0.5
            ) * _CROSS_SCALE_LOG_RATIO_SPAN
            log_ratio_c = (
                torch.rand(1, generator=gen, dtype=torch.float32).item() - 0.5
            ) * _CROSS_SCALE_LOG_RATIO_SPAN
            scale_a = base * math.exp(log_ratio_a)
            scale_b = base
            scale_c = base * math.exp(log_ratio_c)
        scale_out = base

        code_limit = max(8, grid.qmax // 2)
        shape_a = (2, 4, 8)
        shape_b = (2, 6, 8)
        shape_c = (2, 2, 8)
        qa = torch.randint(
            -code_limit, code_limit + 1, shape_a, generator=gen, dtype=torch.int32,
        )
        qb = torch.randint(
            -code_limit, code_limit + 1, shape_b, generator=gen, dtype=torch.int32,
        )
        qc = torch.randint(
            -code_limit, code_limit + 1, shape_c, generator=gen, dtype=torch.int32,
        )
        a = qa.to(torch.float32) * scale_a
        b = qb.to(torch.float32) * scale_b
        c = qc.to(torch.float32) * scale_c

        a_q = _quantize(
            a, scale=scale_a, grid=grid, zero_point=grid.default_zero_point,
        )
        b_q = _quantize(
            b, scale=scale_b, grid=grid, zero_point=grid.default_zero_point,
        )
        c_q = _quantize(
            c, scale=scale_c, grid=grid, zero_point=grid.default_zero_point,
        )

        # For Concat each input must arrive with its own (multiplier, rshift)
        # ratio to the output grid. The kernel's ``align_centered_int32_to_output``
        # consumes ``input.scale → output.scale``; we therefore *only* set
        # multiplier/rshift on the output_encoding when cross-scale, mirroring
        # the Pad / Dropout pattern. Same-scale takes the byte-stream identity
        # branch (no rescale).
        out_enc = _output_encoding(
            scale_in=scale_a if same_scale else scale_b,
            scale_out=scale_out,
            grid=grid, zero_point=grid.default_zero_point,
            with_multiplier=False,
        )

        ref = torch.cat(
            [a.to(torch.float32), b.to(torch.float32), c.to(torch.float32)],
            dim=axis,
        )
        output = get_fixed_kernel(custom.Concat)(
            [a_q, b_q, c_q], {}, out_enc, {"axis": axis},
        )
        mode = "same-scale" if same_scale else "cross-scale"
        _assert_strict_gates(
            output, ref,
            label=f"Concat {mode} {grid.name} trial={trial}",
        )


@pytest.mark.parametrize("grid", _GRID_PARAMS)
def test_concat_same_scale_random_fp32_per_grid(grid: QuantGridSpec):
    _run_concat(grid, same_scale=True, seed_base=790_000)


@pytest.mark.parametrize("grid", _GRID_PARAMS)
def test_concat_cross_scale_random_fp32_per_grid(grid: QuantGridSpec):
    _run_concat(grid, same_scale=False, seed_base=800_000)

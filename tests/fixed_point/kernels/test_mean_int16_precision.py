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
"""custom.Mean single-op precision vs ideal float32 reference.

Sister harness to ``test_avgpool2d_int16_precision.py``. **Same unified
gates** (cosine > 0.9999, max_error_lsb_float < 1.0) and same root
KNOWN_LIMIT (i8 grid + reduction kernels: SNR ceiling is
``code_limit/sqrt(3 N)`` and on i8 + ``N ≥ 4`` plateaus below the
0.9999 cos floor regardless of how clean the kernel arithmetic is).

Per the precision-validation policy (mirror Divide legacy-path), i8
grids are **kept in parametrize and marked ``pytest.mark.xfail(strict
=True)``** so the actual snapshot is recorded in
``doc/precision_validation.md#custommean`` and any unexpected pass turns
the test red.

Hot path under test:
``centered → int32_sum_sat(dim) → saturate_mac_accumulator → requantize_int``.

The ``1/N`` reduction (``N = ∏ shape[d] for d in dim``) is folded
offline into ``M/rshift`` per spec ``doc/04_算子详细规格/04_06_统计类
算子.md`` §4.6.1; the kernel re-derives ``N`` from ``dim`` + input
shape and asserts via ``require_reduce_size_matches_extra``. This file
mirrors that fold step (``real_m = scale_in / (N * scale_out)``).

Coverage:

- Two reduction patterns:
  * ``dim=(2,3), keepdim=True``: spatial mean (matches AdaptiveAvgPool2d
    output_size=(1,1) hot path) on a 4D tensor (1, C, H, W).
  * ``dim=-1, keepdim=False``: last-axis mean — common ReduceMean
    pattern; flattens 1D output to drive a different reduction shape
    through the same kernel.
- Grids: signed i16 / i32 (i8 excluded as KNOWN_LIMIT, see module docstring).
- Modes: same-scale, small cross-scale (±5%).
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

import aimet_torch.fixed_point.kernels  # noqa: E402, F401

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

# Two reduction patterns, parametrized by (input_shape, dim, keepdim, label).
_MEAN_REDUCTIONS: tuple[tuple[tuple[int, ...], tuple[int, ...] | int, bool, str], ...] = (
    ((1, 4, 8, 8), (2, 3), True, "spatial-HW"),
    ((4, 64), -1, False, "last-axis"),
)

_MEAN_GRIDS: tuple[QuantGridSpec, ...] = tuple(
    g for g in SIM_INT32_QUANT_GRIDS if g.signed
)

_I8_XFAIL_REASON = (
    "Mean on i8: reduction-class SNR ceiling — output stddev ∝ "
    "code_limit/sqrt(3·N); i8 (qmax=127) cannot reach SNR > 100 for "
    "N ≥ 4. lsb_max stays ≈ 0.5 (kernel correct), only cosine fails "
    "the unified 0.9999 floor. See doc/precision_validation.md#custommean."
)


def _grid_param(grid: QuantGridSpec):
    if grid.name == "i8":
        return pytest.param(
            grid,
            id=grid.name,
            marks=pytest.mark.xfail(strict=True, reason=_I8_XFAIL_REASON),
        )
    return pytest.param(grid, id=grid.name)


_MEAN_GRID_PARAMS = tuple(_grid_param(g) for g in _MEAN_GRIDS)


def _quantize_float(
    tensor: torch.Tensor,
    *,
    scale: float,
    grid: QuantGridSpec,
    zero_point: int | None = None,
) -> Int16QuantizedTensor:
    zp = grid.default_zero_point if zero_point is None else zero_point
    scale_t = torch.tensor(scale, dtype=torch.float32)
    zp_t = torch.tensor(zp, dtype=torch.int32)
    q = torch.round(tensor.to(torch.float32) / scale_t + zp_t.to(torch.float32))
    return Int16QuantizedTensor(
        int_repr=saturate_sim_tensor(q, grid.qmin, grid.qmax),
        scale=scale_t,
        zero_point=zp_t,
        qmin=grid.qmin,
        qmax=grid.qmax,
    )


def _output_encoding_with_fold(
    *,
    scale_in: float,
    scale_out: float,
    reduce_size: int,
    grid: QuantGridSpec,
    zero_point: int | None = None,
) -> OutputEncoding:
    real_m = scale_in / (reduce_size * scale_out)
    multiplier, rshift = quantize_multiplier(
        torch.tensor(real_m, dtype=torch.float64)
    )
    zp = grid.default_zero_point if zero_point is None else zero_point
    return OutputEncoding(
        scale=torch.tensor(scale_out, dtype=torch.float32),
        zero_point=torch.tensor(zp, dtype=torch.int32),
        qmin=grid.qmin,
        qmax=grid.qmax,
        multiplier=multiplier,
        rshift=rshift,
    )


def _random_fp32_input(
    grid: QuantGridSpec,
    gen: torch.Generator,
    *,
    same_scale: bool,
    shape: tuple[int, ...],
    zero_point: int | None = None,
) -> tuple[torch.Tensor, float, float]:
    scale_out = math.exp(
        torch.rand(1, generator=gen, dtype=torch.float32).item() * 2.5 - 3.5
    )
    if same_scale:
        scale_in = scale_out
    else:
        log_ratio = (
            torch.rand(1, generator=gen, dtype=torch.float32).item() - 0.5
        ) * _CROSS_SCALE_LOG_RATIO_SPAN
        scale_in = scale_out * math.exp(log_ratio)

    # i8 / u8 get the largest physically representable code envelope
    # (~75% of qmax) so the xfail snapshot reflects the true SNR ceiling.
    if grid.name in {"i8", "u8"}:
        code_limit = 96
    else:
        code_limit = 4096

    qx = torch.randint(
        -code_limit, code_limit + 1, shape,
        generator=gen, dtype=torch.int32,
    )
    # x is centered float space; ``_quantize_float`` re-applies zp.
    del zero_point  # noqa: F841 — accepted for symmetry with the avgpool test
    x = qx.to(torch.float32) * scale_in
    return x, scale_in, scale_out


def _assert_strict_fp32_mean_gates(
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


def _run_random_trials(
    grid: QuantGridSpec,
    reduction: tuple[tuple[int, ...], tuple[int, ...] | int, bool, str],
    *,
    same_scale: bool,
    seed_base: int,
    zero_point: int | None = None,
) -> None:
    shape, dim, keepdim, label = reduction
    if isinstance(dim, int):
        reduce_size = shape[dim]
    else:
        reduce_size = 1
        for d in dim:
            reduce_size *= shape[d]
    for trial in range(_NUM_RANDOM_TRIALS):
        gen = torch.Generator().manual_seed(
            seed_base + trial * 9973
            + hash((grid.name, label, zero_point)) % 10000
        )
        x, scale_in, scale_out = _random_fp32_input(
            grid, gen, same_scale=same_scale, shape=shape,
            zero_point=zero_point,
        )
        x_q = _quantize_float(
            x, scale=scale_in, grid=grid, zero_point=zero_point,
        )
        out_enc = _output_encoding_with_fold(
            scale_in=scale_in,
            scale_out=scale_out,
            reduce_size=reduce_size,
            grid=grid,
            zero_point=zero_point,
        )
        ref = torch.mean(x.to(torch.float32), dim=dim, keepdim=keepdim)
        output = get_fixed_kernel(custom.Mean)(
            [x_q],
            {},
            out_enc,
            {
                "dim": dim,
                "keepdim": keepdim,
                "reduce_size": reduce_size,
            },
        )
        mode = "same-scale" if same_scale else "cross-scale"
        _assert_strict_fp32_mean_gates(
            output, ref,
            label=f"Mean {mode} {grid.name} {label} trial={trial}",
        )


@pytest.mark.parametrize("reduction", _MEAN_REDUCTIONS, ids=lambda r: r[3])
@pytest.mark.parametrize("grid", _MEAN_GRID_PARAMS)
def test_mean_same_scale_random_fp32_per_grid(
    grid: QuantGridSpec,
    reduction: tuple[tuple[int, ...], tuple[int, ...] | int, bool, str],
):
    _run_random_trials(grid, reduction, same_scale=True, seed_base=130_000)


@pytest.mark.parametrize("reduction", _MEAN_REDUCTIONS, ids=lambda r: r[3])
@pytest.mark.parametrize("grid", _MEAN_GRID_PARAMS)
def test_mean_cross_scale_random_fp32_per_grid(
    grid: QuantGridSpec,
    reduction: tuple[tuple[int, ...], tuple[int, ...] | int, bool, str],
):
    _run_random_trials(grid, reduction, same_scale=False, seed_base=140_000)


# ----------------------------------------------------------------------------
# Unsigned grid coverage — Layer C3 (test coverage gap, not spec limitation).
# Spec ``04_06 §4.6.1`` Mean explicitly allows ``u8 / u16`` inputs paired with
# non-zero ``zero_point`` calibration. We spot-check the spatial-HW reduction
# (matches AdaptiveAvgPool2d hot path) on u8/u16 with ``zp = qmax//2``. u8 is
# expected to fail the unified 0.9999 cos floor (same SNR ceiling as i8 +
# N≥4 reduction) and is xfail strict=True; u16 should PASS unmarked.
# ----------------------------------------------------------------------------
_U8_UNSIGNED_XFAIL_REASON = (
    "Mean on u8: same SNR ceiling as i8 — centered code envelope ~96 "
    "(qmax=255 with zp=qmax//2=127) gives the same code budget as i8, so "
    "cos < 0.9999 for any N ≥ 4 reduction. lsb_max stays ≈ 0.5 (kernel "
    "arithmetic correct). xfail strict=True mirrors i8. See "
    "doc/precision_validation.md#custommean."
)


def _unsigned_grid_param(g: QuantGridSpec):
    if g.name == "u8":
        return pytest.param(
            g, g.qmax // 2, id=f"{g.name}-zp{g.qmax // 2}",
            marks=pytest.mark.xfail(strict=True, reason=_U8_UNSIGNED_XFAIL_REASON),
        )
    return pytest.param(g, g.qmax // 2, id=f"{g.name}-zp{g.qmax // 2}")


_MEAN_UNSIGNED_PARAMS = tuple(
    _unsigned_grid_param(g)
    for g in SIM_INT32_QUANT_GRIDS
    if (not g.signed) and g.name in {"u8", "u16"}
)

# Use the spatial-HW reduction (N=16) — large enough to exercise the
# reduction hot path, small enough that u16 stays comfortably above floor.
_MEAN_UNSIGNED_REDUCTION = _MEAN_REDUCTIONS[0]


@pytest.mark.parametrize("grid,zp", _MEAN_UNSIGNED_PARAMS)
def test_mean_same_scale_random_fp32_unsigned_zp_centered(
    grid: QuantGridSpec, zp: int,
):
    _run_random_trials(
        grid, _MEAN_UNSIGNED_REDUCTION, same_scale=True,
        seed_base=150_000, zero_point=zp,
    )


@pytest.mark.parametrize("grid,zp", _MEAN_UNSIGNED_PARAMS)
def test_mean_cross_scale_random_fp32_unsigned_zp_centered(
    grid: QuantGridSpec, zp: int,
):
    _run_random_trials(
        grid, _MEAN_UNSIGNED_REDUCTION, same_scale=False,
        seed_base=160_000, zero_point=zp,
    )

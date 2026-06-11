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
"""custom.Subtract single-op precision vs ideal float32 reference.

Mirrors the Path-B harness in ``test_add_ideal_float_reference.py``:
  * Random fp32 activations on each *signed* ``QuantGridSpec`` (i8/i16/i32)
  * Same-scale and small cross-scale ratio (±5%, tighter ±2% on i8)
  * Reference is the ideal float subtraction ``ref = a - b`` (no requant,
    no quantization on the reference path).
  * Gates: cosine_similarity > 0.9999 AND max_error_lsb_float < 1.

Subtract shares the ``_BinaryAlignedKernel`` skeleton with Add — the only
difference at the kernel hot path is ``int32_sub_sat`` vs ``int32_add_sat``
(see ``aimet_torch/fixed_point/kernels/eltwise.py``). Reusing the Add Path-B
gate guarantees Subtract holds the same precision floor on signed grids,
with the case documented at
``doc/precision_validation.md#customsubtract-requantizing``.

Operator-level limitation (intentionally not gated): unsigned grids
(u8/u16/u32) with ``default_zero_point = 0`` cannot represent ``a - b < 0``.
Add's harness happens to pass on those grids because the random codes are
clamped to ``[0, code_limit]`` so ``a + b ≥ 0`` is guaranteed; the same
inputs run through Subtract produce negative outputs that are then
saturated to ``qmin = 0``, breaking the strict ``max_error_lsb < 1`` gate.
This is a spec-level property of unsigned encodings rather than a Subtract
kernel bug, so we restrict the parametrize to signed grids and call it out
in ``doc/precision_validation.md`` rather than papering over it with an
asymmetric ``zero_point`` (which would change the contract being tested).
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
from aimet_torch.fixed_point.quant_grid import (  # noqa: E402
    QuantGridSpec,
    SIM_INT32_QUANT_GRIDS,
)

# Unsigned grids with zp=0 cannot represent negative differences; see the
# module docstring. We test signed grids only — that's where the kernel's
# precision floor is meaningfully observable.
_SUBTRACT_GRIDS: tuple[QuantGridSpec, ...] = tuple(
    g for g in SIM_INT32_QUANT_GRIDS if g.signed
)
from aimet_torch.fixed_point.requantize import saturate_sim_tensor  # noqa: E402

# Strict gates (must hold with strict ``>`` / ``<``, matching Add Path-B).
_MIN_COSINE = 0.9999
_MAX_FLOAT_LSB = 1.0

# Larger sample volume than Add Path-B (which uses 32x64). The precision
# validation campaign for single-op kernels needs enough draws to surface
# rare scale/zp/code-pattern combinations; 128 trials × 256 elements per
# trial = 32_768 effective comparisons per (grid, mode), still under a
# second per parametrize on CPU. Bumped per the precision-record review.
_NUM_RANDOM_TRIALS = 128
_TRIAL_NUMEL = 256
_CROSS_SCALE_LOG_RATIO_SPAN = 0.05
_CROSS_SCALE_LOG_RATIO_SPAN_I8 = 0.02


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


def _output_encoding(
    *, scale: float, grid: QuantGridSpec, zero_point: int | None = None
) -> OutputEncoding:
    zp = grid.default_zero_point if zero_point is None else zero_point
    return OutputEncoding(
        scale=torch.tensor(scale, dtype=torch.float32),
        zero_point=torch.tensor(zp, dtype=torch.int32),
        qmin=grid.qmin,
        qmax=grid.qmax,
        multiplier=torch.tensor(32767, dtype=torch.uint16),
        rshift=torch.tensor(15, dtype=torch.int8),
    )


def _random_fp32_pair(
    grid: QuantGridSpec,
    gen: torch.Generator,
    *,
    same_scale: bool,
    numel: int = _TRIAL_NUMEL,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    scale_out = math.exp(
        torch.rand(1, generator=gen, dtype=torch.float32).item() * 2.5 - 3.5
    )
    if same_scale:
        scale_a = scale_b = scale_out
    else:
        span = (
            _CROSS_SCALE_LOG_RATIO_SPAN_I8
            if grid.name == "i8"
            else _CROSS_SCALE_LOG_RATIO_SPAN
        )
        log_ratio = (
            torch.rand(1, generator=gen, dtype=torch.float32).item() - 0.5
        ) * span
        scale_a = scale_out * math.exp(log_ratio)
        scale_b = scale_out

    zp = grid.default_zero_point
    code_limit = 24 if grid.name == "i8" else min(64, (grid.qmax - grid.qmin) // 8)
    if grid.signed:
        qa = torch.randint(
            -code_limit, code_limit + 1, (numel,), generator=gen, dtype=torch.int32
        )
        qb = torch.randint(
            -code_limit, code_limit + 1, (numel,), generator=gen, dtype=torch.int32
        )
    else:
        qa = torch.randint(0, code_limit + 1, (numel,), generator=gen, dtype=torch.int32)
        qb = torch.randint(0, code_limit + 1, (numel,), generator=gen, dtype=torch.int32)

    a = (qa.to(torch.float32) - float(zp)) * scale_a
    b = (qb.to(torch.float32) - float(zp)) * scale_b
    return a, b, scale_a, scale_b


def _fixed_subtract_vs_ideal_float(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    scale_a: float,
    scale_b: float,
    out_enc: OutputEncoding,
    grid: QuantGridSpec,
    zp_in: int | None = None,
) -> tuple[Int16QuantizedTensor, torch.Tensor]:
    lhs = _quantize_float(a, scale=scale_a, grid=grid, zero_point=zp_in)
    rhs = _quantize_float(b, scale=scale_b, grid=grid, zero_point=zp_in)
    ref = a.to(torch.float32) - b.to(torch.float32)
    output = get_fixed_kernel(custom.Subtract)([lhs, rhs], {}, out_enc, {})
    return output, ref


def _assert_strict_fp32_subtract_gates(
    output: Int16QuantizedTensor,
    ref: torch.Tensor,
    *,
    label: str,
) -> None:
    assert ref.dtype == torch.float32
    with int16_eval_allow_debug_float():
        candidate = output.to_float().to(torch.float32)
    cos = cosine_similarity(ref, candidate)
    if cos <= _MIN_COSINE:
        raise AssertionError(
            f"{label}: cosine_similarity {cos:.8f} <= {_MIN_COSINE} (required >)."
        )
    float_lsb = max_error_lsb_float(
        ref,
        candidate,
        output.scale,
        output.zero_point,
        output.qmin,
        output.qmax,
    )
    if float_lsb >= _MAX_FLOAT_LSB:
        raise AssertionError(
            f"{label}: max_error_lsb_float {float_lsb:.6f} >= {_MAX_FLOAT_LSB} "
            "(required float error < 1 * scale_out)."
        )


def _run_random_trials(grid: QuantGridSpec, *, same_scale: bool, seed_base: int) -> None:
    for trial in range(_NUM_RANDOM_TRIALS):
        gen = torch.Generator().manual_seed(
            seed_base + trial * 9973 + hash(grid.name) % 10000
        )
        a, b, scale_a, scale_b = _random_fp32_pair(grid, gen, same_scale=same_scale)
        out_scale = scale_b
        out_enc = _output_encoding(scale=out_scale, grid=grid)
        output, ref = _fixed_subtract_vs_ideal_float(
            a, b, scale_a=scale_a, scale_b=scale_b, out_enc=out_enc, grid=grid
        )
        mode = "same-scale" if same_scale else "cross-scale"
        _assert_strict_fp32_subtract_gates(
            output,
            ref,
            label=f"Subtract path B {mode} {grid.name} trial={trial}",
        )


@pytest.mark.parametrize("grid", _SUBTRACT_GRIDS, ids=lambda g: g.name)
def test_subtract_path_b_same_scale_random_fp32_per_grid(grid: QuantGridSpec):
    _run_random_trials(grid, same_scale=True, seed_base=30_000)


@pytest.mark.parametrize("grid", _SUBTRACT_GRIDS, ids=lambda g: g.name)
def test_subtract_path_b_cross_scale_random_fp32_per_grid(grid: QuantGridSpec):
    _run_random_trials(grid, same_scale=False, seed_base=40_000)

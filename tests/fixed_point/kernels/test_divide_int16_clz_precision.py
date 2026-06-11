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
"""custom.Divide spec §4.3.4 (reciprocal CLZ LUT) precision vs FP32.

Companion to ``test_divide_int16_precision.py`` (legacy integer-div
path, kept ``xfail`` at the unified 0.9999/1.0 floor) and
``test_divide_int16.py`` (point tests + BN path).

This file covers the **spec §4.3.4 reciprocal-LUT path**: the kernel
decomposes ``y = a / b`` into ``recip = 1/b`` (CLZ LUT) followed by
``y = a * recip`` (Multiply). Worst-case error is ~1 LSB at LUT-domain
inputs, so this path is expected to meet the unified Add/Sub/Mul floor.

Skipped when ``abc_lut-shuai`` (the source of the reciprocal CLZ LUT)
is not in the workspace, mirroring ``test_clz_reciprocal_golden.py``.

Operator-level constraints intentionally *baked in* to the harness:

1. **LUT input domain**: the abc reciprocal LUT is fitted on
   ``b ∈ [-6, 6]``. We pick ``scale_den`` and ``code_limit`` so the
   actual ``b * scale_den`` magnitude stays well inside that band;
   exceeding it would saturate the LUT input alignment step and the
   measured precision would reflect saturation, not the LUT's intrinsic
   floor. This is **the** structural reason this file exists separately
   from ``test_divide_int16_precision.py`` — random-scale inputs cannot
   guarantee LUT-domain residency without a calibrated, per-test LUT.
2. **Non-zero denominator**: ``|b_code| >= max(4, code_limit/8)`` so
   the LUT's internal "zero-saturation" branch never fires; the eps
   semantics are tested separately in
   ``test_divide_int16_eps_clamp.py`` (when present).
3. **Signed grids only**: same operator-level KNOWN_LIMIT as Subtract /
   Multiply / Divide-legacy — unsigned ``zero_point=0`` cannot represent
   negative quotients.
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
from aimet_torch.fixed_point.kernels.clz_lut import (  # noqa: E402
    try_load_default_reciprocal_clz_lut,
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
from aimet_torch.fixed_point.requantize import saturate_sim_tensor  # noqa: E402

_RECIP_LUT = try_load_default_reciprocal_clz_lut()
_HAS_LUT = _RECIP_LUT is not None

pytestmark = pytest.mark.skipif(
    not _HAS_LUT,
    reason="abc_lut-shuai reciprocal_clz_lut.json not in workspace",
)

# Unified floor — same as Add/Sub/Mul, NOT lowered for Divide.
_MIN_COSINE = 0.9999
_MAX_FLOAT_LSB = 1.0

_NUM_RANDOM_TRIALS = 128
_TRIAL_NUMEL = 256

# abc reciprocal LUT properties (read once at import).
_LUT_IN_FMAX = float(_RECIP_LUT["quantization"]["input"]["fmax"]) if _HAS_LUT else 6.0
_LUT_OUT_SCALE = (
    float(_RECIP_LUT["quantization"]["output"]["scale"]) if _HAS_LUT else 3.05e-3
)
_LUT_MARGIN = 0.5

# ``scale_out`` is set to ``_OUTPUT_SCALE_RATIO * _LUT_OUT_SCALE``. This
# is the empirically-located precision sweet spot for Divide-via-LUT:
#   * Smaller ratio → output grid out-resolves the LUT, ``lsb_max``
#     inflates by ``LUT_OUT_LSB / output_LSB``.
#   * Larger ratio → output grid becomes too coarse, ``cosine`` drops
#     below 0.9999 (output grid quantization noise dominates).
#   * 0.4x sits at ``cos ≈ 0.99999, lsb_max ≈ 0.66`` for LUT-domain
#     inputs in the abc default fitted range. Tighter than this is
#     achievable but couples too tightly to the abc segment count;
#     ``0.4`` leaves room for LUT-asset upgrades that shrink LUT-out
#     scale without breaking this test's threshold.
_OUTPUT_SCALE_RATIO = 0.4

# i8 grid is *intentionally* excluded: with ``scale_out ≈
# 1.2e-3`` the i8 output range is ±0.155, which cannot hold typical
# Divide quotients (|a/b| readily exceeds this when |b| ≈ 0.5). This is
# an output-grid resolution constraint, not a LUT-path bug — the i8
# Divide-via-LUT case is recorded as KNOWN_LIMIT in
# ``doc/precision_validation.md`` and best handled by routing 8-bit
# Divide through a different op (e.g. requant to i16, divide, requant
# back) rather than relaxing this floor.
_DIVIDE_GRIDS: tuple[QuantGridSpec, ...] = tuple(
    g for g in SIM_INT32_QUANT_GRIDS if g.signed and g.name != "i8"
)


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
    *,
    scale_out: float,
    grid: QuantGridSpec,
    zero_point: int | None = None,
) -> OutputEncoding:
    zp = grid.default_zero_point if zero_point is None else zero_point
    return OutputEncoding(
        scale=torch.tensor(scale_out, dtype=torch.float32),
        zero_point=torch.tensor(zp, dtype=torch.int32),
        qmin=grid.qmin,
        qmax=grid.qmax,
        # multiplier/rshift are derived inside DivideInt16Kernel from
        # scale_num * scale_recip / scale_out (see _divide_via_reciprocal_lut)
        multiplier=None,
        rshift=None,
    )


def _random_fp32_pair(
    grid: QuantGridSpec,
    gen: torch.Generator,
    *,
    same_scale: bool,
    numel: int = _TRIAL_NUMEL,
) -> tuple[torch.Tensor, torch.Tensor, float, float, float]:
    """Random fp32 (num, den, scale_num, scale_den, scale_out) pinned to LUT domain.

    Layout choices (all aimed at exposing the LUT-path *intrinsic* floor):

    - ``scale_den`` chosen so ``code_limit_den * scale_den`` reaches 90%
      of ``_LUT_IN_FMAX`` (keeps b inside fitted domain after rounding).
    - ``|q_b|`` drawn from ``[den_lo_code, code_limit_den]`` so the
      float magnitude lives in ``[_LUT_MARGIN, ~0.9 * _LUT_IN_FMAX]``.
    - ``scale_out = _LUT_OUT_SCALE`` (exact). This pins the output grid
      to the LUT output grid; smaller would inflate ``lsb_max`` (LUT-out
      LSB > output LSB), larger would crash cosine (output grid coarser
      than the actual quotient resolution).
    - ``code_limit_num`` is sized so the expected quotient ``a/b`` fits
      ~40% of ``qmax(output)`` after LUT-out scale.
    """

    code_limit_den = 1024
    s_den_target = (0.9 * _LUT_IN_FMAX) / code_limit_den
    den_lo_code = max(1, math.ceil(_LUT_MARGIN / s_den_target))

    if same_scale:
        scale_num = scale_den = s_den_target
    else:
        log_ratio = (
            torch.rand(1, generator=gen, dtype=torch.float32).item() - 0.5
        ) * 0.05
        scale_num = s_den_target * math.exp(log_ratio)
        scale_den = s_den_target

    scale_out = _OUTPUT_SCALE_RATIO * _LUT_OUT_SCALE

    # Numerator magnitude budget — multiplicative error propagation:
    # final |output_err| ≈ |a| × LUT_max_LSB_float. The abc default
    # 16-segment LUT has empirical worst-case error ~1 LUT-out LSB
    # (= LUT_OUT_SCALE = 3.05e-3) over its [-6, 6] fitted domain.
    # To stay under 1 output LSB we need:
    #   |a| × LUT_OUT_SCALE × 1 ≤ output_LSB = ratio × LUT_OUT_SCALE
    # i.e. |a| ≤ ratio = _OUTPUT_SCALE_RATIO. With ratio=0.4 the
    # budget is ``|a| ≤ 0.4`` (float). A 0.75 safety factor keeps cos
    # off the 0.9999 cliff while leaving ~1.5 output LSB headroom.
    target_a_max = _OUTPUT_SCALE_RATIO * 0.75
    code_limit_num = max(8, int(target_a_max / scale_num))
    code_limit_num = min(code_limit_num, code_limit_den)

    qa = torch.randint(
        -code_limit_num, code_limit_num + 1, (numel,),
        generator=gen, dtype=torch.int32,
    )
    qb_mag = torch.randint(
        den_lo_code, code_limit_den + 1, (numel,),
        generator=gen, dtype=torch.int32,
    )
    sign = torch.randint(0, 2, (numel,), generator=gen, dtype=torch.int32) * 2 - 1
    qb = qb_mag * sign

    a = qa.to(torch.float32) * scale_num
    b = qb.to(torch.float32) * scale_den

    return a, b, scale_num, scale_den, scale_out


def _assert_strict_fp32_divide_gates(
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
            f"{label}: cosine_similarity {cos:.8f} <= {_MIN_COSINE} "
            "(required >; LUT-domain inputs must meet the unified floor)."
        )
    float_lsb = max_error_lsb_float(
        ref, candidate,
        output.scale, output.zero_point, output.qmin, output.qmax,
    )
    if float_lsb >= _MAX_FLOAT_LSB:
        raise AssertionError(
            f"{label}: max_error_lsb_float {float_lsb:.6f} >= {_MAX_FLOAT_LSB} "
            "(required float error < 1 * scale_out)."
        )


def _run_random_trials(
    grid: QuantGridSpec, *, same_scale: bool, seed_base: int
) -> None:
    for trial in range(_NUM_RANDOM_TRIALS):
        gen = torch.Generator().manual_seed(
            seed_base + trial * 9973 + hash(grid.name) % 10000
        )
        a, b, sn, sd, so = _random_fp32_pair(grid, gen, same_scale=same_scale)
        out_enc = _output_encoding(scale_out=so, grid=grid)
        lhs = _quantize_float(a, scale=sn, grid=grid)
        rhs = _quantize_float(b, scale=sd, grid=grid)
        ref = a.to(torch.float32) / b.to(torch.float32)
        output = get_fixed_kernel(custom.Divide)(
            [lhs, rhs], {}, out_enc, {"eps": 1e-12},
        )
        mode = "same-scale" if same_scale else "cross-scale"
        _assert_strict_fp32_divide_gates(
            output, ref,
            label=f"Divide CLZ-LUT path {mode} {grid.name} trial={trial}",
        )


@pytest.mark.parametrize("grid", _DIVIDE_GRIDS, ids=lambda g: g.name)
def test_divide_clz_lut_path_same_scale_random_fp32_per_grid(
    grid: QuantGridSpec,
):
    _run_random_trials(grid, same_scale=True, seed_base=90_000)


@pytest.mark.parametrize("grid", _DIVIDE_GRIDS, ids=lambda g: g.name)
def test_divide_clz_lut_path_cross_scale_random_fp32_per_grid(
    grid: QuantGridSpec,
):
    _run_random_trials(grid, same_scale=False, seed_base=100_000)

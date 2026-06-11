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
"""custom.Divide *legacy integer-div path* precision vs ideal float32.

Sister harness to ``test_subtract_int16_precision.py`` /
``test_multiply_int16_precision.py``. **Same sampling volume and same
unified gates** (cosine > 0.9999, max_error_lsb_float < 1.0).

This file specifically exercises the **legacy integer-div fallback**
inside ``DivideInt16Kernel`` (selected via ``extra['force_legacy_
integer_divide']=True``). The fallback runs only when the spec §4.3.4
reciprocal CLZ LUT is unavailable in the workspace — but because we
want CI to *always* lock the fallback's precision character regardless
of LUT availability, the harness pins it via ``extra``.

Spec §4.3.4 reciprocal-LUT path precision is locked separately in
``test_divide_int16_clz_precision.py`` (which **does** meet the
unified 0.9999/1 floor under LUT-domain inputs).

KNOWN_LIMIT — the legacy integer-div path does not meet the unified
gates because:

* it is the pre-spec implementation (``num_scaled / den`` round-half
  pipeline, worst-case ≈1.5 LSB by construction),
* it is exercised across full random scales (no LUT-domain pinning),
  so ``scale_out`` swings can push ``real_m`` close to the
  ``quantize_multiplier`` rshift cliff.

Tests are marked ``xfail(strict=True)`` so the gap stays visible in CI
and any unexpected pass is flagged (e.g. someone tightens the
fallback). Thresholds stay aligned with Add/Sub/Mul rather than being
lowered just to make this file green.

Implementation specifics that drive the harness shape:

1. ``DivideInt16Kernel`` derives ``real_m = scale_num / (scale_den *
   scale_out)`` *internally* from the encodings — the test is therefore
   a pure end-to-end against the kernel's own multiplier/rshift folding
   (no need to feed multiplier/rshift in ``OutputEncoding``).
2. Divide is unbounded near zero. We **draw denominator codes from a
   strictly-non-zero band** (``[2*code_step, code_limit]`` plus optional
   sign flip on signed grids) so the eps clamp inside ``DivideInt16Kernel``
   never trips and the precision floor is determined by the
   reciprocal/CLZ path, not by eps clipping. Tests of the eps path live
   in ``test_divide_int16.py`` / ``test_divide_int16_eps_clamp.py``.
3. ``scale_out`` is chosen so the *expected quotient* mostly lives in
   the middle 25% of the output grid — Divide quotients can blow up
   when ``|den|`` shrinks, and we want to measure precision, not
   saturation. The relation we use is
   ``scale_out = (max_num / max_den) / (qmax_out * 0.25)``.

Operator-level KNOWN_LIMIT (recorded in ``doc/precision_validation.md``):
unsigned grids cannot represent negative quotients on ``zero_point=0``.
Mirrors the Subtract / Multiply skip-unsigned policy.
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
from aimet_torch.fixed_point.requantize import saturate_sim_tensor  # noqa: E402

# Unified single-op floor — same as Add/Sub/Mul. Divide is expected to
# *fail* these gates today; see the ``xfail`` decorators below and the
# Divide section in ``doc/precision_validation.md``. We deliberately do
# not relax the floor here — relaxing it would hide the gap and let
# future regressions slip in under the lower bar.
_MIN_COSINE = 0.9999
_MAX_FLOAT_LSB = 1.0

_NUM_RANDOM_TRIALS = 128
_TRIAL_NUMEL = 256
_CROSS_SCALE_LOG_RATIO_SPAN = 0.05
_CROSS_SCALE_LOG_RATIO_SPAN_I8 = 0.02

_DIVIDE_GRIDS: tuple[QuantGridSpec, ...] = tuple(
    g for g in SIM_INT32_QUANT_GRIDS if g.signed
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


def _divide_output_encoding(
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
        # multiplier/rshift left None on purpose: DivideInt16Kernel
        # derives them from the input/output scales.
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
    """Random fp32 (num, den, scale_num, scale_den, scale_out) for Divide.

    Numerator codes ∈ [-code_limit, code_limit]; denominator codes are
    drawn from a non-zero band ``[lo, code_limit]`` then sign-flipped
    so |den_code| ≥ ``2 * code_step`` (eps inside the kernel never
    fires). ``scale_out`` is set so the *expected* quotient sits at
    ~25% of the output grid range (headroom for cross-scale + eps).
    """

    base_num = math.exp(
        torch.rand(1, generator=gen, dtype=torch.float32).item() * 2.5 - 3.5
    )
    if same_scale:
        scale_num = scale_den = base_num
    else:
        span = (
            _CROSS_SCALE_LOG_RATIO_SPAN_I8
            if grid.name == "i8"
            else _CROSS_SCALE_LOG_RATIO_SPAN
        )
        log_ratio = (
            torch.rand(1, generator=gen, dtype=torch.float32).item() - 0.5
        ) * span
        scale_num = base_num * math.exp(log_ratio)
        scale_den = base_num

    if grid.name == "i8":
        code_limit_num = 24
    elif grid.name == "i16":
        code_limit_num = 64
    else:
        code_limit_num = 64

    den_lo = max(4, code_limit_num // 8)
    den_hi = code_limit_num

    qa = torch.randint(
        -code_limit_num,
        code_limit_num + 1,
        (numel,),
        generator=gen,
        dtype=torch.int32,
    )
    qb_mag = torch.randint(
        den_lo, den_hi + 1, (numel,), generator=gen, dtype=torch.int32
    )
    sign = torch.randint(0, 2, (numel,), generator=gen, dtype=torch.int32) * 2 - 1
    qb = qb_mag * sign

    a = qa.to(torch.float32) * scale_num
    b = qb.to(torch.float32) * scale_den

    # Expected quotient envelope; pick scale_out so |q_out| stays under
    # qmax_out * 0.5 with safety, leaving room for cross-scale + eps.
    # ``qmax_out_target`` is capped at INT16 range — even when the
    # output spec is i32, we clamp the *targeted* code magnitude to keep
    # ``real_m = scale_num / (scale_den * scale_out)`` below the upper
    # limit of ``quantize_multiplier`` (rshift becomes negative once
    # real_m > ~2^15). Real i32 deployments do not put dynamic-range-
    # heavy quotients on i32 outputs anyway — i16 is the canonical
    # carrier here.
    max_quot_abs = (code_limit_num * scale_num) / (den_lo * scale_den)
    qmax_out_target = min(grid.qmax, 32767) * 0.4
    scale_out = max_quot_abs / qmax_out_target

    return a, b, scale_num, scale_den, scale_out


def _fixed_divide_vs_ideal_float(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    scale_num: float,
    scale_den: float,
    out_enc: OutputEncoding,
    grid: QuantGridSpec,
) -> tuple[Int16QuantizedTensor, torch.Tensor]:
    lhs = _quantize_float(a, scale=scale_num, grid=grid)
    rhs = _quantize_float(b, scale=scale_den, grid=grid)
    ref = a.to(torch.float32) / b.to(torch.float32)
    output = get_fixed_kernel(custom.Divide)(
        [lhs, rhs],
        {},
        out_enc,
        # Force the *legacy* integer-div hot path even when the spec
        # §4.3.4 reciprocal CLZ LUT is reachable. The LUT path's
        # precision is locked separately in
        # ``test_divide_int16_clz_precision.py``; this file's job is to
        # keep the legacy fallback honest under random-scale inputs.
        {"eps": 1e-12, "force_legacy_integer_divide": True},
    )
    return output, ref


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


def _run_random_trials(
    grid: QuantGridSpec, *, same_scale: bool, seed_base: int
) -> None:
    for trial in range(_NUM_RANDOM_TRIALS):
        gen = torch.Generator().manual_seed(
            seed_base + trial * 9973 + hash(grid.name) % 10000
        )
        a, b, sn, sd, so = _random_fp32_pair(grid, gen, same_scale=same_scale)
        out_enc = _divide_output_encoding(scale_out=so, grid=grid)
        output, ref = _fixed_divide_vs_ideal_float(
            a, b, scale_num=sn, scale_den=sd, out_enc=out_enc, grid=grid
        )
        mode = "same-scale" if same_scale else "cross-scale"
        _assert_strict_fp32_divide_gates(
            output, ref, label=f"Divide path B {mode} {grid.name} trial={trial}",
        )


# --- xfail rationale (kept tightly inline so it shows in test output) ---
#
# Divide's hot path runs ``(num_centered * mult) >> rshift``,  followed
# by an integer ``num_scaled / den_centered`` (round-half), and a final
# requant. The accumulated round-half-LSB worst case is ≈1.5 LSB:
#   ~0.5 LSB  multiplier folding (quantize_multiplier of real_m)
#   ~0.5 LSB  integer division remainder rounding
#   ~0.5 LSB  cross-scale multiplier folding (when scale_a != scale_b)
# This budget cannot be brought below 1.0 LSB without a kernel rewrite
# to the spec-mandated reciprocal-LUT + Newton-iteration pipeline
# (doc/04_算子详细规格/04_03_逐元素运算类算子.md §4.3.4).
#
# ``strict=True`` so any *unexpected pass* (kernel tightened, or HW
# reciprocal path landed) flips this back to red; the doc and the
# follow-up should then be updated explicitly rather than by silent
# strict-pass.
_XFAIL_REASON = (
    "KNOWN_LIMIT: legacy integer-div path has ~1.5 LSB worst-case error; "
    "cannot meet unified Add/Sub/Mul floor (0.9999 cos / 1 LSB) under "
    "random-scale inputs. The spec §4.3.4 reciprocal-LUT path *does* "
    "meet the floor (locked in test_divide_int16_clz_precision.py) "
    "but is restricted to LUT-domain inputs in the abc default asset. "
    "Tracked in doc/precision_validation.md under custom.Divide."
)


@pytest.mark.xfail(strict=True, reason=_XFAIL_REASON)
@pytest.mark.parametrize("grid", _DIVIDE_GRIDS, ids=lambda g: g.name)
def test_divide_path_b_same_scale_random_fp32_per_grid(grid: QuantGridSpec):
    _run_random_trials(grid, same_scale=True, seed_base=70_000)


@pytest.mark.xfail(strict=True, reason=_XFAIL_REASON)
@pytest.mark.parametrize("grid", _DIVIDE_GRIDS, ids=lambda g: g.name)
def test_divide_path_b_cross_scale_random_fp32_per_grid(grid: QuantGridSpec):
    _run_random_trials(grid, same_scale=False, seed_base=80_000)

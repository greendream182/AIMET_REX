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
"""custom.Multiply single-op precision vs ideal float32 reference.

Sister harness to ``test_subtract_int16_precision.py`` — same sampling
volume (128 trials × 256 elements per ``(grid, mode)``), same gates
(cosine > 0.9999, max_error_lsb_float < 1), same Path-B style. The
implementation difference is in the kernel hot path: Multiply runs
``_center_tensor(a) * _center_tensor(b)`` through ``int32_mul_sat``
and then ``_requantize`` — there is no per-input grid alignment
(``_BinaryAlignedKernel``) because the output grid absorbs both input
scales via ``real_m = scale_a * scale_b / scale_out`` (see
``aimet_torch/v2/quantization/affine/fixed_point/adapter.py``).

Output grid is set to ``scale_out = scale_a * scale_b`` for the
"natural" multiplication exit (``real_m ≈ 1`` so multiplier=32767,
rshift=15 stays a faithful approximation); cross-scale perturbs that
ratio by ±5% (i8: ±2%) — the same span as Add/Subtract — so the gate
exercises both bit-aligned and small-multiplier-rounding paths.

Operator-level KNOWN_LIMIT (recorded in ``doc/precision_validation.md``):
unsigned grids with ``zero_point=0`` cannot represent negative products,
which arise whenever exactly one of (a, b) is negative. Add Path-B
sidesteps this because it forces non-negative codes; Subtract sidesteps
it by skipping unsigned grids; Multiply takes the same skip-unsigned
route. Mixed-sign products on signed grids are the case we explicitly
*want* exercised (covered by ``_random_fp32_pair`` drawing from
``[-code_limit, code_limit]``).
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
from aimet_torch.fixed_point.offline.multiplier import (  # noqa: E402
    quantize_multiplier,
)
from aimet_torch.fixed_point.quant_grid import (  # noqa: E402
    QuantGridSpec,
    SIM_INT32_QUANT_GRIDS,
)
from aimet_torch.fixed_point.requantize import saturate_sim_tensor  # noqa: E402

_MIN_COSINE = 0.9999
_MAX_FLOAT_LSB = 1.0

_NUM_RANDOM_TRIALS = 128
_TRIAL_NUMEL = 256
_CROSS_SCALE_LOG_RATIO_SPAN = 0.05
_CROSS_SCALE_LOG_RATIO_SPAN_I8 = 0.02

# Mirrors ``test_subtract_int16_precision.py``: unsigned grids cannot
# represent negative products on zp=0, see module docstring.
_MULTIPLY_GRIDS: tuple[QuantGridSpec, ...] = tuple(
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


def _multiply_output_encoding(
    *,
    scale_a: float,
    scale_b: float,
    scale_out: float,
    grid: QuantGridSpec,
    zero_point: int | None = None,
) -> OutputEncoding:
    """Output encoding with ``multiplier``/``rshift`` derived from ``real_m``.

    For Multiply, ``real_m = scale_a * scale_b / scale_out`` per the
    adapter contract. ``quantize_multiplier`` returns a (uint16, int8)
    pair the kernel consumes through ``_requantize``. This is a more
    faithful test than the fixed (32767, 15) used in Add/Subtract Path-B
    because Multiply genuinely depends on ``real_m`` precision.
    """

    real_m = torch.tensor(scale_a * scale_b / scale_out, dtype=torch.float64)
    multiplier, rshift = quantize_multiplier(real_m)
    zp = grid.default_zero_point if zero_point is None else zero_point
    return OutputEncoding(
        scale=torch.tensor(scale_out, dtype=torch.float32),
        zero_point=torch.tensor(zp, dtype=torch.int32),
        qmin=grid.qmin,
        qmax=grid.qmax,
        multiplier=multiplier,
        rshift=rshift,
    )


def _random_fp32_pair(
    grid: QuantGridSpec,
    gen: torch.Generator,
    *,
    same_scale: bool,
    numel: int = _TRIAL_NUMEL,
) -> tuple[torch.Tensor, torch.Tensor, float, float, float]:
    """Random fp32 (a, b, scale_a, scale_b, scale_out) for Multiply.

    ``scale_out = scale_a * scale_b`` so the natural product lives at
    real_m ≈ 1; cross-scale perturbs by the same ±5% span Add/Subtract
    use. Codes are drawn from ``[-code_limit, code_limit]`` per signed
    grid so mixed-sign products are exercised; ``code_limit`` is kept
    well below ``qmax`` so the product after centering rarely saturates
    INT32 (which would mask precision misses behind unrelated clamps).
    """

    base = math.exp(
        torch.rand(1, generator=gen, dtype=torch.float32).item() * 2.5 - 3.5
    )
    if same_scale:
        scale_a = scale_b = base
    else:
        span = (
            _CROSS_SCALE_LOG_RATIO_SPAN_I8
            if grid.name == "i8"
            else _CROSS_SCALE_LOG_RATIO_SPAN
        )
        log_ratio = (
            torch.rand(1, generator=gen, dtype=torch.float32).item() - 0.5
        ) * span
        scale_a = base * math.exp(log_ratio)
        scale_b = base
    scale_out = scale_a * scale_b

    # Multiply differs from Add/Subtract: the *product* of the input codes
    # must fit the output grid without saturation. With ``scale_out =
    # scale_a * scale_b`` we get ``q_out ≈ q_a * q_b``, so we cap
    # ``code_limit ≤ floor(sqrt(qmax/2))`` (the ``/2`` leaves headroom for
    # the cross-scale ratio + half-LSB rounding on requantize). Add/
    # Subtract get away with a much larger code_limit because their
    # output magnitude is bounded by ``q_a + q_b`` rather than the
    # product.
    if grid.name == "i8":
        code_limit = 8       # 8*8 = 64 < 127
    elif grid.name == "i16":
        code_limit = 64      # 64*64 = 4096 < 32767
    else:
        code_limit = min(64, (grid.qmax - grid.qmin) // 8)
    qa = torch.randint(
        -code_limit, code_limit + 1, (numel,), generator=gen, dtype=torch.int32
    )
    qb = torch.randint(
        -code_limit, code_limit + 1, (numel,), generator=gen, dtype=torch.int32
    )
    a = qa.to(torch.float32) * scale_a
    b = qb.to(torch.float32) * scale_b
    return a, b, scale_a, scale_b, scale_out


def _fixed_multiply_vs_ideal_float(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    scale_a: float,
    scale_b: float,
    out_enc: OutputEncoding,
    grid: QuantGridSpec,
) -> tuple[Int16QuantizedTensor, torch.Tensor]:
    lhs = _quantize_float(a, scale=scale_a, grid=grid)
    rhs = _quantize_float(b, scale=scale_b, grid=grid)
    ref = a.to(torch.float32) * b.to(torch.float32)
    output = get_fixed_kernel(custom.Multiply)([lhs, rhs], {}, out_enc, {})
    return output, ref


def _assert_strict_fp32_multiply_gates(
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
        a, b, scale_a, scale_b, scale_out = _random_fp32_pair(
            grid, gen, same_scale=same_scale
        )
        out_enc = _multiply_output_encoding(
            scale_a=scale_a, scale_b=scale_b, scale_out=scale_out, grid=grid
        )
        output, ref = _fixed_multiply_vs_ideal_float(
            a, b, scale_a=scale_a, scale_b=scale_b, out_enc=out_enc, grid=grid
        )
        mode = "same-scale" if same_scale else "cross-scale"
        _assert_strict_fp32_multiply_gates(
            output, ref, label=f"Multiply path B {mode} {grid.name} trial={trial}",
        )


@pytest.mark.parametrize("grid", _MULTIPLY_GRIDS, ids=lambda g: g.name)
def test_multiply_path_b_same_scale_random_fp32_per_grid(grid: QuantGridSpec):
    _run_random_trials(grid, same_scale=True, seed_base=50_000)


@pytest.mark.parametrize("grid", _MULTIPLY_GRIDS, ids=lambda g: g.name)
def test_multiply_path_b_cross_scale_random_fp32_per_grid(grid: QuantGridSpec):
    _run_random_trials(grid, same_scale=False, seed_base=60_000)

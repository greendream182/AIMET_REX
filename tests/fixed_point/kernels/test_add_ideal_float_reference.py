"""Path B: single-op Add vs ideal float32 sum (``ref = a + b``).

All tensors are **float32**. Inputs are pseudo-random: integer codes drawn on the
input grid, then ``a = (Q - zp) * scale`` (still float32). Scales are random;
cross-scale uses a mild ratio (±5%) so requant error stays small.

Unified gates (output quantizer):

- ``cosine_similarity > 0.9999``
- ``max_error_lsb_float < 1``  (i.e. ``max_abs_error < 1 * scale_out``)

Runs on :data:`~aimet_torch.fixed_point.quant_grid.SIM_INT32_QUANT_GRIDS`.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

import aimet_torch.fixed_point.kernels  # noqa: F401

from aimet_torch._base.nn.modules import custom
from aimet_torch.fixed_point import Int16QuantizedTensor, OutputEncoding, get_fixed_kernel
from aimet_torch.fixed_point.metrics.accuracy import cosine_similarity, max_error_lsb_float
from aimet_torch.fixed_point.metrics.flags import int16_eval_allow_debug_float
from aimet_torch.fixed_point.quant_grid import GRID_U32, QuantGridSpec, SIM_INT32_QUANT_GRIDS
from aimet_torch.fixed_point.requantize import saturate_sim_tensor

# Product gates (strict inequalities).
_MIN_COSINE = 0.9999
_MAX_FLOAT_LSB = 1.0

_NUM_RANDOM_TRIALS = 32
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


def _fixed_add_vs_ideal_float(
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
    ref = a.to(torch.float32) + b.to(torch.float32)
    output = get_fixed_kernel(custom.Add)([lhs, rhs], {}, out_enc, {})
    return output, ref


def _random_fp32_pair(
    grid: QuantGridSpec,
    gen: torch.Generator,
    *,
    same_scale: bool,
    numel: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """Random float32 activations representable on ``grid`` before Add."""

    scale_out = math.exp(torch.rand(1, generator=gen, dtype=torch.float32).item() * 2.5 - 3.5)
    if same_scale:
        scale_a = scale_b = scale_out
    else:
        span = (
            _CROSS_SCALE_LOG_RATIO_SPAN_I8
            if grid.name == "i8"
            else _CROSS_SCALE_LOG_RATIO_SPAN
        )
        log_ratio = (torch.rand(1, generator=gen, dtype=torch.float32).item() - 0.5) * span
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


def _assert_strict_fp32_add_gates(
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
        gen = torch.Generator().manual_seed(seed_base + trial * 9973 + hash(grid.name) % 10000)
        a, b, scale_a, scale_b = _random_fp32_pair(grid, gen, same_scale=same_scale)
        out_scale = scale_b
        out_enc = _output_encoding(scale=out_scale, grid=grid)
        output, ref = _fixed_add_vs_ideal_float(
            a, b, scale_a=scale_a, scale_b=scale_b, out_enc=out_enc, grid=grid
        )
        mode = "same-scale" if same_scale else "cross-scale"
        _assert_strict_fp32_add_gates(
            output,
            ref,
            label=f"Add path B {mode} {grid.name} trial={trial}",
        )


@pytest.mark.parametrize("grid", SIM_INT32_QUANT_GRIDS, ids=lambda g: g.name)
def test_add_path_b_same_scale_random_fp32_per_grid(grid: QuantGridSpec):
    _run_random_trials(grid, same_scale=True, seed_base=10_000)


@pytest.mark.parametrize("grid", SIM_INT32_QUANT_GRIDS, ids=lambda g: g.name)
def test_add_path_b_cross_scale_random_fp32_per_grid(grid: QuantGridSpec):
    _run_random_trials(grid, same_scale=False, seed_base=20_000)


@pytest.mark.skipif(
    not GRID_U32.fits_sim_int32_container,
    reason="full u32 grid needs int64 sim-tensor container (ADR-013)",
)
def test_add_path_b_u32_pending_int64_container():
    pytest.fail("u32 grid should be skipped until int64 container lands")

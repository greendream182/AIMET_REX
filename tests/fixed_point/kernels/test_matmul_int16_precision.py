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
"""``custom.MatMul`` precision vs ideal float32 reference.

Sister harness to ``test_linear_int16_precision.py``. ``custom.MatMul`` is
the user-facing op for plain ``A @ B`` (e.g. BandConverter band matrices)
and shares the kernel hot path with Linear/Conv: both center inputs,
call ``_int32_matmul`` on the centered representation, then drive
``_requantize_output`` with a single (M, rshift) folded from
``scale_a * scale_b / scale_out``.

The unified gates (cosine > 0.9999, max_error_lsb_float < 1.0) apply
without relaxation. Coverage:

- Grids: signed i8 / i16 (matches Linear; i32 not registered as an output
  grid for matmul-class kernels per spec 04_02).
- Modes: same-scale (scale_a = scale_b = scale_out target) and
  mild cross-scale (5% multiplicative ratio on the output scale).
- Shape: ``(B=4, M=16, K=8) @ (B=4, K=8, N=16)`` → ``(4, 16, 16)``
  = 1024 outputs/trial, large enough to give matmul-class SNR head-room
  on i8 (compare with the 64-output Linear case in P4, which sits on
  the edge).

Spec ``doc/04_算子详细规格/04_02_矩阵运算类算子.md`` §MatMul.
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

_BATCH = 4
_M = 16
_K = 8
_N = 16

_MATMUL_GRIDS: tuple[QuantGridSpec, ...] = tuple(
    g for g in SIM_INT32_QUANT_GRIDS if g.signed and g.name in {"i8", "i16"}
)
_GRID_PARAMS = tuple(pytest.param(g, id=g.name) for g in _MATMUL_GRIDS)


def _quantize(
    tensor: torch.Tensor,
    *,
    scale: float,
    grid: QuantGridSpec,
) -> Int16QuantizedTensor:
    scale_t = torch.tensor(scale, dtype=torch.float32)
    zp_t = torch.tensor(grid.default_zero_point, dtype=torch.int32)
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
    scale_a: float,
    scale_b: float,
    scale_out: float,
    grid: QuantGridSpec,
) -> OutputEncoding:
    real_m = (scale_a * scale_b) / scale_out
    multiplier, rshift = quantize_multiplier(
        torch.tensor(real_m, dtype=torch.float64)
    )
    return OutputEncoding(
        scale=torch.tensor(scale_out, dtype=torch.float32),
        zero_point=torch.tensor(grid.default_zero_point, dtype=torch.int32),
        qmin=grid.qmin,
        qmax=grid.qmax,
        multiplier=multiplier,
        rshift=rshift,
    )


def _random_matmul_inputs(
    grid: QuantGridSpec,
    gen: torch.Generator,
    *,
    same_scale: bool,
) -> tuple[torch.Tensor, torch.Tensor, float, float, float]:
    """Random fp32 matmul inputs sized to ``(B,M,K) @ (B,K,N)``.

    Returns ``(a, b, scale_a, scale_b, scale_out)``. ``scale_out`` is
    derived from the actual reference's abs-max so the i8 grid is fully
    used without saturating (mirrors P4 Linear's calibration approach).
    """
    base = math.exp(
        torch.rand(1, generator=gen, dtype=torch.float32).item() * 2.0 - 4.0
    )
    scale_a = base
    scale_b = base

    code_limit = 16 if grid.name == "i8" else 1024
    qa = torch.randint(
        -code_limit, code_limit + 1, (_BATCH, _M, _K),
        generator=gen, dtype=torch.int32,
    )
    qb = torch.randint(
        -code_limit, code_limit + 1, (_BATCH, _K, _N),
        generator=gen, dtype=torch.int32,
    )
    a = qa.to(torch.float32) * scale_a
    b = qb.to(torch.float32) * scale_b

    ref = torch.matmul(a, b)
    ref_abs_max = float(ref.abs().max().item())
    if ref_abs_max == 0.0:
        ref_abs_max = 1.0
    scale_out_base = ref_abs_max / (grid.qmax * 0.85)

    if same_scale:
        scale_out = scale_out_base
    else:
        log_ratio = (
            torch.rand(1, generator=gen, dtype=torch.float32).item() - 0.5
        ) * _CROSS_SCALE_LOG_RATIO_SPAN
        scale_out = scale_out_base * math.exp(log_ratio)

    return a, b, scale_a, scale_b, scale_out


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
        ref.flatten(),
        candidate.flatten(),
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


def _run_matmul(
    grid: QuantGridSpec,
    *,
    same_scale: bool,
    seed_base: int,
) -> None:
    for trial in range(_NUM_RANDOM_TRIALS):
        gen = torch.Generator().manual_seed(
            seed_base + trial * 9973 + hash(grid.name) % 10000
        )
        a, b, scale_a, scale_b, scale_out = _random_matmul_inputs(
            grid, gen, same_scale=same_scale,
        )
        a_q = _quantize(a, scale=scale_a, grid=grid)
        b_q = _quantize(b, scale=scale_b, grid=grid)
        out_enc = _output_encoding(
            scale_a=scale_a, scale_b=scale_b, scale_out=scale_out, grid=grid,
        )
        ref = torch.matmul(a.to(torch.float32), b.to(torch.float32))
        output = get_fixed_kernel(custom.MatMul)(
            [a_q, b_q], {}, out_enc, {},
        )
        mode = "same-scale" if same_scale else "cross-scale"
        _assert_strict_gates(
            output, ref, label=f"MatMul {mode} {grid.name} trial={trial}",
        )


@pytest.mark.parametrize("grid", _GRID_PARAMS)
def test_matmul_same_scale_random_fp32_per_grid(grid: QuantGridSpec):
    _run_matmul(grid, same_scale=True, seed_base=910_000)


@pytest.mark.parametrize("grid", _GRID_PARAMS)
def test_matmul_cross_scale_random_fp32_per_grid(grid: QuantGridSpec):
    _run_matmul(grid, same_scale=False, seed_base=920_000)

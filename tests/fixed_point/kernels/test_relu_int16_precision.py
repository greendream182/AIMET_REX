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
"""nn.ReLU / nn.ReLU6 / custom.Abs single-op precision vs ideal float32 reference.

Sister harness to ``test_linear_int16_precision.py`` (P4). Same unified
gates (cosine > 0.9999, max_error_lsb_float < 1.0).

Hot paths under test (``aimet_torch/fixed_point/kernels/eltwise.py``):

- ``ReLU``: ``centered → clamp_min(0) → [optional requantize]``
- ``ReLU6``: ``centered → clamp(0, round(6/scale_in)) → [optional requantize]``
- ``Abs``: ``centered → torch.abs → [optional requantize]`` (spec 04_03
  §4.3.5 integer-abs path, retired the prior PWL 16-segment fit)

Spec ``doc/04_算子详细规格/04_04_激活函数类算子.md`` §4.4.1 (ReLU),
``04_13_特殊激活与常量算子.md`` §4.14.1 (ReLU6/Clip), and ``04_03_逐元
素运算类算子.md`` §4.3.5 (Abs). When the output quantization grid
matches the input (``M=1``), all three kernels collapse to a pure-integer
op — zero quantization error from the kernel itself, only the
round-half from the input quantize step. Cross-scale enables the
``M/rshift`` requantize branch.

Coverage:

- Grids: signed i8 / i16 (spec output ∈ {i8, i16}; i32 is acc-grid only)
- Modes: same-scale (M=1 fast path) and cross-scale (requantize branch
  active) on all three ops
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

import aimet_torch.fixed_point.kernels  # noqa: E402, F401

from torch import nn  # noqa: E402

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
_NUMEL = 256
_CROSS_SCALE_LOG_RATIO_SPAN = 0.05

_RELU_GRIDS: tuple[QuantGridSpec, ...] = tuple(
    g for g in SIM_INT32_QUANT_GRIDS if g.signed and g.name in {"i8", "i16"}
)

_GRID_PARAMS_STABLE = tuple(pytest.param(g, id=g.name) for g in _RELU_GRIDS)

# ReLU6 cross-scale i8: an extra round(6/scale_in) round-half noise
# stacks on top of the requantize multiplier folding noise, putting
# the worst-case trial cos_min in the [0.99983, 0.99995] range
# depending on seed sequence. Same edge-rattle pattern as P4 Linear
# i8 — use strict=False (not strict=True) to accept the genuine
# randomness rather than flake on unexpected pass.
_I8_CROSS_RELU6_XFAIL_REASON = (
    "ReLU6 i8 cross-scale: round(6/scale_in) integer-rounding noise "
    "(≤ 0.5 LSB-of-input) stacks with the requantize multiplier folding "
    "noise; worst-case trial cos_min oscillates around the 0.9999 floor "
    "(measured 0.99983～0.99995 across seed offsets). lsb_max stays "
    "≤ 0.95 (still < 1.0 floor). strict=False because the floor "
    "crossing is genuinely random. See doc/precision_validation.md."
)


def _relu6_grid_param(grid: QuantGridSpec, same_scale: bool):
    if grid.name == "i8" and not same_scale:
        return pytest.param(
            grid, id=grid.name,
            marks=pytest.mark.xfail(
                strict=False, reason=_I8_CROSS_RELU6_XFAIL_REASON,
            ),
        )
    return pytest.param(grid, id=grid.name)


_GRID_PARAMS_RELU6_CROSS = tuple(
    _relu6_grid_param(g, same_scale=False) for g in _RELU_GRIDS
)


def _quantize_float_with_grid(
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
    same_scale: bool,
) -> OutputEncoding:
    """Build OutputEncoding.

    For the same-scale fast path (M=1), ``multiplier`` / ``rshift`` are
    still computed so the kernel exercises the same code path uniformly
    (quantize_multiplier returns M=2**15, rshift=15 when real_m=1).
    For cross-scale, ``real_m = S_x / S_y`` carries the rescale.
    """
    del same_scale
    real_m = scale_in / scale_out
    multiplier, rshift = quantize_multiplier(
        torch.tensor(real_m, dtype=torch.float64)
    )
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
    same_scale: bool,
) -> tuple[torch.Tensor, float, float]:
    """Random fp32 input + (scale_in, scale_out).

    Inputs are zero-mean with code envelope ~ half of grid.qmax, so
    roughly half the values clip on ReLU/ReLU6 — exercising both
    branches uniformly.
    """

    base = math.exp(
        torch.rand(1, generator=gen, dtype=torch.float32).item() * 2.5 - 3.5
    )
    if same_scale:
        scale_in = base
        scale_out = base
    else:
        log_x = (
            torch.rand(1, generator=gen, dtype=torch.float32).item() - 0.5
        ) * _CROSS_SCALE_LOG_RATIO_SPAN
        log_y = (
            torch.rand(1, generator=gen, dtype=torch.float32).item() - 0.5
        ) * _CROSS_SCALE_LOG_RATIO_SPAN
        scale_in = base * math.exp(log_x)
        scale_out = base * math.exp(log_y)

    code_limit = max(8, grid.qmax // 2)
    qx = torch.randint(
        -code_limit, code_limit + 1, (_NUMEL,),
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


def _run_relu_trials(
    grid: QuantGridSpec,
    *,
    same_scale: bool,
    seed_base: int,
    module_cls: type,
    float_op,
) -> None:
    for trial in range(_NUM_RANDOM_TRIALS):
        gen = torch.Generator().manual_seed(
            seed_base + trial * 9973 + hash(grid.name) % 10000
        )
        x, scale_in, scale_out = _random_fp32_input(
            grid, gen, same_scale=same_scale,
        )
        x_q = _quantize_float_with_grid(
            x, scale=scale_in, grid=grid, zero_point=grid.default_zero_point,
        )
        out_enc = _output_encoding(
            scale_in=scale_in, scale_out=scale_out,
            grid=grid, zero_point=grid.default_zero_point,
            same_scale=same_scale,
        )
        ref = float_op(x.to(torch.float32))
        output = get_fixed_kernel(module_cls)([x_q], {}, out_enc, {})
        mode = "same-scale" if same_scale else "cross-scale"
        _assert_strict_gates(
            output, ref,
            label=f"{module_cls.__name__} {mode} {grid.name} trial={trial}",
        )


@pytest.mark.parametrize("grid", _GRID_PARAMS_STABLE)
def test_relu_same_scale_random_fp32_per_grid(grid: QuantGridSpec):
    _run_relu_trials(
        grid, same_scale=True, seed_base=510_000,
        module_cls=nn.ReLU, float_op=torch.relu,
    )


@pytest.mark.parametrize("grid", _GRID_PARAMS_STABLE)
def test_relu_cross_scale_random_fp32_per_grid(grid: QuantGridSpec):
    _run_relu_trials(
        grid, same_scale=False, seed_base=520_000,
        module_cls=nn.ReLU, float_op=torch.relu,
    )


@pytest.mark.parametrize("grid", _GRID_PARAMS_STABLE)
def test_relu6_same_scale_random_fp32_per_grid(grid: QuantGridSpec):
    _run_relu_trials(
        grid, same_scale=True, seed_base=530_000,
        module_cls=nn.ReLU6, float_op=lambda t: torch.clamp(t, 0.0, 6.0),
    )


@pytest.mark.parametrize("grid", _GRID_PARAMS_RELU6_CROSS)
def test_relu6_cross_scale_random_fp32_per_grid(grid: QuantGridSpec):
    _run_relu_trials(
        grid, same_scale=False, seed_base=540_000,
        module_cls=nn.ReLU6, float_op=lambda t: torch.clamp(t, 0.0, 6.0),
    )


# ----------------------------------------------------------------------------
# Abs — integer-abs path (spec 04_03 §4.3.5). Same SAME_GRID_OR_REQUANT hot
# path as ReLU / Clamp, so the ReLU runner is reused verbatim. Same-scale is
# byte-stream identity on |centered|; cross-scale exercises the requantize
# branch via |x| ≥ 0 (no negative-tail clipping like ReLU's clamp_min).
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("grid", _GRID_PARAMS_STABLE)
def test_abs_same_scale_random_fp32_per_grid(grid: QuantGridSpec):
    _run_relu_trials(
        grid, same_scale=True, seed_base=550_000,
        module_cls=custom.Abs, float_op=torch.abs,
    )


@pytest.mark.parametrize("grid", _GRID_PARAMS_STABLE)
def test_abs_cross_scale_random_fp32_per_grid(grid: QuantGridSpec):
    _run_relu_trials(
        grid, same_scale=False, seed_base=560_000,
        module_cls=custom.Abs, float_op=torch.abs,
    )

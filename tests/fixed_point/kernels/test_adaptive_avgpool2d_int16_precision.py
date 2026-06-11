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
"""custom.AdaptiveAvgPool2d single-op precision vs ideal float32 reference.

``AdaptiveAvgPool2dInt16Kernel`` extends ``_MeanInt16Kernel`` (see
``aimet_torch/fixed_point/kernels/pool.py``) — i.e. it shares the same
``int32_sum_sat → saturate → requantize_int`` hot path as Mean and
AvgPool2d. The only currently-dispatched configuration is
``output_size=(1, 1)`` (the kernel calls
``require_adaptive_avgpool_output_unit``). Adapter populates
``extra={"dim": (2, 3), "keepdim": True, "output_size": (1, 1),
"reduce_size": H*W}`` and the reduction folds ``1/(H*W)`` into
``M/rshift``.

Same unified gates as the rest of the precision suite: cosine > 0.9999,
max_error_lsb_float < 1.0. Same KNOWN_LIMITs (i8 + reduction kernels:
SNR ceiling — see ``test_avgpool2d_int16_precision.py`` module docstring
for the derivation).

Per the precision-validation policy (mirror Divide legacy-path), i8
grids are **kept in parametrize and marked ``pytest.mark.xfail(strict=
True)``** so the actual snapshot is recorded in
``doc/precision_validation.md#customadaptiveavgpool2d``.
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

# (H, W) over which to spatial-mean. Each forms a different N=H*W.
_AAP_SPATIAL: tuple[tuple[int, int], ...] = ((4, 4), (8, 8), (4, 8))
_AAP_BATCH = 1
_AAP_CHANNELS = 4

_AAP_GRIDS: tuple[QuantGridSpec, ...] = tuple(
    g for g in SIM_INT32_QUANT_GRIDS if g.signed
)

_I8_XFAIL_REASON = (
    "AdaptiveAvgPool2d on i8: reduction-class SNR ceiling — output "
    "stddev ∝ code_limit/sqrt(3·H·W); i8 (qmax=127) cannot reach "
    "SNR > 100 for H·W ≥ 4. lsb_max stays ≈ 0.5 (kernel correct), only "
    "cosine fails. See doc/precision_validation.md#customadaptiveavgpool2d."
)


def _grid_param(grid: QuantGridSpec):
    if grid.name == "i8":
        return pytest.param(
            grid,
            id=grid.name,
            marks=pytest.mark.xfail(strict=True, reason=_I8_XFAIL_REASON),
        )
    return pytest.param(grid, id=grid.name)


_AAP_GRID_PARAMS = tuple(_grid_param(g) for g in _AAP_GRIDS)


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

    if grid.name == "i8":
        code_limit = 96
    else:
        code_limit = 4096

    zp = grid.default_zero_point
    qx = torch.randint(
        -code_limit, code_limit + 1, shape,
        generator=gen, dtype=torch.int32,
    )
    x = (qx.to(torch.float32) - float(zp)) * scale_in
    return x, scale_in, scale_out


def _assert_strict_fp32_aap_gates(
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
    spatial: tuple[int, int],
    *,
    same_scale: bool,
    seed_base: int,
) -> None:
    h, w = spatial
    shape = (_AAP_BATCH, _AAP_CHANNELS, h, w)
    reduce_size = h * w
    for trial in range(_NUM_RANDOM_TRIALS):
        gen = torch.Generator().manual_seed(
            seed_base + trial * 9973 + hash((grid.name, h, w)) % 10000
        )
        x, scale_in, scale_out = _random_fp32_input(
            grid, gen, same_scale=same_scale, shape=shape,
        )
        x_q = _quantize_float(x, scale=scale_in, grid=grid)
        out_enc = _output_encoding_with_fold(
            scale_in=scale_in,
            scale_out=scale_out,
            reduce_size=reduce_size,
            grid=grid,
        )
        ref = torch.mean(x.to(torch.float32), dim=(2, 3), keepdim=True)
        output = get_fixed_kernel(custom.AdaptiveAvgPool2d)(
            [x_q],
            {},
            out_enc,
            {
                "dim": (2, 3),
                "keepdim": True,
                "output_size": (1, 1),
                "reduce_size": reduce_size,
            },
        )
        mode = "same-scale" if same_scale else "cross-scale"
        _assert_strict_fp32_aap_gates(
            output, ref,
            label=f"AdaptiveAvgPool2d {mode} {grid.name} HW={h}x{w} trial={trial}",
        )


@pytest.mark.parametrize("spatial", _AAP_SPATIAL, ids=lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize("grid", _AAP_GRID_PARAMS)
def test_adaptive_avgpool2d_same_scale_random_fp32_per_grid(
    grid: QuantGridSpec, spatial: tuple[int, int],
):
    _run_random_trials(grid, spatial, same_scale=True, seed_base=150_000)


@pytest.mark.parametrize("spatial", _AAP_SPATIAL, ids=lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize("grid", _AAP_GRID_PARAMS)
def test_adaptive_avgpool2d_cross_scale_random_fp32_per_grid(
    grid: QuantGridSpec, spatial: tuple[int, int],
):
    _run_random_trials(grid, spatial, same_scale=False, seed_base=160_000)

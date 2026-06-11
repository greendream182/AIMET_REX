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
"""``nn.MaxPool2d`` / ``custom.MaxPool2d`` precision vs ideal float32 max-pool.

MaxPool is :class:`KernelKind.SAME_GRID_VALUE` — kernel runs
``F.max_pool2d`` directly on the int_repr and rewraps with the input's
encoding; the spec contract (04_09) is "input and output share scale /
zp / qmin / qmax (comparator-only HW, no M / rshift)". Float reference is
therefore identical in value space, so we expect **lsb_max = 0.0** for
every well-formed case (the kernel selects existing codes; it never
synthesises a new code).

Coverage matches spec 04_09 §pool operand limits:

- Grids: signed i8 / i16 (the contract is grid-agnostic; we exercise
  the two grids tracked in P-stage precision_validation.md).
- kernel ∈ {2×2, 3×3} (spec allows kt,kf ≤ 3).
- stride / padding mirror common deploy patterns.
- ``custom.MaxPool2d`` shares the kernel with ``nn.MaxPool2d`` so we
  cover both via a single parametrize entry.

Sister harness: ``test_shape_ops_int16_precision.py`` (P6 byte-stream
ops), ``test_avgpool2d_int16_precision.py`` (P3 reduce-class pool).
"""

from __future__ import annotations

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
from aimet_torch.fixed_point.quant_grid import (  # noqa: E402
    QuantGridSpec,
    SIM_INT32_QUANT_GRIDS,
)
from aimet_torch.fixed_point.requantize import saturate_sim_tensor  # noqa: E402


_MIN_COSINE = 0.9999
_MAX_FLOAT_LSB = 1.0

_NUM_RANDOM_TRIALS = 32

_GRIDS: tuple[QuantGridSpec, ...] = tuple(
    g for g in SIM_INT32_QUANT_GRIDS if g.signed and g.name in {"i8", "i16"}
)
_GRID_PARAMS = tuple(pytest.param(g, id=g.name) for g in _GRIDS)

# Spec 04_09: kt,kf ≤ 3, padding 0..3. ``(kernel, stride, padding)``.
_MAXPOOL_CONFIGS: tuple[tuple[int, int, int, str], ...] = (
    (2, 2, 0, "k2-s2-p0"),
    (3, 2, 1, "k3-s2-p1"),
    (3, 1, 1, "k3-s1-p1"),
)


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


def _same_grid_output_encoding(grid: QuantGridSpec, scale: float) -> OutputEncoding:
    """SAME_GRID_VALUE contract: no multiplier / rshift, output mirrors input."""
    return OutputEncoding(
        scale=torch.tensor(scale, dtype=torch.float32),
        zero_point=torch.tensor(grid.default_zero_point, dtype=torch.int32),
        qmin=grid.qmin,
        qmax=grid.qmax,
    )


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


def _run_maxpool(
    grid: QuantGridSpec,
    *,
    module_cls,
    kernel_size: int,
    stride: int,
    padding: int,
    seed_base: int,
    label: str,
) -> None:
    """SAME_GRID_VALUE: x and y share encoding; only same-scale path applies."""
    shape = (1, 4, 8, 8)
    for trial in range(_NUM_RANDOM_TRIALS):
        gen = torch.Generator().manual_seed(
            seed_base + trial * 9973 + hash(grid.name) % 10000
        )
        scale = 0.01 + 0.05 * torch.rand(
            1, generator=gen, dtype=torch.float32
        ).item()
        code_limit = max(8, grid.qmax // 2)
        qx = torch.randint(
            -code_limit, code_limit + 1, shape,
            generator=gen, dtype=torch.int32,
        )
        x = qx.to(torch.float32) * scale

        x_q = _quantize(x, scale=scale, grid=grid)
        out_enc = _same_grid_output_encoding(grid, scale)

        ref = F.max_pool2d(
            x.to(torch.float32),
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        output = get_fixed_kernel(module_cls)(
            [x_q], {}, out_enc,
            {
                "kernel_size": kernel_size,
                "stride": stride,
                "padding": padding,
            },
        )
        _assert_strict_gates(
            output, ref,
            label=f"{label} {grid.name} trial={trial}",
        )


@pytest.mark.parametrize(
    "config",
    [pytest.param(c, id=c[3]) for c in _MAXPOOL_CONFIGS],
)
@pytest.mark.parametrize("grid", _GRID_PARAMS)
def test_nn_maxpool2d_same_grid_random_fp32_per_grid(
    grid: QuantGridSpec,
    config: tuple[int, int, int, str],
):
    k, s, p, _label = config
    _run_maxpool(
        grid, module_cls=nn.MaxPool2d,
        kernel_size=k, stride=s, padding=p,
        seed_base=850_000 + hash(_label) % 10000,
        label=f"nn.MaxPool2d {_label}",
    )


@pytest.mark.parametrize(
    "config",
    [pytest.param(c, id=c[3]) for c in _MAXPOOL_CONFIGS],
)
@pytest.mark.parametrize("grid", _GRID_PARAMS)
def test_custom_maxpool2d_same_grid_random_fp32_per_grid(
    grid: QuantGridSpec,
    config: tuple[int, int, int, str],
):
    """``custom.MaxPool2d`` is the functional wrapper; manifest marks it
    ``dispatchable=False`` (adapter does not yet route it) but the kernel
    is still registered for direct unit-test use — this gate verifies
    parity with ``nn.MaxPool2d`` so a future adapter wiring is safe.
    """
    k, s, p, _label = config
    _run_maxpool(
        grid, module_cls=custom.MaxPool2d,
        kernel_size=k, stride=s, padding=p,
        seed_base=860_000 + hash(_label) % 10000,
        label=f"custom.MaxPool2d {_label}",
    )

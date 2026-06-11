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
"""D3 §4.10.2 Nearest-neighbour resize INT16 precision vs ideal float32.

Sister harness to ``test_shape_ops_int16_precision.py`` (P6 byte-stream
identity ops). Spec ``doc/04_算子详细规格/04_10_Resize类算子.md §4.10.2`` is
explicit: "直接复制量化值，无需重量化" — nearest is SAME_GRID_VALUE, payload
bits unchanged, only the spatial index map mutates. Therefore the strict
gates (cosine > 0.9999, max_error_lsb_float < 1.0) collapse to **bit-exact**
identity for this op: the only source of float-vs-int divergence is the
input quantization rounding, NOT the kernel.

Coverage:

- Grids: signed i8 / i16
- Modes: ``size=(H_out, W_out)`` (explicit shape) and ``scale_factor`` (float).
- Module types: ``nn.Upsample(mode='nearest')`` + ``nn.UpsamplingNearest2d``.
- Negative path: ``nn.Upsample(mode='bilinear')`` MUST raise (spec contract).
"""

from __future__ import annotations

import math
from typing import Any, Dict

import pytest

torch = pytest.importorskip("torch")

import aimet_torch.fixed_point.kernels  # noqa: E402, F401

from torch import nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

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


def _quantize(
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


def _same_grid_output_encoding(
    *,
    scale: float,
    grid: QuantGridSpec,
    zero_point: int,
) -> OutputEncoding:
    """SAME_GRID_VALUE: no multiplier/rshift (kernel rejects mismatched grids)."""
    return OutputEncoding(
        scale=torch.tensor(scale, dtype=torch.float32),
        zero_point=torch.tensor(zero_point, dtype=torch.int32),
        qmin=grid.qmin,
        qmax=grid.qmax,
        multiplier=None,
        rshift=None,
    )


def _random_fp32_input(
    grid: QuantGridSpec,
    gen: torch.Generator,
    *,
    shape: tuple[int, ...],
) -> tuple[torch.Tensor, float]:
    base = math.exp(
        torch.rand(1, generator=gen, dtype=torch.float32).item() * 2.5 - 3.5
    )
    code_limit = max(8, grid.qmax // 2)
    qx = torch.randint(
        -code_limit, code_limit + 1, shape,
        generator=gen, dtype=torch.int32,
    )
    x = qx.to(torch.float32) * base
    return x, base


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


def _run_same_grid_nearest(
    grid: QuantGridSpec,
    module_cls,
    *,
    shape: tuple[int, ...],
    extra: Dict[str, Any],
    seed_base: int,
) -> None:
    """SAME_GRID_VALUE: only same-scale (kernel rejects mismatch)."""
    for trial in range(_NUM_RANDOM_TRIALS):
        gen = torch.Generator().manual_seed(
            seed_base + trial * 9973 + hash(grid.name) % 10000
        )
        x, scale = _random_fp32_input(grid, gen, shape=shape)
        x_q = _quantize(
            x, scale=scale, grid=grid, zero_point=grid.default_zero_point,
        )
        out_enc = _same_grid_output_encoding(
            scale=scale, grid=grid, zero_point=grid.default_zero_point,
        )
        ref = F.interpolate(
            x.to(torch.float32),
            size=extra.get("size"),
            scale_factor=extra.get("scale_factor"),
            mode=str(extra.get("mode", "nearest")),
        )
        output = get_fixed_kernel(module_cls)([x_q], {}, out_enc, extra)
        _assert_strict_gates(
            output, ref,
            label=f"{module_cls.__name__} {extra} {grid.name} trial={trial}",
        )


# ---------- nn.Upsample(mode='nearest') ----------


@pytest.mark.parametrize("grid", _GRID_PARAMS)
def test_upsample_nearest_size_random_fp32_per_grid(grid: QuantGridSpec):
    _run_same_grid_nearest(
        grid, nn.Upsample,
        shape=(2, 4, 8, 8),
        extra={"mode": "nearest", "size": (16, 16), "scale_factor": None},
        seed_base=410_000,
    )


@pytest.mark.parametrize("grid", _GRID_PARAMS)
def test_upsample_nearest_scale_factor_random_fp32_per_grid(grid: QuantGridSpec):
    _run_same_grid_nearest(
        grid, nn.Upsample,
        shape=(2, 4, 8, 8),
        extra={"mode": "nearest", "size": None, "scale_factor": 2.0},
        seed_base=420_000,
    )


@pytest.mark.parametrize("grid", _GRID_PARAMS)
def test_upsample_nearest_downsample_random_fp32_per_grid(grid: QuantGridSpec):
    # Nearest also handles downsample (scale_factor < 1); spec 04_10 §4.10.2
    # uses ``round(idx_out * H_in / H_out)`` which works for both directions.
    _run_same_grid_nearest(
        grid, nn.Upsample,
        shape=(2, 4, 16, 16),
        extra={"mode": "nearest", "size": (8, 8), "scale_factor": None},
        seed_base=430_000,
    )


# ---------- nn.UpsamplingNearest2d ----------


@pytest.mark.parametrize("grid", _GRID_PARAMS)
def test_upsampling_nearest2d_size_random_fp32_per_grid(grid: QuantGridSpec):
    _run_same_grid_nearest(
        grid, nn.UpsamplingNearest2d,
        shape=(2, 4, 8, 8),
        extra={"mode": "nearest", "size": (12, 12), "scale_factor": None},
        seed_base=440_000,
    )


@pytest.mark.parametrize("grid", _GRID_PARAMS)
def test_upsampling_nearest2d_scale_factor_random_fp32_per_grid(grid: QuantGridSpec):
    _run_same_grid_nearest(
        grid, nn.UpsamplingNearest2d,
        shape=(2, 4, 8, 8),
        extra={"mode": "nearest", "size": None, "scale_factor": 3.0},
        seed_base=450_000,
    )


# ---------- Bit-exact identity payload check ----------


@pytest.mark.parametrize("grid", _GRID_PARAMS)
def test_nearest_int_repr_bit_exact_byte_stream_identity(grid: QuantGridSpec):
    """SAME_GRID_VALUE contract: kernel output ``int_repr`` must equal
    ``F.interpolate(int_repr, mode='nearest')`` element-wise (no rounding,
    no rescale). This is the strongest possible gate for the byte-stream
    identity claim in spec 04_10 §4.10.2.
    """

    gen = torch.Generator().manual_seed(990_000 + hash(grid.name) % 10000)
    shape = (2, 3, 5, 7)
    scale = 0.05
    code_limit = max(8, grid.qmax // 2)
    qx = torch.randint(-code_limit, code_limit + 1, shape, generator=gen, dtype=torch.int32)
    x_q = Int16QuantizedTensor(
        int_repr=saturate_sim_tensor(qx, grid.qmin, grid.qmax),
        scale=torch.tensor(scale, dtype=torch.float32),
        zero_point=torch.tensor(grid.default_zero_point, dtype=torch.int32),
        qmin=grid.qmin,
        qmax=grid.qmax,
    )
    out_enc = _same_grid_output_encoding(
        scale=scale, grid=grid, zero_point=grid.default_zero_point,
    )
    extra = {"mode": "nearest", "size": (10, 14), "scale_factor": None}

    output = get_fixed_kernel(nn.Upsample)([x_q], {}, out_enc, extra)
    ref_int = F.interpolate(
        x_q.int_repr.to(torch.float32), size=(10, 14), mode="nearest",
    ).to(x_q.int_repr.dtype)
    assert torch.equal(output.int_repr, ref_int), (
        f"{grid.name}: int_repr is not bit-exact vs F.interpolate (nearest); "
        "SAME_GRID_VALUE contract violated."
    )


# ---------- Negative paths ----------


def test_upsample_bilinear_mode_is_refused():
    """Spec contract: kernel is for nearest only; bilinear must raise so the
    adapter dispatch can fallback to FP32_QDQ instead of silently producing
    wrong values under the SAME_GRID_VALUE encoding contract.
    """

    grid = next(g for g in _GRIDS if g.name == "i16")
    shape = (1, 2, 4, 4)
    scale = 0.05
    gen = torch.Generator().manual_seed(31415)
    qx = torch.randint(-100, 101, shape, generator=gen, dtype=torch.int32)
    x_q = Int16QuantizedTensor(
        int_repr=qx.to(torch.int32),
        scale=torch.tensor(scale, dtype=torch.float32),
        zero_point=torch.tensor(grid.default_zero_point, dtype=torch.int32),
        qmin=grid.qmin,
        qmax=grid.qmax,
    )
    out_enc = _same_grid_output_encoding(
        scale=scale, grid=grid, zero_point=grid.default_zero_point,
    )
    extra = {"mode": "bilinear", "size": (8, 8), "scale_factor": None}

    with pytest.raises(ValueError, match="modes"):
        get_fixed_kernel(nn.Upsample)([x_q], {}, out_enc, extra)


def test_upsample_missing_target_shape_is_refused():
    """Spec ``nn.Upsample`` allows EITHER ``size`` OR ``scale_factor`` but not
    neither; mirror that contract at kernel entry so a misconfigured adapter
    surface surfaces as a precise error instead of an off-by-shape result.
    """

    grid = next(g for g in _GRIDS if g.name == "i16")
    shape = (1, 2, 4, 4)
    scale = 0.05
    gen = torch.Generator().manual_seed(27182)
    qx = torch.randint(-100, 101, shape, generator=gen, dtype=torch.int32)
    x_q = Int16QuantizedTensor(
        int_repr=qx.to(torch.int32),
        scale=torch.tensor(scale, dtype=torch.float32),
        zero_point=torch.tensor(grid.default_zero_point, dtype=torch.int32),
        qmin=grid.qmin,
        qmax=grid.qmax,
    )
    out_enc = _same_grid_output_encoding(
        scale=scale, grid=grid, zero_point=grid.default_zero_point,
    )
    extra = {"mode": "nearest", "size": None, "scale_factor": None}

    with pytest.raises(ValueError, match="size.*scale_factor"):
        get_fixed_kernel(nn.Upsample)([x_q], {}, out_enc, extra)

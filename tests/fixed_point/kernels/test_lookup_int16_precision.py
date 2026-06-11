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
"""P7 LOOKUP-family precision vs ideal float32 reference.

Two LUT shapes are exercised against the unified gates
(cosine > 0.9999, max_error_lsb_float < 1.0):

- CLZ-normalized quadratic LUT (``kernels/clz_lut.py``):
  ``custom.Sqrt``, ``custom.RSqrt``, ``custom.Reciprocal``, ``custom.Square``
  loaded from the abc tree (``abc_lut-shuai/lut_int_general/output/lut_test``).
- PWL LUT (``kernels/lut.py``) for ``custom.Log`` (and Sin/Cos sister
  cases), fitted in-process via ``generate_pwl_lut_for_export``.

Spec references:
  - ``doc/04_算子详细规格/04_04_其他单算子激活类算子.md`` (LUT spec)
  - ``doc/04_算子详细规格/04_05_其他单算子非线性算子.md`` (CLZ-normalized)

LOOKUP kernels carry a physical fit-error ceiling — the PWL family has
per-fn LSB limits in the hundreds-to-thousands (see
``PWL_VS_ANALYTIC_PER_FN_LIMITS``: log/exp = 2048 LSB, tanh = 2400 LSB).
The CLZ family is denser and usually clears the unified 1.0 LSB floor on
the bulk of its domain; the residual near-singular regions (RSqrt and
Reciprocal as x → 0, Sqrt at the bottom of the dynamic range) drive any
test failure. Each case is therefore evaluated against the *unified*
floor and marked ``xfail(strict=True)`` only when the kernel's fit
ceiling is the documented physical limit (see
``doc/precision_validation.md`` P7 section).

The float reference is computed on the full-precision fp32 input *before*
input quantization. The candidate is the kernel's float view of its
int16 output — i.e. quantization-induced error + LUT-fit error are both
charged against the gates.

Coverage:
  - Grids: signed i16 only (LUT kernels are i16-by-design; the spec does
    not register i8 variants).
  - Random fp32 inputs restricted to each operator's valid sub-domain.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable

import pytest

torch = pytest.importorskip("torch")

import aimet_torch.fixed_point.kernels  # noqa: E402, F401

import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from aimet_torch._base.nn.modules import custom  # noqa: E402
from aimet_torch.fixed_point import (  # noqa: E402
    InputEncoding,
    Int16QuantizedTensor,
    OutputEncoding,
    get_fixed_kernel,
)
from aimet_torch.fixed_point.kernels.clz_lut import (  # noqa: E402
    load_clz_lut_from_json,
)
from aimet_torch.fixed_point.metrics.accuracy import (  # noqa: E402
    cosine_similarity,
    max_error_lsb_float,
)
from aimet_torch.fixed_point.metrics.flags import (  # noqa: E402
    int16_eval_allow_debug_float,
)
from aimet_torch.fixed_point.offline.lut_gen import (  # noqa: E402
    generate_pwl_lut_for_export,
)
from aimet_torch.fixed_point.quant_grid import (  # noqa: E402
    QuantGridSpec,
    SIM_INT32_QUANT_GRIDS,
)
from aimet_torch.fixed_point.requantize import saturate_sim_tensor  # noqa: E402

_MIN_COSINE = 0.9999
_MAX_FLOAT_LSB = 1.0

_NUM_RANDOM_TRIALS = 32
_RANDOM_SHAPE = (4, 64)  # 256 elements per trial → 8192 across the parametrize

_ABC_ROOT = Path(__file__).resolve().parents[4] / "abc_lut-shuai"
_ABC_LUT_DIR = _ABC_ROOT / "lut_int_general" / "output" / "lut_test"

_I16_GRID = next(g for g in SIM_INT32_QUANT_GRIDS if g.signed and g.name == "i16")
_I16_PARAM = pytest.param(_I16_GRID, id=_I16_GRID.name)


# -----------------------------------------------------------------------------
# Shared helpers (mirror P5/P6 style)
# -----------------------------------------------------------------------------


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


def _output_encoding(
    *,
    scale_out: float,
    grid: QuantGridSpec,
    zero_point: int,
) -> OutputEncoding:
    return OutputEncoding(
        scale=torch.tensor(scale_out, dtype=torch.float32),
        zero_point=torch.tensor(zero_point, dtype=torch.int32),
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


# -----------------------------------------------------------------------------
# CLZ family — abc JSON-driven
# -----------------------------------------------------------------------------


def _abc_json(name: str) -> Path:
    return _ABC_LUT_DIR / f"{name}_clz_lut.json"


_HAS_ABC = _ABC_LUT_DIR.is_dir() and all(
    _abc_json(n).is_file()
    for n in ("sqrt", "rsqrt", "reciprocal", "power_2")
)


def _load_clz(name: str) -> tuple[dict, InputEncoding, OutputEncoding]:
    path = _abc_json(name)
    func_name, body = load_clz_lut_from_json(path)
    qin = body["quantization"]["input"]
    qout = body["quantization"]["output"]
    in_enc = InputEncoding(
        scale=torch.tensor(qin["scale"], dtype=torch.float32),
        zero_point=torch.tensor(qin["zero_point"], dtype=torch.int32),
        qmin=int(qin["min"]),
        qmax=int(qin["max"]),
    )
    out_enc = OutputEncoding(
        scale=torch.tensor(qout["scale"], dtype=torch.float32),
        zero_point=torch.tensor(qout["zero_point"], dtype=torch.int32),
        qmin=int(qout["min"]),
        qmax=int(qout["max"]),
    )
    assert func_name == name, (func_name, name)
    return body, in_enc, out_enc


def _sample_in_range(
    gen: torch.Generator,
    low: float,
    high: float,
    *,
    shape: tuple[int, ...] = _RANDOM_SHAPE,
) -> torch.Tensor:
    """Uniform fp32 sample on ``[low, high]``."""
    return low + (high - low) * torch.rand(shape, generator=gen, dtype=torch.float32)


def _run_clz_case(
    module_cls,
    abc_name: str,
    *,
    domain_low: float,
    domain_high: float,
    ref_fn: Callable[[torch.Tensor], torch.Tensor],
    seed_base: int,
    skip_near_zero_lsb_codes: int = 0,
    label: str,
) -> None:
    body, lut_in_enc, lut_out_enc = _load_clz(abc_name)
    in_scale = float(lut_in_enc.scale.item())
    out_scale = float(lut_out_enc.scale.item())

    for trial in range(_NUM_RANDOM_TRIALS):
        gen = torch.Generator().manual_seed(seed_base + trial * 9973)
        x = _sample_in_range(gen, domain_low, domain_high)
        if skip_near_zero_lsb_codes > 0:
            min_abs_value = skip_near_zero_lsb_codes * in_scale
            x = torch.where(
                x.abs() >= min_abs_value,
                x,
                torch.sign(x).where(
                    x.abs() >= min_abs_value, torch.full_like(x, min_abs_value)
                ),
            )

        x_q = _quantize(
            x, scale=in_scale, grid=_I16_GRID, zero_point=0,
        )

        out_enc = _output_encoding(
            scale_out=out_scale, grid=_I16_GRID, zero_point=0,
        )

        ref = ref_fn(x.to(torch.float32))
        # Clip ref to the LUT-declared output dynamic range so that codes
        # that saturate at qmax are not unfairly counted as 1-LSB errors.
        ref = torch.clamp(
            ref,
            min=float(lut_out_enc.qmin) * out_scale,
            max=float(lut_out_enc.qmax) * out_scale,
        )

        output = get_fixed_kernel(module_cls)(
            [x_q], {}, out_enc, {"clz_lut": body},
        )
        _assert_strict_gates(
            output, ref, label=f"{label} i16 trial={trial}",
        )


_XFAIL_CLZ_REASON_FIT_CEILING = (
    "CLZ-normalized quadratic LUT physical fit ceiling: kernel emits 16-bit "
    "fixed-segment coefficients on a CLZ-normalized mantissa head; residual "
    "fit error rises above the unified 1.0-LSB floor on a non-trivial code "
    "subset (typically near the singular region of x → 0 or the edge of the "
    "declared dynamic range). See doc/precision_validation.md §P7 for the "
    "spec-anchored physical limit and the per-op observed lsb_max snapshot."
)

_XFAIL_PWL_REASON_FIT_CEILING = (
    "PWL 16-segment LUT physical fit ceiling: per-fn analytic max_lsb limit "
    "(thresholds.PWL_VS_ANALYTIC_PER_FN_LIMITS) is two-to-four orders of "
    "magnitude above the unified 1.0-LSB floor (log/exp = 2048 LSB, sin/cos "
    "= 512 LSB). The single-op cosine still routinely clears 0.9999 because "
    "errors are localised, but max_error_lsb_float is bound by the segment "
    "ceiling. Logged as KNOWN_LIMIT, not a kernel regression."
)


@pytest.mark.skipif(not _HAS_ABC, reason="abc_lut-shuai CLZ JSON resources missing")
@pytest.mark.xfail(strict=True, reason=_XFAIL_CLZ_REASON_FIT_CEILING)
def test_sqrt_clz_random_fp32_i16():
    _run_clz_case(
        custom.Sqrt,
        "sqrt",
        # Sqrt: x ≥ 0; restrict to the declared positive sub-domain.
        domain_low=0.0,
        domain_high=8.0,
        ref_fn=lambda x: torch.sqrt(torch.clamp(x, min=0.0)),
        seed_base=810_000,
        skip_near_zero_lsb_codes=0,  # sqrt(0)=0 is well-defined
        label="Sqrt(clz)",
    )


@pytest.mark.skipif(not _HAS_ABC, reason="abc_lut-shuai CLZ JSON resources missing")
@pytest.mark.xfail(strict=True, reason=_XFAIL_CLZ_REASON_FIT_CEILING)
def test_rsqrt_clz_random_fp32_i16():
    _run_clz_case(
        custom.RSqrt,
        "rsqrt",
        # RSqrt: x > 0; pull samples away from 0 by enough LSBs that the
        # reference does not blow up beyond the LUT's declared output grid.
        # 0.5 ≤ x ≤ 8 keeps 1/sqrt(x) ∈ [0.354, 1.414] inside [-10, 10].
        domain_low=0.5,
        domain_high=8.0,
        ref_fn=lambda x: 1.0 / torch.sqrt(torch.clamp(x, min=1e-6)),
        seed_base=811_000,
        label="RSqrt(clz)",
    )


@pytest.mark.skipif(not _HAS_ABC, reason="abc_lut-shuai CLZ JSON resources missing")
@pytest.mark.xfail(strict=True, reason=_XFAIL_CLZ_REASON_FIT_CEILING)
def test_reciprocal_clz_random_fp32_i16():
    _run_clz_case(
        custom.Reciprocal,
        "reciprocal",
        # Reciprocal: x ≠ 0. Stay in [0.2, 5] so 1/x ∈ [0.2, 5] is well
        # inside the LUT's declared output range.
        domain_low=0.2,
        domain_high=5.0,
        ref_fn=lambda x: 1.0 / torch.clamp(x.abs(), min=1e-6) * torch.sign(x),
        seed_base=812_000,
        label="Reciprocal(clz)",
    )


@pytest.mark.skipif(not _HAS_ABC, reason="abc_lut-shuai CLZ JSON resources missing")
@pytest.mark.xfail(strict=True, reason=_XFAIL_CLZ_REASON_FIT_CEILING)
def test_square_clz_random_fp32_i16():
    _run_clz_case(
        custom.Square,
        "power_2",
        # Square: x ∈ ℝ but output ≤ qmax ⋅ scale_out; pick a band that keeps
        # x² well inside the LUT's declared output dynamic range.
        domain_low=-2.5,
        domain_high=2.5,
        ref_fn=lambda x: x * x,
        seed_base=813_000,
        label="Square(clz)",
    )


# -----------------------------------------------------------------------------
# PWL family — in-process fit via generate_pwl_lut_for_export
# -----------------------------------------------------------------------------


def _pwl_fit(
    fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    in_scale: float,
    in_qmin: int,
    in_qmax: int,
    out_scale: float,
    out_qmin: int,
    out_qmax: int,
    fn_name: str,
) -> tuple[dict, InputEncoding, OutputEncoding]:
    in_enc = InputEncoding(
        scale=torch.tensor(in_scale, dtype=torch.float32),
        zero_point=torch.tensor(0, dtype=torch.int32),
        qmin=in_qmin,
        qmax=in_qmax,
    )
    out_enc = OutputEncoding(
        scale=torch.tensor(out_scale, dtype=torch.float32),
        zero_point=torch.tensor(0, dtype=torch.int32),
        qmin=out_qmin,
        qmax=out_qmax,
    )
    pwl, _, _ = generate_pwl_lut_for_export(
        fn, in_enc, out_enc, enforce_quality=False, fn_name=fn_name,
    )
    return pwl, in_enc, out_enc


def _run_pwl_case(
    module_cls,
    fn_torch: Callable[[torch.Tensor], torch.Tensor],
    *,
    fn_name: str,
    in_scale: float,
    out_scale: float,
    domain_low: float,
    domain_high: float,
    seed_base: int,
    label: str,
) -> None:
    pwl, in_enc, out_enc = _pwl_fit(
        fn_torch,
        in_scale=in_scale,
        in_qmin=-32768,
        in_qmax=32767,
        out_scale=out_scale,
        out_qmin=-32768,
        out_qmax=32767,
        fn_name=fn_name,
    )

    for trial in range(_NUM_RANDOM_TRIALS):
        gen = torch.Generator().manual_seed(seed_base + trial * 9973)
        x = _sample_in_range(gen, domain_low, domain_high)
        x_q = _quantize(
            x, scale=in_scale, grid=_I16_GRID, zero_point=0,
        )

        op_out_enc = _output_encoding(
            scale_out=out_scale, grid=_I16_GRID, zero_point=0,
        )

        ref = fn_torch(x.to(torch.float32))
        # Clip ref to the declared output dynamic range so saturating codes
        # are not double-counted against the gates.
        ref = torch.clamp(
            ref,
            min=float(op_out_enc.qmin) * out_scale,
            max=float(op_out_enc.qmax) * out_scale,
        )

        output = get_fixed_kernel(module_cls)(
            [x_q], {}, op_out_enc,
            {"pwl_lut": pwl, "pwl_input_encoding": in_enc},
        )
        _assert_strict_gates(
            output, ref, label=f"{label} i16 trial={trial}",
        )


@pytest.mark.xfail(strict=True, reason=_XFAIL_PWL_REASON_FIT_CEILING)
def test_log_pwl_random_fp32_i16():
    # x ∈ [0.05, 8] → log(x) ∈ [-3.0, 2.08]
    _run_pwl_case(
        custom.Log,
        lambda x: torch.log(torch.clamp(x, min=1e-3)),
        fn_name="log",
        in_scale=8.0 / 32767,
        out_scale=4.0 / 32767,
        domain_low=0.05,
        domain_high=8.0,
        seed_base=820_000,
        label="Log(pwl)",
    )


@pytest.mark.xfail(strict=True, reason=_XFAIL_PWL_REASON_FIT_CEILING)
def test_exponential_pwl_random_fp32_i16():
    # exp domain narrowed so output ≤ 7 stays inside the symmetric grid.
    _run_pwl_case(
        custom.Exponential,
        torch.exp,
        fn_name="exp",
        in_scale=4.0 / 32767,
        out_scale=8.0 / 32767,
        domain_low=-4.0,
        domain_high=2.0,
        seed_base=821_000,
        label="Exp(pwl)",
    )


@pytest.mark.xfail(strict=True, reason=_XFAIL_PWL_REASON_FIT_CEILING)
def test_sin_pwl_random_fp32_i16():
    _run_pwl_case(
        custom.Sin,
        torch.sin,
        fn_name="sin",
        in_scale=(2.0 * math.pi) / 32767,
        out_scale=1.0 / 32767,
        domain_low=-math.pi,
        domain_high=math.pi,
        seed_base=822_000,
        label="Sin(pwl)",
    )


@pytest.mark.xfail(strict=True, reason=_XFAIL_PWL_REASON_FIT_CEILING)
def test_cos_pwl_random_fp32_i16():
    _run_pwl_case(
        custom.Cos,
        torch.cos,
        fn_name="cos",
        in_scale=(2.0 * math.pi) / 32767,
        out_scale=1.0 / 32767,
        domain_low=-math.pi,
        domain_high=math.pi,
        seed_base=823_000,
        label="Cos(pwl)",
    )


# -----------------------------------------------------------------------------
# S1 PWL activations (Sigmoid / Tanh / GELU / SiLU / Mish / Softplus +
# Hardsigmoid / Hardswish / LeakyReLU / PReLU + Abs).
#
# Smooth activations carry the same PWL fit ceiling as P7 (Log/Exp/Sin/Cos);
# piecewise-linear ones (Hardsigmoid/Hardswish/LeakyReLU/PReLU) are exactly
# representable in 16-segment PWL so they can clear the unified floor —
# however the in-process ``generate_pwl_lut_for_export`` does not detect the
# special structure and still emits a generic fit, so we keep ``xfail`` and
# rely on ``strict=False`` for the piecewise family. (Future asset-driven
# LUT loader can flip these to PASS once shipped tables are wired.)
#
# All cases share the unified ``_MIN_COSINE = 0.9999`` / ``_MAX_FLOAT_LSB =
# 1.0`` gate.
# -----------------------------------------------------------------------------


@pytest.mark.xfail(strict=True, reason=_XFAIL_PWL_REASON_FIT_CEILING)
def test_sigmoid_pwl_random_fp32_i16():
    # domain ±8 covers near-saturation tails; output ∈ [0, 1]
    _run_pwl_case(
        nn.Sigmoid,
        torch.sigmoid,
        fn_name="sigmoid",
        in_scale=8.0 / 32767,
        out_scale=1.0 / 32767,
        domain_low=-8.0,
        domain_high=8.0,
        seed_base=830_000,
        label="Sigmoid(pwl)",
    )


@pytest.mark.xfail(strict=True, reason=_XFAIL_PWL_REASON_FIT_CEILING)
def test_tanh_pwl_random_fp32_i16():
    _run_pwl_case(
        nn.Tanh,
        torch.tanh,
        fn_name="tanh",
        in_scale=4.0 / 32767,
        out_scale=1.0 / 32767,
        domain_low=-4.0,
        domain_high=4.0,
        seed_base=831_000,
        label="Tanh(pwl)",
    )


@pytest.mark.xfail(strict=True, reason=_XFAIL_PWL_REASON_FIT_CEILING)
def test_gelu_pwl_random_fp32_i16():
    _run_pwl_case(
        nn.GELU,
        F.gelu,
        fn_name="gelu",
        in_scale=4.0 / 32767,
        out_scale=4.0 / 32767,
        domain_low=-4.0,
        domain_high=4.0,
        seed_base=832_000,
        label="GELU(pwl)",
    )


@pytest.mark.xfail(strict=True, reason=_XFAIL_PWL_REASON_FIT_CEILING)
def test_silu_pwl_random_fp32_i16():
    _run_pwl_case(
        nn.SiLU,
        F.silu,
        fn_name="silu",
        in_scale=8.0 / 32767,
        out_scale=8.0 / 32767,
        domain_low=-8.0,
        domain_high=8.0,
        seed_base=833_000,
        label="SiLU(pwl)",
    )


@pytest.mark.xfail(strict=True, reason=_XFAIL_PWL_REASON_FIT_CEILING)
def test_mish_pwl_random_fp32_i16():
    _run_pwl_case(
        nn.Mish,
        F.mish,
        fn_name="mish",
        in_scale=4.0 / 32767,
        out_scale=4.0 / 32767,
        domain_low=-4.0,
        domain_high=4.0,
        seed_base=834_000,
        label="Mish(pwl)",
    )


@pytest.mark.xfail(strict=True, reason=_XFAIL_PWL_REASON_FIT_CEILING)
def test_softplus_pwl_random_fp32_i16():
    _run_pwl_case(
        nn.Softplus,
        F.softplus,
        fn_name="softplus",
        in_scale=8.0 / 32767,
        out_scale=8.0 / 32767,
        domain_low=-4.0,
        domain_high=8.0,
        seed_base=835_000,
        label="Softplus(pwl)",
    )


@pytest.mark.xfail(strict=False, reason=_XFAIL_PWL_REASON_FIT_CEILING)
def test_hardsigmoid_pwl_random_fp32_i16():
    # Piecewise-linear: exact PWL fit is *possible* (1/6 * x + 0.5 on [-3, 3])
    # but in-process generator emits a generic fit. strict=False ⇒ accept either.
    _run_pwl_case(
        nn.Hardsigmoid,
        F.hardsigmoid,
        fn_name="hardsigmoid",
        in_scale=4.0 / 32767,
        out_scale=1.0 / 32767,
        domain_low=-4.0,
        domain_high=4.0,
        seed_base=836_000,
        label="Hardsigmoid(pwl)",
    )


@pytest.mark.xfail(strict=True, reason=_XFAIL_PWL_REASON_FIT_CEILING)
def test_hardswish_pwl_random_fp32_i16():
    # Hardswish = x * Hardsigmoid(x); the in-process PWL fit on the smooth
    # boundary regions still leaves ~55 LSB residual → xfail strict.
    _run_pwl_case(
        nn.Hardswish,
        F.hardswish,
        fn_name="hardswish",
        in_scale=4.0 / 32767,
        out_scale=4.0 / 32767,
        domain_low=-4.0,
        domain_high=4.0,
        seed_base=837_000,
        label="Hardswish(pwl)",
    )


def test_leakyrelu_pwl_random_fp32_i16():
    # LeakyReLU is exactly 2-piece linear; the in-process PWL fit captures it
    # within the unified 0.9999 / 1.0 LSB floor (observed lsb_max ≈ 0.5).
    _run_pwl_case(
        nn.LeakyReLU,
        lambda x: F.leaky_relu(x, negative_slope=0.01),
        fn_name="leakyrelu",
        in_scale=4.0 / 32767,
        out_scale=4.0 / 32767,
        domain_low=-4.0,
        domain_high=4.0,
        seed_base=838_000,
        label="LeakyReLU(pwl)",
    )


def test_prelu_pwl_random_fp32_i16():
    # PReLU (slope=0.25 fixed at fit time) is identical-shape to LeakyReLU →
    # also clears the unified floor (observed lsb_max ≈ 0.625).
    _run_pwl_case(
        nn.PReLU,
        lambda x: F.prelu(x, torch.tensor([0.25])),
        fn_name="prelu",
        in_scale=4.0 / 32767,
        out_scale=4.0 / 32767,
        domain_low=-4.0,
        domain_high=4.0,
        seed_base=839_000,
        label="PReLU(pwl)",
    )


# custom.Abs has been migrated from PWL to the integer-abs path
# (spec 04_03 §4.3.5; see ``kernels/eltwise.py::AbsInt16Kernel``).
# Its precision coverage now lives in
# ``test_relu_int16_precision.py::test_abs_{same,cross}_scale_random_fp32_per_grid``
# alongside the rest of the SAME_GRID_OR_REQUANT family.


# -----------------------------------------------------------------------------
# Softmax (composed kernel: stable max-subtract + PWL exp + int32 sum +
# integer normalize). See ``kernels/softmax.py::softmax_int16_pwl``.
#
# ⚠ Output encoding nuance:
#   Softmax output is non-negative ([0, 1]). A *symmetric* i16 grid
#   (scale=1/qmax, zp=0) would saturate every prob > 0.5 to qmax —
#   inflating ``lsb_max`` by 3–4× without telling us anything about the
#   kernel itself. The correct grid uses the full 16-bit dynamic range
#   via ``scale = 1 / (qmax - qmin)`` and ``zp = qmin`` (u16-on-i16
#   storage) so ``q=qmin↔prob=0`` and ``q=qmax↔prob=1``.
#
# Composed-path noise sources (after the encoding fix above):
#   1. 15-bit fixed-point reciprocal-normalize — ``prob_fixed`` step
#      = 1/32768; mapping to 65535-span output ⇒ ~2 LSB output per
#      ``prob_fixed`` step. Dominant lsb contributor.
#   2. PWL exp 16-segment fit residual amplified by ``1/sum``.
#   3. Dynamic ``in_enc`` clamps ``shifted`` to [-32768, 0]; logit-gap
#      tails beyond 2·logit_max collapse onto the boundary.
#   4. int32 sum — exact within spec range, zero noise.
#
# ``cos_min`` is already at / above the v2-path Linear→Softmax floor
# (0.999). ``lsb_max`` stays in the hundreds-to-low-thousands, an order
# of magnitude over the unified 1.0 LSB floor — physical composed-path
# limit. Marked ``xfail(strict=True)`` to lock the ceiling.
# -----------------------------------------------------------------------------


_XFAIL_SOFTMAX_COMPOSED_REASON = (
    "Softmax is a composed kernel (PWL exp + int32 sum + 15-bit integer "
    "normalize). With the correct u16-on-i16 output encoding the cos_min "
    "is ≥ 0.998 (≥ 0.999 for cls≥32) but lsb_max stays ~500–3850 — the "
    "v2-path Linear→Softmax cos floor (0.999) passes for the typical "
    "cls≥32 cases, but the unified single-op floor (0.9999/1.0 LSB) does "
    "not. See doc/precision_validation.md#nnsoftmax-composed-kernelpwl-exp--int32-sum--整数-normalize."
)


def _run_softmax_case(
    *,
    logit_max: float,
    num_classes: int,
    seed_base: int,
    label: str,
) -> None:
    """Run :func:`softmax_int16_pwl` via the registered ``nn.Softmax``
    kernel against a fp32 reference.

    Output encoding uses ``scale = 1 / (qmax - qmin)`` and ``zp = qmin``
    so the full i16 dynamic range maps to softmax's [0, 1] output (the
    natural u16-on-i16 storage). A *symmetric* i16 grid would saturate
    every prob > 0.5 — see header comment.
    """
    in_scale = logit_max / 32767
    span = _I16_GRID.qmax - _I16_GRID.qmin  # 65535
    out_scale = 1.0 / span                  # 1 LSB ≈ 1.5e-5
    out_zp = _I16_GRID.qmin                 # -32768 → prob=0
    for trial in range(_NUM_RANDOM_TRIALS):
        gen = torch.Generator().manual_seed(seed_base + trial * 9973)
        x = (
            (torch.rand((4, num_classes), generator=gen, dtype=torch.float32) - 0.5)
            * 2.0 * logit_max
        )
        x_q = _quantize(
            x, scale=in_scale, grid=_I16_GRID, zero_point=0,
        )

        op_out_enc = _output_encoding(
            scale_out=out_scale, grid=_I16_GRID, zero_point=out_zp,
        )

        ref = torch.softmax(x.to(torch.float32), dim=-1)

        output = get_fixed_kernel(nn.Softmax)(
            [x_q], {}, op_out_enc, {"dim": -1},
        )
        _assert_strict_gates(
            output, ref, label=f"{label} i16 trial={trial}",
        )


@pytest.mark.xfail(strict=True, reason=_XFAIL_SOFTMAX_COMPOSED_REASON)
def test_softmax_logit4_cls8_random_fp32_i16():
    # Small class count — output codes are more spread (avg ~12.5%/class),
    # but small-prob tail (cls=8 with high-confidence input) still drives
    # large lsb_max. Expected to fail strict unified floor.
    _run_softmax_case(
        logit_max=4.0, num_classes=8,
        seed_base=870_000, label="Softmax(±4,cls=8)",
    )


@pytest.mark.xfail(strict=True, reason=_XFAIL_SOFTMAX_COMPOSED_REASON)
def test_softmax_logit4_cls32_random_fp32_i16():
    _run_softmax_case(
        logit_max=4.0, num_classes=32,
        seed_base=871_000, label="Softmax(±4,cls=32)",
    )


@pytest.mark.xfail(strict=True, reason=_XFAIL_SOFTMAX_COMPOSED_REASON)
def test_softmax_logit4_cls128_random_fp32_i16():
    _run_softmax_case(
        logit_max=4.0, num_classes=128,
        seed_base=872_000, label="Softmax(±4,cls=128)",
    )


@pytest.mark.xfail(strict=True, reason=_XFAIL_SOFTMAX_COMPOSED_REASON)
def test_softmax_logit8_cls32_random_fp32_i16():
    _run_softmax_case(
        logit_max=8.0, num_classes=32,
        seed_base=873_000, label="Softmax(±8,cls=32)",
    )

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
"""PWL fit accuracy for the LOOKUP activations that previously had no vs-analytic
gate (S1 of the single-op-precision push).

Sigmoid / Tanh / GELU / Abs already had per-fn LSB tests; this file extends the
same gate to:

  * SiLU       (PER_FN limits in ``thresholds.PWL_VS_ANALYTIC_PER_FN_LIMITS``)
  * Mish       (PER_FN limits)
  * Softplus   (PER_FN limits)
  * Hardsigmoid (no PER_FN entry → fallback to the default looser bound)
  * Hardswish   (no PER_FN entry → fallback)
  * LeakyReLU   (no PER_FN entry → fallback; slope=0.01 default)
  * PReLU       (no PER_FN entry → fallback; tested with weight=0.25, mirroring
                 the adapter's ``base_cls is nn.PReLU`` lambda which folds a
                 scalar weight into a leaky-style fn)

Each case asserts the metrics returned by ``generate_pwl_lut_for_export``
satisfy the limits resolved through ``resolve_pwl_quality_limits`` (per-fn
when available, sane default otherwise) plus the global cosine floor — same
pattern as ``test_pwl_lut_sigmoid_integer_path_within_hardware_analytic_bound``.
A small forward smoke check on ``evaluate_pwl_lut_int16`` confirms the kernel
hot path consumes the fitted LUT and returns the expected dtype/shape — so a
regression that breaks the fit *or* the kernel's PWL eval surfaces here.
"""

from __future__ import annotations

from typing import Callable

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

import aimet_torch.fixed_point.kernels  # noqa: E402, F401
from aimet_torch.fixed_point import (  # noqa: E402
    InputEncoding,
    OutputEncoding,
)
from aimet_torch.fixed_point.kernels.lut import evaluate_pwl_lut_int16  # noqa: E402
from aimet_torch.fixed_point.metrics.thresholds import (  # noqa: E402
    PWL_HARDWARE_NUM_SEGMENTS,
    PWL_VS_ANALYTIC_MIN_COSINE_SIMILARITY,
)
from aimet_torch.fixed_point.offline.lut_gen import (  # noqa: E402
    generate_pwl_lut_for_export,
    resolve_pwl_quality_limits,
)
from aimet_torch.fixed_point.requantize import SIM_TENSOR_DTYPE  # noqa: E402


def _in_enc(scale: float = 8.0 / 32767) -> InputEncoding:
    return InputEncoding(
        scale=torch.tensor(scale, dtype=torch.float32),
        zero_point=torch.tensor(0, dtype=torch.int32),
        qmin=-32768,
        qmax=32767,
    )


def _out_enc(
    *,
    scale: float,
    qmin: int,
    qmax: int,
) -> OutputEncoding:
    return OutputEncoding(
        scale=torch.tensor(scale, dtype=torch.float32),
        zero_point=torch.tensor(0, dtype=torch.int32),
        qmin=qmin,
        qmax=qmax,
    )


# Per-case output grid is picked to match each activation's actual range so
# the LSB error has fair headroom — using a symmetric grid for a [0, 1]
# function would waste half the codes and inflate ``max_lsb`` artificially.
# (Matches the rationale in
# ``test_pwl_lut_sigmoid_integer_path_within_hardware_analytic_bound``.)
_PWL_CASES = [
    (
        "silu",
        F.silu,
        # SiLU: ~[-0.28, 8] on x ∈ [-8, 8]
        dict(scale=8.0 / 32767, qmin=-32768, qmax=32767),
    ),
    (
        "mish",
        F.mish,
        # Mish: ~[-0.31, 8] on x ∈ [-8, 8]
        dict(scale=8.0 / 32767, qmin=-32768, qmax=32767),
    ),
    (
        "softplus",
        # default beta=1, threshold=20
        F.softplus,
        # Softplus: [0, 8.0003] on x ∈ [-8, 8]
        dict(scale=8.0 / 32767, qmin=0, qmax=32767),
    ),
    (
        "hardsigmoid",
        F.hardsigmoid,
        # Hardsigmoid: [0, 1]
        dict(scale=1.0 / 32767, qmin=0, qmax=32767),
    ),
    (
        "hardswish",
        F.hardswish,
        # Hardswish: ~[-0.375, 8] on x ∈ [-8, 8]
        dict(scale=8.0 / 32767, qmin=-32768, qmax=32767),
    ),
    (
        "leaky_relu",
        # Default negative_slope=0.01 (the adapter passes the module's slope;
        # fixing the default keeps the gate deterministic).
        lambda x: F.leaky_relu(x, negative_slope=0.01, inplace=False),
        # LeakyReLU(0.01): [-0.08, 8] on x ∈ [-8, 8]
        dict(scale=8.0 / 32767, qmin=-32768, qmax=32767),
    ),
    (
        "prelu",
        # PReLU with fixed scalar weight (matches the adapter's path which
        # only handles ``weight.numel() == 1``); slope=0.25 lifts the negative
        # tail well above leaky_relu's so the per-fn fit characteristics are
        # actually different.
        lambda x: torch.where(x >= 0, x, x * 0.25),
        # PReLU(0.25): [-2, 8] on x ∈ [-8, 8]
        dict(scale=8.0 / 32767, qmin=-32768, qmax=32767),
    ),
]


@pytest.mark.parametrize("fn_name,torch_fn,out_grid", _PWL_CASES)
def test_pwl_lut_extra_activation_fit_within_analytic_limits(
    fn_name: str,
    torch_fn: Callable[[torch.Tensor], torch.Tensor],
    out_grid: dict,
):
    """Per-fn LSB / cosine gate vs analytic on a 16-segment hardware-fixed PWL.

    This is the missing 'precision floor' for the PWL family: the metrics
    are computed on a uniformly-sampled int16 input grid (4096 samples by
    default in ``generate_pwl_lut_for_export``), so a bad slope/threshold
    fit shows up directly here rather than only as a downstream model-level
    cosine miss.
    """

    in_enc = _in_enc()
    out_enc = _out_enc(**out_grid)

    pwl, num_segments, metrics = generate_pwl_lut_for_export(
        torch_fn,
        in_enc,
        out_enc,
        # ``enforce_quality=False`` so we get the metrics dict back even when
        # a fallback-limited fn slightly grazes a default bound; we then
        # assert the per-fn limits (which may be tighter than the helper's
        # default if PER_FN_LIMITS has an entry) ourselves.
        enforce_quality=False,
        fn_name=fn_name,
    )
    assert num_segments == PWL_HARDWARE_NUM_SEGMENTS

    limits = resolve_pwl_quality_limits(fn_name)
    assert metrics["max_lsb"] <= limits["max_lsb"], (
        f"{fn_name} max_lsb={metrics['max_lsb']:.2f} > limit={limits['max_lsb']}"
    )
    assert metrics["p99_lsb"] <= limits["p99_lsb"], (
        f"{fn_name} p99_lsb={metrics['p99_lsb']:.2f} > limit={limits['p99_lsb']}"
    )
    assert metrics["rmse_lsb"] <= limits["rmse_lsb"], (
        f"{fn_name} rmse_lsb={metrics['rmse_lsb']:.2f} > limit={limits['rmse_lsb']}"
    )
    assert (
        metrics["cosine_similarity"] >= PWL_VS_ANALYTIC_MIN_COSINE_SIMILARITY
    ), (
        f"{fn_name} cosine={metrics['cosine_similarity']:.6f} < "
        f"floor={PWL_VS_ANALYTIC_MIN_COSINE_SIMILARITY}"
    )


@pytest.mark.parametrize("fn_name,torch_fn,out_grid", _PWL_CASES)
def test_evaluate_pwl_lut_int16_consumes_fitted_lut(
    fn_name: str,
    torch_fn: Callable[[torch.Tensor], torch.Tensor],
    out_grid: dict,
):
    """Smoke gate for the kernel hot path on each newly-covered activation:
    the fitted LUT must be consumed by ``evaluate_pwl_lut_int16`` without
    raising and return the canonical sim-tensor dtype + matching shape.
    Together with the fit-quality test above, this means a future
    refactor that breaks either the fit math *or* the kernel's PWL
    integer evaluator surfaces under this same parametrize.
    """

    in_enc = _in_enc()
    out_enc = _out_enc(**out_grid)
    pwl, _, _ = generate_pwl_lut_for_export(
        torch_fn,
        in_enc,
        out_enc,
        enforce_quality=False,
        fn_name=fn_name,
    )

    qx = torch.tensor([-32768, -10000, 0, 10000, 32767], dtype=torch.int16)
    qy = evaluate_pwl_lut_int16(qx, pwl)
    assert qy.shape == qx.shape
    assert qy.dtype is SIM_TENSOR_DTYPE
    assert torch.all(qy >= out_enc.qmin)
    assert torch.all(qy <= out_enc.qmax)

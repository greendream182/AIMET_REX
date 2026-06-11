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
"""Spec § 4.5.1 ``square_mean`` and § 4.5.2 ``variance`` base-instruction
INT16 unit tests.

Spec ``doc/04_算子详细规格/04_05_归一化类算子.md`` defines two
DSP-level base instructions that LayerNorm / RMSNorm-family operators
embed:

  * **§ 4.5.1 ``square_mean``**: ``Σ(q_x − Z_x)² · inv_N >> shift_N``
    rescaled to ``(S_sq, Z_sq)`` via ``M_sq / rshift_sq`` (single-pass).
  * **§ 4.5.2 ``variance``**: two-pass — first ``q_μ = (Σ(q_x − Z_x) ·
    inv_N) >> shift_N``, then ``q_var = (Σd² · inv_N) >> shift_N``
    rescaled to ``(S_var, Z_var)`` via ``M_var / rshift_var``.

Both are exposed as public API entry points in
``aimet_torch/fixed_point/kernels/norm.py`` even though they are not
themselves registered to the kernel dispatch (spec describes them as
DSP base instructions, not user-facing ``nn.Module`` types). The
LayerNorm INT16 kernel uses both; future RMSNorm / cLN2D DSP-parity
work (``FU-NORM-SUBOP-VS-DSP-PARITY``) will reuse the same helpers.

These tests pin the integer carrier output against an idealised fp32
reference (each base instruction's mathematical definition lifted from
spec line 67-75 / line 109-117 with no LUT involvement). Cosine clears
the project-unified ``> 0.9999`` floor easily; ``lsb_max`` is bounded
because there is **no** LUT amplification on the base instructions
themselves (cf. LayerNorm where the RSqrt CLZ LUT residual is the
dominant term). The local ``_BASE_INSTR_LSB_BOUND`` envelope below
allows for accumulator round-half from ``inv_N`` and the final
``M/rshift`` rescale (spec line 71/114) — this is the pure integer
round-half budget LayerNorm pays on top of its LUT residual, isolated
here so the next time the budget grows we know which step regressed.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

# Ensure norm kernel module is loaded so it registers helpers (importing
# the package side-effect chain is required by other kernel tests too;
# we keep the same import-pattern for consistency).
import aimet_torch.fixed_point.kernels  # noqa: E402, F401

from aimet_torch.fixed_point.kernels.norm import (  # noqa: E402
    integer_square_mean_base_instruction,
    integer_variance_base_instruction,
)
from aimet_torch.fixed_point.metrics.accuracy import (  # noqa: E402
    cosine_similarity,
)

_MIN_COSINE = 0.9999

# Pure integer round-half budget for the base instructions (no LUT
# amplification). Allows up to 5 LSB on the (S_var)/(S_sq) output grid:
#   * ``(s · inv_N) >> shift_N`` round-half on mean / sum  → ~1 LSB
#   * ``(v · M_var) >> rshift_var`` final rescale          → ~3 LSB
#   * trial-to-trial input-distribution slack              → +1 LSB
# Tightening past ~3 LSB requires giving up the ``inv_N`` two's-power
# alignment in ``_compute_inv_n_shift_n`` (spec §4.5.2 line 125 freezes
# this convention; do not relax without coordinating with the compiler
# side that produces matching ``inv_N``).
_BASE_INSTR_LSB_BOUND = 5

_NUM_RANDOM_TRIALS = 8

# Test layouts: ``[B, C, T, F]`` reduced over ``(1, 3)`` covers the
# CLN / cLN2D / SimCln2d use cases (spec §§ 4.5.5 / 4.5.7 / 4.5.8 all
# reduce on the (C, F) axes pair). Reduce sizes 8 / 16 / 32 cover the
# small / medium regimes that exercise the ``inv_N`` resolution.
_REDUCE_CASES = (
    pytest.param(((2, 4, 4, 2), (1, 3)), id="cf_4x2_n8"),
    pytest.param(((2, 4, 4, 4), (1, 3)), id="cf_4x4_n16"),
    pytest.param(((1, 8, 4, 4), (1, 3)), id="cf_8x4_n32"),
)


def _make_input(input_shape, gen, *, qmax=32767):
    """Produce ``(x_int_repr, S_x, Z_x, qmax)`` for an i16 grid.

    ``x ~ N(0, 1)``; ``S_x`` calibrated at ``x.abs().max() / (qmax / 1.5)``
    (1.5x headroom margin, mirroring the LayerNorm test's calibrated
    path). Symmetric grid (``Z_x = 0``) keeps the math close to the spec
    closed form ``(q_x − Z_x) = q_x``.
    """
    x = torch.randn(input_shape, generator=gen, dtype=torch.float32)
    scale_x = max(x.abs().max().item() / (qmax / 1.5), 1e-6)
    q_x = torch.round(x / scale_x).clamp(-qmax, qmax).to(torch.int32)
    return q_x, scale_x, 0, qmax


def _quantize_reference(value: torch.Tensor, *, scale: float, zp: int, qmax: int) -> torch.Tensor:
    """``round(value / scale + zp)`` clamped to ``[-qmax, qmax]`` (signed)."""
    q = torch.round(value / scale + zp).clamp(-qmax, qmax).to(torch.int32)
    return q


def _max_int_lsb(out_int: torch.Tensor, ref_int: torch.Tensor) -> float:
    """Worst-case ``|int_repr − ref_int|`` across all elements."""
    return float((out_int.to(torch.int64) - ref_int.to(torch.int64)).abs().max().item())


@pytest.mark.parametrize("shapes", _REDUCE_CASES)
def test_square_mean_base_instruction_matches_fp32(shapes):
    """Spec § 4.5.1 ``square_mean`` integer carrier ≈ fp32 reference.

    Reference: ``mean_sq = ((q_x − Z_x) · S_x).square().mean(dim=dims)``
    quantized to the chosen ``(S_sq, Z_sq)`` grid. Asserts cosine
    similarity > 0.9999 and ``lsb_max ≤ _BASE_INSTR_LSB_BOUND`` on the
    output integer grid (no LUT involvement, so this is a pure
    bit-parity-of-fp32 test).

    The output grid here is independent of any downstream LUT — we pick
    ``S_sq`` calibrated against the reference dynamic range to keep the
    grid utilization tight (1.5x headroom margin same as LayerNorm
    calibration default).
    """
    input_shape, dims = shapes
    qmax_out = 32767
    cos_min = 1.0
    lsb_max = 0.0
    for trial in range(_NUM_RANDOM_TRIALS):
        gen = torch.Generator().manual_seed(800_000 + trial * 9973 + hash(input_shape) % 10000)
        q_x, scale_x, zp_x, _ = _make_input(input_shape, gen)

        x_float = (q_x.to(torch.float32) - zp_x) * scale_x
        ref_float = x_float.square().mean(dim=dims, keepdim=True)
        scale_sq = max(ref_float.abs().max().item() / (qmax_out / 1.5), 1e-6)
        zp_sq = 0
        ref_int = _quantize_reference(ref_float, scale=scale_sq, zp=zp_sq, qmax=qmax_out)

        q_sq, _n_a = integer_square_mean_base_instruction(
            q_x, zp_x, dims,
            scale_x=scale_x,
            out_scale=scale_sq,
            out_zero_point=zp_sq,
            out_qmin=-qmax_out,
            out_qmax=qmax_out,
        )

        cos = cosine_similarity(ref_int.flatten().to(torch.float32), q_sq.flatten().to(torch.float32))
        cos_min = min(cos_min, cos)
        lsb_max = max(lsb_max, _max_int_lsb(q_sq, ref_int))

    assert cos_min > _MIN_COSINE, (
        f"square_mean base instr cos {cos_min:.6f} <= {_MIN_COSINE}"
    )
    assert lsb_max <= _BASE_INSTR_LSB_BOUND, (
        f"square_mean base instr lsb {lsb_max} > {_BASE_INSTR_LSB_BOUND}"
    )


@pytest.mark.parametrize("shapes", _REDUCE_CASES)
def test_variance_base_instruction_matches_fp32(shapes):
    """Spec § 4.5.2 ``variance`` integer carrier ≈ fp32 reference.

    Reference: ``var = ((q_x − Z_x) · S_x − mean).square().mean(dim=dims)``
    quantized to the chosen ``(S_var, Z_var)`` grid. The integer path
    additionally returns ``q_mu_centered = (s_o · inv_N) >> shift_N``
    (centered on ``Z_μ = Z_x``) — the test cross-validates this against
    the fp32 mean as well, which is the same convention the LayerNorm
    INT16 kernel relies on for its ``q_x − q_μ`` subtraction step.
    """
    input_shape, dims = shapes
    qmax_out = 32767
    cos_var_min = 1.0
    lsb_var_max = 0.0
    cos_mu_min = 1.0
    lsb_mu_max = 0.0
    for trial in range(_NUM_RANDOM_TRIALS):
        gen = torch.Generator().manual_seed(810_000 + trial * 9973 + hash(input_shape) % 10000)
        q_x, scale_x, zp_x, _ = _make_input(input_shape, gen)

        x_float = (q_x.to(torch.float32) - zp_x) * scale_x
        mu_float = x_float.mean(dim=dims, keepdim=True)
        var_float = (x_float - mu_float).square().mean(dim=dims, keepdim=True)

        scale_var = max(var_float.abs().max().item() / (qmax_out / 1.5), 1e-6)
        zp_var = 0
        ref_var_int = _quantize_reference(var_float, scale=scale_var, zp=zp_var, qmax=qmax_out)

        q_mu_centered, q_var, _n_a = integer_variance_base_instruction(
            q_x, zp_x, dims,
            scale_x=scale_x,
            var_input_scale=scale_var,
            var_input_zero_point=zp_var,
            var_input_qmin=-qmax_out,
            var_input_qmax=qmax_out,
        )
        # ``q_mu_centered`` = ``q_μ − Z_μ`` on grid ``S_x`` (Default A).
        # Cross-check against fp32 mean quantized to ``S_x`` (Z_μ = Z_x = 0).
        ref_mu_int = _quantize_reference(
            mu_float, scale=scale_x, zp=zp_x, qmax=32767,
        )

        cos_var = cosine_similarity(
            ref_var_int.flatten().to(torch.float32), q_var.flatten().to(torch.float32)
        )
        cos_var_min = min(cos_var_min, cos_var)
        lsb_var_max = max(lsb_var_max, _max_int_lsb(q_var, ref_var_int))

        cos_mu = cosine_similarity(
            ref_mu_int.flatten().to(torch.float32),
            q_mu_centered.flatten().to(torch.float32),
        )
        cos_mu_min = min(cos_mu_min, cos_mu)
        lsb_mu_max = max(lsb_mu_max, _max_int_lsb(q_mu_centered.to(torch.int32), ref_mu_int))

    assert cos_var_min > _MIN_COSINE, (
        f"variance base instr cos {cos_var_min:.6f} <= {_MIN_COSINE}"
    )
    assert lsb_var_max <= _BASE_INSTR_LSB_BOUND, (
        f"variance base instr lsb {lsb_var_max} > {_BASE_INSTR_LSB_BOUND}"
    )
    # Mean intermediate is even tighter — single ``inv_N`` rescale only,
    # no second M/rshift step.
    assert cos_mu_min > _MIN_COSINE, (
        f"variance base instr mu cos {cos_mu_min:.6f} <= {_MIN_COSINE}"
    )
    assert lsb_mu_max <= 2, (
        f"variance base instr mu lsb {lsb_mu_max} > 2 (single round-half budget)"
    )


def test_square_mean_rejects_zero_reduce_size():
    """``N_A`` must be positive — caller bug (empty reduce dim) is loud."""
    q_x = torch.zeros((2, 0, 4, 1), dtype=torch.int32)
    with pytest.raises(ValueError, match="N must be positive"):
        integer_square_mean_base_instruction(
            q_x, 0, (1, 3),
            scale_x=0.01,
            out_scale=0.01,
            out_zero_point=0,
            out_qmin=-32767,
            out_qmax=32767,
        )

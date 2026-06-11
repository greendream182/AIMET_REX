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
"""INT16 fixed-point kernels for normalization operators (spec full bit-parity).

Spec ``doc/04_算子详细规格/04_05_归一化类算子.md`` §4.5.4 (LayerNorm) gives
two views of the same algorithm that this kernel **closes the gap on**:

  * **Mathematical definition** (line 387):
    ``q_y = (S_γ·S_x·S_inv / S_y) · q_γ · q_inv · [(q_x-Z_x) - (q_μ-Z_μ)]
            + (S_β/S_y) · q_β + Z_y``
    where ``q_inv`` comes from the rsqrt LUT (``S_inv · q_inv = f(q_σ²)``).

  * **Hardware integer path** (line 422-428): the same composition with
    each ``S·..·.../S_y`` ratio folded into ``16bit (M, rshift)`` pairs,
    ``q_μ`` / ``q_σ²`` produced by the in-line ``variance`` base
    instruction (spec §4.5.2), and ``q_y`` computed via three integer
    multiply-shift terms ending with ``+ b^LN``.

The registered :class:`LayerNormInt16Kernel` implements the **hardware
integer path** end-to-end (4 steps):

  1. ``q_μ, q_var`` via :func:`_inline_integer_variance` — spec §4.5.2
     two-stage integer reduce: ``s_o = Σ(q_x − Z_x)``, ``q_μ-Z_μ =
     (s_o · inv_N) >> shift_N``, ``v_o = (Σd² · inv_N) >> shift_N``,
     ``q_var = ((v_o · M_var) >> rshift_var) + Z_var``.  ``S_μ = S_x``
     and ``Z_μ = Z_x`` by convention (kept consistent with the ``q_x −
     q_μ`` subtraction in step 3); ``(S_var, Z_var)`` are slaved to the
     RSqrt LUT input grid so the LUT step has zero rescale cost.
  2. ``q_inv = rsqrt_lut(q_var)`` via :func:`_rsqrt_via_clz_lut` — the
     only spec-mandated lookup step (spec line 384), feeding ``q_var``
     to ``_clz_int16_forward(..., "rsqrt")``.
  3. Integer affine per spec line 422-428 via :func:`_integer_affine`:
     ``q_y = ((q_γ · q_inv · (q_x-Z_x) · M_x) >> rshift_x)
            − ((q_γ · q_inv · (q_μ-Z_μ) · M_μ) >> rshift_μ) + b^LN``.
     ``(M_x, rshift_x)`` and ``(M_μ, rshift_μ)`` are equal under the
     ``S_μ = S_x`` convention but are kept as separate fields for
     spec-form parity. ``b^LN = ((q_β · M_β) >> rshift_β) + Z_y``.
     When ``elementwise_affine=False`` the kernel slots ``q_γ = +1`` /
     ``q_β = 0`` and skips the bias term so the integer path still
     applies.
  4. Saturate to ``output_encoding.qmin/qmax`` — there is no further
     ``round`` because the integer affine output already lives on the
     output grid.

Quantization defaults that the spec leaves implicit (documented inline
under "Default A" — see ``precision_validation.md`` ``## nn.LayerNorm``):

  * ``S_γ = γ.abs().max() / qmax_i16, Z_γ = 0`` (symmetric per-tensor
    weight quant, same convention as ``aimet_torch.conv2d``).
  * ``S_β = β.abs().max() / qmax_i16, Z_β = 0`` (same).
  * ``S_μ = S_x, Z_μ = Z_x`` (mean shares the input grid).
  * ``S_var, Z_var`` = RSqrt LUT input quantization grid (taken from
    the LUT body's ``quantization.input`` block).

Reference paths retained but **not registered** (for unit-test oracle
use only): :func:`layer_norm_float_reference` (pure fp32, no LUT) and
:func:`layer_norm_lut_fp32_affine_reference` (LUT step integer, but
variance and affine still in fp32 — the legacy spec-aligned reference
prior to full bit-parity closure).

Precision contract (project-unified strict gates ``cos > 0.9999`` /
``lsb < 1.0`` LSB):

  * Cosine clears 0.9999 on every measured case because the residual is
    magnitude-bounded (LUT PWL fit) rather than direction-biased.
  * ``lsb_max`` is bounded by the RSqrt CLZ LUT physical fit ceiling
    amplified by LayerNorm's ``γ/std`` factor — registered in
    ``thresholds.LAYERNORM_VS_FP32_PER_GRID_LIMITS``. Numerical tests
    therefore use ``xfail(strict=True, reason=...)`` same as the P7
    PWL/CLZ KNOWN_LIMIT bucket. The hardware bit-parity closure does
    not change this floor (verified by the LUT-vs-no-LUT oracle gate);
    integer M/rshift round-half adds ≤ 2 LSB more than the fp32 affine
    reference but stays well inside ``cos > 0.9999``.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from aimet_torch.fixed_point.encoding import OutputEncoding
from aimet_torch.fixed_point.offline.scale_fixed import quantize_scale_to_m_rshift
from aimet_torch.fixed_point.registry import register_fixed_kernel
from aimet_torch.fixed_point.requantize import (
    SIM_TENSOR_DTYPE,
    saturate_sim_tensor,
)
from aimet_torch.fixed_point.tensor import Int16QuantizedTensor

# Spec § 4.5.4 line 422-428 integer affine carrier width. Each of the
# four 16-bit factors (q_γ, q_inv, q_centered, M_x) can hit ±2^15, so
# the unshifted product peaks at 2^60 — int64 is the smallest standard
# carrier that holds it without overflow. We keep the carrier in int64
# until the final ``>> rshift`` and saturation to the output grid.
_LN_INTEGER_AFFINE_CARRIER_DTYPE = torch.int64
_LN_GAMMA_BETA_QMAX = 32767  # int16 symmetric weight grid, Z=0


def _dequant_to_float(tensor: Int16QuantizedTensor) -> torch.Tensor:
    """Return ``(q - Z_x) * S_x`` in fp32 on the same device as ``int_repr``."""

    scale = tensor.scale.to(device=tensor.int_repr.device, dtype=torch.float32)
    while scale.ndim < tensor.int_repr.ndim:
        scale = scale.unsqueeze(-1)
    return tensor.centered_int32().to(torch.float32) * scale


def _requantize_from_float(
    value: torch.Tensor,
    output_encoding: OutputEncoding,
) -> Int16QuantizedTensor:
    """``round(value / S_y) + Z_y`` clamped to ``[qmin, qmax]`` (per-tensor LN output)."""

    scale = output_encoding.scale.to(device=value.device, dtype=torch.float32)
    zp = output_encoding.zero_point.to(device=value.device, dtype=torch.float32)
    while scale.ndim < value.ndim:
        scale = scale.unsqueeze(-1)
        zp = zp.unsqueeze(-1)
    q = torch.round(value / scale + zp)
    q = saturate_sim_tensor(
        q.to(torch.int32), output_encoding.qmin, output_encoding.qmax
    )
    return Int16QuantizedTensor(
        int_repr=q.to(SIM_TENSOR_DTYPE),
        scale=output_encoding.scale.to(device=value.device),
        zero_point=output_encoding.zero_point.to(
            device=value.device, dtype=torch.int32
        ),
        qmin=output_encoding.qmin,
        qmax=output_encoding.qmax,
        axis=output_encoding.axis,
    )


def _normalized_shape_from_extra(extra: Dict[str, Any]) -> Sequence[int]:
    """Normalize ``extra['normalized_shape']`` to a tuple of ints (accept int or list)."""

    raw = extra.get("normalized_shape")
    if raw is None:
        raise ValueError(
            "LayerNorm INT16 kernel: extra['normalized_shape'] is required. "
            "The adapter at ``aimet_torch/v2/quantization/affine/fixed_point/"
            "adapter.py`` must populate it from the qmodule attribute."
        )
    if isinstance(raw, int):
        return (raw,)
    return tuple(int(d) for d in raw)


def _resolve_rsqrt_lut_body(extra: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve the RSqrt CLZ LUT body for spec §4.5.4's ``S_inv · q_inv = f(q_σ²)``.

    Resolution order (mirrors how ``custom.RSqrt`` consumes its LUT via the
    adapter sidecar path):

    1. ``extra['rsqrt_clz_lut']`` if the caller provided it (typical when
       the adapter has loaded the LUT body from a sim sidecar).
    2. ``extra['clz_lut']`` for symmetry with the single-op CLZ kernels
       (``SqrtInt16ClzKernel`` etc.) — but **only** when ``extra`` does
       not also carry the LayerNorm-specific keys, to avoid eating a
       neighbouring op's LUT.
    3. The default abc tree asset at
       ``abc_lut-shuai/lut_int_general/output/lut_test/rsqrt_clz_lut.json``.

    Raises ``RuntimeError`` if no LUT can be resolved — that surfaces an
    integration issue (missing LUT asset, mis-routed adapter) immediately
    rather than silently degrading to a pure-float kernel.
    """

    candidate = extra.get("rsqrt_clz_lut")
    if candidate is not None:
        return candidate
    candidate = extra.get("clz_lut")
    if candidate is not None and "normalized_shape" not in candidate:
        # Heuristic: ``clz_lut`` body has top-level ``quantization`` /
        # ``segments``; if a caller stuffed LayerNorm extras into the
        # same key we'd see ``normalized_shape`` here. The guard keeps a
        # future ``norm`` extra from being misread as a LUT body.
        return candidate

    # pylint: disable=import-outside-toplevel
    from pathlib import Path

    from aimet_torch.fixed_point.kernels.clz_lut import load_clz_lut_from_json
    from aimet_torch.fixed_point.offline.clz_gen import resolve_abc_lut_root

    abc_root = resolve_abc_lut_root()
    if abc_root is None:
        raise RuntimeError(
            "LayerNorm INT16 kernel: RSqrt CLZ LUT could not be resolved. "
            "Provide ``extra['rsqrt_clz_lut']`` (LUT JSON body) or make the "
            "abc_lut-shuai tree available in the workspace so the default "
            "rsqrt_clz_lut.json can be loaded."
        )
    path = Path(abc_root) / "lut_int_general" / "output" / "lut_test" / "rsqrt_clz_lut.json"
    if not path.is_file():
        raise RuntimeError(
            f"LayerNorm INT16 kernel: default RSqrt LUT missing at {path}. "
            "Provide ``extra['rsqrt_clz_lut']`` or restore the asset."
        )
    _func, body = load_clz_lut_from_json(path, func_name="rsqrt")
    return body


def _rsqrt_via_clz_lut(
    value_float: torch.Tensor,
    rsqrt_lut_body: Dict[str, Any],
) -> torch.Tensor:
    """Compute ``1/√x`` via the RSqrt CLZ LUT, returning a fp32 tensor.

    Spec §4.5.4 says ``S_inv · q_inv = f(q_σ²)``. The LUT body declares
    its own input/output quantization grids (``quantization.input`` /
    ``quantization.output``); we quantize the fp32 ``value`` to the LUT
    input grid, call the same ``_clz_int16_forward`` engine the standalone
    ``custom.RSqrt`` kernel uses, and dequantize the output back to fp32
    for the surrounding affine. This keeps the rsqrt error envelope
    identical to ``RSqrtInt16ClzKernel``'s — measured under the P7 CLZ
    section in ``precision_validation.md``.
    """

    # pylint: disable=import-outside-toplevel
    from aimet_torch.fixed_point.kernels.clz_lut import _clz_int16_forward

    qin = rsqrt_lut_body["quantization"]["input"]
    qout = rsqrt_lut_body["quantization"]["output"]

    in_scale = float(qin["scale"])
    in_zp = int(qin["zero_point"])
    in_qmin = int(qin["min"])
    in_qmax = int(qin["max"])
    out_scale = float(qout["scale"])
    out_zp = int(qout["zero_point"])
    out_qmin = int(qout["min"])
    out_qmax = int(qout["max"])

    device = value_float.device
    # Clamp to [0, in_fmax] so the LUT special-case ``x ≤ 0 → out_qmax``
    # branch is never silently triggered by negative round-half noise on
    # ``var + eps`` (theoretically var ≥ 0 + positive eps, but watch out).
    in_fmax = in_qmax * in_scale
    clamped = value_float.clamp(min=0.0, max=in_fmax)
    q_input = torch.round(clamped / in_scale + in_zp)
    q_input = q_input.clamp(in_qmin, in_qmax).to(torch.int32)

    in_tensor = Int16QuantizedTensor(
        int_repr=q_input,
        scale=torch.tensor(in_scale, dtype=torch.float32, device=device),
        zero_point=torch.tensor(in_zp, dtype=torch.int32, device=device),
        qmin=in_qmin,
        qmax=in_qmax,
    )
    out_enc = OutputEncoding(
        scale=torch.tensor(out_scale, dtype=torch.float32, device=device),
        zero_point=torch.tensor(out_zp, dtype=torch.int32, device=device),
        qmin=out_qmin,
        qmax=out_qmax,
    )
    out_tensor = _clz_int16_forward(
        [in_tensor], out_enc, {"clz_lut": rsqrt_lut_body}, "rsqrt"
    )
    # Dequantize: q_inv_float = (q_out - Z_out) * S_out
    return (
        (out_tensor.int_repr.to(torch.float32) - float(out_zp)) * out_scale
    )


# ---------------------------------------------------------------------------
# Spec §4.5.2 integer variance instruction + spec §4.5.4 line 422-428
# integer M/rshift affine. These helpers are reused by ``LayerNormInt16Kernel``
# below and exposed for unit-test direct invocation. See module docstring
# for the four "Default A" quantization conventions they assume.
# ---------------------------------------------------------------------------


def _compute_inv_n_shift_n(n: int, *, multiplier_bits: int = 16) -> Tuple[int, int]:
    """Spec §4.5.2 ``(inv_N, shift_N)`` — fixed-point reciprocal of ``N``.

    Picks ``shift_N`` so ``inv_N = round(2^shift_N / N)`` sits in
    ``[2^(multiplier_bits-1), 2^multiplier_bits)`` (i.e. occupies the full
    upper half of the multiplier range), giving the maximum precision
    while keeping ``inv_N`` within the declared multiplier width. The
    compiler is expected to follow the same convention so sim matches
    hardware bit-exact.
    """

    if n <= 0:
        raise ValueError(f"_compute_inv_n_shift_n: N must be positive, got {n}.")
    # Smallest shift such that 2^shift / N >= 2^(multiplier_bits-1).
    # i.e. shift >= multiplier_bits - 1 + log2(N).
    shift_n = max(0, (multiplier_bits - 1) + int(n - 1).bit_length())
    inv_n = (1 << shift_n) // n + (1 if (((1 << shift_n) % n) * 2 >= n) else 0)
    return int(inv_n), int(shift_n)


def _arithmetic_right_shift(value: torch.Tensor, shift: int) -> torch.Tensor:
    """``value >> shift`` rounded toward -inf (matches hardware ``>>`` semantics).

    Torch's ``//`` on negative integers also floors, but ``torch.bitwise_right_shift``
    has stricter typing requirements — we route through integer floor div by
    ``2**shift`` so the result matches spec line 422-428's ``>>`` semantics
    regardless of value sign.
    """

    if shift == 0:
        return value
    return value // (1 << shift)


def _inline_integer_square_mean(
    x_int_repr: torch.Tensor,
    zero_point_x: int,
    dims: Tuple[int, ...],
    *,
    scale_x: float,
    out_scale: float,
    out_zero_point: int,
    out_qmin: int,
    out_qmax: int,
) -> Tuple[torch.Tensor, int]:
    """Spec §4.5.1 ``square_mean`` base instruction (integer carrier).

    Implements the three steps of spec line 67-75:

      1. ``s_o = Σ_{r ∈ R_A} (q_x − Z_x)²``                (int64 accumulator)
      2. ``m_o = (s_o · inv_N) >> shift_N``                (1/N reciprocal)
      3. ``q_sq = ((m_o · M_sq) >> rshift_sq) + Z_sq``     ``α_sq = S_x²/S_sq``

    Used by RMSNorm-style ops (CLN, §4.5.5) that only need ``mean(x²)``
    rather than the full mean+var of §4.5.2. ``(out_scale, out_zero_point,
    out_qmin, out_qmax)`` is the downstream grid that the caller picks
    (typically the RSqrt LUT input grid for CLN).

    Returns:
        ``(q_sq_on_out_grid, n_a)`` — ``q_sq_on_out_grid`` saturated to
        ``[out_qmin, out_qmax]`` on the ``(out_scale, out_zero_point)``
        grid. The reduce length ``n_a`` is also returned for callers that
        want to verify the ``inv_N`` selection.
    """

    device = x_int_repr.device
    centered = (x_int_repr.to(torch.int64) - int(zero_point_x))
    n_a = 1
    for d in dims:
        n_a *= int(x_int_repr.shape[d])
    inv_n, shift_n = _compute_inv_n_shift_n(n_a)

    sq_sum = (centered * centered).sum(dim=dims, keepdim=True)
    m_o = _arithmetic_right_shift(sq_sum * inv_n, shift_n)

    alpha_sq = (float(scale_x) * float(scale_x)) / float(out_scale)
    m_sq_t, rshift_sq_t = quantize_scale_to_m_rshift(alpha_sq)
    m_sq = int(m_sq_t.item())
    rshift_sq = int(rshift_sq_t.item())

    q_sq = _arithmetic_right_shift(m_o * m_sq, rshift_sq) + int(out_zero_point)
    q_sq = torch.clamp(q_sq, out_qmin, out_qmax)
    return q_sq.to(device=device, dtype=torch.int32), n_a


def _inline_integer_variance(
    x_int_repr: torch.Tensor,
    zero_point_x: int,
    dims: Tuple[int, ...],
    *,
    scale_x: float,
    var_input_scale: float,
    var_input_zero_point: int,
    var_input_qmin: int,
    var_input_qmax: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Spec §4.5.2 in-line ``variance`` base instruction (integer carrier).

    Implements the four steps of spec line 109-117:

      1. ``s_o = Σ_{r ∈ R_A} (q_x − Z_x)``            (int32 accumulator)
      2. ``q_μ − Z_μ = (s_o · inv_N) >> shift_N``     (S_μ = S_x convention)
      3. ``v_o = (Σ_{r ∈ R_A} d² · inv_N) >> shift_N`` (int64 carrier)
      4. ``q_var = ((v_o · M_var) >> rshift_var) + Z_var``  with
         ``M_var, rshift_var`` solved from ``S_x² / S_var``.

    The ``(S_var, Z_var)`` grid is fed in from the RSqrt LUT body so step
    4 lands ``q_var`` directly on the LUT input grid (zero downstream
    rescale).

    Returns:
        ``(q_mu_minus_zmu, q_var_on_lut_grid, n_a)``
        where ``q_mu_minus_zmu`` is centered on ``Z_μ = Z_x`` (so the
        caller can use it directly in ``q_x − q_μ`` subtractions) and
        ``q_var_on_lut_grid`` is saturated to ``[var_qmin, var_qmax]``.
    """

    device = x_int_repr.device
    centered = (x_int_repr.to(torch.int64) - int(zero_point_x))
    n_a = 1
    for d in dims:
        n_a *= int(x_int_repr.shape[d])
    inv_n, shift_n = _compute_inv_n_shift_n(n_a)

    # Stage 1: in-line mean. ``(s_o · inv_N) >> shift_N`` lives on grid
    # ``S_μ = S_x``, centered on Z_μ = Z_x (convention).
    s_o = centered.sum(dim=dims, keepdim=True)
    q_mu_minus_zmu = _arithmetic_right_shift(s_o * inv_n, shift_n)

    # Stage 2: squared diff mean. Use int64 throughout so the squared
    # sum cannot overflow even for N=64 with max-amplitude inputs.
    d_tensor = centered - q_mu_minus_zmu  # broadcast against keepdim shape
    sq_sum = (d_tensor * d_tensor).sum(dim=dims, keepdim=True)
    v_o = _arithmetic_right_shift(sq_sum * inv_n, shift_n)

    # Stage 3: requant v_o (S = S_x²) to var grid (S = var_input_scale).
    # α_var = S_x² / S_var ; (M_var, rshift_var) per quantize_scale_to_m_rshift.
    alpha_var = (float(scale_x) * float(scale_x)) / float(var_input_scale)
    m_var_t, rshift_var_t = quantize_scale_to_m_rshift(alpha_var)
    m_var = int(m_var_t.item())
    rshift_var = int(rshift_var_t.item())

    q_var = _arithmetic_right_shift(v_o * m_var, rshift_var) + int(var_input_zero_point)
    q_var = torch.clamp(q_var, var_input_qmin, var_input_qmax)
    return q_mu_minus_zmu.to(device=device), q_var.to(device=device, dtype=torch.int32), n_a


def _quantize_gamma_beta(
    gamma: Optional[torch.Tensor],
    beta: Optional[torch.Tensor],
    *,
    device: torch.device,
) -> Tuple[
    Optional[torch.Tensor],
    Optional[float],
    Optional[torch.Tensor],
    Optional[float],
]:
    """Offline quantize ``γ`` / ``β`` to int16 symmetric per-tensor (``Z = 0``).

    Default A: ``S_γ = γ.abs().max() / 32767``, ``S_β = β.abs().max() /
    32767``. Returns the int32 ``int_repr`` for each (None when the
    qmodule does not carry the parameter) plus the fp32 scale, which
    the kernel re-uses to fold ``α_x / α_μ / α_β`` into ``(M, rshift)``.
    """

    if gamma is not None:
        gamma_abs_max = float(gamma.abs().max().item())
        if gamma_abs_max == 0.0:
            scale_gamma = 1.0 / _LN_GAMMA_BETA_QMAX
        else:
            scale_gamma = gamma_abs_max / _LN_GAMMA_BETA_QMAX
        q_gamma = torch.round(gamma.to(torch.float32) / scale_gamma).to(torch.int32)
        q_gamma = q_gamma.clamp(-_LN_GAMMA_BETA_QMAX, _LN_GAMMA_BETA_QMAX).to(device)
    else:
        q_gamma, scale_gamma = None, None

    if beta is not None:
        beta_abs_max = float(beta.abs().max().item())
        if beta_abs_max == 0.0:
            scale_beta = 1.0 / _LN_GAMMA_BETA_QMAX
        else:
            scale_beta = beta_abs_max / _LN_GAMMA_BETA_QMAX
        q_beta = torch.round(beta.to(torch.float32) / scale_beta).to(torch.int32)
        q_beta = q_beta.clamp(-_LN_GAMMA_BETA_QMAX, _LN_GAMMA_BETA_QMAX).to(device)
    else:
        q_beta, scale_beta = None, None
    return q_gamma, scale_gamma, q_beta, scale_beta


def _compute_ln_m_rshift_pairs(
    *,
    scale_x: float,
    scale_gamma: Optional[float],
    scale_inv: float,
    scale_beta: Optional[float],
    scale_y: float,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Spec line 410-418 — fold ``α_x`` / ``α_β`` into ``(M, rshift)``.

      * ``α_x = S_γ · S_x · S_inv / S_y → (M_x, rshift_x)``
      * ``α_β = S_β / S_y               → (M_β, rshift_β)``

    Under ``S_μ = S_x`` (Default A) ``α_μ = α_x`` and ``(M_μ, rshift_μ)
    = (M_x, rshift_x)`` — the caller relies on that equivalence to fuse
    spec line 425-426's two terms into a single ``q_centered − q_μ``
    subtraction before the integer multiply. ``S_γ = 1`` is used when
    γ is absent (kernel slots ``q_γ = +1``); ``S_β = 0 → M_β = 0`` so
    the bias term vanishes when β is absent.
    """

    s_g = float(scale_gamma) if scale_gamma is not None else 1.0
    alpha_x = (s_g * float(scale_x) * float(scale_inv)) / float(scale_y)
    m_x_t, rshift_x_t = quantize_scale_to_m_rshift(alpha_x)
    m_x_pair = (int(m_x_t.item()), int(rshift_x_t.item()))

    if scale_beta is not None and scale_beta > 0.0:
        alpha_beta = float(scale_beta) / float(scale_y)
        m_b_t, rshift_b_t = quantize_scale_to_m_rshift(alpha_beta)
        m_b_pair = (int(m_b_t.item()), int(rshift_b_t.item()))
    else:
        m_b_pair = (0, 0)
    return m_x_pair, m_b_pair


def _integer_affine(
    *,
    q_x_centered: torch.Tensor,
    q_mu_centered: torch.Tensor,
    q_inv: torch.Tensor,
    q_gamma: Optional[torch.Tensor],
    q_beta: Optional[torch.Tensor],
    m_x: int,
    rshift_x: int,
    m_beta: int,
    rshift_beta: int,
    output_encoding: OutputEncoding,
    normalized_ndim: int,
) -> Int16QuantizedTensor:
    """Spec §4.5.4 line 422-428 integer affine, fused under ``S_μ = S_x``.

    Computes::

        q_y = ((q_γ · q_inv · ((q_x − Z_x) − (q_μ − Z_μ)) · M_x) >> rshift_x)
              + b^LN
        b^LN = ((q_β · M_β) >> rshift_β) + Z_y

    in int64 carrier, then saturates to ``[output.qmin, output.qmax]``.
    The ``q_γ = +1`` / ``q_β = 0`` slots cover ``elementwise_affine=False``
    without a second code path. ``q_γ`` / ``q_β`` are right-aligned with
    the last ``normalized_ndim`` dimensions and broadcast.
    """

    device = q_x_centered.device
    carrier_dtype = _LN_INTEGER_AFFINE_CARRIER_DTYPE
    centered = (q_x_centered - q_mu_centered).to(carrier_dtype)
    q_inv64 = q_inv.to(carrier_dtype)

    if q_gamma is not None:
        gamma = q_gamma.to(carrier_dtype)
        while gamma.ndim < centered.ndim:
            gamma = gamma.unsqueeze(0)
        product = gamma * q_inv64 * centered
    else:
        product = q_inv64 * centered  # q_γ = +1 slot

    term_x = _arithmetic_right_shift(product * int(m_x), int(rshift_x))

    z_y = int(output_encoding.zero_point.item()) if output_encoding.zero_point.numel() == 1 \
        else int(output_encoding.zero_point.flatten()[0].item())

    if q_beta is not None and m_beta > 0:
        beta = q_beta.to(carrier_dtype)
        while beta.ndim < centered.ndim:
            beta = beta.unsqueeze(0)
        b_ln = _arithmetic_right_shift(beta * int(m_beta), int(rshift_beta)) + z_y
        q_y = term_x + b_ln
    else:
        q_y = term_x + z_y

    q_y = torch.clamp(q_y, output_encoding.qmin, output_encoding.qmax)
    return Int16QuantizedTensor(
        int_repr=q_y.to(device=device, dtype=SIM_TENSOR_DTYPE),
        scale=output_encoding.scale.to(device=device),
        zero_point=output_encoding.zero_point.to(device=device, dtype=torch.int32),
        qmin=output_encoding.qmin,
        qmax=output_encoding.qmax,
        axis=output_encoding.axis,
    )


def _resolve_lut_input_qparams(
    rsqrt_lut_body: Dict[str, Any],
) -> Tuple[float, int, int, int]:
    """Pull ``(S_var, Z_var, qmin, qmax)`` from the RSqrt LUT input grid."""

    qin = rsqrt_lut_body["quantization"]["input"]
    return (
        float(qin["scale"]),
        int(qin["zero_point"]),
        int(qin["min"]),
        int(qin["max"]),
    )


def _q_inv_int_from_qvar(
    q_var: torch.Tensor,
    rsqrt_lut_body: Dict[str, Any],
) -> Tuple[torch.Tensor, float, int]:
    """Run ``q_inv = LUT(q_var)`` purely in integers (no fp32 round-trip).

    Wraps ``_clz_int16_forward`` directly with the var int_repr (already
    on the LUT input grid by construction in
    ``_inline_integer_variance``). Returns ``(q_inv_int32, S_inv, Z_inv)``
    so the affine step can fold ``S_inv`` into ``(M_x, rshift_x)``.
    """

    # pylint: disable=import-outside-toplevel
    from aimet_torch.fixed_point.kernels.clz_lut import _clz_int16_forward

    qin = rsqrt_lut_body["quantization"]["input"]
    qout = rsqrt_lut_body["quantization"]["output"]
    in_scale = float(qin["scale"])
    in_zp = int(qin["zero_point"])
    in_qmin = int(qin["min"])
    in_qmax = int(qin["max"])
    out_scale = float(qout["scale"])
    out_zp = int(qout["zero_point"])
    out_qmin = int(qout["min"])
    out_qmax = int(qout["max"])

    device = q_var.device
    in_tensor = Int16QuantizedTensor(
        int_repr=q_var.to(torch.int32),
        scale=torch.tensor(in_scale, dtype=torch.float32, device=device),
        zero_point=torch.tensor(in_zp, dtype=torch.int32, device=device),
        qmin=in_qmin,
        qmax=in_qmax,
    )
    out_enc = OutputEncoding(
        scale=torch.tensor(out_scale, dtype=torch.float32, device=device),
        zero_point=torch.tensor(out_zp, dtype=torch.int32, device=device),
        qmin=out_qmin,
        qmax=out_qmax,
    )
    out_tensor = _clz_int16_forward(
        [in_tensor], out_enc, {"clz_lut": rsqrt_lut_body}, "rsqrt"
    )
    # q_inv − Z_inv lives on (S_inv) — return centered int32 for the affine
    # carrier (so the centering step inside _integer_affine is implicit).
    return (out_tensor.int_repr.to(torch.int32) - out_zp), out_scale, out_zp


@register_fixed_kernel(nn.LayerNorm)
class LayerNormInt16Kernel:
    """Spec § 4.5.4 LayerNorm — **full integer bit-parity pipeline**.

    Pipeline (see module docstring for per-step details):

      1. ``q_μ`` / ``q_var`` via :func:`_inline_integer_variance`
         (spec § 4.5.2).
      2. ``q_inv`` via the RSqrt CLZ LUT (spec § 4.5.4 line 384).
      3. integer M/rshift affine (spec § 4.5.4 line 422-428).
      4. saturate to ``output_encoding``.

    Contract:

    * ``inputs``: exactly one ``Int16QuantizedTensor``; shape =
      ``prefix_shape + normalized_shape``.
    * ``params``: ignored (γ / β arrive via ``extra``).
    * ``output_encoding``: per-tensor only.
    * ``extra``:
        * ``normalized_shape`` (required): tuple/int per
          ``nn.LayerNorm.normalized_shape``.
        * ``eps`` (default 1e-5): added before the rsqrt (sim-side
          adjustment of ``q_var`` on the LUT input grid).
        * ``weight`` (γ, optional fp32 tensor): None when
          ``elementwise_affine=False``.
        * ``bias`` (β, optional fp32 tensor): None when
          ``elementwise_affine=False``.
        * ``rsqrt_clz_lut`` (optional): pre-loaded LUT JSON body. When
          absent, the default abc-tree asset is loaded.
    """

    module_type = nn.LayerNorm

    def __call__(
        self,
        inputs: List[Int16QuantizedTensor],
        params: Dict[str, Any],
        output_encoding: OutputEncoding,
        extra: Dict[str, Any],
    ) -> Int16QuantizedTensor:
        del params
        if len(inputs) != 1:
            raise ValueError(
                f"LayerNorm expects 1 input; got {len(inputs)}."
            )

        normalized_shape = _normalized_shape_from_extra(extra)
        eps = float(extra.get("eps", 1e-5))
        gamma = extra.get("weight")
        beta = extra.get("bias")
        x_q = inputs[0]
        device = x_q.int_repr.device

        scale_x = float(x_q.scale.item())
        zp_x = int(x_q.zero_point.item())
        dims = tuple(
            range(x_q.int_repr.ndim - len(normalized_shape), x_q.int_repr.ndim)
        )

        rsqrt_lut_body = _resolve_rsqrt_lut_body(extra)
        var_scale, var_zp, var_qmin, var_qmax = _resolve_lut_input_qparams(rsqrt_lut_body)

        # === Step 1: in-line integer variance (spec § 4.5.2) ===
        # Note on eps: spec applies eps on the float side as ``σ² + ε``;
        # in the integer carrier we add the equivalent ``q_eps =
        # round(ε / S_var)`` to ``q_var`` before LUT lookup. This is the
        # nearest-integer rounding for the (S_var) grid; both ``q_var``
        # and ``q_eps`` are clamped jointly to ``[var_qmin, var_qmax]``.
        q_mu_centered, q_var, _n_a = _inline_integer_variance(
            x_q.int_repr.to(device=device, dtype=torch.int64),
            zp_x,
            dims,
            scale_x=scale_x,
            var_input_scale=var_scale,
            var_input_zero_point=var_zp,
            var_input_qmin=var_qmin,
            var_input_qmax=var_qmax,
        )
        if eps > 0.0:
            q_eps = int(round(eps / var_scale))
            if q_eps != 0:
                q_var = torch.clamp(q_var + q_eps, var_qmin, var_qmax)

        # === Step 2: q_inv = LUT(q_var) — integer in/out ===
        q_inv_centered, scale_inv, _zp_inv = _q_inv_int_from_qvar(q_var, rsqrt_lut_body)

        # === Step 3: integer M/rshift affine (spec line 422-428) ===
        q_gamma, scale_gamma, q_beta, scale_beta = _quantize_gamma_beta(
            gamma, beta, device=device
        )
        scale_y = float(output_encoding.scale.item())
        (m_x, rshift_x), (m_beta, rshift_beta) = _compute_ln_m_rshift_pairs(
            scale_x=scale_x,
            scale_gamma=scale_gamma,
            scale_inv=scale_inv,
            scale_beta=scale_beta,
            scale_y=scale_y,
        )

        q_x_centered = (x_q.int_repr.to(torch.int64) - zp_x)
        return _integer_affine(
            q_x_centered=q_x_centered,
            q_mu_centered=q_mu_centered,
            q_inv=q_inv_centered,
            q_gamma=q_gamma,
            q_beta=q_beta,
            m_x=m_x,
            rshift_x=rshift_x,
            m_beta=m_beta,
            rshift_beta=rshift_beta,
            output_encoding=output_encoding,
            normalized_ndim=len(normalized_shape),
        )


def layer_norm_float_reference(
    inputs: List[Int16QuantizedTensor],
    params: Dict[str, Any],
    output_encoding: OutputEncoding,
    extra: Dict[str, Any],
) -> Int16QuantizedTensor:
    """Pure-float LayerNorm reference (dequant → F.layer_norm → requant).

    **Not registered** — exposed for unit tests as a precision oracle to
    quantify the RSqrt-CLZ-LUT residual introduced by the spec-aligned
    kernel above. The previous ``FU-LAYERNORM-FIRST-USE`` closure used
    this function as the registered kernel; the user-driven upgrade to
    spec §4.5.4 alignment (commit 2026-06-09) moved it out of the
    dispatch path.

    Mirrors the registered kernel's ``__call__`` signature so tests can
    swap the two and diff their outputs.
    """

    del params
    if len(inputs) != 1:
        raise ValueError(f"LayerNorm expects 1 input; got {len(inputs)}.")
    x_float = _dequant_to_float(inputs[0])
    normalized_shape = _normalized_shape_from_extra(extra)
    eps = float(extra.get("eps", 1e-5))
    gamma = extra.get("weight")
    beta = extra.get("bias")
    if gamma is not None:
        gamma = gamma.to(device=x_float.device, dtype=torch.float32)
    if beta is not None:
        beta = beta.to(device=x_float.device, dtype=torch.float32)
    y_float = F.layer_norm(
        x_float,
        normalized_shape=normalized_shape,
        weight=gamma,
        bias=beta,
        eps=eps,
    )
    return _requantize_from_float(y_float, output_encoding)


def layer_norm_lut_fp32_affine_reference(
    inputs: List[Int16QuantizedTensor],
    params: Dict[str, Any],
    output_encoding: OutputEncoding,
    extra: Dict[str, Any],
) -> Int16QuantizedTensor:
    """LUT-integer + fp32 variance + fp32 affine reference.

    **Not registered** — this is the previous spec-aligned reference path
    (variance fp32 sim → RSqrt CLZ LUT integer → fp32 multiply-add affine
    → fp32 final requant). Retained as the second precision oracle so
    tests can isolate the residual that the new full-integer pipeline
    picks up *on top of* the LUT step (i.e. the integer-variance +
    integer-affine round-half budget). The registered kernel above
    closes spec full bit-parity by replacing this function's fp32
    variance / affine with their integer counterparts.

    Mirrors the registered kernel's ``__call__`` signature so tests can
    swap the three (this, ``layer_norm_float_reference``, the registered
    integer kernel) and diff their outputs.
    """

    del params
    if len(inputs) != 1:
        raise ValueError(f"LayerNorm expects 1 input; got {len(inputs)}.")
    x_float = _dequant_to_float(inputs[0])
    normalized_shape = _normalized_shape_from_extra(extra)
    eps = float(extra.get("eps", 1e-5))
    gamma = extra.get("weight")
    beta = extra.get("bias")
    dims = tuple(range(x_float.ndim - len(normalized_shape), x_float.ndim))
    if gamma is not None:
        gamma = gamma.to(device=x_float.device, dtype=torch.float32)
    if beta is not None:
        beta = beta.to(device=x_float.device, dtype=torch.float32)
    mu = x_float.mean(dim=dims, keepdim=True)
    centered = x_float - mu
    var = (centered * centered).mean(dim=dims, keepdim=True)
    rsqrt_lut_body = _resolve_rsqrt_lut_body(extra)
    q_inv_float = _rsqrt_via_clz_lut(var + eps, rsqrt_lut_body)
    normalized = centered * q_inv_float
    if gamma is not None:
        normalized = normalized * gamma
    if beta is not None:
        normalized = normalized + beta
    return _requantize_from_float(normalized, output_encoding)


# ---------------------------------------------------------------------------
# Spec §4.5.1 / §4.5.2 base-instruction public API.
#
# These two thin aliases expose the in-line integer reduce helpers as the
# canonical spec "base instruction" entry points so unit tests, future
# kernels (RMSNorm / CLN single-op path / cfLN2D DSP-parity path) can
# call them directly without reaching into module-private symbols.
#
# They are intentionally **not** registered to ``register_fixed_kernel``:
# spec §4.5.1 / §4.5.2 describe DSP base instructions, not user-facing
# ``nn.Module`` types. Project ops that need them today (LayerNorm, and
# any future ``FU-NORM-SUBOP-VS-DSP-PARITY`` rewrite of CLN/cfLN2D)
# embed them in their own ``__call__`` path; tests call them on
# hand-built ``Int16QuantizedTensor`` inputs to assert bit-parity vs
# the fp32 reference of each base instruction.
# ---------------------------------------------------------------------------


def integer_square_mean_base_instruction(
    x_int_repr: torch.Tensor,
    zero_point_x: int,
    dims: Tuple[int, ...],
    *,
    scale_x: float,
    out_scale: float,
    out_zero_point: int,
    out_qmin: int,
    out_qmax: int,
) -> Tuple[torch.Tensor, int]:
    """Spec §4.5.1 ``square_mean`` base instruction — public API.

    Thin wrapper around :func:`_inline_integer_square_mean` exposing the
    spec §4.5.1 base instruction (one-pass integer reduce of ``Σ(q_x −
    Z_x)² · inv_N >> shift_N`` rescaled to ``(out_scale, out_zero_point)``).

    See module docstring for the four "Default A" quantization
    conventions inherited from the LayerNorm reference path.
    """

    return _inline_integer_square_mean(
        x_int_repr,
        zero_point_x,
        dims,
        scale_x=scale_x,
        out_scale=out_scale,
        out_zero_point=out_zero_point,
        out_qmin=out_qmin,
        out_qmax=out_qmax,
    )


def integer_variance_base_instruction(
    x_int_repr: torch.Tensor,
    zero_point_x: int,
    dims: Tuple[int, ...],
    *,
    scale_x: float,
    var_input_scale: float,
    var_input_zero_point: int,
    var_input_qmin: int,
    var_input_qmax: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Spec §4.5.2 ``variance`` base instruction — public API.

    Thin wrapper around :func:`_inline_integer_variance` exposing the
    spec §4.5.2 base instruction (two-pass integer reduce of mean +
    squared-diff-mean, rescaled to ``(var_input_scale, var_input_zero_point)``).

    Returns ``(q_mu_minus_zmu, q_var_on_out_grid, n_a)``. Under the
    ``S_μ = S_x`` / ``Z_μ = Z_x`` Default A convention the centered
    mean is directly usable in ``q_x − q_μ`` subtractions; the variance
    output is saturated to ``[var_input_qmin, var_input_qmax]`` on the
    chosen downstream grid.
    """

    return _inline_integer_variance(
        x_int_repr,
        zero_point_x,
        dims,
        scale_x=scale_x,
        var_input_scale=var_input_scale,
        var_input_zero_point=var_input_zero_point,
        var_input_qmin=var_input_qmin,
        var_input_qmax=var_input_qmax,
    )

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
"""Offline conversion from float scale to (m_int16, rshift) for fixed_scale_qdq."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

from aimet_torch.fixed_point.encoding import FixedScaleEncoding
from aimet_torch.fixed_point.offline.multiplier import _quantize_scalar_multiplier

_FIXED_SCALE_CACHE_ATTR = "_aimet_rx_fixed_scale_encoding"


def _normalize_scalar_m_rshift(m: int, r: int, *, max_rshift: int) -> Tuple[int, int]:
    """Fold ``(m, r)`` so ``r <= max_rshift`` while keeping ``m`` in int16 range."""

    while r > max_rshift and m > 1:
        m = (m + 1) >> 1
        r -= 1
    if r > max_rshift:
        r = max_rshift
        m = max(1, m)
    return m, r


def _quantize_scalar_scale(
    real_scale: float,
    multiplier_bits: int = 15,
    max_rshift: int = 31,
) -> Tuple[int, int]:
    """Scalar scale → ``(m, r)``; normalize when frexp needs ``rshift > max_rshift``."""

    # Boundary scale fold is spec 15 (fixed_scale_qdq), distinct from layer-requant
    # fold (spec 10 §53). Use saturate=False so we don't pollute the
    # ``record_multiplier_saturations`` log; the local try/except path expresses
    # boundary-scale semantics explicitly.
    #
    # _quantize_scalar_multiplier raises ValueError in two cases:
    #   (a) rshift > max_rshift -- real_scale too small; recoverable by widening
    #       max_rshift then half-up folding back to the caller's limit.
    #   (b) rshift < 0          -- real_scale >= 2**multiplier_bits; this is a
    #       genuine config error (output scale smaller than accumulator scale)
    #       that we must surface rather than silently saturate.
    try:
        return _quantize_scalar_multiplier(
            real_scale, multiplier_bits, max_rshift, saturate=False
        )
    except ValueError:
        if real_scale >= float(1 << multiplier_bits):
            # Case (b): widening max_rshift cannot fix rshift<0; re-raise.
            raise
        m, r = _quantize_scalar_multiplier(
            real_scale, multiplier_bits, 63, saturate=False
        )
        return _normalize_scalar_m_rshift(m, r, max_rshift=max_rshift)


def quantize_scale_to_m_rshift(
    scale: Union[float, torch.Tensor],
    multiplier_bits: int = 15,
    max_rshift: int = 31,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert positive scale(s) to ``(m_int16, rshift)`` with ``scale ≈ m / 2**rshift``."""

    if isinstance(scale, torch.Tensor):
        if torch.any(scale <= 0):
            raise ValueError("scale must be positive for quantize_scale_to_m_rshift.")
        flat = scale.detach().cpu().to(torch.float64).reshape(-1)
        multipliers = []
        rshifts = []
        for val in flat:
            m, r = _quantize_scalar_scale(float(val.item()), multiplier_bits, max_rshift)
            multipliers.append(m)
            rshifts.append(r)
        m_t = torch.tensor(multipliers, dtype=torch.int16).reshape(scale.shape)
        r_t = torch.tensor(rshifts, dtype=torch.int8).reshape(scale.shape)
        return m_t.to(device=scale.device), r_t.to(device=scale.device)

    if scale <= 0:
        raise ValueError("scale must be positive for quantize_scale_to_m_rshift.")
    m, r = _quantize_scalar_scale(float(scale), multiplier_bits, max_rshift)
    return torch.tensor(m, dtype=torch.int16), torch.tensor(r, dtype=torch.int8)


def fixed_scale_encoding_from_tensors(
    *,
    scale: torch.Tensor,
    offset: torch.Tensor,
    qmin: int,
    qmax: int,
    m_int16: Optional[torch.Tensor] = None,
    rshift: Optional[torch.Tensor] = None,
    axis: Optional[int] = None,
) -> FixedScaleEncoding:
    """Build :class:`FixedScaleEncoding`; compute ``m_int16``/``rshift`` from ``scale`` if omitted."""

    scale_f = scale.detach().to(device=scale.device, dtype=torch.float32)
    if m_int16 is None or rshift is None:
        m_gen, r_gen = quantize_scale_to_m_rshift(scale_f)
        m_int16 = m_gen if m_int16 is None else m_int16
        rshift = r_gen if rshift is None else rshift
    zp = (-offset.detach()).round().to(torch.int32)
    return FixedScaleEncoding(
        m_int16=m_int16.to(device=scale.device),
        rshift=rshift.to(device=scale.device),
        zero_point=zp.to(device=scale.device),
        qmin=int(qmin),
        qmax=int(qmax),
        axis=axis,
        scale_fp_legacy=scale_f,
    )


def fixed_scale_float_scale(
    m_int16: torch.Tensor,
    rshift: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Recover ``scale ≈ m / 2**r`` for carrier metadata and debug dequant."""

    m = m_int16.to(device=device, dtype=torch.float32)
    r = rshift.to(device=device, dtype=torch.float32)
    return m / torch.pow(2.0, r)


def fixed_scale_encoding_from_affine(encoding) -> FixedScaleEncoding:
    """Build :class:`FixedScaleEncoding` from v2 :class:`~aimet_torch.v2.quantization.affine.encoding.AffineEncoding`."""

    return fixed_scale_encoding_from_tensors(
        scale=encoding.scale,
        offset=encoding.offset,
        qmin=encoding.qmin,
        qmax=encoding.qmax,
    )


def get_or_create_fixed_scale_encoding(encoding) -> FixedScaleEncoding:
    """Return cached :class:`FixedScaleEncoding` for an ``AffineEncoding``."""

    cached = getattr(encoding, _FIXED_SCALE_CACHE_ATTR, None)
    if cached is not None:
        return cached
    fixed = fixed_scale_encoding_from_affine(encoding)
    setattr(encoding, _FIXED_SCALE_CACHE_ATTR, fixed)
    return fixed


def clear_fixed_scale_encoding_cache(encoding) -> None:
    """Drop cached fixed-scale encoding (e.g. after recalibration)."""

    if hasattr(encoding, _FIXED_SCALE_CACHE_ATTR):
        delattr(encoding, _FIXED_SCALE_CACHE_ATTR)


def _iter_quantizers_on_module(module) -> list:
    found = []
    for attr in ("input_quantizers", "output_quantizers"):
        qs = getattr(module, attr, None)
        if qs:
            found.extend(q for q in qs if q is not None)
    pq = getattr(module, "param_quantizers", None)
    if isinstance(pq, dict):
        found.extend(q for q in pq.values() if q is not None)
    return found


def convert_encodings_to_fixed_scale(sim) -> int:
    """Precompute and cache :class:`FixedScaleEncoding` on all initialized affine encodings."""

    # pylint: disable=import-outside-toplevel
    from aimet_torch.v2.quantization.affine.encoding import AffineEncoding
    from aimet_torch.v2.quantization.base import QuantizerBase

    count = 0
    for module in sim.model.modules():
        for quantizer in _iter_quantizers_on_module(module):
            if not isinstance(quantizer, QuantizerBase) or not quantizer.is_initialized():
                continue
            enc = quantizer.get_encodings()
            if isinstance(enc, AffineEncoding):
                get_or_create_fixed_scale_encoding(enc)
                count += 1
    return count

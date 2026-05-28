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
"""Offline conversion from floating scale to int16 multiplier + rshift.

The hardware requantize instruction is fixed to ``M_int16`` and ``rshift in
[0, max_rshift]`` (Ada200 ADR-001, spec 10 §52). When ``frexp`` of a small
``real_multiplier`` yields ``rshift > max_rshift`` the conversion is **folded**
in-place: ``M`` is halved (half-up) and ``rshift`` decremented until the
constraint holds (or ``M==1, rshift==max_rshift`` saturates the smallest
representable scale ``1 / 2**max_rshift``).

This matches what the compiler/toolchain emits before writing the deployable
sidecar — the runtime never sees ``rshift > 31``. Spec 10 §53 then requires the
event (and its relative error) to be recorded; use
:func:`record_multiplier_saturations` to capture it from caller scopes such as
``freeze_int16_fixed`` or the INT16 forward dispatch.
"""

from __future__ import annotations

import math
import os
import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Tuple, Union

import torch

from aimet_torch.fixed_point.requantize import MULTIPLIER_QBITS


_LOCAL = threading.local()

# Diagnostic-only env (spec 10 §53 reporting; never hardware-deployed):
# ``AIMET_RX_INT16_FOLD_MODE``
#   ``fold`` (default) — half-up fold described in spec §52.
#   ``zero``           — when ``rshift > max_rshift`` collapse the channel to
#                        ``(M=0, rshift=max_rshift)``. Output contribution from
#                        that requantize becomes ``zero_point`` only; useful to
#                        A/B-test "does dead-channel fold leakage explain the
#                        INT16 vs FIXED_SCALE_QDQ cosine gap?".
_FOLD_MODE_ENV = "AIMET_RX_INT16_FOLD_MODE"


def _current_fold_mode() -> str:
    return os.environ.get(_FOLD_MODE_ENV, "fold").strip().lower() or "fold"


def _fold_to_max_rshift(
    multiplier: int,
    rshift: int,
    *,
    max_rshift: int,
) -> Tuple[int, int]:
    """Fold ``(M, rshift)`` so ``rshift <= max_rshift`` (spec 10 §52).

    Halves ``multiplier`` (half-up rounding) per step; saturates at
    ``M = max(1, multiplier)`` when ``rshift`` reaches ``max_rshift`` with
    ``multiplier <= 1``. This is the single source of truth for the
    "rshift-overflow" fold used by both the layer-requant multiplier
    (this module) and the quantizer-boundary scale (``scale_fixed.py``).

    Honors the ``AIMET_RX_INT16_FOLD_MODE=zero`` diagnostic override (see
    module docstring) by collapsing the pair to ``(0, max_rshift)``.
    """

    if _current_fold_mode() == "zero":
        return 0, max_rshift

    while rshift > max_rshift and multiplier > 1:
        multiplier = (multiplier + 1) >> 1
        rshift -= 1
    if rshift > max_rshift:
        rshift = max_rshift
        multiplier = max(1, multiplier)
    return multiplier, rshift


def _record_saturation(real_multiplier: float, multiplier: int, rshift: int) -> None:
    """Append a saturation event to the active recorder (no-op when inactive)."""

    log = getattr(_LOCAL, "saturation_log", None)
    if log is None:
        return
    approx = float(multiplier) / float(1 << rshift) if rshift >= 0 else 0.0
    denom = max(abs(real_multiplier), 1e-30)
    rel_err = abs(real_multiplier - approx) / denom
    log.append(
        {
            "real_multiplier": float(real_multiplier),
            "multiplier": int(multiplier),
            "rshift": int(rshift),
            "approx": float(approx),
            "relative_error": float(rel_err),
        }
    )


@contextmanager
def record_multiplier_saturations() -> Iterator[List[Dict[str, Any]]]:
    """Capture layer-requant saturation events for the current thread (spec 10 §53).

    Each entry is a dict with ``real_multiplier`` / ``multiplier`` / ``rshift`` /
    ``approx`` / ``relative_error``. Entries originate from
    :func:`quantize_multiplier` when ``saturate=True`` (the default) folds the
    out-of-range ``rshift`` into ``[0, max_rshift]``.

    Example::

        with record_multiplier_saturations() as events:
            quantize_multiplier(torch.tensor([1e-9, 0.1]))
        # events = [{"real_multiplier": 1e-9, "multiplier": ..., "rshift": 31, ...}]
    """

    prev = getattr(_LOCAL, "saturation_log", None)
    log: List[Dict[str, Any]] = []
    _LOCAL.saturation_log = log
    try:
        yield log
    finally:
        _LOCAL.saturation_log = prev


def _quantize_scalar_multiplier(
    real_multiplier: float,
    multiplier_bits: int,
    max_rshift: int,
    *,
    saturate: bool = True,
) -> Tuple[int, int]:
    if not math.isfinite(real_multiplier):
        # Unify error type: inf would otherwise surface as OverflowError from
        # int(round(inf * ...)), which callers of this module do not catch.
        raise ValueError(
            f"real_multiplier must be finite (got {real_multiplier!r})."
        )
    if real_multiplier < 0:
        raise ValueError("real_multiplier must be non-negative.")
    if real_multiplier == 0:
        return 0, 0

    mantissa, exponent = math.frexp(real_multiplier)
    multiplier = int(round(mantissa * (1 << multiplier_bits)))
    rshift = multiplier_bits - exponent

    if multiplier == (1 << multiplier_bits):
        # round-up carry from mantissa near 1.0: rewrite (M=2^mb, r) as
        # (M=2^(mb-1), r-1); same scale, costs ~1 bit of mantissa precision.
        multiplier >>= 1
        rshift -= 1

    if not 0 <= multiplier < (1 << multiplier_bits):
        raise ValueError(f"multiplier {multiplier} is not representable.")
    if rshift < 0:
        # real_multiplier >= 1: caller has s_x * s_w >= s_y, which is unexpected
        # for layer requant. Surface it rather than silently saturate to rshift=0.
        raise ValueError(
            f"rshift {rshift} is negative (real_multiplier={real_multiplier:.6e}); "
            "this means the accumulator scale already exceeds the output scale."
        )
    if rshift > max_rshift:
        if not saturate:
            raise ValueError(f"rshift {rshift} is outside [0, {max_rshift}].")
        multiplier, rshift = _fold_to_max_rshift(
            multiplier, rshift, max_rshift=max_rshift
        )
        _record_saturation(real_multiplier, multiplier, rshift)

    return multiplier, rshift


def quantize_multiplier(
    real_multiplier: Union[float, torch.Tensor],
    multiplier_bits: int = MULTIPLIER_QBITS,
    max_rshift: int = 31,
    *,
    saturate: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert non-negative float multiplier(s) to int16 Q-format ``(M, rshift)``.

    ``saturate=True`` (default, spec 10 §53): when ``frexp`` yields
    ``rshift > max_rshift`` the pair is folded in-place (see
    :func:`_fold_to_max_rshift`) and the event is appended to any active
    :func:`record_multiplier_saturations` log. The deployable runtime sees
    only ``rshift <= max_rshift``; the recorded relative error tells callers
    which entries lost mantissa precision.

    ``saturate=False`` keeps the strict legacy behavior (raises on
    out-of-range), useful for debugging or strict CI gates.
    """

    if multiplier_bits <= 0 or multiplier_bits > 15:
        raise ValueError("multiplier_bits must be in [1, 15].")
    # rshift is returned as int8; reject ranges that would silently wrap.
    if not 0 <= max_rshift < 128:
        raise ValueError(
            f"max_rshift must be in [0, 127] (got {max_rshift}) to fit in int8."
        )

    if isinstance(real_multiplier, torch.Tensor):
        src_device = real_multiplier.device
        real_multiplier_tensor = real_multiplier.detach().cpu().to(torch.float64)
        if real_multiplier_tensor.dim() == 0:
            multiplier, rshift = _quantize_scalar_multiplier(
                float(real_multiplier_tensor.item()),
                multiplier_bits,
                max_rshift,
                saturate=saturate,
            )
            return (
                torch.tensor(multiplier, dtype=torch.int16, device=src_device),
                torch.tensor(rshift, dtype=torch.int8, device=src_device),
            )

        flat = real_multiplier_tensor.reshape(-1)
        n = int(flat.numel())
        flat_m = torch.empty(n, dtype=torch.int16)
        flat_r = torch.empty(n, dtype=torch.int8)
        for i in range(n):
            m_val, s_val = _quantize_scalar_multiplier(
                float(flat[i].item()),
                multiplier_bits,
                max_rshift,
                saturate=saturate,
            )
            flat_m[i] = m_val
            flat_r[i] = s_val
        return (
            flat_m.reshape(real_multiplier_tensor.shape).to(device=src_device),
            flat_r.reshape(real_multiplier_tensor.shape).to(device=src_device),
        )

    multiplier, rshift = _quantize_scalar_multiplier(
        float(real_multiplier),
        multiplier_bits,
        max_rshift,
        saturate=saturate,
    )
    return torch.tensor(multiplier, dtype=torch.int16), torch.tensor(
        rshift, dtype=torch.int8
    )

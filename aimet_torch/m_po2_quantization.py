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
"""Snap quantizer scales to hardware M/2^n grid (M may be > 1), then optional re-calib.

Unlike legacy ``apply_power_of_2_workflow`` (scale = 1/2^n only), **M_Po2** keeps the
best ``(M_int16, rshift)`` approximation from offline ``frexp`` rounding — the same
representation ``convert_encodings_to_fixed_scale`` emits, but written back into AIMET
float encodings before a second calibration pass.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

import torch

from aimet_torch.fixed_point.offline.scale_fixed import _quantize_scalar_scale
from aimet_torch.v2.nn import QuantizationMixin


def snap_scale_to_m_po2(
    scale: float,
    *,
    multiplier_bits: int = 16,
    max_rshift: int = 31,
) -> Tuple[float, int, int]:
    """Return ``(snapped_scale, M, rshift)`` with ``snapped_scale ≈ M / 2**rshift``."""

    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale!r}.")
    m, r = _quantize_scalar_scale(scale, multiplier_bits, max_rshift)
    return m / float(1 << r), m, r


def _snap_scale_tensor(
    scale: torch.Tensor,
    *,
    multiplier_bits: int = 16,
    max_rshift: int = 31,
) -> torch.Tensor:
    flat = scale.detach().flatten()
    snapped = []
    for val in flat:
        s, _, _ = snap_scale_to_m_po2(float(val.item()), multiplier_bits=multiplier_bits, max_rshift=max_rshift)
        snapped.append(s)
    out = torch.tensor(snapped, dtype=scale.dtype, device=scale.device)
    return out.view(scale.shape)


def _reparametrize_quantizer_to_new_scale(
    quantizer,
    new_scale: torch.Tensor,
    *,
    old_min,
    old_max,
    old_offset,
    qmin: int,
    qmax: int,
    symmetric: bool,
) -> None:
    """Keep qmin/qmax fixed; recompute min/max/offset for ``new_scale`` (Po2-style)."""

    if isinstance(new_scale, torch.Tensor):
        if symmetric or old_offset is None:
            new_qmin_val = qmin
            new_qmax_val = qmax
            new_offset = old_offset
            if new_offset is not None:
                new_min = new_scale * (new_qmin_val + new_offset)
                new_max = new_scale * (new_qmax_val + new_offset)
            else:
                new_min = new_scale * new_qmin_val
                new_max = new_scale * new_qmax_val
        else:
            new_qmin_val = qmin
            new_qmax_val = qmax
            new_offset = old_min / new_scale - qmin
            new_max = new_scale * (qmax + new_offset)
            new_min = old_min
    else:
        old_min_val = old_min.item() if hasattr(old_min, "item") else old_min
        old_max_val = old_max.item() if hasattr(old_max, "item") else old_max
        new_scale_val = new_scale.item() if hasattr(new_scale, "item") else new_scale
        if symmetric or old_offset is None:
            new_qmin_val = qmin
            new_qmax_val = qmax
            if old_offset is not None:
                old_offset_val = old_offset.item() if hasattr(old_offset, "item") else old_offset
                new_offset = old_offset
                new_min_val = new_scale_val * (new_qmin_val + old_offset_val)
                new_max_val = new_scale_val * (new_qmax_val + old_offset_val)
            else:
                new_offset = None
                new_min_val = new_scale_val * new_qmin_val
                new_max_val = new_scale_val * new_qmax_val
            new_min = torch.tensor(new_min_val, dtype=new_scale.dtype, device=new_scale.device)
            new_max = torch.tensor(new_max_val, dtype=new_scale.dtype, device=new_scale.device)
        else:
            new_qmin_val = qmin
            new_qmax_val = qmax
            new_offset_val = old_min_val / new_scale_val - qmin
            new_offset = torch.tensor(new_offset_val, dtype=new_scale.dtype, device=new_scale.device)
            new_max_val = new_scale_val * (qmax + new_offset_val)
            new_min = old_min
            new_max = torch.tensor(new_max_val, dtype=new_scale.dtype, device=new_scale.device)

    with torch.no_grad():
        quantizer.scale.copy_(new_scale if isinstance(new_scale, torch.Tensor) else torch.tensor(new_scale))
        quantizer.qmin = new_qmin_val
        quantizer.qmax = new_qmax_val
        if new_offset is not None and quantizer.offset is not None:
            quantizer.offset.copy_(new_offset)
        quantizer.set_range(new_min, new_max)


def modify_quantizer_to_m_po2(
    quantizer,
    quantizer_name: str,
    *,
    multiplier_bits: int = 16,
    max_rshift: int = 31,
    verbose: bool = False,
) -> bool:
    """Snap one quantizer's scale to the nearest M/2^n grid point."""

    try:
        old_min = quantizer.get_min()
        old_max = quantizer.get_max()
        old_scale = quantizer.get_scale()
        old_offset = quantizer.get_offset()
        qmin = quantizer.qmin
        qmax = quantizer.qmax
        symmetric = quantizer.symmetric

        quantizer._reparametrize_to_scale_offset()

        if isinstance(old_scale, torch.Tensor):
            new_scale = _snap_scale_tensor(
                old_scale, multiplier_bits=multiplier_bits, max_rshift=max_rshift,
            )
        else:
            old_scale_val = old_scale.item() if hasattr(old_scale, "item") else old_scale
            snapped, m, r = snap_scale_to_m_po2(
                float(old_scale_val), multiplier_bits=multiplier_bits, max_rshift=max_rshift,
            )
            new_scale = torch.tensor(snapped, dtype=getattr(old_scale, "dtype", torch.float32))
            if verbose:
                print(
                    f"  {quantizer_name}: scale {old_scale_val:.8g} → {snapped:.8g} "
                    f"(M={m}, rshift={r})"
                )

        _reparametrize_quantizer_to_new_scale(
            quantizer,
            new_scale,
            old_min=old_min,
            old_max=old_max,
            old_offset=old_offset,
            qmin=qmin,
            qmax=qmax,
            symmetric=symmetric,
        )
        return True
    except Exception as exc:
        if verbose:
            print(f"  ⚠️  {quantizer_name}: M_Po2 snap failed: {exc}")
        return False


def apply_m_po2_quantization(
    model,
    *,
    multiplier_bits: int = 16,
    max_rshift: int = 31,
    verbose: bool = False,
) -> Dict[str, int]:
    """Snap all initialized quantizer scales to M/2^n (M may be > 1)."""

    total = 0
    modified = 0
    for name, module in model.named_modules():
        if isinstance(module, QuantizationMixin) and hasattr(module, "input_quantizers"):
            for idx, quantizer in enumerate(module.input_quantizers):
                if quantizer is not None and quantizer.is_initialized():
                    total += 1
                    if modify_quantizer_to_m_po2(
                        quantizer,
                        f"{name}.input_quantizer[{idx}]",
                        multiplier_bits=multiplier_bits,
                        max_rshift=max_rshift,
                        verbose=verbose,
                    ):
                        modified += 1
        if isinstance(module, QuantizationMixin) and hasattr(module, "output_quantizers"):
            for idx, quantizer in enumerate(module.output_quantizers):
                if quantizer is not None and quantizer.is_initialized():
                    total += 1
                    if modify_quantizer_to_m_po2(
                        quantizer,
                        f"{name}.output_quantizer[{idx}]",
                        multiplier_bits=multiplier_bits,
                        max_rshift=max_rshift,
                        verbose=verbose,
                    ):
                        modified += 1
        if hasattr(module, "param_quantizers"):
            for param_name, quantizer in module.param_quantizers.items():
                if quantizer is not None and quantizer.is_initialized():
                    total += 1
                    if modify_quantizer_to_m_po2(
                        quantizer,
                        f"{name}.param_quantizer[{param_name}]",
                        multiplier_bits=multiplier_bits,
                        max_rshift=max_rshift,
                        verbose=verbose,
                    ):
                        modified += 1

    if verbose:
        print(f"M_Po2: modified {modified}/{total} quantizers")
    return {"total": total, "modified": modified}


def clear_sim_fixed_scale_caches(model) -> int:
    """Drop cached ``FixedScaleEncoding`` after encoding mutation."""

    from aimet_torch.fixed_point.offline.scale_fixed import clear_fixed_scale_encoding_cache
    from aimet_torch.v2.quantization.base import QuantizerBase

    cleared = 0
    for module in model.modules():
        for attr in ("input_quantizers", "output_quantizers"):
            if not hasattr(module, attr):
                continue
            for quantizer in getattr(module, attr):
                if quantizer is None or not isinstance(quantizer, QuantizerBase):
                    continue
                if not quantizer.is_initialized():
                    continue
                enc = quantizer.get_encodings()
                if enc is not None and hasattr(enc, "scale"):
                    clear_fixed_scale_encoding_cache(enc)
                    cleared += 1
        if hasattr(module, "param_quantizers"):
            for quantizer in module.param_quantizers.values():
                if quantizer is None or not isinstance(quantizer, QuantizerBase):
                    continue
                if not quantizer.is_initialized():
                    continue
                enc = quantizer.get_encodings()
                if enc is not None:
                    clear_fixed_scale_encoding_cache(enc)
                    cleared += 1
    return cleared


def apply_m_po2_recalib_workflow(
    model,
    calib_loader: Iterable,
    device: torch.device,
    *,
    max_calib_batches: int,
    multiplier_bits: int = 16,
    max_rshift: int = 31,
    verbose: bool = False,
) -> Dict[str, Any]:
    """M_Po2 snap → ``compute_encodings`` re-calib → M_Po2 re-snap."""

    import aimet_torch.v2 as aimet

    pre = apply_m_po2_quantization(
        model, multiplier_bits=multiplier_bits, max_rshift=max_rshift, verbose=verbose,
    )
    clear_sim_fixed_scale_caches(model)

    model.eval()
    with torch.no_grad(), aimet.nn.compute_encodings(model):
        for idx, (x, _) in enumerate(calib_loader):
            if idx >= max_calib_batches:
                break
            model(x.to(device))

    post = apply_m_po2_quantization(
        model, multiplier_bits=multiplier_bits, max_rshift=max_rshift, verbose=verbose,
    )
    cleared = clear_sim_fixed_scale_caches(model)

    return {
        "pre_snap": pre,
        "post_snap": post,
        "cache_cleared": cleared,
        "recalib_batches": max_calib_batches,
    }

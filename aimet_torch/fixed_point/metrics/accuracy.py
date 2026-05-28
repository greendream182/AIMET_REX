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
"""Tensor-level accuracy metrics and assertions for fixed-point vs reference outputs."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

from aimet_torch.fixed_point.metrics.flags import int16_eval_allow_debug_float
from aimet_torch.fixed_point.metrics.thresholds import (
    FP16_VS_FP32_MIN_COSINE_SIMILARITY,
    INT16_VS_FP32_MAX_ERROR_LSB,
    INT16_VS_FP32_MIN_COSINE_SIMILARITY,
    KERNEL_VS_FLOAT_MAX_ERROR_LSB,
)
from aimet_torch.fixed_point.requantize import SIM_TENSOR_DTYPE
from aimet_torch.fixed_point.tensor import Int16QuantizedTensor, align_stat_rank


def _as_float_tensor(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float().cpu()


def cosine_similarity(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> float:
    """Cosine similarity between flattened ``reference`` and ``candidate``."""

    ref = _as_float_tensor(reference).reshape(-1).numpy()
    cur = _as_float_tensor(candidate).reshape(-1).numpy()
    if ref.shape != cur.shape:
        raise ValueError(
            f"Shape mismatch: reference {tuple(reference.shape)} vs candidate {tuple(candidate.shape)}."
        )
    denom = float(np.linalg.norm(ref) * np.linalg.norm(cur))
    if denom <= 0:
        return 1.0
    return float(np.dot(ref, cur) / denom)


# ``torch.quantile`` rejects very large 1-D inputs on some builds; subsample for diagnostics.
_MAX_QUANTILE_SAMPLES = 1_048_576


def p99_abs_error(diff: torch.Tensor) -> float:
    """99th percentile of ``|diff|``, with deterministic subsampling when ``diff`` is huge."""

    if diff.numel() == 0:
        return 0.0
    flat = diff.abs().reshape(-1)
    n = int(flat.numel())
    if n > _MAX_QUANTILE_SAMPLES:
        step = max(1, (n + _MAX_QUANTILE_SAMPLES - 1) // _MAX_QUANTILE_SAMPLES)
        flat = flat[::step][:_MAX_QUANTILE_SAMPLES]
    return float(flat.quantile(0.99).item())


def _broadcast_scale_zero_point(
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    like: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    scale_f = align_stat_rank(scale.to(device=like.device, dtype=torch.float32), like)
    zp_f = align_stat_rank(zero_point.to(device=like.device, dtype=torch.float32), like)
    return scale_f, zp_f


def quantize_float_to_grid(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    qmin: int,
    qmax: int,
) -> torch.Tensor:
    """Round float ``tensor`` onto the INT16 grid defined by ``scale`` / ``zero_point``."""

    scale_b, zp_b = _broadcast_scale_zero_point(scale, zero_point, tensor)
    q = torch.round(tensor.to(torch.float32) / scale_b + zp_b)
    return torch.clamp(q, qmin, qmax)


def max_error_lsb_int(
    reference: torch.Tensor,
    candidate_int_repr: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    qmin: int,
    qmax: int,
) -> float:
    """Maximum integer-grid error ``|q_candidate - q_ref|`` with ``q_ref`` from ``reference``."""

    q_ref = quantize_float_to_grid(reference, scale, zero_point, qmin, qmax)
    q_cur = candidate_int_repr.to(torch.int32)
    if q_ref.shape != q_cur.shape:
        raise ValueError(
            f"Shape mismatch: reference {tuple(reference.shape)} vs "
            f"candidate_int {tuple(candidate_int_repr.shape)}."
        )
    return float(torch.max(torch.abs(q_cur - q_ref.to(torch.int32))).item())


def max_error_lsb_float(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    qmin: int,
    qmax: int,
) -> float:
    """Maximum dequant error in LSB units: ``|y - y_ref| / scale`` (uses output grid)."""

    scale_b, _ = _broadcast_scale_zero_point(scale, zero_point, reference)
    if torch.any(scale_b == 0):
        raise ValueError("scale must not contain zero.")
    ref = reference.to(torch.float32)
    cur = candidate.to(torch.float32)
    if ref.shape != cur.shape:
        raise ValueError(
            f"Shape mismatch: reference {tuple(reference.shape)} vs candidate {tuple(candidate.shape)}."
        )
    err = torch.abs(cur - ref) / scale_b
    return float(torch.max(err).item())


def compute_pair_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    scale: Optional[torch.Tensor] = None,
    zero_point: Optional[torch.Tensor] = None,
    qmin: int = -32768,
    qmax: int = 32767,
    candidate_int_repr: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Compute standard comparison metrics for one tensor pair."""

    metrics: Dict[str, float] = {}
    ref = _as_float_tensor(reference)
    cur = _as_float_tensor(candidate)
    diff = cur - ref
    metrics["max_abs_error"] = float(torch.max(torch.abs(diff)).item())
    metrics["rmse"] = float(torch.sqrt(torch.mean(diff * diff)).item())
    metrics["cosine_similarity"] = cosine_similarity(reference, candidate)
    noise = float(torch.mean(diff * diff).item())
    signal = float(torch.mean(ref * ref).item())
    metrics["sqnr_db"] = float(10.0 * np.log10(signal / noise)) if noise > 0 else float("inf")

    if scale is not None and zero_point is not None:
        metrics["max_error_lsb_float"] = max_error_lsb_float(
            reference, candidate, scale, zero_point, qmin, qmax
        )
        if candidate_int_repr is not None:
            metrics["max_error_lsb"] = max_error_lsb_int(
                reference,
                candidate_int_repr,
                scale,
                zero_point,
                qmin,
                qmax,
            )
        else:
            q_cur = quantize_float_to_grid(candidate, scale, zero_point, qmin, qmax)
            metrics["max_error_lsb"] = max_error_lsb_int(
                reference, q_cur.to(SIM_TENSOR_DTYPE), scale, zero_point, qmin, qmax
            )
    return metrics


def assert_min_cosine_similarity(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    min_cosine: float,
    *,
    label: str = "output",
) -> None:
    actual = cosine_similarity(reference, candidate)
    if actual + 1e-12 < min_cosine:
        raise AssertionError(
            f"{label} cosine_similarity {actual:.6f} < required {min_cosine:.6f}."
        )


def assert_max_error_lsb(
    reference: torch.Tensor,
    candidate: Union[torch.Tensor, Int16QuantizedTensor],
    max_lsb: float,
    *,
    scale: Optional[torch.Tensor] = None,
    zero_point: Optional[torch.Tensor] = None,
    qmin: int = -32768,
    qmax: int = 32767,
    label: str = "output",
    use_integer_grid: bool = True,
) -> None:
    """Assert dequant or integer-grid error is within ``max_lsb`` LSB."""

    if isinstance(candidate, Int16QuantizedTensor):
        scale = candidate.scale
        zero_point = candidate.zero_point
        qmin = candidate.qmin
        qmax = candidate.qmax
        with int16_eval_allow_debug_float():
            cur_float = candidate.to_float()
        cur_int = candidate.int_repr
    elif isinstance(candidate, torch.Tensor) and candidate.dtype == torch.int16:
        if scale is None or zero_point is None:
            raise ValueError("scale and zero_point are required for int16 Tensor candidates.")
        cur_int = candidate
        scale_b, zp_b = _broadcast_scale_zero_point(scale, zero_point, reference)
        cur_float = (candidate.to(torch.int32) - zp_b.to(torch.int32)).to(torch.float32) * scale_b
    else:
        if scale is None or zero_point is None:
            raise ValueError("scale and zero_point are required when candidate is a Tensor.")
        cur_float = candidate
        cur_int = None

    if use_integer_grid and cur_int is not None:
        actual = max_error_lsb_int(reference, cur_int, scale, zero_point, qmin, qmax)
        metric_name = "max_error_lsb"
    else:
        actual = max_error_lsb_float(reference, cur_float, scale, zero_point, qmin, qmax)
        metric_name = "max_error_lsb_float"

    if actual > max_lsb + 1e-6:
        raise AssertionError(
            f"{label} {metric_name} {actual:.4f} > allowed {max_lsb:.4f} LSB."
        )


def assert_int16_vs_fp32_reference(
    quantized: Int16QuantizedTensor,
    reference: torch.Tensor,
    *,
    max_lsb: float = INT16_VS_FP32_MAX_ERROR_LSB,
    min_cosine: float = INT16_VS_FP32_MIN_COSINE_SIMILARITY,
    label: str = "int16_fixed_eval vs fp32_qdq",
) -> None:
    """Assert INT16 fixed eval matches FP32 QDQ within LSB and cosine gates."""

    with int16_eval_allow_debug_float():
        candidate = quantized.to_float()
    assert_max_error_lsb(
        reference,
        quantized,
        max_lsb,
        label=label,
        use_integer_grid=True,
    )
    assert_min_cosine_similarity(reference, candidate, min_cosine, label=label)


def assert_fp16_vs_fp32_reference(
    fp16_output: torch.Tensor,
    fp32_reference: torch.Tensor,
    *,
    min_cosine: float = FP16_VS_FP32_MIN_COSINE_SIMILARITY,
    label: str = "fp16_qdq vs fp32_qdq",
) -> None:
    """Assert FP16 QDQ path matches FP32 QDQ (cosine only — no LSB gate)."""

    assert_min_cosine_similarity(fp32_reference, fp16_output, min_cosine, label=label)


def assert_kernel_vs_float_reference(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    output_scale: torch.Tensor,
    output_zero_point: torch.Tensor,
    *,
    max_lsb: float = KERNEL_VS_FLOAT_MAX_ERROR_LSB,
    min_cosine: float = INT16_VS_FP32_MIN_COSINE_SIMILARITY,
    qmin: int = 0,
    qmax: int = 32767,
    label: str = "kernel vs float",
    use_integer_grid: bool = True,
) -> None:
    """Assert kernel output matches float reference on the output quantization grid."""

    assert_max_error_lsb(
        reference,
        candidate,
        max_lsb,
        scale=output_scale,
        zero_point=output_zero_point,
        qmin=qmin,
        qmax=qmax,
        label=label,
        use_integer_grid=use_integer_grid,
    )
    if candidate.dtype == torch.int16:
        scale_b, zp_b = _broadcast_scale_zero_point(
            output_scale, output_zero_point, reference
        )
        candidate_float = (
            candidate.to(torch.int32) - zp_b.to(torch.int32)
        ).to(torch.float32) * scale_b
    else:
        candidate_float = candidate
    assert_min_cosine_similarity(reference, candidate_float, min_cosine, label=label)

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
"""Offline LUT generation for INT16 fixed-point nonlinear kernels."""

import math
from typing import Any, Callable, Mapping, Optional

import numpy as np
import torch

from aimet_torch.fixed_point.encoding import InputEncoding, OutputEncoding
from aimet_torch.fixed_point.metrics.thresholds import (
    PWL_EXPORT_VALIDATION_SAMPLES,
    PWL_HARDWARE_NUM_SEGMENTS,
    PWL_VS_ANALYTIC_DEFAULT_LIMITS,
    PWL_VS_ANALYTIC_MIN_COSINE_SIMILARITY,
    PWL_VS_ANALYTIC_PER_FN_LIMITS,
)
from aimet_torch.fixed_point.offline.multiplier import quantize_multiplier
from aimet_torch.fixed_point.offline.scale_fixed import quantize_scale_to_m_rshift
from aimet_torch.fixed_point.requantize import (
    INT32_QMAX,
    INT32_QMIN,
    hw_ref_mode_enabled,
    requantize_int,
)

from aimet_torch.fixed_point.rounding import RoundingMode


def _effective_scale_fp64(scale: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Discrete scale ``m / 2^r`` used for PWL/LUT offline fit (ADR-009 / spec 10)."""

    m, r = quantize_scale_to_m_rshift(scale)
    m = m.to(device=device, dtype=torch.float64)
    r = r.to(device=device, dtype=torch.float64)
    divisor = torch.pow(
        torch.tensor(2.0, device=device, dtype=torch.float64),
        r,
    )
    return m / divisor


class PwlLutAccuracyError(RuntimeError):
    """Raised when optional strict quality check vs analytic ``fn`` fails."""


def resolve_pwl_quality_limits(
    fn_name: Optional[str],
    overrides: Optional[Mapping[str, float]] = None,
) -> dict[str, float]:
    """Return ``{max_lsb, p99_lsb, rmse_lsb, min_cosine_similarity}`` for ``fn_name``."""

    base = dict(PWL_VS_ANALYTIC_DEFAULT_LIMITS)
    base["min_cosine_similarity"] = PWL_VS_ANALYTIC_MIN_COSINE_SIMILARITY
    if fn_name:
        per_fn = PWL_VS_ANALYTIC_PER_FN_LIMITS.get(fn_name.lower())
        if per_fn:
            base.update(per_fn)
    if overrides:
        base.update(overrides)
    return base


def _validation_q_samples(
    input_encoding: InputEncoding,
    num_samples: int,
    device: torch.device,
) -> torch.Tensor:
    if num_samples < 2:
        raise ValueError("num_samples must be at least 2.")
    q = torch.linspace(
        input_encoding.qmin,
        input_encoding.qmax,
        num_samples,
        dtype=torch.float64,
        device=device,
    )
    return torch.round(q).to(torch.int16)


def measure_pwl_lut_metrics(
    fn: Callable[[torch.Tensor], torch.Tensor],
    input_encoding: InputEncoding,
    output_encoding: OutputEncoding,
    pwl_lut: dict[str, torch.Tensor],
    *,
    num_samples: int = PWL_EXPORT_VALIDATION_SAMPLES,
) -> dict[str, float]:
    """Measure PWL fit quality vs analytic ``fn`` on uniform ``q`` samples.

    Returns a dict with:
        ``max_lsb`` / ``p99_lsb`` / ``p999_lsb`` / ``rmse_lsb`` — integer-grid LSB stats,
        ``cosine_similarity`` — flattened shape fidelity,
        ``max_relative_error`` — informational, may be huge near zero-crossings.
    """

    # pylint: disable=import-outside-toplevel
    from aimet_torch.fixed_point.kernels.lut import evaluate_pwl_lut_int16
    from aimet_torch.fixed_point.metrics.accuracy import (
        cosine_similarity,
        quantize_float_to_grid,
    )

    device = input_encoding.scale.device
    qx = _validation_q_samples(input_encoding, num_samples, device)
    qy = evaluate_pwl_lut_int16(qx, pwl_lut).to(torch.int32)
    x_scale = input_encoding.scale.to(device=device, dtype=torch.float32)
    x_zp = input_encoding.zero_point.to(device=device, dtype=torch.float32)
    x_float = (qx.to(torch.float32) - x_zp) * x_scale
    y_ref_float = fn(x_float)
    q_ref = quantize_float_to_grid(
        y_ref_float,
        output_encoding.scale,
        output_encoding.zero_point,
        output_encoding.qmin,
        output_encoding.qmax,
    ).to(torch.int32)

    err_lsb = (qy - q_ref).abs().to(torch.float64)
    err_np = err_lsb.detach().cpu().numpy()
    metrics: dict[str, float] = {
        "max_lsb": float(err_np.max(initial=0.0)),
        "p99_lsb": float(np.quantile(err_np, 0.99)),
        "p999_lsb": float(np.quantile(err_np, 0.999)),
        "rmse_lsb": float(np.sqrt(np.mean(err_np * err_np))),
        "cosine_similarity": cosine_similarity(q_ref.to(torch.float32), qy.to(torch.float32)),
    }
    y_scale = output_encoding.scale.to(device=device, dtype=torch.float32)
    y_zp = output_encoding.zero_point.to(device=device, dtype=torch.float32)
    y_cur_float = (qy.to(torch.float32) - y_zp) * y_scale
    eps = float(y_scale.detach().cpu().reshape(-1)[0].item())
    rel = ((y_cur_float - y_ref_float).abs() / (y_ref_float.abs() + eps)).detach().cpu().numpy()
    metrics["max_relative_error"] = float(rel.max(initial=0.0))
    return metrics


def measure_pwl_lut_max_error_lsb(
    fn: Callable[[torch.Tensor], torch.Tensor],
    input_encoding: InputEncoding,
    output_encoding: OutputEncoding,
    pwl_lut: dict[str, torch.Tensor],
    *,
    num_samples: int = PWL_EXPORT_VALIDATION_SAMPLES,
) -> float:
    """Backwards-compatible scalar form of :func:`measure_pwl_lut_metrics` (``max_lsb`` only)."""

    return measure_pwl_lut_metrics(
        fn,
        input_encoding,
        output_encoding,
        pwl_lut,
        num_samples=num_samples,
    )["max_lsb"]


def check_pwl_metrics_within_limits(
    metrics: Mapping[str, float],
    limits: Mapping[str, float],
) -> list[tuple[str, float, float]]:
    """Return ``[(name, actual, threshold), ...]`` for any metric failing ``limits``.

    ``min_cosine_similarity`` is checked as a lower bound; other entries as upper bounds
    keyed by the matching metric name (e.g. ``max_lsb`` checks ``metrics['max_lsb']``).
    """

    failures: list[tuple[str, float, float]] = []
    min_cos = limits.get("min_cosine_similarity")
    if min_cos is not None:
        actual = metrics.get("cosine_similarity")
        if actual is None or actual + 1e-12 < min_cos:
            failures.append(("cosine_similarity", float(actual or 0.0), float(min_cos)))
    for key in ("max_lsb", "p99_lsb", "p999_lsb", "rmse_lsb"):
        threshold = limits.get(key)
        if threshold is None:
            continue
        actual = metrics.get(key)
        if actual is None or actual > threshold + 1e-6:
            failures.append((key, float(actual or 0.0), float(threshold)))
    return failures


def assert_pwl_metrics_within_limits(
    metrics: Mapping[str, float],
    *,
    fn_name: Optional[str] = None,
    limits: Optional[Mapping[str, float]] = None,
    overrides: Optional[Mapping[str, float]] = None,
    label: str = "PWL",
) -> None:
    """Raise :class:`PwlLutAccuracyError` if any metric breaches resolved limits."""

    effective = (
        dict(limits)
        if limits is not None
        else resolve_pwl_quality_limits(fn_name, overrides=overrides)
    )
    failures = check_pwl_metrics_within_limits(metrics, effective)
    if failures:
        rendered = ", ".join(f"{name}={actual:.4f} (limit {thr})" for name, actual, thr in failures)
        raise PwlLutAccuracyError(f"{label} ({fn_name or 'unknown'}): {rendered}.")


def generate_pwl_lut_for_export(
    fn: Callable[[torch.Tensor], torch.Tensor],
    input_encoding: InputEncoding,
    output_encoding: OutputEncoding,
    *,
    num_segments: int | None = None,
    validation_samples: int = PWL_EXPORT_VALIDATION_SAMPLES,
    samples_per_segment: int = 32,
    coeff_b_bit_width: int = 16,
    enforce_quality: bool = False,
    quality_limits: Optional[Mapping[str, float]] = None,
    fn_name: Optional[str] = None,
) -> tuple[dict[str, torch.Tensor], int, dict[str, float]]:
    """Fit PWL for export / runtime INT16 path using the hardware-fixed segment count.

    Segment count defaults to :data:`~aimet_torch.fixed_point.metrics.thresholds.PWL_HARDWARE_NUM_SEGMENTS`
    (16). The accelerator does not support arbitrary segment counts.

    Returned metrics compare PWL integer outputs to ``fn(dequant(q))`` quantized to the output grid —
    i.e. analytic reference, not fp32_qdq. With ``enforce_quality=True`` the function raises
    :class:`PwlLutAccuracyError` when the per-fn limits in
    :data:`~aimet_torch.fixed_point.metrics.thresholds.PWL_VS_ANALYTIC_PER_FN_LIMITS` are breached.

    Returns:
        ``(pwl_lut, num_segments, metrics_dict)`` where metrics include
        ``cosine_similarity`` / ``max_lsb`` / ``p99_lsb`` / ``p999_lsb`` / ``rmse_lsb`` /
        ``max_relative_error``.
    """

    seg = int(PWL_HARDWARE_NUM_SEGMENTS if num_segments is None else num_segments)
    if seg != PWL_HARDWARE_NUM_SEGMENTS:
        raise ValueError(
            f"PWL segment count must match hardware ({PWL_HARDWARE_NUM_SEGMENTS}); got {seg}."
        )

    pwl_lut = generate_pwl_lut(
        fn,
        input_encoding,
        output_encoding,
        num_segments=seg,
        samples_per_segment=samples_per_segment,
        coeff_b_bit_width=coeff_b_bit_width,
    )
    metrics = measure_pwl_lut_metrics(
        fn,
        input_encoding,
        output_encoding,
        pwl_lut,
        num_samples=validation_samples,
    )
    if enforce_quality:
        resolved_name = fn_name or getattr(fn, "__name__", None)
        assert_pwl_metrics_within_limits(
            metrics,
            fn_name=resolved_name,
            limits=quality_limits,
            label="PWL",
        )
    return pwl_lut, seg, metrics


def generate_lut_int16(
    fn: Callable[[torch.Tensor], torch.Tensor],
    input_encoding: InputEncoding,
    output_encoding: OutputEncoding,
    table_size: int = 256,
) -> torch.Tensor:
    """Generate an INT16 LUT by sampling fn over the input encoding range."""

    if table_size <= 1:
        raise ValueError("table_size must be greater than 1.")

    device = input_encoding.scale.device
    input_scale = _effective_scale_fp64(input_encoding.scale, device)
    input_zp = input_encoding.zero_point.to(torch.float32)
    output_scale = _effective_scale_fp64(output_encoding.scale, device)
    output_zp = output_encoding.zero_point.to(torch.float32)

    if torch.any(input_scale == 0) or torch.any(output_scale == 0):
        raise ValueError("input/output scale must be non-zero.")

    q_values = torch.linspace(
        input_encoding.qmin,
        input_encoding.qmax,
        table_size,
        dtype=torch.float32,
        device=input_scale.device,
    )
    x_float = (q_values - input_zp) * input_scale
    y_float = fn(x_float)
    y_int = torch.round(y_float / output_scale + output_zp)
    y_int = torch.clamp(y_int, output_encoding.qmin, output_encoding.qmax)
    return y_int.to(torch.int16)


def _fit_linear_segment(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if x.numel() < 2:
        raise ValueError("Need at least two samples to fit a PWL segment.")
    x64 = x.to(torch.float64).reshape(-1)
    y64 = y.to(torch.float64).reshape(-1)
    x_mean = x64.mean()
    y_mean = y64.mean()
    denom = torch.sum((x64 - x_mean) * (x64 - x_mean))
    if torch.abs(denom) < 1e-24:
        return torch.zeros((), dtype=torch.float64, device=x.device), y_mean
    slope = torch.sum((x64 - x_mean) * (y64 - y_mean)) / denom
    intercept = y_mean - slope * x_mean
    return slope, intercept


def _quantize_signed_multiplier(
    multiplier: torch.Tensor,
    bit_width: int = 16,
    max_shift: int = 31,
) -> tuple[int, int]:
    value = float(multiplier.detach().cpu().item())
    if value == 0.0 or not torch.isfinite(multiplier).item():
        return 0, 0

    max_q = (1 << (bit_width - 1)) - 1
    min_q = -(1 << (bit_width - 1))
    abs_value = abs(value)
    shift = int(torch.floor(torch.log2(torch.tensor(max_q / abs_value))).item())
    shift = max(min(shift, max_shift), -max_shift)
    if shift >= 0:
        q_b = int(round(value * (1 << shift)))
    else:
        q_b = int(round(value / (1 << (-shift))))
    q_b = min(max(q_b, min_q), max_q)
    return q_b, shift


def generate_pwl_lut(
    fn: Callable[[torch.Tensor], torch.Tensor],
    input_encoding: InputEncoding,
    output_encoding: OutputEncoding,
    *,
    num_segments: int = 16,
    samples_per_segment: int = 32,
    coeff_b_bit_width: int = 16,
) -> dict[str, torch.Tensor]:
    """Fit a general-scale piecewise-linear integer LUT.

    Each segment evaluates ``q_y = (q_b * (q_x - x_zp) >> n_bx_total) + term_c``.
    The returned tensors are runtime-ready and contain no floating point values.
    """

    if num_segments <= 0:
        raise ValueError("num_segments must be positive.")
    if samples_per_segment < 2:
        raise ValueError("samples_per_segment must be at least 2.")

    device = input_encoding.scale.device
    x_scale = _effective_scale_fp64(input_encoding.scale, device)
    y_scale = _effective_scale_fp64(output_encoding.scale, device)
    x_zp = input_encoding.zero_point.to(device=device, dtype=torch.int32)
    y_zp = output_encoding.zero_point.to(device=device, dtype=torch.float64)

    if x_scale.numel() != 1 or y_scale.numel() != 1:
        raise NotImplementedError("PWL LUT generation currently supports per-tensor encodings only.")
    if torch.any(x_scale == 0) or torch.any(y_scale == 0):
        raise ValueError("input/output scale must be non-zero.")

    q_edges = torch.linspace(
        input_encoding.qmin,
        input_encoding.qmax,
        num_segments + 1,
        dtype=torch.float64,
        device=device,
    )
    thresholds = torch.round(q_edges).to(torch.int32)
    q_b_values = []
    shifts = []
    terms_c = []

    for index in range(num_segments):
        q_start = q_edges[index]
        q_end = q_edges[index + 1]
        q_samples = torch.linspace(
            q_start,
            q_end,
            samples_per_segment,
            dtype=torch.float64,
            device=device,
        )
        x_float = (q_samples - x_zp.to(torch.float64)) * x_scale
        y_float = fn(x_float.to(torch.float32)).to(torch.float64)
        slope, intercept = _fit_linear_segment(x_float, y_float)

        multiplier = slope * x_scale / y_scale
        term_c = torch.round(intercept / y_scale + y_zp)
        term_c = torch.clamp(term_c, INT32_QMIN, INT32_QMAX).to(torch.int32)
        q_b, shift = _quantize_signed_multiplier(
            multiplier, bit_width=coeff_b_bit_width
        )
        q_b_values.append(q_b)
        shifts.append(shift)
        terms_c.append(int(term_c.item()))

    return {
        "thresholds": thresholds[:-1].to(torch.int32),
        "q_b": torch.tensor(q_b_values, dtype=torch.int16, device=device),
        "n_bx_total": torch.tensor(shifts, dtype=torch.int8, device=device),
        "term_c": torch.tensor(terms_c, dtype=torch.int32, device=device),
        "input_zero_point": x_zp.to(torch.int32),
        "output_qmin": torch.tensor(output_encoding.qmin, dtype=torch.int32, device=device),
        "output_qmax": torch.tensor(output_encoding.qmax, dtype=torch.int32, device=device),
    }


def encodings_share_quant_grid(
    source: InputEncoding,
    target: InputEncoding,
    *,
    rtol: float = 1e-5,
) -> bool:
    """True when two encodings describe the same integer grid (effective scale + zp + qrange)."""

    if source.qmin != target.qmin or source.qmax != target.qmax:
        return False

    devices = {
        source.scale.device,
        target.scale.device,
        source.zero_point.device,
        target.zero_point.device,
    }
    if any(d.type == "cuda" for d in devices):
        device = next(d for d in devices if d.type == "cuda")
    else:
        device = torch.device("cpu")

    z_src = source.zero_point.reshape(-1).to(device=device, dtype=torch.int32)
    z_tgt = target.zero_point.reshape(-1).to(device=device, dtype=torch.int32)
    if not torch.equal(z_src, z_tgt):
        return False
    s_src = _effective_scale_fp64(source.scale.to(device=device), device)
    s_tgt = _effective_scale_fp64(target.scale.to(device=device), device)
    return bool(torch.allclose(s_src, s_tgt, rtol=rtol, atol=0.0))


def periodic_lut_fit_spec(base_cls: type) -> tuple[Callable[[torch.Tensor], torch.Tensor] | None, str | None]:
    """Return ``(fit_fn, phase_fold)`` for sin/cos (cos reuses sin PWL table)."""

    from aimet_torch._base.nn.modules import custom

    if base_cls is custom.Sin:
        return torch.sin, "sin"
    if base_cls is custom.Cos:
        return torch.sin, "cos"
    return None, None


def principal_periodic_input_encoding(
    encoding: InputEncoding | OutputEncoding,
) -> InputEncoding:
    """Narrow fit range to one period ``[-pi, pi)`` in the op integer grid (§3.2)."""

    device = encoding.scale.device
    scale = float(
        _effective_scale_fp64(encoding.scale, device).reshape(-1)[0].item()
    )
    if scale <= 0.0:
        raise ValueError("input scale must be positive for periodic principal range.")

    zp = int(encoding.zero_point.reshape(-1)[0].item())
    q_2pi = int(round(2.0 * math.pi / scale))
    if q_2pi <= 0:
        return InputEncoding(
            scale=encoding.scale,
            zero_point=encoding.zero_point,
            qmin=int(encoding.qmin),
            qmax=int(encoding.qmax),
            axis=encoding.axis,
        )

    half = q_2pi // 2
    qmin = max(int(encoding.qmin), zp - half)
    qmax = min(int(encoding.qmax), zp + half - 1)
    if qmax < qmin:
        qmax = qmin
    return InputEncoding(
        scale=encoding.scale,
        zero_point=encoding.zero_point,
        qmin=qmin,
        qmax=qmax,
        axis=encoding.axis,
    )


def fold_periodic_input_to_principal_range(
    q_x: torch.Tensor,
    func_name: str,
    input_encoding: InputEncoding,
) -> torch.Tensor:
    """Fold ``sin``/``cos`` inputs into ``[-pi, pi)`` in the op integer grid (§3.2).

    ``cos`` is handled as ``sin(x + pi/2)`` via an integer phase offset ``Q_HALFPI``.
    """

    name = func_name.strip().lower()
    if name not in ("sin", "cos"):
        raise ValueError(f"phase_fold must be 'sin' or 'cos'; got {func_name!r}.")

    device = q_x.device
    scale = float(
        _effective_scale_fp64(input_encoding.scale, device).reshape(-1)[0].item()
    )
    if scale <= 0.0:
        raise ValueError("input scale must be positive for periodic fold.")

    zp = input_encoding.zero_point.to(device=device, dtype=torch.int32)
    qmin = int(input_encoding.qmin)
    qmax = int(input_encoding.qmax)

    q_2pi = int(round(2.0 * torch.pi / scale))
    if q_2pi <= 0:
        return torch.clamp(q_x.to(torch.int32), qmin, qmax)

    q_halfpi = int(round((torch.pi / 2.0) / scale))
    x_offset = q_x.to(torch.int64) - zp.to(torch.int64)
    if name == "cos":
        x_offset = x_offset + q_halfpi

    half = q_2pi // 2
    k = torch.where(
        x_offset >= 0,
        (x_offset + half) // q_2pi,
        -((-x_offset + half) // q_2pi),
    )
    folded = x_offset - k * q_2pi + zp.to(torch.int64)
    return torch.clamp(folded, qmin, qmax).to(torch.int32)


def _real_signed_multiplier(q_b: int, n_bx: int) -> float:
    if n_bx >= 0:
        return float(q_b) / float(1 << n_bx)
    return float(q_b) * float(1 << (-n_bx))


def _remap_lut_q_to_op_grid(
    q_lut: torch.Tensor,
    lut_encoding: InputEncoding,
    op_encoding: InputEncoding,
) -> torch.Tensor:
    """Map LUT-fit grid indices to the upstream op grid (inverse of §3.0 align)."""

    device = q_lut.device
    if encodings_share_quant_grid(lut_encoding, op_encoding):
        return q_lut.to(torch.int32)

    s_lut = _effective_scale_fp64(lut_encoding.scale, device)
    s_op = _effective_scale_fp64(op_encoding.scale, device)
    if torch.any(s_op == 0):
        raise ValueError("op input scale must be non-zero for threshold remap.")

    ratio = s_lut / s_op
    m, r = quantize_multiplier(ratio)
    zp_lut = lut_encoding.zero_point.to(device=device, dtype=torch.int32)
    zp_op = op_encoding.zero_point.to(device=device, dtype=torch.int32)
    centered = q_lut.to(torch.int32) - zp_lut
    rounding = (
        RoundingMode.HALF_UP if hw_ref_mode_enabled() else RoundingMode.HALF_TO_EVEN
    )
    return requantize_int(
        centered,
        m.to(device=device),
        r.to(device=device),
        zp_op,
        op_encoding.qmin,
        op_encoding.qmax,
        rounding_mode=rounding,
    )


def bake_op_scale_adapter_into_pwl_lut(
    pwl_lut: dict[str, torch.Tensor],
    op_encoding: InputEncoding,
    lut_encoding: InputEncoding,
    *,
    coeff_b_bit_width: int = 16,
) -> dict[str, torch.Tensor]:
    """Bake §3.0 scale adapter into per-segment ``q_b`` (mitigation 2, general-scale doc).

    After baking, runtime may evaluate with ``q_x`` on the **op** grid (skip
    :func:`align_op_quant_grid_to_lut_quant_grid`) when ``scale_adapter_baked`` is set.
    Segment thresholds and ``input_zero_point`` are remapped to the op grid.
    """

    if encodings_share_quant_grid(op_encoding, lut_encoding):
        baked = {key: value for key, value in pwl_lut.items()}
        baked["scale_adapter_baked"] = True
        return baked

    device = lut_encoding.scale.device
    s_op = _effective_scale_fp64(op_encoding.scale, device)
    s_lut = _effective_scale_fp64(lut_encoding.scale, device)
    if torch.any(s_lut == 0):
        raise ValueError("LUT input scale must be non-zero for adapter bake.")

    q_r, n_r = quantize_multiplier((s_op / s_lut).reshape(1))
    q_r_i = int(q_r.reshape(-1)[0].item())
    n_r_i = int(n_r.reshape(-1)[0].item())

    q_b_values = []
    shift_values = []
    for q_b, n_bx in zip(
        pwl_lut["q_b"].tolist(),
        pwl_lut["n_bx_total"].tolist(),
    ):
        combined = _real_signed_multiplier(int(q_b), int(n_bx)) * _real_signed_multiplier(
            q_r_i, n_r_i
        )
        qb_new, shift_new = _quantize_signed_multiplier(
            torch.tensor(combined, dtype=torch.float64),
            bit_width=coeff_b_bit_width,
        )
        q_b_values.append(qb_new)
        shift_values.append(shift_new)

    thresholds_op = _remap_lut_q_to_op_grid(
        pwl_lut["thresholds"].to(torch.int32), lut_encoding, op_encoding
    )

    baked = {key: value for key, value in pwl_lut.items()}
    baked["q_b"] = torch.tensor(q_b_values, dtype=torch.int16, device=device)
    baked["n_bx_total"] = torch.tensor(shift_values, dtype=torch.int8, device=device)
    baked["thresholds"] = thresholds_op.to(torch.int32)
    baked["input_zero_point"] = op_encoding.zero_point.to(
        device=device, dtype=torch.int32
    )
    baked["scale_adapter_baked"] = True
    return baked


def align_op_quant_grid_to_lut_quant_grid(
    q_x: torch.Tensor,
    op_encoding: InputEncoding,
    lut_encoding: InputEncoding,
) -> torch.Tensor:
    """Map ``q_x`` from the upstream op grid to the LUT fit grid (general-scale §3.0).

    Uses ``requantize_int`` with ``(M, rshift)`` approximating ``s_op / s_lut`` on
    effective discrete scales. No-op when :func:`encodings_share_quant_grid` is true.
    """

    if encodings_share_quant_grid(op_encoding, lut_encoding):
        return q_x

    device = q_x.device
    s_op = _effective_scale_fp64(op_encoding.scale, device)
    s_lut = _effective_scale_fp64(lut_encoding.scale, device)
    if torch.any(s_lut == 0):
        raise ValueError("LUT input scale must be non-zero for grid alignment.")

    ratio = s_op / s_lut
    m, r = quantize_multiplier(ratio)
    centered = q_x.to(torch.int32) - op_encoding.zero_point.to(
        device=device, dtype=torch.int32
    )
    rounding = (
        RoundingMode.HALF_UP if hw_ref_mode_enabled() else RoundingMode.HALF_TO_EVEN
    )
    return requantize_int(
        centered,
        m.to(device=device),
        r.to(device=device),
        lut_encoding.zero_point.to(device=device, dtype=torch.int32),
        lut_encoding.qmin,
        lut_encoding.qmax,
        rounding_mode=rounding,
    )


def pwl_lut_to_json_dict(
    pwl_lut: dict[str, torch.Tensor],
    *,
    func_name: str,
    input_encoding: InputEncoding,
    output_encoding: OutputEncoding,
    quality_metrics: Optional[Mapping[str, float]] = None,
    quality_limits: Optional[Mapping[str, float]] = None,
) -> dict[str, Any]:
    """Export runtime PWL LUT tensors to a JSON-friendly dict.

    The segment schema intentionally mirrors ``abc_lut-shuai/lut_int_general``:
    ``threshold_quantized``, ``q_b``, ``n_bx_total`` and ``term_c_precomputed``.

    If ``quality_metrics`` is provided, a ``quality`` block is attached with the metric
    values, the corresponding limits (if any) and a ``status`` indicating PASS / FAIL.
    """

    thresholds = pwl_lut["thresholds"].detach().cpu().to(torch.int32).tolist()
    q_b = pwl_lut["q_b"].detach().cpu().to(torch.int16).tolist()
    shifts = pwl_lut["n_bx_total"].detach().cpu().to(torch.int8).tolist()
    terms = pwl_lut["term_c"].detach().cpu().to(torch.int32).tolist()
    segments = []
    for index, threshold in enumerate(thresholds):
        next_threshold = (
            thresholds[index + 1] if index + 1 < len(thresholds) else input_encoding.qmax
        )
        segments.append(
            {
                "segment_id": index,
                "threshold_quantized": [int(threshold), int(next_threshold)],
                "coefficients_quantized": {"q_b": int(q_b[index]), "q_c": 0},
                "shift_bits": {"n_bx_total": int(shifts[index])},
                "term_c_precomputed": int(terms[index]),
            }
        )

    def _float_range(enc: InputEncoding | OutputEncoding) -> tuple[float, float]:
        scale = float(enc.scale.detach().cpu().reshape(-1)[0].item())
        zp = int(enc.zero_point.detach().cpu().reshape(-1)[0].item())
        fmin = scale * (enc.qmin - zp)
        fmax = scale * (enc.qmax - zp)
        return fmin, fmax

    in_fmin, in_fmax = _float_range(input_encoding)
    out_fmin, out_fmax = _float_range(output_encoding)

    body: dict[str, Any] = {
        "quantization": {
            "input": {
                "scale": float(input_encoding.scale.detach().cpu().reshape(-1)[0].item()),
                "zero_point": int(input_encoding.zero_point.detach().cpu().reshape(-1)[0].item()),
                "qmin": int(input_encoding.qmin),
                "qmax": int(input_encoding.qmax),
                "fmin": in_fmin,
                "fmax": in_fmax,
            },
            "output": {
                "scale": float(output_encoding.scale.detach().cpu().reshape(-1)[0].item()),
                "zero_point": int(output_encoding.zero_point.detach().cpu().reshape(-1)[0].item()),
                "qmin": int(output_encoding.qmin),
                "qmax": int(output_encoding.qmax),
                "fmin": out_fmin,
                "fmax": out_fmax,
            },
            "coeff_b_bit_width": 16,
            "term_c_bit_width": 32,
            "internal_accumulator_bit_width": 32,
        },
        "num_segments": len(segments),
        "segments": segments,
    }

    if quality_metrics is not None:
        metrics_block = {key: float(value) for key, value in quality_metrics.items()}
        quality_block: dict[str, Any] = {"metrics": metrics_block}
        if quality_limits is not None:
            limits_block = {key: float(value) for key, value in quality_limits.items()}
            quality_block["limits"] = limits_block
            failures = check_pwl_metrics_within_limits(metrics_block, limits_block)
            quality_block["failures"] = [
                {"metric": name, "actual": float(actual), "threshold": float(threshold)}
                for name, actual, threshold in failures
            ]
            quality_block["status"] = "PASS" if not failures else "FAIL"
        else:
            quality_block["status"] = "PASS"
        body["quality"] = quality_block

    return {func_name: body}


def attach_pwl_sidecar_metadata(
    pwl_json: dict[str, Any],
    *,
    phase_fold: str | None = None,
    pwl_input_encoding: InputEncoding | None = None,
) -> dict[str, Any]:
    """Add runtime hints (``phase_fold``, ``pwl_input_encoding``) to exported PWL JSON."""

    if not pwl_json:
        return pwl_json
    func_name = next(iter(pwl_json))
    body = pwl_json[func_name]
    if phase_fold is not None:
        body["phase_fold"] = phase_fold
    if pwl_input_encoding is not None:
        scale = float(
            pwl_input_encoding.scale.detach().cpu().reshape(-1)[0].item()
        )
        body["pwl_input_encoding"] = {
            "scale": scale,
            "zero_point": int(
                pwl_input_encoding.zero_point.detach().cpu().reshape(-1)[0].item()
            ),
            "qmin": int(pwl_input_encoding.qmin),
            "qmax": int(pwl_input_encoding.qmax),
        }
    return pwl_json


def pwl_lut_from_json_dict(data: dict[str, Any], func_name: str | None = None) -> dict[str, torch.Tensor]:
    """Load AIMET RX / lut_int_general-style PWL LUT JSON into runtime tensors."""

    if func_name is None:
        func_name = next(iter(data))
    func_data = data[func_name]
    segments = func_data["segments"]
    quant = func_data["quantization"]
    input_quant = quant["input"]
    output_quant = quant["output"]

    return {
        "thresholds": torch.tensor(
            [seg["threshold_quantized"][0] for seg in segments], dtype=torch.int32
        ),
        "q_b": torch.tensor(
            [seg["coefficients_quantized"]["q_b"] for seg in segments],
            dtype=torch.int16,
        ),
        "n_bx_total": torch.tensor(
            [seg["shift_bits"]["n_bx_total"] for seg in segments], dtype=torch.int8
        ),
        "term_c": torch.tensor(
            [seg["term_c_precomputed"] for seg in segments], dtype=torch.int32
        ),
        "input_zero_point": torch.tensor(input_quant["zero_point"], dtype=torch.int32),
        "output_qmin": torch.tensor(output_quant.get("qmin", -32768), dtype=torch.int32),
        "output_qmax": torch.tensor(output_quant.get("qmax", 32767), dtype=torch.int32),
    }

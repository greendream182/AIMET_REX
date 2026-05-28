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
"""Offline CLZ-normalized LUT generation (delegates to ``lut_int_general`` when available)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch

from aimet_torch.fixed_point.encoding import InputEncoding, OutputEncoding
from aimet_torch.fixed_point.metrics.thresholds import PWL_HARDWARE_NUM_SEGMENTS
from aimet_torch.fixed_point.offline.lut_gen import _effective_scale_fp64

CLZ_HW_NUM_SEGMENTS = PWL_HARDWARE_NUM_SEGMENTS
CLZ_INPUT_BIT_WIDTH = 16
CLZ_INTERNAL_BIT_WIDTH = 32
CLZ_COEFF_B_BIT_WIDTH = 16
CLZ_TERM_C_BIT_WIDTH = 32

_CLZ_MODULE_NAMES = {
    "sqrt": "sqrt",
    "rsqrt": "rsqrt",
    "reciprocal": "reciprocal",
    "power_2": "power_2",
    "power2": "power_2",
}


class ClzLutGenerationError(RuntimeError):
    """Raised when CLZ LUT cannot be generated (missing abc tree or fit failure)."""


def resolve_clz_fit_float_range(
    qmodule: Any,
    func_name: str,
    input_encoding: InputEncoding,
) -> tuple[float | None, float | None]:
    """Use quantizer ``min``/``max`` only when encoding range is misleading for the op."""

    calib = quantizer_calibration_float_range(qmodule)
    if calib is None:
        return None, None
    enc_min, enc_max = _float_range_from_encoding(input_encoding)
    del enc_max
    if func_name.lower() == "reciprocal":
        return calib
    # sqrt/rsqrt/power_2: keep encoding-derived range unless a future op needs calib.
    if enc_min <= 0 and func_name.lower() == "power_2":
        return calib
    return None, None


def quantizer_calibration_float_range(qmodule: Any) -> tuple[float, float] | None:
    """Return calibrated ``(min, max)`` from the first input quantizer when set."""

    quantizers = getattr(qmodule, "input_quantizers", None)
    if not quantizers:
        return None
    iq = quantizers[0]
    min_p = getattr(iq, "min", None)
    max_p = getattr(iq, "max", None)
    if min_p is None or max_p is None:
        return None
    try:
        fmin = float(min_p.detach().cpu().reshape(-1)[0].item())
        fmax = float(max_p.detach().cpu().reshape(-1)[0].item())
    except (TypeError, ValueError):
        return None
    if not (fmax > fmin):
        return None
    return fmin, fmax


def clz_activation_name(base_cls: type) -> str | None:
    """Map AIMET ``custom`` module class to CLZ function name."""

    from aimet_torch._base.nn.modules import custom

    if base_cls is custom.Sqrt:
        return "sqrt"
    if base_cls is custom.RSqrt:
        return "rsqrt"
    if base_cls is custom.Reciprocal:
        return "reciprocal"
    if base_cls is custom.Square:
        return "power_2"
    return None


def resolve_abc_lut_root(explicit: str | Path | None = None) -> Path | None:
    """Locate ``abc_lut-shuai`` (env ``AIMET_RX_ABC_LUT_ROOT`` or workspace sibling)."""

    if explicit is not None:
        root = Path(explicit)
        if (root / "lut_int_general" / "quantization").is_dir():
            return root
        return None

    env = os.environ.get("AIMET_RX_ABC_LUT_ROOT")
    if env:
        root = Path(env)
        if (root / "lut_int_general" / "quantization").is_dir():
            return root

    sibling = Path(__file__).resolve().parents[3].parent / "abc_lut-shuai"
    if (sibling / "lut_int_general" / "quantization").is_dir():
        return sibling
    return None


def _float_range_from_encoding(enc: InputEncoding | OutputEncoding) -> tuple[float, float]:
    device = enc.scale.device
    scale = _effective_scale_fp64(enc.scale, device)
    zp = enc.zero_point.to(device=device, dtype=torch.float64)
    qmin = torch.tensor(enc.qmin, device=device, dtype=torch.float64)
    qmax = torch.tensor(enc.qmax, device=device, dtype=torch.float64)
    fmin = float(((qmin - zp) * scale).reshape(-1).min().item())
    fmax = float(((qmax - zp) * scale).reshape(-1).max().item())
    return fmin, fmax


def _clz_domain_float_range(
    func_name: str,
    input_encoding: InputEncoding,
    *,
    fit_float_min: float | None = None,
    fit_float_max: float | None = None,
) -> tuple[float, float]:
    """Positive-domain CLZ fit range (avoids symmetric-negative artefact on reciprocal/rsqrt)."""

    if fit_float_min is not None and fit_float_max is not None:
        x_min, x_max = float(fit_float_min), float(fit_float_max)
    else:
        x_min, x_max = _float_range_from_encoding(input_encoding)

    name = func_name.lower()
    if name == "sqrt":
        x_min = max(0.0, x_min)
    elif name in ("rsqrt", "reciprocal"):
        if x_max <= 0:
            raise ClzLutGenerationError(
                f"CLZ fit range invalid for {func_name}: need positive x_max (got {x_max})."
            )
        x_min = max(x_min, 1e-6)
        # Symmetric encodings can report negative fmin; do not sample near zero (1/x blow-up).
        if x_min <= 0 or x_min < 0.05 * x_max:
            x_min = max(0.05 * x_max, 1e-6)
    elif name == "power_2":
        x_min = max(0.0, x_min)
        if x_max <= x_min:
            x_max = x_min * 2.0
    if x_max <= x_min:
        raise ClzLutGenerationError(
            f"CLZ fit range invalid for {func_name}: x_min={x_min}, x_max={x_max}."
        )
    return x_min, x_max


def _import_abc_clz_fitter(abc_root: Path):
    general = abc_root / "lut_int_general"
    path_str = str(general)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    from quantization.clz_normalized_fitter import CLZNormalizedLUTFitter  # type: ignore
    from quantization._internal.nonlinear_registry import get_nonlinear_callable  # type: ignore
    from quantization.lut import clz_fitter_to_lut_json  # type: ignore

    return CLZNormalizedLUTFitter, get_nonlinear_callable, clz_fitter_to_lut_json


def generate_clz_lut_for_export(
    func_name: str,
    input_encoding: InputEncoding,
    output_encoding: OutputEncoding | None = None,
    *,
    fit_float_min: float | None = None,
    fit_float_max: float | None = None,
    num_segments: int = CLZ_HW_NUM_SEGMENTS,
    abc_lut_root: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Fit a CLZ LUT body for ``extra['clz_lut']`` (``lut_int_general`` fitter).

    ``output_encoding`` is reserved for future grid alignment; the abc fitter
    derives output quantizers from the analytic function over the input range.
    Pass ``fit_float_min`` / ``fit_float_max`` from calibrated quantizer ``min``/``max``
    when symmetric encodings span negative floats but the op domain is positive-only.
    """

    del output_encoding
    name = _CLZ_MODULE_NAMES.get(func_name.lower())
    if name is None:
        raise ClzLutGenerationError(f"unsupported CLZ function: {func_name}")

    if num_segments != CLZ_HW_NUM_SEGMENTS:
        raise ValueError(
            f"CLZ segment count must match hardware ({CLZ_HW_NUM_SEGMENTS}); got {num_segments}."
        )

    root = resolve_abc_lut_root(abc_lut_root)
    if root is None:
        raise ClzLutGenerationError(
            "abc_lut-shuai/lut_int_general not found. Set AIMET_RX_ABC_LUT_ROOT "
            "or place the repo beside aimet_rx."
        )

    x_min, x_max = _clz_domain_float_range(
        name,
        input_encoding,
        fit_float_min=fit_float_min,
        fit_float_max=fit_float_max,
    )
    fitter_cls, get_fn, to_json = _import_abc_clz_fitter(root)
    func = get_fn(name)
    fitter = fitter_cls(
        num_segments=num_segments,
        normalized_space_bit_width=CLZ_INPUT_BIT_WIDTH,
        normalized_output_bit_width=CLZ_INPUT_BIT_WIDTH,
        global_input_bit_width=CLZ_INPUT_BIT_WIDTH,
        coeff_b_bit_width=CLZ_COEFF_B_BIT_WIDTH,
        term_c_bit_width=CLZ_TERM_C_BIT_WIDTH,
        internal_acc_bit_width=CLZ_INTERNAL_BIT_WIDTH,
    )
    fitter.fit_normalized_function(func, name, x_min, x_max)
    payload = to_json(fitter)
    body = payload[name]
    metrics = {"clz_x_min": x_min, "clz_x_max": x_max, "num_segments": float(num_segments)}
    return body, metrics


def clz_lut_to_json_dict(
    clz_body: dict[str, Any],
    func_name: str,
    *,
    fit_metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Wrap CLZ LUT body for sidecar / ``lut_int_general``-style JSON."""

    body = dict(clz_body)
    if fit_metrics is not None:
        body["export_metrics"] = {key: float(value) for key, value in fit_metrics.items()}
    return {func_name: body}

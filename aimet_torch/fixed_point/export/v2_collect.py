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
"""Derive INT16 fixed-point encodings from AIMET v2 quantized modules (offline export)."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from aimet_torch._base.nn.modules import custom
from aimet_torch.fixed_point.encoding import InputEncoding, OutputEncoding
from aimet_torch.fixed_point.encoding_export import _tensor_int_json
from aimet_torch.fixed_point.offline.clz_gen import (
    ClzLutGenerationError,
    clz_activation_name,
    clz_lut_to_json_dict,
    generate_clz_lut_for_export,
    resolve_clz_fit_float_range,
)
from aimet_torch.fixed_point.offline.lut_gen import (
    attach_pwl_sidecar_metadata,
    generate_pwl_lut_for_export,
    periodic_lut_fit_spec,
    principal_periodic_input_encoding,
    pwl_lut_to_json_dict,
    resolve_pwl_quality_limits,
)
from aimet_torch.fixed_point.offline.multiplier import quantize_multiplier
from aimet_torch.fixed_point.registry import KernelNotFoundError, get_fixed_kernel

# Side-effect import: kernel registrations live as import-time decorators on each
# kernel submodule. ``fixed_point/__init__.py`` intentionally avoids pulling them at
# package root to keep the root import lightweight, so we eagerly load them here —
# the export/freeze pipeline can't probe the registry before this side-effect runs.
# Without this, running ``freeze_int16_fixed`` in a process that never touched the
# kernels subpackage (e.g. ``test_pipeline.py`` in isolation) silently sees an empty
# registry and skips every layer.
import aimet_torch.fixed_point.kernels  # noqa: E402,F401  pylint: disable=wrong-import-position,unused-import
from aimet_torch.v2.quantization.affine.encoding import AffineEncoding
from aimet_torch.v2.quantization.base import QuantizerBase

# Shared with runtime adapter; export must stay bit-aligned with dispatch_int16_fixed.
from aimet_torch.v2.quantization.affine.fixed_point.adapter import (  # pylint: disable=protected-access
    _affine_output_encoding,
    _affine_to_fixed_encoding,
    _broadcast_output_encoding_conv2d,
    _broadcast_output_encoding_linear,
    _is_weighted_module,
    _pwl_activation_fn,
)


def _first_initialized_affine(
    quantizers,
) -> Optional[AffineEncoding]:
    if not quantizers:
        return None
    for q in quantizers:
        if isinstance(q, QuantizerBase) and q.is_initialized():
            enc = q.get_encodings()
            if isinstance(enc, AffineEncoding):
                return enc
    return None


def _pwl_fit_input_encoding(
    base_cls: type,
    input_enc: InputEncoding,
) -> InputEncoding:
    """Adjust fit grid for saturating activations (e.g. ``log`` needs positive domain)."""

    if base_cls is not custom.Log:
        return input_enc

    device = input_enc.scale.device
    scale = float(
        input_enc.scale.detach().cpu().reshape(-1)[0].item()
    )
    if scale <= 0.0:
        return input_enc
    zp = int(input_enc.zero_point.detach().cpu().reshape(-1)[0].item())
    fmin = scale * (int(input_enc.qmin) - zp)
    if fmin > 1e-6:
        return input_enc
    qmin_pos = zp + int(round(1e-4 / scale))
    qmin_pos = max(int(input_enc.qmin), min(qmin_pos, int(input_enc.qmax)))
    return InputEncoding(
        scale=input_enc.scale,
        zero_point=input_enc.zero_point,
        qmin=qmin_pos,
        qmax=int(input_enc.qmax),
        axis=input_enc.axis,
    )


def _resolve_pwl_fit_fn(
    qmodule: nn.Module,
    base_cls: type,
) -> tuple[Any | None, str | None]:
    """Return ``(callable, phase_fold)`` aligned with runtime adapter LUT generation."""

    pwl_fn = _pwl_activation_fn(qmodule, base_cls)
    periodic_fit_fn, phase_fold = periodic_lut_fit_spec(base_cls)
    if pwl_fn is None and periodic_fit_fn is not None:
        pwl_fn = periodic_fit_fn
    return pwl_fn, phase_fold


def derive_int16_real_multiplier(
    qmodule: nn.Module,
    *,
    base_cls: type,
    device: Optional[torch.device] = None,
) -> Optional[torch.Tensor]:
    """Floating requant ratio used before ``quantize_multiplier`` (offline / export)."""

    dev = device or torch.device("cpu")
    oq = qmodule.output_quantizers[0] if getattr(qmodule, "output_quantizers", None) else None
    if not isinstance(oq, QuantizerBase) or not oq.is_initialized():
        return None
    y_enc = oq.get_encodings()
    if not isinstance(y_enc, AffineEncoding):
        return None

    if base_cls is custom.Concat:
        input_quantizers = qmodule.input_quantizers
    else:
        input_quantizers = qmodule.input_quantizers[:1] if qmodule.input_quantizers else []

    x_enc = _first_initialized_affine(input_quantizers)
    if base_cls is custom.Concat:
        y_scale = y_enc.scale.to(device=dev, dtype=torch.float32)
        return torch.ones_like(y_scale)

    if x_enc is None and not _is_weighted_module(base_cls):
        return None

    x_scale = x_enc.scale.to(device=dev, dtype=torch.float32) if x_enc is not None else None

    if _is_weighted_module(base_cls):
        pq = getattr(qmodule, "param_quantizers", None)
        wq = pq["weight"] if pq is not None and "weight" in pq else None
        if not isinstance(wq, QuantizerBase) or not wq.is_initialized():
            return None
        w_enc = wq.get_encodings()
        if not isinstance(w_enc, AffineEncoding) or x_scale is None:
            return None
        w_scale = w_enc.scale.to(device=dev, dtype=torch.float32)
        y_scale = y_enc.scale.to(device=dev, dtype=torch.float32)
        return (x_scale * w_scale) / y_scale

    if x_scale is None:
        return None
    y_scale = y_enc.scale.to(device=dev, dtype=torch.float32)

    if base_cls is custom.Multiply and len(qmodule.input_quantizers) >= 2:
        scales = []
        for iq in qmodule.input_quantizers[:2]:
            if not isinstance(iq, QuantizerBase) or not iq.is_initialized():
                return None
            enc = iq.get_encodings()
            if not isinstance(enc, AffineEncoding):
                return None
            scales.append(enc.scale.to(device=dev, dtype=torch.float32))
        prod_scale = scales[0] * scales[1]
        return prod_scale / y_scale

    return x_scale / y_scale


def derive_int16_output_encoding(
    qmodule: nn.Module,
    *,
    base_cls: type,
    device: Optional[torch.device] = None,
) -> Optional[OutputEncoding]:
    """Build :class:`OutputEncoding` with ``(multiplier, rshift)`` for one quantized module."""

    dev = device or torch.device("cpu")
    oq = qmodule.output_quantizers[0] if getattr(qmodule, "output_quantizers", None) else None
    if not isinstance(oq, QuantizerBase) or not oq.is_initialized():
        return None
    y_enc = oq.get_encodings()
    if not isinstance(y_enc, AffineEncoding):
        return None

    real_m = derive_int16_real_multiplier(qmodule, base_cls=base_cls, device=dev)
    if real_m is None:
        return _affine_to_fixed_encoding(y_enc, dev)

    out_enc = _affine_output_encoding(y_enc, real_m, dev)
    if base_cls is nn.Linear:
        out_enc = _broadcast_output_encoding_linear(out_enc, qmodule.out_features)
    elif base_cls in (nn.Conv1d, nn.Conv2d):
        out_enc = _broadcast_output_encoding_conv2d(out_enc, qmodule.out_channels)
    return out_enc


def derive_input_requant_records(
    qmodule: nn.Module,
    *,
    base_cls: type,
    device: Optional[torch.device] = None,
) -> Optional[list[dict[str, Any]]]:
    """Per-input-branch ``(multiplier, rshift)`` for Add/Subtract/Multiply/Concat."""

    if base_cls not in (custom.Add, custom.Subtract, custom.Multiply, custom.Concat):
        return None

    dev = device or torch.device("cpu")
    oq = qmodule.output_quantizers[0] if getattr(qmodule, "output_quantizers", None) else None
    if not isinstance(oq, QuantizerBase) or not oq.is_initialized():
        return None
    y_enc = oq.get_encodings()
    if not isinstance(y_enc, AffineEncoding):
        return None
    y_scale = y_enc.scale.to(device=dev, dtype=torch.float32)

    input_quantizers = qmodule.input_quantizers
    if base_cls in (custom.Add, custom.Subtract):
        input_quantizers = input_quantizers[:2]

    records: list[dict[str, Any]] = []
    for iq in input_quantizers:
        if not isinstance(iq, QuantizerBase) or not iq.is_initialized():
            return None
        enc = iq.get_encodings()
        if not isinstance(enc, AffineEncoding):
            return None
        x_scale = enc.scale.to(device=dev, dtype=torch.float32)
        real_m = x_scale / y_scale
        mult, rsh = quantize_multiplier(real_m.detach())
        records.append(
            {
                "multiplier": _tensor_int_json(mult),
                "rshift": _tensor_int_json(rsh),
            }
        )
    return records


def derive_int16_pwl_json(
    qmodule: nn.Module,
    *,
    base_cls: type,
    out_enc: OutputEncoding,
    func_name: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Build lut_int_general-style PWL JSON for a quantized activation module."""

    pwl_fn, phase_fold = _resolve_pwl_fit_fn(qmodule, base_cls)
    if pwl_fn is None:
        return None

    device = torch.device("cpu")
    x_enc = _first_initialized_affine(qmodule.input_quantizers)
    if x_enc is None:
        return None

    input_enc = _affine_to_fixed_encoding(x_enc, device)
    input_enc = _pwl_fit_input_encoding(base_cls, input_enc)
    fit_enc = (
        principal_periodic_input_encoding(input_enc)
        if phase_fold is not None
        else input_enc
    )
    if func_name is None:
        func_name = phase_fold or base_cls.__name__.lower()
    pwl_lut, _, metrics = generate_pwl_lut_for_export(
        pwl_fn, fit_enc, out_enc, fn_name=func_name
    )
    limits = resolve_pwl_quality_limits(func_name)
    pwl_json = pwl_lut_to_json_dict(
        pwl_lut,
        func_name=func_name,
        input_encoding=fit_enc,
        output_encoding=out_enc,
        quality_metrics=metrics,
        quality_limits=limits,
    )
    return attach_pwl_sidecar_metadata(
        pwl_json,
        phase_fold=phase_fold,
        pwl_input_encoding=fit_enc,
    )


def derive_int16_clz_json(
    qmodule: nn.Module,
    *,
    base_cls: type,
    out_enc: OutputEncoding,
) -> Optional[dict[str, Any]]:
    """Build CLZ-normalized LUT JSON (``lut_int_general`` schema) for sqrt/rsqrt/reciprocal."""

    func_name = clz_activation_name(base_cls)
    if func_name is None:
        return None

    device = torch.device("cpu")
    x_enc = _first_initialized_affine(qmodule.input_quantizers)
    if x_enc is None:
        return None
    input_enc = _affine_to_fixed_encoding(x_enc, device)

    fit_min, fit_max = resolve_clz_fit_float_range(qmodule, func_name, input_enc)
    try:
        body, metrics = generate_clz_lut_for_export(
            func_name,
            input_enc,
            out_enc,
            fit_float_min=fit_min,
            fit_float_max=fit_max,
        )
    except ClzLutGenerationError:
        return None
    return clz_lut_to_json_dict(body, func_name, fit_metrics=metrics)


def collect_v2_int16_layer_record(
    qmodule: nn.Module,
    layer_name: str,
) -> Optional[dict[str, Any]]:
    """Collect one layer's export record, or ``None`` if not an INT16-capable quantized op."""

    # pylint: disable=import-outside-toplevel
    from aimet_torch.v2.nn.true_quant import QuantizationMixin

    base_cls = QuantizationMixin.qcls_to_cls.get(type(qmodule))
    if base_cls is None:
        return None
    try:
        get_fixed_kernel(base_cls)
    except KernelNotFoundError:
        return None

    out_enc = derive_int16_output_encoding(qmodule, base_cls=base_cls)
    if out_enc is None:
        return None

    record: dict[str, Any] = {
        "op": base_cls.__name__,
        "output_encoding": out_enc,
    }
    pwl_json = derive_int16_pwl_json(qmodule, base_cls=base_cls, out_enc=out_enc)
    if pwl_json is not None:
        record["pwl"] = pwl_json
        func_block = pwl_json[next(iter(pwl_json))]
        if "phase_fold" in func_block:
            record["phase_fold"] = func_block["phase_fold"]
    clz_json = derive_int16_clz_json(qmodule, base_cls=base_cls, out_enc=out_enc)
    if clz_json is not None:
        record["clz"] = clz_json
    input_requants = derive_input_requant_records(qmodule, base_cls=base_cls)
    if input_requants is not None:
        record["input_requants"] = input_requants
    record["layer"] = layer_name
    return record


def collect_v2_int16_layers(
    model: nn.Module,
) -> Dict[str, dict[str, Any]]:
    """Return ``{qualified_name: record}`` for all INT16-dispatchable v2 quantized modules."""

    # pylint: disable=import-outside-toplevel
    from aimet_torch.v2.nn.true_quant import QuantizationMixin

    layers: Dict[str, dict[str, Any]] = {}
    for name, module in model.named_modules():
        if not isinstance(module, QuantizationMixin):
            continue
        record = collect_v2_int16_layer_record(module, name or "<root>")
        if record is not None:
            layers[name or "<root>"] = record
    return layers

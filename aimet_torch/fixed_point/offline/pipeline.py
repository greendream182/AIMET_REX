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
"""Offline freeze pipeline: calibrated QuantSim → INT16 sidecar + per-layer report."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from aimet_torch.fixed_point.encoding import OutputEncoding
from aimet_torch.fixed_point.encoding_export import output_encoding_to_dict
from aimet_torch.fixed_point.offline.bias import quantize_bias_int32
from aimet_torch.fixed_point.registry import KernelNotFoundError, get_fixed_kernel
from aimet_torch.v2.quantization.affine.encoding import AffineEncoding
from aimet_torch.v2.quantization.base import QuantizerBase


def _is_weighted_module(base_cls: type) -> bool:
    # pylint: disable=import-outside-toplevel
    import torch.nn as _nn

    return base_cls in (_nn.Linear, _nn.Conv1d, _nn.Conv2d)


def _resolve_quantized_model(sim_model: Any) -> nn.Module:
    """Accept a :class:`QuantizationSimModel` or a raw v2 quantized ``nn.Module``."""

    model = getattr(sim_model, "model", sim_model)
    if not isinstance(model, nn.Module):
        raise TypeError(
            "sim_model must be an nn.Module or an object with a .model attribute; "
            f"got {type(sim_model).__name__}."
        )
    return model


def _safe_layer_filename(layer_name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", layer_name.strip() or "layer")
    return safe[:120]


def _write_int32_binary(tensor: torch.Tensor, path: str) -> None:
    arr = tensor.detach().cpu().numpy().astype("<i4", copy=False)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    arr.tofile(path)


def _approx_multiplier_tensor(
    multiplier: torch.Tensor,
    rshift: torch.Tensor,
) -> torch.Tensor:
    return multiplier.to(torch.float64) / torch.pow(
        2.0,
        rshift.to(torch.float64),
    )


def multiplier_relative_error(
    real_multiplier: torch.Tensor,
    multiplier: torch.Tensor,
    rshift: torch.Tensor,
) -> float:
    """Max relative error between float ratio and ``multiplier / 2**rshift``."""

    real = real_multiplier.detach().to(torch.float64).reshape(-1)
    approx = _approx_multiplier_tensor(multiplier, rshift).reshape(-1)
    if real.numel() == 0:
        return 0.0
    denom = torch.clamp(real.abs(), min=1e-12)
    rel = (approx - real).abs() / denom
    mask = real.abs() > 1e-12
    if not torch.any(mask):
        return 0.0
    return float(rel[mask].max().item())


def _quantize_weighted_bias(
    qmodule: nn.Module,
    *,
    base_cls: type,
    device: torch.device,
) -> Optional[torch.Tensor]:
    bias = getattr(qmodule, "bias", None)
    if bias is None:
        return None
    if not _is_weighted_module(base_cls):
        return None

    iq = qmodule.input_quantizers[0] if qmodule.input_quantizers else None
    pq = getattr(qmodule, "param_quantizers", None)
    wq = pq["weight"] if pq is not None and "weight" in pq else None
    if not isinstance(iq, QuantizerBase) or not iq.is_initialized():
        return None
    if not isinstance(wq, QuantizerBase) or not wq.is_initialized():
        return None
    x_enc = iq.get_encodings()
    w_enc = wq.get_encodings()
    if not isinstance(x_enc, AffineEncoding) or not isinstance(w_enc, AffineEncoding):
        return None

    x_scale = x_enc.scale.to(device=device, dtype=torch.float32)
    w_scale = w_enc.scale.to(device=device, dtype=torch.float32)
    if hasattr(qmodule, "_derive_bias_scale"):
        acc_scale = qmodule._derive_bias_scale(x_scale, w_scale)
        if acc_scale is None:
            return None
        ones = torch.ones_like(acc_scale, dtype=acc_scale.dtype, device=acc_scale.device)
        return quantize_bias_int32(bias, acc_scale, ones)
    return quantize_bias_int32(bias, x_scale, w_scale)


def _layer_freeze_report(
    qmodule: nn.Module,
    *,
    layer_name: str,
    base_cls: type,
    out_enc: OutputEncoding,
    real_m: torch.Tensor,
    error_threshold: float,
    bias_int32_path: Optional[str],
) -> dict[str, Any]:
    mult = out_enc.multiplier
    rsh = out_enc.rshift
    if mult is None or rsh is None:
        raise ValueError(f"Layer {layer_name!r} is missing multiplier/rshift.")

    rel_err = multiplier_relative_error(real_m, mult, rsh)
    report: dict[str, Any] = {
        "op": base_cls.__name__,
        "status": "ok",
        "multiplier_int16": output_encoding_to_dict(out_enc).get("multiplier"),
        "rshift_int8": output_encoding_to_dict(out_enc).get("rshift"),
        "relative_error": rel_err,
    }
    if rel_err > error_threshold:
        report["status"] = "warning"
        report["warning"] = (
            f"multiplier relative error {rel_err:.6f} exceeds threshold {error_threshold}."
        )
    if bias_int32_path is not None:
        report["bias_int32_path"] = bias_int32_path
    return report


def _iter_quant_modules_without_int16_export(
    model: nn.Module,
    *,
    exported_layer_names: set[str],
) -> Dict[str, str]:
    # pylint: disable=import-outside-toplevel
    from aimet_torch.fixed_point.export.v2_collect import collect_v2_int16_layer_record
    from aimet_torch.v2.nn.true_quant import QuantizationMixin

    skipped: Dict[str, str] = {}
    for name, module in model.named_modules():
        key = name or "<root>"
        if key in exported_layer_names:
            continue
        if not isinstance(module, QuantizationMixin):
            continue
        base_cls = QuantizationMixin.qcls_to_cls.get(type(module))
        if base_cls is None:
            skipped[key] = "skipped: not a dispatchable QuantizationMixin"
            continue
        try:
            get_fixed_kernel(base_cls)
        except KernelNotFoundError:
            skipped[key] = f"skipped: no fixed kernel for {base_cls.__name__}"
            continue
        if collect_v2_int16_layer_record(module, key) is None:
            skipped[key] = "skipped: encodings not ready for INT16 export"
    return skipped


def freeze_int16_fixed(
    sim_model: Any,
    output_path: str,
    *,
    write_binaries: bool = True,
    error_threshold: float = 0.005,
    aimet_encoding_path: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Freeze INT16 fixed-point parameters and write deployable sidecar JSON.

    Walks calibrated v2 quantized modules, derives ``multiplier`` / ``rshift`` (and
    optional ``bias_int32`` binaries), writes ``output_path``, and returns a per-layer
    report dict suitable for CI or manual review.

    Parameters
    ----------
    sim_model:
        :class:`~aimet_torch.v2.quantsim.QuantizationSimModel` or its inner ``.model``.
    output_path:
        Path to ``*.int16.json`` sidecar (parent dir used for ``*.bias_int32.bin``).
    write_binaries:
        When True, Conv/Linear biases are written as little-endian int32 files.
    error_threshold:
        Relative multiplier approximation warning threshold (default 0.5%).
    aimet_encoding_path:
        Optional AIMET ``.encodings`` path for ONNX name hints in the sidecar.
    metadata:
        Optional metadata merged into the sidecar document.

    Returns
    -------
    dict
        ``{"layers": {layer_name: report, ...}, "skipped": {...}, "sidecar_path": str}``
    """

    # pylint: disable=import-outside-toplevel
    from aimet_torch.fixed_point.export.sidecar import build_int16_sidecar_document
    from aimet_torch.fixed_point.export.v2_collect import (
        collect_v2_int16_layers,
        derive_int16_real_multiplier,
    )

    model = _resolve_quantized_model(sim_model)
    device = torch.device("cpu")
    output_path = os.path.abspath(output_path)
    binary_dir = os.path.dirname(output_path)
    base_name = os.path.splitext(os.path.basename(output_path))[0]

    layer_reports: Dict[str, Any] = {}
    raw_layers = collect_v2_int16_layers(model)

    for layer_name, record in raw_layers.items():
        out_enc: OutputEncoding = record["output_encoding"]
        # pylint: disable=import-outside-toplevel
        from aimet_torch.v2.nn.true_quant import QuantizationMixin

        qmodule = dict(model.named_modules())[layer_name]
        base_cls = QuantizationMixin.qcls_to_cls[type(qmodule)]
        real_m = derive_int16_real_multiplier(qmodule, base_cls=base_cls, device=device)
        if real_m is None:
            layer_reports[layer_name] = {
                "op": record.get("op"),
                "status": "error",
                "error": "could not derive real_multiplier",
            }
            continue

        bias_path: Optional[str] = None
        if write_binaries and _is_weighted_module(base_cls):
            bias_int32 = _quantize_weighted_bias(qmodule, base_cls=base_cls, device=device)
            if bias_int32 is not None:
                fname = f"{base_name}.{_safe_layer_filename(layer_name)}.bias_int32.bin"
                bias_path = os.path.join(binary_dir, fname)
                _write_int32_binary(bias_int32, bias_path)

        layer_reports[layer_name] = _layer_freeze_report(
            qmodule,
            layer_name=layer_name,
            base_cls=base_cls,
            out_enc=out_enc,
            real_m=real_m,
            error_threshold=error_threshold,
            bias_int32_path=bias_path,
        )

    skipped = _iter_quant_modules_without_int16_export(
        model,
        exported_layer_names=set(raw_layers.keys()),
    )
    for name, reason in skipped.items():
        layer_reports[name] = {"status": "skipped", "reason": reason}

    doc = build_int16_sidecar_document(
        model,
        aimet_encoding_path=aimet_encoding_path,
        metadata=metadata,
    )
    if write_binaries:
        for layer_name, report in layer_reports.items():
            bias_rel = report.get("bias_int32_path")
            if bias_rel and layer_name in doc.get("layers", {}):
                doc["layers"][layer_name]["bias_int32_path"] = os.path.relpath(
                    bias_rel,
                    start=binary_dir,
                )

    os.makedirs(binary_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, sort_keys=True)

    summary = {
        "sidecar_path": output_path,
        "layer_count": len(raw_layers),
        "skipped_count": len(skipped),
        "warning_count": sum(
            1 for r in layer_reports.values() if r.get("status") == "warning"
        ),
        "layers": layer_reports,
        "skipped": skipped,
    }
    return summary


def freeze_int16_fixed_report_only(
    sim_model: Any,
    *,
    error_threshold: float = 0.005,
) -> dict[str, Any]:
    """Run freeze diagnostics without writing files (for quick checks)."""

    # pylint: disable=import-outside-toplevel
    from aimet_torch.fixed_point.export.v2_collect import (
        collect_v2_int16_layers,
        derive_int16_real_multiplier,
    )
    from aimet_torch.v2.nn.true_quant import QuantizationMixin

    model = _resolve_quantized_model(sim_model)
    device = torch.device("cpu")
    layer_reports: Dict[str, Any] = {}

    for layer_name, record in collect_v2_int16_layers(model).items():
        qmodule = dict(model.named_modules())[layer_name]
        base_cls = QuantizationMixin.qcls_to_cls[type(qmodule)]
        out_enc = record["output_encoding"]
        real_m = derive_int16_real_multiplier(qmodule, base_cls=base_cls, device=device)
        if real_m is None or out_enc.multiplier is None or out_enc.rshift is None:
            layer_reports[layer_name] = {"status": "error", "op": record.get("op")}
            continue
        layer_reports[layer_name] = _layer_freeze_report(
            qmodule,
            layer_name=layer_name,
            base_cls=base_cls,
            out_enc=out_enc,
            real_m=real_m,
            error_threshold=error_threshold,
            bias_int32_path=None,
        )
    return {
        "layers": layer_reports,
        "skipped": _iter_quant_modules_without_int16_export(
            model,
            exported_layer_names=set(layer_reports.keys()),
        ),
    }

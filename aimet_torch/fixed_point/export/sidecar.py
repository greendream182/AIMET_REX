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
"""INT16 fixed-point sidecar JSON export / load / consistency checks."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn

from aimet_torch.fixed_point.encoding import OutputEncoding
from aimet_torch.fixed_point.encoding_export import (
    fixed_point_tensor_bundle,
    output_encoding_from_dict,
    output_encoding_to_dict,
)
from aimet_torch.fixed_point.export.v2_collect import collect_v2_int16_layers

INT16_SIDECAR_VERSION = "1.0.0-int16-fixed"


def _load_aimet_encoding_tensor_names(encodings_path: str) -> list[str]:
    with open(encodings_path, encoding="utf-8") as f:
        data = json.load(f)
    version = str(data.get("version", "0.6.1"))
    if version >= "2.0.0":
        return [
            str(entry["name"])
            for entry in data.get("encodings", [])
            if isinstance(entry, dict) and entry.get("name")
        ]
    activation = data.get("activation_encodings", [])
    if isinstance(activation, list):
        return [
            str(entry["name"])
            for entry in activation
            if isinstance(entry, dict) and entry.get("name")
        ]
    if isinstance(activation, dict):
        return [str(name) for name in activation.keys()]
    return []


def build_onnx_name_hints_for_layers(
    pytorch_layer_names: list[str],
    aimet_encoding_path: str,
) -> dict[str, list[str]]:
    """Best-effort map PyTorch module paths to AIMET/ONNX encoding tensor names."""

    tensor_names = _load_aimet_encoding_tensor_names(aimet_encoding_path)
    hints: dict[str, list[str]] = {}
    for layer in pytorch_layer_names:
        matches: list[str] = []
        for name in tensor_names:
            if name == layer or name.endswith(f".{layer}") or name.startswith(f"{layer}."):
                matches.append(name)
        if matches:
            hints[layer] = sorted(set(matches))
    return hints


def attach_onnx_name_hints(
    doc: dict[str, Any],
    aimet_encoding_path: str,
) -> dict[str, Any]:
    """Attach ``onnx_tensor_names`` hints to each layer bundle (in-place)."""

    layers = doc.get("layers", {})
    if not isinstance(layers, dict):
        return doc
    hints = build_onnx_name_hints_for_layers(list(layers.keys()), aimet_encoding_path)
    for layer_name, bundle in layers.items():
        if layer_name in hints:
            bundle["onnx_tensor_names"] = hints[layer_name]
    doc["onnx_name_hints_attached"] = bool(hints)
    return doc


def build_int16_sidecar_document(
    model: nn.Module,
    *,
    aimet_encoding_path: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build a deployable sidecar dict aligned with runtime INT16 kernels.

    Parameters
    ----------
    model:
        AIMET v2 model containing :class:`~aimet_torch.v2.nn.true_quant.QuantizationMixin` modules.
    aimet_encoding_path:
        Optional path to an existing ``.encodings`` file produced by QuantSim ONNX export
        (stored as reference metadata only).
    metadata:
        Optional user metadata merged into the top-level document.
    """

    raw_layers = collect_v2_int16_layers(model)
    layers: Dict[str, Any] = {}
    for layer_name, record in raw_layers.items():
        out_enc: OutputEncoding = record["output_encoding"]
        bundle = fixed_point_tensor_bundle(
            layer_name=layer_name,
            output_encoding=out_enc,
            pwl_json=record.get("pwl"),
            input_requants=record.get("input_requants"),
        )
        bundle["op"] = record.get("op")
        if record.get("clz") is not None:
            bundle["clz"] = record["clz"]
        if record.get("phase_fold") is not None:
            bundle["phase_fold"] = record["phase_fold"]
        layers[layer_name] = bundle

    doc: dict[str, Any] = {
        "version": INT16_SIDECAR_VERSION,
        "format": "aimet_rx_int16_fixed_sidecar",
        "layer_count": len(layers),
        "layers": layers,
    }
    if aimet_encoding_path is not None:
        doc["aimet_encoding_reference"] = aimet_encoding_path
        attach_onnx_name_hints(doc, aimet_encoding_path)
    if metadata:
        doc["metadata"] = metadata
    return doc


def default_int16_sidecar_path(export_dir: str, filename_prefix: str) -> str:
    """Standard sidecar path: ``{export_dir}/{prefix}.int16.json``."""

    return os.path.join(export_dir, f"{filename_prefix}.int16.json")


def export_int16_sidecar_json(
    model: nn.Module,
    path: str,
    *,
    aimet_encoding_path: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Write INT16 sidecar JSON to ``path`` and return the document."""

    doc = build_int16_sidecar_document(
        model,
        aimet_encoding_path=aimet_encoding_path,
        metadata=metadata,
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, sort_keys=True)
    return doc


def load_int16_sidecar_json(path: str) -> dict[str, Any]:
    """Load sidecar JSON written by :func:`export_int16_sidecar_json`."""

    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("format") != "aimet_rx_int16_fixed_sidecar":
        raise ValueError(
            f"Not an AIMET RX INT16 sidecar (format={doc.get('format')!r})."
        )
    version = doc.get("version")
    if version is not None and version != INT16_SIDECAR_VERSION:
        raise ValueError(
            f"Unsupported AIMET RX INT16 sidecar version {version!r}; "
            f"expected {INT16_SIDECAR_VERSION!r}."
        )
    if not isinstance(doc.get("layers"), dict):
        raise ValueError("sidecar['layers'] must be a mapping.")
    return doc


def _encoding_dicts_match(
    exported: dict[str, Any],
    runtime: dict[str, Any],
    *,
    scale_rtol: float = 1e-5,
    scale_atol: float = 1e-7,
) -> list[str]:
    mismatches: list[str] = []
    for key in ("scale", "zero_point", "qmin", "qmax"):
        if exported.get(key) != runtime.get(key):
            if key == "scale":
                exp = exported[key]
                run = runtime[key]
                if isinstance(exp, list) and isinstance(run, list):
                    if len(exp) != len(run):
                        mismatches.append(f"{key}: length {len(exp)} vs {len(run)}")
                    else:
                        for i, (e, r) in enumerate(zip(exp, run)):
                            if abs(e - r) > scale_atol + scale_rtol * max(abs(e), abs(r), 1e-12):
                                mismatches.append(f"{key}[{i}]: {e} vs {r}")
                elif isinstance(exp, (int, float)) and isinstance(run, (int, float)):
                    if abs(exp - run) > scale_atol + scale_rtol * max(abs(exp), abs(run), 1e-12):
                        mismatches.append(f"{key}: {exp} vs {run}")
                else:
                    mismatches.append(f"{key}: {exported.get(key)!r} vs {runtime.get(key)!r}")
            else:
                mismatches.append(f"{key}: {exported.get(key)!r} vs {runtime.get(key)!r}")

    for key in ("multiplier", "rshift"):
        if key in exported or key in runtime:
            if exported.get(key) != runtime.get(key):
                mismatches.append(f"{key}: {exported.get(key)!r} vs {runtime.get(key)!r}")
    return mismatches


def compare_sidecar_with_model(
    model: nn.Module,
    sidecar: Mapping[str, Any],
    *,
    scale_rtol: float = 1e-5,
    scale_atol: float = 1e-7,
) -> dict[str, Any]:
    """Re-derive runtime encodings and compare against a loaded sidecar.

    Returns a report dict with ``ok``, ``missing_in_sidecar``, ``extra_in_sidecar``,
    and per-layer ``mismatches``.
    """

    runtime_layers = collect_v2_int16_layers(model)
    named_modules = dict(model.named_modules())
    sidecar_layers = sidecar.get("layers", {})
    if not isinstance(sidecar_layers, dict):
        raise ValueError("sidecar['layers'] must be a mapping.")

    missing = sorted(set(runtime_layers) - set(sidecar_layers))
    extra = sorted(set(sidecar_layers) - set(runtime_layers))
    layer_reports: Dict[str, Any] = {}

    for name in sorted(set(runtime_layers) & set(sidecar_layers)):
        runtime_enc = runtime_layers[name]["output_encoding"]
        exported_dict = sidecar_layers[name].get("output_encoding", {})
        runtime_dict = output_encoding_to_dict(runtime_enc)
        mismatches = _encoding_dicts_match(
            exported_dict,
            runtime_dict,
            scale_rtol=scale_rtol,
            scale_atol=scale_atol,
        )
        if mismatches:
            layer_reports[name] = {"ok": False, "mismatches": mismatches}
        else:
            layer_reports[name] = {"ok": True}

        from aimet_torch.fixed_point.export.sidecar_loader import (  # noqa: WPS433
            get_int16_sidecar_extra,
        )

        module = named_modules.get(name)
        attached = get_int16_sidecar_extra(module) if module is not None else None

        if "pwl" in sidecar_layers[name]:
            has_pwl = "pwl" in runtime_layers.get(name, {}) or (
                attached is not None and "pwl_lut" in attached
            )
            if not has_pwl:
                layer_reports[name] = {
                    "ok": False,
                    "mismatches": ["pwl present in sidecar but not derivable at runtime"],
                }
        if "clz" in sidecar_layers[name]:
            has_clz = "clz" in runtime_layers.get(name, {}) or (
                attached is not None and "clz_lut" in attached
            )
            if not has_clz:
                layer_reports[name] = {
                    "ok": False,
                    "mismatches": ["clz present in sidecar but not derivable at runtime"],
                }

    ok = not missing and not extra and all(r.get("ok") for r in layer_reports.values())
    return {
        "ok": ok,
        "missing_in_sidecar": missing,
        "extra_in_sidecar": extra,
        "layers": layer_reports,
    }


def layers_from_sidecar(
    sidecar: Mapping[str, Any],
    *,
    device: Optional[torch.device] = None,
) -> Dict[str, OutputEncoding]:
    """Parse ``output_encoding`` fields from a sidecar into :class:`OutputEncoding` objects."""

    dev = device or torch.device("cpu")
    out: Dict[str, OutputEncoding] = {}
    for name, bundle in sidecar.get("layers", {}).items():
        enc_dict = bundle.get("output_encoding")
        if enc_dict is None:
            continue
        out[name] = output_encoding_from_dict(enc_dict, device=dev)
    return out

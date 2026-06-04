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
"""Load INT16 sidecar LUT payloads into runtime ``extra`` for ``dispatch_int16_fixed``."""

from __future__ import annotations

import json
import os
import weakref
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn

from aimet_torch.fixed_point.encoding import InputEncoding
from aimet_torch.fixed_point.offline.lut_gen import pwl_lut_from_json_dict

_INT16_SIDECAR_FORMAT = "aimet_rx_int16_fixed_sidecar"
_INT16_SIDECAR_VERSION = "1.0.0-int16-fixed"
_INT16_SIDECAR_PATH_ENV = "AIMET_RX_INT16_SIDECAR_PATH"

_INT16_SIDECAR_EXTRA_ATTR = "_int16_sidecar_extra"
_INT16_ONLINE_EXTRA_ATTR = "_int16_online_extra"
_ATTACHED_PATH_BY_MODEL: "weakref.WeakKeyDictionary[nn.Module, str]" = (
    weakref.WeakKeyDictionary()
)


def _parse_sidecar_document(sidecar: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(sidecar, (str, Path)):
        with open(sidecar, encoding="utf-8") as handle:
            doc = json.load(handle)
    else:
        doc = dict(sidecar)
    if doc.get("format") != _INT16_SIDECAR_FORMAT:
        raise ValueError(
            f"Not an AIMET RX INT16 sidecar (format={doc.get('format')!r})."
        )
    version = doc.get("version")
    if version is not None and version != _INT16_SIDECAR_VERSION:
        raise ValueError(
            f"Unsupported AIMET RX INT16 sidecar version {version!r}; "
            f"expected {_INT16_SIDECAR_VERSION!r}."
        )
    if not isinstance(doc.get("layers"), dict):
        raise ValueError("sidecar['layers'] must be a mapping.")
    return doc


def _input_encoding_from_sidecar_dict(data: Mapping[str, Any]) -> InputEncoding:
    return InputEncoding(
        scale=torch.tensor(data["scale"], dtype=torch.float32),
        zero_point=torch.tensor(data["zero_point"], dtype=torch.int32),
        qmin=int(data["qmin"]),
        qmax=int(data["qmax"]),
    )


def layer_bundle_to_runtime_extra(
    bundle: Mapping[str, Any],
    *,
    device: Optional[torch.device] = None,
) -> dict[str, Any]:
    """Convert one sidecar layer bundle to kernel ``extra`` keys.

    Populates ``pwl_lut``, ``pwl_input_encoding``, ``phase_fold``, ``clz_lut`` /
    ``clz_func_name`` when present in the bundle.
    """

    dev = device or torch.device("cpu")
    extra: dict[str, Any] = {}

    pwl_block = bundle.get("pwl")
    if "pwl" in bundle and not (isinstance(pwl_block, dict) and pwl_block):
        raise ValueError("sidecar layer 'pwl' must be a non-empty mapping.")
    if isinstance(pwl_block, dict) and pwl_block:
        func_name = next(iter(pwl_block))
        func_body = pwl_block[func_name]
        if not isinstance(func_body, Mapping):
            raise ValueError(f"sidecar layer 'pwl[{func_name}]' must be a mapping.")
        extra["pwl_lut"] = pwl_lut_from_json_dict(pwl_block, func_name)
        for key in ("thresholds", "q_b", "n_bx_total", "term_c", "input_zero_point", "output_qmin", "output_qmax"):
            if key in extra["pwl_lut"]:
                extra["pwl_lut"][key] = extra["pwl_lut"][key].to(dev)

        phase_fold = func_body.get("phase_fold") or bundle.get("phase_fold")
        if phase_fold is not None:
            extra["phase_fold"] = str(phase_fold)

        pwl_in = func_body.get("pwl_input_encoding")
        if isinstance(pwl_in, dict):
            enc = _input_encoding_from_sidecar_dict(pwl_in)
            extra["pwl_input_encoding"] = InputEncoding(
                scale=enc.scale.to(dev),
                zero_point=enc.zero_point.to(dev),
                qmin=enc.qmin,
                qmax=enc.qmax,
                axis=enc.axis,
            )

    clz_block = bundle.get("clz")
    if "clz" in bundle and not (isinstance(clz_block, dict) and clz_block):
        raise ValueError("sidecar layer 'clz' must be a non-empty mapping.")
    if isinstance(clz_block, dict) and clz_block:
        func_name = next(iter(clz_block))
        if not isinstance(clz_block[func_name], Mapping):
            raise ValueError(f"sidecar layer 'clz[{func_name}]' must be a mapping.")
        extra["clz_lut"] = clz_block[func_name]
        extra["clz_func_name"] = func_name

    return extra


def resolve_sidecar_layer_to_module_names(
    sidecar_layers: Mapping[str, Any],
    model: nn.Module,
) -> dict[str, str]:
    """Map sidecar layer keys to ``model.named_modules()`` paths (ONNX / suffix fallback)."""

    named = dict(model.named_modules())
    keys = list(named.keys())
    resolved: dict[str, str] = {}

    for sidecar_name, bundle in sidecar_layers.items():
        if sidecar_name in named:
            resolved[sidecar_name] = sidecar_name
            continue

        onnx_names = bundle.get("onnx_tensor_names") if isinstance(bundle, dict) else None
        if isinstance(onnx_names, list):
            for onnx_name in onnx_names:
                if onnx_name in named:
                    resolved[sidecar_name] = onnx_name
                    break
            if sidecar_name in resolved:
                continue

        candidates = [
            key
            for key in keys
            if key == sidecar_name or key.endswith(f".{sidecar_name}")
        ]
        if len(candidates) == 1:
            resolved[sidecar_name] = candidates[0]
        elif len(candidates) > 1:
            resolved[sidecar_name] = min(candidates, key=len)

    return resolved


def build_runtime_extra_by_layer(
    sidecar: Mapping[str, Any],
    *,
    device: Optional[torch.device] = None,
) -> dict[str, dict[str, Any]]:
    """Map ``sidecar['layers'][name]`` → runtime ``extra`` dict."""

    layers = sidecar.get("layers", {})
    if not isinstance(layers, dict):
        raise ValueError("sidecar['layers'] must be a mapping.")
    return {
        name: layer_bundle_to_runtime_extra(bundle, device=device)
        for name, bundle in layers.items()
    }


def attach_int16_sidecar_to_model(
    model: nn.Module,
    sidecar: str | Path | Mapping[str, Any],
    *,
    strict: bool = False,
    device: Optional[torch.device] = None,
) -> dict[str, dict[str, Any]]:
    """Attach per-module ``_int16_sidecar_extra`` used by :func:`dispatch_int16_fixed`.

    Returns the ``{layer_name: extra}`` mapping that was attached. Layers listed in
    the sidecar but missing from ``model`` are skipped unless ``strict=True``.
    """

    doc = _parse_sidecar_document(sidecar)

    layers = doc.get("layers", {})
    if not isinstance(layers, dict):
        raise ValueError("sidecar['layers'] must be a mapping.")

    extras_by_layer = build_runtime_extra_by_layer(doc, device=device)
    name_map = resolve_sidecar_layer_to_module_names(layers, model)
    named = dict(model.named_modules())

    missing_in_model: list[str] = []
    for sidecar_name, extra in extras_by_layer.items():
        module_name = name_map.get(sidecar_name)
        if module_name is None or module_name not in named:
            missing_in_model.append(sidecar_name)
            continue
        setattr(named[module_name], _INT16_SIDECAR_EXTRA_ATTR, extra)

    if strict and missing_in_model:
        raise KeyError(
            "Sidecar layers not found on model: " + ", ".join(sorted(missing_in_model))
        )

    return extras_by_layer


def detach_int16_sidecar_from_model(model: nn.Module) -> None:
    """Remove ``_int16_sidecar_extra`` from all modules."""

    for module in model.modules():
        if hasattr(module, _INT16_SIDECAR_EXTRA_ATTR):
            delattr(module, _INT16_SIDECAR_EXTRA_ATTR)
    _ATTACHED_PATH_BY_MODEL.pop(model, None)


def get_int16_sidecar_extra(module: nn.Module) -> Optional[dict[str, Any]]:
    """Return attached sidecar ``extra`` for one module, if any."""

    value = getattr(module, _INT16_SIDECAR_EXTRA_ATTR, None)
    return value if isinstance(value, dict) else None


def _move_lut_payload_to_device(payload: Any, device: torch.device) -> Any:
    if isinstance(payload, dict):
        return {
            key: _move_lut_payload_to_device(value, device)
            for key, value in payload.items()
        }
    if torch.is_tensor(payload):
        return payload.to(device)
    return payload


def get_int16_online_extra(
    module: nn.Module,
    *,
    device: Optional[torch.device] = None,
) -> Optional[dict[str, Any]]:
    """Return module-local LUT cache populated by online INT16 dispatch."""

    cached = getattr(module, _INT16_ONLINE_EXTRA_ATTR, None)
    if not isinstance(cached, dict) or not cached:
        return None
    if device is None:
        return dict(cached)
    return {
        key: _move_lut_payload_to_device(value, device)
        for key, value in cached.items()
    }


def merge_int16_online_extra(module: nn.Module, updates: Mapping[str, Any]) -> None:
    """Persist generated PWL/CLZ payloads on ``module`` for later forwards."""

    if not updates:
        return
    bucket = getattr(module, _INT16_ONLINE_EXTRA_ATTR, None)
    if not isinstance(bucket, dict):
        bucket = {}
        setattr(module, _INT16_ONLINE_EXTRA_ATTR, bucket)
    for key, value in updates.items():
        if key.endswith("_metrics") or key.endswith("_error"):
            continue
        bucket[key] = value


def clear_int16_online_extra(model: nn.Module) -> None:
    """Drop online LUT cache from all submodules (e.g. after encoding change)."""

    for module in model.modules():
        if hasattr(module, _INT16_ONLINE_EXTRA_ATTR):
            delattr(module, _INT16_ONLINE_EXTRA_ATTR)


def maybe_attach_int16_sidecar_from_env(
    model: nn.Module,
    *,
    env_var: str = _INT16_SIDECAR_PATH_ENV,
    strict: bool = False,
    device: Optional[torch.device] = None,
) -> Optional[str]:
    """Attach sidecar LUTs when ``AIMET_RX_INT16_SIDECAR_PATH`` points to a JSON file.

    Idempotent per ``model`` instance: re-attaches only when the path changes.
    Returns the path used, or ``None`` when the env var is unset/empty.
    """

    path = os.environ.get(env_var, "").strip()
    if not path:
        return None
    resolved = str(Path(path).expanduser().resolve())
    if not Path(resolved).is_file():
        raise FileNotFoundError(f"{env_var}={resolved!r} is not a file.")

    prior = _ATTACHED_PATH_BY_MODEL.get(model)
    if prior == resolved:
        if any(get_int16_sidecar_extra(m) is not None for m in model.modules()):
            return resolved

    attach_int16_sidecar_to_model(model, resolved, strict=strict, device=device)
    _ATTACHED_PATH_BY_MODEL[model] = resolved
    return resolved

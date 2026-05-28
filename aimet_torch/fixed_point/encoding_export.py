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
"""JSON-serializable views of encodings for sidecar export / round-trip tests."""

from __future__ import annotations

from typing import Any, List, Optional, Union

import torch

from aimet_torch.fixed_point.encoding import FixedScaleEncoding, InputEncoding, OutputEncoding


def _tensor_float_json(t: torch.Tensor) -> Union[float, List[float]]:
    flat = t.detach().cpu().reshape(-1).to(torch.float32)
    if flat.numel() == 1:
        return float(flat.item())
    return flat.tolist()


def _tensor_int_json(t: torch.Tensor) -> Union[int, List[int]]:
    flat = t.detach().cpu().reshape(-1)
    if flat.numel() == 1:
        return int(flat.item())
    return [int(x) for x in flat.tolist()]


def input_encoding_to_dict(enc: InputEncoding) -> dict[str, Any]:
    """Serialize :class:`InputEncoding` to plain dict (ONNX sidecar / JSON friendly)."""

    return {
        "scale": _tensor_float_json(enc.scale),
        "zero_point": _tensor_int_json(enc.zero_point),
        "qmin": int(enc.qmin),
        "qmax": int(enc.qmax),
        "axis": enc.axis,
    }


def fixed_scale_encoding_to_dict(enc: FixedScaleEncoding) -> dict[str, Any]:
    """Serialize :class:`FixedScaleEncoding` including ``m_int16`` / ``rshift``."""

    d = {
        "m_int16": _tensor_int_json(enc.m_int16),
        "rshift": _tensor_int_json(enc.rshift),
        "zero_point": _tensor_int_json(enc.zero_point),
        "qmin": int(enc.qmin),
        "qmax": int(enc.qmax),
        "axis": enc.axis,
    }
    if enc.scale_fp_legacy is not None:
        d["scale"] = _tensor_float_json(enc.scale_fp_legacy)
    return d


def fixed_scale_encoding_from_dict(
    data: dict[str, Any], *, device: Optional[torch.device] = None
) -> FixedScaleEncoding:
    """Rebuild :class:`FixedScaleEncoding`; requires ``m_int16`` and ``rshift`` keys."""

    if "m_int16" not in data or "rshift" not in data:
        raise ValueError(
            "fixed_scale encoding dict must include 'm_int16' and 'rshift'. "
            "Run convert_encodings_to_fixed_scale(sim) after PTQ/QAT freeze."
        )
    dev = device or torch.device("cpu")
    m = _jsonable_to_tensor(data["m_int16"], dtype=torch.int16, device=dev)
    r = _jsonable_to_tensor(data["rshift"], dtype=torch.int8, device=dev)
    zp = _jsonable_to_tensor(data["zero_point"], dtype=torch.int32, device=dev)
    scale_legacy = None
    if "scale" in data:
        scale_legacy = _jsonable_to_tensor(data["scale"], dtype=torch.float32, device=dev)
    return FixedScaleEncoding(
        m_int16=m,
        rshift=r,
        zero_point=zp,
        qmin=int(data["qmin"]),
        qmax=int(data["qmax"]),
        axis=data.get("axis"),
        scale_fp_legacy=scale_legacy,
    )


def output_encoding_to_dict(enc: OutputEncoding) -> dict[str, Any]:
    """Serialize :class:`OutputEncoding` including optional multiplier / shift."""

    d = input_encoding_to_dict(enc)
    if enc.multiplier is not None:
        d["multiplier"] = _tensor_int_json(enc.multiplier)
    if enc.rshift is not None:
        d["rshift"] = _tensor_int_json(enc.rshift)
    return d


def _jsonable_to_tensor(
    value: Union[float, int, List[Any]],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(value, list):
        return torch.tensor(value, dtype=dtype, device=device)
    return torch.tensor(value, dtype=dtype, device=device)


def input_encoding_from_dict(
    data: dict[str, Any], *, device: Optional[torch.device] = None
) -> InputEncoding:
    """Rebuild :class:`InputEncoding` from :func:`input_encoding_to_dict`."""

    dev = device or torch.device("cpu")
    scale = _jsonable_to_tensor(data["scale"], dtype=torch.float32, device=dev)
    zp = _jsonable_to_tensor(data["zero_point"], dtype=torch.int32, device=dev)
    return InputEncoding(
        scale=scale,
        zero_point=zp,
        qmin=int(data["qmin"]),
        qmax=int(data["qmax"]),
        axis=data.get("axis"),
    )


def output_encoding_from_dict(
    data: dict[str, Any], *, device: Optional[torch.device] = None
) -> OutputEncoding:
    """Rebuild :class:`OutputEncoding` from :func:`output_encoding_to_dict`."""

    dev = device or torch.device("cpu")
    base = input_encoding_from_dict(data, device=dev)
    mult = data.get("multiplier")
    rsh = data.get("rshift")
    return OutputEncoding(
        scale=base.scale,
        zero_point=base.zero_point,
        qmin=base.qmin,
        qmax=base.qmax,
        axis=base.axis,
        multiplier=_jsonable_to_tensor(mult, dtype=torch.int16, device=dev) if mult is not None else None,
        rshift=_jsonable_to_tensor(rsh, dtype=torch.int8, device=dev) if rsh is not None else None,
    )


def fixed_point_tensor_bundle(
    *,
    layer_name: str,
    output_encoding: OutputEncoding,
    pwl_json: dict[str, Any] | None = None,
    input_requants: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Bundle a layer name with encoding dict and optional PWL sidecar (same quantization block)."""

    out: dict[str, Any] = {
        "layer": layer_name,
        "output_encoding": output_encoding_to_dict(output_encoding),
    }
    if pwl_json is not None:
        out["pwl"] = pwl_json
    if input_requants is not None:
        out["input_requants"] = input_requants
    return out

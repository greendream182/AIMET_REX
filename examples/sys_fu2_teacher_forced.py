"""Shared teacher-forced / chained SQNR helpers for SYS-FU-2 diagnostics."""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn

from aimet_torch.fixed_point import ExecutionMode, quant_execution_mode
from aimet_torch.fixed_point.metrics.isolated import (
    _quantize_with_carrier,
    _to_float,
    per_layer_isolated_cosine,
)


def sqnr_db(ref: torch.Tensor, cand: torch.Tensor) -> float:
    diff = ref.reshape(-1).float() - cand.reshape(-1).float()
    sig = ref.reshape(-1).float().pow(2).mean().item()
    noise = diff.pow(2).mean().item()
    if noise <= 0 or sig <= 0:
        return float("inf")
    return float(10.0 * math.log10(sig / noise))


def find_modules(
    model: nn.Module,
    suffixes: Iterable[str],
) -> dict[str, tuple[str, nn.Module]]:
    found: dict[str, tuple[str, nn.Module]] = {}
    for qualname, mod in model.named_modules():
        for suffix in suffixes:
            if qualname == suffix or qualname.endswith("." + suffix):
                found[suffix] = (qualname, mod)
    return found


@torch.no_grad()
def teacher_forced_sqnr(
    model: nn.Module,
    sample: torch.Tensor,
    suffixes: Iterable[str],
    *,
    ref_mode: ExecutionMode = ExecutionMode.FP32_QDQ,
    cand_mode: ExecutionMode = ExecutionMode.INT16_FIXED_EVAL,
) -> dict[str, float]:
    """Isolated SQNR via :func:`per_layer_isolated_cosine` (suffix filter)."""

    del ref_mode
    wanted = set(suffixes)
    rows = per_layer_isolated_cosine(model, sample, cand_mode=cand_mode)
    out: dict[str, float] = {}
    for row in rows:
        name = row["module"]
        for suffix in wanted:
            if name == suffix or name.endswith("." + suffix):
                out[suffix] = float(row["sqnr_db"])
    return out


@torch.no_grad()
def _teacher_forced_sqnr_direct(
    model: nn.Module,
    sample: torch.Tensor,
    suffixes: Iterable[str],
    *,
    ref_mode: ExecutionMode = ExecutionMode.FP32_QDQ,
    cand_mode: ExecutionMode = ExecutionMode.INT16_FIXED_EVAL,
) -> dict[str, float]:
    """Direct re-invoke protocol (fallback / debug)."""

    targets = find_modules(model, suffixes)
    cached_in: dict[str, torch.Tensor] = {}
    cached_ref: dict[str, torch.Tensor] = {}

    def _pre(name: str):
        def _fn(_mod, inputs):
            if inputs and isinstance(inputs[0], torch.Tensor):
                cached_in[name] = inputs[0].detach().clone()
        return _fn

    def _fwd(name: str):
        def _fn(_mod, _ins, out):
            t = _to_float(out)
            if t is not None:
                cached_ref[name] = t
        return _fn

    handles = []
    for _suffix, (name, mod) in targets.items():
        handles.append(mod.register_forward_pre_hook(_pre(name)))
        handles.append(mod.register_forward_hook(_fwd(name)))
    with quant_execution_mode(ref_mode):
        model(sample)
    for h in handles:
        h.remove()

    carriers: dict[str, dict] = {}

    def _carrier_pre(name: str):
        def _fn(_mod, inputs):
            if not inputs:
                return
            first = inputs[0]
            from aimet_torch.fixed_point.tensor import FixedPointSimTensor

            if isinstance(first, FixedPointSimTensor):
                carriers[name] = {
                    "scale": first.scale.detach().clone(),
                    "zero_point": first.zero_point.detach().clone(),
                    "qmin": first.qmin,
                    "qmax": first.qmax,
                    "axis": first.axis,
                }
        return _fn

    carrier_handles = [
        mod.register_forward_pre_hook(_carrier_pre(name))
        for _suffix, (name, mod) in targets.items()
    ]
    try:
        with quant_execution_mode(cand_mode):
            model(sample)
    except (RuntimeError, TypeError, AttributeError):
        pass
    finally:
        for h in carrier_handles:
            h.remove()

    out: dict[str, float] = {}
    for suffix, (name, mod) in targets.items():
        if name not in cached_in or name not in cached_ref:
            continue
        x_args = (cached_in[name],)
        if name in carriers:
            try:
                x0 = _quantize_with_carrier(cached_in[name], carriers[name])
                x_args = (x0,)
            except (RuntimeError, TypeError, ValueError):
                pass
        else:
            iq = getattr(mod, "input_quantizers", None)
            if iq is not None and iq[0] is not None and iq[0].is_initialized():
                from aimet_torch.fixed_point.boundary_quantize import (
                    quantize_boundary_from_affine,
                )

                x_enc = iq[0].get_encodings()
                if x_enc is not None:
                    try:
                        x_args = (
                            quantize_boundary_from_affine(
                                cached_in[name], x_enc
                            ).to(cached_in[name].device),
                        )
                    except (RuntimeError, TypeError, ValueError):
                        pass
        with quant_execution_mode(cand_mode):
            y_cand = _to_float(mod(*x_args))
        if y_cand is None or y_cand.shape != cached_ref[name].shape:
            continue
        out[suffix] = sqnr_db(cached_ref[name], y_cand)
    return out


@torch.no_grad()
def chained_sqnr(
    model: nn.Module,
    sample: torch.Tensor,
    suffixes: Iterable[str],
    *,
    ref_mode: ExecutionMode = ExecutionMode.FP32_QDQ,
    cand_mode: ExecutionMode = ExecutionMode.INT16_FIXED_EVAL,
) -> dict[str, float]:
    """Chained SQNR: full-model forward under cand_mode vs ref at each hook."""

    targets = find_modules(model, suffixes)
    ref_out: dict[str, torch.Tensor] = {}
    cand_out: dict[str, torch.Tensor] = {}

    def _fwd(name: str, store: dict[str, torch.Tensor]):
        def _fn(_mod, _ins, out):
            t = _to_float(out)
            if t is not None:
                store[name] = t
        return _fn

    handles = [
        mod.register_forward_hook(_fwd(name, ref_out))
        for _suffix, (name, mod) in targets.items()
    ]
    with quant_execution_mode(ref_mode):
        model(sample)
    for h in handles:
        h.remove()

    handles = [
        mod.register_forward_hook(_fwd(name, cand_out))
        for _suffix, (name, mod) in targets.items()
    ]
    try:
        with quant_execution_mode(cand_mode):
            model(sample)
    except (RuntimeError, TypeError, AttributeError):
        pass
    for h in handles:
        h.remove()

    out: dict[str, float] = {}
    for suffix, (name, _mod) in targets.items():
        if name not in ref_out or name not in cand_out:
            continue
        if ref_out[name].shape != cand_out[name].shape:
            continue
        out[suffix] = sqnr_db(ref_out[name], cand_out[name])
    return out

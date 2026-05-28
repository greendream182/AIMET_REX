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
"""INT16 fixed-point execution profiling (per-layer INT16 carrier statistics)."""

from __future__ import annotations

import json
from typing import Any, Dict

import torch
import torch.nn as nn

from aimet_torch.fixed_point.metrics.flags import (
    int16_eval_debug_float_allowed,
    set_int16_eval_debug_float_allowed,
)
from aimet_torch.fixed_point.tensor import Int16QuantizedTensor, align_stat_rank


def _collect_int16_tensors(obj: Any, out: list[Int16QuantizedTensor]) -> None:
    if isinstance(obj, Int16QuantizedTensor):
        out.append(obj)
        return
    if isinstance(obj, tuple):
        for item in obj:
            _collect_int16_tensors(item, out)
        return
    if isinstance(obj, list):
        for item in obj:
            _collect_int16_tensors(item, out)
        return
    if isinstance(obj, dict):
        for item in obj.values():
            _collect_int16_tensors(item, out)


def _scalar_requant_params_from_module(module: nn.Module) -> tuple[int, int]:
    """Read offline multiplier/rshift when ``module`` is a v2 INT16-dispatchable layer."""

    try:
        # pylint: disable=import-outside-toplevel
        from aimet_torch.fixed_point.export.v2_collect import collect_v2_int16_layer_record
        from aimet_torch.v2.nn.true_quant import QuantizationMixin

        if not isinstance(module, QuantizationMixin):
            return 0, 0
        record = collect_v2_int16_layer_record(module, "")
        if record is None:
            return 0, 0
        enc = record["output_encoding"]
        mult = enc.multiplier
        rsh = enc.rshift
        if mult is None or rsh is None:
            return 0, 0
        mult_i = int(mult.detach().cpu().reshape(-1).max().item())
        rsh_i = int(rsh.detach().cpu().reshape(-1).max().item())
        return mult_i, rsh_i
    except Exception:  # noqa: BLE001
        return 0, 0


def _layer_stats_from_tensor(t: Int16QuantizedTensor) -> Dict[str, Any]:
    q = t.int_repr.detach()
    centered = t.centered_int32().detach()
    qf = q.to(torch.float32)
    zp = align_stat_rank(t.zero_point.to(torch.float32), q)
    at_low = (qf <= float(t.qmin)).float().mean().item()
    at_high = (qf >= float(t.qmax)).float().mean().item()
    saturation_ratio = max(0.0, min(1.0, at_low + at_high))
    acc_max_abs = int(torch.max(torch.abs(centered)).item())
    dyn = float(max(abs(t.qmin), abs(t.qmax)))
    centered_f = centered.to(torch.float32)
    bit_utilization = float(torch.mean(torch.abs(centered_f)).item() / dyn) if dyn > 0 else 0.0
    return {
        "saturation_ratio": float(saturation_ratio),
        "acc_max_abs": acc_max_abs,
        "multiplier": 0,
        "rshift": 0,
        "bit_utilization": float(bit_utilization),
    }


class FixedPointProfiler:
    """Context manager that enables ``Int16QuantizedTensor.to_float`` in eval and records per-layer stats."""

    def __init__(self, model: nn.Module):
        self._model = model
        self._handles: list[Any] = []
        self._records: Dict[str, Dict[str, Any]] = {}
        self._merged: Dict[str, Dict[str, Any]] = {}
        self._prev_allow_debug_float: bool = False

    def __enter__(self) -> "FixedPointProfiler":
        self._prev_allow_debug_float = int16_eval_debug_float_allowed()
        set_int16_eval_debug_float_allowed(True)
        self._records.clear()
        self._merged.clear()

        def make_hook(layer: str, module: nn.Module):
            def _hook(_mod, _inp, out) -> None:
                tensors: list[Int16QuantizedTensor] = []
                _collect_int16_tensors(out, tensors)
                if not tensors:
                    return
                agg_sat = 0.0
                agg_bit = 0.0
                acc_max = 0
                for t in tensors:
                    st = _layer_stats_from_tensor(t)
                    agg_sat = max(agg_sat, st["saturation_ratio"])
                    agg_bit = max(agg_bit, st["bit_utilization"])
                    acc_max = max(acc_max, st["acc_max_abs"])
                mult_i, rsh_i = _scalar_requant_params_from_module(module)
                self._records[layer] = {
                    "saturation_ratio": agg_sat,
                    "acc_max_abs": acc_max,
                    "multiplier": mult_i,
                    "rshift": rsh_i,
                    "bit_utilization": agg_bit,
                }

            return _hook

        for mod_name, module in self._model.named_modules():
            layer = mod_name if mod_name else "<root>"
            h = module.register_forward_hook(make_hook(layer, module))
            self._handles.append(h)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        set_int16_eval_debug_float_allowed(self._prev_allow_debug_float)
        self._merged = dict(self._records)
        return None

    def to_dict(self) -> dict[str, Any]:
        """Return per-module statistics collected during the last context (same keys as INTERFACE.md §8.1)."""

        return dict(self._merged)

    def to_json(self, path: str) -> None:
        """Write :meth:`to_dict` as UTF-8 JSON."""

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

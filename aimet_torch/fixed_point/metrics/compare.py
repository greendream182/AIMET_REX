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
"""Compare tensor outputs across quantization execution modes."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from aimet_torch.fixed_point import ExecutionMode, quant_execution_mode
from aimet_torch.fixed_point.metrics.accuracy import compute_pair_metrics
from aimet_torch.fixed_point.metrics.flags import int16_eval_allow_debug_float
from aimet_torch.fixed_point.metrics.profiler import FixedPointProfiler
from aimet_torch.fixed_point.tensor import Int16QuantizedTensor

# Default mode order for reports: reference first, then QDQ variants, then INT16 sim.
DEFAULT_COMPARE_MODES: Tuple[ExecutionMode, ...] = (
    ExecutionMode.FP32_QDQ,
    ExecutionMode.FP16_QDQ,
    ExecutionMode.FIXED_SCALE_QDQ,
    ExecutionMode.INT16_FIXED_EVAL,
)


def _unwrap_output(out: Any) -> Tuple[torch.Tensor, Optional[Int16QuantizedTensor]]:
    """Return float tensor for metrics and optional INT16 carrier for LSB metrics."""

    if isinstance(out, Int16QuantizedTensor):
        with int16_eval_allow_debug_float():
            return out.to_float(), out
    if isinstance(out, torch.Tensor):
        return out, None
    if hasattr(out, "dequantize"):
        return out.dequantize(), None
    raise TypeError(f"Unsupported model output type: {type(out).__name__}")


def _normalize_inputs(
    input_data: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
) -> Tuple[torch.Tensor, ...]:
    if isinstance(input_data, torch.Tensor):
        return (input_data,)
    return input_data


def _capture_leaf_outputs(
    model: nn.Module,
    inputs: Tuple[torch.Tensor, ...],
    mode: ExecutionMode,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Optional[Int16QuantizedTensor]]:
    """Forward under ``mode``; return leaf-module float outputs and final output."""

    captured: Dict[str, torch.Tensor] = {}
    handles: List[Any] = []

    def _make_hook(layer_name: str):
        def _hook(_module, _inp, out):
            tensor = _unwrap_output(out)[0]
            captured[layer_name] = tensor.detach().cpu()

        return _hook

    for name, module in model.named_modules():
        label = name if name else "<root>"
        if name and list(module.children()):
            continue
        handles.append(module.register_forward_hook(_make_hook(label)))

    try:
        with quant_execution_mode(mode):
            with torch.no_grad():
                raw = model(*inputs)
        y_float, carrier = _unwrap_output(raw)
        return captured, y_float, carrier
    finally:
        for handle in handles:
            handle.remove()


def _pairwise_metrics_block(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    carrier: Optional[Int16QuantizedTensor],
    metrics: Tuple[str, ...],
) -> Dict[str, float]:
    """Compare tensors; use ``carrier`` only when its grid matches ``reference`` shape."""

    use_carrier = carrier
    if use_carrier is not None and tuple(use_carrier.int_repr.shape) != tuple(reference.shape):
        use_carrier = None

    pair_metrics = compute_pair_metrics(
        reference,
        candidate,
        scale=use_carrier.scale if use_carrier is not None else None,
        zero_point=use_carrier.zero_point if use_carrier is not None else None,
        qmin=use_carrier.qmin if use_carrier is not None else -32768,
        qmax=use_carrier.qmax if use_carrier is not None else 32767,
        candidate_int_repr=use_carrier.int_repr if use_carrier is not None else None,
    )
    return {key: pair_metrics[key] for key in metrics if key in pair_metrics}


def write_per_layer_csv(report: Dict[str, Any], path: Union[str, Path]) -> None:
    """Write ``report['per_layer']`` to CSV (spec 12)."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "layer",
        "pair",
        "cosine_similarity",
        "max_abs_error",
        "rmse",
        "sqnr_db",
        "max_error_lsb",
        "max_error_lsb_float",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for layer, pairs in report.get("per_layer", {}).items():
            for pair_key, metrics in pairs.items():
                row = {"layer": layer, "pair": pair_key, **metrics}
                writer.writerow(row)


def compare_modes(
    model: nn.Module,
    input_data: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    modes: Optional[List[ExecutionMode]] = None,
    metrics: Tuple[str, ...] = (
        "max_abs_error",
        "rmse",
        "cosine_similarity",
        "sqnr_db",
        "max_error_lsb",
        "max_error_lsb_float",
    ),
    *,
    include_per_layer: bool = False,
    include_saturation: bool = False,
    max_layers: Optional[int] = 256,
) -> Dict[str, Any]:
    """Run ``model`` under each mode and compare float outputs to the first mode's output.

    If ``modes`` is omitted, uses :data:`DEFAULT_COMPARE_MODES` (``fp32_qdq`` first as reference).

    When ``include_per_layer`` is True, leaf-module activations are compared vs the reference mode.
    When ``include_saturation`` is True, records :class:`FixedPointProfiler` stats for
    ``int16_fixed_eval`` (if present in ``modes``).
    """

    if modes is None:
        modes = list(DEFAULT_COMPARE_MODES)

    model.eval()
    inputs = _normalize_inputs(input_data)

    layer_captures: List[Dict[str, torch.Tensor]] = []
    floats: List[torch.Tensor] = []
    int16_carriers: List[Optional[Int16QuantizedTensor]] = []

    for mode in modes:
        if include_per_layer:
            layers, y_float, carrier = _capture_leaf_outputs(model, inputs, mode)
            layer_captures.append(layers)
        else:
            with quant_execution_mode(mode):
                with torch.no_grad():
                    raw = model(*inputs)
                y_float, carrier = _unwrap_output(raw)
            layer_captures.append({})
        floats.append(y_float)
        int16_carriers.append(carrier)

    ref = floats[0]
    result: Dict[str, Any] = {"modes": [m.value for m in modes], "pairwise": {}}

    for i in range(1, len(floats)):
        cur = floats[i]
        if cur.shape != ref.shape:
            raise ValueError(
                f"Output shape mismatch between {modes[0].value} and {modes[i].value}."
            )
        pair = _pairwise_metrics_block(ref, cur, int16_carriers[i], metrics)
        result["pairwise"][f"{modes[0].value}_vs_{modes[i].value}"] = pair

    if include_per_layer and layer_captures:
        ref_layers = layer_captures[0]
        per_layer: Dict[str, Dict[str, Dict[str, float]]] = {}
        layer_names = sorted(ref_layers.keys())
        if max_layers is not None:
            layer_names = layer_names[: max_layers]
        for layer in layer_names:
            ref_t = ref_layers[layer]
            per_layer[layer] = {}
            for i in range(1, len(modes)):
                cur_map = layer_captures[i]
                if layer not in cur_map:
                    continue
                cur_t = cur_map[layer]
                if cur_t.shape != ref_t.shape:
                    continue
                pair_key = f"{modes[0].value}_vs_{modes[i].value}"
                # Per-layer activations differ in shape from final logits; never reuse the
                # model-output INT16 carrier here (would raise shape mismatch on LSB).
                per_layer[layer][pair_key] = _pairwise_metrics_block(
                    ref_t,
                    cur_t,
                    None,
                    metrics,
                )
        result["per_layer"] = per_layer

    if include_saturation and ExecutionMode.INT16_FIXED_EVAL in modes:
        with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
            with FixedPointProfiler(model) as prof:
                with torch.no_grad():
                    model(*inputs)
            result["saturation"] = prof.to_dict()

    return result


def compare_modes_against_reference(
    model: nn.Module,
    input_data: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    reference_mode: ExecutionMode,
    other_modes: List[ExecutionMode],
    metrics: Tuple[str, ...] = (
        "max_abs_error",
        "rmse",
        "cosine_similarity",
        "sqnr_db",
        "max_error_lsb",
        "max_error_lsb_float",
    ),
) -> Dict[str, Any]:
    """Convenience wrapper: first run is ``reference_mode``, then each of ``other_modes``."""

    order = [reference_mode] + list(other_modes)
    return compare_modes(model, input_data, order, metrics=metrics)

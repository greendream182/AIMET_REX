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
"""Per-layer *chained* (accumulated) cosine across execution modes.

This is the natural sibling of :mod:`isolated`: both probe every leaf
quantized module via forward hooks, but here we let each mode run the
**full network** forward and capture each module's output in-place — so
the per-layer cosine includes upstream accumulation noise.

Use :func:`per_layer_chained_cosine` together with
:func:`per_layer_isolated_cosine` to distinguish accumulated error from
intrinsic per-layer quantization noise; see ``doc/FixedPoint_Quantization_Design_v2.md``.

The function is **model-agnostic**: it accepts any ``(model, inputs)``
pair, mirroring :func:`per_layer_isolated_cosine`'s signature so ImageNet,
ResNet, BERT, etc. can all reuse it.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from aimet_torch.fixed_point import ExecutionMode, quant_execution_mode
from aimet_torch.fixed_point.metrics.accuracy import p99_abs_error
from aimet_torch.fixed_point.metrics.flags import int16_eval_allow_debug_float
from aimet_torch.fixed_point.tensor import Int16QuantizedTensor

_LOGGER = logging.getLogger(__name__)


def _default_filter(_name: str, module: nn.Module) -> bool:
    """Match leaf qmodules that own an output quantizer slot.

    Mirrors :func:`isolated._default_filter`: standalone quantizers
    (``Quantize`` / ``QuantizeDequantize``) are excluded so the table only
    lists user-facing layers.
    """

    try:  # pylint: disable=import-outside-toplevel
        from aimet_torch.v2.quantization.base.quantizer import QuantizerBase
    except ImportError:  # pragma: no cover - v2 always available at runtime
        QuantizerBase = None  # type: ignore[assignment]
    if QuantizerBase is not None and isinstance(module, QuantizerBase):
        return False

    output_quantizers = getattr(module, "output_quantizers", None)
    if output_quantizers is None or len(output_quantizers) == 0:
        return False
    return output_quantizers[0] is not None


def _to_float(out: Any) -> Optional[torch.Tensor]:
    """Coerce a module output to a float ``Tensor``; ``None`` if unsupported."""

    if isinstance(out, Int16QuantizedTensor):
        with int16_eval_allow_debug_float():
            return out.to_float().detach().to(torch.float32).clone()
    if hasattr(out, "dequantize"):
        deq = out.dequantize()
        if isinstance(deq, torch.Tensor):
            return deq.detach().to(torch.float32).clone()
    if isinstance(out, torch.Tensor):
        # NOTE: ``.clone()`` defeats in-place activations (e.g. MobileNet V2
        # ``ReLU6(inplace=True)``) that would otherwise mutate the captured
        # handle before we compare it against the candidate output.
        return out.detach().to(torch.float32).clone()
    return None


def _normalize_inputs(
    input_data: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
) -> Tuple[torch.Tensor, ...]:
    if isinstance(input_data, torch.Tensor):
        return (input_data,)
    return tuple(input_data)


@torch.no_grad()
def per_layer_chained_cosine(
    model: nn.Module,
    inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    *,
    ref_mode: ExecutionMode = ExecutionMode.FP32_QDQ,
    cand_mode: ExecutionMode = ExecutionMode.INT16_FIXED_EVAL,
    module_filter: Optional[Callable[[str, nn.Module], bool]] = None,
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Per-module *chained* cosine between two execution modes.

    For every module passing ``module_filter`` (default: any leaf qmodule
    with an output-quantizer slot):

    1. Run ``model(*inputs)`` under ``ref_mode`` and hook each target's
       output (defensively cloned).
    2. Re-run ``model(*inputs)`` under ``cand_mode``, capture the same
       module's output the same way.
    3. Compute the Tier-1 metric set on the two captured tensors.

    Because both runs traverse the **full network**, the per-layer cosine
    reflects accumulated upstream noise — pair with
    :func:`per_layer_isolated_cosine` to factor that out.

    Parameters
    ----------
    model
        Any ``nn.Module`` (typically ``sim.model``).
    inputs
        A single ``Tensor`` or a tuple of positional input tensors.
    ref_mode
        Reference execution mode (default :attr:`ExecutionMode.FP32_QDQ`).
    cand_mode
        Candidate execution mode (default :attr:`ExecutionMode.INT16_FIXED_EVAL`).
    module_filter
        Optional predicate ``(name, module) -> bool``; defaults to "leaf
        modules with an output quantizer".
    top_k
        If set, return only the ``top_k`` lowest-cosine rows.

    Returns
    -------
    list of dict
        Each row carries the Tier-1 metric set:
        ``module``, ``cosine``, ``shape``, ``ref_rms``, ``max_abs_err``,
        ``norm_max_err``, ``rmse``, ``sqnr_db``, ``p99_abs_err``.
        Sorted ascending by ``cosine``.
    """

    in_args = _normalize_inputs(inputs)
    filt = module_filter or _default_filter
    targets: List[Tuple[str, nn.Module]] = [
        (n, m) for n, m in model.named_modules() if filt(n, m)
    ]
    if not targets:
        _LOGGER.warning(
            "per_layer_chained_cosine: no modules matched the filter (model=%s)",
            type(model).__name__,
        )
        return []

    ref_outputs: Dict[str, torch.Tensor] = {}
    cand_outputs: Dict[str, torch.Tensor] = {}

    def make_hook(name: str, store: Dict[str, torch.Tensor]):
        def _hook(_mod, _ins, out):  # pylint: disable=unused-argument
            tensor = _to_float(out)
            if tensor is not None and tensor.is_floating_point():
                store[name] = tensor
        return _hook

    def _attach(store: Dict[str, torch.Tensor]) -> List[Any]:
        return [m.register_forward_hook(make_hook(n, store)) for n, m in targets]

    handles = _attach(ref_outputs)
    try:
        with quant_execution_mode(ref_mode):
            model(*in_args)
    finally:
        for h in handles:
            h.remove()

    handles = _attach(cand_outputs)
    try:
        with quant_execution_mode(cand_mode):
            model(*in_args)
    finally:
        for h in handles:
            h.remove()

    rows: List[Dict[str, Any]] = []
    for name, _ in targets:
        a = ref_outputs.get(name)
        b = cand_outputs.get(name)
        if a is None or b is None or a.shape != b.shape:
            continue
        a_f = a.reshape(-1).to(torch.float32)
        b_f = b.reshape(-1).to(torch.float32)
        denom = (a_f.norm() * b_f.norm()).clamp_min(1e-30)
        cos = float(torch.dot(a_f, b_f).item() / float(denom))
        rms = float(a_f.pow(2).mean().sqrt().clamp_min(1e-30).item())
        diff = a_f - b_f
        max_abs_err = float(diff.abs().max().item())
        rmse = float(diff.pow(2).mean().sqrt().item())
        signal_power = float(a_f.pow(2).mean().item())
        noise_power = float(diff.pow(2).mean().item())
        if noise_power > 0 and signal_power > 0:
            sqnr_db = float(10.0 * math.log10(signal_power / noise_power))
        else:
            sqnr_db = float("inf")
        p99_abs_err = p99_abs_error(diff)
        rows.append(
            {
                "module": name,
                "cosine": cos,
                "shape": tuple(a.shape),
                "ref_rms": rms,
                "max_abs_err": max_abs_err,
                "norm_max_err": max_abs_err / rms if rms > 0 else 0.0,
                "rmse": rmse,
                "sqnr_db": sqnr_db,
                "p99_abs_err": p99_abs_err,
            }
        )

    rows.sort(key=lambda r: r["cosine"])
    if top_k is not None:
        rows = rows[: int(top_k)]
    return rows

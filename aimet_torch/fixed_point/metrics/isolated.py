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
"""Teacher-forced per-layer cosine.

Unlike the "chained" comparison in :mod:`compare`, the *isolated* comparison
feeds **the same float input** (captured from a reference forward pass) into
every leaf module under the candidate execution mode and measures the
resulting cosine against the reference output. This isolates the module's
*own* quantization noise, removing upstream accumulation, so we can answer:

* If ``isolated_cosine`` is high but the chained ``per_layer_cosine_across_modes``
  value is low, the layer is innocent — drop comes from accumulation.
* If ``isolated_cosine`` is low, the layer itself is a hot-spot.

The function is **model-agnostic**: it accepts any ``(model, inputs)`` pair,
not just ImageNet / MobileNet, so other networks (ResNet, BERT, ...) can
reuse it directly.

Example
-------

>>> from aimet_torch.fixed_point.metrics import per_layer_isolated_cosine
>>> rows = per_layer_isolated_cosine(sim.model, dummy_input)
>>> for row in rows[:5]:
...     print(row["module"], row["isolated_cosine"])
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
from aimet_torch.fixed_point.requantize import SIM_TENSOR_DTYPE, saturate_sim_tensor
from aimet_torch.fixed_point.tensor import FixedPointSimTensor, Int16QuantizedTensor

_LOGGER = logging.getLogger(__name__)


def _default_filter(_name: str, module: nn.Module) -> bool:
    """Match user-facing quantized leaves: modules with an output quantizer.

    Such modules are exactly the per-layer comparison points: every
    QuantizedConv2d/QuantizedLinear/QuantizedReLU/... — but NOT the
    standalone :class:`Quantize` quantizers themselves (those are
    sub-components that would otherwise also match ``output_quantizers``
    via attribute lookup on the base class).
    """

    try:  # local import: keep this file model-agnostic at module load
        from aimet_torch.v2.quantization.base.quantizer import QuantizerBase
    except ImportError:  # pragma: no cover - v2 always available at runtime
        QuantizerBase = None  # type: ignore[assignment]
    if QuantizerBase is not None and isinstance(module, QuantizerBase):
        return False

    output_quantizers = getattr(module, "output_quantizers", None)
    if output_quantizers is None or len(output_quantizers) == 0:
        return False
    oq = output_quantizers[0]
    return oq is not None and bool(getattr(oq, "is_initialized", lambda: True)())


def _to_float(out: Any) -> Optional[torch.Tensor]:
    """Best-effort coerce a module output to a float ``Tensor`` for cosine."""

    if isinstance(out, Int16QuantizedTensor):
        with int16_eval_allow_debug_float():
            return out.to_float().detach().to(torch.float32).clone()
    if hasattr(out, "dequantize"):
        deq = out.dequantize()
        if isinstance(deq, torch.Tensor):
            return deq.detach().to(torch.float32).clone()
    if isinstance(out, torch.Tensor):
        # ``.clone()`` defends against ``inplace=True`` activations (e.g.
        # MobileNet V2 ``ReLU6``) that would mutate the cached handle.
        return out.detach().to(torch.float32).clone()
    return None


def _normalize_inputs(
    input_data: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
) -> Tuple[torch.Tensor, ...]:
    if isinstance(input_data, torch.Tensor):
        return (input_data,)
    return tuple(input_data)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.reshape(-1).float()
    bf = b.reshape(-1).float()
    denom = (af.norm() * bf.norm()).clamp_min(1e-30)
    return float(torch.dot(af, bf).item() / float(denom))


def _quantize_with_carrier(
    x_float: torch.Tensor, carrier: Dict[str, Any]
) -> Int16QuantizedTensor:
    """Quantize a clean float tensor using a captured predecessor's grid.

    Re-implements :meth:`Int16QuantizedTensor.from_float` but honours the
    carrier's actual ``qmin``/``qmax`` (which may describe a U8 / signed-8
    super-group boundary, not the default INT16 range).
    """

    scale = carrier["scale"].to(device=x_float.device, dtype=torch.float32)
    zp = carrier["zero_point"].to(device=x_float.device, dtype=torch.int32)
    q = torch.round(x_float / scale + zp.to(torch.float32))
    int_repr = saturate_sim_tensor(q, carrier["qmin"], carrier["qmax"])
    return FixedPointSimTensor(
        int_repr=int_repr,
        scale=scale,
        zero_point=zp,
        qmin=carrier["qmin"],
        qmax=carrier["qmax"],
        axis=carrier.get("axis"),
    )


@torch.no_grad()
def per_layer_isolated_cosine(
    model: nn.Module,
    inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    *,
    cand_mode: ExecutionMode = ExecutionMode.FIXED_SCALE_QDQ,
    ref_mode: ExecutionMode = ExecutionMode.FP32_QDQ,
    module_filter: Optional[Callable[[str, nn.Module], bool]] = None,
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Teacher-forced per-layer cosine between two execution modes.

    For every module passing ``module_filter`` (default: any module whose
    first ``output_quantizers`` slot is initialized):

    1. Run ``model(*inputs)`` under ``ref_mode`` and **cache** that module's
       ``(forward_inputs, output)`` tensors (cloned defensively, so in-place
       activations downstream cannot corrupt the cache).
    2. Re-invoke the module *directly* with the cached inputs but under
       ``cand_mode``; convert the output back to ``float32`` for cosine.
    3. Record ``isolated_cosine`` against the cached reference output.

    Because each layer sees the **same clean reference input**, the resulting
    cosine reflects only that layer's quantization noise — there is no
    accumulated upstream error.

    Notes
    -----
    - ``cand_mode=FIXED_SCALE_QDQ`` (default) accepts ``float`` inputs for
      any module, so the function works on all leaf qmodules in one pass.
    - ``cand_mode=INT16_FIXED_EVAL`` is also supported on every module: we
      run an extra *carrier* pass under INT16 to snapshot the predecessor's
      ``(scale, zero_point, qmin, qmax)``, then re-quantize the clean
      ``ref_mode`` cached input onto that grid before calling the module.
      The teacher-forced property holds (the **values** still come from
      the FP32 reference, not from the cumulative INT16 path).
    - This is a diagnostic tool. It does *not* validate the network's
      end-to-end accuracy — use :mod:`compare` or top-1 metrics for that.

    Parameters
    ----------
    model
        The quantized ``sim.model`` (or any ``nn.Module`` with
        ``output_quantizers`` attached to its leaf modules).
    inputs
        Either a single ``Tensor`` or a tuple of positional input tensors,
        matching ``model.forward``'s signature.
    cand_mode
        The candidate execution mode to evaluate per-layer. Default
        :attr:`ExecutionMode.FIXED_SCALE_QDQ` for broad compatibility.
    ref_mode
        Reference execution mode (default :attr:`ExecutionMode.FP32_QDQ`).
    module_filter
        Predicate ``(name, module) -> bool`` selecting layers to test.
        Defaults to "leaf modules with an initialized output quantizer".
    top_k
        If set, only the ``top_k`` worst (lowest-cosine) rows are returned.

    Returns
    -------
    list of dict
        One row per probed module, each with keys:
        ``module``, ``isolated_cosine``, ``shape``, ``ref_rms``,
        ``max_abs_err``, ``norm_max_err``. Sorted ascending by cosine.
    """

    in_args = _normalize_inputs(inputs)
    filt = module_filter or _default_filter
    targets: List[Tuple[str, nn.Module]] = [
        (n, m) for n, m in model.named_modules() if filt(n, m)
    ]
    if not targets:
        _LOGGER.warning(
            "per_layer_isolated_cosine: no modules matched the filter (model=%s)",
            type(model).__name__,
        )
        return []

    cached_inputs: Dict[str, Tuple[Any, ...]] = {}
    cached_outputs: Dict[str, torch.Tensor] = {}

    def make_hook(name: str):
        def _hook(_mod, ins, out):  # pylint: disable=unused-argument
            captured_ins: List[Any] = []
            for arg in ins:
                tensor = _to_float(arg)
                if tensor is not None:
                    # NOTE: clone to defeat downstream inplace activations
                    # (e.g. MobileNet V2 ``ReLU6(inplace=True)``).
                    captured_ins.append(tensor)
                else:
                    captured_ins.append(arg)
            cached_inputs[name] = tuple(captured_ins)
            tensor = _to_float(out)
            if tensor is not None:
                cached_outputs[name] = tensor
        return _hook

    handles = [m.register_forward_hook(make_hook(n)) for n, m in targets]
    try:
        with quant_execution_mode(ref_mode):
            model(*in_args)
    finally:
        for h in handles:
            h.remove()

    int16_carriers: Dict[str, Dict[str, Any]] = {}
    if cand_mode is ExecutionMode.INT16_FIXED_EVAL:
        # Extra pass under INT16 to capture each module's predecessor
        # output-quantizer grid (scale/zp/qmin/qmax/axis). We do NOT keep
        # the int data; only metadata, so teacher-forced semantics hold
        # when we re-quantize the FP32 cached input below.

        def make_carrier_hook(name: str):
            def _hook(_mod, ins, _out):  # pylint: disable=unused-argument
                if not ins:
                    return
                first = ins[0]
                if isinstance(first, Int16QuantizedTensor):
                    int16_carriers[name] = {
                        "scale": first.scale.detach().clone(),
                        "zero_point": first.zero_point.detach().clone(),
                        "qmin": first.qmin,
                        "qmax": first.qmax,
                        "axis": first.axis,
                    }
            return _hook

        carrier_handles = [
            m.register_forward_hook(make_carrier_hook(n)) for n, m in targets
        ]
        try:
            with quant_execution_mode(cand_mode):
                model(*in_args)
        except (RuntimeError, AttributeError, TypeError) as exc:
            _LOGGER.warning(
                "per_layer_isolated_cosine: INT16 carrier pass failed "
                "(%s: %s); falling back to per-module direct invocation. "
                "Modules with uninitialized input quantizers will be skipped.",
                type(exc).__name__,
                exc,
            )
        finally:
            for h in carrier_handles:
                h.remove()

    rows: List[Dict[str, Any]] = []
    skipped_count = 0
    for name, module in targets:
        if name not in cached_inputs or name not in cached_outputs:
            continue
        x_args = cached_inputs[name]
        y_ref = cached_outputs[name]
        if (
            cand_mode is ExecutionMode.INT16_FIXED_EVAL
            and name in int16_carriers
            and x_args
            and isinstance(x_args[0], torch.Tensor)
        ):
            try:
                x0 = _quantize_with_carrier(x_args[0], int16_carriers[name])
                x_args = (x0,) + x_args[1:]
            except (RuntimeError, TypeError, ValueError) as exc:
                _LOGGER.debug(
                    "isolated INT16 re-quant skipped for %s: %s: %s",
                    name,
                    type(exc).__name__,
                    exc,
                )
        try:
            with quant_execution_mode(cand_mode):
                y_cand_raw = module(*x_args)
        except (RuntimeError, AttributeError, TypeError) as exc:
            skipped_count += 1
            _LOGGER.debug(
                "isolated cosine skipped for %s (cand_mode=%s): %s: %s",
                name,
                cand_mode,
                type(exc).__name__,
                exc,
            )
            continue
        y_cand = _to_float(y_cand_raw)
        if y_cand is None or y_cand.shape != y_ref.shape:
            skipped_count += 1
            continue
        a = y_cand.reshape(-1).float()
        b = y_ref.reshape(-1).float()
        denom = (a.norm() * b.norm()).clamp_min(1e-30)
        cos = float(torch.dot(a, b).item() / float(denom))
        rms = float(b.pow(2).mean().sqrt().clamp_min(1e-30).item())
        diff = a - b
        max_abs_err = float(diff.abs().max().item())
        rmse = float(diff.pow(2).mean().sqrt().item())
        # SQNR (dB): 10*log10(signal_power / noise_power). +inf when noise=0,
        # which surfaces "perfect" layers without messing up table sort.
        signal_power = float(b.pow(2).mean().item())
        noise_power = float(diff.pow(2).mean().item())
        if noise_power > 0 and signal_power > 0:
            sqnr_db = float(10.0 * math.log10(signal_power / noise_power))
        else:
            sqnr_db = float("inf")
        # P99 absolute error — the long-tail metric cosine cannot see.
        p99_abs_err = p99_abs_error(diff)
        rows.append(
            {
                "module": name,
                "isolated_cosine": cos,
                "shape": tuple(y_ref.shape),
                "ref_rms": rms,
                "max_abs_err": max_abs_err,
                "norm_max_err": max_abs_err / rms if rms > 0 else 0.0,
                "rmse": rmse,
                "sqnr_db": sqnr_db,
                "p99_abs_err": p99_abs_err,
            }
        )

    if skipped_count:
        _LOGGER.info(
            "per_layer_isolated_cosine: skipped %d / %d modules (cand_mode=%s); "
            "run with logging level DEBUG to see the per-module reasons.",
            skipped_count,
            len(targets),
            cand_mode,
        )

    rows.sort(key=lambda r: r["isolated_cosine"])
    if top_k is not None:
        rows = rows[: int(top_k)]
    return rows

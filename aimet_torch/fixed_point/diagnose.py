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
"""INT16_FIXED_EVAL readiness diagnostics for v2 QuantizationSimModel."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch.nn as nn

from aimet_torch.fixed_point.registry import KernelNotFoundError, get_fixed_kernel
from aimet_torch.fixed_point.sim_utils import (
    INT16_DISPATCHABLE_MODULES,
    iter_missing_output_quantizers,
)

__all__ = [
    "diagnose_int16_readiness",
    "is_int16_ready",
]

Report = Dict[str, List]


def _iter_quantizers(module: nn.Module):
    """Yield non-None quantizers attached to a v2 quantized module."""

    # pylint: disable=import-outside-toplevel
    from aimet_torch.v2.quantization.base import QuantizerBase

    param_q = getattr(module, "param_quantizers", None)
    if isinstance(param_q, dict):
        for q in param_q.values():
            if q is not None:
                yield q

    for attr in ("input_quantizers", "output_quantizers"):
        qlist = getattr(module, attr, None)
        if not isinstance(qlist, nn.ModuleList):
            continue
        for q in qlist:
            if q is not None:
                yield q


def _resolve_base_cls(module: nn.Module):
    # pylint: disable=import-outside-toplevel
    from aimet_torch.v2.nn.true_quant import QuantizationMixin

    base_cls = QuantizationMixin.qcls_to_cls.get(type(module))
    if base_cls is not None:
        return base_cls

    from aimet_torch.v2.nn.fake_quant._legacy_impl import FakeQuantizationMixin

    return FakeQuantizationMixin.qcls_to_cls.get(type(module))


def diagnose_int16_readiness(sim: Any) -> Report:
    """Return INT16 readiness issue lists for a calibrated v2 sim.

    Keys (each maps to ``list[(module_name, class_name)]`` except
    ``uncalibrated_quantgru`` which is ``list[str]``):

    - ``missing_output_quantizer``
    - ``uninitialized_encoding``
    - ``uncalibrated_quantgru``
    - ``missing_fixed_kernel``
    """

    # pylint: disable=import-outside-toplevel
    from aimet_torch.fixed_point.shape_meta import (
        is_intentionally_unquantized_module,
        is_shape_meta_only_quantized_module,
    )
    from aimet_torch.v2.quantization.base import QuantizerBase

    dispatchable_fn = INT16_DISPATCHABLE_MODULES
    dispatchable = (
        dispatchable_fn() if callable(dispatchable_fn) else dispatchable_fn
    )

    try:
        from aimet_torch.v2.nn.modules.custom import QuantizedQuantGRU
    except ImportError:
        QuantizedQuantGRU = None

    missing_output_quantizer = list(iter_missing_output_quantizers(sim))
    uninitialized_encoding: List[Tuple[str, str]] = []
    uncalibrated_quantgru: List[str] = []
    missing_fixed_kernel: List[Tuple[str, str]] = []

    for name, module in sim.model.named_modules():
        label = name or type(module).__name__

        if QuantizedQuantGRU is not None and isinstance(module, QuantizedQuantGRU):
            if not module.is_calibrated():
                uncalibrated_quantgru.append(label)
            continue

        if is_intentionally_unquantized_module(module):
            continue

        if is_shape_meta_only_quantized_module(module):
            continue

        for quantizer in _iter_quantizers(module):
            if isinstance(quantizer, QuantizerBase) and not quantizer.is_initialized():
                uninitialized_encoding.append((label, type(module).__name__))
                break

        if not isinstance(module, dispatchable):
            continue
        if is_shape_meta_only_quantized_module(module):
            continue

        base_cls = _resolve_base_cls(module)
        if base_cls is None:
            missing_fixed_kernel.append((label, type(module).__name__))
            continue

        try:
            get_fixed_kernel(base_cls)
        except KernelNotFoundError:
            missing_fixed_kernel.append((label, type(module).__name__))

    return {
        "missing_output_quantizer": missing_output_quantizer,
        "uninitialized_encoding": uninitialized_encoding,
        "uncalibrated_quantgru": uncalibrated_quantgru,
        "missing_fixed_kernel": missing_fixed_kernel,
    }


def is_int16_ready(sim: Any) -> bool:
    """Return True when :func:`diagnose_int16_readiness` reports no blockers."""

    report = diagnose_int16_readiness(sim)
    return all(len(report[key]) == 0 for key in report)

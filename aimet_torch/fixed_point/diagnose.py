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


def _collect_diagnose_bitwidths(module: nn.Module, attr: str) -> List[int]:
    """Pull initialized quantizer bitwidths from a ``ModuleList``/``ModuleDict``.

    Mirrors :func:`adapter._collect_quantizer_bitwidths`; kept local to
    diagnose to avoid a circular import on the v2 adapter package.
    """

    # pylint: disable=import-outside-toplevel
    from aimet_torch.v2.quantization.base import QuantizerBase

    out: List[int] = []
    quants = getattr(module, attr, None)
    if not isinstance(quants, (nn.ModuleList, nn.ModuleDict)):
        return out
    iterator = (
        quants.values() if isinstance(quants, nn.ModuleDict) else iter(quants)
    )
    for quant in iterator:
        if not isinstance(quant, QuantizerBase) or not quant.is_initialized():
            continue
        bw = getattr(quant, "bitwidth", None)
        if bw is None:
            continue
        out.append(int(bw))
    return out


def diagnose_int16_readiness(sim: Any) -> Report:
    """Return INT16 readiness issue lists for a calibrated v2 sim.

    Keys (each maps to ``list[(module_name, class_name)]`` except
    ``uncalibrated_quantgru`` and ``blackbox_native_ops`` which are ``list[str]``,
    and ``unsupported_activation_bitwidth`` which is
    ``list[(module_name, class_name, bitwidth)]``):

    - ``missing_output_quantizer`` — dispatchable module without an output quantizer.
    - ``uninitialized_encoding`` — quantizer attached but not calibrated.
    - ``uncalibrated_quantgru`` — ``QuantGRU`` blackbox without calibration.
    - ``missing_fixed_kernel`` — dispatchable module whose base class has no
      registered Python kernel (informational; should not happen on a
      manifest-aligned codebase).
    - ``unsupported_activation_bitwidth`` — input/output quantizer with a
      bitwidth outside ``capabilities.SUPPORTED_ACTIVATION_BITWIDTHS``. The
      adapter dispatch entry will refuse such modules under
      ``INT16_FIXED_EVAL`` (see audit-int16-activation-quantizer-contract);
      surfacing them here lets callers fix the configuration before forward.
    - ``blackbox_native_ops`` — modules driven by a native blackbox contract
      (e.g. ``QuantizedQuantGRU``); intentionally not registered as a Python
      fixed kernel. Listed for visibility, not as a readiness blocker.
    """

    # pylint: disable=import-outside-toplevel
    from aimet_torch.fixed_point.capabilities import (
        CapabilityStatus,
        assert_requantizing_combo_supported,
        get_capability,
        requires_activation_bitwidth_gate,
    )
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
    unsupported_activation_bitwidth: List[Tuple[str, str, int]] = []
    blackbox_native_ops: List[str] = []

    for name, module in sim.model.named_modules():
        label = name or type(module).__name__

        if QuantizedQuantGRU is not None and isinstance(module, QuantizedQuantGRU):
            blackbox_native_ops.append(label)
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

        # Activation-bitwidth contract: REQUANTIZING kernels only (LOOKUP
        # kernels validate 16-bit through their LUT path; SAME_GRID kernels
        # never re-scale). Mirrors
        # ``adapter._enforce_supported_activation_bitwidths`` so a diagnose
        # pass and a forward pass cannot disagree on what counts as an
        # unsupported configuration.
        #
        # PR-2 (W5 SYS-FU-1.B): the contract is now the per-MAC combo gate,
        # which gates ``input_bw + weight_bw`` against
        # ``REQUANTIZING_COMBO_BITWIDTH_BUDGET`` for ``is_reduction=True``
        # kernels. We surface a violation by running the same validator the
        # adapter calls under ``try/except`` and recording the offending
        # bitwidth (the maximum operand bitwidth in the unsupported combo
        # — preserves the legacy ``(label, type, bw)`` tuple shape so
        # readiness reports stay backward-compatible).
        base_for_gate = _resolve_base_cls(module)
        if base_for_gate is not None:
            cap_for_gate = get_capability(base_for_gate)
            if cap_for_gate is not None and requires_activation_bitwidth_gate(
                cap_for_gate.kernel_kind
            ):
                input_bws = _collect_diagnose_bitwidths(module, "input_quantizers")
                output_bws = _collect_diagnose_bitwidths(module, "output_quantizers")
                weight_bws: List[int] = []
                pq = getattr(module, "param_quantizers", None)
                if isinstance(pq, nn.ModuleDict) and "weight" in pq:
                    wq = pq["weight"]
                    if (
                        isinstance(wq, QuantizerBase)
                        and wq.is_initialized()
                        and getattr(wq, "bitwidth", None) is not None
                    ):
                        weight_bws.append(int(wq.bitwidth))
                # Same input/weight vs output split as the adapter; see
                # the long comment there for why output_bw is NOT a MAC
                # operand.
                gate_failed = False
                try:
                    assert_requantizing_combo_supported(
                        input_bws,
                        weight_bws,
                        where="input/weight quantizers",
                        qualname=type(module).__name__,
                        is_reduction=bool(cap_for_gate.is_reduction),
                    )
                    if output_bws:
                        assert_requantizing_combo_supported(
                            output_bws,
                            (),
                            where="output quantizers",
                            qualname=type(module).__name__,
                            is_reduction=False,
                        )
                except ValueError:
                    gate_failed = True
                if gate_failed:
                    bws_all = input_bws + output_bws + weight_bws
                    if bws_all:
                        unsupported_activation_bitwidth.append(
                            (label, type(module).__name__, max(bws_all))
                        )

        if not isinstance(module, dispatchable):
            continue
        if is_shape_meta_only_quantized_module(module):
            continue

        base_cls = _resolve_base_cls(module)
        if base_cls is None:
            missing_fixed_kernel.append((label, type(module).__name__))
            continue

        capability = get_capability(base_cls)
        if capability is not None and capability.status is CapabilityStatus.BLACKBOX:
            # Manifest declares this op as a native blackbox: do not flag a
            # missing Python kernel even if the registry has no entry.
            blackbox_native_ops.append(label)
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
        "unsupported_activation_bitwidth": unsupported_activation_bitwidth,
        "blackbox_native_ops": blackbox_native_ops,
    }


_READINESS_BLOCKER_KEYS: Tuple[str, ...] = (
    "missing_output_quantizer",
    "uninitialized_encoding",
    "uncalibrated_quantgru",
    "missing_fixed_kernel",
    "unsupported_activation_bitwidth",
)
"""Keys in :func:`diagnose_int16_readiness` that block INT16 readiness.

``blackbox_native_ops`` is intentionally excluded — it is informational
(e.g. lists ``QuantizedQuantGRU`` instances) and a network containing a
calibrated ``QuantGRU`` should still be considered ready.
"""


def is_int16_ready(sim: Any) -> bool:
    """Return True when :func:`diagnose_int16_readiness` reports no blockers."""

    report = diagnose_int16_readiness(sim)
    return all(len(report.get(key, [])) == 0 for key in _READINESS_BLOCKER_KEYS)

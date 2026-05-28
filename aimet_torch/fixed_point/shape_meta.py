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
"""Shape / layout metadata ops for INT16 fixed-point dispatch.

``model_preparer`` may materialize ``b * f``, ``t + pad`` etc. as
``QuantizedMultiply`` / ``QuantizedAdd`` in the FX graph. Those operands come
from ``tensor.shape`` indexing and are **not** quantized activations — they must
not require boundary output quantizers in ``INT16_FIXED_*`` modes.

This module detects such meta-only binary ops and evaluates them with plain
Python / integer tensor semantics (mirroring ``aimet_torch._base.nn.modules.custom``).
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
from torch import nn

from aimet_torch._base.nn.modules import custom
from aimet_torch.fixed_point.tensor import FixedPointSimTensor, Int16QuantizedTensor
from aimet_torch.v2.quantization.tensor import QuantizedTensorBase

__all__ = [
    "is_fp32_preserve_quantized_module",
    "is_shape_meta_only_quantized_module",
    "is_intentionally_unquantized_module",
    "is_shape_meta_scalar",
    "try_dispatch_shape_meta_op",
]

_SHAPE_META_BINARY = (
    custom.Multiply,
    custom.Add,
    custom.Subtract,
    custom.Divide,
    custom.FloorDivide,
)


def is_shape_meta_scalar(value: Any) -> bool:
    """Return True if *value* is layout metadata, not a quantized activation."""

    if isinstance(value, (Int16QuantizedTensor, FixedPointSimTensor)):
        return False
    if isinstance(value, QuantizedTensorBase):
        return False
    if isinstance(value, bool):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return True
    if isinstance(value, torch.Tensor):
        if value.is_floating_point():
            return False
        return True
    if isinstance(value, (list, tuple)):
        return all(is_shape_meta_scalar(item) for item in value)
    return False


def _eval_binary(base_cls: type, a: Any, b: Any) -> Any:
    if base_cls is custom.Multiply:
        if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
            return torch.mul(a, b)
        return a * b
    if base_cls is custom.Add:
        if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
            return torch.add(a, b)
        return a + b
    if base_cls is custom.Subtract:
        if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
            return torch.sub(a, b)
        return a - b
    if base_cls is custom.Divide:
        if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
            return torch.div(a, b)
        return a / b
    if base_cls is custom.FloorDivide:
        if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
            return torch.floor_divide(a, b)
        return a // b
    raise TypeError(f"Unsupported shape-meta binary op: {base_cls!r}")


def try_dispatch_shape_meta_op(
    base_cls: Optional[type],
    *args: Any,
    **kwargs: Any,
) -> Tuple[bool, Any]:
    """Try to evaluate a shape-meta binary op outside INT16 quant dispatch.

    Returns:
        ``(True, result)`` when *base_cls* is a supported meta binary op and all
        operand args are shape-meta scalars; ``(False, None)`` otherwise.
    """

    del kwargs
    if base_cls not in _SHAPE_META_BINARY:
        return False, None
    if len(args) < 2:
        return False, None
    a, b = args[0], args[1]
    if not (is_shape_meta_scalar(a) and is_shape_meta_scalar(b)):
        return False, None
    return True, _eval_binary(base_cls, a, b)


def is_shape_meta_only_quantized_module(module: Any) -> bool:
    """Return True when a v2 quantized module only computes layout metadata.

    ``model_preparer`` may emit ``QuantizedMultiply`` for ``b * f`` etc. Those
    modules intentionally have **no** calibrated input/output quantizers and are
    dispatched via :func:`try_dispatch_shape_meta_op` instead of INT16 kernels.
    """

    # pylint: disable=import-outside-toplevel
    from aimet_torch.v2.nn.true_quant import QuantizationMixin
    from aimet_torch.v2.quantization.base import QuantizerBase

    base_cls = QuantizationMixin.qcls_to_cls.get(type(module))
    if base_cls not in _SHAPE_META_BINARY:
        return False

    for attr in ("input_quantizers", "output_quantizers"):
        qlist = getattr(module, attr, None)
        if not isinstance(qlist, nn.ModuleList):
            continue
        for q in qlist:
            if isinstance(q, QuantizerBase) and q.is_initialized():
                return False
    return True


def is_fp32_preserve_quantized_module(module: Any) -> bool:
    """Return True for FP32 black-box wrappers (no INT16 calib expected)."""

    del module
    return False


def is_intentionally_unquantized_module(module: Any) -> bool:
    """Return True for FP32 black-box wrappers with all-quantizer slots set to None."""

    if is_fp32_preserve_quantized_module(module):
        return True

    # pylint: disable=import-outside-toplevel
    from aimet_torch.v2.quantization.base import QuantizerBase

    saw_slot = False
    for attr in ("input_quantizers", "output_quantizers"):
        qlist = getattr(module, attr, None)
        if not isinstance(qlist, nn.ModuleList):
            continue
        for q in qlist:
            saw_slot = True
            if q is not None:
                return False

    param_q = getattr(module, "param_quantizers", None)
    if isinstance(param_q, dict):
        for q in param_q.values():
            if q is not None:
                return False

    return saw_slot

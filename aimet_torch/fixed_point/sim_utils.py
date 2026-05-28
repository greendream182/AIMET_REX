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
"""Helpers to make a v2 ``QuantizationSimModel`` ready for ``INT16_FIXED_EVAL``.

The default v2 sim config applies super-group fusion (e.g. ``Conv → BN → ReLU``)
which leaves intermediate ``Conv`` modules with **no output quantizer**. The
INT16 fixed-point dispatch path requires every emitted op to have an output
quantizer (the hardware needs an explicit output scale at each layer boundary),
so the user must materialize those quantizers before ``compute_encodings``.

``ensure_output_quantizers_for_int16_eval`` is a one-liner that walks the sim
and inserts a default symmetric output quantizer wherever it is missing. It is
idempotent and only touches modules whose ``output_quantizers[0]`` is either
``None`` or uninitialized.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Tuple

import torch.nn as nn

__all__ = [
    "ensure_output_quantizers_for_int16_eval",
    "INT16_DISPATCHABLE_MODULES",
]


def _dispatchable_module_types():
    """Module classes that the INT16 fixed-point adapter knows how to dispatch.

    Imports are deferred to avoid pulling v2 stack at module import time.
    """

    # pylint: disable=import-outside-toplevel
    from aimet_torch._base.nn.modules import custom

    types = [
        nn.Linear,
        nn.Conv1d,
        nn.Conv2d,
        nn.MaxPool2d,
        nn.AvgPool2d,
        nn.ReLU,
        nn.ReLU6,
        nn.Hardtanh,
        nn.Sigmoid,
        nn.Tanh,
        nn.GELU,
        nn.SiLU,
        nn.Mish,
        nn.Softplus,
        nn.Hardsigmoid,
        nn.Hardswish,
        nn.LeakyReLU,
        nn.PReLU,
        nn.Softmax,
        nn.Flatten,
        custom.Sin,
        custom.Cos,
        custom.Sqrt,
        custom.RSqrt,
        custom.Reciprocal,
        custom.Square,
        custom.Abs,
        custom.ElementwiseUnarySign,
        custom.Exponential,
        custom.Log,
        nn.Identity,
        nn.Dropout,
        custom.Add,
        custom.Subtract,
        custom.Multiply,
        custom.Divide,
        custom.MatMul,
        custom.Concat,
        custom.Reshape,
        custom.Permute,
        custom.Mean,
        custom.MaxPool2d,
        custom.AvgPool2d,
        custom.AdaptiveAvgPool2d,
        custom.Pad,
        custom.Clamp,
        custom.Clip,
    ]

    try:
        from aimet_torch.v2.nn.modules.custom import QuantizedQuantGRU
    except ImportError:
        QuantizedQuantGRU = None

    if QuantizedQuantGRU is not None:
        types.append(QuantizedQuantGRU)

    return tuple(types)


# Public alias resolved lazily so importing this module is cheap.
INT16_DISPATCHABLE_MODULES = _dispatchable_module_types


def ensure_output_quantizers_for_int16_eval(
    sim: Any,
    bitwidth: int = 8,
    symmetric: bool = True,
) -> List[Tuple[str, str]]:
    """Materialize input/output quantizers on dispatchable modules that lack them.

    Args:
        sim: A v2 ``QuantizationSimModel`` whose ``sim.model`` has been built
            but **before** ``compute_encodings`` is called.
        bitwidth: Bitwidth used for the freshly created ``Quantize`` instances.
        symmetric: Whether the new quantizers should be symmetric.

    Returns:
        A list of ``(module_name, module_class)`` pairs that were patched.
        Use this to log/report how many layers had missing output quantizers.

    Notes:
        Run this **before** ``sim.compute_encodings(...)`` so the new quantizers
        get statistics collected during calibration. Calling it again is a no-op.
    """

    # pylint: disable=import-outside-toplevel
    from aimet_torch.v2.quantization.affine import Quantize
    from aimet_torch.fixed_point.shape_meta import is_shape_meta_only_quantized_module
    from aimet_torch.v2.quantization.base import QuantizerBase

    dispatchable = _dispatchable_module_types()
    patched: List[Tuple[str, str]] = []

    try:
        from aimet_torch.v2.nn.modules.custom import QuantizedQuantGRU
    except ImportError:
        QuantizedQuantGRU = None

    for name, module in sim.model.named_modules():
        if QuantizedQuantGRU is not None and isinstance(module, QuantizedQuantGRU):
            continue
        if not isinstance(module, dispatchable):
            continue
        if is_shape_meta_only_quantized_module(module):
            continue

        iq_list = getattr(module, "input_quantizers", None)
        if iq_list is not None:
            for index, iq in enumerate(list(iq_list)):
                if isinstance(iq, QuantizerBase) and iq.is_initialized():
                    continue
                iq_list[index] = Quantize((), bitwidth, symmetric=symmetric)
                patched.append((f"{name}:input[{index}]", type(module).__name__))

        oq_list = getattr(module, "output_quantizers", None)
        if oq_list is None or len(oq_list) == 0:
            continue
        for index, oq in enumerate(list(oq_list)):
            if isinstance(oq, QuantizerBase) and oq.is_initialized():
                continue
            oq_list[index] = Quantize((), bitwidth, symmetric=symmetric)
            patched.append((name or type(module).__name__, type(module).__name__))

    return patched


def iter_missing_output_quantizers(sim: Any) -> Iterable[Tuple[str, str]]:
    """Yield ``(name, classname)`` for every dispatchable module with a missing oq.

    Useful for reporting before calling :func:`ensure_output_quantizers_for_int16_eval`.
    """

    # pylint: disable=import-outside-toplevel
    from aimet_torch.fixed_point.shape_meta import is_shape_meta_only_quantized_module
    from aimet_torch.v2.quantization.base import QuantizerBase

    dispatchable = _dispatchable_module_types()

    try:
        from aimet_torch.v2.nn.modules.custom import QuantizedQuantGRU
    except ImportError:
        QuantizedQuantGRU = None

    for name, module in sim.model.named_modules():
        if QuantizedQuantGRU is not None and isinstance(module, QuantizedQuantGRU):
            continue
        oq_list = getattr(module, "output_quantizers", None)
        if oq_list is None or len(oq_list) == 0:
            continue
        if not isinstance(module, dispatchable):
            continue
        if is_shape_meta_only_quantized_module(module):
            continue
        for oq in oq_list:
            if not (isinstance(oq, QuantizerBase) and oq.is_initialized()):
                yield (name or type(module).__name__, type(module).__name__)
                break

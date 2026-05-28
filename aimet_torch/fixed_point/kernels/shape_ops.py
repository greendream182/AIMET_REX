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
"""Reference INT16 fixed-point kernels for shape-only operations."""

from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch import nn

from aimet_torch._base.nn.modules import custom
from aimet_torch.fixed_point.encoding import OutputEncoding
from aimet_torch.fixed_point.kernels.eltwise import align_centered_int32_to_output
from aimet_torch.fixed_point.registry import register_fixed_kernel
from aimet_torch.fixed_point.requantize import (
    SIM_TENSOR_DTYPE,
    int32_add_sat,
    requantize_int,
    saturate_sim_tensor,
)
from aimet_torch.fixed_point.tensor import Int16QuantizedTensor


def _require_same_encoding(
    tensor: Int16QuantizedTensor,
    output_encoding: OutputEncoding,
) -> None:
    output_scale = output_encoding.scale.to(
        device=tensor.int_repr.device, dtype=tensor.scale.dtype
    )
    output_zp = output_encoding.zero_point.to(
        device=tensor.int_repr.device, dtype=torch.int32
    )
    if not torch.equal(tensor.scale.to(device=tensor.int_repr.device), output_scale):
        raise ValueError("Shape-only INT16 kernel requires output scale to match input scale.")
    if not torch.equal(tensor.zero_point.to(device=tensor.int_repr.device), output_zp):
        raise ValueError("Shape-only INT16 kernel requires output zero_point to match input zero_point.")


def _require_all_same_encoding(
    inputs: List[Int16QuantizedTensor],
    output_encoding: OutputEncoding,
) -> None:
    for tensor in inputs:
        _require_same_encoding(tensor, output_encoding)


def _wrap(
    int_repr: torch.Tensor,
    output_encoding: OutputEncoding,
) -> Int16QuantizedTensor:
    if int_repr.dtype != SIM_TENSOR_DTYPE:
        int_repr = int_repr.to(SIM_TENSOR_DTYPE)
    return Int16QuantizedTensor(
        int_repr=int_repr,
        scale=output_encoding.scale.to(device=int_repr.device),
        zero_point=output_encoding.zero_point.to(
            device=int_repr.device, dtype=torch.int32
        ),
        qmin=output_encoding.qmin,
        qmax=output_encoding.qmax,
        axis=output_encoding.axis,
    )


def _requantize_identity_output(
    tensor: Int16QuantizedTensor,
    output_encoding: OutputEncoding,
) -> Int16QuantizedTensor:
    """Requantize an identity-like op from input grid to output grid."""

    if output_encoding.multiplier is None or output_encoding.rshift is None:
        _require_same_encoding(tensor, output_encoding)
        return _wrap(tensor.int_repr, output_encoding)

    centered = tensor.centered_int32()
    int_repr = requantize_int(
        centered,
        output_encoding.multiplier.to(device=centered.device),
        output_encoding.rshift.to(device=centered.device),
        output_encoding.zero_point.to(device=centered.device, dtype=torch.int32),
        output_encoding.qmin,
        output_encoding.qmax,
    )
    return _wrap(int_repr, output_encoding)


class _IdentityLikeInt16Kernel:
    """Shared no-op kernel: copies the int_repr and requires the encoding to match.

    Used for ``nn.Identity`` and for eval-mode-only no-ops such as ``nn.Dropout``
    (Dropout has no parameters and degenerates to identity in ``eval()``).
    """

    def __call__(
        self,
        inputs: List[Int16QuantizedTensor],
        params: Dict[str, Any],
        output_encoding: OutputEncoding,
        extra: Dict[str, Any],
    ) -> Int16QuantizedTensor:
        del params, extra
        if len(inputs) != 1:
            raise ValueError(f"Identity/Dropout expects 1 input; got {len(inputs)}.")
        _require_same_encoding(inputs[0], output_encoding)
        return _wrap(inputs[0].int_repr, output_encoding)


@register_fixed_kernel(nn.Identity)
class IdentityInt16Kernel(_IdentityLikeInt16Kernel):
    """Reference INT16 identity kernel."""

    module_type = nn.Identity


@register_fixed_kernel(nn.Dropout)
class DropoutInt16Kernel:
    """Reference INT16 Dropout kernel.

    In ``eval()`` mode Dropout is a no-op in value space, but it may still sit
    between two different quantization grids. Requantize to the destination
    output encoding instead of only relabeling the integer payload.
    """

    module_type = nn.Dropout

    def __call__(
        self,
        inputs: List[Int16QuantizedTensor],
        params: Dict[str, Any],
        output_encoding: OutputEncoding,
        extra: Dict[str, Any],
    ) -> Int16QuantizedTensor:
        del params, extra
        if len(inputs) != 1:
            raise ValueError(f"Dropout expects 1 input; got {len(inputs)}.")
        return _requantize_identity_output(inputs[0], output_encoding)


@register_fixed_kernel(nn.Flatten)
class FlattenInt16Kernel:
    """Reference INT16 flatten kernel."""

    module_type = nn.Flatten

    def __call__(
        self,
        inputs: List[Int16QuantizedTensor],
        params: Dict[str, Any],
        output_encoding: OutputEncoding,
        extra: Dict[str, Any],
    ) -> Int16QuantizedTensor:
        del params
        if len(inputs) != 1:
            raise ValueError(f"Flatten expects 1 input; got {len(inputs)}.")
        _require_same_encoding(inputs[0], output_encoding)
        start_dim = int(extra.get("start_dim", 1))
        end_dim = int(extra.get("end_dim", -1))
        return _wrap(torch.flatten(inputs[0].int_repr, start_dim, end_dim), output_encoding)


@register_fixed_kernel(custom.Reshape)
class ReshapeInt16Kernel:
    """Reference INT16 reshape kernel."""

    module_type = custom.Reshape

    def __call__(
        self,
        inputs: List[Int16QuantizedTensor],
        params: Dict[str, Any],
        output_encoding: OutputEncoding,
        extra: Dict[str, Any],
    ) -> Int16QuantizedTensor:
        del params
        if len(inputs) != 1:
            raise ValueError(f"Reshape expects 1 input; got {len(inputs)}.")
        _require_same_encoding(inputs[0], output_encoding)
        shape = extra.get("shape")
        if shape is None:
            raise ValueError("Reshape kernel requires extra['shape'].")
        return _wrap(torch.reshape(inputs[0].int_repr, tuple(shape)), output_encoding)


@register_fixed_kernel(custom.Permute)
class PermuteInt16Kernel:
    """Reference INT16 permute kernel."""

    module_type = custom.Permute

    def __call__(
        self,
        inputs: List[Int16QuantizedTensor],
        params: Dict[str, Any],
        output_encoding: OutputEncoding,
        extra: Dict[str, Any],
    ) -> Int16QuantizedTensor:
        del params
        if len(inputs) != 1:
            raise ValueError(f"Permute expects 1 input; got {len(inputs)}.")
        _require_same_encoding(inputs[0], output_encoding)
        dims = extra.get("dims")
        if dims is None:
            raise ValueError("Permute kernel requires extra['dims'].")
        return _wrap(torch.permute(inputs[0].int_repr, tuple(dims)), output_encoding)


def _normalize_pad_tuple(pad: Any) -> tuple[int, ...]:
    if isinstance(pad, int):
        return (int(pad),)
    return tuple(int(x) for x in pad)


def _pad_value_int_repr(value: float, output_encoding: OutputEncoding, device: torch.device) -> float:
    """Map a float pad constant to ``F.pad`` value in integer grid."""

    scale = output_encoding.scale.to(device=device, dtype=torch.float32).reshape(-1)[0]
    zp = output_encoding.zero_point.to(device=device, dtype=torch.int32).reshape(-1)[0]
    return float(torch.round(torch.tensor(value, device=device) / scale + zp).item())


@register_fixed_kernel(custom.Pad)
class PadInt16Kernel:
    """Reference INT16 constant pad: align input grid, then pad with quantized fill value."""

    module_type = custom.Pad

    def __call__(
        self,
        inputs: List[Int16QuantizedTensor],
        params: Dict[str, Any],
        output_encoding: OutputEncoding,
        extra: Dict[str, Any],
    ) -> Int16QuantizedTensor:
        del params
        if len(inputs) != 1:
            raise ValueError(f"Pad expects 1 input; got {len(inputs)}.")
        pad = extra.get("pad")
        if pad is None:
            raise ValueError("Pad kernel requires extra['pad'].")
        mode = str(extra.get("mode", "constant"))
        if mode != "constant":
            raise ValueError(f"Pad INT16 kernel only supports mode='constant'; got {mode!r}.")
        aligned = _requantize_identity_output(inputs[0], output_encoding)
        pad_value = _pad_value_int_repr(float(extra.get("value", 0.0)), output_encoding, aligned.int_repr.device)
        padded = F.pad(aligned.int_repr, _normalize_pad_tuple(pad), mode="constant", value=pad_value)
        return _wrap(padded, output_encoding)


@register_fixed_kernel(custom.Concat)
class ConcatInt16Kernel:
    """Reference INT16 concat: align each branch to output grid, then concat int_repr."""

    module_type = custom.Concat

    def __call__(
        self,
        inputs: List[Int16QuantizedTensor],
        params: Dict[str, Any],
        output_encoding: OutputEncoding,
        extra: Dict[str, Any],
    ) -> Int16QuantizedTensor:
        del params
        if not inputs:
            raise ValueError("Concat expects at least one input.")
        axis = int(extra.get("axis", 0))
        aligned = [align_centered_int32_to_output(item, output_encoding) for item in inputs]
        acc = torch.cat(aligned, dim=axis)
        acc = int32_add_sat(
            acc,
            output_encoding.zero_point.to(device=acc.device, dtype=torch.int32),
        )
        return _wrap(
            saturate_sim_tensor(acc, output_encoding.qmin, output_encoding.qmax),
            output_encoding,
        )

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
"""Reference INT16 fixed-point kernels for elementwise operations."""

from typing import Any, Dict, List

import torch
from torch import nn

from aimet_torch._base.nn.modules import custom
from aimet_torch.fixed_point.encoding import OutputEncoding
from aimet_torch.fixed_point.offline.multiplier import quantize_multiplier
from aimet_torch.fixed_point.registry import register_fixed_kernel
from aimet_torch.fixed_point.rounding import RoundingMode
from aimet_torch.fixed_point.requantize import (
    SIM_TENSOR_DTYPE,
    hw_ref_mode_enabled,
    int32_add_sat,
    int32_mul_sat,
    int32_sub_sat,
    requantize_int,
    saturate_mac_accumulator,
    saturate_sim_tensor,
)
from aimet_torch.fixed_point.tensor import Int16QuantizedTensor


def _center_tensor(tensor: Int16QuantizedTensor) -> torch.Tensor:
    return tensor.centered_int32()


def align_centered_int32_to_output(
    tensor: Int16QuantizedTensor,
    output_encoding: OutputEncoding,
) -> torch.Tensor:
    """Align one INT16 tensor to the output quantizer grid (Add / Concat inputs)."""

    output_scale = output_encoding.scale.to(
        device=tensor.int_repr.device, dtype=torch.float32
    )
    input_scale = tensor.scale.to(device=tensor.int_repr.device, dtype=torch.float32)
    input_zp = tensor.zero_point.to(
        device=tensor.int_repr.device, dtype=torch.int32
    )
    output_zp = output_encoding.zero_point.to(
        device=tensor.int_repr.device, dtype=torch.int32
    )

    if torch.equal(input_scale, output_scale) and torch.equal(input_zp, output_zp):
        return _center_tensor(tensor)

    multiplier, rshift = quantize_multiplier((input_scale / output_scale).detach())
    rounding = (
        RoundingMode.HALF_UP if hw_ref_mode_enabled() else RoundingMode.HALF_TO_EVEN
    )
    aligned = requantize_int(
        _center_tensor(tensor),
        multiplier.to(device=tensor.int_repr.device),
        rshift.to(device=tensor.int_repr.device),
        torch.zeros_like(output_zp, dtype=torch.int32),
        output_encoding.qmin,
        output_encoding.qmax,
        rounding_mode=rounding,
    )
    return aligned.to(torch.int32)


def _wrap_like_output(
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


def _requantize(
    acc: torch.Tensor,
    output_encoding: OutputEncoding,
) -> Int16QuantizedTensor:
    if output_encoding.multiplier is None or output_encoding.rshift is None:
        raise ValueError("output_encoding must provide multiplier and rshift.")
    return _wrap_like_output(
        requantize_int(
            saturate_mac_accumulator(acc),
            output_encoding.multiplier.to(device=acc.device),
            output_encoding.rshift.to(device=acc.device),
            output_encoding.zero_point.to(device=acc.device, dtype=torch.int32),
            output_encoding.qmin,
            output_encoding.qmax,
        ),
        output_encoding,
    )


class _BinaryAlignedKernel:
    """Base class for binary integer kernels with per-input scale alignment."""

    op_name = "binary"

    def _apply(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _align_to_output(
        self,
        tensor: Int16QuantizedTensor,
        output_encoding: OutputEncoding,
    ) -> torch.Tensor:
        return align_centered_int32_to_output(tensor, output_encoding)

    def __call__(
        self,
        inputs: List[Int16QuantizedTensor],
        params: Dict[str, Any],
        output_encoding: OutputEncoding,
        extra: Dict[str, Any],
    ) -> Int16QuantizedTensor:
        del params, extra
        if len(inputs) != 2:
            raise ValueError(f"{self.op_name} expects 2 inputs; got {len(inputs)}.")

        lhs = self._align_to_output(inputs[0], output_encoding)
        rhs = self._align_to_output(inputs[1], output_encoding)
        acc = self._apply(lhs, rhs)
        acc = int32_add_sat(
            acc,
            output_encoding.zero_point.to(device=acc.device, dtype=torch.int32),
        )
        return _wrap_like_output(
            saturate_sim_tensor(acc, output_encoding.qmin, output_encoding.qmax),
            output_encoding,
        )


@register_fixed_kernel(custom.Add)
class AddInt16Kernel(_BinaryAlignedKernel):
    """Reference INT16 Add kernel with input scale alignment."""

    module_type = custom.Add
    op_name = "Add"

    def _apply(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return int32_add_sat(x, y)


@register_fixed_kernel(custom.Subtract)
class SubtractInt16Kernel(_BinaryAlignedKernel):
    """Reference INT16 Subtract kernel with input scale alignment."""

    module_type = custom.Subtract
    op_name = "Subtract"

    def _apply(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return int32_sub_sat(x, y)


@register_fixed_kernel(custom.Multiply)
class MultiplyInt16Kernel:
    """Reference INT16 Multiply kernel."""

    module_type = custom.Multiply

    def __call__(
        self,
        inputs: List[Int16QuantizedTensor],
        params: Dict[str, Any],
        output_encoding: OutputEncoding,
        extra: Dict[str, Any],
    ) -> Int16QuantizedTensor:
        del params, extra
        if len(inputs) != 2:
            raise ValueError(f"Multiply expects 2 inputs; got {len(inputs)}.")
        acc = int32_mul_sat(_center_tensor(inputs[0]), _center_tensor(inputs[1]))
        return _requantize(acc, output_encoding)


@register_fixed_kernel(nn.ReLU)
class ReLUInt16Kernel:
    """Reference INT16 ReLU kernel."""

    module_type = nn.ReLU

    def __call__(
        self,
        inputs: List[Int16QuantizedTensor],
        params: Dict[str, Any],
        output_encoding: OutputEncoding,
        extra: Dict[str, Any],
    ) -> Int16QuantizedTensor:
        del params
        if len(inputs) != 1:
            raise ValueError(f"ReLU expects 1 input; got {len(inputs)}.")
        centered = _center_tensor(inputs[0])
        clamped = torch.clamp_min(centered, 0)
        if output_encoding.multiplier is None or output_encoding.rshift is None:
            y = clamped + output_encoding.zero_point.to(
                device=clamped.device, dtype=torch.int32
            )
            return _wrap_like_output(
                saturate_sim_tensor(y, output_encoding.qmin, output_encoding.qmax),
                output_encoding,
            )
        return _requantize(clamped, output_encoding)


@register_fixed_kernel(nn.ReLU6)
class ReLU6Int16Kernel(ReLUInt16Kernel):
    """Reference INT16 ReLU6 kernel."""

    module_type = nn.ReLU6

    def __call__(
        self,
        inputs: List[Int16QuantizedTensor],
        params: Dict[str, Any],
        output_encoding: OutputEncoding,
        extra: Dict[str, Any],
    ) -> Int16QuantizedTensor:
        x = inputs[0]
        centered = _center_tensor(x)
        scale = x.scale.to(device=x.int_repr.device, dtype=torch.float32)
        relu6_max = torch.round(torch.tensor(6.0, device=x.int_repr.device) / scale).to(
            torch.int32
        )
        clamped = torch.clamp(centered, min=0, max=int(relu6_max.item()))
        if output_encoding.multiplier is None or output_encoding.rshift is None:
            y = clamped + output_encoding.zero_point.to(
                device=clamped.device, dtype=torch.int32
            )
            return _wrap_like_output(
                saturate_sim_tensor(y, output_encoding.qmin, output_encoding.qmax),
                output_encoding,
            )
        return _requantize(clamped, output_encoding)


@register_fixed_kernel(nn.Hardtanh)
class ClampInt16Kernel:
    """Reference INT16 clamp-like kernel."""

    module_type = nn.Hardtanh

    def __call__(
        self,
        inputs: List[Int16QuantizedTensor],
        params: Dict[str, Any],
        output_encoding: OutputEncoding,
        extra: Dict[str, Any],
    ) -> Int16QuantizedTensor:
        del params
        if len(inputs) != 1:
            raise ValueError(f"Clamp expects 1 input; got {len(inputs)}.")
        min_int = extra.get("min_int", output_encoding.qmin)
        max_int = extra.get("max_int", output_encoding.qmax)
        clamped = torch.clamp(inputs[0].int_repr, int(min_int), int(max_int))
        return _wrap_like_output(clamped, output_encoding)


@register_fixed_kernel(custom.Clamp)
class FunctionalClampInt16Kernel(ClampInt16Kernel):
    """``torch.clamp`` / ``F.clamp`` INT16 kernel."""

    module_type = custom.Clamp


def _dequant_float(tensor: Int16QuantizedTensor) -> torch.Tensor:
    scale = tensor.scale.to(device=tensor.int_repr.device, dtype=torch.float32)
    while scale.ndim < tensor.int_repr.ndim:
        scale = scale.unsqueeze(-1)
    return tensor.centered_int32().to(torch.float32) * scale


def _quantize_from_float(
    value: torch.Tensor,
    output_encoding: OutputEncoding,
) -> Int16QuantizedTensor:
    scale = output_encoding.scale.to(device=value.device, dtype=torch.float32)
    zp = output_encoding.zero_point.to(device=value.device, dtype=torch.float32)
    while scale.ndim < value.ndim:
        scale = scale.unsqueeze(-1)
        zp = zp.unsqueeze(-1)
    q = torch.round(value / scale + zp)
    q = saturate_sim_tensor(q.to(torch.int32), output_encoding.qmin, output_encoding.qmax)
    return _wrap_like_output(q.to(SIM_TENSOR_DTYPE), output_encoding)


class _UnaryFloatRefKernel:
    """Reference INT16 unary op via dequant → float fn → requant."""

    op_name = "unary"

    def _apply(self, value: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def __call__(
        self,
        inputs: List[Int16QuantizedTensor],
        params: Dict[str, Any],
        output_encoding: OutputEncoding,
        extra: Dict[str, Any],
    ) -> Int16QuantizedTensor:
        del params, extra
        if len(inputs) != 1:
            raise ValueError(f"{self.op_name} expects 1 input; got {len(inputs)}.")
        return _quantize_from_float(self._apply(_dequant_float(inputs[0])), output_encoding)


@register_fixed_kernel(custom.Abs)
class AbsInt16Kernel(_UnaryFloatRefKernel):
    """Reference INT16 Abs kernel."""

    module_type = custom.Abs
    op_name = "Abs"

    def _apply(self, value: torch.Tensor) -> torch.Tensor:
        return value.abs()


@register_fixed_kernel(custom.ElementwiseUnarySign)
class SignInt16Kernel(_UnaryFloatRefKernel):
    """Reference INT16 sign kernel."""

    module_type = custom.ElementwiseUnarySign
    op_name = "Sign"

    def _apply(self, value: torch.Tensor) -> torch.Tensor:
        return value.sign()


@register_fixed_kernel(custom.Divide)
class DivideInt16Kernel:
    """Reference INT16 Divide kernel (float reference path for BN / normalize)."""

    module_type = custom.Divide

    def __call__(
        self,
        inputs: List[Int16QuantizedTensor],
        params: Dict[str, Any],
        output_encoding: OutputEncoding,
        extra: Dict[str, Any],
    ) -> Int16QuantizedTensor:
        del params, extra
        if len(inputs) != 2:
            raise ValueError(f"Divide expects 2 inputs; got {len(inputs)}.")
        num = _dequant_float(inputs[0])
        den = _dequant_float(inputs[1])
        eps = torch.tensor(1e-12, device=den.device, dtype=den.dtype)
        out = num / torch.where(den.abs() < eps, eps * den.sign().clamp(min=1.0), den)
        return _quantize_from_float(out, output_encoding)


@register_fixed_kernel(custom.Clip)
class FunctionalClipInt16Kernel(ClampInt16Kernel):
    """``torch.clip`` INT16 kernel."""

    module_type = custom.Clip

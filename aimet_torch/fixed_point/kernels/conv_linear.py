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
"""Reference INT16 fixed-point kernels for Conv and Linear modules."""

from typing import Any, Dict, List, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from aimet_torch._base.nn.modules import custom
from aimet_torch.fixed_point.encoding import OutputEncoding
from aimet_torch.fixed_point.kernels._im2col import im2col_int
from aimet_torch.fixed_point.registry import register_fixed_kernel
from aimet_torch.fixed_point.requantize import (
    SIM_TENSOR_DTYPE,
    _env_truthy,
    mac_accumulator_int32_sat_enabled,
    requantize_int,
    saturate_mac_accumulator,
)
from aimet_torch.fixed_point.tensor import Int16QuantizedTensor


def _require_single_input(inputs: List[Int16QuantizedTensor]) -> Int16QuantizedTensor:
    if len(inputs) != 1:
        raise ValueError(f"Expected one input tensor; got {len(inputs)}.")
    return inputs[0]


def _get_weight(params: Dict[str, Any]) -> Int16QuantizedTensor:
    weight = params.get("weight")
    if not isinstance(weight, Int16QuantizedTensor):
        raise TypeError("params['weight'] must be an Int16QuantizedTensor.")
    return weight


def _get_bias(params: Dict[str, Any], device: torch.device) -> torch.Tensor:
    bias = params.get("bias")
    if bias is None:
        return torch.zeros((), dtype=torch.int32, device=device)
    if not isinstance(bias, torch.Tensor):
        raise TypeError("params['bias'] must be a torch.Tensor when provided.")
    if bias.dtype != torch.int32:
        raise TypeError(f"params['bias'] must be torch.int32; got {bias.dtype}.")
    return bias


def _center_tensor(tensor: Int16QuantizedTensor) -> torch.Tensor:
    return tensor.centered_int32()


def _requantize_output(
    acc: torch.Tensor,
    output_encoding: OutputEncoding,
) -> Int16QuantizedTensor:
    if output_encoding.multiplier is None or output_encoding.rshift is None:
        raise ValueError("output_encoding must provide multiplier and rshift.")

    int_repr = requantize_int(
        saturate_mac_accumulator(acc),
        output_encoding.multiplier.to(device=acc.device),
        output_encoding.rshift.to(device=acc.device),
        output_encoding.zero_point.to(device=acc.device, dtype=torch.int32),
        output_encoding.qmin,
        output_encoding.qmax,
    )
    return Int16QuantizedTensor(
        int_repr=int_repr,
        scale=output_encoding.scale.to(device=acc.device),
        zero_point=output_encoding.zero_point.to(device=acc.device, dtype=torch.int32),
        qmin=output_encoding.qmin,
        qmax=output_encoding.qmax,
        axis=output_encoding.axis,
    )


def _as_tuple(value: Any, ndim: int) -> Tuple[int, ...]:
    if isinstance(value, Sequence):
        value = tuple(value)
        if len(value) != ndim:
            raise ValueError(f"Expected tuple of length {ndim}; got {value}.")
        return value
    return (int(value),) * ndim


def _add_bias(acc: torch.Tensor, bias: torch.Tensor, view_shape: Tuple[int, ...]):
    if bias.numel() == 1:
        return acc + bias.to(device=acc.device, dtype=acc.dtype)
    return acc + bias.to(device=acc.device, dtype=acc.dtype).view(view_shape)


def _matmul_fast_fp32_enabled() -> bool:
    """Opt-in: use float32 CUDA matmul for speed at the cost of bit-exactness.

    A single ``int16 * int16`` product can reach 2**30, exceeding float32's
    24-bit exact-integer range (2**24); the K-accumulation diverges further.
    This is a *fast approximate preview* only and MUST NOT back sign-off.
    Default is float64 accumulation, which is bit-exact for int16 ranges and
    matches the CPU integer reference across devices.
    """

    return _env_truthy("AIMET_RX_MATMUL_FAST_FP32")


def _int32_matmul(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Integer MAC with CUDA-safe fallback (PyTorch CUDA lacks integer matmul)."""

    if mac_accumulator_int32_sat_enabled():
        prod = torch.matmul(lhs.to(torch.float64), rhs.to(torch.float64))
        return saturate_mac_accumulator(prod.to(torch.int64))

    if lhs.is_cuda or rhs.is_cuda:
        if _matmul_fast_fp32_enabled():
            prod = torch.matmul(lhs.to(torch.float32), rhs.to(torch.float32))
            return prod.round().to(torch.int32)
        # Default: float64 is bit-exact for int16 operands and device-invariant.
        prod = torch.matmul(lhs.to(torch.float64), rhs.to(torch.float64))
        return prod.round().to(torch.int32)

    return torch.matmul(lhs, rhs)


@register_fixed_kernel(nn.Linear)
class LinearInt16Kernel:
    """Reference INT16 kernel for torch.nn.Linear."""

    module_type = nn.Linear

    def __call__(
        self,
        inputs: List[Int16QuantizedTensor],
        params: Dict[str, Any],
        output_encoding: OutputEncoding,
        extra: Dict[str, Any],
    ) -> Int16QuantizedTensor:
        del extra
        x = _require_single_input(inputs)
        weight = _get_weight(params)
        bias = _get_bias(params, x.int_repr.device)

        x_centered = _center_tensor(x)
        w_centered = _center_tensor(weight)
        acc = _int32_matmul(x_centered, w_centered.transpose(-1, -2))
        acc = saturate_mac_accumulator(_add_bias(acc, bias, (1,) * (acc.dim() - 1) + (-1,)))

        return _requantize_output(acc, output_encoding)


@register_fixed_kernel(nn.Conv2d)
class Conv2dInt16Kernel:
    """Reference INT16 kernel for torch.nn.Conv2d using unfold + int32 matmul."""

    module_type = nn.Conv2d

    def __call__(
        self,
        inputs: List[Int16QuantizedTensor],
        params: Dict[str, Any],
        output_encoding: OutputEncoding,
        extra: Dict[str, Any],
    ) -> Int16QuantizedTensor:
        x = _require_single_input(inputs)
        weight = _get_weight(params)
        bias = _get_bias(params, x.int_repr.device)

        stride = _as_tuple(extra.get("stride", 1), 2)
        padding = _as_tuple(extra.get("padding", 0), 2)
        dilation = _as_tuple(extra.get("dilation", 1), 2)
        groups = int(extra.get("groups", 1))

        x_centered = _center_tensor(x)
        w_centered = _center_tensor(weight)

        if mac_accumulator_int32_sat_enabled():
            x_mac = x_centered.to(torch.float64)
            w_mac = w_centered.to(torch.float64)
        else:
            x_mac = x_centered.to(torch.float32)
            w_mac = w_centered.to(torch.float32)

        acc = F.conv2d(
            x_mac,
            w_mac,
            bias=None,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        if mac_accumulator_int32_sat_enabled():
            acc = saturate_mac_accumulator(acc.to(torch.int64))
        else:
            acc = acc.round().to(torch.int32)
        acc = saturate_mac_accumulator(_add_bias(acc, bias, (1, -1, 1, 1)))

        return _requantize_output(acc, output_encoding)

    @staticmethod
    def _conv2d_unfold(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        stride: Tuple[int, int],
        padding: Tuple[int, int],
        dilation: Tuple[int, int],
    ) -> torch.Tensor:
        n_batch, _, input_h, input_w = x.shape
        out_channels, _, kernel_h, kernel_w = weight.shape

        # Pure-integer im2col (ADR-002: no float intermediates in int16_fixed_eval).
        x_unfold = im2col_int(
            x.to(SIM_TENSOR_DTYPE),
            (kernel_h, kernel_w),
            dilation=dilation,
            padding=padding,
            stride=stride,
        )
        weight_matrix = weight.view(out_channels, -1)
        acc = _int32_matmul(weight_matrix, x_unfold).transpose(1, 2)

        out_h = (
            input_h + 2 * padding[0] - dilation[0] * (kernel_h - 1) - 1
        ) // stride[0] + 1
        out_w = (
            input_w + 2 * padding[1] - dilation[1] * (kernel_w - 1) - 1
        ) // stride[1] + 1
        acc = acc.transpose(1, 2).reshape(n_batch, out_channels, out_h, out_w)
        return saturate_mac_accumulator(_add_bias(acc, bias, (1, -1, 1, 1)))

    @classmethod
    def _grouped_conv2d(
        cls,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        stride: Tuple[int, int],
        padding: Tuple[int, int],
        dilation: Tuple[int, int],
        groups: int,
    ) -> torch.Tensor:
        x_groups = x.chunk(groups, dim=1)
        w_groups = weight.chunk(groups, dim=0)
        bias_groups = bias.chunk(groups) if bias.numel() > 1 else [bias] * groups
        outputs = [
            cls._conv2d_unfold(x_g, w_g, b_g, stride, padding, dilation)
            for x_g, w_g, b_g in zip(x_groups, w_groups, bias_groups)
        ]
        return torch.cat(outputs, dim=1)


@register_fixed_kernel(nn.Conv1d)
class Conv1dInt16Kernel:
    """Reference INT16 kernel for torch.nn.Conv1d via Conv2d kernel."""

    module_type = nn.Conv1d

    def __call__(
        self,
        inputs: List[Int16QuantizedTensor],
        params: Dict[str, Any],
        output_encoding: OutputEncoding,
        extra: Dict[str, Any],
    ) -> Int16QuantizedTensor:
        x = _require_single_input(inputs)
        weight = _get_weight(params)

        x_2d = Int16QuantizedTensor(
            int_repr=x.int_repr.unsqueeze(-2),
            scale=x.scale,
            zero_point=x.zero_point,
            qmin=x.qmin,
            qmax=x.qmax,
            axis=x.axis,
        )
        weight_2d = Int16QuantizedTensor(
            int_repr=weight.int_repr.unsqueeze(-2),
            scale=weight.scale,
            zero_point=weight.zero_point,
            qmin=weight.qmin,
            qmax=weight.qmax,
            axis=weight.axis,
        )
        extra_2d = {
            "stride": (1, _as_tuple(extra.get("stride", 1), 1)[0]),
            "padding": (0, _as_tuple(extra.get("padding", 0), 1)[0]),
            "dilation": (1, _as_tuple(extra.get("dilation", 1), 1)[0]),
            "groups": extra.get("groups", 1),
        }
        params_2d = dict(params)
        params_2d["weight"] = weight_2d
        output = Conv2dInt16Kernel()(inputs=[x_2d], params=params_2d, output_encoding=output_encoding, extra=extra_2d)
        return Int16QuantizedTensor(
            int_repr=output.int_repr.squeeze(-2),
            scale=output.scale,
            zero_point=output.zero_point,
            qmin=output.qmin,
            qmax=output.qmax,
            axis=output.axis,
        )


@register_fixed_kernel(nn.Conv3d)
class Conv3dInt16Kernel:
    """Reference INT16 kernel for torch.nn.Conv3d.

    Default: float32 ``conv3d`` + round (e2e parity). HW ref: float64 MAC from exact
    int64 centered operands, then :func:`saturate_mac_accumulator` (no int32 wrap).
    """

    module_type = nn.Conv3d

    def __call__(
        self,
        inputs: List[Int16QuantizedTensor],
        params: Dict[str, Any],
        output_encoding: OutputEncoding,
        extra: Dict[str, Any],
    ) -> Int16QuantizedTensor:
        x = _require_single_input(inputs)
        weight = _get_weight(params)
        bias = _get_bias(params, x.int_repr.device)

        stride = _as_tuple(extra.get("stride", 1), 3)
        padding = _as_tuple(extra.get("padding", 0), 3)
        dilation = _as_tuple(extra.get("dilation", 1), 3)
        groups = int(extra.get("groups", 1))

        x_centered = _center_tensor(x)
        w_centered = _center_tensor(weight)
        if mac_accumulator_int32_sat_enabled():
            mac_dtype = torch.float64
            x_mac = x_centered.to(torch.float64)
            w_mac = w_centered.to(torch.float64)
        else:
            mac_dtype = torch.float32
            x_mac = x_centered.to(torch.float32)
            w_mac = w_centered.to(torch.float32)

        acc = F.conv3d(
            x_mac,
            w_mac,
            bias=None,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        if mac_dtype is torch.float64:
            acc = saturate_mac_accumulator(acc.to(torch.int64))
        else:
            acc = acc.round().to(torch.int32)
        acc = saturate_mac_accumulator(
            _add_bias(acc, bias, (1, -1, 1, 1, 1))
        )

        return _requantize_output(acc, output_encoding)


@register_fixed_kernel(custom.MatMul)
class MatMulInt16Kernel:
    """Reference INT16 kernel for ``custom.MatMul`` (e.g. BandConverter band matrix)."""

    module_type = custom.MatMul

    def __call__(
        self,
        inputs: List[Int16QuantizedTensor],
        params: Dict[str, Any],
        output_encoding: OutputEncoding,
        extra: Dict[str, Any],
    ) -> Int16QuantizedTensor:
        del params, extra
        if len(inputs) != 2:
            raise ValueError(f"MatMul expects 2 inputs; got {len(inputs)}.")
        lhs = _center_tensor(inputs[0])
        rhs = _center_tensor(inputs[1])
        acc = _int32_matmul(lhs, rhs)
        return _requantize_output(acc, output_encoding)

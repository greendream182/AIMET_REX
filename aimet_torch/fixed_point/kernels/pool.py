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
"""Reference INT16 fixed-point kernels for pooling operations."""

from typing import Any, Dict, List, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from aimet_torch._base.nn.modules import custom
from aimet_torch.fixed_point.capabilities import KernelKind
from aimet_torch.fixed_point.encoding import OutputEncoding
from aimet_torch.fixed_point.kernels._contracts import (
    require_adaptive_avgpool_output_unit,
    require_int32_saturated_accumulator,
    require_kernel_kind_encoding_contract,
    require_pool2d_operand_limits,
    require_reduce_size_matches_extra,
)
from aimet_torch.fixed_point.kernels._im2col import im2col_int
from aimet_torch.fixed_point.registry import register_fixed_kernel
from aimet_torch.fixed_point.requantize import (
    SIM_TENSOR_DTYPE,
    int32_sum_sat,
    requantize_int,
    saturate_mac_accumulator,
    saturate_sim_tensor,
)
from aimet_torch.fixed_point.tensor import Int16QuantizedTensor

# Spec doc/04_算子详细规格/04_09 pinned set: {2x2, 4x4, 4x2, 2x4}.
_AVGPOOL_ALLOWED_KERNELS: Tuple[Tuple[int, int], ...] = (
    (2, 2),
    (4, 4),
    (4, 2),
    (2, 4),
)


def _as_tuple(value: Any, ndim: int) -> Tuple[int, ...]:
    if isinstance(value, Sequence):
        value = tuple(value)
        if len(value) != ndim:
            raise ValueError(f"Expected tuple of length {ndim}; got {value}.")
        return value
    return (int(value),) * ndim


def _single_input(inputs: List[Int16QuantizedTensor]) -> Int16QuantizedTensor:
    if len(inputs) != 1:
        raise ValueError(f"Pooling expects 1 input; got {len(inputs)}.")
    return inputs[0]


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


class _MaxPool2dKernel:
    def __call__(
        self,
        inputs: List[Int16QuantizedTensor],
        params: Dict[str, Any],
        output_encoding: OutputEncoding,
        extra: Dict[str, Any],
    ) -> Int16QuantizedTensor:
        del params
        x = _single_input(inputs)
        require_kernel_kind_encoding_contract(
            KernelKind.SAME_GRID_VALUE,
            output_encoding,
            op_name="MaxPool2d",
            tensor=x,
        )
        kernel_size = _as_tuple(extra.get("kernel_size", 1), 2)
        stride = extra.get("stride", kernel_size)
        stride = _as_tuple(stride, 2)
        padding = _as_tuple(extra.get("padding", 0), 2)
        dilation = _as_tuple(extra.get("dilation", 1), 2)
        ceil_mode = bool(extra.get("ceil_mode", False))
        require_pool2d_operand_limits(
            kernel_size=kernel_size,
            padding=padding,
            op_name="MaxPool2d",
            max_kernel=3,
        )

        y = F.max_pool2d(
            x.int_repr,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
        )
        return _wrap(y, output_encoding)


@register_fixed_kernel(nn.MaxPool2d)
class MaxPool2dInt16Kernel(_MaxPool2dKernel):
    """Reference INT16 MaxPool2d kernel."""

    module_type = nn.MaxPool2d


@register_fixed_kernel(custom.MaxPool2d)
class CustomMaxPool2dInt16Kernel(_MaxPool2dKernel):
    """Reference INT16 MaxPool2d kernel for AIMET custom wrapper."""

    module_type = custom.MaxPool2d


class _AvgPool2dKernel:
    def __call__(
        self,
        inputs: List[Int16QuantizedTensor],
        params: Dict[str, Any],
        output_encoding: OutputEncoding,
        extra: Dict[str, Any],
    ) -> Int16QuantizedTensor:
        del params
        require_kernel_kind_encoding_contract(
            KernelKind.REQUANTIZING,
            output_encoding,
            op_name="AvgPool2d",
        )

        x = _single_input(inputs)
        kernel_size = _as_tuple(extra.get("kernel_size", 1), 2)
        stride = extra.get("stride", kernel_size)
        stride = _as_tuple(stride, 2)
        padding = _as_tuple(extra.get("padding", 0), 2)
        require_pool2d_operand_limits(
            kernel_size=kernel_size,
            padding=padding,
            op_name="AvgPool2d",
            allowed_kernels=_AVGPOOL_ALLOWED_KERNELS,
        )
        # Spec 04_09 folds 1/N (N=k_t*k_f) into output ``M/rshift``; the
        # adapter must publish that N so a stale-extra path cannot silently
        # mis-scale the result.
        require_reduce_size_matches_extra(
            extra.get("reduce_size"),
            kernel_size[0] * kernel_size[1],
            op_name="AvgPool2d",
        )

        # Pure-integer im2col (see _im2col.py). int32 container preserves
        # values exactly within the [qmin, qmax] grid (ADR-013).
        unfolded = im2col_int(
            x.int_repr.to(SIM_TENSOR_DTYPE),
            kernel_size,
            padding=padding,
            stride=stride,
        )
        kernel_area = kernel_size[0] * kernel_size[1]
        n_batch, channels, input_h, input_w = x.int_repr.shape
        del input_h, input_w
        out_h = int((x.int_repr.shape[-2] + 2 * padding[0] - kernel_size[0]) / stride[0]) + 1
        out_w = int((x.int_repr.shape[-1] + 2 * padding[1] - kernel_size[1]) / stride[1]) + 1
        centered = unfolded - x.zero_point.to(device=unfolded.device, dtype=torch.int32)
        acc = int32_sum_sat(centered.view(n_batch, channels, kernel_area, -1), dim=2)
        acc = acc.view(n_batch, channels, out_h, out_w)
        acc_sat = saturate_mac_accumulator(acc)
        require_int32_saturated_accumulator(acc_sat, op_name="AvgPool2d")

        y = requantize_int(
            acc_sat,
            output_encoding.multiplier.to(device=acc.device),
            output_encoding.rshift.to(device=acc.device),
            output_encoding.zero_point.to(device=acc.device, dtype=torch.int32),
            output_encoding.qmin,
            output_encoding.qmax,
        )
        return _wrap(y, output_encoding)


@register_fixed_kernel(nn.AvgPool2d)
class AvgPool2dInt16Kernel(_AvgPool2dKernel):
    """Reference INT16 AvgPool2d kernel."""

    module_type = nn.AvgPool2d


@register_fixed_kernel(custom.AvgPool2d)
class CustomAvgPool2dInt16Kernel(_AvgPool2dKernel):
    """Reference INT16 AvgPool2d kernel for AIMET custom wrapper."""

    module_type = custom.AvgPool2d


class _MeanInt16Kernel:
    """Reference INT16 reduce-mean kernel.

    The adapter folds ``1/N`` (with ``N`` the product of the reduced-axis
    lengths) into the output ``real_multiplier`` and publishes that ``N`` via
    ``extra["reduce_size"]``. This kernel re-derives ``N`` from ``dim`` plus
    the input shape and asserts equality through
    :func:`require_reduce_size_matches_extra`, so any adapter path that
    forgets / mis-folds ``1/N`` raises immediately rather than producing a
    silently mis-scaled output. Once that contract holds, the kernel only
    needs to zero-centre, accumulate, and pass through ``requantize_int``.
    """

    def __call__(
        self,
        inputs: List[Int16QuantizedTensor],
        params: Dict[str, Any],
        output_encoding: OutputEncoding,
        extra: Dict[str, Any],
    ) -> Int16QuantizedTensor:
        del params
        require_kernel_kind_encoding_contract(
            KernelKind.REQUANTIZING,
            output_encoding,
            op_name=type(self).__name__,
        )

        x = _single_input(inputs)
        require_adaptive_avgpool_output_unit(
            output_size=extra.get("output_size"),
            op_name=type(self).__name__,
        )
        dim = extra.get("dim")
        if dim is None:
            dims: Tuple[int, ...] = tuple(range(x.int_repr.dim()))
        elif isinstance(dim, int):
            dims = (int(dim) % x.int_repr.dim(),)
        else:
            dims = tuple(int(d) % x.int_repr.dim() for d in dim)
        keepdim = bool(extra.get("keepdim", False))

        derived_reduce_size = 1
        for d in dims:
            derived_reduce_size *= int(x.int_repr.shape[d])
        require_reduce_size_matches_extra(
            extra.get("reduce_size"),
            derived_reduce_size,
            op_name=type(self).__name__,
        )

        zp = x.zero_point.to(device=x.int_repr.device, dtype=torch.int32)
        centered = x.int_repr.to(torch.int32) - zp
        acc = int32_sum_sat(centered, dim=dims, keepdim=keepdim)
        acc_sat = saturate_mac_accumulator(acc)
        require_int32_saturated_accumulator(acc_sat, op_name=type(self).__name__)

        y = requantize_int(
            acc_sat,
            output_encoding.multiplier.to(device=acc.device),
            output_encoding.rshift.to(device=acc.device),
            output_encoding.zero_point.to(device=acc.device, dtype=torch.int32),
            output_encoding.qmin,
            output_encoding.qmax,
        )
        return _wrap(y, output_encoding)


@register_fixed_kernel(custom.Mean)
class MeanInt16Kernel(_MeanInt16Kernel):
    """Reference INT16 kernel for ``aimet_torch`` ``custom.Mean``."""

    module_type = custom.Mean


@register_fixed_kernel(custom.AdaptiveAvgPool2d)
class AdaptiveAvgPool2dInt16Kernel(_MeanInt16Kernel):
    """Reference INT16 kernel for ``custom.AdaptiveAvgPool2d``.

    Only ``output_size=(1,1)`` is dispatched today; the adapter populates
    ``extra={"dim": (2, 3), "keepdim": True, ...}``, making this op a thin
    spatial-mean over ``(H, W)``. Non-(1,1) outputs are refused at dispatch
    time so they fall back to the float QDQ path.
    """

    module_type = custom.AdaptiveAvgPool2d

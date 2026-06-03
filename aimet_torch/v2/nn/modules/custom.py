# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
"""Quantized definitions for custom modules of AIMET"""

import copy
import importlib
from contextlib import contextmanager
from typing import Optional
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from aimet_torch._base.nn.modules.custom import *  # pylint: disable=wildcard-import, unused-wildcard-import
from aimet_torch.v2.quantization.tensor import QuantizedTensorBase
from aimet_torch.v2.quantization.affine import QuantizeDequantize
from ..true_quant import (
    QuantizationMixin,
    _DispatchMixin,
    _quantize_if_applicable,
    _quantize_dequantize_if_applicable,
)

# NOTE: Disabling due to pylint false alarm in ModuleList
# pylint: disable=not-callable

try:
    _OptionalQuantGRU = importlib.import_module("quant_gru").QuantGRU
except (ImportError, AttributeError):
    _OptionalQuantGRU = None


@QuantizationMixin.implements(Sin)
class QuantizedSin(_DispatchMixin, QuantizationMixin, Sin):
    """Quantized Sin"""

    _builtin_torch_fn = torch.sin


@QuantizationMixin.implements(Cos)
class QuantizedCos(_DispatchMixin, QuantizationMixin, Cos):
    """Quantized Cos"""

    _builtin_torch_fn = torch.cos


@QuantizationMixin.implements(AvgPool2d)
class QuantizedAvgPool2d(_DispatchMixin, QuantizationMixin, AvgPool2d):
    """Quantized AvgPool2d"""

    _builtin_torch_fn = F.avg_pool2d

    def _is_dynamo_traceable(self):
        # F.avg_pool2d isn't dynamo-traceable
        return False


@QuantizationMixin.implements(Reshape)
class QuantizedReshape(_DispatchMixin, QuantizationMixin, Reshape):
    """Quantized Reshape"""

    _builtin_torch_fn = torch.reshape

    @staticmethod
    def _normalize_shape(shape):
        if isinstance(shape, Tensor):
            shape = shape.detach().cpu().reshape(-1).tolist()
        if isinstance(shape, torch.Size):
            return tuple(shape)
        dims = []
        for dim in shape:
            dim_int = int(dim)
            if dim_int != dim:
                raise ValueError(f"Reshape shape dimensions must be integral; got {dim}.")
            dims.append(dim_int)
        return tuple(dims)

    def _builtin_torch_fn_helper(self, fn):
        def reshape(input, shape):  # pylint: disable=redefined-builtin
            input_qtzr = self.input_quantizers[0] if self.input_quantizers else None
            input = _quantize_dequantize_if_applicable(input, input_qtzr)
            output = fn(input, self._normalize_shape(shape))
            return _quantize_dequantize_if_applicable(output, self.output_quantizers[0])

        return reshape

    def _custom_kernel_helper(self, fn):
        def reshape(input, shape):  # pylint: disable=redefined-builtin
            input_qtzr = self.input_quantizers[0] if self.input_quantizers else None
            input = _quantize_if_applicable(input, input_qtzr)
            output_encodings = (
                self.output_quantizers[0].get_encodings()
                if self.output_quantizers[0]
                else None
            )
            return fn(input, self._normalize_shape(shape), output_encodings=output_encodings)

        return reshape

    def _can_use_torch_function_dispatch(self, builtin_torch_fn):
        return False

    def _is_dynamo_traceable(self):
        # torch.reshape isn't dynamo-traceable
        return False


@QuantizationMixin.implements(RSqrt)
class QuantizedRSqrt(_DispatchMixin, QuantizationMixin, RSqrt):
    """Quantized RSqrt"""

    _builtin_torch_fn = torch.rsqrt


@QuantizationMixin.implements(MatMul)
class QuantizedMatMul(_DispatchMixin, QuantizationMixin, MatMul):
    """Quantized MatMul"""

    __quant_init__ = QuantizationMixin.__binary__
    _builtin_torch_fn = torch.matmul


@QuantizationMixin.implements(Add)
class QuantizedAdd(_DispatchMixin, QuantizationMixin, Add):
    """Quantized Add"""

    __quant_init__ = QuantizationMixin.__binary__
    _builtin_torch_fn = torch.add


@QuantizationMixin.implements(Multiply)
class QuantizedMultiply(_DispatchMixin, QuantizationMixin, Multiply):
    """Quantized Multiply"""

    __quant_init__ = QuantizationMixin.__binary__
    _builtin_torch_fn = torch.mul


@QuantizationMixin.implements(Subtract)
class QuantizedSubtract(_DispatchMixin, QuantizationMixin, Subtract):
    """Quantized Subtract"""

    __quant_init__ = QuantizationMixin.__binary__
    _builtin_torch_fn = torch.sub


def _safe_div(x, y, *args, **kwargs):
    """``torch.div`` with ``0/0 -> 0`` and ``x/0 -> 0`` protection (backward-safe).

    Needed for ``fixed_scale_qdq`` (and any other path where divisor activations
    may be rounded to integer-grid zero by quantization). The fp32_qdq fake-quant
    path empirically masks 0/0 to non-NaN through downstream effects, but
    fixed_scale_qdq propagates exact ``0/0 -> NaN``, which then poisons the rest
    of the forward graph (e.g. CLN ``x / sqrt(clamp(var, min=eps))`` when ``eps``
    rounds to 0). Returning 0 there matches the implicit fp32_qdq behavior.

    Implementation notes:
      * Replace ``y`` with 1 *before* the division so ``torch.div`` never actually
        evaluates ``0/0`` — important under autograd: the naive
        ``where(y==0, 0, x/y)`` form still computes ``x/y`` internally and
        ``torch.where``'s backward pass routes grad through both branches, so
        NaN from the dead branch poisons grads during QAT.
      * Pure tensor ops (no python-level ``if``) → tracing / TorchScript safe.
    """
    if not isinstance(y, torch.Tensor) or not y.is_floating_point():
        return torch.div(x, y, *args, **kwargs)
    zero_mask = (y == 0)
    safe_y = torch.where(zero_mask, torch.ones_like(y), y)
    out = torch.div(x, safe_y, *args, **kwargs)
    return torch.where(zero_mask, torch.zeros_like(out), out)


@QuantizationMixin.implements(Divide)
class QuantizedDivide(_DispatchMixin, QuantizationMixin, Divide):
    """Quantized Divide"""

    __quant_init__ = QuantizationMixin.__binary__
    _builtin_torch_fn = _safe_div


@QuantizationMixin.implements(Outer)
class QuantizedOuter(_DispatchMixin, QuantizationMixin, Outer):
    """Quantized Outer"""

    __quant_init__ = QuantizationMixin.__binary__
    _builtin_torch_fn = torch.outer


@QuantizationMixin.implements(Concat)
class QuantizedConcat(_DispatchMixin, QuantizationMixin, Concat):
    """Quantized Concat"""

    _builtin_torch_fn = torch.cat

    # pylint: disable=attribute-defined-outside-init
    def __quant_init__(self):
        super().__quant_init__()
        self._num_inputs = 1

    def export_input_encodings(self, encoding_version: str):
        """
        Extends super().export to repeat input quantizer's encodings :attr:`self._num_inputs` times
        """
        input_encodings = super().export_input_encodings(encoding_version)
        # Create separate encoding objects to avoid overriding of attributes added/updated later while exporting encodings
        return [
            copy.deepcopy(encoding) for encoding in input_encodings * self._num_inputs
        ]

    def import_input_encodings(
        self,
        encodings,
        strict: bool,
        partial: bool,
        requires_grad: Optional[bool],
        allow_overwrite: bool,
    ):
        """
        Extends super().import_input_encodings to set `self._num_inputs` based on length of encodings.
        """
        self._num_inputs = len(encodings)
        super().import_input_encodings(
            encodings,
            strict=strict,
            partial=partial,
            requires_grad=requires_grad,
            allow_overwrite=allow_overwrite,
        )

    def forward(self, *x):  # pylint: disable=arguments-differ
        """
        Quantized forward impl for custom.Concat.
        """
        self._num_inputs = len(x)
        return super().forward(*x)

    def _builtin_torch_fn_helper(self, fn: Callable[..., Tensor]):
        def cat(tensors, dim=0, *, out=None):
            input_qtzr = self.input_quantizers[0]
            tensors = tuple(
                _quantize_dequantize_if_applicable(x, input_qtzr) for x in tensors
            )
            output = fn(tensors, dim=dim, out=out)
            return _quantize_dequantize_if_applicable(output, self.output_quantizers[0])

        return cat

    def _custom_kernel_helper(self, fn: Callable[..., QuantizedTensorBase]):
        def cat(tensors, dim=0, *, out=None):
            input_qtzr = self.input_quantizers[0]
            tensors = tuple(_quantize_if_applicable(x, input_qtzr) for x in tensors)
            output_encodings = (
                self.output_quantizers[0].get_encodings()
                if self.output_quantizers[0]
                else None
            )
            return fn(tensors, dim=dim, out=out, output_encodings=output_encodings)

        return cat


@QuantizationMixin.implements(FloorDivide)
class QuantizedFloorDivide(_DispatchMixin, QuantizationMixin, FloorDivide):
    """ Quantized FloorDivide """
    __quant_init__ = QuantizationMixin.__binary__
    _builtin_torch_fn = torch.floor_divide
#
#
@QuantizationMixin.implements(Norm)
class QuantizedNorm(_DispatchMixin, QuantizationMixin, Norm):
    """Quantized Norm"""

    _builtin_torch_fn = torch.norm


@QuantizationMixin.implements(Exponential)
class QuantizedExponential(_DispatchMixin, QuantizationMixin, Exponential):
    """Quantized Exponential"""

    _builtin_torch_fn = torch.exp


@QuantizationMixin.implements(Erf)
class QuantizedErf(_DispatchMixin, QuantizationMixin, Erf):
    """Quantized Erf"""

    _builtin_torch_fn = torch.erf


@QuantizationMixin.implements(Sqrt)
class QuantizedSqrt(_DispatchMixin, QuantizationMixin, Sqrt):
    """Quantized Sqrt"""

    _builtin_torch_fn = torch.sqrt


# @QuantizationMixin.implements(Maximum)
# class QuantizedMaximum(_DispatchMixin, QuantizationMixin, Maximum):
#     """ Quantized Maximum """
#     _builtin_torch_fn = torch.maximum
#
#
# @QuantizationMixin.implements(Max)
# class QuantizedMax(_DispatchMixin, QuantizationMixin, Max):
#     """ Quantized Max """
#     _builtin_torch_fn = torch.max
#
# @QuantizationMixin.implements(AMax)
# class QuantizedAMax(_DispatchMixin, QuantizationMixin, AMax):
#     """ Quantized AMax """
#     _builtin_torch_fn = torch.amax
#
#
# @QuantizationMixin.implements(Minimum)
# class QuantizedMinimum(_DispatchMixin, QuantizationMixin, Minimum):
#     """ Quantized Minimum """
#     _builtin_torch_fn = torch.minimum
#
#
# @QuantizationMixin.implements(Min)
# class QuantizedMin(_DispatchMixin, QuantizationMixin, Min):
#     """ Quantized Min """
#     _builtin_torch_fn = torch.min
#
# @QuantizationMixin.implements(AMin)
# class QuantizedAMin(_DispatchMixin, QuantizationMixin, AMin):
#     """ Quantized AMin """
#     _builtin_torch_fn = torch.amin
#
#
# @QuantizationMixin.implements(Where)
# class QuantizedWhere(_DispatchMixin, QuantizationMixin, Where):
#     """ Quantized Where """
#     _builtin_torch_fn = torch.where
#
#
# @QuantizationMixin.implements(Greater)
# class QuantizedGreater(_DispatchMixin, QuantizationMixin, Greater):
#     """ Quantized Greater """
#     _builtin_torch_fn = torch.gt
#
#
# @QuantizationMixin.implements(Less)
# class QuantizedLess(_DispatchMixin, QuantizationMixin, Less):
#     """ Quantized Less """
#     _builtin_torch_fn = torch.lt
#
#
# @QuantizationMixin.implements(GreaterEqual)
# class QuantizedGreaterEqual(_DispatchMixin, QuantizationMixin, GreaterEqual):
#     """ Quantized GreaterEqual """
#     _builtin_torch_fn = torch.ge
#
#
# @QuantizationMixin.implements(LessEqual)
# class QuantizedLessEqual(_DispatchMixin, QuantizationMixin, LessEqual):
#     """ Quantized LessEqual """
#     _builtin_torch_fn = torch.le
#
#
# @QuantizationMixin.implements(NotEqual)
# class QuantizedNotEqual(_DispatchMixin, QuantizationMixin, NotEqual):
#     """ Quantized NotEqual """
#     _builtin_torch_fn = torch.ne
#
#
# @QuantizationMixin.implements(Equal)
# class QuantizedEqual(_DispatchMixin, QuantizationMixin, Equal):
#     """ Quantized Equal """
#     _builtin_torch_fn = torch.eq


@QuantizationMixin.implements(Bmm)
class QuantizedBmm(_DispatchMixin, QuantizationMixin, Bmm):
    """Quantized Bmm"""

    __quant_init__ = QuantizationMixin.__binary__
    _builtin_torch_fn = torch.bmm


@QuantizationMixin.implements(CumSum)
class QuantizedCumSum(_DispatchMixin, QuantizationMixin, CumSum):
    """Quantized CumSum"""

    _builtin_torch_fn = torch.cumsum

    def _is_dynamo_traceable(self):
        # torch.cumsum isn't dynamo-traceable
        return False

@QuantizationMixin.implements(Reciprocal)
class QuantizedReciprocal(_DispatchMixin, QuantizationMixin, Reciprocal):
    """Quantized Reciprocal"""
    _builtin_torch_fn = torch.reciprocal


@QuantizationMixin.implements(Clamp)
class QuantizedClamp(_DispatchMixin, QuantizationMixin, Clamp):
    """Quantized Clamp"""
    _builtin_torch_fn = torch.clamp


@QuantizationMixin.implements(Clip)
class QuantizedClip(_DispatchMixin, QuantizationMixin, Clip):
    """Quantized Clip"""
    _builtin_torch_fn = torch.clip



# @QuantizationMixin.implements(MaskedFill)
# class QuantizedMaskedFill(_DispatchMixin, QuantizationMixin, MaskedFill):
#     """ Quantized MaskedFill """
#     _builtin_torch_fn = torch.Tensor.masked_fill_
#
#
@QuantizationMixin.implements(Mean)
class QuantizedMean(_DispatchMixin, QuantizationMixin, Mean):
    """Quantized Mean"""
    
    _builtin_torch_fn = torch.mean


@QuantizationMixin.implements(Var)
class QuantizedVar(_DispatchMixin, QuantizationMixin, Var):
    """Quantized Var"""

    _builtin_torch_fn = torch.var

    def __quant_init__(self):
        super().__quant_init__()
        # torch.var 没有标准 ONNX op 映射，JSON config 无法自动创建 input quantizer。
        # 此处保持 None 占位（避免 sim 初始化 trace 阶段因未 calibrate 而报错）；
        # 需由 apply_mixed_precision_bitwidth 在 sim 创建后按需创建真实 quantizer。
        if not self.input_quantizers or len(self.input_quantizers) == 0:
            self.input_quantizers = nn.ModuleList([None])


# @QuantizationMixin.implements(Sum)
# class QuantizedSum(_DispatchMixin, QuantizationMixin, Sum):
#     """ Quantized Sum """
#     _builtin_torch_fn = torch.sum
#
#
# @QuantizationMixin.implements(Prod)
# class QuantizedProd(_DispatchMixin, QuantizationMixin, Prod):
#     """ Quantized Prod """
#     _builtin_torch_fn = torch.prod
#
#
@QuantizationMixin.implements(Log)
class QuantizedLog(_DispatchMixin, QuantizationMixin, Log):
    """Quantized Log"""

    _builtin_torch_fn = torch.log


@QuantizationMixin.implements(Abs)
class QuantizedAbs(_DispatchMixin, QuantizationMixin, Abs):
    """Quantized Abs"""

    _builtin_torch_fn = torch.abs


@QuantizationMixin.implements(Neg)
class QuantizedNeg(_DispatchMixin, QuantizationMixin, Neg):
    """Quantized Neg"""

    _builtin_torch_fn = torch.neg


#
#
# @QuantizationMixin.implements(Argmin)
# class QuantizedArgmin(_DispatchMixin, QuantizationMixin, Argmin):
#     """ Quantized Argmin """
#     _builtin_torch_fn = torch.argmin
#
#
# @QuantizationMixin.implements(Argmax)
# class QuantizedArgmax(_DispatchMixin, QuantizationMixin, Argmax):
#     """ Quantized Argmax """
#     _builtin_torch_fn = torch.argmax
#
#
# @QuantizationMixin.implements(ElementwiseCeil)
# class QuantizedElementwiseCeil(_DispatchMixin, QuantizationMixin, ElementwiseCeil):
#     """ Quantized ElementwiseCeil """
#     _builtin_torch_fn = torch.ceil
#
#
# @QuantizationMixin.implements(ElementwiseFloor)
# class QuantizedElementwiseFloor(_DispatchMixin, QuantizationMixin, ElementwiseFloor):
#     """ Quantized ElementwiseFloor """
#     _builtin_torch_fn = torch.floor
#
#
# @QuantizationMixin.implements(Asin)
# class QuantizedAsin(_DispatchMixin, QuantizationMixin, Asin):
#     """ Quantized Asin """
#     _builtin_torch_fn = torch.asin
#
#
# @QuantizationMixin.implements(Atan)
# class QuantizedAtan(_DispatchMixin, QuantizationMixin, Atan):
#     """ Quantized Atan """
#     _builtin_torch_fn = torch.atan
#
#
# @QuantizationMixin.implements(Round)
# class QuantizedRound(_DispatchMixin, QuantizationMixin, Round):
#     """ Quantized Round """
#     _builtin_torch_fn = torch.round
#
#
# @QuantizationMixin.implements(Gather)
# class QuantizedGather(_DispatchMixin, QuantizationMixin, Gather):
#     """ Quantized Gather """
#     _builtin_torch_fn = torch.gather
#
#
# @QuantizationMixin.implements(LogicalOr)
# class QuantizedLogicalOr(_DispatchMixin, QuantizationMixin, LogicalOr):
#     """ Quantized LogicalOr """
#     _builtin_torch_fn = torch.logical_or
#
#
# @QuantizationMixin.implements(LogicalAnd)
# class QuantizedLogicalAnd(_DispatchMixin, QuantizationMixin, LogicalAnd):
#     """ Quantized LogicalAnd """
#     _builtin_torch_fn = torch.logical_and
#
#
# @QuantizationMixin.implements(LogicalNot)
# class QuantizedLogicalNot(_DispatchMixin, QuantizationMixin, LogicalNot):
#     """ Quantized LogicalNot """
#     _builtin_torch_fn = torch.logical_not
#
#
# @QuantizationMixin.implements(Split)
# class QuantizedSplit(_DispatchMixin, QuantizationMixin, Split):
#     """ Quantized Split """
#     _builtin_torch_fn = torch.split
#
#
# @QuantizationMixin.implements(Permute)
# class QuantizedPermute(_DispatchMixin, QuantizationMixin, Permute):
#     """ Quantized Permute """
#     _builtin_torch_fn = torch.permute
#
#
# @QuantizationMixin.implements(Remainder)
# class QuantizedRemainder(_DispatchMixin, QuantizationMixin, Remainder):
#     """ Quantized Remainder """
#     _builtin_torch_fn = torch.remainder
#
#
# @QuantizationMixin.implements(IndexSelect)
# class QuantizedIndexSelect(_DispatchMixin, QuantizationMixin, IndexSelect):
#     """ Quantized IndexSelect """
#     _builtin_torch_fn = torch.index_select
#
#
# @QuantizationMixin.implements(Fmod)
# class QuantizedFmod(_DispatchMixin, QuantizationMixin, Fmod):
#     """ Quantized Fmod """
#     _builtin_torch_fn = torch.fmod
#
#
# @QuantizationMixin.implements(NonZero)
# class QuantizedNonZero(_DispatchMixin, QuantizationMixin, NonZero):
#     """ Quantized NonZero """
#     _builtin_torch_fn = torch.nonzero
#
#
# @QuantizationMixin.implements(TopK)
# class QuantizedTopK(_DispatchMixin, QuantizationMixin, TopK):
#     """ Quantized TopK """
#     _builtin_torch_fn = torch.topk
#
#
# @QuantizationMixin.implements(Shape)
# class QuantizedShape(_DispatchMixin, QuantizationMixin, Shape):
#     """ Quantized Shape """
#     _builtin_torch_fn = torch.Tensor.size
#
#
# @QuantizationMixin.implements(Tile)
# class QuantizedTile(_DispatchMixin, QuantizationMixin, Tile):
#     """ Quantized Tile """
#     _builtin_torch_fn = torch.tile

@QuantizationMixin.implements(ElementwiseUnarySign)
class QuantizedElementwiseUnarySign(_DispatchMixin, QuantizationMixin, ElementwiseUnarySign):
    """Quantized ElementwiseUnarySign (Sign function)"""
    
    _builtin_torch_fn = torch.sign


@QuantizationMixin.implements(Baddbmm)
class QuantizedBaddbmm(_DispatchMixin, QuantizationMixin, Baddbmm):
    """Quantized Baddbmm"""

    __quant_init__ = QuantizationMixin.__ternary__
    _builtin_torch_fn = torch.baddbmm


@QuantizationMixin.implements(Addmm)
class QuantizedAddmm(_DispatchMixin, QuantizationMixin, Addmm):
    """Quantized Addmm"""

    __quant_init__ = QuantizationMixin.__ternary__
    _builtin_torch_fn = torch.addmm


@QuantizationMixin.implements(RmsNorm)
class QuantizedRmsNorm(QuantizationMixin, RmsNorm):
    """Custom module for RmsNorm"""

    # pylint: disable=arguments-differ
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RmsNorm
        """
        if self.input_quantizers[0]:
            x = self.input_quantizers[0](x)

        with self._patch_quantized_parameters():
            out = super().forward(x)

        if self.output_quantizers[0]:
            out = self.output_quantizers[0](out)

        return out


@QuantizationMixin.implements(HadamardRotation)
class QuantizedHadamardRotation(QuantizationMixin, HadamardRotation):
    """Custom module for HadamardRotation"""

    # pylint: disable=arguments-differ
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for HadamardRotation
        """
        # Quantize input tensors
        if self.input_quantizers[0]:
            x = self.input_quantizers[0](x)

        # Run forward with quantized inputs and parameters
        with self._patch_quantized_parameters():
            ret = super().forward(x)

        # Quantize output tensors
        if self.output_quantizers[0]:
            ret = self.output_quantizers[0](ret)

        return ret


@QuantizationMixin.implements(Square)
class QuantizedSquare(_DispatchMixin, QuantizationMixin, Square):
    """Quantized Square"""

    _builtin_torch_fn = torch.square
#
#
# @QuantizationMixin.implements(Select)
# class QuantizedSelect(_DispatchMixin, QuantizationMixin, Select):
#     """ Quantized Select """
#     _builtin_torch_fn = torch.select
#
#
#
# # modules for functional operations defined under torch.nn.functional package
# @QuantizationMixin.implements(Interpolate)
# class QuantizedInterpolate(_DispatchMixin, QuantizationMixin, Interpolate):
#     """ Quantized Interpolate """
#     _builtin_torch_fn = torch.nn.functional.interpolate
#
#
# @QuantizationMixin.implements(MaxPool2d)
# class QuantizedMaxPool2d(_DispatchMixin, QuantizationMixin, MaxPool2d):
#     """ Quantized MaxPool2d """
#     _builtin_torch_fn = torch.nn.functional.max_pool2d
#
#
# @QuantizationMixin.implements(AdaptiveAvgPool2d)
# class QuantizedAdaptiveAvgPool2d(_DispatchMixin, QuantizationMixin, AdaptiveAvgPool2d):
#     """ Quantized AdaptiveAvgPool2d """
#     _builtin_torch_fn = torch.nn.functional.adaptive_avg_pool2d
#
#
@QuantizationMixin.implements(BatchNorm)
class QuantizedBatchNorm(_DispatchMixin, QuantizationMixin, BatchNorm):
    """Quantized BatchNorm"""

    _builtin_torch_fn = torch.nn.functional.batch_norm

    def __quant_init__(self):
        super().__quant_init__()
        # pylint: disable=attribute-defined-outside-init
        self.input_quantizers = nn.ModuleList([None, None, None, None, None])

    def _builtin_torch_fn_helper(self, fn: Callable[..., Tensor]):
        # pylint: disable=redefined-builtin
        def batch_norm_wrapper(
            input: Tensor,
            running_mean: Optional[Tensor],
            running_var: Optional[Tensor],
            weight: Optional[Tensor] = None,
            bias: Optional[Tensor] = None,
            training: bool = False,
            momentum: float = 0.1,
            eps: float = 1e-5,
        ) -> Tensor:
            if training:
                if (
                    self.input_quantizers[1] is not None
                    or self.input_quantizers[2] is not None
                ):
                    raise RuntimeError(
                        f"{self.__class__} doesn't support quantizing running_mean or running_var in training mode"
                    )

            input = _quantize_dequantize_if_applicable(input, self.input_quantizers[0])
            running_mean = _quantize_dequantize_if_applicable(
                running_mean, self.input_quantizers[1]
            )
            running_var = _quantize_dequantize_if_applicable(
                running_var, self.input_quantizers[2]
            )
            weight = _quantize_dequantize_if_applicable(
                weight, self.input_quantizers[3]
            )
            bias = _quantize_dequantize_if_applicable(bias, self.input_quantizers[4])

            # PyTorch doesn't support gradient calculation of running_mean/var
            output = fn(
                input,
                running_mean.detach(),
                running_var.detach(),
                weight,
                bias,
                training,
                momentum,
                eps,
            )

            return _quantize_dequantize_if_applicable(output, self.output_quantizers[0])

        return batch_norm_wrapper

    def _custom_kernel_helper(self, fn: Callable[..., Tensor]):
        # pylint: disable=redefined-builtin
        def batch_norm_wrapper(
            input: Tensor,
            running_mean: Optional[Tensor],
            running_var: Optional[Tensor],
            weight: Optional[Tensor] = None,
            bias: Optional[Tensor] = None,
            training: bool = False,
            momentum: float = 0.1,
            eps: float = 1e-5,
        ) -> Tensor:
            if training:
                if (
                    self.input_quantizers[1] is not None
                    or self.input_quantizers[2] is not None
                ):
                    raise RuntimeError(
                        f"{self.__class__} doesn't support quantizing running_mean or running_var in training mode"
                    )

            input = _quantize_if_applicable(input, self.input_quantizers[0])
            running_mean = _quantize_if_applicable(
                running_mean, self.input_quantizers[1]
            )
            running_var = _quantize_if_applicable(running_var, self.input_quantizers[2])
            weight = _quantize_if_applicable(weight, self.input_quantizers[3])
            bias = _quantize_if_applicable(bias, self.input_quantizers[4])

            # PyTorch doesn't support gradient calculation of running_mean/var
            output = fn(
                input,
                running_mean.detach(),
                running_var.detach(),
                weight,
                bias,
                training,
                momentum,
                eps,
            )
            return _quantize_if_applicable(output, self.output_quantizers[0])

        return batch_norm_wrapper


@QuantizationMixin.implements(GroupNorm)
class QuantizedGroupNorm(_DispatchMixin, QuantizationMixin, GroupNorm):
    """Quantized GroupNorm"""

    _builtin_torch_fn = F.group_norm

    def _is_dynamo_traceable(self):
        # F.group_norm isn't dynamo-traceable
        return False


@QuantizationMixin.implements(Normalize)
class QuantizedNormalize(_DispatchMixin, QuantizationMixin, Normalize):
    """Quantized Normalize"""

    _builtin_torch_fn = torch.nn.functional.normalize


@QuantizationMixin.implements(NullRequant)
class QuantizedNullRequant(QuantizationMixin, NullRequant):
    """Quantized module for NullRequant"""

    # pylint: disable=arguments-differ
    def forward(self, x: torch.Tensor, shape: list) -> torch.Tensor:
        """
        Forward pass for NullRequant
        """
        if self.input_quantizers[0]:
            x = self.input_quantizers[0](x)

        with self._patch_quantized_parameters():
            out = super().forward(x, shape)

        if self.output_quantizers[0]:
            out = self.output_quantizers[0](out)

        return out


@QuantizationMixin.implements(Pad)
class QuantizedPad(_DispatchMixin, QuantizationMixin, Pad):
    """ Quantized Pad """
    _builtin_torch_fn = torch.nn.functional.pad


@QuantizationMixin.implements(GridSample)
class QuantizedGridSample(_DispatchMixin, QuantizationMixin, GridSample):
    """Quantized GridSample"""

    __quant_init__ = QuantizationMixin.__binary__
    _builtin_torch_fn = torch.nn.functional.grid_sample


@QuantizationMixin.implements(Snake2d)
class QuantizedSnake2d(QuantizationMixin, Snake2d):
    """Quantized Snake2d activation"""

    def __quant_init__(self):
        super().__quant_init__()
        # input_quantizers[0]: None 占位，对应输入 x（可由 apply_mixed_precision_bitwidth 创建）
        # input_quantizers[1]: None，不量化 alpha（alpha 由父 Module 的 param_quantizer 处理）
        self.input_quantizers = nn.ModuleList([None, None])

    def forward(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        input_qtzr = self.input_quantizers[0] if self.input_quantizers else None
        output_qtzr = self.output_quantizers[0] if self.output_quantizers else None

        if input_qtzr:
            x = input_qtzr(x)

        out = super().forward(x, alpha)

        if output_qtzr:
            out = output_qtzr(out)

        return out


# @QuantizationMixin.implements(DynamicConv2d)
# class QuantizedDynamicConv2d(QuantizationMixin, DynamicConv2d):
#     """ Quantized DynamicConv2d """
#
#
@QuantizationMixin.implements(Pow)
class QuantizedPow(QuantizationMixin, Pow):
    """Quantized Pow"""

    __quant_init__ = QuantizationMixin.__binary__
    
    def forward(self, x, exponent):  # pylint: disable=arguments-differ
        """Quantized forward for Pow operation"""
        # Get quantizers
        input_qtzr_x = self.input_quantizers[0] if self.input_quantizers else None
        input_qtzr_exp = self.input_quantizers[1] if len(self.input_quantizers) > 1 else None
        output_qtzr = self.output_quantizers[0] if self.output_quantizers else None
        
        # Quantize inputs
        if input_qtzr_x:
            x = input_qtzr_x(x)
        
        # Only quantize exponent if it's a tensor
        if input_qtzr_exp and isinstance(exponent, torch.Tensor):
            exponent = input_qtzr_exp(exponent)
        
        # Compute
        out = torch.pow(x, exponent)
        
        # Quantize output
        if output_qtzr:
            out = output_qtzr(out)
        
        return out


@QuantizationMixin.implements(CustomSiLU)
class QuantizedCustomSiLU(QuantizationMixin, CustomSiLU):
    """Quantized CustomSiLU"""

    __quant_init__ = QuantizationMixin.__unary__

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-differ
        (input_qtzr,) = self.input_quantizers
        (output_qtzr,) = self.output_quantizers

        if input_qtzr:
            x = input_qtzr(x)

        out = super().forward(x)

        if output_qtzr:
            out = output_qtzr(out)

        return out


if _OptionalQuantGRU is not None:
    from aimet_torch.fixed_point import ExecutionMode, get_quant_execution_mode
    from aimet_torch.fixed_point.errors import QuantGRUFlagLockedError
    from aimet_torch.fixed_point.quantgru_adapter import (
        EXPECTED_ADAPTER_MAJOR,
        MIN_SUPPORTED_ADAPTER_MINOR,
        aimet_capabilities as _quantgru_aimet_capabilities,
        aimet_configure as _quantgru_aimet_configure,
        check_adapter_version,
        forward_quantized as _quantgru_forward_quantized,
        get_io_quant_meta as _quantgru_get_io_quant_meta,
        resolve_quantgru_mode_str as _resolve_quantgru_mode_str,
    )

    @QuantizationMixin.implements(_OptionalQuantGRU)
    class QuantizedQuantGRU(_DispatchMixin, QuantizationMixin, _OptionalQuantGRU):
        """Black-box AIMET wrapper for QuantGRU.

        Routes forward by ExecutionMode; never re-implements GRU integer math.
        INT16 modes use ``forward_quantized`` for bit-exact boundaries.
        """

        _builtin_torch_fn = None
        _LOCKED_FLAGS = ("use_quantization", "calibrating", "export_mode", "export_format")
        EXPECTED_ADAPTER_MAJOR = EXPECTED_ADAPTER_MAJOR
        MIN_SUPPORTED_ADAPTER_MINOR = MIN_SUPPORTED_ADAPTER_MINOR

        def __quant_init__(self):
            super().__quant_init__()
            object.__setattr__(self, "_aimet_lock", False)
            check_adapter_version(self)
            # QuantGRU owns IO quantization semantics; AIMET boundary quantizers stay None.
            self.input_quantizers = nn.ModuleList([None, None])
            self.output_quantizers = nn.ModuleList([None, None])
            object.__setattr__(self, "_aimet_lock", True)

        def __setattr__(self, name, value):
            if name in self._LOCKED_FLAGS:
                try:
                    locked = object.__getattribute__(self, "_aimet_lock")
                except AttributeError:
                    locked = False
                if locked:
                    raise QuantGRUFlagLockedError(
                        f"`{name}` is managed by AIMET sim and cannot be set directly. "
                        "Use sim.set_execution_mode() / sim.compute_encodings() instead."
                    )
            super().__setattr__(name, value)

        @contextmanager
        def _aimet_unlock_ctx(self):
            prev = object.__getattribute__(self, "_aimet_lock")
            object.__setattr__(self, "_aimet_lock", False)
            try:
                yield
            finally:
                object.__setattr__(self, "_aimet_lock", prev)

        def _aimet_compute_encodings_enter(self):
            with self._aimet_unlock_ctx():
                _quantgru_aimet_configure(self, "calibrating")

        def _aimet_compute_encodings_exit(self):
            with self._aimet_unlock_ctx():
                if self.quant_ranges is not None or (
                    hasattr(self, "hist_collectors")
                    and self.hist_collectors is not None
                    and self.hist_collectors.is_valid()
                ):
                    try:
                        self.finalize_calibration(verbose=False)
                    except Exception:
                        pass
                # plan §0.2.1: FIXED_SCALE_QDQ 在 adapter 层别名为 fp32_qdq。
                _quantgru_aimet_configure(
                    self, _resolve_quantgru_mode_str(get_quant_execution_mode())
                )

        def aimet_configure(self, mode: str) -> None:
            _quantgru_aimet_configure(self, mode)

        def aimet_capabilities(self):
            return _quantgru_aimet_capabilities(self)

        def get_io_quant_meta(self):
            return _quantgru_get_io_quant_meta(self)

        def forward_quantized(self, input: torch.Tensor, hx=None):
            with self._aimet_unlock_ctx():
                return _quantgru_forward_quantized(self, input, hx)

        # plan §2.5: hidden 端 hx/h_n 必须共享同一 EncodingBase 引用。
        # sim builder (`_V2LazyQuantizeWrapper.realize`) 默认按 propagate 规则
        # 给 input_quantizers[1] / output_quantizers[1] 注入两个独立实例；
        # 长序列 + BiGRU 双向会累积漂移误差。本方法 lazy 在第一次 forward 前
        # 把 input_quantizers[1] 替换为 output_quantizers[1] 的同一引用。
        def _ensure_hidden_quantizer_shared(self) -> None:
            iq_list = self.input_quantizers
            oq_list = self.output_quantizers
            if (
                iq_list is not None
                and oq_list is not None
                and len(iq_list) >= 2
                and len(oq_list) >= 2
                and oq_list[1] is not None
                and iq_list[1] is not oq_list[1]
            ):
                with self._aimet_unlock_ctx():
                    iq_list[1] = oq_list[1]

        # Boundary helper for FP32_QDQ / FP16_QDQ / 校准期 (plan §3.1.1 D)
        # _DispatchMixin.forward 在 _builtin_torch_fn=None 时会以 fn=None 调用本 helper；
        # QuantGRU 没有原子 builtin，helper 直接用 _OptionalQuantGRU.forward 走浮点分支。
        def _builtin_torch_fn_helper(self, fn):
            del fn

            def gru_with_boundary_qdq(input, hx=None):
                in_qtzr = (
                    self.input_quantizers[0] if self.input_quantizers else None
                )
                hx_qtzr = (
                    self.input_quantizers[1]
                    if self.input_quantizers and len(self.input_quantizers) > 1
                    else None
                )
                out_qtzr = (
                    self.output_quantizers[0] if self.output_quantizers else None
                )
                hn_qtzr = (
                    self.output_quantizers[1]
                    if self.output_quantizers and len(self.output_quantizers) > 1
                    else None
                )

                x_q = _quantize_dequantize_if_applicable(input, in_qtzr)
                # contract §1.1: QuantGRU forward 入口必须 fp32 (FP16_QDQ 下 quantizer 可能返回 fp16)
                if isinstance(x_q, Tensor) and x_q.dtype != torch.float32:
                    x_q = x_q.to(torch.float32)

                h_q = None
                if hx is not None:
                    h_q = _quantize_dequantize_if_applicable(hx, hx_qtzr)
                    if isinstance(h_q, Tensor) and h_q.dtype != torch.float32:
                        h_q = h_q.to(torch.float32)

                out, h_n = _OptionalQuantGRU.forward(self, x_q, h_q)

                out = _quantize_dequantize_if_applicable(out, out_qtzr)
                h_n = _quantize_dequantize_if_applicable(h_n, hn_qtzr)
                return out, h_n

            return gru_with_boundary_qdq

        def forward(self, input: torch.Tensor, hx=None):
            mode = get_quant_execution_mode()
            # ExecutionMode 当前枚举: FP32_QDQ / FP16_QDQ / FIXED_SCALE_QDQ /
            # INT16_FIXED_EVAL / INT16_FIXED_QAT_SIM (无 "纯 FP32"；FP32 透传通过
            # 全 None boundary quantizer 自然实现)。
            if mode in (ExecutionMode.INT16_FIXED_EVAL, ExecutionMode.INT16_FIXED_QAT_SIM):
                # INT16: dispatch_quantgru_blackbox 直出 bit-exact，不经 boundary helper
                return _DispatchMixin.forward(self, input, hx)

            # FP32_QDQ / FP16_QDQ / FIXED_SCALE_QDQ / 校准期：
            # plan §2.5: 在走 boundary helper 之前确保 hx/h_n 端 quantizer 共享同一 EncodingBase。
            self._ensure_hidden_quantizer_shared()

            # 校准期内勿按 ExecutionMode 重配 flag（compute_encodings hook 已切到 calibrating）
            if not self.calibrating:
                with self._aimet_unlock_ctx():
                    _quantgru_aimet_configure(self, _resolve_quantgru_mode_str(mode))
            # 通过 _DispatchMixin.forward 触发 _builtin_torch_fn_helper -> boundary Q/DQ
            return _DispatchMixin.forward(self, input, hx)


# @QuantizationMixin.implements(StridedSlice)
# class QuantizedStridedSlice(QuantizationMixin, StridedSlice):
#     """ Quantized StridedSlice """
#
#
# @QuantizationMixin.implements(ChannelShuffle)
# class QuantizedChannelShuffle(QuantizationMixin, ChannelShuffle):
#     """ Quantized ChannelShuffle """
#
#
# @QuantizationMixin.implements(Cast)
# class QuantizedCast(QuantizationMixin, Cast):
#     """ Quantized Cast """
#
#
# @QuantizationMixin.implements(CustomGather)
# class QuantizedCustomGather(QuantizationMixin, CustomGather):
#     """ Quantized CustomGather """
#
#
# @QuantizationMixin.implements(DepthToSpaceCRDMode)
# class QuantizedDepthToSpaceCRDMode(QuantizationMixin, DepthToSpaceCRDMode):
#     """ Quantized DepthToSpaceCRDMode """
#
#
# @QuantizationMixin.implements(DepthToSpaceDCRMode)
# class QuantizedDepthToSpaceDCRMode(QuantizationMixin, DepthToSpaceDCRMode):
#     """ Quantized DepthToSpaceDCRMode """
#
#
# @QuantizationMixin.implements(CustomSparseConv3DLayer)
# class QuantizedCustomSparseConv3DLayer(QuantizationMixin, CustomSparseConv3DLayer):
#     """ Quantized CustomSparseConv3DLayer """
#
#
# @QuantizationMixin.implements(SparseTensorWrapper)
# class QuantizedSparseTensorWrapper(QuantizationMixin, SparseTensorWrapper):
#     """ Quantized SparseTensorWrapper """
#
#
# @QuantizationMixin.implements(ScatterDense)
# class QuantizedScatterDense(QuantizationMixin, ScatterDense):
#     """ Quantized ScatterDense """
#
#
# @QuantizationMixin.implements(ScatterND)
# class QuantizedScatterND(QuantizationMixin, ScatterND):
#     """ Quantized ScatterND """
#
#
# @QuantizationMixin.implements(RoiAlign)
# class QuantizedRoiAlign(QuantizationMixin, RoiAlign):
#     """ Quantized RoiAlign """
#
#
# @QuantizationMixin.implements(NonMaxSuppression)
# class QuantizedNonMaxSuppression(QuantizationMixin, NonMaxSuppression):
#     """ Quantized NonMaxSuppression """
#
#
# @QuantizationMixin.implements(GatherNd)
# class QuantizedGatherNd(QuantizationMixin, GatherNd):
#     """ Quantized GatherNd """
#
#
# @QuantizationMixin.implements(ScatterElements)
# class QuantizedScatterElements(QuantizationMixin, ScatterElements):
#     """ Quantized ScatterElements """
#
#
# @QuantizationMixin.implements(OneHot)
# class QuantizedOneHot(QuantizationMixin, OneHot):
#     """ Quantized OneHot """
#
#
# @QuantizationMixin.implements(Expand)
# class QuantizedExpand(QuantizationMixin, Expand):
#     """ Quantized Expand """
#
#
# @QuantizationMixin.implements(DynamicLinear)
# class QuantizedDynamicLinear(QuantizationMixin, DynamicLinear):
#     """ Quantized DynamicLinear """


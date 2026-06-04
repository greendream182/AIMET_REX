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
    round_shift,
    saturate_int32,
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
    # ``requantize_int`` saturates the *centered* accumulator (target zp = 0).
    # Do not pass the output quant grid ``[qmin, qmax]`` here: for asymmetric
    # quantizers (e.g. uint8 0..255 with zp>0) that would clip negative centered
    # values to 0 and break Add/Concat scale alignment on residual branches.
    centered_qmin = int(output_encoding.qmin) - output_zp.to(torch.int64)
    centered_qmax = int(output_encoding.qmax) - output_zp.to(torch.int64)
    aligned = requantize_int(
        _center_tensor(tensor),
        multiplier.to(device=tensor.int_repr.device),
        rshift.to(device=tensor.int_repr.device),
        torch.zeros_like(output_zp, dtype=torch.int32),
        int(centered_qmin.min().item()),
        int(centered_qmax.max().item()),
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
    """Reference INT16 clamp: dequant → float clamp → output requant."""

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
        value = _dequant_float(inputs[0])
        min_v = extra.get("min")
        max_v = extra.get("max")
        if min_v is not None:
            value = torch.clamp(value, min=float(min_v))
        if max_v is not None:
            value = torch.clamp(value, max=float(max_v))
        return _quantize_from_float(value, output_encoding)


@register_fixed_kernel(custom.Clamp)
class FunctionalClampInt16Kernel(ClampInt16Kernel):
    """``torch.clamp`` / ``F.clamp`` INT16 kernel."""

    module_type = custom.Clamp


def _dequant_float(tensor: Int16QuantizedTensor) -> torch.Tensor:
    scale = tensor.scale.to(device=tensor.int_repr.device, dtype=torch.float32)
    while scale.ndim < tensor.int_repr.ndim:
        scale = scale.unsqueeze(-1)
    return tensor.centered_int32().to(torch.float32) * scale


def sign_int16_from_float_input(
    float_input: torch.Tensor,
    output_encoding: OutputEncoding,
) -> Int16QuantizedTensor:
    """``sign(float)`` then output requant — matches float_native / QDQ sign_input_bypass."""

    return _quantize_from_float(torch.sign(float_input), output_encoding)


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


@register_fixed_kernel(custom.ElementwiseUnarySign)
class SignInt16Kernel:
    """INT16 sign on dequantized float (then output requant).

    Sign must not run on pre-quantized centered integers: small STFT/complex
    values collapse to zero on the input grid and flip signs (MRNN pc1).
    """

    module_type = custom.ElementwiseUnarySign

    def __call__(
        self,
        inputs: List[Int16QuantizedTensor],
        params: Dict[str, Any],
        output_encoding: OutputEncoding,
        extra: Dict[str, Any],
    ) -> Int16QuantizedTensor:
        del params, extra
        if len(inputs) != 1:
            raise ValueError(f"Sign expects 1 input; got {len(inputs)}.")
        return sign_int16_from_float_input(_dequant_float(inputs[0]), output_encoding)


def _divide_rounding_mode() -> RoundingMode:
    return RoundingMode.HALF_UP if hw_ref_mode_enabled() else RoundingMode.HALF_TO_EVEN


def _safe_divide_denominator(
    den_centered: torch.Tensor,
    den_scale: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    """Clamp near-zero divisors (matches float ``Divide`` reference)."""

    scale_f = float(den_scale.detach().reshape(-1)[0].item())
    min_abs = max(1, int(round(eps / max(scale_f, 1e-30))))
    min_abs_t = torch.tensor(min_abs, device=den_centered.device, dtype=torch.int32)
    abs_d = den_centered.abs()
    sign_d = torch.sign(den_centered.to(torch.float32)).to(torch.int32)
    sign_d = torch.where(sign_d == 0, torch.ones_like(sign_d), sign_d)
    return sign_d * torch.maximum(abs_d, min_abs_t)


def _integer_div_round(
    num: torch.Tensor,
    den: torch.Tensor,
    rounding_mode: RoundingMode,
) -> torch.Tensor:
    num_i64 = num.to(torch.int64)
    den_i64 = den.to(torch.int64)
    if rounding_mode == RoundingMode.TRUNCATE:
        return num_i64 // den_i64
    half = den_i64.abs() // 2
    bias = torch.where(num_i64 >= 0, half, -half)
    return (num_i64 + bias) // den_i64


@register_fixed_kernel(custom.Divide)
class DivideInt16Kernel:
    """INT16 Divide: scale numerator, integer division, output requant."""

    module_type = custom.Divide

    def __call__(
        self,
        inputs: List[Int16QuantizedTensor],
        params: Dict[str, Any],
        output_encoding: OutputEncoding,
        extra: Dict[str, Any],
    ) -> Int16QuantizedTensor:
        del params
        if len(inputs) != 2:
            raise ValueError(f"Divide expects 2 inputs; got {len(inputs)}.")
        eps = float(extra.get("eps", 1e-12))
        num_c = _center_tensor(inputs[0])
        den_c = _center_tensor(inputs[1])
        s_num = inputs[0].scale.to(device=num_c.device, dtype=torch.float32)
        s_den = inputs[1].scale.to(device=num_c.device, dtype=torch.float32)
        s_out = output_encoding.scale.to(device=num_c.device, dtype=torch.float32)
        real_m = (s_num / s_den / s_out).detach()
        multiplier, rshift = quantize_multiplier(real_m)
        rounding = _divide_rounding_mode()
        prod = num_c.to(torch.int64) * multiplier.to(device=num_c.device, dtype=torch.int64)
        if hw_ref_mode_enabled():
            prod = saturate_int32(prod).to(torch.int64)
        num_scaled = round_shift(prod, rshift, rounding)
        den_safe = _safe_divide_denominator(den_c, s_den, eps=eps)
        quotient = _integer_div_round(num_scaled, den_safe, rounding)
        acc = int32_add_sat(
            quotient.to(torch.int32),
            output_encoding.zero_point.to(device=num_c.device, dtype=torch.int32),
        )
        return _wrap_like_output(
            saturate_sim_tensor(acc, output_encoding.qmin, output_encoding.qmax),
            output_encoding,
        )


@register_fixed_kernel(custom.Clip)
class FunctionalClipInt16Kernel(ClampInt16Kernel):
    """``torch.clip`` INT16 kernel."""

    module_type = custom.Clip

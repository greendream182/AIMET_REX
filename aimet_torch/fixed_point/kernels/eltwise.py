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

from typing import Any, Dict, List, Tuple

import torch
from torch import nn

from aimet_torch._base.nn.modules import custom
from aimet_torch.fixed_point.encoding import OutputEncoding
from aimet_torch.fixed_point.kernels._contracts import (
    require_int32_saturated_accumulator,
)
from aimet_torch.fixed_point.offline.multiplier import quantize_multiplier
from aimet_torch.fixed_point.registry import register_fixed_kernel
from aimet_torch.fixed_point.rounding import RoundingMode
from aimet_torch.fixed_point.requantize import (
    SIM_TENSOR_DTYPE,
    _forbid_float_fallback_in_eval,
    hw_ref_mode_enabled,
    int16_fixed_eval_mode,
    int32_add_sat,
    int32_mul_sat,
    int32_sub_sat,
    requantize_int,
    round_shift,
    saturate_int32,
    saturate_mac_accumulator,
    saturate_sim_tensor,
    sign_float_ref_enabled,
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
    *,
    op_name: str = "eltwise",
) -> Int16QuantizedTensor:
    if output_encoding.multiplier is None or output_encoding.rshift is None:
        raise ValueError("output_encoding must provide multiplier and rshift.")
    acc_sat = saturate_mac_accumulator(acc)
    require_int32_saturated_accumulator(acc_sat, op_name=op_name)
    return _wrap_like_output(
        requantize_int(
            acc_sat,
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


@register_fixed_kernel(custom.Abs)
class AbsInt16Kernel:
    """Reference INT16 Abs kernel — integer-abs path per spec 04_03 §4.3.5.

    Spec contract::

        x' = |q_x − Z_x|                       # int32 centered absolute value
        y_q = sat((x' · M) ≫ rshift) + Z_y     # SAME_GRID_OR_REQUANT

    Same-grid (no rescale) reduces to byte-stream identity on the centered
    representation; cross-grid hits ``_requantize`` (same hot path as ReLU /
    Clamp). The previous PWL implementation (16-segment fit of ``|x|``) has
    been retired in favour of the integer-abs path so the kernel matches
    spec and the manifest can advertise ``SAME_GRID_OR_REQUANT`` again.
    """

    module_type = custom.Abs

    def __call__(
        self,
        inputs: List[Int16QuantizedTensor],
        params: Dict[str, Any],
        output_encoding: OutputEncoding,
        extra: Dict[str, Any],
    ) -> Int16QuantizedTensor:
        del params, extra
        if len(inputs) != 1:
            raise ValueError(f"Abs expects 1 input; got {len(inputs)}.")
        centered = _center_tensor(inputs[0])
        abs_val = torch.abs(centered)
        if output_encoding.multiplier is None or output_encoding.rshift is None:
            y = abs_val + output_encoding.zero_point.to(
                device=abs_val.device, dtype=torch.int32
            )
            return _wrap_like_output(
                saturate_sim_tensor(y, output_encoding.qmin, output_encoding.qmax),
                output_encoding,
            )
        return _requantize(abs_val, output_encoding, op_name="Abs")


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


def _encoding_grids_match(
    tensor: Int16QuantizedTensor,
    output_encoding: OutputEncoding,
) -> bool:
    device = tensor.int_repr.device
    in_scale = tensor.scale.to(device=device, dtype=torch.float32)
    out_scale = output_encoding.scale.to(device=device, dtype=torch.float32)
    in_zp = tensor.zero_point.to(device=device, dtype=torch.int32)
    out_zp = output_encoding.zero_point.to(device=device, dtype=torch.int32)
    return torch.equal(in_scale, out_scale) and torch.equal(in_zp, out_zp)


def _centered_bounds_from_extra(
    extra: Dict[str, Any],
    tensor: Int16QuantizedTensor,
    output_encoding: OutputEncoding,
) -> Tuple[int, int]:
    """Map float / input-grid ``min_int``/``max_int`` to centered bounds on the output scale."""

    device = tensor.int_repr.device
    out_scale = output_encoding.scale.to(device=device, dtype=torch.float32).reshape(-1)
    out_zp = int(output_encoding.zero_point.reshape(-1)[0].item())
    scale0 = float(out_scale[0].item())

    min_v = extra.get("min")
    max_v = extra.get("max")
    if min_v is not None:
        cmin = int(round(float(min_v) / scale0))
    elif extra.get("min_int") is not None:
        min_int = int(extra["min_int"])
        if _encoding_grids_match(tensor, output_encoding):
            cmin = min_int - out_zp
        else:
            in_scale = float(tensor.scale.reshape(-1)[0].item())
            in_zp = int(tensor.zero_point.reshape(-1)[0].item())
            cmin = int(round((min_int - in_zp) * in_scale / scale0))
    else:
        cmin = int(output_encoding.qmin) - out_zp

    if max_v is not None:
        cmax = int(round(float(max_v) / scale0))
    elif extra.get("max_int") is not None:
        max_int = int(extra["max_int"])
        if _encoding_grids_match(tensor, output_encoding):
            cmax = max_int - out_zp
        else:
            in_scale = float(tensor.scale.reshape(-1)[0].item())
            in_zp = int(tensor.zero_point.reshape(-1)[0].item())
            cmax = int(round((max_int - in_zp) * in_scale / scale0))
    else:
        cmax = int(output_encoding.qmax) - out_zp

    return cmin, cmax


def _grid_aware_floor_clamp_extra(
    extra: Dict[str, Any],
    tensor: Int16QuantizedTensor,
) -> Dict[str, Any]:
    """Floor ``extra['min']/['max']`` to ±1 LSB of the input grid (R2 fix).

    Spec ``doc/04_算子详细规格/04_03_逐元素运算类算子.md`` §4.3.4 "边界与
    保护" and §4.13.1 Clip 注释 jointly require the upstream near-zero
    guard ``Clamp(min=ε)`` to **remain effective** at INT16 precision. Yet
    when ``ε`` is below 1 LSB of the input grid (e.g. ``CLN.EPS = 1e-8``
    with a ``mean_sq`` scale of ``~1e-4``), the adapter's
    ``round(ε / S_x + zp)`` pulls ``min_int`` straight back to ``zp`` and
    the same-grid fast path of :func:`clamp_int16` becomes a no-op. The
    near-zero ``mean_sq`` then flows untouched into ``Sqrt → Divide``,
    where ``reciprocal_via_clz_lut`` saturates ``q_in = 0 → out_qmax``
    (the spec-mandated divide-by-zero guard) and the resulting ``+100×``
    payload explodes the activation (observed ``norm_max ≈ 7e30`` on
    ``cln.module_div_*``).

    The fix: when the user-supplied ``min`` is strictly positive but the
    integer-domain bound collapses to ``≤ zp`` (i.e. ``< 1`` LSB above
    zero on the centered grid), promote ``min_int`` to ``zp + 1`` and
    ``min`` to one input LSB so every consumer of ``extra`` (the
    grid-match fast path, the cross-grid fallback, and
    :func:`_centered_bounds_from_extra`) keeps a non-trivial clamp.
    Symmetric treatment for negative ``max``. ``min_int`` /
    ``max_int`` callers without an accompanying ``min`` / ``max`` float
    hint are presumed already integer-aware and are NOT auto-floored.

    fp32 / QDQ paths bypass this helper entirely: this is an
    ``INT16_FIXED_EVAL``-only correction that never alters training
    semantics.
    """

    min_v = extra.get("min")
    max_v = extra.get("max")
    if min_v is None and max_v is None:
        return extra

    in_scale = float(tensor.scale.reshape(-1)[0].item())
    in_zp = int(tensor.zero_point.reshape(-1)[0].item())
    new_extra = dict(extra)

    if min_v is not None:
        min_f = float(min_v)
        if min_f > 0.0:
            cur_min_int = new_extra.get("min_int")
            if cur_min_int is None or int(cur_min_int) <= in_zp:
                new_extra["min_int"] = in_zp + 1
                new_extra["min"] = max(min_f, in_scale)

    if max_v is not None:
        max_f = float(max_v)
        if max_f < 0.0:
            cur_max_int = new_extra.get("max_int")
            if cur_max_int is None or int(cur_max_int) >= in_zp:
                new_extra["max_int"] = in_zp - 1
                new_extra["max"] = min(max_f, -in_scale)

    return new_extra


def clamp_int16(
    tensor: Int16QuantizedTensor,
    output_encoding: OutputEncoding,
    extra: Dict[str, Any],
) -> Int16QuantizedTensor:
    """Integer-domain clamp (Hardtanh / ``torch.clamp`` / Clip).

    Spec ``doc/04_算子详细规格/04_13_特殊激活与常量算子.md`` §4.14.1
    (ReLU6/Clip) is explicit: "若 S_y/Z_y 与 S_x/Z_x 不一致，编译器应在
    前后图优化中插入或融合重定标，**不应让纯 Clip 比较逻辑隐式承担量化
    域转换**" — i.e. the Clip kernel is the wrong place to fold a second
    ``M/rshift``.

    The cross-grid path therefore does the ``S_x → S_y`` rescale exactly
    once via :func:`align_centered_int32_to_output` (whose own
    ``quantize_multiplier(S_x/S_y)`` folding stays internal to that
    helper). After ``align``, the centered payload is already on the
    output grid; the clamp bounds from :func:`_centered_bounds_from_extra`
    are also computed in the output centered domain. So the only work
    left is ``+Z_y + saturate`` to land back on the integer grid.

    Historically the kernel chained ``align → clamp → _requantize`` and
    let ``_requantize`` fold ``output_encoding.multiplier`` (typically the
    same ``S_x/S_y``) a second time, multiplying error by ``(S_x/S_y)²``
    and inflating ``lsb_max`` to ~150 on real adapter paths (the test
    suite sidestepped the bug by forcing ``multiplier=None`` in
    :func:`OutputEncoding`). This was tracked as
    ``FU-P5-CLAMP-DOUBLE-RESCALE`` in ``doc/precision_validation.md``.

    The fix is to **ignore** ``output_encoding.multiplier`` /
    ``rshift`` in the cross-grid branch — they remain in
    ``OutputEncoding`` because the adapter dispatch site at
    ``aimet_torch/v2/quantization/affine/fixed_point/adapter.py``
    populates them from ``real_m = x_scale / y_scale`` for every
    non-special-cased op (line ~1016) and the Clamp family currently
    has no opt-out branch there. Skipping the fold here is the minimal
    change that fixes the precision regression without touching adapter
    routing for the other SAME_GRID_OR_REQUANT consumers of that
    fallback.
    """

    extra = _grid_aware_floor_clamp_extra(extra, tensor)

    if _encoding_grids_match(tensor, output_encoding) and extra.get("min_int") is not None:
        min_q = int(extra["min_int"])
        max_q = int(extra.get("max_int", output_encoding.qmax))
        q = torch.clamp(tensor.int_repr, min=min_q, max=max_q)
        return _wrap_like_output(
            saturate_sim_tensor(q, output_encoding.qmin, output_encoding.qmax),
            output_encoding,
        )

    centered = align_centered_int32_to_output(tensor, output_encoding)
    cmin, cmax = _centered_bounds_from_extra(extra, tensor, output_encoding)
    clamped = torch.clamp(centered, min=cmin, max=cmax)
    zp = output_encoding.zero_point.to(device=centered.device, dtype=torch.int32)
    y = int32_add_sat(clamped, zp)
    return _wrap_like_output(
        saturate_sim_tensor(y, output_encoding.qmin, output_encoding.qmax),
        output_encoding,
    )


@register_fixed_kernel(nn.Hardtanh)
class ClampInt16Kernel:
    """Reference INT16 clamp on the quantized grid (no float intermediate)."""

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
        return clamp_int16(inputs[0], output_encoding, extra)


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


def sign_int16_centered(
    tensor: Int16QuantizedTensor,
    output_encoding: OutputEncoding,
) -> Int16QuantizedTensor:
    """Integer sign on centered input: ``+1`` / ``0`` / ``-1``, then output requant."""

    centered = tensor.centered_int32()
    signs = (centered > 0).to(torch.int32) - (centered < 0).to(torch.int32)

    if output_encoding.multiplier is None or output_encoding.rshift is None:
        zp = output_encoding.zero_point.to(device=centered.device, dtype=torch.int32)
        y = int32_add_sat(signs, zp)
        return _wrap_like_output(
            saturate_sim_tensor(y, output_encoding.qmin, output_encoding.qmax),
            output_encoding,
        )
    return _requantize(signs, output_encoding)


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
    """INT16 sign: integer compare on centered values (default in eval).

    Optional float reference (``AIMET_RX_SIGN_FLOAT_REF=1`` or
    ``extra['sign_float_ref']=True``) matches float QDQ when near-zero
    activations quantize to 0 on the input grid before sign (MRNN STFT).
    """

    module_type = custom.ElementwiseUnarySign

    def __call__(
        self,
        inputs: List[Int16QuantizedTensor],
        params: Dict[str, Any],
        output_encoding: OutputEncoding,
        extra: Dict[str, Any],
    ) -> Int16QuantizedTensor:
        del params
        if len(inputs) != 1:
            raise ValueError(f"Sign expects 1 input; got {len(inputs)}.")
        if extra.get("sign_float_ref"):
            if int16_fixed_eval_mode():
                _forbid_float_fallback_in_eval("sign_float_ref")
            return sign_int16_from_float_input(
                _dequant_float(inputs[0]), output_encoding
            )
        if sign_float_ref_enabled():
            return sign_int16_from_float_input(
                _dequant_float(inputs[0]), output_encoding
            )
        return sign_int16_centered(inputs[0], output_encoding)


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
    sign_d = (den_centered > 0).to(torch.int32) - (den_centered < 0).to(torch.int32)
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


def _legacy_integer_divide(
    inputs: List[Int16QuantizedTensor],
    output_encoding: OutputEncoding,
    *,
    eps: float,
) -> Int16QuantizedTensor:
    """Pre-spec integer-div Divide path; kept as fallback when LUT unavailable.

    Worst-case error ≈1.5 LSB (multiplier folding + integer round-half +
    cross-scale folding). See ``doc/precision_validation.md`` Divide entry.
    """

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


def _try_resolve_reciprocal_clz_lut(
    extra: Dict[str, Any],
) -> Dict[str, Any] | None:
    """Resolve a reciprocal CLZ LUT body for Divide's spec §4.3.4 path.

    Order of resolution (first hit wins):
    1. ``extra['reciprocal_clz_lut']`` user-supplied dict (already loaded)
    2. ``extra['reciprocal_clz_lut_path']`` JSON file path
    3. abc default ``reciprocal_clz_lut.json`` discovered via
       ``resolve_abc_lut_root`` (no abc tree → ``None``)

    ``None`` means "fall back to the legacy integer-div path".
    """

    raw = extra.get("reciprocal_clz_lut")
    if raw is not None and isinstance(raw, dict) and "segments" in raw:
        return raw
    path = extra.get("reciprocal_clz_lut_path")
    if path is not None:
        from aimet_torch.fixed_point.kernels.clz_lut import (
            load_clz_lut_from_json,
        )

        try:
            _func, body = load_clz_lut_from_json(path, func_name="reciprocal")
            return body
        except (FileNotFoundError, KeyError, ValueError):
            return None
    if extra.get("force_legacy_integer_divide", False):
        return None
    from aimet_torch.fixed_point.kernels.clz_lut import (
        try_load_default_reciprocal_clz_lut,
    )

    return try_load_default_reciprocal_clz_lut()


def _divide_via_reciprocal_lut(
    inputs: List[Int16QuantizedTensor],
    output_encoding: OutputEncoding,
    clz_body: Dict[str, Any],
) -> Int16QuantizedTensor:
    """spec §4.3.4 path: ``y = a * (1/b)`` via reciprocal CLZ LUT + Multiply.

    Decomposition:
      1. ``q_recip = ReciprocalCLZ(q_b)``  — at LUT output grid (e.g. abc
         default: signed int16 fmin/fmax = ±100). LUT internally handles
         |b|→0 saturation per spec "边界与保护".
      2. ``y = num * q_recip`` — folded through the standard Multiply hot
         path with ``real_m = scale_num * scale_recip / scale_out``. This
         is the **same** ``int32_mul_sat(_center, _center) → _requantize``
         path used by ``custom.Multiply``, so the precision character
         matches Multiply rather than the integer-div fallback.
    """

    from aimet_torch.fixed_point.kernels.clz_lut import (
        reciprocal_via_clz_lut,
    )

    num, den = inputs
    recip_t = reciprocal_via_clz_lut(den, clz_body)

    num_dev = num.scale.device
    s_num = num.scale.to(device=num_dev, dtype=torch.float32)
    s_recip = recip_t.scale.to(device=num_dev, dtype=torch.float32)
    s_out = output_encoding.scale.to(device=num_dev, dtype=torch.float32)
    real_m = (s_num * s_recip / s_out).detach().to(torch.float64)
    multiplier, rshift = quantize_multiplier(real_m)

    out_enc_with_m = OutputEncoding(
        scale=output_encoding.scale,
        zero_point=output_encoding.zero_point,
        qmin=output_encoding.qmin,
        qmax=output_encoding.qmax,
        multiplier=multiplier,
        rshift=rshift,
        axis=output_encoding.axis,
    )
    return MultiplyInt16Kernel()([num, recip_t], {}, out_enc_with_m, {})


@register_fixed_kernel(custom.Divide)
class DivideInt16Kernel:
    """INT16 Divide kernel.

    Two paths, gated by reciprocal-LUT availability:

    - **spec §4.3.4 path** (default when reciprocal CLZ LUT is reachable):
      ``y = num * (1/den)`` via reciprocal CLZ LUT + Multiply. Precision
      character matches Multiply (≤ ~1 LSB at LUT-domain inputs).
    - **legacy integer-div path** (fallback when no LUT, or
      ``extra['force_legacy_integer_divide']=True``): the original
      ``num_scaled // den`` integer round-half pipeline. Worst-case
      ≈ 1.5 LSB; kept for environments without ``abc_lut-shuai``.

    See ``doc/precision_validation.md`` for the precision split between
    paths and ``doc/04_算子详细规格/04_03_逐元素运算类算子.md`` §4.3.4 for
    the underlying spec path.
    """

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

        clz_body = _try_resolve_reciprocal_clz_lut(extra)
        if clz_body is not None:
            return _divide_via_reciprocal_lut(inputs, output_encoding, clz_body)
        return _legacy_integer_divide(inputs, output_encoding, eps=eps)


@register_fixed_kernel(custom.Clip)
class FunctionalClipInt16Kernel(ClampInt16Kernel):
    """``torch.clip`` INT16 kernel."""

    module_type = custom.Clip

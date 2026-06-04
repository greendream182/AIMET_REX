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
"""Reference INT16 fixed-point LUT kernels for nonlinear functions."""

from typing import Any, Dict, List

import torch
from torch import nn

from aimet_torch.fixed_point.encoding import InputEncoding, OutputEncoding
from aimet_torch.fixed_point.kernels.softmax import softmax_int16_pwl
from aimet_torch.fixed_point.offline.lut_gen import (
    align_op_quant_grid_to_lut_quant_grid,
    fold_periodic_input_to_principal_range,
)
from aimet_torch.fixed_point.metrics.accuracy import quantize_float_to_grid
from aimet_torch.fixed_point.metrics.flags import int16_eval_allow_debug_float
from aimet_torch.fixed_point.rounding import RoundingMode
from aimet_torch.fixed_point.requantize import (
    INT16_QMAX,
    INT16_QMIN,
    SIM_TENSOR_DTYPE,
    _env_truthy,
    hw_ref_mode_enabled,
    round_shift,
    saturate_int32,
    saturate_sim_tensor,
    saturate_to_range,
)
from aimet_torch._base.nn.modules import custom
from aimet_torch.fixed_point.registry import register_fixed_kernel
from aimet_torch.fixed_point.tensor import Int16QuantizedTensor


def _pwl_hw_mac_sat_enabled() -> bool:
    """PE multiply INT32 saturation (``AIMET_RX_PWL_HW_MAC_SAT=1``). Implied by HW_REF."""

    return _env_truthy("AIMET_RX_PWL_HW_MAC_SAT") or hw_ref_mode_enabled()


def _pwl_hw_ref_enabled() -> bool:
    """Full abc/PE INT32 tap-point model; see :func:`hw_ref_mode_enabled`."""

    return hw_ref_mode_enabled()


def _sat_int32_acc(x: torch.Tensor) -> torch.Tensor:
    """Clamp to INT32 range; return int64 tensor for downstream shift/add ops."""

    return saturate_int32(x).to(torch.int64)


def _arith_right_shift_half_up(x: torch.Tensor, rshift: torch.Tensor) -> torch.Tensor:
    """Per-element arithmetic right shift with half-up bias (``lut_int_general``)."""

    if x.dtype != torch.int64:
        raise TypeError(f"x must be torch.int64; got {x.dtype}.")
    out = torch.empty_like(x)
    pos_mask = rshift >= 0
    if torch.any(pos_mask):
        s = rshift[pos_mask].to(torch.int64)
        bias = torch.where(s > 0, torch.ones_like(s) << (s - 1), torch.zeros_like(s))
        out[pos_mask] = (x[pos_mask] + bias) >> s
    neg_mask = ~pos_mask
    if torch.any(neg_mask):
        out[neg_mask] = x[neg_mask] << (-rshift[neg_mask].to(torch.int64))
    return out


def _apply_pwl_shift(
    prod: torch.Tensor,
    shifts: torch.Tensor,
    *,
    hw_ref: bool,
) -> torch.Tensor:
    shifted = torch.empty_like(prod)
    pos_mask = shifts >= 0
    if torch.any(pos_mask):
        if hw_ref:
            shifted[pos_mask] = _arith_right_shift_half_up(
                prod[pos_mask], shifts[pos_mask]
            )
        else:
            shifted[pos_mask] = round_shift(
                prod[pos_mask],
                shifts[pos_mask],
                RoundingMode.HALF_AWAY_FROM_ZERO,
            )
    if torch.any(~pos_mask):
        left_shift = (-shifts[~pos_mask]).to(torch.int64)
        shifted[~pos_mask] = prod[~pos_mask] << left_shift
    if hw_ref:
        shifted = _sat_int32_acc(shifted)
    return shifted


def _check_sim_lut_input(x: torch.Tensor) -> None:
    if x.dtype not in (torch.int16, SIM_TENSOR_DTYPE):
        raise TypeError(
            f"LUT input must be torch.int16 or {SIM_TENSOR_DTYPE}; got {x.dtype}."
        )


def lookup_lut_int16(
    x_int16: torch.Tensor,
    lut_int16: torch.Tensor,
    *,
    input_qmin: int = -32768,
    input_qmax: int = 32767,
) -> torch.Tensor:
    """Lookup an INT16 LUT by uniformly mapping input integer range to table indices."""

    _check_sim_lut_input(x_int16)
    if lut_int16.dtype != torch.int16:
        raise TypeError(f"lut_int16 must be torch.int16; got {lut_int16.dtype}.")
    if lut_int16.dim() != 1:
        raise ValueError("lut_int16 must be a 1-D tensor.")
    if input_qmin >= input_qmax:
        raise ValueError("input_qmin must be smaller than input_qmax.")

    table_size = lut_int16.numel()
    x_int32 = x_int16.to(torch.int32)
    numerator = (x_int32 - input_qmin) * (table_size - 1)
    denominator = input_qmax - input_qmin
    indices = torch.div(numerator, denominator, rounding_mode="floor")
    indices = torch.clamp(indices, 0, table_size - 1).to(torch.long)
    out = lut_int16.to(device=x_int16.device).index_select(0, indices.flatten())
    return out.view(x_int16.shape).to(SIM_TENSOR_DTYPE)


def evaluate_pwl_lut_int16(
    x_int16: torch.Tensor,
    pwl_lut: Dict[str, Any],
) -> torch.Tensor:
    """Evaluate a general-scale piecewise-linear LUT using integer arithmetic."""

    _check_sim_lut_input(x_int16)

    device = x_int16.device
    thresholds = torch.as_tensor(pwl_lut["thresholds"], dtype=torch.int32, device=device)
    q_b = torch.as_tensor(pwl_lut["q_b"], dtype=torch.int16, device=device)
    n_bx_total = torch.as_tensor(pwl_lut["n_bx_total"], dtype=torch.int8, device=device)
    term_c = torch.as_tensor(pwl_lut["term_c"], dtype=torch.int32, device=device)
    input_zero_point = torch.as_tensor(
        pwl_lut["input_zero_point"], dtype=torch.int32, device=device
    )
    output_qmin = int(torch.as_tensor(pwl_lut["output_qmin"]).item())
    output_qmax = int(torch.as_tensor(pwl_lut["output_qmax"]).item())

    if thresholds.dim() != 1 or thresholds.numel() == 0:
        raise ValueError("pwl_lut['thresholds'] must be a non-empty 1-D tensor.")

    x_flat = x_int16.to(torch.int32).flatten()
    segment_idx = torch.searchsorted(thresholds, x_flat, right=True) - 1
    segment_idx = torch.clamp(segment_idx, 0, thresholds.numel() - 1).to(torch.long)

    hw_ref = _pwl_hw_ref_enabled()
    centered = x_flat - input_zero_point
    if _pwl_hw_mac_sat_enabled():
        # PE BxC: INT16 multiply operands, INT32 product with saturation (ADR-015).
        centered = saturate_to_range(centered, INT16_QMIN, INT16_QMAX)
        prod = _sat_int32_acc(
            centered.to(torch.int64) * q_b.index_select(0, segment_idx).to(torch.int64)
        )
    else:
        prod = centered.to(torch.int64) * q_b.index_select(0, segment_idx).to(torch.int64)

    shifts = n_bx_total.index_select(0, segment_idx)
    shifted = _apply_pwl_shift(prod, shifts, hw_ref=hw_ref)

    y = shifted + term_c.index_select(0, segment_idx).to(torch.int64)
    if hw_ref:
        y = _sat_int32_acc(y)
    y = saturate_sim_tensor(y, output_qmin, output_qmax)
    return y.view(x_int16.shape)


class _LutInt16Kernel:
    """Base class for 1-input LUT nonlinear kernels."""

    module_type = None

    def __call__(
        self,
        inputs: List[Int16QuantizedTensor],
        params: Dict[str, Any],
        output_encoding: OutputEncoding,
        extra: Dict[str, Any],
    ) -> Int16QuantizedTensor:
        del params
        if len(inputs) != 1:
            raise ValueError(f"{type(self).__name__} expects 1 input; got {len(inputs)}.")

        pwl_lut = extra.get("pwl_lut")
        if pwl_lut is not None:
            x_repr = inputs[0].int_repr
            phase_fold = extra.get("phase_fold")
            if phase_fold is not None:
                x_repr = fold_periodic_input_to_principal_range(
                    x_repr,
                    str(phase_fold),
                    InputEncoding(
                        scale=inputs[0].scale,
                        zero_point=inputs[0].zero_point,
                        qmin=inputs[0].qmin,
                        qmax=inputs[0].qmax,
                        axis=inputs[0].axis,
                    ),
                )
            lut_input_enc = extra.get("pwl_input_encoding")
            baked = bool(pwl_lut.get("scale_adapter_baked", False))
            if lut_input_enc is not None and not baked:
                if not isinstance(lut_input_enc, InputEncoding):
                    raise TypeError("extra['pwl_input_encoding'] must be InputEncoding.")
                op_enc = InputEncoding(
                    scale=inputs[0].scale,
                    zero_point=inputs[0].zero_point,
                    qmin=inputs[0].qmin,
                    qmax=inputs[0].qmax,
                    axis=inputs[0].axis,
                )
                x_repr = align_op_quant_grid_to_lut_quant_grid(
                    x_repr, op_enc, lut_input_enc
                )
            int_repr = evaluate_pwl_lut_int16(x_repr, pwl_lut)
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

        lut_int16 = extra.get("lut_int16")
        if lut_int16 is None:
            raise ValueError("LUT kernel requires extra['pwl_lut'] or extra['lut_int16'].")
        if not isinstance(lut_int16, torch.Tensor):
            lut_int16 = torch.tensor(lut_int16, dtype=torch.int16)

        input_qmin = int(extra.get("input_qmin", inputs[0].qmin))
        input_qmax = int(extra.get("input_qmax", inputs[0].qmax))
        int_repr = lookup_lut_int16(
            inputs[0].int_repr,
            lut_int16,
            input_qmin=input_qmin,
            input_qmax=input_qmax,
        )
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


@register_fixed_kernel(nn.Sigmoid)
class SigmoidInt16Kernel(_LutInt16Kernel):
    """INT16 LUT kernel for torch.nn.Sigmoid."""

    module_type = nn.Sigmoid


@register_fixed_kernel(nn.Tanh)
class TanhInt16Kernel(_LutInt16Kernel):
    """INT16 LUT kernel for torch.nn.Tanh."""

    module_type = nn.Tanh


@register_fixed_kernel(nn.GELU)
class GELUInt16Kernel(_LutInt16Kernel):
    """INT16 PWL LUT kernel for torch.nn.GELU."""

    module_type = nn.GELU


@register_fixed_kernel(nn.SiLU)
class SiLUInt16Kernel(_LutInt16Kernel):
    """INT16 PWL LUT kernel for torch.nn.SiLU."""

    module_type = nn.SiLU


@register_fixed_kernel(nn.Mish)
class MishInt16Kernel(_LutInt16Kernel):
    """INT16 PWL LUT kernel for torch.nn.Mish."""

    module_type = nn.Mish


@register_fixed_kernel(nn.Softplus)
class SoftplusInt16Kernel(_LutInt16Kernel):
    """INT16 PWL LUT kernel for torch.nn.Softplus."""

    module_type = nn.Softplus


@register_fixed_kernel(nn.Hardsigmoid)
class HardsigmoidInt16Kernel(_LutInt16Kernel):
    """INT16 PWL LUT kernel for torch.nn.Hardsigmoid."""

    module_type = nn.Hardsigmoid


@register_fixed_kernel(nn.Hardswish)
class HardswishInt16Kernel(_LutInt16Kernel):
    """INT16 PWL LUT kernel for torch.nn.Hardswish."""

    module_type = nn.Hardswish


@register_fixed_kernel(nn.LeakyReLU)
class LeakyReLUInt16Kernel(_LutInt16Kernel):
    """INT16 PWL LUT kernel for torch.nn.LeakyReLU."""

    module_type = nn.LeakyReLU


@register_fixed_kernel(nn.PReLU)
class PReLUInt16Kernel(_LutInt16Kernel):
    """INT16 PWL LUT kernel for torch.nn.PReLU."""

    module_type = nn.PReLU


@register_fixed_kernel(custom.Sin)
class SinInt16Kernel(_LutInt16Kernel):
    """INT16 PWL ``sin`` with optional ``extra['phase_fold']='sin'``."""

    module_type = custom.Sin


@register_fixed_kernel(custom.Cos)
class CosInt16Kernel(_LutInt16Kernel):
    """INT16 PWL ``cos`` via sin table + ``phase_fold='cos'`` (adapter default)."""

    module_type = custom.Cos


@register_fixed_kernel(custom.Exponential)
class ExponentialInt16Kernel(_LutInt16Kernel):
    """INT16 PWL kernel for ``custom.Exponential`` (``torch.exp``)."""

    module_type = custom.Exponential


@register_fixed_kernel(custom.Log)
class LogInt16Kernel(_LutInt16Kernel):
    """INT16 PWL kernel for ``custom.Log`` (``torch.log``)."""

    module_type = custom.Log


@register_fixed_kernel(custom.Abs)
class AbsInt16Kernel(_LutInt16Kernel):
    """INT16 PWL kernel for ``custom.Abs`` (PowerCompress / frontend)."""

    module_type = custom.Abs


@register_fixed_kernel(nn.Softmax)
class SoftmaxInt16Kernel:
    """INT16 Softmax: PWL ``exp`` + integer sum (default), or legacy float reference."""

    module_type = nn.Softmax

    def __call__(
        self,
        inputs: List[Int16QuantizedTensor],
        params: Dict[str, Any],
        output_encoding: OutputEncoding,
        extra: Dict[str, Any],
    ) -> Int16QuantizedTensor:
        if len(inputs) != 1:
            raise ValueError(f"SoftmaxInt16Kernel expects 1 input; got {len(inputs)}.")
        dim = int(extra.get("dim", params.get("dim", -1)))
        if extra.get("legacy_float_softmax"):
            with int16_eval_allow_debug_float():
                x_float = inputs[0].to_float()
            y_float = torch.softmax(x_float, dim=dim)
            q = quantize_float_to_grid(
                y_float,
                output_encoding.scale,
                output_encoding.zero_point,
                output_encoding.qmin,
                output_encoding.qmax,
            ).to(SIM_TENSOR_DTYPE)
            return Int16QuantizedTensor(
                int_repr=q,
                scale=output_encoding.scale.to(device=q.device),
                zero_point=output_encoding.zero_point.to(device=q.device, dtype=torch.int32),
                qmin=output_encoding.qmin,
                qmax=output_encoding.qmax,
                axis=output_encoding.axis,
            )
        return softmax_int16_pwl(inputs[0], dim=dim, output_encoding=output_encoding)

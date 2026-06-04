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
"""CLZ-normalized LUT inference (sqrt/rsqrt/reciprocal/power_2).

Reference: ``abc_lut-shuai/lut_int_general/quantization/lut.py`` +
``clz_normalized_fitter.py``; hardware notes in ``lut_int_po2/docs/PE_CLZ_BxC_Design.md``.
"""

from __future__ import annotations

import bisect
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Union

import torch

from aimet_torch._base.nn.modules import custom
from aimet_torch.fixed_point.encoding import InputEncoding, OutputEncoding
from aimet_torch.fixed_point.offline.lut_gen import (
    align_op_quant_grid_to_lut_quant_grid,
    encodings_share_quant_grid,
    _remap_lut_q_to_op_grid,
)
from aimet_torch.fixed_point.offline.multiplier import quantize_multiplier
from aimet_torch.fixed_point.registry import register_fixed_kernel
from aimet_torch.fixed_point.requantize import SIM_TENSOR_DTYPE, saturate_sim_tensor
from aimet_torch.fixed_point.tensor import Int16QuantizedTensor

SQRT2_Q16 = 92682
INV_SQRT2_Q16 = 46341
Q_SHIFT = 16


def _signed_int_range(bit_width: int) -> tuple[int, int]:
    if bit_width <= 0:
        raise ValueError(f"bit_width must be positive; got {bit_width}.")
    if bit_width == 1:
        return -1, 0
    return (-(1 << (bit_width - 1)), (1 << (bit_width - 1)) - 1)


def _sat_signed_int(value: int, bit_width: int) -> int:
    lo, hi = _signed_int_range(bit_width)
    return max(lo, min(hi, int(value)))


def _arith_right_shift_round_half_up(value: int, shift: int) -> int:
    s = int(shift)
    if s == 0:
        return int(value)
    if s > 0:
        return (int(value) + (1 << (s - 1))) >> s
    return int(value) << (-s)


def clz_normalize_fixed_point(x_offset: int, bit_width: int) -> tuple[int, int]:
    """Return ``(mantissa_int, e_offset)`` for ``x_offset > 0`` (abc CLZ)."""

    if x_offset <= 0:
        return 0, 0
    msb_position = int(x_offset).bit_length()
    leading_zeros = bit_width - msb_position
    if leading_zeros >= 0:
        mantissa_int = int(x_offset) << leading_zeros
    else:
        mantissa_int = int(x_offset) >> (-leading_zeros)
    return int(mantissa_int), msb_position - 1


def denormalize_clz(
    q_y_norm: int,
    exponent: int,
    func_name: str,
    *,
    norm_output_zp: int,
    output_zp: int,
    r_q: int,
    r_shift: int,
    acc_bw: int = 32,
) -> int:
    """De-normalise normalized-space LUT output to the output quant grid."""

    name = func_name.strip().lower()
    y_offset = int(q_y_norm) - int(norm_output_zp)

    if name == "reciprocal":
        e_int = -int(exponent)
        parity_factor = None
    elif name == "sqrt":
        e_int = int(exponent) // 2
        parity_factor = SQRT2_Q16 if (int(exponent) & 1) else None
    elif name == "rsqrt":
        e_int = -(int(exponent) // 2)
        parity_factor = INV_SQRT2_Q16 if (int(exponent) & 1) else None
    elif name in ("power_2", "power2"):
        e_int = 2 * int(exponent)
        parity_factor = None
    else:
        raise ValueError(f"unsupported CLZ function: {func_name}")

    val = int(y_offset) * int(r_q)
    if parity_factor is not None:
        val = _sat_signed_int(val, acc_bw)
        val = (val * int(parity_factor)) >> Q_SHIFT
    net_shift = int(r_shift) - int(e_int)
    if net_shift >= 0:
        val = _arith_right_shift_round_half_up(int(val), net_shift)
    else:
        val = int(val) << (-net_shift)
    return _sat_signed_int(int(val) + int(output_zp), acc_bw)


def load_clz_lut_from_json(
    data: Union[dict[str, Any], str, Path],
    func_name: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Load CLZ LUT JSON (``{func: {quantization, segments}}``)."""

    if isinstance(data, (str, Path)):
        path = Path(data)
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = data
    if func_name is None:
        func_name = next(iter(payload))
    body = payload[func_name]
    if "clz_params" not in body.get("quantization", {}):
        raise ValueError(f"{func_name}: not a CLZ-normalized LUT (missing clz_params).")
    return func_name, body


def _evaluate_clz_reference(
    q_x: torch.Tensor,
    clz_lut: dict[str, Any],
    func_name: str,
) -> torch.Tensor:
    """Per-element golden reference (CPU Python loop). Kept as bit-exact oracle."""

    quant = clz_lut["quantization"]
    clz_params = quant["clz_params"]
    input_params = quant["input"]
    output_params = quant["output"]
    norm_input_params = quant["normalized_input"]
    norm_output_params = quant["normalized_output"]

    input_bit_width = int(clz_params["input_bit_width"])
    r_q = int(clz_params["denorm_R_q"])
    r_shift = int(clz_params["denorm_R_shift"])
    acc_bw = int(clz_params.get("internal_acc_bw", 32))

    out_qmin = int(output_params["min"])
    out_qmax = int(output_params["max"])
    zp_x = int(input_params["zero_point"])
    zp_m = int(norm_input_params["zero_point"])
    zp_norm_y = int(norm_output_params["zero_point"])
    zp_y = int(output_params["zero_point"])
    s_m = float(norm_input_params["scale"])

    segments = clz_lut["segments"]
    thr_starts = [int(seg["threshold_quantized"][0]) for seg in segments]

    q_factor, n_factor = quantize_multiplier(
        torch.tensor(1.0 / s_m, dtype=torch.float64)
    )
    q_factor_i = int(q_factor.reshape(-1)[0].item())
    n_factor_i = int(n_factor.reshape(-1)[0].item())
    map_total_shift = input_bit_width + n_factor_i

    outputs: list[int] = []
    name = func_name.lower()
    for q_val in q_x.to(torch.int32).flatten().tolist():
        q_x_i = int(q_val)
        x_offset = q_x_i - zp_x
        sign_neg = False
        if name in ("power_2", "power2"):
            if x_offset == 0:
                outputs.append(zp_y)
                continue
            x_for_clz = abs(x_offset)
        elif name == "reciprocal":
            if x_offset == 0:
                outputs.append(out_qmax)
                continue
            sign_neg = x_offset < 0
            x_for_clz = abs(x_offset)
        elif name == "rsqrt":
            if x_offset <= 0:
                outputs.append(out_qmax)
                continue
            x_for_clz = x_offset
        elif name == "sqrt":
            if x_offset <= 0:
                outputs.append(zp_y)
                continue
            x_for_clz = x_offset
        else:
            if x_offset <= 0:
                outputs.append(out_qmax if name in ("reciprocal", "rsqrt") else zp_y)
                continue
            x_for_clz = x_offset

        m_int, e_offset = clz_normalize_fixed_point(x_for_clz, input_bit_width)
        exponent = int(e_offset) + 1

        prod = int(m_int) * q_factor_i
        if map_total_shift >= 0:
            m_scaled = _arith_right_shift_round_half_up(prod, map_total_shift)
        else:
            m_scaled = prod << (-map_total_shift)
        q_m = int(m_scaled) + zp_m

        seg_idx = max(
            0,
            min(
                len(segments) - 1,
                bisect.bisect_right(thr_starts, q_m) - 1,
            ),
        )
        seg = segments[seg_idx]
        q_b = int(seg["coefficients_quantized"]["q_b"])
        n_bx = int(seg["shift_bits"]["n_bx_total"])
        term_c = int(seg["term_c_precomputed"])

        m_offset = q_m - zp_m
        bm = _sat_signed_int(q_b * m_offset, acc_bw)
        if n_bx >= 0:
            term_bm = _sat_signed_int(_arith_right_shift_round_half_up(bm, n_bx), acc_bw)
        else:
            term_bm = _sat_signed_int(bm << (-n_bx), acc_bw)
        q_y_norm = _sat_signed_int(term_bm + term_c, acc_bw)
        q_y = denormalize_clz(
            q_y_norm,
            exponent,
            name,
            norm_output_zp=zp_norm_y,
            output_zp=zp_y,
            r_q=r_q,
            r_shift=r_shift,
            acc_bw=acc_bw,
        )
        if sign_neg:
            q_y = _sat_signed_int(2 * int(zp_y) - int(q_y), acc_bw)
        outputs.append(max(out_qmin, min(out_qmax, q_y)))

    return torch.tensor(outputs, dtype=SIM_TENSOR_DTYPE, device=q_x.device).view(q_x.shape)


# Max iterations for vectorized bit_length; CLZ inputs are <= int16 magnitude,
# but keep generous headroom for power_2 (x^2 grid) inputs.
_BIT_LENGTH_ITERS = 40


def _bit_length_vec(x: torch.Tensor) -> torch.Tensor:
    """Vectorized ``int.bit_length`` for non-negative int64 tensor (data-independent)."""

    bl = torch.zeros_like(x)
    v = x.clone()
    for _ in range(_BIT_LENGTH_ITERS):
        bl = bl + (v > 0).to(x.dtype)
        v = v >> 1
    return bl


def _shift_lr_vec(x: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """Left shift where ``shift >= 0`` else arithmetic right shift (no rounding)."""

    left = x << shift.clamp(min=0)
    right = x >> (-shift).clamp(min=0)
    return torch.where(shift >= 0, left, right)


def _arsh_round_vec(value: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """Vectorized ``_arith_right_shift_round_half_up`` with per-element int64 shift."""

    one = torch.ones_like(value)
    s_pos = shift.clamp(min=1)
    add = torch.where(shift > 0, one << (s_pos - 1), torch.zeros_like(value))
    rsh = (value + add) >> s_pos
    lsh = value << (-shift).clamp(min=1)
    return torch.where(shift > 0, rsh, torch.where(shift < 0, lsh, value))


def _sat_signed_vec(value: torch.Tensor, bit_width: int) -> torch.Tensor:
    lo, hi = _signed_int_range(bit_width)
    return value.clamp(min=lo, max=hi)


def _evaluate_clz_vectorized(
    q_x: torch.Tensor,
    clz_lut: dict[str, Any],
    func_name: str,
) -> torch.Tensor:
    """On-device vectorized CLZ LUT (bit-exact with ``_evaluate_clz_reference``)."""

    quant = clz_lut["quantization"]
    clz_params = quant["clz_params"]
    input_params = quant["input"]
    output_params = quant["output"]
    norm_input_params = quant["normalized_input"]
    norm_output_params = quant["normalized_output"]

    input_bit_width = int(clz_params["input_bit_width"])
    r_q = int(clz_params["denorm_R_q"])
    r_shift = int(clz_params["denorm_R_shift"])
    acc_bw = int(clz_params.get("internal_acc_bw", 32))

    out_qmin = int(output_params["min"])
    out_qmax = int(output_params["max"])
    zp_x = int(input_params["zero_point"])
    zp_m = int(norm_input_params["zero_point"])
    zp_norm_y = int(norm_output_params["zero_point"])
    zp_y = int(output_params["zero_point"])
    s_m = float(norm_input_params["scale"])

    name = func_name.strip().lower()
    if name == "power2":
        name = "power_2"
    if name not in ("sqrt", "rsqrt", "reciprocal", "power_2"):
        raise ValueError(f"unsupported CLZ function: {func_name}")

    segments = clz_lut["segments"]
    device = q_x.device

    thr_starts = torch.tensor(
        [int(seg["threshold_quantized"][0]) for seg in segments],
        dtype=torch.int64,
        device=device,
    )
    seg_q_b = torch.tensor(
        [int(seg["coefficients_quantized"]["q_b"]) for seg in segments],
        dtype=torch.int64,
        device=device,
    )
    seg_n_bx = torch.tensor(
        [int(seg["shift_bits"]["n_bx_total"]) for seg in segments],
        dtype=torch.int64,
        device=device,
    )
    seg_term_c = torch.tensor(
        [int(seg["term_c_precomputed"]) for seg in segments],
        dtype=torch.int64,
        device=device,
    )

    q_factor, n_factor = quantize_multiplier(torch.tensor(1.0 / s_m, dtype=torch.float64))
    q_factor_i = int(q_factor.reshape(-1)[0].item())
    n_factor_i = int(n_factor.reshape(-1)[0].item())
    map_total_shift = input_bit_width + n_factor_i

    q = q_x.to(torch.int64)
    x_offset = q - zp_x

    # Branch-specific special-case masks + clz input (mirrors reference early-outs).
    if name == "power_2":
        special_mask = x_offset == 0
        special_val = zp_y
        x_for_clz = x_offset.abs()
        sign_neg = torch.zeros_like(x_offset, dtype=torch.bool)
    elif name == "reciprocal":
        special_mask = x_offset == 0
        special_val = out_qmax
        x_for_clz = x_offset.abs()
        sign_neg = x_offset < 0
    elif name == "rsqrt":
        special_mask = x_offset <= 0
        special_val = out_qmax
        x_for_clz = x_offset
        sign_neg = torch.zeros_like(x_offset, dtype=torch.bool)
    else:  # sqrt
        special_mask = x_offset <= 0
        special_val = zp_y
        x_for_clz = x_offset
        sign_neg = torch.zeros_like(x_offset, dtype=torch.bool)

    # Guard invalid entries so bit_length/shift stay well-defined; masked out later.
    x_safe = torch.where(x_for_clz > 0, x_for_clz, torch.ones_like(x_for_clz))

    msb_position = _bit_length_vec(x_safe)
    leading_zeros = input_bit_width - msb_position
    mantissa = _shift_lr_vec(x_safe, leading_zeros)
    exponent = msb_position

    prod = mantissa * q_factor_i
    if map_total_shift >= 0:
        m_scaled = _arsh_round_vec(
            prod, torch.full_like(prod, map_total_shift)
        )
    else:
        m_scaled = prod << (-map_total_shift)
    q_m = m_scaled + zp_m

    seg_idx = (torch.searchsorted(thr_starts, q_m, right=True) - 1).clamp(
        min=0, max=len(segments) - 1
    )
    q_b = seg_q_b[seg_idx]
    n_bx = seg_n_bx[seg_idx]
    term_c = seg_term_c[seg_idx]

    m_offset = q_m - zp_m
    bm = _sat_signed_vec(q_b * m_offset, acc_bw)
    term_bm = _sat_signed_vec(_arsh_round_vec(bm, n_bx), acc_bw)
    q_y_norm = _sat_signed_vec(term_bm + term_c, acc_bw)

    # ---- denormalize_clz (vectorized) ----
    y_offset = q_y_norm - zp_norm_y
    if name == "reciprocal":
        e_int = -exponent
    elif name == "sqrt":
        e_int = exponent >> 1
    elif name == "rsqrt":
        e_int = -(exponent >> 1)
    else:  # power_2
        e_int = 2 * exponent

    val = y_offset * r_q
    if name in ("sqrt", "rsqrt"):
        parity = SQRT2_Q16 if name == "sqrt" else INV_SQRT2_Q16
        is_odd = (exponent & 1).to(torch.bool)
        val_parity = (_sat_signed_vec(val, acc_bw) * parity) >> Q_SHIFT
        val = torch.where(is_odd, val_parity, val)

    net_shift = r_shift - e_int
    val = _arsh_round_vec(val, net_shift)
    q_y = _sat_signed_vec(val + zp_y, acc_bw)

    if sign_neg.any():
        q_y = torch.where(sign_neg, _sat_signed_vec(2 * zp_y - q_y, acc_bw), q_y)

    q_y = q_y.clamp(min=out_qmin, max=out_qmax)
    out = torch.where(special_mask, torch.full_like(q_y, special_val), q_y)
    return out.to(SIM_TENSOR_DTYPE).view(q_x.shape)


def evaluate_clz_normalized_lut_int16(
    q_x: torch.Tensor,
    clz_lut: dict[str, Any],
    func_name: str,
) -> torch.Tensor:
    """Evaluate one CLZ LUT on integer ``q_x`` (per-sample abc semantics).

    Dispatches to the on-device vectorized kernel by default; set
    ``AIMET_RX_CLZ_VECTORIZED=0`` to force the per-element Python reference
    (bit-exact oracle, used for parity tests and debugging).
    """

    if _clz_vectorized_enabled():
        return _evaluate_clz_vectorized(q_x, clz_lut, func_name)
    return _evaluate_clz_reference(q_x, clz_lut, func_name)


def _clz_vectorized_enabled() -> bool:
    raw = os.environ.get("AIMET_RX_CLZ_VECTORIZED", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _encoding_from_clz_quant(quant: dict[str, Any]) -> InputEncoding:
    return InputEncoding(
        scale=torch.tensor(quant["scale"], dtype=torch.float32),
        zero_point=torch.tensor(quant["zero_point"], dtype=torch.int32),
        qmin=int(quant["min"]),
        qmax=int(quant["max"]),
    )


def _resolve_clz_body_and_func(
    extra: Dict[str, Any], default_func: str
) -> tuple[str, dict[str, Any]]:
    raw = extra.get("clz_lut")
    if raw is None:
        path = extra.get("clz_lut_json_path")
        if path is None:
            raise ValueError(
                "CLZ kernel requires extra['clz_lut'] or extra['clz_lut_json_path']."
            )
        return load_clz_lut_from_json(path, func_name=default_func)
    if isinstance(raw, dict) and "segments" in raw:
        return str(extra.get("clz_func_name", default_func)), raw
    return load_clz_lut_from_json(raw, func_name=default_func)


def _clz_int16_forward(
    inputs: List[Int16QuantizedTensor],
    output_encoding: OutputEncoding,
    extra: Dict[str, Any],
    default_func: str,
) -> Int16QuantizedTensor:
    if len(inputs) != 1:
        raise ValueError(f"CLZ kernel expects 1 input; got {len(inputs)}.")
    func_name, clz_body = _resolve_clz_body_and_func(extra, default_func)
    x_in = inputs[0]
    lut_in_enc = _encoding_from_clz_quant(clz_body["quantization"]["input"])
    op_in_enc = InputEncoding(
        scale=x_in.scale,
        zero_point=x_in.zero_point,
        qmin=x_in.qmin,
        qmax=x_in.qmax,
        axis=x_in.axis,
    )
    q_x = x_in.int_repr
    if not encodings_share_quant_grid(lut_in_enc, op_in_enc):
        q_x = align_op_quant_grid_to_lut_quant_grid(q_x, op_in_enc, lut_in_enc)
    int_repr = evaluate_clz_normalized_lut_int16(q_x, clz_body, func_name)
    lut_out_enc = _encoding_from_clz_quant(clz_body["quantization"]["output"])
    op_out_enc = InputEncoding(
        scale=output_encoding.scale,
        zero_point=output_encoding.zero_point,
        qmin=output_encoding.qmin,
        qmax=output_encoding.qmax,
        axis=output_encoding.axis,
    )
    if not encodings_share_quant_grid(lut_out_enc, op_out_enc):
        int_repr = _remap_lut_q_to_op_grid(int_repr, lut_out_enc, op_out_enc)
    return Int16QuantizedTensor(
        int_repr=saturate_sim_tensor(
            int_repr, output_encoding.qmin, output_encoding.qmax
        ),
        scale=output_encoding.scale.to(device=int_repr.device),
        zero_point=output_encoding.zero_point.to(
            device=int_repr.device, dtype=torch.int32
        ),
        qmin=output_encoding.qmin,
        qmax=output_encoding.qmax,
        axis=output_encoding.axis,
    )


@register_fixed_kernel(custom.Sqrt)
class SqrtInt16ClzKernel:
    """``custom.Sqrt`` via CLZ LUT (``extra['clz_lut']`` or adapter auto-fit)."""

    module_type = custom.Sqrt

    def __call__(
        self,
        inputs: List[Int16QuantizedTensor],
        params: Dict[str, Any],
        output_encoding: OutputEncoding,
        extra: Dict[str, Any],
    ) -> Int16QuantizedTensor:
        del params
        return _clz_int16_forward(inputs, output_encoding, extra, "sqrt")


@register_fixed_kernel(custom.RSqrt)
class RSqrtInt16ClzKernel:
    """``custom.RSqrt`` via CLZ LUT."""

    module_type = custom.RSqrt

    def __call__(
        self,
        inputs: List[Int16QuantizedTensor],
        params: Dict[str, Any],
        output_encoding: OutputEncoding,
        extra: Dict[str, Any],
    ) -> Int16QuantizedTensor:
        del params
        return _clz_int16_forward(inputs, output_encoding, extra, "rsqrt")


@register_fixed_kernel(custom.Reciprocal)
class ReciprocalInt16ClzKernel:
    """``custom.Reciprocal`` via CLZ LUT."""

    module_type = custom.Reciprocal

    def __call__(
        self,
        inputs: List[Int16QuantizedTensor],
        params: Dict[str, Any],
        output_encoding: OutputEncoding,
        extra: Dict[str, Any],
    ) -> Int16QuantizedTensor:
        del params
        return _clz_int16_forward(inputs, output_encoding, extra, "reciprocal")


@register_fixed_kernel(custom.Square)
class SquareInt16ClzKernel:
    """``custom.Square`` via CLZ ``power_2`` LUT (abc CLZ-normalized x^2 path)."""

    module_type = custom.Square

    def __call__(
        self,
        inputs: List[Int16QuantizedTensor],
        params: Dict[str, Any],
        output_encoding: OutputEncoding,
        extra: Dict[str, Any],
    ) -> Int16QuantizedTensor:
        del params
        return _clz_int16_forward(inputs, output_encoding, extra, "power_2")

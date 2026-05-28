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
from pathlib import Path
from typing import Any, Dict, List, Union

import torch

from aimet_torch._base.nn.modules import custom
from aimet_torch.fixed_point.encoding import InputEncoding, OutputEncoding
from aimet_torch.fixed_point.offline.lut_gen import (
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


def evaluate_clz_normalized_lut_int16(
    q_x: torch.Tensor,
    clz_lut: dict[str, Any],
    func_name: str,
) -> torch.Tensor:
    """Evaluate one CLZ LUT on integer ``q_x`` (per-sample abc semantics)."""

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
    int_repr = evaluate_clz_normalized_lut_int16(
        inputs[0].int_repr, clz_body, func_name
    )
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

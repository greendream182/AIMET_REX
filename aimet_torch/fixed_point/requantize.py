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
"""Integer-only requantization helpers for INT16 fixed-point execution."""

import os

import torch

from aimet_torch.fixed_point.rounding import RoundingMode

INT16_QMIN = -32768
INT16_QMAX = 32767
INT32_QMIN = -2147483648
INT32_QMAX = 2147483647
MULTIPLIER_QBITS = 16
MULTIPLIER_MAX = (1 << MULTIPLIER_QBITS) - 1

# ADR-013: G3 sim-tensor 段载体容器统一为 torch.int32（FixedPointSimTensor.int_repr 的 dtype）。
#
# Hard constraints (anti-patterns that MUST NOT appear elsewhere in the code):
#   - 这是 *模块级常量*，不读环境变量、不暴露 setter / context manager。
#   - 业务代码禁止 `if int_repr.dtype == torch.int16: ... else: ...` 类的 dtype 分支。
#   - 业务代码禁止散写 `.to(torch.int32)` 字面量，应统一调 `saturate_sim_tensor` 收口。
#   - 改容器（例如未来扩到 torch.int64 支持 >16bit 量化器）需改这一行常量 + 放宽 ADR-014
#     的 qmin/qmax 守门 assert + 重跑全量 bit-exact 回归 + byte-stream 等价测试。
#
# CI / pre-commit 守门（PR-4 启用）：
#   rg "os\\.environ.*SIM_TENSOR_DTYPE"            必须 0 命中
#   rg "SIM_TENSOR_DTYPE\\s*=\\s*torch\\."         仅 1 处（此处常量定义）
#   rg "dtype\\s*==\\s*torch\\.int16" aimet_torch/fixed_point/   0 命中（v1 兼容层可豁免）
SIM_TENSOR_DTYPE: torch.dtype = torch.int32


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes")


def hw_ref_mode_enabled() -> bool:
    """Strict hardware/abc simulation (``AIMET_RX_HW_REF`` or ``AIMET_RX_PWL_HW_REF``)."""

    return _env_truthy("AIMET_RX_HW_REF") or _env_truthy("AIMET_RX_PWL_HW_REF")


def int16_fixed_eval_mode() -> bool:
    """True when the process is in ``INT16_FIXED_EVAL`` (G3 pure fixed-point path)."""

    # pylint: disable=import-outside-toplevel
    from aimet_torch.fixed_point.execution_mode import (
        ExecutionMode,
        get_quant_execution_mode,
    )

    return get_quant_execution_mode() is ExecutionMode.INT16_FIXED_EVAL


def _forbid_float_fallback_in_eval(feature: str) -> None:
    if int16_fixed_eval_mode():
        raise RuntimeError(
            f"{feature} is not allowed in INT16_FIXED_EVAL; all kernels must use "
            "pure integer fixed-point compute."
        )


def sign_float_ref_enabled() -> bool:
    """Use dequant→``torch.sign``→requant instead of integer centered sign.

    Default off (integer centered sign, hardware-style). Enable with
    ``AIMET_RX_SIGN_FLOAT_REF=1`` when the graph must match float QDQ on
    near-zero inputs that quantize to 0 on the incoming grid (e.g. MRNN STFT).
    """

    if int16_fixed_eval_mode():
        if _env_truthy("AIMET_RX_SIGN_FLOAT_REF"):
            _forbid_float_fallback_in_eval("AIMET_RX_SIGN_FLOAT_REF")
        return False
    return _env_truthy("AIMET_RX_SIGN_FLOAT_REF")


def requantize_int32_prod_sat_enabled() -> bool:
    """INT32 saturate after ``acc * multiplier`` before rshift (ADR-015 strict mode).

    Default **off**: ``int64`` product is kept through ``round_shift`` (spec 05,
    e2e fidelity). Enable only for strict HW regression via
    ``AIMET_RX_REQUANTIZE_INT32_SAT=1`` or ``AIMET_RX_HW_REF``.
    """

    return _env_truthy("AIMET_RX_REQUANTIZE_INT32_SAT") or hw_ref_mode_enabled()


def mac_accumulator_int32_sat_enabled() -> bool:
    """INT32 saturate dot-product / conv accumulator before output requantize.

    Default ON (HW-faithful: the accumulator can exceed int32 for realistic K,
    so we clamp to the INT32 ALU width instead of silently wrapping). Set
    ``AIMET_RX_ACC_INT32_SAT=0`` to opt out (legacy non-saturating int32 cast).
    ``AIMET_RX_HW_REF`` always forces it on.

    ``INT16_FIXED_EVAL`` always forces it on (no float MAC fallback).
    """

    if hw_ref_mode_enabled() or int16_fixed_eval_mode():
        return True
    raw = os.environ.get("AIMET_RX_ACC_INT32_SAT")
    if raw is not None:
        return raw.strip().lower() in ("1", "true", "yes")
    return True


def saturate_mac_accumulator(acc: torch.Tensor) -> torch.Tensor:
    """Return ``torch.int32`` MAC result; clamp to INT32 ALU width when HW ref is on."""

    if mac_accumulator_int32_sat_enabled():
        return saturate_int32(acc.to(torch.int64))
    if acc.dtype != torch.int32:
        return acc.to(torch.int32)
    return acc


def int32_add_sat(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Add two int32 tensors; optional INT32 ALU saturation (eltwise Add/Sub)."""

    if mac_accumulator_int32_sat_enabled():
        return saturate_int32(lhs.to(torch.int64) + rhs.to(torch.int64))
    return lhs + rhs


def int32_sub_sat(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    if mac_accumulator_int32_sat_enabled():
        return saturate_int32(lhs.to(torch.int64) - rhs.to(torch.int64))
    return lhs - rhs


def int32_mul_sat(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    if mac_accumulator_int32_sat_enabled():
        return saturate_int32(lhs.to(torch.int64) * rhs.to(torch.int64))
    return lhs * rhs


def int32_sum_sat(
    tensor: torch.Tensor,
    dim: int | tuple[int, ...],
    *,
    keepdim: bool = False,
) -> torch.Tensor:
    if mac_accumulator_int32_sat_enabled():
        return saturate_int32(tensor.to(torch.int64).sum(dim=dim, keepdim=keepdim))
    return tensor.sum(dim=dim, keepdim=keepdim).to(torch.int32)


def _validate_requantize_inputs(
    acc: torch.Tensor,
    multiplier: torch.Tensor,
    rshift: torch.Tensor,
    y_zp: torch.Tensor,
):
    if acc.dtype != torch.int32:
        raise TypeError(f"acc must be torch.int32; got {acc.dtype}.")
    if multiplier.dtype != torch.uint16:
        raise TypeError(f"multiplier must be torch.uint16; got {multiplier.dtype}.")
    if rshift.dtype != torch.int8:
        raise TypeError(f"rshift must be torch.int8; got {rshift.dtype}.")
    if y_zp.dtype != torch.int32:
        raise TypeError(f"y_zp must be torch.int32; got {y_zp.dtype}.")

    if torch.any(multiplier.to(torch.int64) < 0) or torch.any(
        multiplier.to(torch.int64) > MULTIPLIER_MAX
    ):
        raise ValueError(f"multiplier must be in [0, {MULTIPLIER_MAX}].")
    if torch.any(rshift < 0) or torch.any(rshift > 31):
        raise ValueError("rshift must be in [0, 31].")


def saturate_to_range(x: torch.Tensor, qmin: int, qmax: int) -> torch.Tensor:
    """Clamp integer tensor values into the target quantization range."""

    if qmin > qmax:
        raise ValueError(f"qmin ({qmin}) must be <= qmax ({qmax}).")
    return torch.clamp(x, qmin, qmax)


def saturate_int16(x: torch.Tensor) -> torch.Tensor:
    """Saturate tensor values to signed INT16 range and return torch.int16.

    Compat alias kept for the int16-container era. New kernel code should
    call :func:`saturate_sim_tensor` instead; PR-3 migrates business paths.
    """

    return saturate_to_range(x, INT16_QMIN, INT16_QMAX).to(torch.int16)


def saturate_int32(x: torch.Tensor) -> torch.Tensor:
    """Saturate to signed INT32 range (Ada200 accumulator / ALU width).

    Used after fixed-point multiply-add steps in PWL and requantize paths so
    the simulator matches hardware saturation instead of silent int64 carry.
    """

    return saturate_to_range(x, INT32_QMIN, INT32_QMAX).to(torch.int32)


def saturate_sim_tensor(
    x: torch.Tensor,
    qmin: int = INT16_QMIN,
    qmax: int = INT16_QMAX,
) -> torch.Tensor:
    """Saturate to ``[qmin, qmax]`` and cast to :data:`SIM_TENSOR_DTYPE` (int32).

    Single source of truth for "clamp + container-dtype cast" in fixed-point
    kernels. Values stay numerically in ``[qmin, qmax]``; the int32 container
    is only the storage slot (ADR-013).
    """

    return saturate_to_range(x, qmin, qmax).to(SIM_TENSOR_DTYPE)


def _as_int64_power_of_two(rshift: torch.Tensor) -> torch.Tensor:
    one = torch.ones((), dtype=torch.int64, device=rshift.device)
    return one << rshift.to(torch.int64)


def round_shift(
    x: torch.Tensor,
    rshift: torch.Tensor,
    rounding_mode: RoundingMode,
) -> torch.Tensor:
    """Right shift int64 tensor with the requested integer rounding policy."""

    if x.dtype != torch.int64:
        raise TypeError(f"x must be torch.int64; got {x.dtype}.")
    if rshift.dtype != torch.int8:
        raise TypeError(f"rshift must be torch.int8; got {rshift.dtype}.")

    if torch.any(rshift < 0) or torch.any(rshift > 31):
        raise ValueError("rshift must be in [0, 31].")

    if torch.all(rshift == 0):
        return x

    rshift_i64 = rshift.to(torch.int64)
    if rounding_mode == RoundingMode.TRUNCATE:
        return x >> rshift_i64

    if rounding_mode == RoundingMode.HALF_AWAY_FROM_ZERO:
        offset = _as_int64_power_of_two(rshift) >> 1
        abs_rounded = (torch.abs(x) + offset) >> rshift_i64
        return torch.where(x < 0, -abs_rounded, abs_rounded)

    if rounding_mode == RoundingMode.HALF_UP:
        bias = torch.where(
            rshift_i64 > 0,
            torch.ones_like(rshift_i64) << (rshift_i64 - 1),
            torch.zeros_like(rshift_i64),
        )
        return (x + bias) >> rshift_i64

    if rounding_mode == RoundingMode.HALF_TO_EVEN:
        truncated = x >> rshift_i64
        denominator = _as_int64_power_of_two(rshift)
        remainder = x - (truncated << rshift_i64)
        half = denominator >> 1
        bump = (remainder > half) | (
            (remainder == half) & ((truncated & 1) == 1)
        )
        return truncated + bump.to(torch.int64)

    raise ValueError(f"Unsupported rounding mode: {rounding_mode}.")


def requantize_int(
    acc: torch.Tensor,
    multiplier: torch.Tensor,
    rshift: torch.Tensor,
    y_zp: torch.Tensor,
    qmin: int = INT16_QMIN,
    qmax: int = INT16_QMAX,
    rounding_mode: RoundingMode = RoundingMode.HALF_TO_EVEN,
) -> torch.Tensor:
    """Requantize int32 accumulator using integer multiplier + rshift.

    Returns a tensor in the :data:`SIM_TENSOR_DTYPE` container (int32) whose
    values are clamped to ``[qmin, qmax]``. The container is independent of
    the semantic bitwidth carried by ``qmin``/``qmax`` (ADR-013).
    """

    _validate_requantize_inputs(acc, multiplier, rshift, y_zp)

    prod = acc.to(torch.int64) * multiplier.to(torch.int64)
    if requantize_int32_prod_sat_enabled():
        prod = saturate_int32(prod).to(torch.int64)
    rounded = round_shift(prod, rshift, rounding_mode)
    shifted = rounded + y_zp.to(torch.int64)
    return saturate_sim_tensor(shifted, qmin, qmax)

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
"""QuantGRU black-box adapter helpers (native API + AIMET-side stubs)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from aimet_torch.fixed_point import ExecutionMode, get_quant_execution_mode
from aimet_torch.fixed_point.errors import IncompatibleAdapterVersionError
from aimet_torch.fixed_point.gradient_helpers import (
    is_quantized_activation,
    requantize_fp_to_int,
    stop_grad_dequantize,
    wrap_int_tensor_with_meta,
)
from aimet_torch.fixed_point.qat.carrier import publish_int16_carrier
from aimet_torch.fixed_point.tensor import Int16QuantizedTensor

__all__ = [
    "EXPECTED_ADAPTER_MAJOR",
    "MIN_SUPPORTED_ADAPTER_MINOR",
    "check_adapter_version",
    "get_io_quant_meta",
    "aimet_configure",
    "aimet_capabilities",
    "forward_quantized",
    "dispatch_quantgru_blackbox",
    "resolve_quantgru_mode_str",
]

EXPECTED_ADAPTER_MAJOR = 1
MIN_SUPPORTED_ADAPTER_MINOR = 0

# QuantGRU contract v1 supported_modes (see plan §1.5).
# NOTE: AIMET's FIXED_SCALE_QDQ is intentionally NOT in this set; QuantGRU
# only does floating-point forward in that mode (boundary fixed-scale grid is
# enforced by AIMET's own QuantizeDequantize). See plan §0.2.1.
_SUPPORTED_MODES = (
    "fp32",
    "fp32_qdq",
    "fp16_qdq",
    "int16_fixed_eval",
    "int16_fixed_qat_sim",
    "calibrating",
)


def resolve_quantgru_mode_str(mode: ExecutionMode) -> str:
    """Map AIMET ``ExecutionMode`` to QuantGRU contract v1 mode string.

    QuantGRU only knows the strings in :data:`_SUPPORTED_MODES`. AIMET's
    ``FIXED_SCALE_QDQ`` is wrapper-side aliased to ``"fp32_qdq"`` because:

    * QuantGRU itself only runs floating-point forward in this mode
      (``use_quantization=False``); the (m_int16, rshift) grid is enforced
      by AIMET's boundary :class:`QuantizeDequantize` (see plan §0.2.1 /
      ``aimet_torch/v2/quantization/affine/backends/torch_builtins.py:230``).
    * Keeping FIXED_SCALE_QDQ out of contract v1 avoids forcing the
      ``quant-gru-pytorch`` project to add an AIMET-specific mode string.

    All other ``ExecutionMode`` values map 1-to-1 to their lowercase string.
    """
    if mode == ExecutionMode.FIXED_SCALE_QDQ:
        return ExecutionMode.FP32_QDQ.value
    return mode.value


def _shift_to_scale(shift: int) -> float:
    return float(2.0 ** (-int(shift)))


def _meta_from_quant_params(
    quant_params: Any,
    prefix: str,
    bitwidth_config: Any = None,
) -> Dict[str, Any]:
    shift = getattr(quant_params, f"shift_{prefix}")
    zp = getattr(quant_params, f"zp_{prefix}", 0)
    if bitwidth_config is not None:
        bitwidth = int(getattr(bitwidth_config, prefix))
        is_symmetric = bool(getattr(bitwidth_config, f"{prefix}symmetric_"))
        unsigned_attr = f"{prefix}unsigned_"
        is_unsigned = bool(getattr(bitwidth_config, unsigned_attr, False))
    else:
        bitwidth = 16
        is_symmetric = True
        is_unsigned = False
    if is_unsigned and is_symmetric:
        is_symmetric = False
    return {
        "scale": _shift_to_scale(shift),
        "zp": int(zp),
        "bitwidth": bitwidth,
        "is_symmetric": is_symmetric,
    }


def _native_on_quant_gru(module: nn.Module, method_name: str) -> bool:
    for cls in type(module).__mro__:
        if cls.__name__ != "QuantGRU":
            continue
        return method_name in cls.__dict__
    return False


def _call_native(module: nn.Module, method_name: str, *args, **kwargs):
    for cls in type(module).__mro__:
        if cls.__name__ != "QuantGRU":
            continue
        method = cls.__dict__.get(method_name)
        if method is None:
            break
        return method(module, *args, **kwargs)
    raise AttributeError(method_name)


def check_adapter_version(module: nn.Module) -> None:
    caps = aimet_capabilities(module)
    version = caps.get("adapter_version", "0.0")
    parts = str(version).split(".", maxsplit=1)
    major = int(parts[0])
    minor = int(parts[1]) if len(parts) > 1 else 0
    if major != EXPECTED_ADAPTER_MAJOR or minor < MIN_SUPPORTED_ADAPTER_MINOR:
        raise IncompatibleAdapterVersionError(
            f"QuantGRU adapter_version={version} not supported. "
            f"AIMET requires major == {EXPECTED_ADAPTER_MAJOR} "
            f"and minor >= {MIN_SUPPORTED_ADAPTER_MINOR}."
        )


def stub_get_io_quant_meta(module: nn.Module) -> Dict[str, Dict[str, Any]]:
    if not module.is_calibrated():
        raise RuntimeError("QuantGRU not calibrated")

    quant_params = module.quant_params
    if quant_params is None:
        raise RuntimeError("QuantGRU not calibrated")

    bw = getattr(module, "_bitwidth_config", None)
    return {
        "input": _meta_from_quant_params(quant_params, "x_", bw),
        "output": _meta_from_quant_params(quant_params, "h_", bw),
        "hidden": _meta_from_quant_params(quant_params, "h_", bw),
    }


def get_io_quant_meta(module: nn.Module) -> Dict[str, Dict[str, Any]]:
    if _native_on_quant_gru(module, "get_io_quant_meta"):
        return _call_native(module, "get_io_quant_meta")
    return stub_get_io_quant_meta(module)


def stub_aimet_configure(module: nn.Module, mode: str) -> None:
    normalized = str(mode).lower()
    if normalized not in _SUPPORTED_MODES:
        raise ValueError(
            f"Unsupported aimet_configure mode: {mode!r}. "
            f"Expected one of: {', '.join(_SUPPORTED_MODES)}."
        )

    if normalized == "calibrating":
        module.calibrating = True
        module.use_quantization = False
        module.export_mode = False
        return

    module.calibrating = False
    module.export_mode = False
    module.use_quantization = normalized in ("int16_fixed_eval", "int16_fixed_qat_sim")


def aimet_configure(module: nn.Module, mode: str) -> None:
    if _native_on_quant_gru(module, "aimet_configure"):
        return _call_native(module, "aimet_configure", mode)
    stub_aimet_configure(module, mode)


def stub_aimet_capabilities(module: nn.Module) -> Dict[str, Any]:
    del module
    return {
        "adapter_version": "1.0",
        "supported_modes": list(_SUPPORTED_MODES),
        "forward_io_dtype": "float32",
        "forward_io_device": "cuda",
        "requires_calibration_for": ["int16_fixed_eval", "int16_fixed_qat_sim"],
        "supports_forward_quantized": True,
    }


def aimet_capabilities(module: nn.Module) -> Dict[str, Any]:
    if _native_on_quant_gru(module, "aimet_capabilities"):
        return _call_native(module, "aimet_capabilities")
    return stub_aimet_capabilities(module)


def stub_forward_quantized(
    module: nn.Module,
    input: torch.Tensor,
    hx: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Round-trip stub: dequantized ``forward`` → requantize with IO meta.

    TODO(quant-gru-pytorch): replace with bit-exact CUDA integer output.
    """

    if not module.is_calibrated():
        raise RuntimeError("QuantGRU not calibrated")

    was_quant = bool(getattr(module, "use_quantization", False))
    if not was_quant:
        module.use_quantization = True

    try:
        fp_out, fp_hn = nn.Module.forward(module, input, hx)
    finally:
        if not was_quant:
            module.use_quantization = False

    meta = get_io_quant_meta(module)
    int_out = requantize_fp_to_int(fp_out, meta["output"])
    int_hn = requantize_fp_to_int(fp_hn, meta["hidden"])
    return int_out, int_hn


def forward_quantized(
    module: nn.Module,
    input: torch.Tensor,
    hx: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if _native_on_quant_gru(module, "forward_quantized"):
        return _call_native(module, "forward_quantized", input, hx)
    return stub_forward_quantized(module, input, hx)


def _maybe_dequantize_input(
    data: Any,
) -> torch.Tensor:
    if data is None:
        raise TypeError("QuantGRU input must not be None.")
    if is_quantized_activation(data):
        return stop_grad_dequantize(data)
    if isinstance(data, torch.Tensor) and data.is_floating_point():
        return data
    raise TypeError(f"Unsupported QuantGRU input type: {type(data).__name__}.")


def dispatch_quantgru_blackbox(
    module: nn.Module,
    input: torch.Tensor,
    hx: Optional[torch.Tensor] = None,
) -> Tuple[Any, Any]:
    """INT16 black-box dispatch for :class:`QuantizedQuantGRU`."""

    fp_input = _maybe_dequantize_input(input)
    fp_hx = _maybe_dequantize_input(hx) if hx is not None else None

    unlock = getattr(module, "_aimet_unlock_ctx", None)
    has_bound_fwd_q = callable(getattr(module, "forward_quantized", None))

    if unlock is not None and has_bound_fwd_q:
        with unlock():
            aimet_configure(module, get_quant_execution_mode().value)
            int_out, int_hn = module.forward_quantized(fp_input, fp_hx)
    else:
        aimet_configure(module, get_quant_execution_mode().value)
        int_out, int_hn = forward_quantized(module, fp_input, fp_hx)
    meta = get_io_quant_meta(module)
    out_q = wrap_int_tensor_with_meta(int_out, meta["output"])
    hn_q = wrap_int_tensor_with_meta(int_hn, meta["hidden"])

    mode = get_quant_execution_mode()
    if mode is ExecutionMode.INT16_FIXED_QAT_SIM:
        out_fp = out_q.to_float(torch.float32)
        hn_fp = hn_q.to_float(torch.float32)
        publish_int16_carrier(out_fp, out_q)
        publish_int16_carrier(hn_fp, hn_q)
        return out_fp, hn_fp

    return out_q, hn_q

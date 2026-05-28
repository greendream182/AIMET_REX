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
"""INT16 fixed-point fast path for v1 :class:`StaticGridQuantWrapper` (eval / ACTIVE)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

import aimet_torch.fixed_point.kernels  # noqa: F401  # register kernels
from aimet_common.defs import QuantizationDataType
from aimet_torch.custom import custom_tensor_utils
from aimet_torch.fixed_point import (
    ExecutionMode,
    KernelNotFoundError,
    get_fixed_kernel,
    get_quant_execution_mode,
)
from aimet_torch.fixed_point.offline.bias import quantize_bias_int32
from aimet_torch.fixed_point.tensor import Int16QuantizedTensor
from aimet_torch.v1.tensor_quantizer import (
    StaticGridPerChannelQuantizer,
    StaticGridPerTensorQuantizer,
    TensorQuantizer,
)
from aimet_torch.v2.nn.true_quant import _derive_bias_scale
from aimet_torch.v2.quantization.affine.encoding import AffineEncoding
from aimet_torch.v2.quantization.affine.backends import _derive_qmin_qmax
from aimet_torch.v2.quantization.affine.fixed_point import adapter as _fp_v2_adapter


def _stitch_per_channel_tf_encodings(
    ref_shape: torch.Size,
    enc_list: List,
    ch_axis: int,
    device: torch.device,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if ref_shape[ch_axis] != len(enc_list):
        return None
    deltas = torch.tensor(
        [float(e.delta) for e in enc_list], dtype=torch.float32, device=device
    )
    offsets = torch.tensor(
        [float(e.offset) for e in enc_list], dtype=torch.float32, device=device
    )
    shape = [1] * len(ref_shape)
    shape[ch_axis] = len(enc_list)
    return deltas.reshape(shape), offsets.reshape(shape)


def _v1_quantizer_to_affine_encoding(
    ref_tensor: torch.Tensor, quantizer: TensorQuantizer
) -> Optional[AffineEncoding]:
    if quantizer.data_type != QuantizationDataType.int or quantizer.bitwidth == 32:
        return None
    enc = quantizer.encoding
    if enc is None:
        return None

    qmin, qmax = _derive_qmin_qmax(bitwidth=quantizer.bitwidth, signed=True)
    device = ref_tensor.device

    if isinstance(quantizer, StaticGridPerChannelQuantizer):
        if not isinstance(enc, list) or not enc:
            return None
        stitched = _stitch_per_channel_tf_encodings(
            ref_tensor.shape, enc, quantizer.channel_axis, device
        )
        if stitched is None:
            return None
        scale, offset = stitched
    elif isinstance(quantizer, StaticGridPerTensorQuantizer):
        scale = torch.tensor(float(enc.delta), dtype=torch.float32, device=device)
        offset = torch.tensor(float(enc.offset), dtype=torch.float32, device=device)
    else:
        return None

    return AffineEncoding(
        scale,
        offset,
        qmin,
        qmax,
        quantizer.use_symmetric_encodings,
        None,
        None,
    )


def _output_ref_tensor(module: nn.Module, x0: torch.Tensor) -> torch.Tensor:
    """1-D placeholder matching output channel count for per-channel output encodings."""

    if isinstance(module, nn.Linear):
        return torch.zeros(module.out_features, device=x0.device, dtype=torch.float32)
    if isinstance(module, nn.Conv2d):
        return torch.zeros(module.out_channels, device=x0.device, dtype=torch.float32)
    if isinstance(module, nn.Conv1d):
        return torch.zeros(module.out_channels, device=x0.device, dtype=torch.float32)
    return torch.zeros((), device=x0.device, dtype=torch.float32)


def try_static_grid_int16_forward(
    wrapper: Any,
    *inputs: Any,
    **kwargs: Any,
) -> Optional[Int16QuantizedTensor]:
    """Return INT16 carrier output if the fixed-point kernel applies."""

    del kwargs

    mode = get_quant_execution_mode()
    if mode not in (
        ExecutionMode.INT16_FIXED_EVAL,
        ExecutionMode.INT16_FIXED_QAT_SIM,
    ):
        return None

    module = wrapper._module_to_wrap
    base_cls = type(module)
    if base_cls not in (nn.Linear, nn.Conv1d, nn.Conv2d):
        return None

    try:
        kernel = get_fixed_kernel(base_cls)
    except KernelNotFoundError:
        return None

    if not getattr(wrapper, "input_quantizers", None) or not wrapper.input_quantizers:
        return None
    if not getattr(wrapper, "output_quantizers", None) or not wrapper.output_quantizers:
        return None

    iq = wrapper.input_quantizers[0]
    wq = wrapper.param_quantizers.get("weight")
    oq = wrapper.output_quantizers[0]
    if wq is None:
        return None

    if not inputs:
        return None
    x0 = inputs[0]
    if isinstance(x0, Int16QuantizedTensor):
        x_int = x0
        ref_tensor = x0.int_repr
    elif isinstance(x0, torch.Tensor) and x0.is_floating_point():
        ref_tensor = x0
        x_enc = _v1_quantizer_to_affine_encoding(x0, iq)
        if x_enc is None:
            return None
        x_int = Int16QuantizedTensor.from_affine_encoding(x0, x_enc).to(x0.device)
    else:
        return None

    w_float = module.weight
    w_enc = _v1_quantizer_to_affine_encoding(w_float, wq)
    y_enc = _v1_quantizer_to_affine_encoding(_output_ref_tensor(module, ref_tensor), oq)
    if w_enc is None or y_enc is None:
        return None
    w_int = Int16QuantizedTensor.from_affine_encoding(w_float, w_enc).to(ref_tensor.device)

    params: Dict[str, Any] = {"weight": w_int}
    bias = getattr(module, "bias", None)
    if bias is not None:
        acc_scale = _derive_bias_scale(
            x_int.scale, w_enc.scale, bias.shape, channel_axis=0
        )
        if acc_scale is None:
            return None
        ones = torch.ones_like(acc_scale, dtype=acc_scale.dtype, device=acc_scale.device)
        params["bias"] = quantize_bias_int32(bias, acc_scale, ones)

    x_scale = x_int.scale.to(device=ref_tensor.device, dtype=torch.float32)
    w_scale = w_enc.scale.to(device=ref_tensor.device, dtype=torch.float32)
    y_scale = y_enc.scale.to(device=ref_tensor.device, dtype=torch.float32)
    real_m = (x_scale * w_scale) / y_scale
    out_enc = _fp_v2_adapter._affine_output_encoding(y_enc, real_m, ref_tensor.device)
    if base_cls is nn.Linear:
        out_enc = _fp_v2_adapter._broadcast_output_encoding_linear(
            out_enc, w_float.shape[0]
        )
    elif base_cls in (nn.Conv1d, nn.Conv2d):
        out_enc = _fp_v2_adapter._broadcast_output_encoding_conv2d(
            out_enc, w_float.shape[0]
        )

    extra: Dict[str, Any]
    if isinstance(module, nn.Conv2d):
        extra = {
            "stride": module.stride,
            "padding": module.padding,
            "dilation": module.dilation,
            "groups": module.groups,
        }
    elif isinstance(module, nn.Conv1d):
        extra = {
            "stride": module.stride,
            "padding": module.padding,
            "dilation": module.dilation,
            "groups": module.groups,
        }
    else:
        extra = {}

    out = kernel([x_int], params, out_enc, extra)
    return out

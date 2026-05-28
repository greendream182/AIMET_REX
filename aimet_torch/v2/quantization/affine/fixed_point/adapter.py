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
"""Connect AIMET v2 quantized modules to INT16 fixed-point kernels."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from aimet_torch._base.nn.modules import custom
from aimet_torch.fixed_point import (
    ExecutionMode,
    KernelNotFoundError,
    get_fixed_kernel,
    get_quant_execution_mode,
)
from aimet_torch.fixed_point.encoding import InputEncoding, OutputEncoding
from aimet_torch.fixed_point.offline.bias import quantize_bias_int32
from aimet_torch.fixed_point.offline.clz_gen import (
    ClzLutGenerationError,
    clz_activation_name,
    generate_clz_lut_for_export,
    resolve_clz_fit_float_range,
)
from aimet_torch.fixed_point.offline.lut_gen import (
    bake_op_scale_adapter_into_pwl_lut,
    generate_pwl_lut_for_export,
    periodic_lut_fit_spec,
    principal_periodic_input_encoding,
)
from aimet_torch.fixed_point.requantize import _env_truthy
from aimet_torch.fixed_point.offline.multiplier import quantize_multiplier
from aimet_torch.fixed_point.boundary_quantize import quantize_boundary_from_affine
from aimet_torch.fixed_point.qat.carrier import maybe_int16_carrier, publish_int16_carrier
from aimet_torch.fixed_point.tensor import Int16QuantizedTensor
from aimet_torch.v2.quantization.affine.encoding import AffineEncoding
from aimet_torch.v2.quantization.base import QuantizerBase
from aimet_torch.v2.quantization.tensor import QuantizedTensorBase


def _unwrap_float_tensor(data: Any) -> Optional[torch.Tensor]:
    if isinstance(data, QuantizedTensorBase):
        data = data.dequantize()
    if isinstance(data, torch.Tensor) and data.is_floating_point():
        return data
    return None


def _normalize_shape_arg(shape: Any) -> tuple[int, ...]:
    if isinstance(shape, torch.Tensor):
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


def _normalize_pad_tuple(pad: Any) -> tuple[int, ...]:
    if isinstance(pad, int):
        return (int(pad),)
    return tuple(int(x) for x in pad)


def _affine_to_fixed_encoding(enc: AffineEncoding, device: torch.device) -> OutputEncoding:
    return OutputEncoding(
        scale=enc.scale.to(device=device, dtype=torch.float32),
        zero_point=(-enc.offset).round().to(torch.int32).to(device=device),
        qmin=enc.qmin,
        qmax=enc.qmax,
        axis=None,
    )


def _affine_output_encoding(
    y_enc: AffineEncoding,
    real_multiplier: torch.Tensor,
    device: torch.device,
) -> OutputEncoding:
    mult, rsh = quantize_multiplier(real_multiplier.detach())
    return OutputEncoding(
        scale=y_enc.scale.to(device=device, dtype=torch.float32),
        zero_point=(-y_enc.offset).round().to(torch.int32).to(device=device),
        qmin=y_enc.qmin,
        qmax=y_enc.qmax,
        multiplier=mult.to(device=device),
        rshift=rsh.to(device=device),
        axis=None,
    )


def _is_weighted_module(base_cls: type) -> bool:
    return base_cls in (nn.Linear, nn.Conv1d, nn.Conv2d)


def _resolve_adaptive_avg_pool2d_output_size(
    args: tuple,
    kwargs: dict,
) -> Optional[Tuple[int, int]]:
    """Extract ``output_size`` from ``F.adaptive_avg_pool2d(input, output_size)``.

    ``custom.AdaptiveAvgPool2d`` is a thin wrapper around the functional; callers
    may pass ``output_size`` positionally or as a kwarg.
    """

    output_size = kwargs.get("output_size", None)
    if output_size is None and len(args) >= 2:
        output_size = args[1]
    if output_size is None:
        return None
    if isinstance(output_size, int):
        return int(output_size), int(output_size)
    if isinstance(output_size, (list, tuple)) and len(output_size) == 2:
        return int(output_size[0]), int(output_size[1])
    return None


def _resolve_mean_dim_keepdim(
    args: tuple,
    kwargs: dict,
    input_shape: torch.Size,
) -> tuple:
    """Extract ``dim`` and ``keepdim`` from the forward args of ``torch.mean``.

    ``custom.Mean`` is a thin wrapper around ``torch.mean``; users may call it
    as ``module(x)`` (full reduce), ``module(x, dim)``, ``module(x, dim, keepdim)``
    or via keyword arguments.

    Returns ``(dim, keepdim)`` where ``dim`` may be ``None`` (full reduce),
    an ``int`` or a tuple of ``int``.
    """
    del input_shape  # only used by caller for size computation
    dim = kwargs.get("dim", None)
    keepdim = kwargs.get("keepdim", False)
    extra_pos = args[1:]
    if extra_pos:
        dim = extra_pos[0]
        if len(extra_pos) >= 2:
            keepdim = extra_pos[1]
    if isinstance(dim, list):
        dim = tuple(dim)
    return dim, bool(keepdim)


def _pwl_activation_fn(module: nn.Module, base_cls: type):
    if base_cls is nn.Sigmoid:
        return torch.sigmoid
    if base_cls is nn.Tanh:
        return torch.tanh
    if base_cls is nn.GELU:
        return lambda x: F.gelu(x, approximate=getattr(module, "approximate", "none"))
    if base_cls is nn.SiLU:
        return F.silu
    if base_cls is nn.Mish:
        return F.mish
    if base_cls is nn.Softplus:
        return lambda x: F.softplus(
            x,
            beta=getattr(module, "beta", 1),
            threshold=getattr(module, "threshold", 20),
        )
    if base_cls is nn.Hardsigmoid:
        return F.hardsigmoid
    if base_cls is nn.Hardswish:
        return F.hardswish
    if base_cls is nn.LeakyReLU:
        return lambda x: F.leaky_relu(
            x,
            negative_slope=getattr(module, "negative_slope", 0.01),
            inplace=False,
        )
    if base_cls is nn.PReLU:
        weight = getattr(module, "weight", None)
        if weight is None or weight.numel() != 1:
            return None
        negative_slope = float(weight.detach().flatten()[0].item())
        return lambda x: torch.where(x >= 0, x, x * negative_slope)
    if base_cls is custom.Exponential:
        return torch.exp
    if base_cls is custom.Log:
        return torch.log
    return None


def _qat_surrogate_float(
    qmodule: nn.Module,
    base_cls: type,
    surrogate_inputs: list[torch.Tensor],
    params: Dict[str, Any],
    extra: Dict[str, Any],
) -> Optional[torch.Tensor]:
    if _is_weighted_module(base_cls):
        bias = getattr(qmodule, "bias", None)
        if base_cls is nn.Linear:
            return F.linear(surrogate_inputs[0], qmodule.weight, bias)
        if base_cls is nn.Conv2d:
            return F.conv2d(
                surrogate_inputs[0],
                qmodule.weight,
                bias,
                stride=extra["stride"],
                padding=extra["padding"],
                dilation=extra["dilation"],
                groups=extra["groups"],
            )
        if base_cls is nn.Conv1d:
            return F.conv1d(
                surrogate_inputs[0],
                qmodule.weight,
                bias,
                stride=extra["stride"],
                padding=extra["padding"],
                dilation=extra["dilation"],
                groups=extra["groups"],
            )
    if base_cls is custom.Add and len(surrogate_inputs) == 2:
        return surrogate_inputs[0] + surrogate_inputs[1]
    if base_cls is custom.Subtract and len(surrogate_inputs) == 2:
        return surrogate_inputs[0] - surrogate_inputs[1]
    if base_cls is custom.Multiply and len(surrogate_inputs) == 2:
        return surrogate_inputs[0] * surrogate_inputs[1]
    if base_cls is nn.ReLU:
        return F.relu(surrogate_inputs[0])
    if base_cls is nn.ReLU6:
        return F.relu6(surrogate_inputs[0])
    if base_cls is nn.Hardtanh:
        return F.hardtanh(surrogate_inputs[0], qmodule.min_val, qmodule.max_val)
    if base_cls is nn.MaxPool2d:
        return F.max_pool2d(
            surrogate_inputs[0],
            kernel_size=extra["kernel_size"],
            stride=extra["stride"],
            padding=extra["padding"],
            dilation=extra["dilation"],
            ceil_mode=extra["ceil_mode"],
        )
    if base_cls is nn.AvgPool2d:
        return F.avg_pool2d(
            surrogate_inputs[0],
            kernel_size=extra["kernel_size"],
            stride=extra["stride"],
            padding=extra["padding"],
            ceil_mode=extra["ceil_mode"],
        )
    if base_cls is nn.Flatten:
        return torch.flatten(surrogate_inputs[0], extra["start_dim"], extra["end_dim"])
    if base_cls is custom.Reshape:
        return torch.reshape(surrogate_inputs[0], extra["shape"])
    if base_cls is custom.Pad:
        return F.pad(
            surrogate_inputs[0],
            extra["pad"],
            mode=extra.get("mode", "constant"),
            value=extra.get("value", 0.0),
        )
    if base_cls in (custom.Clamp, custom.Clip):
        min_v = extra.get("min")
        max_v = extra.get("max")
        return torch.clamp(surrogate_inputs[0], min=min_v, max=max_v)
    if base_cls is custom.Mean:
        return surrogate_inputs[0].mean(
            dim=extra.get("dim"),
            keepdim=bool(extra.get("keepdim", False)),
        )
    if base_cls is custom.AdaptiveAvgPool2d:
        return F.adaptive_avg_pool2d(
            surrogate_inputs[0],
            output_size=extra.get("output_size", (1, 1)),
        )
    if base_cls is nn.Dropout:
        return surrogate_inputs[0]
    if base_cls is custom.Sin:
        return torch.sin(surrogate_inputs[0])
    if base_cls is custom.Cos:
        return torch.cos(surrogate_inputs[0])
    if base_cls is custom.Square:
        return torch.square(surrogate_inputs[0])
    pwl_fn = _pwl_activation_fn(qmodule, base_cls)
    if pwl_fn is not None:
        return pwl_fn(surrogate_inputs[0])
    del params
    return None


def _broadcast_output_encoding_linear(enc: OutputEncoding, out_features: int) -> OutputEncoding:
    """Make per-output-channel fields broadcast with Linear accumulator shape (N, out)."""

    mult = enc.multiplier
    if mult is None or mult.numel() in (0, 1) or mult.numel() != out_features:
        return enc
    rsh = enc.rshift
    if rsh is None:
        return enc
    scale = enc.scale
    zp = enc.zero_point
    if scale.numel() == out_features:
        scale = scale.reshape(1, -1)
    if zp.numel() == out_features:
        zp = zp.reshape(1, -1)
    return OutputEncoding(
        scale=scale,
        zero_point=zp,
        qmin=enc.qmin,
        qmax=enc.qmax,
        multiplier=mult.reshape(1, -1),
        rshift=rsh.reshape(1, -1),
        axis=enc.axis,
    )


def _broadcast_output_encoding_conv2d(enc: OutputEncoding, out_channels: int) -> OutputEncoding:
    """Make per-output-channel fields broadcast with Conv2d accumulator (N, C, H, W)."""

    mult = enc.multiplier
    if mult is None or mult.numel() in (0, 1) or mult.numel() != out_channels:
        return enc
    rsh = enc.rshift
    if rsh is None:
        return enc
    scale = enc.scale
    zp = enc.zero_point
    if scale.numel() == out_channels:
        scale = scale.reshape(1, out_channels, 1, 1)
    if zp.numel() == out_channels:
        zp = zp.reshape(1, out_channels, 1, 1)
    return OutputEncoding(
        scale=scale,
        zero_point=zp,
        qmin=enc.qmin,
        qmax=enc.qmax,
        multiplier=mult.reshape(1, out_channels, 1, 1),
        rshift=rsh.reshape(1, out_channels, 1, 1),
        axis=enc.axis,
    )


def dispatch_int16_fixed(qmodule: nn.Module, *args, **kwargs) -> Optional[Union[Int16QuantizedTensor, torch.Tensor]]:
    """Run the module on INT16 fixed-point kernels when execution mode allows.

    In :class:`~aimet_torch.fixed_point.execution_mode.ExecutionMode.INT16_FIXED_EVAL`, returns
    an :class:`~aimet_torch.fixed_point.tensor.Int16QuantizedTensor` (debug comparison should use
    :meth:`~aimet_torch.fixed_point.tensor.Int16QuantizedTensor.to_float` outside the INT16 mode
    context, or wrap with :class:`~aimet_torch.fixed_point.metrics.profiler.FixedPointProfiler`).

    In :class:`~aimet_torch.fixed_point.execution_mode.ExecutionMode.INT16_FIXED_QAT_SIM`, returns
    a ``float32`` tensor using fixed-point values with a surrogate gradient (see surrogate wiring in
    this module).

    Returns ``None`` when the module cannot dispatch to INT16 fixed-point.
    """

    mode = get_quant_execution_mode()
    if mode not in (
        ExecutionMode.INT16_FIXED_EVAL,
        ExecutionMode.INT16_FIXED_QAT_SIM,
    ):
        return None

    # pylint: disable=import-outside-toplevel
    from aimet_torch.v2.nn.true_quant import QuantizationMixin, _is_computing_encodings

    if _is_computing_encodings(qmodule):
        return None

    # QuantGRU black-box: bypass generic kernel registry (contract v1).
    qmodule_cls_name = type(qmodule).__name__
    if qmodule_cls_name == "QuantizedQuantGRU":
        from aimet_torch.fixed_point.quantgru_adapter import dispatch_quantgru_blackbox

        return dispatch_quantgru_blackbox(qmodule, *args, **kwargs)

    base_cls = QuantizationMixin.qcls_to_cls.get(type(qmodule))
    if base_cls is None:
        # Fake-quant wrappers (e.g. ``FakeQuantizedAdaptiveAvgPool2d`` from
        # ``_legacy_impl.py``) are not registered with ``QuantizationMixin``;
        # they share the same dispatch contract once we resolve the parent op.
        from aimet_torch.v2.nn.fake_quant._legacy_impl import FakeQuantizationMixin
        base_cls = FakeQuantizationMixin.qcls_to_cls.get(type(qmodule))
    if base_cls is None:
        return None

    # Shape/layout metadata (e.g. b*f from tensor.shape for view) is not a
    # quantized activation segment; evaluate with plain scalar semantics.
    from aimet_torch.fixed_point.shape_meta import try_dispatch_shape_meta_op

    is_meta, meta_result = try_dispatch_shape_meta_op(base_cls, *args, **kwargs)
    if is_meta:
        return meta_result

    try:
        kernel = get_fixed_kernel(base_cls)
    except KernelNotFoundError:
        return None

    if not args:
        return None

    first_arg = args[0]
    first_float = _unwrap_float_tensor(first_arg)
    if isinstance(first_arg, Int16QuantizedTensor):
        device = first_arg.int_repr.device
    elif first_float is not None:
        device = first_float.device
    else:
        return None

    pq = getattr(qmodule, "param_quantizers", None)
    wq = pq["weight"] if pq is not None and "weight" in pq else None
    oq = qmodule.output_quantizers[0] if qmodule.output_quantizers else None

    inputs_int = []
    x_encodings = []
    surrogate_inputs = []
    collect_surrogate = mode is ExecutionMode.INT16_FIXED_QAT_SIM
    if base_cls is custom.Concat:
        input_args = args
    else:
        input_args = args[: len(qmodule.input_quantizers)]

    for index, arg in enumerate(input_args):
        if isinstance(arg, Int16QuantizedTensor):
            inputs_int.append(arg)
            x_encodings.append(None)
            if collect_surrogate:
                surrogate_inputs.append(arg.to_float())
            continue
        input_tensor = _unwrap_float_tensor(arg)
        quant_index = 0 if base_cls is custom.Concat else index
        iq = qmodule.input_quantizers[quant_index] if qmodule.input_quantizers else None
        if input_tensor is None:
            return None
        if isinstance(iq, QuantizerBase) and iq.is_initialized():
            x_enc = iq.get_encodings()
            if not isinstance(x_enc, AffineEncoding):
                return None
            inputs_int.append(quantize_boundary_from_affine(input_tensor, x_enc).to(device))
            x_encodings.append(x_enc)
            if collect_surrogate:
                surrogate_inputs.append(input_tensor)
            continue
        carrier = maybe_int16_carrier(arg)
        if carrier is not None:
            inputs_int.append(carrier.to(device))
            x_encodings.append(None)
            if collect_surrogate:
                surrogate_inputs.append(input_tensor)
            continue
        if not isinstance(iq, QuantizerBase) or not iq.is_initialized():
            # Super-groups disable iq on the trailing op (e.g. ReLU after Conv).
            # INT16_FIXED_EVAL still works because the predecessor emits
            # Int16QuantizedTensor. In QAT the predecessor publishes an INT16
            # carrier on its float STE output (see fixed_point.qat.carrier).
            if collect_surrogate:
                surrogate_inputs.append(input_tensor)
                continue
            return None

    if inputs_int and len(inputs_int) != len(input_args):
        inputs_int = []
        x_encodings = []

    if not inputs_int:
        if collect_surrogate and surrogate_inputs:
            qat_extra: Dict[str, Any] = {}
            if isinstance(qmodule, nn.Conv2d):
                qat_extra.update(
                    {
                        "stride": qmodule.stride,
                        "padding": qmodule.padding,
                        "dilation": qmodule.dilation,
                        "groups": qmodule.groups,
                    }
                )
            elif isinstance(qmodule, nn.Conv1d):
                qat_extra.update(
                    {
                        "stride": qmodule.stride,
                        "padding": qmodule.padding,
                        "dilation": qmodule.dilation,
                        "groups": qmodule.groups,
                    }
                )
            elif isinstance(qmodule, nn.Flatten):
                qat_extra.update({"start_dim": qmodule.start_dim, "end_dim": qmodule.end_dim})
            elif base_cls is custom.Reshape and len(args) > 1:
                qat_extra["shape"] = _normalize_shape_arg(args[1])
            elif base_cls is custom.Pad and len(args) > 1:
                pad = args[1]
                qat_extra["pad"] = pad if isinstance(pad, tuple) else tuple(pad)
                qat_extra["mode"] = kwargs.get("mode", "constant")
                pad_value = kwargs.get("value", 0.0)
                qat_extra["value"] = 0.0 if pad_value is None else float(pad_value)
            elif base_cls is custom.Mean:
                mean_dim, mean_keepdim = _resolve_mean_dim_keepdim(
                    args, kwargs, first_float.shape if first_float is not None else ()
                )
                qat_extra.update({"dim": mean_dim, "keepdim": mean_keepdim})
            elif base_cls is custom.AdaptiveAvgPool2d:
                output_size = _resolve_adaptive_avg_pool2d_output_size(args, kwargs)
                qat_extra.update({"output_size": output_size or (1, 1)})
            elif isinstance(qmodule, (nn.MaxPool2d, nn.AvgPool2d)):
                qat_extra.update(
                    {
                        "kernel_size": qmodule.kernel_size,
                        "stride": qmodule.stride,
                        "padding": qmodule.padding,
                        "dilation": getattr(qmodule, "dilation", 1),
                        "ceil_mode": qmodule.ceil_mode,
                    }
                )
            surrogate = _qat_surrogate_float(
                qmodule, base_cls, surrogate_inputs, {}, qat_extra
            )
            if surrogate is not None:
                if isinstance(oq, QuantizerBase) and oq.is_initialized():
                    return oq(surrogate)
                return surrogate
        return None
    x_int = inputs_int[0]

    if _is_weighted_module(base_cls) and (
        not isinstance(wq, QuantizerBase) or not wq.is_initialized()
    ):
        return None
    if not isinstance(oq, QuantizerBase) or not oq.is_initialized():
        return None

    w_enc = wq.get_encodings() if wq is not None else None
    y_enc = oq.get_encodings()
    if _is_weighted_module(base_cls) and not isinstance(w_enc, AffineEncoding):
        return None
    if not isinstance(y_enc, AffineEncoding):
        return None

    x_scale = x_int.scale.to(device=device, dtype=torch.float32)

    params: Dict[str, Any] = {}

    if _is_weighted_module(base_cls):
        w_float = qmodule.weight
        w_int = quantize_boundary_from_affine(w_float, w_enc).to(device)
        params["weight"] = w_int
        bias = getattr(qmodule, "bias", None)
        if bias is not None:
            if hasattr(qmodule, "_derive_bias_scale"):
                acc_scale = qmodule._derive_bias_scale(x_scale, w_enc.scale)
                if acc_scale is None:
                    return None
                ones = torch.ones_like(acc_scale, dtype=acc_scale.dtype, device=acc_scale.device)
                params["bias"] = quantize_bias_int32(bias, acc_scale, ones)
            else:
                params["bias"] = quantize_bias_int32(bias, x_scale, w_enc.scale)

        w_scale = w_enc.scale.to(device=device, dtype=torch.float32)
        y_scale = y_enc.scale.to(device=device, dtype=torch.float32)
        real_m = (x_scale * w_scale) / y_scale
        out_enc = _affine_output_encoding(y_enc, real_m, device)
        if base_cls is nn.Linear:
            out_enc = _broadcast_output_encoding_linear(out_enc, w_float.shape[0])
        elif base_cls in (nn.Conv1d, nn.Conv2d):
            # Conv1d kernel delegates to Conv2d with an extra spatial dim; (1,C,1,1) still broadcasts.
            out_enc = _broadcast_output_encoding_conv2d(out_enc, w_float.shape[0])
    else:
        y_scale = y_enc.scale.to(device=device, dtype=torch.float32)
        if base_cls is custom.Multiply and len(inputs_int) == 2:
            real_m = (
                inputs_int[0].scale.to(device=device, dtype=torch.float32)
                * inputs_int[1].scale.to(device=device, dtype=torch.float32)
            ) / y_scale
        elif base_cls is custom.MatMul and len(inputs_int) == 2:
            real_m = (
                inputs_int[0].scale.to(device=device, dtype=torch.float32)
                * inputs_int[1].scale.to(device=device, dtype=torch.float32)
            ) / y_scale
        elif base_cls is nn.AvgPool2d:
            kernel_size = qmodule.kernel_size
            if isinstance(kernel_size, tuple):
                kernel_area = int(kernel_size[0]) * int(kernel_size[1])
            else:
                kernel_area = int(kernel_size) * int(kernel_size)
            real_m = x_scale / (float(kernel_area) * y_scale)
        elif base_cls is custom.Mean:
            # ``torch.mean(x, dim, keepdim=...)``: forward args[1:] / kwargs carry dim/keepdim.
            mean_dim, mean_keepdim = _resolve_mean_dim_keepdim(args, kwargs, x_int.int_repr.shape)
            if mean_dim is None:
                reduce_size = int(x_int.int_repr.numel())
                dims_resolved: tuple[int, ...] = tuple(range(x_int.int_repr.dim()))
            else:
                if isinstance(mean_dim, int):
                    dims_resolved = (int(mean_dim) % x_int.int_repr.dim(),)
                else:
                    dims_resolved = tuple(int(d) % x_int.int_repr.dim() for d in mean_dim)
                reduce_size = 1
                for d in dims_resolved:
                    reduce_size *= int(x_int.int_repr.shape[d])
            if reduce_size <= 0:
                return None
            real_m = x_scale / (float(reduce_size) * y_scale)
        elif base_cls is custom.AdaptiveAvgPool2d:
            # Only ``output_size=(1,1)`` reduces to a single mean over spatial dims,
            # which matches MobileNet-V2 GAP. Non-(1,1) requires per-block grouping
            # that is not yet implemented; surface the gap by refusing dispatch.
            output_size = _resolve_adaptive_avg_pool2d_output_size(args, kwargs)
            if output_size != (1, 1):
                return None
            shape = x_int.int_repr.shape
            if x_int.int_repr.dim() != 4:
                return None
            reduce_size = int(shape[2]) * int(shape[3])
            if reduce_size <= 0:
                return None
            real_m = x_scale / (float(reduce_size) * y_scale)
        else:
            real_m = x_scale / y_scale
        out_enc = _affine_output_encoding(y_enc, real_m, device)

    extra: Dict[str, Any]
    if isinstance(qmodule, nn.Conv2d):
        extra = {
            "stride": qmodule.stride,
            "padding": qmodule.padding,
            "dilation": qmodule.dilation,
            "groups": qmodule.groups,
        }
    elif isinstance(qmodule, nn.Conv1d):
        extra = {
            "stride": qmodule.stride,
            "padding": qmodule.padding,
            "dilation": qmodule.dilation,
            "groups": qmodule.groups,
        }
    else:
        extra = {}
        if isinstance(qmodule, nn.Flatten):
            extra.update({"start_dim": qmodule.start_dim, "end_dim": qmodule.end_dim})
        elif base_cls is custom.Reshape and len(args) > 1:
            extra["shape"] = _normalize_shape_arg(args[1])
        elif base_cls is custom.Pad and len(args) > 1:
            pad = args[1]
            extra["pad"] = _normalize_pad_tuple(pad)
            extra["mode"] = kwargs.get("mode", "constant")
            pad_value = kwargs.get("value", 0.0)
            extra["value"] = 0.0 if pad_value is None else float(pad_value)
        elif isinstance(qmodule, custom.Concat):
            extra["axis"] = getattr(qmodule, "_axis", getattr(qmodule, "axis", 0))
        elif isinstance(qmodule, (nn.MaxPool2d, nn.AvgPool2d)):
            extra.update(
                {
                    "kernel_size": qmodule.kernel_size,
                    "stride": qmodule.stride,
                    "padding": qmodule.padding,
                    "dilation": getattr(qmodule, "dilation", 1),
                    "ceil_mode": qmodule.ceil_mode,
                }
            )
        elif isinstance(qmodule, nn.Hardtanh):
            scale = x_int.scale.to(device=device, dtype=torch.float32)
            zp = x_int.zero_point.to(device=device, dtype=torch.int32)
            extra.update(
                {
                    "min_int": int(torch.round(torch.tensor(qmodule.min_val, device=device) / scale + zp).item()),
                    "max_int": int(torch.round(torch.tensor(qmodule.max_val, device=device) / scale + zp).item()),
                }
            )
        elif base_cls in (custom.Clamp, custom.Clip):
            scale = x_int.scale.to(device=device, dtype=torch.float32)
            zp = x_int.zero_point.to(device=device, dtype=torch.int32)
            min_v = kwargs.get("min", args[1] if len(args) > 1 else None)
            max_v = kwargs.get("max", args[2] if len(args) > 2 else None)
            extra["min"] = min_v
            extra["max"] = max_v
            if min_v is not None:
                extra["min_int"] = int(torch.round(torch.tensor(float(min_v), device=device) / scale + zp).item())
            else:
                extra["min_int"] = int(x_int.qmin)
            if max_v is not None:
                extra["max_int"] = int(torch.round(torch.tensor(float(max_v), device=device) / scale + zp).item())
            else:
                extra["max_int"] = int(x_int.qmax)
        elif base_cls is custom.Mean:
            mean_dim, mean_keepdim = _resolve_mean_dim_keepdim(args, kwargs, x_int.int_repr.shape)
            extra.update({"dim": mean_dim, "keepdim": mean_keepdim})
        elif base_cls is custom.AdaptiveAvgPool2d:
            # ``output_size=(1,1)`` is the only path that reaches dispatch (see above);
            # the Mean kernel reduces the spatial dims with ``keepdim=True``.
            extra.update({"dim": (2, 3), "keepdim": True, "output_size": (1, 1)})
        elif base_cls is nn.Softmax:
            extra["dim"] = getattr(qmodule, "dim", None)
            if extra["dim"] is None:
                extra["dim"] = kwargs.get("dim", -1)

    from aimet_torch.fixed_point.export.sidecar_loader import (  # noqa: WPS433
        get_int16_sidecar_extra,
    )

    sidecar_extra = get_int16_sidecar_extra(qmodule)
    if sidecar_extra:
        extra.update(sidecar_extra)

    clz_name = clz_activation_name(base_cls)
    periodic_fit_fn, phase_fold = periodic_lut_fit_spec(base_cls)
    pwl_fn = _pwl_activation_fn(qmodule, base_cls) if clz_name is None else None
    if pwl_fn is None and periodic_fit_fn is not None:
        pwl_fn = periodic_fit_fn
    needs_lut_enc = pwl_fn is not None or (
        clz_name is not None and extra.get("clz_lut") is None
    )
    if needs_lut_enc:
        first_x_enc = x_encodings[0]
        lut_input_enc = (
            _affine_to_fixed_encoding(first_x_enc, device)
            if first_x_enc is not None
            else OutputEncoding(
                scale=x_int.scale.to(device=device, dtype=torch.float32),
                zero_point=x_int.zero_point.to(device=device, dtype=torch.int32),
                qmin=x_int.qmin,
                qmax=x_int.qmax,
            )
        )
        if clz_name is not None and extra.get("clz_lut") is None:
            fit_min, fit_max = resolve_clz_fit_float_range(
                qmodule, clz_name, lut_input_enc
            )
            try:
                clz_body, clz_metrics = generate_clz_lut_for_export(
                    clz_name,
                    lut_input_enc,
                    out_enc,
                    fit_float_min=fit_min,
                    fit_float_max=fit_max,
                )
                extra["clz_lut"] = clz_body
                extra["clz_func_name"] = clz_name
                extra["clz_lut_metrics"] = clz_metrics
            except ClzLutGenerationError as exc:
                import os

                if os.environ.get("AIMET_RX_REQUIRE_CLZ_LUT", "").strip() in (
                    "1",
                    "true",
                    "yes",
                ):
                    raise
                extra.setdefault("clz_lut_error", str(exc))
        if clz_name is not None and extra.get("clz_lut") is None:
            import os

            if os.environ.get("AIMET_RX_REQUIRE_CLZ_LUT", "").strip() in (
                "1",
                "true",
                "yes",
            ):
                raise ClzLutGenerationError(
                    extra.get("clz_lut_error", "CLZ LUT generation failed.")
                )
            return None
        if pwl_fn is not None and extra.get("pwl_lut") is None:
            fit_enc = (
                principal_periodic_input_encoding(lut_input_enc)
                if phase_fold is not None
                else lut_input_enc
            )
            extra["pwl_lut"], _, _metrics = generate_pwl_lut_for_export(
                pwl_fn,
                fit_enc,
                out_enc,
                fn_name=phase_fold or getattr(pwl_fn, "__name__", None),
            )
            extra["pwl_input_encoding"] = fit_enc
            if phase_fold is not None:
                extra["phase_fold"] = phase_fold
            if _env_truthy("AIMET_RX_BAKE_PWL_SCALE"):
                op_enc = InputEncoding(
                    scale=x_int.scale,
                    zero_point=x_int.zero_point,
                    qmin=x_int.qmin,
                    qmax=x_int.qmax,
                    axis=x_int.axis,
                )
                extra["pwl_lut"] = bake_op_scale_adapter_into_pwl_lut(
                    extra["pwl_lut"], op_enc, fit_enc
                )

    out = kernel(inputs_int, params, out_enc, extra)
    if mode == ExecutionMode.INT16_FIXED_QAT_SIM:
        surrogate = _qat_surrogate_float(qmodule, base_cls, surrogate_inputs, params, extra)
        if surrogate is None:
            float_out = out.to_float(torch.float32)
            publish_int16_carrier(float_out, out)
            return float_out
        fixed_float = out.to_float(torch.float32)
        ste_out = fixed_float + (surrogate - surrogate.detach())
        publish_int16_carrier(ste_out, out)
        return ste_out
    return out

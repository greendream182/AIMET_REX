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

import contextlib
import dataclasses
import os
from typing import Any, Dict, List, Optional, Tuple, Union

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
from aimet_torch.fixed_point.capabilities import (
    KernelKind,
    assert_requantizing_combo_supported,
    get_capability,
    requires_activation_bitwidth_gate,
)
from aimet_torch.fixed_point.encoding import InputEncoding, OutputEncoding
from aimet_torch.fixed_point.offline.bias import quantize_bias_int
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


def _resolve_bias_bits(qmodule: Any) -> int:
    """Pick bias storage bit-width for Conv/Linear (explicit_config).

    Default is 32 (legacy/simulator). A model author opts into the spec-04_01
    canonical 16-bit hardware bias by setting ``qmodule._fp_bias_bits = 16``
    on the quantized module. Any other value raises so silent typos surface.
    """

    bits = getattr(qmodule, "_fp_bias_bits", 32)
    if bits not in (16, 32):
        raise ValueError(
            f"_fp_bias_bits must be 16 or 32; got {bits!r} on {type(qmodule).__name__}."
        )
    return int(bits)


def _maxpool_encodings_match(
    x_int: Int16QuantizedTensor,
    y_enc: AffineEncoding,
) -> bool:
    """Return True iff MaxPool input/output share the same quant grid.

    Per ``doc/04_算子详细规格/04_09_池化类算子.md`` Max-pooling: hardware is a
    comparator tree only (no ``M/rshift``); input and output must share
    ``scale``/``zero_point``/``qmin``/``qmax``. Numerical equality is used
    here (not Python identity), so two distinct quantizer instances that
    converge to the same grid still match.
    """

    if x_int.qmin != y_enc.qmin or x_int.qmax != y_enc.qmax:
        return False
    in_scale = x_int.scale.detach().reshape(-1)
    out_scale = y_enc.scale.detach().to(
        device=in_scale.device, dtype=in_scale.dtype
    ).reshape(-1)
    if in_scale.shape != out_scale.shape or not torch.equal(in_scale, out_scale):
        return False
    in_zp = x_int.zero_point.detach().to(torch.int32).reshape(-1)
    out_zp = (-y_enc.offset).detach().round().to(
        device=in_zp.device, dtype=torch.int32
    ).reshape(-1)
    if in_zp.shape != out_zp.shape or not torch.equal(in_zp, out_zp):
        return False
    return True


# INT16 kernels that only need output scale/zp/qmin/qmax (no MAC ``M,rshift``).
_UNARY_GRID_ONLY_OUTPUT_OPS = frozenset(
    {
        custom.ElementwiseUnarySign,
    }
)


def _output_encoding_from_scales(
    y_enc: AffineEncoding,
    real_multiplier: torch.Tensor,
    device: torch.device,
    *,
    base_cls: type,
) -> OutputEncoding:
    """Build :class:`OutputEncoding`; skip ``quantize_multiplier`` when not needed."""

    if base_cls in _UNARY_GRID_ONLY_OUTPUT_OPS:
        return _affine_to_fixed_encoding(y_enc, device)
    if not torch.isfinite(real_multiplier).all():
        return _affine_to_fixed_encoding(y_enc, device)
    return _affine_output_encoding(y_enc, real_multiplier, device)


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


def _compute_mean_reduce_size_and_dims(
    mean_dim, input_shape: torch.Size
) -> tuple:
    """Resolve ``(reduce_size, dims_resolved)`` for ``torch.mean(..., dim)``.

    Single source of truth shared between the ``real_m = S_x / (N * S_y)``
    fold (which feeds output ``M/rshift``) and the ``extra['reduce_size']``
    contract written into the kernel ``extra``. Keeping these two values in
    physical sync — instead of computing them in two parallel branches —
    avoids drift bugs that would otherwise be surfaced only at runtime by
    :func:`require_reduce_size_matches_extra`.

    ``mean_dim=None`` means the full reduce; otherwise ``mean_dim`` may be
    an ``int`` or an iterable of ``int`` and is normalised against
    ``input_shape``'s rank with the standard ``% ndim`` wrap.
    """

    ndim = len(input_shape)
    if mean_dim is None:
        # ``ndim == 0`` (rank-0 scalar) would make later ``% ndim`` blow up;
        # there's nothing to reduce in that case so return the no-op pair and
        # let the caller's ``reduce_size <= 0`` guard refuse dispatch.
        if ndim == 0:
            return 0, ()
        return int(torch.Size(input_shape).numel()), tuple(range(ndim))
    if ndim == 0:
        return 0, ()
    if isinstance(mean_dim, int):
        dims_resolved: tuple = (int(mean_dim) % ndim,)
    else:
        dims_resolved = tuple(int(d) % ndim for d in mean_dim)
    reduce_size = 1
    for d in dims_resolved:
        reduce_size *= int(input_shape[d])
    if reduce_size <= 0:
        # Empty / 0-length axis: the 1/N fold is undefined; signal "no go" to
        # the caller so it can ``return None`` and fall back to fp32 QDQ.
        return 0, dims_resolved
    return reduce_size, dims_resolved


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
    if base_cls is custom.Abs:
        return torch.abs
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
    if base_cls is custom.MatMul and len(surrogate_inputs) == 2:
        return torch.matmul(surrogate_inputs[0], surrogate_inputs[1])
    if base_cls is custom.Divide and len(surrogate_inputs) == 2:
        num, den = surrogate_inputs
        eps = float(extra.get("eps", 1e-12))
        return num / torch.where(
            den.abs() < eps,
            eps * den.sign().clamp(min=1.0),
            den,
        )
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
    if base_cls in (nn.Upsample, nn.UpsamplingNearest2d):
        # Surrogate runs in the value space (fp32 fake-quant), so ``F.interpolate``
        # accepts the tensor directly. Mode/size/scale_factor were captured at
        # the ``extra``-build site below.
        return F.interpolate(
            surrogate_inputs[0],
            size=extra.get("size"),
            scale_factor=extra.get("scale_factor"),
            mode=str(extra.get("mode", "nearest")),
        )
    if base_cls is nn.LayerNorm:
        # LayerNorm surrogate is symmetric with the float-reference kernel
        # in ``norm.LayerNormInt16Kernel`` — both call ``F.layer_norm`` on
        # the fp32 value-space input. ``normalized_shape`` / ``eps`` /
        # ``weight`` / ``bias`` come from ``extra`` (captured below at the
        # extra-build site) so this branch matches the kernel byte-for-byte
        # when fed equivalent inputs.
        return F.layer_norm(
            surrogate_inputs[0],
            normalized_shape=extra["normalized_shape"],
            weight=extra.get("weight"),
            bias=extra.get("bias"),
            eps=float(extra.get("eps", 1e-5)),
        )
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
    if base_cls is custom.Sqrt:
        return torch.sqrt(surrogate_inputs[0].clamp_min(0.0))
    if base_cls is custom.RSqrt:
        return torch.rsqrt(surrogate_inputs[0].clamp_min(0.0))
    if base_cls is custom.ElementwiseUnarySign:
        return torch.sign(surrogate_inputs[0])
    pwl_fn = _pwl_activation_fn(qmodule, base_cls)
    if pwl_fn is not None:
        return pwl_fn(surrogate_inputs[0])
    del params
    return None


def _qat_surrogate_checkpoint_enabled() -> bool:
    """Activation-checkpoint surrogate ops during INT16 QAT when explicitly enabled."""

    if not torch.is_grad_enabled():
        return False
    return os.environ.get("AIMET_RX_QAT_SURROGATE_CHECKPOINT", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _run_qat_surrogate(
    qmodule: nn.Module,
    base_cls: type,
    surrogate_inputs: list[torch.Tensor],
    params: Dict[str, Any],
    extra: Dict[str, Any],
) -> Optional[torch.Tensor]:
    """Run float surrogate for STE; optionally checkpoint to save activation memory."""

    if not surrogate_inputs:
        return None
    if not _qat_surrogate_checkpoint_enabled():
        return _qat_surrogate_float(qmodule, base_cls, surrogate_inputs, params, extra)

    def _checkpointed(*flat_inputs: torch.Tensor) -> torch.Tensor:
        result = _qat_surrogate_float(qmodule, base_cls, list(flat_inputs), params, extra)
        if result is None:
            raise RuntimeError(
                f"QAT surrogate checkpoint failed for {getattr(base_cls, '__name__', base_cls)!r}."
            )
        return result

    return torch.utils.checkpoint.checkpoint(
        _checkpointed,
        *surrogate_inputs,
        use_reentrant=False,
    )


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
        bias_bits=enc.bias_bits,
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
        bias_bits=enc.bias_bits,
    )


def _resolve_sign_float_input(
    arg: Any,
    int_value_ctx: contextlib.AbstractContextManager,
) -> Optional[torch.Tensor]:
    if isinstance(arg, Int16QuantizedTensor):
        with int_value_ctx:
            return arg.to_float(torch.float32)
    return _unwrap_float_tensor(arg)


def _dispatch_sign_int16_on_float(
    qmodule: nn.Module,
    args: tuple,
    kwargs: dict,
    *,
    device: torch.device,
    mode: ExecutionMode,
    collect_surrogate: bool,
    int_value_ctx: contextlib.AbstractContextManager,
) -> Optional[Union[Int16QuantizedTensor, torch.Tensor]]:
    """Run sign on float input before input-grid collapse (CLZ / STFT near-zero)."""

    oq_list = getattr(qmodule, "output_quantizers", None)
    oq = oq_list[0] if oq_list else None
    if not isinstance(oq, QuantizerBase) or not oq.is_initialized():
        return None
    y_enc = oq.get_encodings()
    if not isinstance(y_enc, AffineEncoding):
        return None

    float_in = _resolve_sign_float_input(args[0], int_value_ctx)
    if float_in is None:
        return None
    float_in = float_in.to(device=device)

    # pylint: disable=import-outside-toplevel
    from aimet_torch.fixed_point.kernels.eltwise import sign_int16_from_float_input

    out_enc = _affine_to_fixed_encoding(y_enc, device)
    with int_value_ctx:
        out = sign_int16_from_float_input(float_in, out_enc)

    if mode == ExecutionMode.INT16_FIXED_QAT_SIM:
        surrogate = _run_qat_surrogate(
            qmodule,
            custom.ElementwiseUnarySign,
            [float_in],
            {},
            {},
        )
        if surrogate is None:
            with int_value_ctx:
                float_out = out.to_float(torch.float32)
            publish_int16_carrier(float_out, out)
            return float_out
        with int_value_ctx:
            fixed_float = out.to_float(torch.float32)
        ste_out = fixed_float + (surrogate - surrogate.detach())
        publish_int16_carrier(ste_out, out)
        return ste_out
    return out


def _collect_quantizer_bitwidths(
    qmodule: nn.Module, attr: str
) -> List[int]:
    """Pull initialized ``bitwidth`` values from a ``nn.ModuleList`` of
    quantizers (input / output / param). Quantizers that exist but are
    not initialized contribute nothing to the contract — same convention
    as the legacy gate that ``return``-ed on ``not is_initialized()``.
    """

    out: List[int] = []
    quants = getattr(qmodule, attr, None)
    if not isinstance(quants, (nn.ModuleList, nn.ModuleDict)):
        return out
    iterator = (
        quants.values() if isinstance(quants, nn.ModuleDict) else iter(quants)
    )
    for quant in iterator:
        if not isinstance(quant, QuantizerBase) or not quant.is_initialized():
            continue
        bw = getattr(quant, "bitwidth", None)
        if bw is None:
            continue
        out.append(int(bw))
    return out


def _enforce_supported_activation_bitwidths(
    qmodule: nn.Module, base_cls: type
) -> None:
    """Refuse REQUANTIZING-kernel dispatch on an unvalidated bitwidth combo.

    PR-2 (W5 SYS-FU-1.B) routes the gate through
    :func:`assert_requantizing_combo_supported`: instead of asserting each
    activation bitwidth in isolation, the gate now collects the full
    operand set (input quantizers, output quantizers, weight param
    quantizer) and validates the ``input_bw + weight_bw`` combo against
    :data:`REQUANTIZING_COMBO_BITWIDTH_BUDGET` for kernels with
    ``cap.is_reduction=True`` (Conv/Linear/MatMul). Element-wise
    REQUANTIZING ops (Multiply/Divide/cross-grid Add/Subtract — N=1) and
    sum-only reduction ops (AvgPool/Mean/LayerNorm — no operand×operand
    MAC) bypass the budget but still require each bitwidth to be in
    :data:`SUPPORTED_ACTIVATION_BITWIDTHS`.

    Bias quantizers stay out of the contract (``bias_bits ∈ {16,32}`` is
    enforced by the kernel/offline path independently).
    """

    capability = get_capability(base_cls)
    if capability is None or not requires_activation_bitwidth_gate(
        capability.kernel_kind
    ):
        return

    qualname = type(qmodule).__name__

    input_bws = _collect_quantizer_bitwidths(qmodule, "input_quantizers")
    output_bws = _collect_quantizer_bitwidths(qmodule, "output_quantizers")

    weight_bws: List[int] = []
    param_q = getattr(qmodule, "param_quantizers", None)
    if isinstance(param_q, nn.ModuleDict) and "weight" in param_q:
        wq = param_q["weight"]
        if (
            isinstance(wq, QuantizerBase)
            and wq.is_initialized()
            and getattr(wq, "bitwidth", None) is not None
        ):
            weight_bws.append(int(wq.bitwidth))

    # The combo gate runs on the operands carrying the MAC reduction
    # — for Conv/Linear those are ``inputs × weights``; for MatMul they
    # are ``inputs × inputs``. The OUTPUT bitwidth is the requantize
    # target grid, NOT a MAC operand, so feeding it as a phantom
    # "input" would build a false ``(output_bw, weight_bw)`` pair and
    # spuriously trip the 16+16 budget on legitimate combos like
    # ``input=8, weight=16, output=16`` (where the only real MAC pair
    # is ``8+16=24`` ≤ budget). We therefore:
    #   * pass ``input_bws`` + ``weight_bws`` to the budget check, and
    #   * sanity-check each ``output_bw`` independently via the
    #     bitwidth-list half of the same gate.
    assert_requantizing_combo_supported(
        input_bws,
        weight_bws,
        where="input/weight quantizers",
        qualname=qualname,
        is_reduction=bool(capability.is_reduction),
    )
    if output_bws:
        assert_requantizing_combo_supported(
            output_bws,
            (),
            where="output quantizers",
            qualname=qualname,
            is_reduction=False,
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

    # Activation bitwidth contract gate: REQUANTIZING kernels (Linear, Conv,
    # AvgPool, Mean, ...) are validated only at the activation bitwidths
    # declared in ``capabilities.SUPPORTED_ACTIVATION_BITWIDTHS``. LOOKUP
    # kernels (sigmoid/sin/sqrt/...) are gated separately by their LUT/CLZ
    # generators, which DO support 16-bit activations, so the helper
    # consults the manifest and skips them. We refuse loudly here rather
    # than ``return None``-ing so callers see a precise error instead of
    # either a generic "not implemented" or — worse — a silent dispatch
    # that produces an arithmetic-but-wrong result (>2k LSB drift on
    # Linear / AvgPool / Mean was the original repro). QAT_SIM runs
    # through a float surrogate path with different numerics and is
    # intentionally not gated here; see
    # audit-int16-activation-quantizer-contract.
    if mode is ExecutionMode.INT16_FIXED_EVAL:
        _enforce_supported_activation_bitwidths(qmodule, base_cls)

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

    collect_surrogate = mode is ExecutionMode.INT16_FIXED_QAT_SIM
    int_value_ctx = torch.no_grad() if collect_surrogate else contextlib.nullcontext()

    first_arg = args[0]
    first_float = _unwrap_float_tensor(first_arg)
    if isinstance(first_arg, Int16QuantizedTensor):
        device = first_arg.int_repr.device
    elif first_float is not None:
        device = first_float.device
    else:
        return None

    if base_cls is custom.ElementwiseUnarySign:
        # pylint: disable=import-outside-toplevel
        from aimet_torch.fixed_point.requantize import (
            _forbid_float_fallback_in_eval,
            int16_fixed_eval_mode,
            sign_float_ref_enabled,
        )

        if kwargs.get("sign_float_ref") and int16_fixed_eval_mode():
            _forbid_float_fallback_in_eval("sign_float_ref")
        use_float_sign = collect_surrogate or sign_float_ref_enabled()
        if use_float_sign:
            sign_out = _dispatch_sign_int16_on_float(
                qmodule,
                args,
                kwargs,
                device=device,
                mode=mode,
                collect_surrogate=collect_surrogate,
                int_value_ctx=int_value_ctx,
            )
            if sign_out is not None:
                return sign_out

    pq = getattr(qmodule, "param_quantizers", None)
    wq = pq["weight"] if pq is not None and "weight" in pq else None
    oq = qmodule.output_quantizers[0] if qmodule.output_quantizers else None

    inputs_int = []
    x_encodings = []
    surrogate_inputs = []
    if base_cls is custom.Concat:
        input_args = args
    else:
        input_args = args[: len(qmodule.input_quantizers)]

    for index, arg in enumerate(input_args):
        if isinstance(arg, Int16QuantizedTensor):
            inputs_int.append(arg)
            x_encodings.append(None)
            if collect_surrogate:
                with int_value_ctx:
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
            with int_value_ctx:
                inputs_int.append(quantize_boundary_from_affine(input_tensor, x_enc).to(device))
            x_encodings.append(x_enc)
            if collect_surrogate:
                surrogate_inputs.append(input_tensor)
            continue
        carrier = maybe_int16_carrier(arg)
        if carrier is not None:
            with int_value_ctx:
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
            elif isinstance(qmodule, (nn.Upsample, nn.UpsamplingNearest2d)):
                # QAT-side surrogate uses the same ``F.interpolate`` branch as
                # the fp32-eval surrogate above; mirror the same extra keys so
                # the surrogate fp32 path matches the int16 dispatch path.
                mode_attr = getattr(qmodule, "mode", "nearest")
                qat_extra.update(
                    {
                        "mode": str(mode_attr) if mode_attr is not None else "nearest",
                        "size": getattr(qmodule, "size", None),
                        "scale_factor": getattr(qmodule, "scale_factor", None),
                    }
                )
            elif isinstance(qmodule, nn.LayerNorm):
                # γ/β are nn.LayerNorm's native fp32 parameters (only present
                # when ``elementwise_affine=True``). For the QAT surrogate
                # they go in as fp32 directly — no separate quantization,
                # matching ``LayerNormInt16Kernel`` semantics.
                qat_extra.update(
                    {
                        "normalized_shape": tuple(qmodule.normalized_shape),
                        "eps": float(getattr(qmodule, "eps", 1e-5)),
                        "weight": getattr(qmodule, "weight", None),
                        "bias": getattr(qmodule, "bias", None),
                    }
                )
            surrogate = _run_qat_surrogate(
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
        bias_bits = _resolve_bias_bits(qmodule)
        with int_value_ctx:
            w_int = quantize_boundary_from_affine(w_float, w_enc).to(device)
            params["weight"] = w_int
            bias = getattr(qmodule, "bias", None)
            if bias is not None:
                if hasattr(qmodule, "_derive_bias_scale"):
                    acc_scale = qmodule._derive_bias_scale(x_scale, w_enc.scale)
                    if acc_scale is None:
                        return None
                    ones = torch.ones_like(
                        acc_scale, dtype=acc_scale.dtype, device=acc_scale.device
                    )
                    params["bias"] = quantize_bias_int(
                        bias, acc_scale, ones, bits=bias_bits
                    )
                else:
                    params["bias"] = quantize_bias_int(
                        bias, x_scale, w_enc.scale, bits=bias_bits
                    )

        w_scale = w_enc.scale.to(device=device, dtype=torch.float32)
        y_scale = y_enc.scale.to(device=device, dtype=torch.float32)
        real_m = (x_scale * w_scale) / y_scale
        out_enc = _affine_output_encoding(y_enc, real_m, device)
        out_enc = dataclasses.replace(out_enc, bias_bits=bias_bits)
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
        elif base_cls is custom.Divide and len(inputs_int) == 2:
            real_m = (
                inputs_int[0].scale.to(device=device, dtype=torch.float32)
                / inputs_int[1].scale.to(device=device, dtype=torch.float32)
            ) / y_scale
        elif base_cls is nn.MaxPool2d:
            # Spec doc/04_算子详细规格/04_09_池化类算子.md §Max-pooling:
            # comparator tree only, no ``M/rshift``; input/output share the
            # same quant grid. Refuse dispatch when encodings disagree to fall
            # back to float QDQ instead of silently rewrapping with a different
            # scale/zero-point. Sentinel handled below: skip
            # ``_output_encoding_from_scales`` when ``real_m`` is ``None``.
            if not _maxpool_encodings_match(x_int, y_enc):
                return None
            out_enc = _affine_to_fixed_encoding(y_enc, device)
            real_m = None
        elif base_cls is nn.AvgPool2d:
            # Spec 04_09 only models ``count_include_pad=True`` (the divisor
            # is fixed at ``k_t * k_f``). ``count_include_pad=False`` would
            # mean a per-window divisor that the HW does not support; refuse
            # dispatch so the op falls back to the float QDQ path instead of
            # producing a silently mis-scaled result.
            #
            # Note: this guard is intentionally configuration-level rather
            # than padding-aware. ``count_include_pad=False`` with
            # ``padding=0`` is numerically equivalent to the spec path, but
            # we still refuse dispatch to keep the rule trivially auditable
            # ("the only supported AvgPool2d config is the spec-04_09 fold")
            # at the cost of occasionally falling back to fp32 QDQ for a
            # config the math would have allowed.
            if not getattr(qmodule, "count_include_pad", True):
                return None
            kernel_size = qmodule.kernel_size
            if isinstance(kernel_size, tuple):
                kernel_area = int(kernel_size[0]) * int(kernel_size[1])
            else:
                kernel_area = int(kernel_size) * int(kernel_size)
            real_m = x_scale / (float(kernel_area) * y_scale)
        elif base_cls is custom.Mean:
            # ``torch.mean(x, dim, keepdim=...)``: forward args[1:] / kwargs carry dim/keepdim.
            mean_dim, mean_keepdim = _resolve_mean_dim_keepdim(args, kwargs, x_int.int_repr.shape)
            reduce_size, _ = _compute_mean_reduce_size_and_dims(
                mean_dim, x_int.int_repr.shape
            )
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
        if real_m is not None:
            out_enc = _output_encoding_from_scales(
                y_enc, real_m, device, base_cls=base_cls
            )

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
        elif isinstance(qmodule, (nn.Upsample, nn.UpsamplingNearest2d)):
            # ``nn.Upsample`` accepts EITHER ``size`` (explicit output spatial
            # shape) OR ``scale_factor`` (multiplier on input shape); exactly
            # one is set on the module. Mode defaults to ``nearest`` for
            # ``UpsamplingNearest2d`` and is whatever was passed for
            # ``Upsample``; the kernel rejects non-nearest modes so the
            # adapter does NOT pre-filter here.
            mode_attr = getattr(qmodule, "mode", "nearest")
            extra.update(
                {
                    "mode": str(mode_attr) if mode_attr is not None else "nearest",
                    "size": getattr(qmodule, "size", None),
                    "scale_factor": getattr(qmodule, "scale_factor", None),
                }
            )
        elif isinstance(qmodule, nn.LayerNorm):
            # Spec doc/04_算子详细规格/04_05_归一化类算子.md §4.5.4. The
            # float-reference kernel at ``norm.LayerNormInt16Kernel`` takes
            # ``normalized_shape`` / ``eps`` / ``weight`` (γ) / ``bias`` (β)
            # from this ``extra`` dict and feeds them straight into
            # ``F.layer_norm``. γ/β remain fp32 here (not quantized along
            # the activation grid) — they are nn.LayerNorm's native fp32
            # parameters and the kernel consumes them directly. The spec
            # big-op variant would quantize γ/β to integer M/rshift coefs
            # at compile time, but that path is the dedicated DSP kernel
            # tracked under ``FU-LAYERNORM-DSP-PARITY``.
            extra.update(
                {
                    "normalized_shape": tuple(qmodule.normalized_shape),
                    "eps": float(getattr(qmodule, "eps", 1e-5)),
                    "weight": getattr(qmodule, "weight", None),
                    "bias": getattr(qmodule, "bias", None),
                }
            )
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
            if isinstance(qmodule, nn.AvgPool2d):
                # Spec 04_09 folds ``1/N`` (N = k_t * k_f) into the offline
                # ``M/rshift`` stream; the kernel re-derives N from
                # ``kernel_size`` and asserts it matches this value, so any
                # adapter path that forgets the fold surfaces immediately.
                ks = qmodule.kernel_size
                if isinstance(ks, tuple):
                    extra["reduce_size"] = int(ks[0]) * int(ks[1])
                else:
                    extra["reduce_size"] = int(ks) * int(ks)
        elif isinstance(qmodule, nn.Hardtanh):
            scale = x_int.scale.to(device=device, dtype=torch.float32)
            zp = x_int.zero_point.to(device=device, dtype=torch.int32)
            extra.update(
                {
                    "min": float(qmodule.min_val),
                    "max": float(qmodule.max_val),
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
            # ``reduce_size`` shares its source of truth with the ``1/N``
            # fold above (``_compute_mean_reduce_size_and_dims``), so
            # ``M/rshift`` and the kernel-side contract cannot drift apart.
            mean_reduce_size, _ = _compute_mean_reduce_size_and_dims(
                mean_dim, x_int.int_repr.shape
            )
            # The ``real_m`` branch above already returns None on
            # ``reduce_size <= 0``; this mirror-guard keeps the two branches
            # symmetric so that a future re-ordering of dispatch stages does
            # not silently produce ``extra['reduce_size'] = 0``.
            if mean_reduce_size <= 0:
                return None
            extra.update(
                {
                    "dim": mean_dim,
                    "keepdim": mean_keepdim,
                    "reduce_size": mean_reduce_size,
                }
            )
        elif base_cls is custom.AdaptiveAvgPool2d:
            # ``output_size=(1,1)`` is the only path that reaches dispatch (see above);
            # the Mean kernel reduces the spatial dims with ``keepdim=True``.
            shape = x_int.int_repr.shape
            extra.update(
                {
                    "dim": (2, 3),
                    "keepdim": True,
                    "output_size": (1, 1),
                    "reduce_size": int(shape[2]) * int(shape[3]),
                }
            )
        elif base_cls is nn.Softmax:
            extra["dim"] = getattr(qmodule, "dim", None)
            if extra["dim"] is None:
                extra["dim"] = kwargs.get("dim", -1)

    from aimet_torch.fixed_point.export.sidecar_loader import (  # noqa: WPS433
        get_int16_online_extra,
        get_int16_sidecar_extra,
        merge_int16_online_extra,
    )

    sidecar_extra = get_int16_sidecar_extra(qmodule)
    if sidecar_extra:
        extra.update(sidecar_extra)
    online_extra = get_int16_online_extra(qmodule, device=device)
    if online_extra:
        for key, value in online_extra.items():
            extra.setdefault(key, value)

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
                merge_int16_online_extra(
                    qmodule,
                    {"clz_lut": clz_body, "clz_func_name": clz_name},
                )
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
            merge_int16_online_extra(
                qmodule,
                {
                    "pwl_lut": extra["pwl_lut"],
                    "pwl_input_encoding": extra.get("pwl_input_encoding"),
                    "phase_fold": extra.get("phase_fold"),
                },
            )

    with int_value_ctx:
        out = kernel(inputs_int, params, out_enc, extra)
    if mode == ExecutionMode.INT16_FIXED_QAT_SIM:
        surrogate = _run_qat_surrogate(qmodule, base_cls, surrogate_inputs, params, extra)
        if surrogate is None:
            with int_value_ctx:
                float_out = out.to_float(torch.float32)
            publish_int16_carrier(float_out, out)
            return float_out
        with int_value_ctx:
            fixed_float = out.to_float(torch.float32)
        ste_out = fixed_float + (surrogate - surrogate.detach())
        publish_int16_carrier(ste_out, out)
        return ste_out
    return out

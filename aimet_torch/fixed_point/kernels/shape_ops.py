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
"""Reference INT16 fixed-point kernels for shape-only operations."""

from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch import nn

from aimet_torch._base.nn.modules import custom
from aimet_torch.fixed_point.capabilities import KernelKind
from aimet_torch.fixed_point.encoding import OutputEncoding
from aimet_torch.fixed_point.kernels._contracts import (
    require_kernel_kind_encoding_contract,
)
from aimet_torch.fixed_point.kernels.eltwise import align_centered_int32_to_output
from aimet_torch.fixed_point.registry import register_fixed_kernel
from aimet_torch.fixed_point.requantize import (
    SIM_TENSOR_DTYPE,
    int32_add_sat,
    requantize_int,
    saturate_sim_tensor,
)
from aimet_torch.fixed_point.tensor import Int16QuantizedTensor


def _wrap(
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


def _requantize_identity_output(
    tensor: Int16QuantizedTensor,
    output_encoding: OutputEncoding,
) -> Int16QuantizedTensor:
    """Requantize an identity-like op from input grid to output grid."""

    if output_encoding.multiplier is None or output_encoding.rshift is None:
        require_kernel_kind_encoding_contract(
            KernelKind.SAME_GRID_VALUE,
            output_encoding,
            op_name="shape-only",
            tensor=tensor,
        )
        return _wrap(tensor.int_repr, output_encoding)

    centered = tensor.centered_int32()
    int_repr = requantize_int(
        centered,
        output_encoding.multiplier.to(device=centered.device),
        output_encoding.rshift.to(device=centered.device),
        output_encoding.zero_point.to(device=centered.device, dtype=torch.int32),
        output_encoding.qmin,
        output_encoding.qmax,
    )
    return _wrap(int_repr, output_encoding)


class _IdentityLikeInt16Kernel:
    """Strict same-grid no-op kernel: rewraps ``int_repr`` and requires the
    output encoding to numerically match the input grid.

    Used by ``nn.Identity`` only. ``nn.Dropout`` has its own
    :class:`DropoutInt16Kernel` because the graph may legitimately place a
    Dropout between two different quant grids in eval (kind
    ``SAME_GRID_OR_REQUANT``); this strict variant would refuse that case.
    """

    def __call__(
        self,
        inputs: List[Int16QuantizedTensor],
        params: Dict[str, Any],
        output_encoding: OutputEncoding,
        extra: Dict[str, Any],
    ) -> Int16QuantizedTensor:
        del params, extra
        if len(inputs) != 1:
            raise ValueError(f"Identity/Dropout expects 1 input; got {len(inputs)}.")
        require_kernel_kind_encoding_contract(
            KernelKind.SAME_GRID_VALUE,
            output_encoding,
            op_name="Identity/Dropout",
            tensor=inputs[0],
        )
        return _wrap(inputs[0].int_repr, output_encoding)


@register_fixed_kernel(nn.Identity)
class IdentityInt16Kernel(_IdentityLikeInt16Kernel):
    """Reference INT16 identity kernel."""

    module_type = nn.Identity


@register_fixed_kernel(nn.Dropout)
class DropoutInt16Kernel:
    """Reference INT16 Dropout kernel.

    In ``eval()`` mode Dropout is a no-op in value space, but it may still sit
    between two different quantization grids. Requantize to the destination
    output encoding instead of only relabeling the integer payload.
    """

    module_type = nn.Dropout

    def __call__(
        self,
        inputs: List[Int16QuantizedTensor],
        params: Dict[str, Any],
        output_encoding: OutputEncoding,
        extra: Dict[str, Any],
    ) -> Int16QuantizedTensor:
        del params, extra
        if len(inputs) != 1:
            raise ValueError(f"Dropout expects 1 input; got {len(inputs)}.")
        return _requantize_identity_output(inputs[0], output_encoding)


@register_fixed_kernel(nn.Flatten)
class FlattenInt16Kernel:
    """Reference INT16 flatten kernel."""

    module_type = nn.Flatten

    def __call__(
        self,
        inputs: List[Int16QuantizedTensor],
        params: Dict[str, Any],
        output_encoding: OutputEncoding,
        extra: Dict[str, Any],
    ) -> Int16QuantizedTensor:
        del params
        if len(inputs) != 1:
            raise ValueError(f"Flatten expects 1 input; got {len(inputs)}.")
        require_kernel_kind_encoding_contract(
            KernelKind.SAME_GRID_VALUE,
            output_encoding,
            op_name="Flatten",
            tensor=inputs[0],
        )
        start_dim = int(extra.get("start_dim", 1))
        end_dim = int(extra.get("end_dim", -1))
        return _wrap(torch.flatten(inputs[0].int_repr, start_dim, end_dim), output_encoding)


@register_fixed_kernel(custom.Reshape)
class ReshapeInt16Kernel:
    """Reference INT16 reshape kernel."""

    module_type = custom.Reshape

    def __call__(
        self,
        inputs: List[Int16QuantizedTensor],
        params: Dict[str, Any],
        output_encoding: OutputEncoding,
        extra: Dict[str, Any],
    ) -> Int16QuantizedTensor:
        del params
        if len(inputs) != 1:
            raise ValueError(f"Reshape expects 1 input; got {len(inputs)}.")
        require_kernel_kind_encoding_contract(
            KernelKind.SAME_GRID_VALUE,
            output_encoding,
            op_name="Reshape",
            tensor=inputs[0],
        )
        shape = extra.get("shape")
        if shape is None:
            raise ValueError("Reshape kernel requires extra['shape'].")
        return _wrap(torch.reshape(inputs[0].int_repr, tuple(shape)), output_encoding)


@register_fixed_kernel(custom.Permute)
class PermuteInt16Kernel:
    """Reference INT16 permute kernel."""

    module_type = custom.Permute

    def __call__(
        self,
        inputs: List[Int16QuantizedTensor],
        params: Dict[str, Any],
        output_encoding: OutputEncoding,
        extra: Dict[str, Any],
    ) -> Int16QuantizedTensor:
        del params
        if len(inputs) != 1:
            raise ValueError(f"Permute expects 1 input; got {len(inputs)}.")
        require_kernel_kind_encoding_contract(
            KernelKind.SAME_GRID_VALUE,
            output_encoding,
            op_name="Permute",
            tensor=inputs[0],
        )
        dims = extra.get("dims")
        if dims is None:
            raise ValueError("Permute kernel requires extra['dims'].")
        return _wrap(torch.permute(inputs[0].int_repr, tuple(dims)), output_encoding)


def _normalize_pad_tuple(pad: Any) -> tuple[int, ...]:
    if isinstance(pad, int):
        return (int(pad),)
    return tuple(int(x) for x in pad)


def _pad_value_int_repr(value: float, output_encoding: OutputEncoding, device: torch.device) -> float:
    """Map a float pad constant to ``F.pad`` value in integer grid."""

    scale = output_encoding.scale.to(device=device, dtype=torch.float32).reshape(-1)[0]
    zp = output_encoding.zero_point.to(device=device, dtype=torch.int32).reshape(-1)[0]
    return float(torch.round(torch.tensor(value, device=device) / scale + zp).item())


_PAD_MODE_CONSTANT = "constant"
_PAD_MODE_REPLICATE = "replicate"
_PAD_MODE_REFLECT = "reflect"
_PAD_SUPPORTED_MODES = frozenset(
    {_PAD_MODE_CONSTANT, _PAD_MODE_REPLICATE, _PAD_MODE_REFLECT}
)


def _pad_int_via_float_roundtrip(
    int_repr: torch.Tensor,
    pad_tuple: tuple[int, ...],
    mode: str,
) -> torch.Tensor:
    """``F.pad`` for ``replicate``/``reflect`` on integer ``int_repr``.

    PyTorch's ``F.pad`` reflect/replicate kernels historically required
    floating-point dtype on a couple of backends. Both modes are pure
    position-based copies (no arithmetic on values), so a fp32 round-trip
    is **bit-exact** for any sim-int payload bounded by ``±2^24`` — INT16
    qmin/qmax (``[-32768, 32767]``) trivially satisfies this. Returns a
    tensor in the original ``SIM_TENSOR_DTYPE``.
    """
    src_dtype = int_repr.dtype
    out = F.pad(int_repr.to(torch.float32), pad_tuple, mode=mode)
    return out.to(src_dtype)


@register_fixed_kernel(custom.Pad)
class PadInt16Kernel:
    """Reference INT16 pad: align to output grid then pad with mode-specific rule.

    Spec ``04_11 §4.11.1 Pad`` lists three modes (``constant`` / ``reflect`` /
    ``replicate``). ``constant`` requantizes the input to the output grid first
    and uses the rounded pad-value on the output grid; ``reflect`` and
    ``replicate`` are byte-stream position copies on the centred int_repr and
    therefore equivalent in the quant domain (no new code values introduced).

    Spec calls out ``constant`` as the priority hardware mode; the other two
    are reference-software-side extensions that match the documented op
    semantics so callers don't need to special-case them at the adapter
    boundary.
    """

    module_type = custom.Pad

    def __call__(
        self,
        inputs: List[Int16QuantizedTensor],
        params: Dict[str, Any],
        output_encoding: OutputEncoding,
        extra: Dict[str, Any],
    ) -> Int16QuantizedTensor:
        del params
        if len(inputs) != 1:
            raise ValueError(f"Pad expects 1 input; got {len(inputs)}.")
        pad = extra.get("pad")
        if pad is None:
            raise ValueError("Pad kernel requires extra['pad'].")
        mode = str(extra.get("mode", _PAD_MODE_CONSTANT))
        if mode not in _PAD_SUPPORTED_MODES:
            raise ValueError(
                f"Pad INT16 kernel supports modes {sorted(_PAD_SUPPORTED_MODES)}; "
                f"got {mode!r}."
            )
        aligned = _requantize_identity_output(inputs[0], output_encoding)
        pad_tuple = _normalize_pad_tuple(pad)
        if mode == _PAD_MODE_CONSTANT:
            pad_value = _pad_value_int_repr(
                float(extra.get("value", 0.0)),
                output_encoding,
                aligned.int_repr.device,
            )
            padded = F.pad(
                aligned.int_repr, pad_tuple, mode="constant", value=pad_value
            )
        else:
            padded = _pad_int_via_float_roundtrip(aligned.int_repr, pad_tuple, mode)
        return _wrap(padded, output_encoding)


@register_fixed_kernel(custom.Concat)
class ConcatInt16Kernel:
    """Reference INT16 concat: align each branch to output grid, then concat int_repr."""

    module_type = custom.Concat

    def __call__(
        self,
        inputs: List[Int16QuantizedTensor],
        params: Dict[str, Any],
        output_encoding: OutputEncoding,
        extra: Dict[str, Any],
    ) -> Int16QuantizedTensor:
        del params
        if not inputs:
            raise ValueError("Concat expects at least one input.")
        axis = int(extra.get("axis", 0))
        aligned = [align_centered_int32_to_output(item, output_encoding) for item in inputs]
        acc = torch.cat(aligned, dim=axis)
        acc = int32_add_sat(
            acc,
            output_encoding.zero_point.to(device=acc.device, dtype=torch.int32),
        )
        return _wrap(
            saturate_sim_tensor(acc, output_encoding.qmin, output_encoding.qmax),
            output_encoding,
        )


# Spec doc/04_算子详细规格/04_10_Resize类算子.md §4.10.2 explicitly states
# "直接复制量化值，无需重量化" (copy quant values, no requantize). The kernel is
# therefore SAME_GRID_VALUE — only the spatial index map changes; payload bits
# are byte-stream identical. Bilinear (§4.10.1) is REQUANTIZING and lives in a
# separate kernel (not implemented this round).
_NEAREST_MODES = frozenset({"nearest", "nearest-exact"})


def _nearest_resample_int_repr(
    int_repr: torch.Tensor,
    *,
    size: Any,
    scale_factor: Any,
) -> torch.Tensor:
    """Run ``F.interpolate(mode='nearest')`` on the integer payload.

    ``F.interpolate`` requires a floating dtype for the nearest path on a few
    backends but the operation is a pure position-based copy (no arithmetic),
    so the fp32 round-trip is bit-exact for any sim-int payload bounded by
    ``±2^24`` — INT16 qmin/qmax (``[-32768, 32767]``) trivially satisfies this.
    Same trick as :func:`_pad_int_via_float_roundtrip` above.
    """

    src_dtype = int_repr.dtype
    interp_kwargs: Dict[str, Any] = {"mode": "nearest"}
    if size is not None:
        interp_kwargs["size"] = size
    elif scale_factor is not None:
        # scale_factor must remain float for F.interpolate's recompute path.
        interp_kwargs["scale_factor"] = scale_factor
    else:
        raise ValueError(
            "Nearest resize requires either extra['size'] or extra['scale_factor']."
        )
    out = F.interpolate(int_repr.to(torch.float32), **interp_kwargs)
    return out.to(src_dtype)


class _NearestResizeInt16Kernel:
    """Shared kernel for nearest-mode resize: byte-stream-identity spatial resample.

    Used by both ``nn.Upsample`` (when ``mode='nearest'``) and the dedicated
    ``nn.UpsamplingNearest2d``. The output encoding is required to match the
    input grid (SAME_GRID_VALUE contract per spec 04_10 §4.10.2).
    """

    op_name = "ResizeNearest"

    def __call__(
        self,
        inputs: List[Int16QuantizedTensor],
        params: Dict[str, Any],
        output_encoding: OutputEncoding,
        extra: Dict[str, Any],
    ) -> Int16QuantizedTensor:
        del params
        if len(inputs) != 1:
            raise ValueError(
                f"{self.op_name} expects 1 input; got {len(inputs)}."
            )
        mode = str(extra.get("mode", "nearest"))
        if mode not in _NEAREST_MODES:
            # Bilinear / other modes are out of scope; refuse explicitly so a
            # silent grid mismatch never reaches the SAME_GRID_VALUE contract.
            raise ValueError(
                f"{self.op_name} INT16 kernel supports modes {sorted(_NEAREST_MODES)}; "
                f"got {mode!r}. Bilinear lives on a separate REQUANTIZING path "
                "(spec 04_10 §4.10.1) and is not implemented yet."
            )
        require_kernel_kind_encoding_contract(
            KernelKind.SAME_GRID_VALUE,
            output_encoding,
            op_name=self.op_name,
            tensor=inputs[0],
        )
        resampled = _nearest_resample_int_repr(
            inputs[0].int_repr,
            size=extra.get("size"),
            scale_factor=extra.get("scale_factor"),
        )
        return _wrap(resampled, output_encoding)


@register_fixed_kernel(nn.Upsample)
class UpsampleInt16Kernel(_NearestResizeInt16Kernel):
    """Reference INT16 Upsample kernel — nearest mode only.

    ``nn.Upsample`` covers both nearest and bilinear; only the nearest branch
    is currently registered (SAME_GRID_VALUE per spec 04_10 §4.10.2). Bilinear
    invocations are refused at kernel entry so the adapter dispatch can
    fallback to FP32_QDQ instead of silently producing wrong values.
    """

    module_type = nn.Upsample
    op_name = "Upsample"


@register_fixed_kernel(nn.UpsamplingNearest2d)
class UpsamplingNearest2dInt16Kernel(_NearestResizeInt16Kernel):
    """Reference INT16 UpsamplingNearest2d kernel (always nearest)."""

    module_type = nn.UpsamplingNearest2d
    op_name = "UpsamplingNearest2d"

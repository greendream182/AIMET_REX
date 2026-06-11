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
"""Quantized integer carrier for G3 fixed-point simulation (``int16_fixed_*`` modes).

:class:`FixedPointSimTensor` is the primary G3 segment carrier (ADR-013). It does
**not** mean activations/weights are always 16-bit: ``qmin``/``qmax`` come from
the AIMET quantizer (U8, S8, U16, …). ``int_repr`` uses :data:`~aimet_torch.fixed_point.requantize.SIM_TENSOR_DTYPE`
(``torch.int32``) as the storage slot; layer MAC + requant use separate
``multiplier``/``rshift`` on :class:`~aimet_torch.fixed_point.encoding.OutputEncoding`.
See Design v2 §3.7 and spec 04.

:class:`Int16QuantizedTensor` is a deprecated subclass kept for API compatibility.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import torch

from aimet_torch.fixed_point.quant_grid import MAX_SEMANTIC_LEVELS
from aimet_torch.fixed_point.requantize import (
    INT16_QMAX,
    INT16_QMIN,
    INT32_QMAX,
    INT32_QMIN,
    SIM_TENSOR_DTYPE,
    saturate_sim_tensor,
)

UINT16_QMAX = (1 << 16) - 1
MAX_16BIT_LEVELS = 1 << 16

if TYPE_CHECKING:
    from aimet_torch.fixed_point.encoding import FixedScaleEncoding


def align_stat_rank(stat: torch.Tensor, int_repr: torch.Tensor) -> torch.Tensor:
    """Pad ``stat`` with trailing length-1 dims so it broadcasts over spatial/tail axes of ``int_repr``.

    Without this, subtracting e.g. ``zero_point`` shaped like ``(out,1,1)`` from ``int_repr``
    ``(out,in,1,k)`` mis-aligns the channel dimension of ``zero_point`` with the inner ``in``
    axis instead of only ``out`` (PyTorch aligns trailing dimensions first).
    """

    t = stat.to(device=int_repr.device)
    while t.ndim < int_repr.ndim:
        t = t.unsqueeze(-1)
    return t


@dataclass(frozen=True)
class FixedPointSimTensor:
    """Immutable carrier for **integer simulation** segments (G3), not "semantic INT16 only".

    - **Storage**: ``int_repr`` is :data:`SIM_TENSOR_DTYPE` (``torch.int32``; ADR-013).
    - **Semantic grid**: ``qmin``/``qmax`` follow the quantizer bitwidth (may be U8, etc.).
    - **Scale metadata**: ``scale``/``zero_point`` describe the boundary grid; optional
      boundary Q via ``(m_int16, rshift)`` uses :meth:`from_fixed_scale_encoding`.
    - **Not used** on the ``fixed_scale_qdq`` main path (G2 uses float QDQ between ops).
    """

    int_repr: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor
    qmin: int = INT16_QMIN
    qmax: int = INT16_QMAX
    axis: Optional[int] = None

    def __post_init__(self):
        int_repr = self.int_repr
        if int_repr.dtype == torch.int16:
            # PR-2 transition: legacy kernels still emit int16; normalize in-place.
            int_repr = int_repr.to(SIM_TENSOR_DTYPE)
            object.__setattr__(self, "int_repr", int_repr)
        elif int_repr.dtype != SIM_TENSOR_DTYPE:
            raise TypeError(
                f"int_repr must be {SIM_TENSOR_DTYPE}; got {self.int_repr.dtype}."
            )

        num_levels = int(self.qmax) - int(self.qmin) + 1
        if num_levels <= 0 or num_levels > MAX_SEMANTIC_LEVELS:
            raise ValueError(
                f"qmin/qmax ({self.qmin}, {self.qmax}) describe {num_levels} levels; "
                f"valid range is [1, {MAX_SEMANTIC_LEVELS}]."
            )
        if self.qmin < INT32_QMIN or self.qmax > INT32_QMAX:
            raise ValueError(
                f"qmin/qmax ({self.qmin}, {self.qmax}) exceed the int32 sim-tensor "
                f"container [{INT32_QMIN}, {INT32_QMAX}] (e.g. full u32 needs int64; "
                "ADR-013 follow-up)."
            )

        if self.zero_point.dtype != torch.int32:
            raise TypeError(
                f"zero_point must be torch.int32; got {self.zero_point.dtype}."
            )
        if not self.scale.is_floating_point():
            raise TypeError(f"scale must be floating point; got {self.scale.dtype}.")
        if self.qmin > self.qmax:
            raise ValueError(f"qmin ({self.qmin}) must be <= qmax ({self.qmax}).")
        if self.axis is not None and self.scale.numel() not in (
            1,
            self.int_repr.shape[self.axis],
        ):
            raise ValueError(
                "Per-channel scale must be scalar or match int_repr.shape[axis]."
            )

    @classmethod
    def from_float(
        cls,
        tensor: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        axis: Optional[int] = None,
        *,
        qmin: int = INT16_QMIN,
        qmax: int = INT16_QMAX,
    ) -> FixedPointSimTensor:
        """Quantize a floating-point tensor into a sim-tensor container."""

        if not tensor.is_floating_point():
            raise TypeError(f"tensor must be floating point; got {tensor.dtype}.")
        if torch.any(torch.isnan(tensor)):
            raise ValueError("tensor contains NaN values.")
        if torch.any(scale == 0):
            raise ValueError("scale must not contain zero.")

        scale = scale.to(device=tensor.device, dtype=torch.float32)
        zero_point = zero_point.to(device=tensor.device, dtype=torch.int32)
        quantized = torch.round(tensor / scale + zero_point.to(torch.float32))
        int_repr = saturate_sim_tensor(quantized, qmin, qmax)
        return cls(
            int_repr=int_repr,
            scale=scale,
            zero_point=zero_point,
            qmin=qmin,
            qmax=qmax,
            axis=axis,
        )

    @classmethod
    def from_affine_encoding(
        cls, tensor: torch.Tensor, encoding
    ) -> FixedPointSimTensor:
        """Quantize floats using AIMET v2 :class:`AffineEncoding` (same grid as QDQ)."""

        # pylint: disable=import-outside-toplevel
        from aimet_torch.v2.quantization.affine.encoding import AffineEncoding

        if not isinstance(encoding, AffineEncoding):
            raise TypeError(
                f"encoding must be AffineEncoding; got {type(encoding).__name__}."
            )
        if encoding.block_size not in (None, ()):
            raise NotImplementedError(
                "Fixed-point adapter does not support block-wise AffineEncoding yet."
            )

        scale = encoding.scale.to(device=tensor.device, dtype=torch.float32)
        offset = encoding.offset.to(device=tensor.device, dtype=torch.float32)
        if torch.any(scale == 0):
            raise ValueError("encoding.scale must not contain zero.")

        q_float = torch.round(tensor / scale - offset)
        int_repr = saturate_sim_tensor(q_float, encoding.qmin, encoding.qmax)
        zero_point = (-offset).round().to(torch.int32)

        axis: Optional[int] = None
        if scale.numel() != 1 and scale.dim() == tensor.dim():
            candidates = [
                d
                for d in range(tensor.dim())
                if tensor.shape[d] == scale.shape[d] and tensor.shape[d] != 1
            ]
            if len(candidates) == 1:
                axis = candidates[0]

        return cls(
            int_repr=int_repr,
            scale=scale,
            zero_point=zero_point,
            qmin=encoding.qmin,
            qmax=encoding.qmax,
            axis=axis,
        )

    @classmethod
    def from_fixed_scale_encoding(
        cls,
        tensor: torch.Tensor,
        encoding: FixedScaleEncoding,
    ) -> FixedPointSimTensor:
        """Quantize using ``(m_int16, rshift)`` — same grid as ``fixed_scale_qdq`` (spec 15)."""

        # pylint: disable=import-outside-toplevel
        from aimet_torch.fixed_point.encoding import FixedScaleEncoding
        from aimet_torch.fixed_point.fixed_scale_qdq import quantize_with_fixed_scale
        from aimet_torch.fixed_point.offline.scale_fixed import fixed_scale_float_scale

        if not isinstance(encoding, FixedScaleEncoding):
            raise TypeError(
                f"encoding must be FixedScaleEncoding; got {type(encoding).__name__}."
            )

        q = quantize_with_fixed_scale(tensor, encoding)
        int_repr = saturate_sim_tensor(q, encoding.qmin, encoding.qmax)
        device = tensor.device
        scale = fixed_scale_float_scale(
            encoding.m_int16,
            encoding.rshift,
            device=device,
        )
        zp = encoding.zero_point.to(device=device, dtype=torch.int32)
        axis: Optional[int] = None
        if scale.numel() != 1 and scale.dim() == tensor.dim():
            candidates = [
                d
                for d in range(tensor.dim())
                if tensor.shape[d] == scale.shape[d] and tensor.shape[d] != 1
            ]
            if len(candidates) == 1:
                axis = candidates[0]

        return cls(
            int_repr=int_repr,
            scale=scale,
            zero_point=zp,
            qmin=encoding.qmin,
            qmax=encoding.qmax,
            axis=axis,
        )

    def centered_int32(self) -> torch.Tensor:
        """Integer centered representation (``int_repr - zero_point``) with safe broadcasting."""

        zp = align_stat_rank(self.zero_point.to(torch.int32), self.int_repr)
        return self.int_repr.to(torch.int32) - zp

    def to_float(self, debug_dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Convert to float for debug and metrics only."""

        # pylint: disable=import-outside-toplevel
        import os

        from aimet_torch.fixed_point.execution_mode import ExecutionMode, get_quant_execution_mode
        from aimet_torch.fixed_point.metrics.flags import int16_eval_debug_float_allowed

        mode = get_quant_execution_mode()
        if mode is ExecutionMode.INT16_FIXED_EVAL and not int16_eval_debug_float_allowed():
            strict = os.environ.get("AIMET_RX_INT16_STRICT_TO_FLOAT", "").lower() in (
                "1",
                "true",
                "yes",
            )
            message = (
                "FixedPointSimTensor.to_float() in INT16_FIXED_EVAL without "
                "FixedPointProfiler / int16_eval_allow_debug_float(); "
                "this path is for debug/metrics only."
            )
            if strict:
                raise RuntimeError(message)
            warnings.warn(message, stacklevel=2)

        scale = align_stat_rank(
            self.scale.to(device=self.int_repr.device, dtype=debug_dtype), self.int_repr
        )
        return self.centered_int32().to(debug_dtype) * scale

    def quantized_repr(self) -> torch.Tensor:
        """Return the underlying quantized integer representation."""

        return self.int_repr

    def saturate(self) -> FixedPointSimTensor:
        """Return a saturated copy in the configured ``[qmin, qmax]`` range."""

        return FixedPointSimTensor(
            int_repr=saturate_sim_tensor(
                self.int_repr.to(torch.int32), self.qmin, self.qmax
            ),
            scale=self.scale,
            zero_point=self.zero_point,
            qmin=self.qmin,
            qmax=self.qmax,
            axis=self.axis,
        )

    def to(self, device: torch.device) -> FixedPointSimTensor:
        """Move tensor and metadata to another device."""

        return FixedPointSimTensor(
            int_repr=self.int_repr.to(device),
            scale=self.scale.to(device),
            zero_point=self.zero_point.to(device),
            qmin=self.qmin,
            qmax=self.qmax,
            axis=self.axis,
        )

    # ---- Tensor-like layout introspection (FX-traced graphs) ----------------
    # model_preparer 会把 ``b, c, t, f = x.shape`` 烘焙进 FX 图；INT16 路径下
    # 中间激活是 FixedPointSimTensor，必须暴露与 int_repr 一致的 layout 属性，
    # 否则 ``getattr(module_output, 'shape')`` 在 e2e forward 里直接 AttributeError。
    @property
    def shape(self) -> torch.Size:
        return self.int_repr.shape

    @property
    def ndim(self) -> int:
        return self.int_repr.ndim

    @property
    def device(self) -> torch.device:
        return self.int_repr.device

    @property
    def dtype(self) -> torch.dtype:
        return self.int_repr.dtype

    def size(self, dim: Optional[int] = None):
        """Mirror ``torch.Tensor.size`` for FX ``size`` / ``shape`` call sites."""
        if dim is None:
            return self.int_repr.size()
        return self.int_repr.size(dim)

    def dim(self) -> int:
        """Mirror ``torch.Tensor.dim`` (alias of ``ndim``)."""
        return self.int_repr.dim()

    def _layout_transform(self, new_int_repr: torch.Tensor) -> FixedPointSimTensor:
        """Apply a pure layout change to ``int_repr``; metadata unchanged, axis cleared."""
        return FixedPointSimTensor(
            int_repr=new_int_repr,
            scale=self.scale,
            zero_point=self.zero_point,
            qmin=self.qmin,
            qmax=self.qmax,
            axis=None,
        )

    def permute(self, *dims) -> FixedPointSimTensor:
        return self._layout_transform(self.int_repr.permute(*dims))

    def contiguous(self) -> FixedPointSimTensor:
        return self._layout_transform(self.int_repr.contiguous())

    def view(self, *shape) -> FixedPointSimTensor:
        if len(shape) == 1 and not isinstance(shape[0], int):
            return self._layout_transform(self.int_repr.view(shape[0]))
        return self._layout_transform(self.int_repr.view(*shape))

    def reshape(self, *shape) -> FixedPointSimTensor:
        if len(shape) == 1 and not isinstance(shape[0], int):
            return self._layout_transform(self.int_repr.reshape(shape[0]))
        return self._layout_transform(self.int_repr.reshape(*shape))

    def transpose(self, dim0: int, dim1: int) -> FixedPointSimTensor:
        return self._layout_transform(self.int_repr.transpose(dim0, dim1))

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> FixedPointSimTensor:
        return self._layout_transform(self.int_repr.flatten(start_dim, end_dim))

    def squeeze(self, dim: Optional[int] = None) -> FixedPointSimTensor:
        if dim is None:
            return self._layout_transform(self.int_repr.squeeze())
        return self._layout_transform(self.int_repr.squeeze(dim))

    def unsqueeze(self, dim: int) -> FixedPointSimTensor:
        return self._layout_transform(self.int_repr.unsqueeze(dim))

    def __getitem__(self, index) -> FixedPointSimTensor:
        return self._layout_transform(self.int_repr[index])

    # ---- Arithmetic guards -------------------------------------------------
    def _arith_guard(self, op: str):
        raise TypeError(
            f"FixedPointSimTensor does not support the '{op}' operator directly. "
            "Wrap the elementwise op in a QuantizedAdd / QuantizedSubtract / "
            "QuantizedMultiply module (or run aimet_torch.model_preparer."
            "prepare_model on the source model) so that an output quantizer "
            "owns the destination encoding for the INT16 fixed-point dispatch."
        )

    def __add__(self, other):  # pragma: no cover - exercised via runtime tests
        self._arith_guard("+")

    def __radd__(self, other):  # pragma: no cover
        self._arith_guard("+")

    def __sub__(self, other):  # pragma: no cover
        self._arith_guard("-")

    def __rsub__(self, other):  # pragma: no cover
        self._arith_guard("-")

    def __mul__(self, other):  # pragma: no cover
        self._arith_guard("*")

    def __rmul__(self, other):  # pragma: no cover
        self._arith_guard("*")

    _LAYOUT_TORCH_FUNCS = ()  # populated below after the class body

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func in cls._LAYOUT_TORCH_FUNCS:
            carrier = next(
                (a for a in args if isinstance(a, FixedPointSimTensor)),
                None,
            )
            if carrier is not None:
                int_args = tuple(
                    a.int_repr if isinstance(a, FixedPointSimTensor) else a
                    for a in args
                )
                new_int_repr = func(*int_args, **kwargs)
                return carrier._layout_transform(new_int_repr)
        return NotImplemented


# Historical API name; same class as FixedPointSimTensor (Design v2 §3.7).
Int16QuantizedTensor = FixedPointSimTensor

# Layout-only torch functions whose semantics are pure ``int_repr`` rearrangement.
FixedPointSimTensor._LAYOUT_TORCH_FUNCS = (  # type: ignore[attr-defined]
    torch.flatten,
    torch.reshape,
    torch.permute,
    torch.transpose,
    torch.squeeze,
    torch.unsqueeze,
)

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
"""Shared contract validators for INT16 fixed-point reference kernels.

Each ``KernelKind`` (see :mod:`aimet_torch.fixed_point.capabilities`) carries
an implicit contract on the surrounding :class:`OutputEncoding` and per-op
``extra`` dict. Historically every kernel re-implemented its own version of
those checks (``_require_same_encoding`` in ``shape_ops.py``,
``_require_maxpool_same_encoding`` in ``pool.py``, ``_encoding_grids_match``
in ``eltwise.py``), which silently drifted apart. This module centralises the
canonical helpers so that:

* requantizing kernels (Conv/Linear/MatMul/Mul/Divide/AvgPool/Mean/...) are
  guaranteed to receive a fully populated ``(multiplier, rshift)`` pair on
  the output encoding;
* same-grid-value kernels (MaxPool, layout-only shape ops) refuse to silently
  paper over a quant-grid mismatch, mirroring the comparator/relabel
  semantics described in ``doc/04_算子详细规格``;
* spec-mandated operand limits (e.g. MaxPool ``kt,kf <= 3``,
  AvgPool ``kernel ∈ {(2,2),(4,4),(4,2),(2,4)}``, ``padding <= 3``) can be
  enforced uniformly under :func:`hw_ref_mode_enabled` without breaking
  cross-op functional tests that exercise the pure software reference.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import torch

from aimet_torch.fixed_point.capabilities import KernelKind
from aimet_torch.fixed_point.encoding import OutputEncoding
from aimet_torch.fixed_point.requantize import (
    INT32_QMAX,
    INT32_QMIN,
    hw_ref_mode_enabled,
)
from aimet_torch.fixed_point.tensor import Int16QuantizedTensor


_EXPLICITLY_HANDLED_KERNEL_KINDS = frozenset(KernelKind)


def explicitly_handled_kernel_kinds() -> frozenset[KernelKind]:
    """Return KernelKind values handled by ``require_kernel_kind_encoding_contract``.

    Tests assert this matches ``set(KernelKind)`` so adding a new manifest
    category forces the contract dispatch to be updated deliberately instead
    of falling through to a wrong default (especially treating
    ``SAME_GRID_OR_REQUANT`` as strict ``SAME_GRID_VALUE``).
    """

    return _EXPLICITLY_HANDLED_KERNEL_KINDS


def require_requantizing_encoding(
    output_encoding: OutputEncoding,
    *,
    op_name: str,
) -> None:
    """Requantizing kernels must carry a fully populated ``(multiplier, rshift)``.

    Used by Conv/Linear/MatMul/Mul/Divide/AvgPool/Mean/AdaptiveAvgPool2d and
    every PWL/CLZ lookup kernel — anything that reduces or recombines values
    across grids and therefore needs the spec-mandated requantize step.
    """

    if output_encoding.multiplier is None or output_encoding.rshift is None:
        raise ValueError(
            f"{op_name} output encoding must provide multiplier/rshift "
            "(requantizing kernel)."
        )


def _require_complete_or_absent_requant_params(
    output_encoding: OutputEncoding,
    *,
    op_name: str,
) -> bool:
    """Return True iff a complete ``(multiplier, rshift)`` pair is present."""

    has_multiplier = output_encoding.multiplier is not None
    has_rshift = output_encoding.rshift is not None
    if has_multiplier != has_rshift:
        raise ValueError(
            f"{op_name} output encoding must provide multiplier and rshift "
            "together, or neither."
        )
    return has_multiplier


def _scale_equal(
    tensor: Int16QuantizedTensor,
    output_encoding: OutputEncoding,
) -> bool:
    """Numerically equal in fp32, with a tiny relative tolerance.

    Same-grid kernels are *spec-strict*: the hardware comparator/relabel keeps
    the quant grid bit-for-bit (``04_09`` MaxPool, layout-only ops). But the
    simulator's input quantizer can go through the fixed-scale boundary path
    (``M_int / 2**rshift`` quantized via :func:`quantize_multiplier`), which is
    mathematically equivalent to the affine path yet differs by up to one
    ``quantize_multiplier`` ULP — at uint16 multipliers that's ``~2**-15``
    relative ≈ ``3e-5``. Returning False on those phantom mismatches would
    force perfectly valid layouts (Reshape/Permute/Flatten) to fall back to
    float QDQ. We therefore accept relative error up to ``5e-5`` (still well
    below 1 LSB even on int8 grids, where the LSB ≈ ``7.87e-3`` of full
    range) while still flagging genuine grid changes (e.g. ``scale=1.0`` vs
    ``scale=0.5``). When tighter checking is desired (HW conformance tests),
    callers should drive the kernel directly with strictly equal encodings —
    the test in ``test_eltwise_pool_shape.py`` does exactly that.
    """

    in_scale = tensor.scale.detach().reshape(-1)
    out_scale = (
        output_encoding.scale.detach()
        .to(device=in_scale.device, dtype=in_scale.dtype)
        .reshape(-1)
    )
    if in_scale.shape != out_scale.shape:
        return False
    if bool(torch.equal(in_scale, out_scale)):
        return True
    return bool(torch.allclose(in_scale, out_scale, rtol=5e-5, atol=0.0))


def _zero_point_equal(
    tensor: Int16QuantizedTensor,
    output_encoding: OutputEncoding,
) -> bool:
    in_zp = tensor.zero_point.detach().to(torch.int32).reshape(-1)
    out_zp = (
        output_encoding.zero_point.detach()
        .to(device=in_zp.device, dtype=torch.int32)
        .reshape(-1)
    )
    return in_zp.shape == out_zp.shape and bool(torch.equal(in_zp, out_zp))


def same_grid_encoding(
    tensor: Int16QuantizedTensor,
    output_encoding: OutputEncoding,
) -> bool:
    """Return True iff ``tensor`` and ``output_encoding`` describe the same
    quant grid (``scale`` / ``zero_point`` / ``qmin`` / ``qmax``).

    Numerical equality is used (not Python identity), so two distinct
    quantizer instances that converge on the same grid still compare equal.
    """

    if tensor.qmin != output_encoding.qmin or tensor.qmax != output_encoding.qmax:
        return False
    return _scale_equal(tensor, output_encoding) and _zero_point_equal(
        tensor, output_encoding
    )


def require_same_grid_encoding(
    tensor: Int16QuantizedTensor,
    output_encoding: OutputEncoding,
    *,
    op_name: str,
) -> None:
    """Strict same-grid contract — used by ``KernelKind.SAME_GRID_VALUE``.

    Covers MaxPool2d (comparator-only HW per ``04_09``) and the pure layout
    ops Identity/Reshape/Permute/Flatten — kernels that don't even have a
    requantize fallback. **Do not** use this on
    ``KernelKind.SAME_GRID_OR_REQUANT`` ops (ReLU/Hardtanh/Clamp/Pad/Concat/
    Dropout/Abs/Sign): those kernels are allowed to take ``(multiplier,
    rshift)`` and align grids, so a strict equality guard would force them
    onto a needless float fallback. Error messages include the spec section
    so blast-radius is easy to trace.
    """

    if tensor.qmin != output_encoding.qmin or tensor.qmax != output_encoding.qmax:
        raise ValueError(
            f"{op_name} input/output qmin/qmax must match per spec "
            "(same-grid kernel, no requantize); got "
            f"input=({tensor.qmin},{tensor.qmax}), "
            f"output=({output_encoding.qmin},{output_encoding.qmax})."
        )
    if not _scale_equal(tensor, output_encoding):
        raise ValueError(
            f"{op_name} input/output scale must be numerically equal per spec "
            f"(input={tensor.scale.detach().cpu().tolist()!r}, "
            f"output={output_encoding.scale.detach().cpu().tolist()!r})."
        )
    if not _zero_point_equal(tensor, output_encoding):
        raise ValueError(
            f"{op_name} input/output zero_point must be numerically equal per spec "
            f"(input={tensor.zero_point.detach().cpu().tolist()!r}, "
            f"output={output_encoding.zero_point.detach().cpu().tolist()!r})."
        )


def require_inputs_same_grid_encoding(
    inputs: Sequence[Int16QuantizedTensor],
    output_encoding: OutputEncoding,
    *,
    op_name: str,
) -> None:
    for tensor in inputs:
        require_same_grid_encoding(tensor, output_encoding, op_name=op_name)


def require_kernel_kind_encoding_contract(
    kernel_kind: KernelKind,
    output_encoding: OutputEncoding,
    *,
    op_name: str,
    tensor: Optional[Int16QuantizedTensor] = None,
) -> None:
    """Apply the output-encoding contract implied by a manifest ``KernelKind``.

    This is the single switch statement for category-level contract checks:
    every :class:`KernelKind` must appear here explicitly. In particular,
    ``SAME_GRID_OR_REQUANT`` is *not* strict same-grid. It accepts either:

    * no ``(multiplier, rshift)`` pair, in which case the input/output grids
      must match and ``tensor`` is required; or
    * a complete ``(multiplier, rshift)`` pair, in which case the kernel is
      expected to align/requantize internally and no same-grid check is made.
    """

    if kernel_kind is KernelKind.REQUANTIZING:
        require_requantizing_encoding(output_encoding, op_name=op_name)
        return

    if kernel_kind is KernelKind.LOOKUP:
        require_requantizing_encoding(output_encoding, op_name=op_name)
        return

    if kernel_kind is KernelKind.SAME_GRID_VALUE:
        if tensor is None:
            raise ValueError(
                f"{op_name} strict same-grid contract requires an input tensor."
            )
        require_same_grid_encoding(tensor, output_encoding, op_name=op_name)
        return

    if kernel_kind is KernelKind.SAME_GRID_OR_REQUANT:
        has_requant = _require_complete_or_absent_requant_params(
            output_encoding, op_name=op_name
        )
        if has_requant:
            return
        if tensor is None:
            raise ValueError(
                f"{op_name} same-grid fast path requires an input tensor when "
                "multiplier/rshift are absent."
            )
        require_same_grid_encoding(tensor, output_encoding, op_name=op_name)
        return

    if kernel_kind is KernelKind.NONE:
        raise ValueError(f"{op_name} has no fixed-point kernel contract.")

    raise AssertionError(f"Unhandled KernelKind: {kernel_kind!r}")


# ---------------------------------------------------------------------------
# Spec-mandated operand-shape limits.
#
# These are *hardware* constraints (doc/04_算子详细规格/04_09) and would break
# pure-software reference tests that legitimately use 7×7 GAP, large stride
# pooling etc. They are therefore only enforced when ``hw_ref_mode_enabled()``
# returns True (env ``AIMET_RX_HW_REF=1`` and friends), matching how
# ``requantize.py`` / ``lut.py`` already gate their HW-faithful branches.
# ---------------------------------------------------------------------------


def _as_pair(value, *, name: str) -> Tuple[int, int]:
    if isinstance(value, int):
        return (int(value), int(value))
    if isinstance(value, Iterable):
        seq = tuple(int(v) for v in value)
        if len(seq) == 2:
            return seq
        if len(seq) == 1:
            return (seq[0], seq[0])
    raise ValueError(f"{name} must be int or 2-tuple of int; got {value!r}.")


def require_pool2d_operand_limits(
    *,
    kernel_size,
    padding,
    op_name: str,
    max_kernel: int = 0,
    allowed_kernels: Tuple[Tuple[int, int], ...] = (),
    max_padding: int = 3,
    spec_ref: str = "doc/04_算子详细规格/04_09_池化类算子.md",
) -> None:
    """Enforce 04_09 pooling operand-shape limits (HW-strict mode only).

    * ``max_kernel`` (>0): reject ``kt`` or ``kf`` greater than the bound
      (MaxPool2d uses ``max_kernel=3``).
    * ``allowed_kernels``: when non-empty, ``(kh, kw)`` must appear in this
      whitelist (AvgPool2d uses ``{(2,2),(4,4),(4,2),(2,4)}``).
    * ``max_padding`` (default 3): both ``ph`` and ``pw`` must be in
      ``[0, max_padding]``.

    No-op when :func:`hw_ref_mode_enabled` is False, so functional tests on
    arbitrary kernel sizes keep working under the pure-software reference.
    """

    if not hw_ref_mode_enabled():
        return

    kh, kw = _as_pair(kernel_size, name=f"{op_name} kernel_size")
    ph, pw = _as_pair(padding, name=f"{op_name} padding")

    if ph < 0 or pw < 0:
        raise ValueError(
            f"{op_name} padding must be non-negative; got ({ph},{pw})."
        )
    if ph > max_padding or pw > max_padding:
        raise ValueError(
            f"{op_name} padding must be <= {max_padding}; got ({ph},{pw}) "
            f"(spec {spec_ref})."
        )
    if max_kernel and (kh > max_kernel or kw > max_kernel):
        raise ValueError(
            f"{op_name} kernel kt,kf must be <= {max_kernel}; got ({kh},{kw}) "
            f"(spec {spec_ref})."
        )
    if allowed_kernels and (kh, kw) not in allowed_kernels:
        listed = sorted(set(allowed_kernels))
        raise ValueError(
            f"{op_name} kernel must be one of {listed!r}; got ({kh},{kw}) "
            f"(spec {spec_ref})."
        )


def require_reduce_size_matches_extra(
    extra_reduce_size,
    derived_reduce_size: int,
    *,
    op_name: str,
    spec_ref: str = "doc/04_算子详细规格/04_09_池化类算子.md",
) -> None:
    """Cross-check the adapter-supplied ``extra['reduce_size']`` against the
    value the kernel re-derives from its own operand shapes.

    Spec 04_09 splits AvgPool/Mean into two stages: a divide-by-N step
    (``inv/shift``, with ``N = k_t*k_f`` for AvgPool, ``∏(input.shape[d])`` for
    Mean over ``dim``) followed by a quantize-grid requantize (``M/rshift``).
    The simulator currently folds the ``1/N`` factor into the offline
    ``M/rshift`` (mathematically equivalent), so the kernel never explicitly
    multiplies by ``inv``. To keep that fold honest, the adapter must publish
    the ``N`` it used, and the kernel must reject any mismatch (or absent
    field) so a future adapter path that forgets the fold cannot silently
    produce a wrong-scale output.
    """

    if extra_reduce_size is None:
        raise ValueError(
            f"{op_name} extra['reduce_size'] is required: it pins the spec-04_09 "
            f"1/N fold the adapter folded into output multiplier/rshift "
            f"(spec {spec_ref})."
        )
    try:
        extra_value = int(extra_reduce_size)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{op_name} extra['reduce_size'] must be an integer; got "
            f"{extra_reduce_size!r}."
        ) from exc
    if extra_value <= 0:
        raise ValueError(
            f"{op_name} extra['reduce_size'] must be positive; got {extra_value}."
        )
    if extra_value != derived_reduce_size:
        raise ValueError(
            f"{op_name} extra['reduce_size']={extra_value} disagrees with "
            f"kernel-derived N={derived_reduce_size}; the adapter likely "
            f"folded a different 1/N into output multiplier/rshift "
            f"(spec {spec_ref})."
        )


def require_adaptive_avgpool_output_unit(
    *,
    output_size,
    op_name: str = "AdaptiveAvgPool2d",
) -> None:
    """Reject any ``output_size`` other than ``(1, 1)``.

    The adapter only dispatches the ``(1,1)`` case (it falls back to float
    QDQ for non-unit output sizes), but we mirror the same guard inside the
    kernel so unit tests / direct-call paths cannot accidentally request a
    spatially-resampling kernel that doesn't exist yet.
    """

    if output_size is None:
        return
    if isinstance(output_size, int):
        out_h = out_w = int(output_size)
    elif isinstance(output_size, Iterable):
        seq = tuple(int(v) for v in output_size)
        if len(seq) == 1:
            out_h = out_w = seq[0]
        elif len(seq) == 2:
            out_h, out_w = seq
        else:
            raise ValueError(
                f"{op_name} output_size must be int or 2-tuple; got {output_size!r}."
            )
    else:
        raise ValueError(
            f"{op_name} output_size must be int or 2-tuple; got {output_size!r}."
        )

    if (out_h, out_w) != (1, 1):
        raise ValueError(
            f"{op_name} INT16 kernel only supports output_size=(1,1); "
            f"got ({out_h},{out_w}). Non-unit outputs must fall back to "
            "the float QDQ path (see adapter dispatch)."
        )


def require_int32_saturated_accumulator(
    acc: torch.Tensor,
    *,
    op_name: str,
) -> None:
    """Assert ``acc`` is the int32-typed accumulator that ``requantize_int`` expects.

    Spec context (see ``doc/04_算子详细规格`` MAC sections + ADR-013/015):
    every requantizing kernel folds its zero-centered MAC / reduce sum into
    an int32 accumulator clamped to the INT32 ALU width before feeding
    ``requantize_int``. The clamp itself lives in
    :func:`aimet_torch.fixed_point.requantize.saturate_mac_accumulator` /
    ``int32_sum_sat`` / ``int32_add_sat`` (which widen to int64 first, then
    clamp, then cast to int32) and is enabled by default under
    ``INT16_FIXED_EVAL`` / ``hw_ref_mode_enabled``.

    What this gate adds is the *kernel-side dtype contract*: a kernel must
    hand ``requantize_int`` an int32 accumulator, not an int64 one and not
    a float. Catching a wrong dtype here surfaces "the kernel forgot to
    call ``saturate_mac_accumulator``" as a precise error at the kernel
    boundary instead of leaking into ``requantize_int`` as a downstream
    type/value bug.

    Limitation worth being honest about: this gate **cannot detect** a
    silent ``acc.to(torch.int32)`` wrap. Once a value is bit-cast into
    int32 it is, by construction, inside the int32 ALU width — there is
    no observable distinguishing "saturated to ``INT32_QMAX``" from
    "wrapped to a wildly different value". Defending against that
    requires keeping the int64 form alive long enough to clamp, which is
    exactly what the helpers above do; this contract's job is to enforce
    that the helpers actually ran (by refusing the int64 input that would
    arise if they didn't).

    Skipped on empty tensors so reduce-along-empty-axis kernels stay safe.
    """

    if not isinstance(acc, torch.Tensor):
        raise TypeError(
            f"{op_name} accumulator must be a torch.Tensor; got {type(acc).__name__}."
        )
    if acc.dtype != torch.int32:
        raise TypeError(
            f"{op_name} accumulator must be torch.int32 (the ALU container "
            f"width per ADR-013/015); got {acc.dtype}. Call "
            f"``saturate_mac_accumulator`` / ``int32_sum_sat`` before "
            f"``requantize_int``."
        )
    if acc.numel() == 0:
        return


def require_int64_within_int32_range(
    acc: torch.Tensor,
    *,
    op_name: str,
) -> None:
    """Assert a PWL/CLZ-style int64 intermediate stays within the INT32 ALU width.

    Companion to :func:`require_int32_saturated_accumulator`, but for the
    *intermediate* accumulators in LUT / PWL kernels — those tap points are
    deliberately kept as ``torch.int64`` so the next stage can fold further
    multiplies/adds without re-widening. ``saturate_int32`` is called after
    each tap to clamp the values, but the dtype stays int64 because the
    next op needs the headroom.

    Why a separate gate from the int32 sibling:

      * ``int64`` here is *intentional*: rejecting it (as the int32 gate
        does) would force a roundtrip to int32 only to widen back, which
        would re-introduce the very wrap risk this whole layer is trying
        to avoid.
      * Because the dtype is int64, a *value*-domain check is actually
        meaningful — we can distinguish "saturated to ``INT32_QMAX``"
        from "wrapped from 5e9 to a small magnitude" because the values
        are still in their pre-cast int64 form. This is exactly the
        check the int32-typed gate cannot do (see its docstring).

    Cost discipline: the value-domain check is gated on
    :func:`hw_ref_mode_enabled` so functional / unit tests pay only the
    O(1) dtype probe; the strict regression mode pays the extra
    ``aminmax`` to surface "the kernel forgot ``saturate_int32`` on this
    tap point". Empty tensors are skipped to keep degenerate-shape paths
    safe.
    """

    if not isinstance(acc, torch.Tensor):
        raise TypeError(
            f"{op_name} intermediate must be a torch.Tensor; got {type(acc).__name__}."
        )
    if acc.dtype != torch.int64:
        raise TypeError(
            f"{op_name} intermediate must be torch.int64 (PWL/CLZ tap point "
            f"keeps headroom for further mac); got {acc.dtype}. If this is "
            f"the requantize boundary instead of an intermediate, use "
            f"``require_int32_saturated_accumulator``."
        )
    if acc.numel() == 0:
        return
    if not hw_ref_mode_enabled():
        return
    lo, hi = torch.aminmax(acc)
    if int(lo) < INT32_QMIN or int(hi) > INT32_QMAX:
        raise ValueError(
            f"{op_name} intermediate value range "
            f"[{int(lo)}, {int(hi)}] exceeds INT32 ALU width "
            f"[{INT32_QMIN}, {INT32_QMAX}]; insert ``saturate_int32`` on "
            f"this tap point so subsequent stages see a correctly clamped "
            f"int64 carrier."
        )


def require_int64_within_signed_bit_width(
    acc: torch.Tensor,
    *,
    bit_width: int,
    op_name: str,
) -> None:
    """Assert a CLZ-style ``raw`` mac/add result fits in a custom signed bit width.

    Built for ``clz_lut.py``'s vectorized path, which carries int64
    accumulators but explicitly clamps every tap to a *configurable*
    width via ``_sat_signed_vec(value, acc_bw)``. The width comes from
    ``clz_params['internal_acc_bw']`` (default 32, but LUT manifests can
    set it to other values e.g. 24/40 to model PE BxC accumulator
    widths). Hardcoding ``[INT32_QMIN, INT32_QMAX]`` would lose that
    knob, so we accept ``bit_width`` as a parameter.

    Why a third gate (vs.  ``require_int64_within_int32_range``):

      * The CLZ path's "natural" ALU width is *not necessarily* 32 —
        the abc design lets each LUT pick its own. Routing CLZ through
        the int32 sibling would silently lose width-mismatch bugs.
      * The contract is meant to be called on the **pre-saturation**
        raw value (i.e. before ``_sat_signed_vec``), not after, because
        a check after clamp is by construction in-range and therefore
        useless. See ``clz_lut.py`` callers — every tap follows the
        ``raw → contract → sat`` pattern, so a tripped gate means the
        kernel is producing values that wouldn't fit the configured
        bit width without saturation, which is exactly the silent error
        we want to surface (typical cause: ``acc_bw`` set too small in
        the LUT manifest or a forgotten ``_sat_signed_vec`` call).

    Cost discipline mirrors the int32 sibling: only the dtype check
    runs in normal mode; the value-domain ``aminmax`` runs only under
    :func:`hw_ref_mode_enabled` so functional tests stay cheap and the
    strict regression mode catches forgotten clamps. ``bit_width`` is
    validated to keep the helper from silently passing a misuse.
    """

    if not isinstance(bit_width, int):
        raise TypeError(
            f"{op_name} bit_width must be int; got {type(bit_width).__name__}."
        )
    if bit_width < 1 or bit_width > 64:
        raise ValueError(
            f"{op_name} bit_width must be in [1, 64] (signed int64 carrier); "
            f"got {bit_width}."
        )
    if not isinstance(acc, torch.Tensor):
        raise TypeError(
            f"{op_name} accumulator must be a torch.Tensor; got {type(acc).__name__}."
        )
    if acc.dtype != torch.int64:
        raise TypeError(
            f"{op_name} accumulator must be torch.int64 (CLZ/PWL tap "
            f"carrier); got {acc.dtype}."
        )
    if acc.numel() == 0:
        return
    if not hw_ref_mode_enabled():
        return
    if bit_width == 1:
        lo_bound, hi_bound = -1, 0
    else:
        lo_bound = -(1 << (bit_width - 1))
        hi_bound = (1 << (bit_width - 1)) - 1
    lo, hi = torch.aminmax(acc)
    if int(lo) < lo_bound or int(hi) > hi_bound:
        raise ValueError(
            f"{op_name} raw value range [{int(lo)}, {int(hi)}] exceeds "
            f"signed {bit_width}-bit ALU range [{lo_bound}, {hi_bound}]; "
            f"either set ``acc_bw`` wider in the LUT manifest or insert "
            f"a missing ``_sat_signed_vec`` on this tap point."
        )

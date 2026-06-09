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
"""Central capability manifest for INT16 fixed-point operators.

Single source of truth for "which operators participate in the INT16
fixed-point path, and how". Three call sites used to maintain their own
overlapping lists:

* ``sim_utils.py`` — which modules need an output quantizer materialized.
* ``diagnose.py`` — which modules count as missing a fixed kernel.
* ``export/v2_collect.py`` — which modules produce a sidecar/binary record.

Each entry below records:

* ``status`` — one of :class:`CapabilityStatus` (``implemented`` /
  ``blackbox`` …). Future ``planned`` / ``composed`` / ``unsupported`` are
  reserved for the ``classify-unimplemented-ops`` follow-up todo.
* ``kernel_kind`` — semantic class per the ``add-kernel-contracts`` plan
  (``requantizing`` / ``same_grid_value`` / ``lookup`` / ``none``).
* ``dispatchable`` — adapter dispatches to the op (drives ``sim_utils``).
* ``int16_eval`` — allowed under ``INT16_FIXED_EVAL`` execution mode.
* ``exportable`` — produces sidecar/binary records (drives the export
  layer-gate).
* ``constraints`` — free-form notes pulled straight from the spec docs
  (``doc/04_算子详细规格``); kernels and tests should refuse configs that
  violate these.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Sequence, Tuple

import torch.nn as nn

__all__ = [
    "CapabilityStatus",
    "KernelKind",
    "OperatorCapability",
    "SUPPORTED_ACTIVATION_BITWIDTHS",
    "REQUANTIZING_COMBO_BITWIDTH_BUDGET",
    "assert_activation_bitwidth_supported",
    "assert_requantizing_combo_supported",
    "dispatchable_module_types",
    "get_capability",
    "get_manifest",
    "is_dispatchable",
    "is_exportable",
    "is_reduction_requantizing",
    "is_supported_activation_bitwidth",
    "requires_activation_bitwidth_gate",
]


# --- Activation bitwidth contract (REQUANTIZING kernels only) -----------------
#
# ``INT16_FIXED_EVAL`` describes the MAC carrier / accumulator path, NOT the
# activation grid. The activation-bitwidth contract is **per** :class:`KernelKind`:
#
#   * ``REQUANTIZING`` (Linear / Conv / AvgPool / Mean / MatMul / Mul / Divide):
#     only 8-bit activations are validated today. 16-bit activations on these
#     ops produce >2k LSB drift vs. FP32_QDQ — the offline ``M/rshift`` path
#     was tuned for 8-bit grids and the multiplier/rshift/carrier edges are
#     not safe at 16-bit (root cause TBD; tracked under the
#     audit-int16-activation-quantizer-contract follow-up).
#   * ``LOOKUP`` (sigmoid / tanh / sin / cos / sqrt / rsqrt / reciprocal /
#     square / log / exp / softmax / mish / ...): 16-bit activations are
#     validated and exercised by the existing ``test_quantized_*_int16_*``
#     suite, because the LUT/CLZ generators target 16-bit grids directly.
#   * ``SAME_GRID_VALUE`` / ``SAME_GRID_OR_REQUANT`` / ``NONE``: input and
#     output share encoding, so the activation bitwidth is whatever the
#     producer hands in — no separate gate needed here.
#
# The dispatch entry consults the manifest to decide whether to enforce, so
# this single tuple intentionally only covers the REQUANTIZING contract.
SUPPORTED_ACTIVATION_BITWIDTHS: Tuple[int, ...] = (8, 16)
"""Per-operand activation bitwidths validated for **REQUANTIZING**
kernels under ``INT16_FIXED_EVAL``.

W5 SYS-FU-1.B (PR-2, 2026-06-09) extended the legacy ``(8,)`` to
``(8, 16)`` after the W5.1 probe showed the asymmetric subset
``16+8 / 8+16`` is safe (SQNR ≥ 39 dB up to N=4096). The remaining
``16+16``-on-MAC-reduction case (Conv/Linear/MatMul) is gated
separately by :func:`assert_requantizing_combo_supported` against
:data:`REQUANTIZING_COMBO_BITWIDTH_BUDGET`. Element-wise REQUANTIZING
ops (Multiply/Divide/cross-grid Add/Subtract — N=1) and sum-only
reduction ops (AvgPool/Mean/LayerNorm — no operand×operand MAC)
accept the full ``16+16`` because the int32 ALU never reduces them.

LOOKUP / SAME_GRID kernels are NOT gated by this set; see the
per-kind contract above.
"""


def is_supported_activation_bitwidth(bitwidth: int) -> bool:
    """Return True iff ``bitwidth`` is in :data:`SUPPORTED_ACTIVATION_BITWIDTHS`.

    Note: callers must already have decided that the surrounding kernel is
    ``KernelKind.REQUANTIZING`` (the only kind this gate covers); LOOKUP /
    SAME_GRID kernels can legitimately use bitwidths outside this set.
    """

    try:
        return int(bitwidth) in SUPPORTED_ACTIVATION_BITWIDTHS
    except (TypeError, ValueError):
        return False


def requires_activation_bitwidth_gate(kernel_kind: "KernelKind") -> bool:
    """Return True for kernel kinds whose activation grid is gated.

    Only ``REQUANTIZING`` is gated today: those kernels consume the offline
    ``(multiplier, rshift)`` pair generated for an 8-bit activation grid, so
    feeding them a 16-bit grid silently mis-scales the output. LOOKUP kernels
    are intentionally NOT gated because they target 16-bit grids by design,
    and SAME_GRID kernels never re-scale.
    """

    return kernel_kind is KernelKind.REQUANTIZING


def assert_activation_bitwidth_supported(
    bitwidth: int,
    *,
    where: str,
    qualname: Optional[str] = None,
) -> None:
    """Raise ``ValueError`` when ``bitwidth`` is not a validated REQUANTIZING activation grid.

    Used by the adapter dispatch entry under ``INT16_FIXED_EVAL`` and mirrored
    by ``diagnose_int16_readiness`` so an unsupported activation bitwidth
    surfaces as a precise readiness blocker rather than as a silent dispatch
    that produces an arithmetic-but-wrong result.

    The caller is responsible for filtering to ``KernelKind.REQUANTIZING``
    modules first (see :func:`requires_activation_bitwidth_gate`); calling
    this on a LOOKUP module would falsely reject a configuration the LUT
    path already validates.

    ``where`` is a short tag (e.g. ``"input_quantizers[0]"``) and ``qualname``
    is the offending module's class qualname; both go straight into the error
    message so the caller can fix the configuration without having to read
    dispatch-internal stack frames.
    """

    if is_supported_activation_bitwidth(bitwidth):
        return
    qual = f" on {qualname}" if qualname else ""
    supported = ", ".join(str(b) for b in SUPPORTED_ACTIVATION_BITWIDTHS)
    raise ValueError(
        f"INT16_FIXED_EVAL activation bitwidth={bitwidth}{qual} at {where} "
        f"is not validated for REQUANTIZING kernels; supported bitwidths: "
        f"{{{supported}}}. Tracked under audit-int16-activation-quantizer-contract; "
        f"either reconfigure the quantizer to a supported bitwidth or extend "
        f"SUPPORTED_ACTIVATION_BITWIDTHS once the underlying multiplier/rshift "
        f"path is validated."
    )


# --- Combo bitwidth budget for REQUANTIZING-with-reduction kernels ----------
#
# Background (W5.1 root-cause probe, 2026-06-09): the HW INT32 ALU forces
# ``saturate_mac_accumulator`` to clamp the MAC accumulator to ±(2^31 − 1).
# For a single zero-centered MAC the operand product magnitude is bounded by
# ``2^(input_bw - 1) * 2^(weight_bw - 1) = 2^(input_bw + weight_bw - 2)``.
# Reduction kernels (Conv/Linear/MatMul/AvgPool/Mean/LayerNorm) sum many such
# MACs, so unless ``input_bw + weight_bw ≤ 24`` (leaving 8 bits of N
# headroom) the int32 accumulator saturates well before the spec maximum
# reduce length. Empirically:
#
#   - 8+8  combo:  per-MAC ≤ ±2^14, safe up to N≈2^17 → all real layers.
#   - 16+8 / 8+16: per-MAC ≤ ±2^22, validated SQNR ≥ 39.4 dB at N up to
#     4096 (W5.1-extended probe).
#   - 16+16:       per-MAC ≤ ±2^30, ALU saturates at N≥2 → 9.4 dB at
#     N=1024 (catastrophic).
#
# Element-wise REQUANTIZING ops (Multiply/Divide/cross-grid Add/Subtract)
# do not reduce; their effective N=1 and the int32 ceiling is comfortably
# above the worst-case ±2^30 product, so the combo gate intentionally does
# **not** apply to them.
REQUANTIZING_COMBO_BITWIDTH_BUDGET: int = 24
"""Per-MAC bitwidth ceiling (``input_bw + weight_bw``) for
``REQUANTIZING`` kernels whose capability has ``is_reduction=True``.
8 bits of headroom over the INT32 ALU ceiling cover N ≤ 2^8 = 256
without pre-saturation; W5.1 validates the budget up to N=4096 in
practice. Element-wise (non-reduction) REQUANTIZING kernels are
unaffected — their ``N=1`` reduction bypasses the int32-ALU concern.
"""

_REQUANTIZING_COMBO_VALIDATED_BITWIDTHS: Tuple[int, ...] = (8, 16)
"""Per-operand bitwidths validated by W5.1 probe for the combo gate.

Intentionally **wider** than :data:`SUPPORTED_ACTIVATION_BITWIDTHS`: the
legacy "single-bitwidth" gate (used by
:func:`assert_activation_bitwidth_supported`) still rejects 16-bit
activations because PR-1 keeps that contract intact. The new combo gate
has its own validated set (8/16 each), and PR-2 will reconcile both
gates by extending :data:`SUPPORTED_ACTIVATION_BITWIDTHS` to ``(8, 16)``
plus rerouting adapter dispatch through
:func:`assert_requantizing_combo_supported`.
"""


def is_reduction_requantizing(base_cls: type) -> bool:
    """True iff ``base_cls`` is a REQUANTIZING kernel with reduction.

    Reduction kernels (Conv/Linear/MatMul/AvgPool/Mean/LayerNorm) sum many
    MACs and are subject to :data:`REQUANTIZING_COMBO_BITWIDTH_BUDGET`.
    Element-wise REQUANTIZING (Multiply/Divide/cross-grid Add/Subtract)
    have N=1 and bypass that gate.

    Returns False for non-REQUANTIZING kinds (LOOKUP / SAME_GRID_*) and
    for unknown ``base_cls`` (callers should treat absence as "no entry,
    no contract" — same convention as :func:`get_capability`).
    """

    cap = get_capability(base_cls)
    if cap is None or cap.kernel_kind is not KernelKind.REQUANTIZING:
        return False
    return bool(cap.is_reduction)


def assert_requantizing_combo_supported(
    input_bitwidths: Sequence[int],
    weight_bitwidths: Sequence[int] = (),
    *,
    where: str,
    qualname: Optional[str] = None,
    is_reduction: bool,
) -> None:
    """Validate the (input × weight) bitwidth combo for a REQUANTIZING kernel.

    The contract differs by reduction status:

    * ``is_reduction=True`` (Conv/Linear/MatMul — the ops with a MAC sum
      reduction): every operand-pair carrying the MAC (one ``input_bw``
      with one ``weight_bw``) must satisfy
      ``input_bw + weight_bw ≤ REQUANTIZING_COMBO_BITWIDTH_BUDGET``.

      - Single-weight kernels (Linear/Conv*): pass scalar/list
        ``input_bitwidths`` and the single ``weight_bitwidths``;
        every input is paired with the (single) weight.
      - Multi-input no-weight kernels (MatMul): pass both activation
        bitwidths via ``input_bitwidths`` and leave ``weight_bitwidths``
        empty; the kernel pairs ``inputs[i]`` with ``inputs[j]`` for
        ``i < j``.

    * ``is_reduction=False``: every per-operand bitwidth must
      independently be in :data:`SUPPORTED_ACTIVATION_BITWIDTHS`, but
      no cross-budget gate. Covers element-wise REQUANTIZING
      (Multiply/Divide/cross-grid Add/Subtract — N=1) and sum-only
      reduction (AvgPool/Mean/LayerNorm — no operand×operand MAC).

    Empty bitwidth tuples short-circuit (a kernel with no operand to
    quantize has no contract to violate); callers feed ``None``
    quantizer states in as nothing to gate.

    ``where`` and ``qualname`` are propagated into the error message so
    the offending quantizer is identifiable without dispatch-internal
    stack frames (mirrors :func:`assert_activation_bitwidth_supported`).
    """

    inputs = [int(b) for b in input_bitwidths if b is not None]
    weights = [int(b) for b in weight_bitwidths if b is not None]
    if not inputs and not weights:
        return

    qual = f" on {qualname}" if qualname else ""

    for bw in inputs + weights:
        if bw not in _REQUANTIZING_COMBO_VALIDATED_BITWIDTHS:
            supported = ", ".join(
                str(b) for b in _REQUANTIZING_COMBO_VALIDATED_BITWIDTHS
            )
            raise ValueError(
                f"INT16_FIXED_EVAL bitwidth={bw}{qual} at {where} is not "
                f"validated for REQUANTIZING kernels; supported bitwidths: "
                f"{{{supported}}}."
            )

    if not is_reduction:
        return

    if weights:
        pairs = [(ib, wb) for ib in inputs for wb in weights]
    elif len(inputs) >= 2:
        pairs = [
            (inputs[i], inputs[j])
            for i in range(len(inputs))
            for j in range(i + 1, len(inputs))
        ]
    else:
        pairs = []

    for ib, wb in pairs:
        if ib + wb > REQUANTIZING_COMBO_BITWIDTH_BUDGET:
            raise ValueError(
                f"INT16_FIXED_EVAL combo (operand_bw={ib}, operand_bw={wb})"
                f"{qual} at {where} exceeds the REQUANTIZING-with-MAC-"
                f"reduction budget operand_bw + operand_bw ≤ "
                f"{REQUANTIZING_COMBO_BITWIDTH_BUDGET} (W5.1 root-cause: "
                f"INT32 ALU saturates at full 16+16 MAC). Validated "
                f"combos: 8+8 / 16+8 / 8+16. Either downgrade one side "
                f"to 8-bit or wait for the W5 ALU upgrade."
            )


class CapabilityStatus(str, Enum):
    """Lifecycle status of an operator on the INT16 fixed-point path."""

    IMPLEMENTED = "implemented"
    """Has a registered Python fixed-point kernel."""

    BLACKBOX = "blackbox"
    """Native blackbox contract (e.g. ``QuantGRU``); not a registered kernel."""

    # Reserved for ``classify-unimplemented-ops`` (P2 in plan):
    #   PLANNED = "planned"
    #   COMPOSED = "composed"
    #   UNSUPPORTED = "unsupported"
    #   SHAPE_META_ONLY = "shape_meta_only"


class KernelKind(str, Enum):
    """Semantic kernel category per the ``add-kernel-contracts`` plan."""

    REQUANTIZING = "requantizing"
    """Output via ``M/rshift`` requantize (Conv/Linear/MatMul/Mul/Divide/AvgPool/Mean)."""

    SAME_GRID_VALUE = "same_grid_value"
    """Strict same-grid: input/output share scale/zero_point/qmin/qmax and the
    kernel never requantizes (MaxPool comparator tree per spec 04_09; pure
    layout ops Reshape/Permute/Flatten/Identity).
    """

    SAME_GRID_OR_REQUANT = "same_grid_or_requant"
    """Same-grid optimised, requantize-capable. Input/output usually share the
    quant grid (the kernel takes the no-multiplier fast path), but the kernel
    also accepts an output ``(multiplier, rshift)`` and aligns input → output
    grid through ``align_centered_int32_to_output`` + ``requantize_int``.
    Covers value-clamp activations (ReLU/ReLU6/Hardtanh/Clamp/Clip),
    integer-domain unaries (Abs/Sign), Pad (constant fill on aligned grid),
    Concat (per-input grid alignment), and Dropout (no-op or grid relabel).
    Distinct from :attr:`SAME_GRID_VALUE` because contract checks must NOT
    require strict grid equality on these ops.
    """

    LOOKUP = "lookup"
    """PWL/CLZ/LUT activations (sigmoid/tanh/sin/cos/sqrt/reciprocal/...)."""

    NONE = "none"
    """No fixed-point kernel (blackbox contract or planned op)."""


@dataclass(frozen=True)
class OperatorCapability:
    """Per-operator capability record."""

    qualname: str
    status: CapabilityStatus
    kernel_kind: KernelKind
    dispatchable: bool
    int16_eval: bool
    exportable: bool
    constraints: Tuple[str, ...] = field(default_factory=tuple)
    is_reduction: bool = False
    """``True`` for ``REQUANTIZING`` kernels whose MAC has a reduction
    sum (Conv/Linear/MatMul) and is therefore subject to
    :data:`REQUANTIZING_COMBO_BITWIDTH_BUDGET`. Element-wise
    REQUANTIZING (Multiply/Divide/cross-grid Add/Subtract) and
    sum-only reduction (AvgPool/Mean/LayerNorm — no operand×operand
    MAC) keep the default ``False``. Non-REQUANTIZING kinds also
    leave it ``False`` (the field is undefined for them; check
    ``kernel_kind`` first via :func:`is_reduction_requantizing`).
    """


@functools.lru_cache(maxsize=1)
def get_manifest() -> Dict[type, OperatorCapability]:
    """Return the full ``{module_class: OperatorCapability}`` manifest.

    Imports are deferred to first access so importing this module is cheap
    and free of v2-stack side effects.
    """

    # pylint: disable=import-outside-toplevel
    from aimet_torch._base.nn.modules import custom

    entries: Dict[type, OperatorCapability] = {}

    def _register(
        cls: type,
        *,
        status: CapabilityStatus,
        kernel_kind: KernelKind,
        dispatchable: bool = True,
        int16_eval: bool = True,
        exportable: bool = True,
        constraints: Tuple[str, ...] = (),
        is_reduction: bool = False,
    ) -> None:
        entries[cls] = OperatorCapability(
            qualname=f"{cls.__module__}.{cls.__qualname__}",
            status=status,
            kernel_kind=kernel_kind,
            dispatchable=dispatchable,
            int16_eval=int16_eval,
            exportable=exportable,
            constraints=tuple(constraints),
            is_reduction=is_reduction,
        )

    # --- Weighted requantizing (doc/04_01) ----------------------------------
    for cls in (nn.Linear, nn.Conv1d, nn.Conv2d):
        _register(
            cls,
            status=CapabilityStatus.IMPLEMENTED,
            kernel_kind=KernelKind.REQUANTIZING,
            is_reduction=True,
            constraints=(
                "weight zero_point must equal 0 per spec 04_01 (Z_w=0); "
                "input zero_point folded into q_bias",
            ),
        )

    # ``nn.Conv3d`` has a registered kernel (shares the Conv im2col path) but
    # the project does not currently use 3D convolution. The kernel is kept
    # for ad-hoc unit tests; adapter dispatch / int16_eval / export are all
    # disabled so the production path does not accidentally route 3D conv
    # without explicit precision validation. Re-enable by flipping the three
    # flags once Conv3d precision is recorded in precision_validation.md.
    _register(
        nn.Conv3d,
        status=CapabilityStatus.IMPLEMENTED,
        kernel_kind=KernelKind.REQUANTIZING,
        is_reduction=True,
        dispatchable=False,
        int16_eval=False,
        exportable=False,
        constraints=(
            "weight zero_point must equal 0 per spec 04_01 (Z_w=0); "
            "input zero_point folded into q_bias",
            "out-of-scope: project does not currently use 3D conv; "
            "precision_validation.md P4 section reserves the slot but no "
            "snapshot is taken until the path is needed",
        ),
    )

    # --- Pooling (doc/04_09) -----------------------------------------------
    _maxpool_constraints = (
        "input/output share scale/zero_point/qmin/qmax per 04_09 "
        "(comparator-only HW, no M/rshift)",
        "kernel kt,kf<=3; padding 0..3",
    )
    _register(
        nn.MaxPool2d,
        status=CapabilityStatus.IMPLEMENTED,
        kernel_kind=KernelKind.SAME_GRID_VALUE,
        constraints=_maxpool_constraints,
    )
    # ``custom.MaxPool2d`` is a ``create_wrapper_module`` over
    # ``F.max_pool2d`` (see _base/nn/modules/custom.py); it is NOT a subclass
    # of ``nn.MaxPool2d``. The adapter dispatch path in
    # ``aimet_torch/v2/quantization/affine/fixed_point/adapter.py`` only
    # special-cases ``base_cls is nn.MaxPool2d`` for the same-grid sentinel
    # and pulls ``kernel_size``/``padding``/``stride`` from the qmodule
    # attributes — the functional wrapper doesn't expose those, so the
    # custom variant cannot dispatch yet. Tracked by plan item
    # ``route-custom-pool-through-adapter``. Kernel is still registered for
    # direct unit-test use.
    _register(
        custom.MaxPool2d,
        status=CapabilityStatus.IMPLEMENTED,
        kernel_kind=KernelKind.SAME_GRID_VALUE,
        dispatchable=False,
        int16_eval=False,
        exportable=False,
        constraints=_maxpool_constraints
        + (
            "adapter does not yet route custom.MaxPool2d (functional wrapper "
            "needs args/kwargs unpack); see plan item "
            "route-custom-pool-through-adapter",
        ),
    )
    _avgpool_constraints = (
        "kernel ∈ {2x2, 4x4, 4x2, 2x4} per 04_09",
        "padding 0..3",
        "1/N folded into output real_multiplier",
    )
    _register(
        nn.AvgPool2d,
        status=CapabilityStatus.IMPLEMENTED,
        kernel_kind=KernelKind.REQUANTIZING,
        constraints=_avgpool_constraints,
    )
    _register(
        custom.AvgPool2d,
        status=CapabilityStatus.IMPLEMENTED,
        kernel_kind=KernelKind.REQUANTIZING,
        dispatchable=False,
        int16_eval=False,
        exportable=False,
        constraints=_avgpool_constraints
        + (
            "adapter does not yet route custom.AvgPool2d (functional wrapper "
            "needs args/kwargs unpack); see plan item "
            "route-custom-pool-through-adapter",
        ),
    )
    _register(
        custom.AdaptiveAvgPool2d,
        status=CapabilityStatus.IMPLEMENTED,
        kernel_kind=KernelKind.REQUANTIZING,
        constraints=(
            "output_size=(1,1) only; non-(1,1) refused at dispatch",
            "spatial mean over (H,W) with keepdim=True",
        ),
    )

    # --- Value-clamp activations (dual-mode: same-grid fast path + optional
    # output requantize when ``(multiplier, rshift)`` is present). ----------
    for cls in (nn.ReLU, nn.ReLU6, nn.Hardtanh, custom.Clamp, custom.Clip):
        _register(
            cls,
            status=CapabilityStatus.IMPLEMENTED,
            kernel_kind=KernelKind.SAME_GRID_OR_REQUANT,
            constraints=(
                "kernel takes the same-grid path when output_encoding has "
                "no multiplier; otherwise aligns input → output via "
                "align_centered_int32_to_output + requantize_int",
            ),
        )

    # --- Integer-domain unary: sign (compare-only, integer-domain) ----------
    _register(
        custom.ElementwiseUnarySign,
        status=CapabilityStatus.IMPLEMENTED,
        kernel_kind=KernelKind.SAME_GRID_OR_REQUANT,
    )

    # --- Normalization: LayerNorm (doc/04_05 §4.5.4) --------------------------
    # Kernel at ``norm.LayerNormInt16Kernel`` is **spec §4.5.4 aligned** —
    # a 4-step pipeline:
    #   1. dequant INT16 input to fp32.
    #   2. fp32 ``variance`` instruction simulation (mu, σ² on the reduce).
    #      Spec §4.5.2 doesn't constrain the intermediate carrier; fp32 is
    #      the project's reference implementation of the base instruction.
    #   3. ``q_inv = rsqrt_lut(σ² + ε)`` via the RSqrt CLZ LUT — the only
    #      step spec §4.5.4 line 384 explicitly requires be lookup-based.
    #      Re-uses the same ``abc_lut-shuai/lut_int_general/output/lut_test/
    #      rsqrt_clz_lut.json`` asset that ``custom.RSqrt`` consumes.
    #   4. fp32 affine ``γ·(x-μ)·q_inv + β`` then requantize to output.
    #
    # We mark it ``REQUANTIZING`` because input/output encodings differ
    # (LN changes magnitude via 1/std); the dispatcher then enforces the
    # 8-bit activation gate, which matches the precision validation suite
    # (cos≥0.999, lsb_max ≤ 8.0 — bound by RSqrt LUT PWL residual ×
    # ``γ/std`` amplification).
    #
    # Two follow-ups distinguish the residual gap to the *hardware*
    # integer LayerNorm path:
    #   * ``FU-LAYERNORM-AFFINE-INTEGER`` — replace fp32 affine with the
    #     ``M/rshift`` folded integer affine (spec §4.5.4 line 422-428).
    #     The current fp32 affine introduces *no* residual on top of the
    #     RSqrt LUT (proven by the LUT-vs-float-oracle gate at
    #     ``test_layernorm_spec_aligned_diff_from_float_oracle_bounded``).
    #   * ``FU-LAYERNORM-DSP-PARITY`` — replace fp32 variance with the
    #     hardware in-line integer ``variance`` base instruction (spec
    #     §4.5.2). Needed only when bit-parity with a DSP single-
    #     instruction LayerNorm is required.
    _register(
        nn.LayerNorm,
        status=CapabilityStatus.IMPLEMENTED,
        kernel_kind=KernelKind.REQUANTIZING,
        constraints=(
            "spec 04_05 §4.5.4: nn.LayerNorm via **full spec bit-parity "
            "integer pipeline** — §4.5.2 in-line integer variance "
            "(_inline_integer_variance) + RSqrt CLZ LUT (integer-in / "
            "integer-out) + spec line 422-428 16-bit M/rshift integer "
            "affine (_integer_affine).",
            "per-tensor output encoding only; per-channel LN not in scope",
            "γ / β offline-quantized to int16 symmetric per-tensor (Z=0) "
            "with S_γ = γ.abs().max()/32767 / S_β = β.abs().max()/32767 "
            "(Default A; tracked in precision_validation.md ## nn.LayerNorm)",
            "S_μ = S_x, Z_μ = Z_x (mean shares input grid; Default A)",
            "S_var, Z_var = RSqrt CLZ LUT input grid (zero downstream rescale)",
            "RSqrt CLZ LUT resolved from extra['rsqrt_clz_lut'] or "
            "abc_lut-shuai/lut_int_general/output/lut_test/rsqrt_clz_lut.json",
            # KNOWN_LIMIT 1 — RSqrt CLZ LUT physical fit ceiling, shared
            # with P7 custom.RSqrt. Recovery requires hardware-side LUT
            # segment count upgrade.
            #
            # KNOWN_LIMIT 2 — spec line 414's 16-bit M + max_rshift=31
            # physical ceiling. Under i16 calibrated grid + γ quant,
            # α_x = S_γ·S_x·S_inv/S_y ≈ 1e-8 falls below the M precision
            # floor; multiplier degrades to ~5-bit and affine path lsb
            # climbs to ~1000 LSB. Cosine still clears 0.9999. Per-grid
            # ceilings (affine / noaffine split) registered in
            # thresholds.LAYERNORM_VS_FP32_PER_GRID_LIMITS. Recovery
            # tracked as FU-LAYERNORM-ALPHA-X-CALIBRATION (wider M ABI,
            # alternate factoring of α_x, or γ-grid recalibration).",
        ),
    )

    # --- Abs (integer-abs SAME_GRID_OR_REQUANT) ------------------------------
    # ``custom.Abs`` follows spec ``04_03 §4.3.5 abs``: integer-abs path
    # ``x' = |q_x − Z_x|`` then ``y_q = sat(x' · M ≫ rshift) + Z_y``. Kernel
    # is registered to :class:`AbsInt16Kernel` in ``kernels/eltwise.py`` and
    # shares the ReLU/Clamp hot path. The earlier PWL `_LutInt16Kernel`
    # implementation has been retired (kept here in a comment for diff
    # archaeology); ``thresholds.PWL_VS_ANALYTIC_PER_FN_LIMITS["abs"]`` is
    # now dead code for the runtime path but stays around as a sanity check
    # for any historical PWL-export consumers.
    _register(
        custom.Abs,
        status=CapabilityStatus.IMPLEMENTED,
        kernel_kind=KernelKind.SAME_GRID_OR_REQUANT,
        constraints=(
            "integer-abs path (spec 04_03 §4.3.5): "
            "x' = |q_x − Z_x|; y_q = sat(x' · M ≫ rshift) + Z_y",
        ),
    )

    # --- Lookup activations: PWL ------------------------------------------
    # Entries below are *base operator classes* (not quantized wrapper
    # classes). Two sources coexist:
    #   * ``torch.nn.*`` — PyTorch-native activations (Sigmoid/Tanh/...).
    #   * ``aimet_torch._base.nn.modules.custom.*`` — AIMET wrapper modules
    #     for ops PyTorch exposes only as functions (Exponential/Log/...).
    # ``QuantizationMixin.qcls_to_cls`` resolves any quantized module back
    # to one of these base classes, which is what the manifest keys on.
    _PWL_BASE_CLASSES = (
        nn.Sigmoid,
        nn.Tanh,
        nn.GELU,
        nn.SiLU,
        nn.Mish,
        nn.Softplus,
        nn.Hardsigmoid,
        nn.Hardswish,
        nn.LeakyReLU,
        nn.PReLU,
        custom.Exponential,
    )
    for cls in _PWL_BASE_CLASSES:
        _register(
            cls,
            status=CapabilityStatus.IMPLEMENTED,
            kernel_kind=KernelKind.LOOKUP,
        )

    # log is currently handled via PWL (``adapter._pwl_activation_fn`` maps
    # ``custom.Log`` to ``torch.log``), but the spec classifies it as CLZ
    # (``LUT_Binary_Storage_General.md`` §5: log uses ``q_ln2``/``q_ln_sx``
    # head slots, with the special-case formula
    # ``q_y = round_shift(y_offset · r_q, r_shift) + E · q_ln2 + q_ln_sx + zp``).
    # The blocker is *upstream LUT data*, not software:
    #   - ``abc_lut-shuai/lut_int_general/output/lut_test/log_lut.json`` is a
    #     PWL 16-segment asset (same family as sigmoid/gelu); same for
    #     ``lut_int_po2/output/lut_test/log_lut*.json``.
    #   - ``abc_lut-shuai/lut_fp/output/fp32_test/log_normalized_fp32_lut.json``
    #     is a CLZ-style mantissa LUT (input ∈ [1.0, 2.0], output ∈ [0, ln 2])
    #     — **conceptually correct** but in fp32, **not yet quantized to int**.
    #   - ``lut_int_general/quantization/clz_normalized_fitter.py`` only
    #     supports ``reciprocal/sqrt/rsqrt/power_2``; log fitting is not
    #     implemented in the CLZ fitter yet.
    # Implementing ``LogInt16ClzKernel`` end-to-end requires:
    #   1. abc_lut-shuai data-science side: extend ``clz_normalized_fitter``
    #      with a ``log`` branch and emit ``log_clz_lut.json`` (q_norm/n_norm/
    #      r_q/r_shift/q_ln2/q_ln_sx + 16 mantissa segments) — the fp32
    #      normalized asset already gives the segment fit; the missing piece
    #      is the int16 quantization plus q_ln2/q_ln_sx constants.
    #   2. software side (this repo): extend ``denormalize_clz`` with a
    #      ``"log"`` branch (~30 LOC) and register ``LogInt16ClzKernel``.
    #   3. tests: spec golden LUT bit-exact oracle in test_clz_*_golden.py.
    # Until step 1 lands upstream the runtime is correctly stuck on PWL —
    # tracked by the ``review-nonlinear-lut`` plan item.
    _register(
        custom.Log,
        status=CapabilityStatus.IMPLEMENTED,
        kernel_kind=KernelKind.LOOKUP,
        constraints=(
            "spec classifies log as CLZ (q_ln2/q_ln_sx); runtime stays on "
            "PWL because abc_lut-shuai upstream has no log_clz_lut.json — "
            "see plan item review-nonlinear-lut",
        ),
    )

    # --- Lookup activations: periodic (sin/cos) ----------------------------
    for cls in (custom.Sin, custom.Cos):
        _register(
            cls,
            status=CapabilityStatus.IMPLEMENTED,
            kernel_kind=KernelKind.LOOKUP,
            constraints=(
                "phase fold using q_inv_2pi/q_2pi/q_halfpi (LUT_Binary_Storage_General §4)",
            ),
        )

    # --- Lookup activations: CLZ-normalized -------------------------------
    for cls in (custom.Sqrt, custom.RSqrt, custom.Reciprocal, custom.Square):
        _register(
            cls,
            status=CapabilityStatus.IMPLEMENTED,
            kernel_kind=KernelKind.LOOKUP,
            constraints=(
                "CLZ normalization (LUT_Binary_Storage_General §5: "
                "q_norm/n_norm + r_q/r_shift)",
            ),
        )

    # --- Softmax (composed lookup over exp + reciprocal) -------------------
    _register(
        nn.Softmax,
        status=CapabilityStatus.IMPLEMENTED,
        kernel_kind=KernelKind.LOOKUP,
    )

    # --- Pure layout-only data movement (strict same-grid) ----------------
    # These kernels call ``require_same_grid_encoding`` unconditionally and
    # must NOT receive a different output encoding.
    for cls in (
        nn.Flatten,
        custom.Reshape,
        custom.Permute,
        nn.Identity,
    ):
        _register(
            cls,
            status=CapabilityStatus.IMPLEMENTED,
            kernel_kind=KernelKind.SAME_GRID_VALUE,
            constraints=("input/output share encoding (no requantize)",),
        )

    # --- Resize / Upsample (doc/04_10) ------------------------------------
    # Nearest-neighbour resize is a pure spatial-index remap: spec
    # ``04_10 §4.10.2`` says "直接复制量化值，无需重量化". Therefore
    # SAME_GRID_VALUE — input/output must share encoding. Bilinear (§4.10.1)
    # is REQUANTIZING and is intentionally NOT in the manifest yet; the kernel
    # at ``shape_ops.UpsampleInt16Kernel`` refuses any mode other than
    # ``nearest`` so a Bilinear ``nn.Upsample`` falls back to FP32_QDQ via the
    # adapter null-dispatch path instead of going through SAME_GRID_VALUE.
    _resize_nearest_constraints = (
        "spec 04_10 §4.10.2: mode='nearest' only (byte-stream identity); "
        "Bilinear path not yet implemented and refused at kernel entry",
        "input/output share encoding (no requantize)",
    )
    _register(
        nn.UpsamplingNearest2d,
        status=CapabilityStatus.IMPLEMENTED,
        kernel_kind=KernelKind.SAME_GRID_VALUE,
        constraints=_resize_nearest_constraints,
    )
    _register(
        nn.Upsample,
        status=CapabilityStatus.IMPLEMENTED,
        kernel_kind=KernelKind.SAME_GRID_VALUE,
        constraints=_resize_nearest_constraints,
    )

    # --- Concat / Pad: dual-mode (per-input grid alignment + optional
    # output requantize). Not strict same-grid because ``Concat`` aligns
    # different input grids to the output via ``align_centered_int32_to_output``
    # and ``Pad`` may go through ``_requantize_identity_output`` when the
    # output encoding carries multiplier/rshift. ---------------------------
    _register(
        custom.Concat,
        status=CapabilityStatus.IMPLEMENTED,
        kernel_kind=KernelKind.SAME_GRID_OR_REQUANT,
        constraints=("each branch aligned to output grid before concat",),
    )
    _register(
        custom.Pad,
        status=CapabilityStatus.IMPLEMENTED,
        kernel_kind=KernelKind.SAME_GRID_OR_REQUANT,
        constraints=(
            "mode='constant' only",
            "input requantized to output grid before constant fill",
        ),
    )

    # --- Dropout: identity in eval; kernel allows multiplier path so the
    # graph can place a Dropout between two different quant grids. ---------
    _register(
        nn.Dropout,
        status=CapabilityStatus.IMPLEMENTED,
        kernel_kind=KernelKind.SAME_GRID_OR_REQUANT,
        constraints=(
            "no-op in eval; kernel may relabel via _requantize_identity_output",
        ),
    )

    # --- Reduce (doc/04_06) ------------------------------------------------
    _register(
        custom.Mean,
        status=CapabilityStatus.IMPLEMENTED,
        kernel_kind=KernelKind.REQUANTIZING,
        constraints=("1/N folded into output real_multiplier",),
    )

    # --- Elementwise / matmul requantizing --------------------------------
    for cls in (custom.Add, custom.Subtract):
        _register(
            cls,
            status=CapabilityStatus.IMPLEMENTED,
            kernel_kind=KernelKind.REQUANTIZING,
            constraints=("inputs aligned to output grid then summed",),
        )
    for cls in (custom.Multiply, custom.Divide):
        _register(
            cls,
            status=CapabilityStatus.IMPLEMENTED,
            kernel_kind=KernelKind.REQUANTIZING,
        )
    _register(
        custom.MatMul,
        status=CapabilityStatus.IMPLEMENTED,
        kernel_kind=KernelKind.REQUANTIZING,
        is_reduction=True,
        constraints=(
            "two activation inputs (no parameter weight); reduction over "
            "the inner contracted dim — combo gate pairs both activation "
            "bitwidths against REQUANTIZING_COMBO_BITWIDTH_BUDGET",
        ),
    )

    # --- Native blackbox: QuantGRU ----------------------------------------
    try:
        from aimet_torch.v2.nn.modules.custom import QuantizedQuantGRU
    except ImportError:
        QuantizedQuantGRU = None
    if QuantizedQuantGRU is not None:
        _register(
            QuantizedQuantGRU,
            status=CapabilityStatus.BLACKBOX,
            kernel_kind=KernelKind.NONE,
            dispatchable=False,
            exportable=False,
            constraints=(
                "native blackbox contract; not a registered fixed_point kernel",
            ),
        )

    return entries


def dispatchable_module_types() -> Tuple[type, ...]:
    """Tuple of module classes the INT16 adapter dispatches to."""

    return tuple(cls for cls, cap in get_manifest().items() if cap.dispatchable)


def get_capability(base_cls: type) -> Optional[OperatorCapability]:
    """Return the capability record for ``base_cls`` (or ``None`` if absent)."""

    return get_manifest().get(base_cls)


def is_dispatchable(base_cls: type) -> bool:
    """True iff ``base_cls`` is in the manifest with ``dispatchable=True``."""

    cap = get_capability(base_cls)
    return cap is not None and cap.dispatchable


def is_exportable(base_cls: type) -> bool:
    """True iff ``base_cls`` is in the manifest with ``exportable=True``."""

    cap = get_capability(base_cls)
    return cap is not None and cap.exportable

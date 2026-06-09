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
"""Smoke tests for the central INT16 fixed-point capability manifest."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from aimet_torch._base.nn.modules import custom  # noqa: E402

from aimet_torch.fixed_point.capabilities import (  # noqa: E402
    CapabilityStatus,
    KernelKind,
    OperatorCapability,
    REQUANTIZING_COMBO_BITWIDTH_BUDGET,
    assert_requantizing_combo_supported,
    dispatchable_module_types,
    get_capability,
    get_manifest,
    is_dispatchable,
    is_exportable,
    is_reduction_requantizing,
)
from aimet_torch.fixed_point.encoding import OutputEncoding  # noqa: E402
from aimet_torch.fixed_point.kernels._contracts import (  # noqa: E402
    explicitly_handled_kernel_kinds,
    require_kernel_kind_encoding_contract,
)
from aimet_torch.fixed_point.sim_utils import _dispatchable_module_types  # noqa: E402
from aimet_torch.fixed_point.tensor import Int16QuantizedTensor  # noqa: E402


def _int16_tensor(scale: float = 1.0, zero_point: int = 0) -> Int16QuantizedTensor:
    return Int16QuantizedTensor(
        int_repr=torch.tensor([0], dtype=torch.int16),
        scale=torch.tensor(scale, dtype=torch.float32),
        zero_point=torch.tensor(zero_point, dtype=torch.int32),
    )


def _output_encoding(
    *,
    scale: float = 1.0,
    zero_point: int = 0,
    with_requant: bool = False,
    partial_requant: bool = False,
) -> OutputEncoding:
    multiplier = None
    rshift = None
    if with_requant or partial_requant:
        multiplier = torch.tensor(32768, dtype=torch.uint16)
    if with_requant:
        rshift = torch.tensor(15, dtype=torch.int8)
    return OutputEncoding(
        scale=torch.tensor(scale, dtype=torch.float32),
        zero_point=torch.tensor(zero_point, dtype=torch.int32),
        qmin=-32768,
        qmax=32767,
        multiplier=multiplier,
        rshift=rshift,
    )


def test_manifest_entries_have_consistent_fields():
    """Every entry must carry valid enums and a non-empty qualname."""
    manifest = get_manifest()
    assert len(manifest) > 0
    for cls, capability in manifest.items():
        assert isinstance(capability, OperatorCapability)
        assert capability.qualname
        assert isinstance(capability.status, CapabilityStatus)
        assert isinstance(capability.kernel_kind, KernelKind)
        assert isinstance(capability.constraints, tuple)
        # blackbox ops must not advertise a Python kernel kind.
        if capability.status is CapabilityStatus.BLACKBOX:
            assert capability.kernel_kind is KernelKind.NONE
            assert not capability.dispatchable
            assert not capability.exportable


def test_kernel_kind_contract_dispatch_handles_every_kind_explicitly():
    """Adding a KernelKind must force the contracts switch to be updated."""

    assert set(explicitly_handled_kernel_kinds()) == set(KernelKind)


def test_same_grid_or_requant_contract_is_not_strict_same_grid_when_requantizing():
    """``SAME_GRID_OR_REQUANT`` must not accidentally behave like strict
    ``SAME_GRID_VALUE`` once a complete ``(multiplier, rshift)`` pair is
    present. This is the guard for ReLU/Clamp/Pad/Concat/Dropout style kernels.
    """

    tensor = _int16_tensor(scale=1.0, zero_point=0)
    require_kernel_kind_encoding_contract(
        KernelKind.SAME_GRID_OR_REQUANT,
        _output_encoding(scale=0.5, zero_point=0, with_requant=True),
        op_name="dual-mode-op",
        tensor=tensor,
    )

    with pytest.raises(ValueError, match="dual-mode-op input/output scale"):
        require_kernel_kind_encoding_contract(
            KernelKind.SAME_GRID_OR_REQUANT,
            _output_encoding(scale=0.5, zero_point=0, with_requant=False),
            op_name="dual-mode-op",
            tensor=tensor,
        )

    with pytest.raises(ValueError, match="multiplier and rshift together"):
        require_kernel_kind_encoding_contract(
            KernelKind.SAME_GRID_OR_REQUANT,
            _output_encoding(scale=0.5, zero_point=0, partial_requant=True),
            op_name="dual-mode-op",
            tensor=tensor,
        )


def test_core_implemented_ops_are_dispatchable_and_exportable():
    """Spec-mandated ops appear with the expected kernel kinds."""
    expected = {
        nn.Linear: KernelKind.REQUANTIZING,
        nn.Conv2d: KernelKind.REQUANTIZING,
        nn.MaxPool2d: KernelKind.SAME_GRID_VALUE,
        nn.AvgPool2d: KernelKind.REQUANTIZING,
        custom.Add: KernelKind.REQUANTIZING,
        custom.MatMul: KernelKind.REQUANTIZING,
        custom.Sqrt: KernelKind.LOOKUP,
        nn.Sigmoid: KernelKind.LOOKUP,
        custom.Pad: KernelKind.SAME_GRID_OR_REQUANT,
    }
    for cls, expected_kind in expected.items():
        cap = get_capability(cls)
        assert cap is not None, cls
        assert cap.status is CapabilityStatus.IMPLEMENTED
        assert cap.kernel_kind is expected_kind
        assert cap.dispatchable is True
        assert cap.exportable is True
        assert is_dispatchable(cls)
        assert is_exportable(cls)


def test_value_clamp_activations_are_dual_mode():
    """ReLU/Hardtanh/Clamp/Clip kernels accept both same-grid and requantize
    paths — they MUST NOT be classified as strict ``SAME_GRID_VALUE`` or the
    contracts layer would (incorrectly) require strict grid equality."""

    for cls in (nn.ReLU, nn.ReLU6, nn.Hardtanh, custom.Clamp, custom.Clip):
        cap = get_capability(cls)
        assert cap is not None, cls
        assert cap.kernel_kind is KernelKind.SAME_GRID_OR_REQUANT, cls


def test_pad_concat_dropout_are_dual_mode():
    """Pad/Concat/Dropout: kernel calls ``_requantize_identity_output`` /
    ``align_centered_int32_to_output`` and accepts a different output grid
    when the encoding carries ``(multiplier, rshift)``."""

    for cls in (custom.Pad, custom.Concat, nn.Dropout):
        cap = get_capability(cls)
        assert cap is not None, cls
        assert cap.kernel_kind is KernelKind.SAME_GRID_OR_REQUANT, cls


def test_layout_only_ops_are_strict_same_grid():
    """Reshape/Permute/Flatten/Identity must remain strict ``SAME_GRID_VALUE``
    because their kernels call ``require_same_grid_encoding`` unconditionally."""

    for cls in (nn.Flatten, custom.Reshape, custom.Permute, nn.Identity):
        cap = get_capability(cls)
        assert cap is not None, cls
        assert cap.kernel_kind is KernelKind.SAME_GRID_VALUE, cls


def test_custom_pool_wrappers_are_marked_non_dispatchable():
    """``custom.MaxPool2d``/``AvgPool2d`` are functional wrappers without
    ``kernel_size``/``padding`` attributes; the adapter only special-cases
    the ``nn.*`` versions today, so the manifest must reflect that or
    callers will get silent-numerical-corruption from the default
    ``real_m=x_scale/y_scale`` branch."""

    for cls in (custom.MaxPool2d, custom.AvgPool2d):
        cap = get_capability(cls)
        assert cap is not None, cls
        assert cap.dispatchable is False, cls
        assert cap.exportable is False, cls
        joined = " | ".join(cap.constraints)
        assert "route-custom-pool-through-adapter" in joined, cls


def test_maxpool_has_same_grid_constraint_in_manifest():
    """``add-kernel-contracts`` plan: spec 04_09 same-grid rule must be visible."""
    cap = get_capability(nn.MaxPool2d)
    assert cap is not None
    joined = " | ".join(cap.constraints)
    assert "scale/zero_point/qmin/qmax" in joined or "comparator-only" in joined


def test_avgpool_kernel_size_constraint_in_manifest():
    """Spec 04_09 restricts AvgPool kernels to {2x2, 4x4, 4x2, 2x4}."""
    cap = get_capability(nn.AvgPool2d)
    assert cap is not None
    joined = " | ".join(cap.constraints)
    assert "2x2" in joined and "4x4" in joined


def test_quantgru_is_blackbox_when_available():
    try:
        from aimet_torch.v2.nn.modules.custom import QuantizedQuantGRU
    except ImportError:
        pytest.skip("QuantizedQuantGRU not available")
    cap = get_capability(QuantizedQuantGRU)
    assert cap is not None
    assert cap.status is CapabilityStatus.BLACKBOX
    assert cap.kernel_kind is KernelKind.NONE
    assert not cap.dispatchable
    assert not cap.exportable


def test_sim_utils_dispatchable_aligns_with_manifest():
    """``sim_utils._dispatchable_module_types`` must come straight from the manifest.

    Guards against drift between the manifest and the sim_utils helper (the
    whole point of P0 was to remove the duplicated list).
    """
    sim_set = set(_dispatchable_module_types())
    manifest_set = set(dispatchable_module_types())
    assert sim_set == manifest_set


def test_unknown_class_returns_none():
    class _NotInManifest(nn.Module):
        pass

    assert get_capability(_NotInManifest) is None
    assert not is_dispatchable(_NotInManifest)
    assert not is_exportable(_NotInManifest)


# ============================================================================
# PR-1 (W5 SYS-FU-1.B): is_reduction flag + combo bitwidth budget
# ----------------------------------------------------------------------------
# Pure-additive API: these tests pin the new manifest property and the
# REQUANTIZING-with-MAC-reduction combo gate without changing the legacy
# ``assert_activation_bitwidth_supported`` contract (which PR-2 will adapt).
# ============================================================================


def test_is_reduction_marks_only_mac_kernels():
    """``is_reduction=True`` is exactly the set of REQUANTIZING ops with a
    MAC sum (Conv/Linear/MatMul).

    AvgPool/Mean/LayerNorm are REQUANTIZING but reduce by sum-only with no
    operand×operand multiplication; element-wise REQUANTIZING ops
    (Multiply/Divide) have N=1; both keep ``is_reduction=False``. Pinning
    this prevents drift when new REQUANTIZING ops are added (the budget
    gate must opt in via the manifest, not by default).
    """

    expected_reduction = {
        nn.Linear,
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        custom.MatMul,
    }
    expected_non_reduction = {
        nn.AvgPool2d,
        custom.Mean,
        nn.LayerNorm,
        custom.Multiply,
        custom.Divide,
        custom.Add,
        custom.Subtract,
    }
    for cls in expected_reduction:
        cap = get_capability(cls)
        assert cap is not None, cls
        assert cap.kernel_kind is KernelKind.REQUANTIZING, cls
        assert cap.is_reduction is True, cls
        assert is_reduction_requantizing(cls), cls

    for cls in expected_non_reduction:
        cap = get_capability(cls)
        assert cap is not None, cls
        assert cap.kernel_kind is KernelKind.REQUANTIZING, cls
        assert cap.is_reduction is False, cls
        assert not is_reduction_requantizing(cls), cls

    # Non-REQUANTIZING and unknown classes never report as reduction.
    for cls in (nn.Sigmoid, nn.MaxPool2d, custom.Pad, nn.Identity):
        assert not is_reduction_requantizing(cls), cls

    class _NotInManifest(nn.Module):
        pass

    assert not is_reduction_requantizing(_NotInManifest)


def test_combo_budget_constant_matches_w51_root_cause():
    """W5.1 probe pins the budget at ``input_bw + weight_bw <= 24``.

    Hard-coded here so accidentally relaxing the constant in
    ``capabilities.py`` (e.g. to 32 by analogy with ALU width) trips a
    test instead of silently re-introducing the >9 dB SQNR cliff at
    ``16+16``.
    """

    assert REQUANTIZING_COMBO_BITWIDTH_BUDGET == 24


def test_assert_requantizing_combo_accepts_validated_combos():
    """The four combos validated by W5.1 / W5.1-extended (8+8, 16+8, 8+16,
    plus all 8-bit) must pass for any reduction kernel."""

    for ib, wb in [(8, 8), (16, 8), (8, 16)]:
        assert_requantizing_combo_supported(
            (ib,), (wb,),
            where="input_quantizers[0]",
            qualname="QuantizedLinear",
            is_reduction=True,
        )


def test_assert_requantizing_combo_rejects_full_16bit_reduction():
    """16+16 combo on a reduction kernel must surface the W5 root-cause
    error message so callers can find the budget rule."""

    with pytest.raises(ValueError, match="REQUANTIZING-with-MAC-reduction"):
        assert_requantizing_combo_supported(
            (16,), (16,),
            where="input_quantizers[0]",
            qualname="QuantizedLinear",
            is_reduction=True,
        )


def test_assert_requantizing_combo_pairs_matmul_inputs():
    """MatMul has no parameter weight; the combo gate pairs activation
    inputs against each other (``inputs[i]+inputs[j] <= budget``)."""

    assert_requantizing_combo_supported(
        (16, 8),
        (),
        where="input_quantizers",
        qualname="QuantizedMatMul",
        is_reduction=True,
    )

    with pytest.raises(ValueError, match="REQUANTIZING-with-MAC-reduction"):
        assert_requantizing_combo_supported(
            (16, 16),
            (),
            where="input_quantizers",
            qualname="QuantizedMatMul",
            is_reduction=True,
        )


def test_assert_requantizing_combo_allows_full_16bit_when_no_reduction():
    """Element-wise REQUANTIZING (Multiply/Divide/Add/Subtract) and
    sum-only reduction (AvgPool/Mean/LayerNorm) bypass the budget — both
    sides 16-bit is OK because there is no operand×operand MAC.
    """

    assert_requantizing_combo_supported(
        (16, 16),
        (),
        where="input_quantizers",
        qualname="QuantizedMultiply",
        is_reduction=False,
    )
    assert_requantizing_combo_supported(
        (16,),
        (),
        where="input_quantizers[0]",
        qualname="QuantizedAvgPool2d",
        is_reduction=False,
    )


def test_assert_requantizing_combo_rejects_unsupported_bitwidth():
    """Bitwidths outside ``SUPPORTED_ACTIVATION_BITWIDTHS`` (e.g. 4-bit
    activation or 32-bit) must still be rejected, regardless of the
    reduction flag — that gate is independent of the combo budget."""

    with pytest.raises(ValueError, match="not.*validated for REQUANTIZING"):
        assert_requantizing_combo_supported(
            (4,), (8,),
            where="input_quantizers[0]",
            qualname="QuantizedLinear",
            is_reduction=True,
        )

    with pytest.raises(ValueError, match="not.*validated for REQUANTIZING"):
        assert_requantizing_combo_supported(
            (32,), (),
            where="input_quantizers[0]",
            qualname="QuantizedAvgPool2d",
            is_reduction=False,
        )


def test_assert_requantizing_combo_short_circuits_when_no_operands():
    """Empty bitwidth tuples / all-``None`` tuples must early-return —
    quantizers that haven't been initialized contribute no contract."""

    assert_requantizing_combo_supported(
        (), (),
        where="input_quantizers",
        qualname="QuantizedLinear",
        is_reduction=True,
    )
    assert_requantizing_combo_supported(
        (None, None),  # type: ignore[arg-type]
        (None,),  # type: ignore[arg-type]
        where="input_quantizers",
        qualname="QuantizedLinear",
        is_reduction=True,
    )

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
"""``require_int32_saturated_accumulator`` (enforce-bitwidth-ceiling, S1)."""

import pytest

torch = pytest.importorskip("torch")

from aimet_torch.fixed_point.kernels._contracts import (  # noqa: E402
    require_int32_saturated_accumulator,
    require_int64_within_int32_range,
    require_int64_within_signed_bit_width,
)
from aimet_torch.fixed_point.requantize import (  # noqa: E402
    INT32_QMAX,
    INT32_QMIN,
    saturate_int32,
    saturate_mac_accumulator,
)


def test_require_int32_saturated_accumulator_accepts_int32_dtype():
    """Normal post-saturation accumulator (int32, in-range) must pass."""

    acc = torch.tensor([0, 1, -1, INT32_QMAX, INT32_QMIN], dtype=torch.int32)
    require_int32_saturated_accumulator(acc, op_name="Test")


def test_require_int32_saturated_accumulator_rejects_int64_dtype():
    """``int64`` acc means the kernel forgot to call
    ``saturate_mac_accumulator``; surface that as a TypeError so the
    forgotten call is fixed at the kernel rather than leaking into
    requantize as a wrap-around bug downstream.
    """

    acc = torch.tensor([0, 1, 2], dtype=torch.int64)
    with pytest.raises(TypeError, match="must be torch.int32"):
        require_int32_saturated_accumulator(acc, op_name="Test")


def test_require_int32_saturated_accumulator_rejects_float_dtype():
    acc = torch.tensor([0.0, 1.0], dtype=torch.float32)
    with pytest.raises(TypeError, match="must be torch.int32"):
        require_int32_saturated_accumulator(acc, op_name="Test")


def test_require_int32_saturated_accumulator_rejects_non_tensor():
    with pytest.raises(TypeError, match="must be a torch.Tensor"):
        require_int32_saturated_accumulator([0, 1, 2], op_name="Test")  # type: ignore[arg-type]


def test_require_int32_saturated_accumulator_skips_empty_tensor():
    """Reduce-along-empty-axis kernels can produce empty acc tensors;
    ``aminmax`` would error on those, so the contract must skip cleanly.
    """

    acc = torch.empty(0, dtype=torch.int32)
    require_int32_saturated_accumulator(acc, op_name="Test")


def test_require_int32_saturated_accumulator_traps_forgotten_saturate_call():
    """The whole point of this contract: a kernel that forgets the
    ``saturate_mac_accumulator`` step and hands ``requantize_int`` an
    int64 sum directly must trip *before* the data leaks into the
    requantize path. The int64 dtype is the cheapest, deterministic
    fingerprint of "saturate was skipped" — a future ``acc.to(int32)``
    would silently wrap and become indistinguishable from a saturated
    value, so we deliberately reject the int64 form rather than try to
    detect wrap after the fact.
    """

    raw_int64_sum = torch.tensor(
        [INT32_QMAX + 1, INT32_QMIN - 1, 0],
        dtype=torch.int64,
    )
    with pytest.raises(TypeError, match="must be torch.int32"):
        require_int32_saturated_accumulator(raw_int64_sum, op_name="Test")


def test_saturate_mac_accumulator_clamps_int64_overflow_to_int32_max(monkeypatch):
    """Sanity for the helper the contract guards: an int64 sum exceeding
    ``INT32_QMAX`` must come back as int32 ``INT32_QMAX``, not wrap.
    Belt-and-braces vs. ``test_hw_ref_eltwise_pool.py`` which exercises
    the same helper at a different magnitude and via the env-toggle path.
    """

    monkeypatch.setenv("AIMET_RX_ACC_INT32_SAT", "1")
    overflow = torch.tensor(
        [INT32_QMAX + 1234, INT32_QMIN - 5678, 0],
        dtype=torch.int64,
    )
    out = saturate_mac_accumulator(overflow)
    assert out.dtype == torch.int32
    assert out.tolist() == [INT32_QMAX, INT32_QMIN, 0]


def test_saturate_mac_accumulator_output_passes_contract(monkeypatch):
    """End-to-end: an int64 sum that would have wrapped, run through
    ``saturate_mac_accumulator``, must satisfy
    ``require_int32_saturated_accumulator``. This is the round-trip the
    kernel hot path actually does (``saturate_mac_accumulator(acc)``
    followed by the contract assertion before ``requantize_int``).
    """

    monkeypatch.setenv("AIMET_RX_ACC_INT32_SAT", "1")
    overflow = torch.tensor([INT32_QMAX + 1, 0, INT32_QMIN - 1], dtype=torch.int64)
    acc_sat = saturate_mac_accumulator(overflow)
    require_int32_saturated_accumulator(acc_sat, op_name="Test")


# --- ``require_int64_within_int32_range`` (S3, PWL/CLZ-style int64 tap point) ----


def test_require_int64_within_int32_range_accepts_clamped_int64():
    """Normal post-``saturate_int32`` PWL intermediate: int64 dtype, values
    inside ALU width — the canonical state at lut.py's PWL tap points."""

    acc = torch.tensor([0, INT32_QMAX, INT32_QMIN, 1234], dtype=torch.int64)
    require_int64_within_int32_range(acc, op_name="PWL.test")


def test_require_int64_within_int32_range_rejects_int32_dtype():
    """The PWL tap deliberately keeps int64 headroom; an int32 input
    means the kernel either (a) used the wrong contract — should call
    ``require_int32_saturated_accumulator`` for the requantize boundary,
    or (b) accidentally narrowed before it should have. Either way,
    surface it.
    """

    acc = torch.tensor([0, 1, 2], dtype=torch.int32)
    with pytest.raises(TypeError, match="must be torch.int64"):
        require_int64_within_int32_range(acc, op_name="PWL.test")


def test_require_int64_within_int32_range_skips_value_check_outside_hw_ref(monkeypatch):
    """Performance: outside ``hw_ref_mode_enabled`` only the dtype gate
    runs; the (deliberately heavier) ``aminmax`` value-domain check is
    reserved for the strict regression mode where we accept extra cost
    in exchange for catching a forgotten ``saturate_int32`` call.
    """

    monkeypatch.delenv("AIMET_RX_HW_REF", raising=False)
    monkeypatch.delenv("AIMET_RX_PWL_HW_REF", raising=False)
    monkeypatch.delenv("AIMET_RX_HW_REF_MODE", raising=False)
    overflow_uncheckable = torch.tensor(
        [INT32_QMAX + 7, INT32_QMIN - 9, 0],
        dtype=torch.int64,
    )
    require_int64_within_int32_range(overflow_uncheckable, op_name="PWL.test")


def test_require_int64_within_int32_range_traps_overflow_in_hw_ref(monkeypatch):
    """Strict-mode value gate: an int64 PWL intermediate that **wasn't**
    fed through ``saturate_int32`` — the very mistake this contract is
    supposed to catch — must raise. This is the key reason the int64
    sibling exists at all (the int32 version cannot make this distinction;
    see its docstring).
    """

    monkeypatch.setenv("AIMET_RX_HW_REF", "1")
    overflow = torch.tensor(
        [INT32_QMAX + 7, INT32_QMIN - 9, 0],
        dtype=torch.int64,
    )
    with pytest.raises(ValueError, match="exceeds INT32 ALU width"):
        require_int64_within_int32_range(overflow, op_name="PWL.test")


def test_require_int64_within_int32_range_passes_after_saturate_int32(monkeypatch):
    """The hot-path round trip in ``lut.py``:
    ``acc = saturate_int32(...).to(int64)`` followed by the contract.
    Same overflow input as the negative case above, just clamped first;
    must now pass even under hw_ref.
    """

    monkeypatch.setenv("AIMET_RX_HW_REF", "1")
    raw = torch.tensor(
        [INT32_QMAX + 7, INT32_QMIN - 9, 0],
        dtype=torch.int64,
    )
    clamped = saturate_int32(raw).to(torch.int64)
    require_int64_within_int32_range(clamped, op_name="PWL.test")


def test_require_int64_within_int32_range_skips_empty_tensor(monkeypatch):
    monkeypatch.setenv("AIMET_RX_HW_REF", "1")
    require_int64_within_int32_range(
        torch.empty(0, dtype=torch.int64), op_name="PWL.test"
    )


# --- ``require_int64_within_signed_bit_width`` (S3 follow-up, CLZ acc_bw) ---


def test_require_int64_within_signed_bit_width_accepts_in_range_default32():
    """Default ``acc_bw=32``: behaves like the int32 sibling but
    expressed via the explicit-width helper (canonical CLZ usage)."""

    acc = torch.tensor([0, INT32_QMAX, INT32_QMIN, 1234], dtype=torch.int64)
    require_int64_within_signed_bit_width(acc, bit_width=32, op_name="CLZ.test")


def test_require_int64_within_signed_bit_width_accepts_smaller_width(monkeypatch):
    """LUT manifests can shrink ``acc_bw`` below 32 to match a narrower
    PE BxC accumulator (e.g. 24-bit). Values inside that narrower window
    must pass even under hw_ref."""

    monkeypatch.setenv("AIMET_RX_HW_REF", "1")
    lo24 = -(1 << 23)
    hi24 = (1 << 23) - 1
    acc = torch.tensor([0, hi24, lo24, 1], dtype=torch.int64)
    require_int64_within_signed_bit_width(acc, bit_width=24, op_name="CLZ.test")


def test_require_int64_within_signed_bit_width_traps_overflow_against_smaller_width(
    monkeypatch,
):
    """The actual catch-this-bug path: an int64 raw value that's *fine*
    in 32-bit but *exceeds* the configured ``acc_bw=24`` must trip in
    hw_ref mode. This is the bug class the third gate exists for —
    surfacing "acc_bw set too small / forgotten ``_sat_signed_vec``"
    that the int32 sibling cannot model.
    """

    monkeypatch.setenv("AIMET_RX_HW_REF", "1")
    hi24 = (1 << 23) - 1
    overflow = torch.tensor([hi24 + 5, 0, -hi24 - 7], dtype=torch.int64)
    with pytest.raises(ValueError, match=r"signed 24-bit ALU range"):
        require_int64_within_signed_bit_width(
            overflow, bit_width=24, op_name="CLZ.test"
        )


def test_require_int64_within_signed_bit_width_rejects_int32_dtype():
    acc = torch.tensor([0, 1], dtype=torch.int32)
    with pytest.raises(TypeError, match="must be torch.int64"):
        require_int64_within_signed_bit_width(acc, bit_width=32, op_name="CLZ.test")


@pytest.mark.parametrize("bad_bw", [0, -1, 65, 100])
def test_require_int64_within_signed_bit_width_rejects_invalid_bit_width(bad_bw):
    """``bit_width`` is the contract knob — silently accepting nonsense
    (0, negative, > 64) would defeat the whole point of asserting the
    LUT manifest plumbed a sane width through."""

    acc = torch.tensor([0], dtype=torch.int64)
    with pytest.raises(ValueError, match=r"bit_width must be in \[1, 64\]"):
        require_int64_within_signed_bit_width(
            acc, bit_width=bad_bw, op_name="CLZ.test"
        )


def test_require_int64_within_signed_bit_width_skips_value_check_outside_hw_ref(
    monkeypatch,
):
    monkeypatch.delenv("AIMET_RX_HW_REF", raising=False)
    monkeypatch.delenv("AIMET_RX_PWL_HW_REF", raising=False)
    monkeypatch.delenv("AIMET_RX_HW_REF_MODE", raising=False)
    overflow = torch.tensor([(1 << 40), -(1 << 40), 0], dtype=torch.int64)
    require_int64_within_signed_bit_width(overflow, bit_width=32, op_name="CLZ.test")


def test_require_int64_within_signed_bit_width_skips_empty_tensor():
    require_int64_within_signed_bit_width(
        torch.empty(0, dtype=torch.int64), bit_width=32, op_name="CLZ.test"
    )

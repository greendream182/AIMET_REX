import pytest
import torch

from aimet_torch.fixed_point import (
    RoundingMode,
    requantize_int,
    round_shift,
    saturate_int16,
)
from aimet_torch.fixed_point.requantize import (
    INT32_QMAX,
    INT32_QMIN,
    SIM_TENSOR_DTYPE,
    saturate_int32,
    saturate_sim_tensor,
)


def test_requantize_int_basic_half_to_even():
    acc = torch.tensor([100, -100, 32766], dtype=torch.int32)
    multiplier = torch.tensor(16384, dtype=torch.uint16)
    rshift = torch.tensor(15, dtype=torch.int8)
    y_zp = torch.tensor(0, dtype=torch.int32)

    output = requantize_int(acc, multiplier, rshift, y_zp)

    assert output.dtype is SIM_TENSOR_DTYPE
    assert output.tolist() == [50, -50, 16383]


def test_requantize_int_saturates():
    acc = torch.tensor([2_000_000_000, -2_000_000_000], dtype=torch.int32)
    multiplier = torch.tensor(32767, dtype=torch.uint16)
    rshift = torch.tensor(15, dtype=torch.int8)
    y_zp = torch.tensor(0, dtype=torch.int32)

    output = requantize_int(acc, multiplier, rshift, y_zp)

    assert output.tolist() == [32767, -32768]


def test_round_shift_half_away_from_zero_signed():
    x = torch.tensor([3, -3, 5, -5], dtype=torch.int64)
    rshift = torch.tensor(1, dtype=torch.int8)

    output = round_shift(x, rshift, RoundingMode.HALF_AWAY_FROM_ZERO)

    assert output.tolist() == [2, -2, 3, -3]


def test_saturate_int16_returns_int16():
    output = saturate_int16(torch.tensor([40000, -40000, 100], dtype=torch.int32))

    assert output.dtype == torch.int16
    assert output.tolist() == [32767, -32768, 100]


# ---- PR-1 additions: SIM_TENSOR_DTYPE + saturate_sim_tensor --------------
# These cover the new int32 sim-tensor container constant (ADR-013) and the
# unified saturate+dtype-cast helper. saturate_int16 / requantize_int output
# dtypes intentionally remain int16 in PR-1; PR-3 will migrate them.


def test_sim_tensor_dtype_is_int32():
    # ADR-013 hardcodes the sim-tensor container to torch.int32 as a single
    # module-level constant. Guard against accidental flips back to int16.
    assert SIM_TENSOR_DTYPE is torch.int32


def test_saturate_sim_tensor_default_range_is_int16_grid():
    output = saturate_sim_tensor(torch.tensor([40000, -40000, 100], dtype=torch.int32))

    assert output.dtype is SIM_TENSOR_DTYPE
    assert output.tolist() == [32767, -32768, 100]


def test_saturate_sim_tensor_custom_qmin_qmax():
    # U8 grid: should clamp to [0, 255] yet keep the int32 container.
    raw = torch.tensor([-5, 0, 100, 300], dtype=torch.int32)

    output = saturate_sim_tensor(raw, qmin=0, qmax=255)

    assert output.dtype is SIM_TENSOR_DTYPE
    assert output.tolist() == [0, 0, 100, 255]


def test_saturate_sim_tensor_accepts_int64_input():
    # Multiplier paths produce int64 accumulators; the helper must downcast
    # cleanly without overflow inside the [qmin, qmax] grid.
    raw = torch.tensor([1 << 40, -(1 << 40), 50], dtype=torch.int64)

    output = saturate_sim_tensor(raw)

    assert output.dtype is SIM_TENSOR_DTYPE
    assert output.tolist() == [32767, -32768, 50]


def test_saturate_sim_tensor_rejects_inverted_range():
    with pytest.raises(ValueError):
        saturate_sim_tensor(torch.tensor([0], dtype=torch.int32), qmin=10, qmax=0)


def test_saturate_int32_clamps_accumulator_width():
    raw = torch.tensor([INT32_QMAX + 1, INT32_QMIN - 1, 0], dtype=torch.int64)

    output = saturate_int32(raw)

    assert output.dtype == torch.int32
    assert output.tolist() == [INT32_QMAX, INT32_QMIN, 0]


def test_requantize_int32_product_saturates_before_shift_when_env_enabled(monkeypatch):
    """``AIMET_RX_REQUANTIZE_INT32_SAT=1``: prod clamps to INT32 before rshift."""
    monkeypatch.setenv("AIMET_RX_REQUANTIZE_INT32_SAT", "1")
    acc = torch.tensor([66_000], dtype=torch.int32)
    multiplier = torch.tensor(32767, dtype=torch.uint16)
    rshift = torch.tensor(17, dtype=torch.int8)
    y_zp = torch.tensor(0, dtype=torch.int32)

    output = requantize_int(acc, multiplier, rshift, y_zp, qmin=-32768, qmax=32767)

    assert output.item() == 16_384


def test_requantize_int32_product_no_sat_by_default(monkeypatch):
    """Default path keeps int64 product through rshift (differs from INT32-sat path)."""
    monkeypatch.delenv("AIMET_RX_REQUANTIZE_INT32_SAT", raising=False)
    monkeypatch.delenv("AIMET_RX_HW_REF", raising=False)
    monkeypatch.delenv("AIMET_RX_PWL_HW_REF", raising=False)
    acc = torch.tensor([66_000], dtype=torch.int32)
    multiplier = torch.tensor(32767, dtype=torch.uint16)
    rshift = torch.tensor(17, dtype=torch.int8)
    y_zp = torch.tensor(0, dtype=torch.int32)

    output = requantize_int(acc, multiplier, rshift, y_zp, qmin=-32768, qmax=32767)

    assert output.item() == 16_499


def test_round_shift_half_up_matches_bias_formula():
    x = torch.tensor([7, -7, 4], dtype=torch.int64)
    rshift = torch.tensor(2, dtype=torch.int8)

    output = round_shift(x, rshift, RoundingMode.HALF_UP)

    assert output.tolist() == [2, -2, 1]

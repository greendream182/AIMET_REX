import math

import pytest
import torch

from aimet_torch.fixed_point import (
    quantize_bias_int,
    quantize_bias_int32,
    quantize_multiplier,
    record_multiplier_saturations,
)
from aimet_torch.fixed_point.offline.scale_fixed import (
    clear_fixed_scale_encoding_cache,
    fixed_scale_float_scale,
    quantize_scale_to_m_rshift,
)
from aimet_torch.fixed_point.encoding import FixedScaleEncoding


def test_quantize_multiplier_known_value():
    multiplier, rshift = quantize_multiplier(0.1234)

    assert multiplier.dtype == torch.uint16
    assert rshift.dtype == torch.int8
    assert multiplier.item() == 64697
    assert rshift.item() == 19


def test_quantize_multiplier_tensor():
    multiplier, rshift = quantize_multiplier(torch.tensor([0.5, 0.25]))

    assert multiplier.dtype == torch.uint16
    assert rshift.dtype == torch.int8
    assert multiplier.tolist() == [32768, 32768]
    assert rshift.tolist() == [16, 17]


def test_quantize_multiplier_folds_when_rshift_exceeds_max():
    """Spec 10 §53: tiny ``real_m`` folds to (M, max_rshift) instead of raising."""

    real_m = 6.751979864105806e-09  # observed on MobileNet-V2 features.0.0 channel
    multiplier, rshift = quantize_multiplier(real_m)

    assert 0 <= int(rshift.item()) <= 31
    assert int(rshift.item()) == 31
    assert 1 <= int(multiplier.item()) <= 65535
    approx = float(multiplier.item()) / (1 << int(rshift.item()))
    rel_err = abs(approx - real_m) / real_m
    # Folding to (M, 31) keeps relative error within a few percent for this magnitude.
    assert rel_err < 0.1


def test_quantize_multiplier_fold_mode_zero_collapses_overflow(monkeypatch):
    """``AIMET_RX_INT16_FOLD_MODE=zero`` collapses out-of-range pairs to ``(0, max_rshift)``.

    This is a diagnostic A/B knob (multiplier.py module docstring): does the
    dead-channel fold residue explain the INT16 vs FIXED_SCALE_QDQ cosine gap?
    """

    monkeypatch.setenv("AIMET_RX_INT16_FOLD_MODE", "zero")
    multiplier, rshift = quantize_multiplier(6.751979864105806e-09)
    assert int(multiplier.item()) == 0
    assert int(rshift.item()) == 31


def test_quantize_multiplier_strict_mode_still_raises_on_overflow():
    """``saturate=False`` keeps the legacy strict behavior for debug / CI gates."""

    with pytest.raises(ValueError, match=r"rshift.*outside \[0, 31\]"):
        quantize_multiplier(1e-12, saturate=False)


def test_record_multiplier_saturations_captures_folded_event():
    """Context manager records folded events with relative error (spec 10 §53)."""

    real_m = torch.tensor([0.1234, 6.75e-9, 0.5])
    with record_multiplier_saturations() as events:
        multiplier, rshift = quantize_multiplier(real_m)

    assert int(rshift[1].item()) == 31  # tiny entry folded to max_rshift
    assert len(events) == 1
    entry = events[0]
    assert entry["rshift"] == 31
    # float32 tensor → float64 promotion has minor jitter; tolerate ULPs.
    assert math.isclose(entry["real_multiplier"], 6.75e-9, rel_tol=1e-6)
    assert entry["relative_error"] > 0.0


def test_quantize_bias_int32():
    bias = torch.tensor([0.25, -0.5], dtype=torch.float32)
    x_scale = torch.tensor(0.5, dtype=torch.float32)
    w_scale = torch.tensor(0.25, dtype=torch.float32)

    bias_int32 = quantize_bias_int32(bias, x_scale, w_scale, saturate=False)

    assert bias_int32.dtype == torch.int32
    assert bias_int32.tolist() == [2, -4]


def test_quantize_bias_int_bits16_dtype_and_value():
    """``bits=16`` outputs torch.int16 with the same scale ratio as bits=32."""

    bias = torch.tensor([0.25, -0.5], dtype=torch.float32)
    x_scale = torch.tensor(0.5, dtype=torch.float32)
    w_scale = torch.tensor(0.25, dtype=torch.float32)

    bias_int16 = quantize_bias_int(
        bias, x_scale, w_scale, bits=16, saturate=False
    )
    bias_int32 = quantize_bias_int(
        bias, x_scale, w_scale, bits=32, saturate=False
    )

    assert bias_int16.dtype == torch.int16
    assert bias_int32.dtype == torch.int32
    assert bias_int16.tolist() == bias_int32.tolist() == [2, -4]


def test_quantize_bias_int_bits16_default_overflow_raises():
    """``bits=16`` defaults to ``saturate=False``: silent clamping would
    otherwise mutate model semantics under explicit_config, so an out-of-range
    int16 bias must surface as a hard error rather than be silently clipped."""

    bias = torch.tensor([1e6], dtype=torch.float32)
    x_scale = torch.tensor(1.0, dtype=torch.float32)
    w_scale = torch.tensor(1.0, dtype=torch.float32)

    with pytest.raises(ValueError, match=r"exceeds int16 range"):
        quantize_bias_int(bias, x_scale, w_scale, bits=16)


def test_quantize_bias_int_bits16_explicit_saturate_clamps_to_int16_range():
    """``bits=16`` with explicit ``saturate=True`` clamps to int16 range
    (callers that accept clipping must opt in)."""

    bias = torch.tensor([1e6], dtype=torch.float32)
    x_scale = torch.tensor(1.0, dtype=torch.float32)
    w_scale = torch.tensor(1.0, dtype=torch.float32)

    bias_int16 = quantize_bias_int(bias, x_scale, w_scale, bits=16, saturate=True)
    assert bias_int16.dtype == torch.int16
    assert int(bias_int16.item()) == 32767


def test_quantize_bias_int_bits32_default_saturates():
    """``bits=32`` keeps the legacy ``saturate=True`` default for QAT stability."""

    bias = torch.tensor([1e6], dtype=torch.float32)
    x_scale = torch.tensor(1e-6, dtype=torch.float32)
    w_scale = torch.tensor(1.0, dtype=torch.float32)

    bias_int32 = quantize_bias_int(bias, x_scale, w_scale, bits=32)
    assert int(bias_int32.item()) == 2147483647


def test_quantize_bias_int_per_channel_acc_scale_aligned_linear():
    """Linear-style per-channel ``w_scale`` shape ``(out, 1)`` must divide
    a 1-D ``(out,)`` bias element-wise (no extra broadcast dim)."""

    bias = torch.tensor([0.25, -0.5], dtype=torch.float32)
    x_scale = torch.tensor(0.5, dtype=torch.float32)
    w_scale = torch.full((2, 1), 0.25, dtype=torch.float32)

    out = quantize_bias_int(bias, x_scale, w_scale, bits=32, saturate=False)
    assert out.shape == bias.shape
    assert out.tolist() == [2, -4]


def test_quantize_bias_int_per_channel_acc_scale_aligned_conv():
    """Conv-style per-channel ``w_scale`` shape ``(out, 1, 1, 1)`` must
    divide a 1-D ``(out,)`` bias element-wise."""

    bias = torch.tensor([0.25, -0.5], dtype=torch.float32)
    x_scale = torch.tensor(0.5, dtype=torch.float32)
    w_scale = torch.full((2, 1, 1, 1), 0.25, dtype=torch.float32)

    out = quantize_bias_int(bias, x_scale, w_scale, bits=32, saturate=False)
    assert out.shape == bias.shape
    assert out.tolist() == [2, -4]


def test_quantize_bias_int_acc_scale_shape_mismatch_raises():
    """Any acc_scale with neither scalar nor bias.numel() elements is rejected,
    so silent broadcasting bugs surface early."""

    bias = torch.tensor([0.25, -0.5], dtype=torch.float32)
    x_scale = torch.tensor(0.5, dtype=torch.float32)
    w_scale = torch.full((3, 1), 0.25, dtype=torch.float32)

    with pytest.raises(ValueError, match=r"not broadcastable per-channel"):
        quantize_bias_int(bias, x_scale, w_scale, bits=32, saturate=False)


def test_quantize_bias_int_invalid_bits_raises():
    bias = torch.tensor([0.25], dtype=torch.float32)
    with pytest.raises(ValueError, match=r"bits must be 16 or 32"):
        quantize_bias_int(
            bias, torch.tensor(1.0), torch.tensor(1.0), bits=24
        )


def test_quantize_bias_int_non_finite_bias_raises():
    """NaN / Inf in ``bias_float`` is rejected up-front; otherwise downstream
    rounding / saturation produces platform-dependent garbage."""

    x_scale = torch.tensor(0.5, dtype=torch.float32)
    w_scale = torch.tensor(0.25, dtype=torch.float32)
    for bad in (
        torch.tensor([0.25, float("nan")], dtype=torch.float32),
        torch.tensor([float("inf"), -0.5], dtype=torch.float32),
    ):
        with pytest.raises(ValueError, match=r"bias_float must be finite"):
            quantize_bias_int(bad, x_scale, w_scale, bits=32, saturate=False)


def test_quantize_bias_int_non_finite_scale_raises():
    """NaN / Inf in either scale is rejected; ``acc_scale = S_x * S_w`` would
    otherwise propagate the non-finite value into the quantized bias."""

    bias = torch.tensor([0.25], dtype=torch.float32)
    good = torch.tensor(0.5, dtype=torch.float32)
    bad_nan = torch.tensor(float("nan"), dtype=torch.float32)
    bad_inf = torch.tensor(float("inf"), dtype=torch.float32)
    with pytest.raises(ValueError, match=r"must be finite"):
        quantize_bias_int(bias, bad_nan, good, bits=32, saturate=False)
    with pytest.raises(ValueError, match=r"must be finite"):
        quantize_bias_int(bias, good, bad_inf, bits=32, saturate=False)


def test_quantize_bias_int_non_positive_scale_raises():
    """Scales must be strictly positive: zero or negative scales have no
    valid quantization grid and previously slipped past the ``!= 0`` check
    when negative."""

    bias = torch.tensor([0.25], dtype=torch.float32)
    good = torch.tensor(0.5, dtype=torch.float32)
    for bad in (
        torch.tensor(0.0, dtype=torch.float32),
        torch.tensor(-0.25, dtype=torch.float32),
    ):
        with pytest.raises(ValueError, match=r"must be strictly positive"):
            quantize_bias_int(bias, bad, good, bits=32, saturate=False)
        with pytest.raises(ValueError, match=r"must be strictly positive"):
            quantize_bias_int(bias, good, bad, bits=32, saturate=False)


def test_quantize_bias_int32_saturates_by_default():
    bias = torch.tensor([1e6], dtype=torch.float32)
    x_scale = torch.tensor(1e-6, dtype=torch.float32)
    w_scale = torch.tensor(1.0, dtype=torch.float32)

    bias_int32 = quantize_bias_int32(bias, x_scale, w_scale)
    assert int(bias_int32.item()) == 2147483647


def test_quantize_scale_to_m_rshift_batched_tensor():
    scales = torch.tensor([0.125, 0.25], dtype=torch.float32)
    m, r = quantize_scale_to_m_rshift(scales)
    assert m.dtype == torch.uint16
    assert r.dtype == torch.int8
    assert m.numel() == 2


def test_fixed_scale_float_scale_recovery():
    device = torch.device("cpu")
    m = torch.tensor(32768, dtype=torch.uint16)
    r = torch.tensor(16, dtype=torch.int8)
    scale = fixed_scale_float_scale(m, r, device=device)
    assert abs(scale.item() - 0.5) < 1e-4


def test_clear_fixed_scale_encoding_cache_removes_attr():
    enc = FixedScaleEncoding(
        m_int16=torch.tensor(32767, dtype=torch.int16),
        rshift=torch.tensor(15, dtype=torch.int8),
        zero_point=torch.tensor(0, dtype=torch.int32),
        qmin=-128,
        qmax=127,
        axis=None,
        scale_fp_legacy=torch.tensor(1.0, dtype=torch.float32),
    )
    class _Holder:
        pass

    holder = _Holder()
    setattr(holder, "_aimet_rx_fixed_scale_encoding", enc)
    clear_fixed_scale_encoding_cache(holder)
    assert not hasattr(holder, "_aimet_rx_fixed_scale_encoding")

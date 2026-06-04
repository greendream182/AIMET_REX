import math

import pytest
import torch

from aimet_torch.fixed_point import (
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

    assert multiplier.dtype == torch.int16
    assert rshift.dtype == torch.int8
    assert multiplier.item() == 32349
    assert rshift.item() == 18


def test_quantize_multiplier_tensor():
    multiplier, rshift = quantize_multiplier(torch.tensor([0.5, 0.25]))

    assert multiplier.dtype == torch.int16
    assert rshift.dtype == torch.int8
    assert multiplier.tolist() == [16384, 16384]
    assert rshift.tolist() == [15, 16]


def test_quantize_multiplier_folds_when_rshift_exceeds_max():
    """Spec 10 §53: tiny ``real_m`` folds to (M, max_rshift) instead of raising."""

    real_m = 6.751979864105806e-09  # observed on MobileNet-V2 features.0.0 channel
    multiplier, rshift = quantize_multiplier(real_m)

    assert 0 <= int(rshift.item()) <= 31
    assert int(rshift.item()) == 31
    assert 1 <= int(multiplier.item()) <= 32767
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


def test_quantize_bias_int32_saturates_by_default():
    bias = torch.tensor([1e6], dtype=torch.float32)
    x_scale = torch.tensor(1e-6, dtype=torch.float32)
    w_scale = torch.tensor(1.0, dtype=torch.float32)

    bias_int32 = quantize_bias_int32(bias, x_scale, w_scale)
    assert int(bias_int32.item()) == 2147483647


def test_quantize_scale_to_m_rshift_batched_tensor():
    scales = torch.tensor([0.125, 0.25], dtype=torch.float32)
    m, r = quantize_scale_to_m_rshift(scales)
    assert m.dtype == torch.int16
    assert r.dtype == torch.int8
    assert m.numel() == 2


def test_fixed_scale_float_scale_recovery():
    device = torch.device("cpu")
    m = torch.tensor(16384, dtype=torch.int16)
    r = torch.tensor(15, dtype=torch.int8)
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

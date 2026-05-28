"""Offline CLZ LUT generation via lut_int_general."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from aimet_torch.fixed_point.encoding import InputEncoding, OutputEncoding
from aimet_torch.fixed_point.kernels.clz_lut import evaluate_clz_normalized_lut_int16
from aimet_torch.fixed_point.offline.clz_gen import (
    ClzLutGenerationError,
    generate_clz_lut_for_export,
    resolve_abc_lut_root,
    resolve_clz_fit_float_range,
)

_ABC = resolve_abc_lut_root()
_HAS_ABC = _ABC is not None
_GOLDEN = (
    _ABC / "lut_int_general" / "output" / "lut_test" / "sqrt_clz_lut.json"
    if _ABC
    else None
)


@pytest.mark.skipif(not _HAS_ABC, reason="abc_lut-shuai not in workspace")
def test_generate_clz_sqrt_produces_segments():
    enc = OutputEncoding(
        scale=torch.tensor(0.00024414807580797754, dtype=torch.float32),
        zero_point=torch.tensor(0, dtype=torch.int32),
        qmin=-32768,
        qmax=32767,
    )
    body, metrics = generate_clz_lut_for_export("sqrt", enc, enc)
    assert "segments" in body
    assert len(body["segments"]) == 16
    assert metrics["num_segments"] == 16.0


@pytest.mark.skipif(not _HAS_ABC, reason="abc_lut-shuai not in workspace")
def test_generate_clz_sqrt_matches_golden_json_on_samples():
    import json

    assert _GOLDEN is not None and _GOLDEN.is_file()
    golden = json.loads(_GOLDEN.read_text(encoding="utf-8"))["sqrt"]
    enc = OutputEncoding(
        scale=torch.tensor(
            golden["quantization"]["input"]["scale"], dtype=torch.float32
        ),
        zero_point=torch.tensor(
            golden["quantization"]["input"]["zero_point"], dtype=torch.int32
        ),
        qmin=int(golden["quantization"]["input"]["min"]),
        qmax=int(golden["quantization"]["input"]["max"]),
    )
    body, _ = generate_clz_lut_for_export("sqrt", enc, enc)
    q = torch.tensor([1000, 4000, 20000], dtype=torch.int16)
    y_new = evaluate_clz_normalized_lut_int16(q, body, "sqrt")
    y_ref = evaluate_clz_normalized_lut_int16(q, golden, "sqrt")
    torch.testing.assert_close(
        y_new.to(torch.int32), y_ref.to(torch.int32), rtol=0, atol=0
    )


@pytest.mark.skipif(not _HAS_ABC, reason="abc_lut-shuai not in workspace")
def test_generate_clz_reciprocal_output_scale_sane_with_fit_domain():
    """Symmetric encodings must not drive reciprocal global output quant to ~1e6 scale."""

    class _Q:
        pass

    q = _Q()
    q.min = torch.nn.Parameter(torch.tensor(0.2))
    q.max = torch.nn.Parameter(torch.tensor(2.0))
    fake = type("FakeModule", (), {"input_quantizers": [q]})()

    enc = InputEncoding(
        scale=torch.tensor(2.7466237952467054e-05, dtype=torch.float32),
        zero_point=torch.tensor(0, dtype=torch.int32),
        qmin=-32768,
        qmax=32767,
    )
    fit_min, fit_max = resolve_clz_fit_float_range(fake, "reciprocal", enc)
    assert fit_min == pytest.approx(0.2)
    assert fit_max == pytest.approx(2.0)
    body, _ = generate_clz_lut_for_export(
        "reciprocal", enc, enc, fit_float_min=fit_min, fit_float_max=fit_max
    )
    out_scale = float(body["quantization"]["output"]["scale"])
    assert out_scale < 1.0
    assert abs(body["quantization"]["output"]["fmax"]) < 200.0


@pytest.mark.skipif(not _HAS_ABC, reason="abc_lut-shuai not in workspace")
def test_generate_clz_power2_produces_segments():
    enc = OutputEncoding(
        scale=torch.tensor(0.00024414807580797754, dtype=torch.float32),
        zero_point=torch.tensor(0, dtype=torch.int32),
        qmin=-32768,
        qmax=32767,
    )
    body, metrics = generate_clz_lut_for_export("power_2", enc, enc)
    assert "segments" in body
    assert len(body["segments"]) == 16
    assert metrics["num_segments"] == 16.0


def test_generate_clz_raises_without_abc(monkeypatch):
    monkeypatch.delenv("AIMET_RX_ABC_LUT_ROOT", raising=False)
    enc = OutputEncoding(
        scale=torch.tensor(0.01, dtype=torch.float32),
        zero_point=torch.tensor(0, dtype=torch.int32),
        qmin=-32768,
        qmax=32767,
    )

    def _no_root(_explicit=None):
        return None

    monkeypatch.setattr(
        "aimet_torch.fixed_point.offline.clz_gen.resolve_abc_lut_root",
        _no_root,
    )
    with pytest.raises(ClzLutGenerationError):
        generate_clz_lut_for_export("sqrt", enc, enc, abc_lut_root="/nonexistent")

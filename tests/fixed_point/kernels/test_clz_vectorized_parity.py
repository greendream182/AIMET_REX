"""Vectorized CLZ kernel must be bit-exact with the Python reference oracle.

The reference (`_evaluate_clz_reference`) is the per-element golden path; the
vectorized path (`_evaluate_clz_vectorized`) is the on-device fast path used by
default. They must agree exactly on every int16 input, on CPU and CUDA.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from aimet_torch.fixed_point.kernels import clz_lut

_ABC_ROOT = Path(__file__).resolve().parents[4] / "abc_lut-shuai"
_LUT_DIR = _ABC_ROOT / "lut_int_general" / "output" / "lut_test"
_FUNCS = ["sqrt", "rsqrt", "reciprocal", "power_2"]
_HAS_ABC = _LUT_DIR.is_dir() and all(
    (_LUT_DIR / f"{f}_clz_lut.json").is_file() for f in _FUNCS
)

_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _full_int16_inputs() -> torch.Tensor:
    return torch.arange(-32768, 32768, dtype=torch.int32)


@pytest.mark.skipif(not _HAS_ABC, reason="abc CLZ LUT JSONs not in workspace")
@pytest.mark.parametrize("func", _FUNCS)
@pytest.mark.parametrize("device", _DEVICES)
def test_vectorized_matches_reference_full_range(func: str, device: str):
    _, clz_body = clz_lut.load_clz_lut_from_json(
        _LUT_DIR / f"{func}_clz_lut.json", func_name=func
    )
    q = _full_int16_inputs().to(device)

    ref = clz_lut._evaluate_clz_reference(q.cpu(), clz_body, func)
    vec = clz_lut._evaluate_clz_vectorized(q, clz_body, func).cpu()

    mismatches = (ref.to(torch.int64) != vec.to(torch.int64)).sum().item()
    assert mismatches == 0, (
        f"{func} on {device}: {mismatches} mismatching elements vs reference"
    )


@pytest.mark.skipif(not _HAS_ABC, reason="abc CLZ LUT JSONs not in workspace")
@pytest.mark.parametrize("func", _FUNCS)
def test_default_dispatch_is_vectorized_and_consistent(func: str, monkeypatch):
    _, clz_body = clz_lut.load_clz_lut_from_json(
        _LUT_DIR / f"{func}_clz_lut.json", func_name=func
    )
    q = torch.tensor([-32768, -100, 0, 100, 1000, 20000, 32767], dtype=torch.int32)

    monkeypatch.setenv("AIMET_RX_CLZ_VECTORIZED", "1")
    on = clz_lut.evaluate_clz_normalized_lut_int16(q, clz_body, func)
    monkeypatch.setenv("AIMET_RX_CLZ_VECTORIZED", "0")
    off = clz_lut.evaluate_clz_normalized_lut_int16(q, clz_body, func)

    torch.testing.assert_close(
        on.to(torch.int64), off.to(torch.int64), rtol=0, atol=0
    )

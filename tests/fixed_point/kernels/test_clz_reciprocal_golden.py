"""CLZ reciprocal LUT vs lut_int_general."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

from aimet_torch.fixed_point.kernels.clz_lut import (
    evaluate_clz_normalized_lut_int16,
    load_clz_lut_from_json,
)

_ABC_PKG_ROOT = Path(__file__).resolve().parents[4] / "abc_lut-shuai"
_JSON = _ABC_PKG_ROOT / "lut_int_general" / "output" / "lut_test" / "reciprocal_clz_lut.json"
_HAS_ABC = _JSON.is_file()
if _HAS_ABC:
    sys.path.insert(0, str(_ABC_PKG_ROOT))


@pytest.mark.skipif(not _HAS_ABC, reason="abc reciprocal_clz_lut.json not in workspace")
def test_evaluate_clz_reciprocal_matches_lut_int_general():
    from lut_int_general.quantization.lut import infer_with_clz_normalized_lut  # type: ignore
    func_name, clz_body = load_clz_lut_from_json(_JSON)
    assert func_name == "reciprocal"

    q_samples = torch.tensor([500, 2000, 8000, 20000], dtype=torch.int16)
    y_aimet = evaluate_clz_normalized_lut_int16(
        q_samples, clz_body, func_name
    ).to(torch.int32)
    y_abc = torch.tensor(
        infer_with_clz_normalized_lut(
            json.loads(_JSON.read_text(encoding="utf-8")),
            q_samples.numpy(),
            input_dtype="int16",
        )["output_quantized"],
        dtype=torch.int32,
    )
    assert (y_aimet - y_abc).abs().max().item() <= 3


@pytest.mark.skipif(not _HAS_ABC, reason="abc reciprocal_clz_lut.json not in workspace")
def test_reciprocal_negative_q_sign_extension_not_abc_saturation():
    """AIMET reflects sign for x<0; abc golden saturates negative q to out_qmax."""

    from lut_int_general.quantization.lut import infer_with_clz_normalized_lut  # type: ignore

    func_name, clz_body = load_clz_lut_from_json(_JSON)
    q_neg = torch.tensor([-2000, -500], dtype=torch.int16)
    y_aimet = evaluate_clz_normalized_lut_int16(q_neg, clz_body, func_name).to(torch.int32)
    assert (y_aimet < 0).any().item()

    payload = json.loads(_JSON.read_text(encoding="utf-8"))
    out_max = int(clz_body["quantization"]["output"]["max"])
    y_abc = torch.tensor(
        infer_with_clz_normalized_lut(
            payload, q_neg.numpy(), input_dtype="int16"
        )["output_quantized"],
        dtype=torch.int32,
    )
    assert (y_abc == out_max).all().item()

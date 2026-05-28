"""CLZ power_2 LUT vs lut_int_general (maps to ``custom.Square`` kernel)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

from aimet_torch._base.nn.modules import custom
from aimet_torch.fixed_point import get_fixed_kernel
from aimet_torch.fixed_point.encoding import OutputEncoding
from aimet_torch.fixed_point.kernels.clz_lut import (
    evaluate_clz_normalized_lut_int16,
    load_clz_lut_from_json,
)
from aimet_torch.fixed_point.tensor import Int16QuantizedTensor

_ABC_PKG_ROOT = Path(__file__).resolve().parents[4] / "abc_lut-shuai"
_JSON = _ABC_PKG_ROOT / "lut_int_general" / "output" / "lut_test" / "power_2_clz_lut.json"
_HAS_ABC = _JSON.is_file()
if _HAS_ABC:
    sys.path.insert(0, str(_ABC_PKG_ROOT))


@pytest.mark.skipif(not _HAS_ABC, reason="abc power_2_clz_lut.json not in workspace")
def test_evaluate_clz_power2_matches_lut_int_general():
    from lut_int_general.quantization.lut import infer_with_clz_normalized_lut  # type: ignore
    func_name, clz_body = load_clz_lut_from_json(_JSON)
    assert func_name == "power_2"

    q_samples = torch.tensor([2000, 8000, 16000], dtype=torch.int16)
    y_aimet = evaluate_clz_normalized_lut_int16(
        q_samples, clz_body, func_name
    ).to(torch.int32)
    payload = json.loads(_JSON.read_text(encoding="utf-8"))
    y_abc = torch.tensor(
        infer_with_clz_normalized_lut(
            payload, q_samples.numpy(), input_dtype="int16"
        )["output_quantized"],
        dtype=torch.int32,
    )
    assert (y_aimet - y_abc).abs().max().item() <= 3


@pytest.mark.skipif(not _HAS_ABC, reason="abc power_2_clz_lut.json not in workspace")
def test_square_clz_kernel_registered():
    import aimet_torch.fixed_point.kernels  # noqa: F401

    _, clz_body = load_clz_lut_from_json(_JSON, func_name="power_2")
    x = Int16QuantizedTensor(
        int_repr=torch.tensor([4000, 12000], dtype=torch.int32),
        scale=torch.tensor(0.00024414807580797754, dtype=torch.float32),
        zero_point=torch.tensor(0, dtype=torch.int32),
        qmin=-32768,
        qmax=32767,
    )
    oq = clz_body["quantization"]["output"]
    out_enc = OutputEncoding(
        scale=torch.tensor(oq["scale"], dtype=torch.float32),
        zero_point=torch.tensor(oq["zero_point"], dtype=torch.int32),
        qmin=int(oq["min"]),
        qmax=int(oq["max"]),
    )
    y = get_fixed_kernel(custom.Square)([x], {}, out_enc, {"clz_lut": clz_body, "clz_func_name": "power_2"})
    y_ref = evaluate_clz_normalized_lut_int16(
        x.int_repr.to(torch.int16), clz_body, "power_2"
    )
    assert (y.int_repr.to(torch.int32) - y_ref.to(torch.int32)).abs().max().item() <= 3


@pytest.mark.skipif(not _HAS_ABC, reason="abc power_2_clz_lut.json not in workspace")
def test_power2_negative_q_matches_abs_square_not_abc_zero():
    """AIMET uses |x| for power_2; abc golden returns 0 for negative q (positive-domain LUT)."""

    import json

    from lut_int_general.quantization.lut import infer_with_clz_normalized_lut  # type: ignore

    func_name, clz_body = load_clz_lut_from_json(_JSON)
    q_neg = torch.tensor([-8000, -2000], dtype=torch.int16)
    y_aimet = evaluate_clz_normalized_lut_int16(q_neg, clz_body, func_name).to(torch.int32)
    y_pos = evaluate_clz_normalized_lut_int16(
        torch.tensor([8000, 2000], dtype=torch.int16), clz_body, func_name
    ).to(torch.int32)
    torch.testing.assert_close(y_aimet, y_pos, rtol=0, atol=0)

    payload = json.loads(_JSON.read_text(encoding="utf-8"))
    y_abc = torch.tensor(
        infer_with_clz_normalized_lut(
            payload, q_neg.numpy(), input_dtype="int16"
        )["output_quantized"],
        dtype=torch.int32,
    )
    assert torch.all(y_abc == 0).item()
    assert (y_aimet - y_abc).abs().max().item() > 0

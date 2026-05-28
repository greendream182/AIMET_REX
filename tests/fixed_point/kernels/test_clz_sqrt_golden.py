"""CLZ sqrt LUT: aimet evaluator vs abc_lut-shuai/lut_int_general."""

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

_ABC_ROOT = Path(__file__).resolve().parents[4] / "abc_lut-shuai"
_JSON = _ABC_ROOT / "lut_int_general" / "output" / "lut_test" / "sqrt_clz_lut.json"
_HAS_ABC = _JSON.is_file()
if _HAS_ABC:
    sys.path.insert(0, str(_ABC_ROOT))
    _HAS_INFER = True
else:
    _HAS_INFER = False


@pytest.mark.skipif(not _HAS_ABC, reason="abc sqrt_clz_lut.json not in workspace")
def test_evaluate_clz_sqrt_matches_lut_int_general():
    from lut_int_general.quantization.lut import infer_with_clz_normalized_lut  # type: ignore

    func_name, clz_body = load_clz_lut_from_json(_JSON)
    assert func_name == "sqrt"

    q_samples = torch.tensor(
        [0, 100, 1000, 5000, 20000, 32767], dtype=torch.int16
    )
    y_aimet = evaluate_clz_normalized_lut_int16(
        q_samples, clz_body, func_name
    ).to(torch.int32)
    abc_out = infer_with_clz_normalized_lut(
        json.loads(_JSON.read_text(encoding="utf-8")),
        q_samples.numpy(),
        input_dtype="int16",
    )
    y_abc = torch.tensor(abc_out["output_quantized"], dtype=torch.int32)
    torch.testing.assert_close(y_aimet, y_abc, rtol=0, atol=0)


@pytest.mark.skipif(not _HAS_ABC, reason="abc sqrt_clz_lut.json not in workspace")
def test_rsqrt_reciprocal_kernels_registered():
    import aimet_torch.fixed_point.kernels  # noqa: F401
    from aimet_torch._base.nn.modules import custom
    from aimet_torch.fixed_point import get_fixed_kernel

    get_fixed_kernel(custom.RSqrt)
    get_fixed_kernel(custom.Reciprocal)


@pytest.mark.skipif(not _HAS_ABC, reason="abc sqrt_clz_lut.json not in workspace")
def test_sqrt_clz_kernel_registered():
    import aimet_torch.fixed_point.kernels  # noqa: F401
    from aimet_torch._base.nn.modules import custom
    from aimet_torch.fixed_point import get_fixed_kernel
    from aimet_torch.fixed_point.encoding import OutputEncoding
    from aimet_torch.fixed_point.tensor import Int16QuantizedTensor

    _, clz_body = load_clz_lut_from_json(_JSON)
    x = Int16QuantizedTensor(
        int_repr=torch.tensor([1000, 4000], dtype=torch.int32),
        scale=torch.tensor(0.00024414807580797754, dtype=torch.float32),
        zero_point=torch.tensor(0, dtype=torch.int32),
        qmin=-32768,
        qmax=32767,
    )
    out_enc = OutputEncoding(
        scale=torch.tensor(8.63193800087341e-05, dtype=torch.float32),
        zero_point=torch.tensor(0, dtype=torch.int32),
        qmin=-32768,
        qmax=32767,
    )
    y = get_fixed_kernel(custom.Sqrt)([x], {}, out_enc, {"clz_lut": clz_body})
    y_ref = evaluate_clz_normalized_lut_int16(
        x.int_repr.to(torch.int16), clz_body, "sqrt"
    )
    torch.testing.assert_close(
        y.int_repr.to(torch.int32), y_ref.to(torch.int32), rtol=0, atol=0
    )

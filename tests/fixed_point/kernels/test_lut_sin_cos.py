"""sin/cos PWL + phase fold vs abc_lut-shuai/lut_int_general."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

from aimet_torch._base.nn.modules import custom
from aimet_torch.fixed_point import InputEncoding, OutputEncoding, get_fixed_kernel
from aimet_torch.fixed_point.kernels.lut import evaluate_pwl_lut_int16
from aimet_torch.fixed_point.offline.lut_gen import (
    fold_periodic_input_to_principal_range,
    pwl_lut_from_json_dict,
)
from aimet_torch.fixed_point.tensor import Int16QuantizedTensor

_ABC_ROOT = Path(__file__).resolve().parents[4] / "abc_lut-shuai"
_JSON = _ABC_ROOT / "lut_int_general" / "output" / "lut_test" / "sin_lut.json"
_HAS_ABC = _JSON.is_file()


def _enc_from_sin_json(body: dict) -> tuple[InputEncoding, OutputEncoding]:
    iq = body["quantization"]["input"]
    oq = body["quantization"]["output"]
    in_enc = InputEncoding(
        scale=torch.tensor(iq["scale"], dtype=torch.float32),
        zero_point=torch.tensor(iq["zero_point"], dtype=torch.int32),
        qmin=int(iq["min"]),
        qmax=int(iq["max"]),
    )
    out_enc = OutputEncoding(
        scale=torch.tensor(oq["scale"], dtype=torch.float32),
        zero_point=torch.tensor(oq["zero_point"], dtype=torch.int32),
        qmin=int(oq["min"]),
        qmax=int(oq["max"]),
    )
    return in_enc, out_enc


@pytest.mark.skipif(not _HAS_ABC, reason="sin_lut.json not in workspace")
def test_sin_pwl_matches_infer_with_lut_phase_fold():
    sys.path.insert(0, str(_ABC_ROOT))
    from lut_int_general.quantization.lut import infer_with_lut  # type: ignore

    payload = json.loads(_JSON.read_text(encoding="utf-8"))
    body = payload["sin"]
    in_enc, out_enc = _enc_from_sin_json(body)
    pwl = pwl_lut_from_json_dict(payload, "sin")

    q_wide = torch.tensor([-40000, -20000, 0, 15000, 30000], dtype=torch.int32)
    q_fold = fold_periodic_input_to_principal_range(q_wide, "sin", in_enc)
    y_aimet = evaluate_pwl_lut_int16(q_fold.to(torch.int16), pwl).to(torch.int32)

    y_abc = torch.tensor(
        infer_with_lut(
            payload,
            q_wide.numpy(),
            input_dtype="int16",
            enable_periodic_fold=True,
        )["output_quantized"],
        dtype=torch.int32,
    )
    assert (y_aimet - y_abc).abs().max().item() <= 3


@pytest.mark.skipif(not _HAS_ABC, reason="sin_lut.json not in workspace")
def test_cos_reuses_sin_lut_with_phase_fold():
    sys.path.insert(0, str(_ABC_ROOT))
    from lut_int_general.quantization.lut import infer_with_lut  # type: ignore

    payload = json.loads(_JSON.read_text(encoding="utf-8"))
    body = payload["sin"]
    in_enc, out_enc = _enc_from_sin_json(body)
    pwl = pwl_lut_from_json_dict(payload, "sin")

    q_wide = torch.tensor([-25000, -5000, 10000, 28000], dtype=torch.int32)
    q_fold = fold_periodic_input_to_principal_range(q_wide, "cos", in_enc)
    y_aimet = evaluate_pwl_lut_int16(q_fold.to(torch.int16), pwl).to(torch.int32)

    y_abc = torch.tensor(
        infer_with_lut(
            payload,
            q_wide.numpy(),
            input_dtype="int16",
            enable_periodic_fold=True,
            cos_uses_sin_lut=True,
        )["output_quantized"],
        dtype=torch.int32,
    )
    assert (y_aimet - y_abc).abs().max().item() <= 3


@pytest.mark.skipif(not _HAS_ABC, reason="sin_lut.json not in workspace")
def test_sin_cos_kernels_registered():
    import aimet_torch.fixed_point.kernels  # noqa: F401

    payload = json.loads(_JSON.read_text(encoding="utf-8"))
    body = payload["sin"]
    _, out_enc = _enc_from_sin_json(body)
    x = Int16QuantizedTensor(
        int_repr=torch.tensor([0, 8000, -12000], dtype=torch.int32),
        scale=torch.tensor(body["quantization"]["input"]["scale"], dtype=torch.float32),
        zero_point=torch.tensor(0, dtype=torch.int32),
        qmin=-32768,
        qmax=32767,
    )
    extra = {
        "pwl_lut": pwl_lut_from_json_dict(payload, "sin"),
        "pwl_input_encoding": _enc_from_sin_json(body)[0],
        "phase_fold": "sin",
    }
    y_sin = get_fixed_kernel(custom.Sin)([x], {}, out_enc, extra)
    extra["phase_fold"] = "cos"
    y_cos = get_fixed_kernel(custom.Cos)([x], {}, out_enc, extra)
    assert y_sin.int_repr.shape == x.int_repr.shape
    assert y_cos.int_repr.shape == x.int_repr.shape

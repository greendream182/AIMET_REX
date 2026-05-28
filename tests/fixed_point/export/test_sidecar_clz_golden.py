"""Frozen abc CLZ JSON in sidecar bundle → kernel path vs golden evaluator."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

import aimet_torch.fixed_point.kernels  # noqa: F401
from aimet_torch._base.nn.modules import custom
from aimet_torch.fixed_point import get_fixed_kernel
from aimet_torch.fixed_point.encoding import OutputEncoding
from aimet_torch.fixed_point.encoding_export import fixed_point_tensor_bundle
from aimet_torch.fixed_point.export import INT16_SIDECAR_VERSION
from aimet_torch.fixed_point.export.sidecar_loader import layer_bundle_to_runtime_extra
from aimet_torch.fixed_point.kernels.clz_lut import (
    evaluate_clz_normalized_lut_int16,
    load_clz_lut_from_json,
)
from aimet_torch.fixed_point.tensor import Int16QuantizedTensor

_ABC_ROOT = Path(__file__).resolve().parents[4] / "abc_lut-shuai"
_RECIP_JSON = _ABC_ROOT / "lut_int_general" / "output" / "lut_test" / "reciprocal_clz_lut.json"
_POWER2_JSON = _ABC_ROOT / "lut_int_general" / "output" / "lut_test" / "power_2_clz_lut.json"
_HAS_RECIP = _RECIP_JSON.is_file()
_HAS_POWER2 = _POWER2_JSON.is_file()
if _HAS_RECIP or _HAS_POWER2:
    sys.path.insert(0, str(_ABC_ROOT))


def _encoding_from_quant_block(block: dict) -> OutputEncoding:
    return OutputEncoding(
        scale=torch.tensor(block["scale"], dtype=torch.float32),
        zero_point=torch.tensor(block["zero_point"], dtype=torch.int32),
        qmin=int(block["min"]),
        qmax=int(block["max"]),
    )


def _sidecar_bundle_from_golden(path: Path, func_name: str, *, op: str) -> dict:
    func_name_loaded, clz_body = load_clz_lut_from_json(path, func_name=func_name)
    assert func_name_loaded == func_name
    out_enc = _encoding_from_quant_block(clz_body["quantization"]["output"])
    bundle = fixed_point_tensor_bundle(layer_name="0", output_encoding=out_enc)
    bundle["op"] = op
    bundle["clz"] = {func_name: clz_body}
    return bundle


def _int16_from_golden_input(q: torch.Tensor, clz_body: dict) -> Int16QuantizedTensor:
    inp = clz_body["quantization"]["input"]
    return Int16QuantizedTensor(
        int_repr=q.to(torch.int32),
        scale=torch.tensor(inp["scale"], dtype=torch.float32),
        zero_point=torch.tensor(inp["zero_point"], dtype=torch.int32),
        qmin=int(inp["min"]),
        qmax=int(inp["max"]),
    )


@pytest.mark.skipif(not _HAS_RECIP, reason="abc reciprocal_clz_lut.json not in workspace")
def test_frozen_reciprocal_sidecar_extra_matches_golden_evaluator():
    from lut_int_general.quantization.lut import infer_with_clz_normalized_lut  # type: ignore

    func_name, clz_body = load_clz_lut_from_json(_RECIP_JSON)
    bundle = _sidecar_bundle_from_golden(_RECIP_JSON, func_name, op="Reciprocal")
    extra = layer_bundle_to_runtime_extra(bundle)
    assert extra["clz_func_name"] == "reciprocal"
    assert "clz_lut" in extra

    q_samples = torch.tensor([500, 2000, 8000, 20000], dtype=torch.int16)
    y_ref = evaluate_clz_normalized_lut_int16(q_samples, clz_body, func_name).to(torch.int32)
    y_abc = torch.tensor(
        infer_with_clz_normalized_lut(
            json.loads(_RECIP_JSON.read_text(encoding="utf-8")),
            q_samples.numpy(),
            input_dtype="int16",
        )["output_quantized"],
        dtype=torch.int32,
    )
    assert (y_ref - y_abc).abs().max().item() <= 3

    out_enc = _encoding_from_quant_block(clz_body["quantization"]["output"])
    x = _int16_from_golden_input(q_samples, clz_body)
    y_kernel = get_fixed_kernel(custom.Reciprocal)([x], {}, out_enc, extra)
    torch.testing.assert_close(
        y_kernel.int_repr.to(torch.int32).flatten(),
        y_ref.flatten(),
        rtol=0,
        atol=3,
    )


@pytest.mark.skipif(not _HAS_POWER2, reason="abc power_2_clz_lut.json not in workspace")
def test_frozen_power2_sidecar_extra_matches_golden_evaluator():
    from lut_int_general.quantization.lut import infer_with_clz_normalized_lut  # type: ignore

    func_name, clz_body = load_clz_lut_from_json(_POWER2_JSON, func_name="power_2")
    bundle = _sidecar_bundle_from_golden(_POWER2_JSON, func_name, op="Square")
    extra = layer_bundle_to_runtime_extra(bundle)
    assert extra["clz_func_name"] == "power_2"

    q_samples = torch.tensor([2000, 8000, 16000], dtype=torch.int16)
    y_ref = evaluate_clz_normalized_lut_int16(q_samples, clz_body, func_name).to(torch.int32)
    y_abc = torch.tensor(
        infer_with_clz_normalized_lut(
            json.loads(_POWER2_JSON.read_text(encoding="utf-8")),
            q_samples.numpy(),
            input_dtype="int16",
        )["output_quantized"],
        dtype=torch.int32,
    )
    assert (y_ref - y_abc).abs().max().item() <= 3

    out_enc = _encoding_from_quant_block(clz_body["quantization"]["output"])
    x = _int16_from_golden_input(q_samples, clz_body)
    y_kernel = get_fixed_kernel(custom.Square)([x], {}, out_enc, extra)
    torch.testing.assert_close(
        y_kernel.int_repr.to(torch.int32).flatten(),
        y_ref.flatten(),
        rtol=0,
        atol=3,
    )


def test_sidecar_doc_shape_from_golden_reciprocal():
    pytest.importorskip("pathlib")
    if not _HAS_RECIP:
        pytest.skip("abc reciprocal_clz_lut.json not in workspace")
    bundle = _sidecar_bundle_from_golden(_RECIP_JSON, "reciprocal", op="Reciprocal")
    assert "clz" in bundle and "reciprocal" in bundle["clz"]
    assert bundle["output_encoding"]["scale"] is not None

"""Sidecar JSON → runtime ``extra`` for INT16 dispatch."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxscript")
import torch.nn as nn  # noqa: E402

import aimet_torch.fixed_point.kernels  # noqa: F401, E402
from aimet_torch.fixed_point import (  # noqa: E402
    ExecutionMode,
    Int16QuantizedTensor,
    quant_execution_mode,
)
from aimet_torch.fixed_point.export import (  # noqa: E402
    attach_int16_sidecar_to_model,
    build_int16_sidecar_document,
    detach_int16_sidecar_from_model,
    export_int16_sidecar_json,
    get_int16_sidecar_extra,
    layer_bundle_to_runtime_extra,
    load_int16_sidecar_json,
    maybe_attach_int16_sidecar_from_env,
)
from aimet_torch.fixed_point.export.sidecar_loader import (
    build_runtime_extra_by_layer,
    resolve_sidecar_layer_to_module_names,
)
from aimet_torch.fixed_point.offline.clz_gen import resolve_abc_lut_root
from aimet_torch.v2.nn import QuantizedSigmoid  # noqa: E402
from aimet_torch.v2.nn.modules.custom import (  # noqa: E402
    QuantizedReciprocal,
    QuantizedSin,
    QuantizedSqrt,
    QuantizedSquare,
)
from aimet_torch.v2.quantization.affine import Quantize  # noqa: E402


def _init_unary(m: nn.Module, in_range=2.0, out_min=-1.0, out_max=1.0):
    m.input_quantizers[0] = Quantize((), 16, symmetric=True)
    m.output_quantizers[0] = Quantize((), 16, symmetric=True)
    m.input_quantizers[0].min = nn.Parameter(torch.tensor(-in_range))
    m.input_quantizers[0].max = nn.Parameter(torch.tensor(in_range))
    m.output_quantizers[0].min = nn.Parameter(torch.tensor(out_min))
    m.output_quantizers[0].max = nn.Parameter(torch.tensor(out_max))


def test_resolve_sidecar_layer_via_onnx_tensor_names():
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.op = QuantizedSin()

    model = nn.Sequential(Block())
    _init_unary(model[0].op, in_range=3.15)
    doc = build_int16_sidecar_document(model)
    true_key = "0.op"
    assert true_key in doc["layers"]

    legacy_bundle = dict(doc["layers"][true_key])
    legacy_bundle["onnx_tensor_names"] = [true_key]
    sidecar_layers = {"legacy_sin": legacy_bundle}

    name_map = resolve_sidecar_layer_to_module_names(sidecar_layers, model)
    assert name_map["legacy_sin"] == true_key

    detach_int16_sidecar_from_model(model)
    attach_int16_sidecar_to_model(
        model, {"format": "aimet_rx_int16_fixed_sidecar", "layers": sidecar_layers}
    )
    assert get_int16_sidecar_extra(model[0].op) is not None
    detach_int16_sidecar_from_model(model)


def test_sidecar_loader_rejects_invalid_schema(tmp_path):
    bad_format = tmp_path / "bad_format.int16.json"
    bad_format.write_text(json.dumps({"format": "not_aimet", "layers": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="Not an AIMET RX INT16 sidecar"):
        load_int16_sidecar_json(str(bad_format))

    bad_version = tmp_path / "bad_version.int16.json"
    bad_version.write_text(
        json.dumps(
            {
                "format": "aimet_rx_int16_fixed_sidecar",
                "version": "0.0.0",
                "layers": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsupported AIMET RX INT16 sidecar version"):
        load_int16_sidecar_json(str(bad_version))

    bad_layers = tmp_path / "bad_layers.int16.json"
    bad_layers.write_text(
        json.dumps(
            {
                "format": "aimet_rx_int16_fixed_sidecar",
                "version": "1.0.0-int16-fixed",
                "layers": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sidecar\\['layers'\\] must be a mapping"):
        load_int16_sidecar_json(str(bad_layers))


def test_sidecar_runtime_extra_rejects_empty_payloads():
    with pytest.raises(ValueError, match="'pwl' must be a non-empty mapping"):
        layer_bundle_to_runtime_extra({"pwl": {}})
    with pytest.raises(ValueError, match="'clz' must be a non-empty mapping"):
        layer_bundle_to_runtime_extra({"clz": {}})
    with pytest.raises(ValueError, match="'clz\\[sqrt\\]' must be a mapping"):
        layer_bundle_to_runtime_extra({"clz": {"sqrt": []}})


def test_attach_sidecar_strict_missing_layer_fails():
    model = nn.Sequential(QuantizedSin())
    _init_unary(model[0], in_range=3.15)
    doc = build_int16_sidecar_document(model)
    moved = dict(doc)
    moved["layers"] = {"missing_sin": doc["layers"]["0"]}

    with pytest.raises(KeyError, match="Sidecar layers not found on model"):
        attach_int16_sidecar_to_model(model, moved, strict=True)


def test_maybe_attach_from_env(monkeypatch):
    model = nn.Sequential(QuantizedSin())
    _init_unary(model[0], in_range=3.15)
    x = torch.tensor([[0.2, -0.8]], dtype=torch.float32)

    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "m.int16.json")
        export_int16_sidecar_json(model, path)
        monkeypatch.setenv("AIMET_RX_INT16_SIDECAR_PATH", path)

        used = maybe_attach_int16_sidecar_from_env(model)
        assert used == str(Path(path).resolve())
        assert maybe_attach_int16_sidecar_from_env(model) == used

        with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
            y = model(x)
        assert isinstance(y, Int16QuantizedTensor)

    monkeypatch.delenv("AIMET_RX_INT16_SIDECAR_PATH", raising=False)
    detach_int16_sidecar_from_model(model)


def test_maybe_attach_from_env_reattaches_after_detach(monkeypatch):
    model = nn.Sequential(QuantizedSin())
    _init_unary(model[0], in_range=3.15)

    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "m.int16.json")
        export_int16_sidecar_json(model, path)
        monkeypatch.setenv("AIMET_RX_INT16_SIDECAR_PATH", path)

        maybe_attach_int16_sidecar_from_env(model)
        assert get_int16_sidecar_extra(model[0]) is not None

        detach_int16_sidecar_from_model(model)
        assert get_int16_sidecar_extra(model[0]) is None

        maybe_attach_int16_sidecar_from_env(model)
        assert get_int16_sidecar_extra(model[0]) is not None

    monkeypatch.delenv("AIMET_RX_INT16_SIDECAR_PATH", raising=False)
    detach_int16_sidecar_from_model(model)


def test_layer_bundle_to_runtime_extra_pwl_keys():
    model = nn.Sequential(QuantizedSigmoid())
    _init_unary(model[0], in_range=4.0, out_min=0.0, out_max=1.0)
    doc = build_int16_sidecar_document(model)
    extra = layer_bundle_to_runtime_extra(doc["layers"]["0"])
    assert "pwl_lut" in extra
    assert "thresholds" in extra["pwl_lut"]
    assert extra["pwl_lut"]["thresholds"].dtype == torch.int32


def test_attach_sidecar_skips_online_lut_regen():
    model = nn.Sequential(QuantizedSin())
    _init_unary(model[0], in_range=3.15)
    x = torch.tensor([[0.5, -1.0, 1.2]], dtype=torch.float32)

    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "m.int16.json")
        export_int16_sidecar_json(model, path)
        doc = load_int16_sidecar_json(path)

    detach_int16_sidecar_from_model(model)
    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y_online = model(x)

    attach_int16_sidecar_to_model(model, doc)
    assert get_int16_sidecar_extra(model[0]) is not None
    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y_sidecar = model(x)

    detach_int16_sidecar_from_model(model)
    assert get_int16_sidecar_extra(model[0]) is None

    assert isinstance(y_online, Int16QuantizedTensor)
    assert isinstance(y_sidecar, Int16QuantizedTensor)
    torch.testing.assert_close(
        y_online.int_repr.to(torch.int32),
        y_sidecar.int_repr.to(torch.int32),
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(resolve_abc_lut_root() is None, reason="abc_lut-shuai not found")
def test_attach_sidecar_sqrt_clz_matches_online():
    model = nn.Sequential(QuantizedSqrt())
    _init_unary(model[0], in_range=4.0, out_min=0.2, out_max=2.0)
    model[0].input_quantizers[0].min = nn.Parameter(torch.tensor(0.1))
    x = torch.tensor([[0.25, 1.0, 2.25]], dtype=torch.float32)

    doc = build_int16_sidecar_document(model)
    extras = build_runtime_extra_by_layer(doc)
    assert "clz_lut" in extras["0"]

    detach_int16_sidecar_from_model(model)
    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y_online = model(x)

    attach_int16_sidecar_to_model(model, doc)
    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y_sidecar = model(x)
    detach_int16_sidecar_from_model(model)

    torch.testing.assert_close(
        y_online.int_repr.to(torch.int32),
        y_sidecar.int_repr.to(torch.int32),
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(resolve_abc_lut_root() is None, reason="abc_lut-shuai not found")
def test_attach_sidecar_reciprocal_clz_matches_online():
    model = nn.Sequential(QuantizedReciprocal())
    model[0].input_quantizers[0] = Quantize((), 16, symmetric=True)
    model[0].output_quantizers[0] = Quantize((), 16, symmetric=True)
    model[0].input_quantizers[0].min = nn.Parameter(torch.tensor(0.2))
    model[0].input_quantizers[0].max = nn.Parameter(torch.tensor(2.0))
    model[0].output_quantizers[0].min = nn.Parameter(torch.tensor(0.2))
    model[0].output_quantizers[0].max = nn.Parameter(torch.tensor(3.0))
    x = torch.tensor([[0.5, 1.0, 1.5]], dtype=torch.float32)
    model(x)

    doc = build_int16_sidecar_document(model)
    extras = build_runtime_extra_by_layer(doc)
    assert "clz_lut" in extras["0"]
    assert extras["0"].get("clz_func_name") == "reciprocal"

    detach_int16_sidecar_from_model(model)
    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y_online = model(x)

    attach_int16_sidecar_to_model(model, doc)
    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y_sidecar = model(x)
    detach_int16_sidecar_from_model(model)

    torch.testing.assert_close(
        y_online.int_repr.to(torch.int32),
        y_sidecar.int_repr.to(torch.int32),
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(resolve_abc_lut_root() is None, reason="abc_lut-shuai not found")
def test_attach_sidecar_square_clz_matches_online():
    model = nn.Sequential(QuantizedSquare())
    model[0].input_quantizers[0] = Quantize((), 16, symmetric=True)
    model[0].output_quantizers[0] = Quantize((), 16, symmetric=True)
    model[0].input_quantizers[0].min = nn.Parameter(torch.tensor(-2.0))
    model[0].input_quantizers[0].max = nn.Parameter(torch.tensor(2.0))
    model[0].output_quantizers[0].min = nn.Parameter(torch.tensor(0.0))
    model[0].output_quantizers[0].max = nn.Parameter(torch.tensor(4.0))
    x = torch.tensor([[-1.5, 0.5, 1.5]], dtype=torch.float32)
    model(x)

    doc = build_int16_sidecar_document(model)
    assert "clz_lut" in build_runtime_extra_by_layer(doc)["0"]
    detach_int16_sidecar_from_model(model)
    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y_online = model(x)

    attach_int16_sidecar_to_model(model, doc)
    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y_sidecar = model(x)
    detach_int16_sidecar_from_model(model)

    torch.testing.assert_close(
        y_online.int_repr.to(torch.int32),
        y_sidecar.int_repr.to(torch.int32),
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(resolve_abc_lut_root() is None, reason="abc_lut-shuai not found")
def test_export_json_reload_reciprocal_matches_online_int_repr():
    """Float quantizer → export ``*.int16.json`` → load → INT16 must match online."""

    model = nn.Sequential(QuantizedReciprocal())
    model[0].input_quantizers[0] = Quantize((), 16, symmetric=True)
    model[0].output_quantizers[0] = Quantize((), 16, symmetric=True)
    model[0].input_quantizers[0].min = nn.Parameter(torch.tensor(0.2))
    model[0].input_quantizers[0].max = nn.Parameter(torch.tensor(2.0))
    model[0].output_quantizers[0].min = nn.Parameter(torch.tensor(0.2))
    model[0].output_quantizers[0].max = nn.Parameter(torch.tensor(3.0))
    x = torch.tensor([[0.5, 1.0, 1.5]], dtype=torch.float32)
    model(x)

    detach_int16_sidecar_from_model(model)
    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y_online = model(x)

    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "recip.int16.json")
        export_int16_sidecar_json(model, path)
        loaded = load_int16_sidecar_json(path)
        assert "clz" in loaded["layers"]["0"]

        detach_int16_sidecar_from_model(model)
        attach_int16_sidecar_to_model(model, path)
        with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
            y_reload = model(x)
        detach_int16_sidecar_from_model(model)

    torch.testing.assert_close(
        y_online.int_repr.to(torch.int32),
        y_reload.int_repr.to(torch.int32),
        rtol=0,
        atol=0,
    )

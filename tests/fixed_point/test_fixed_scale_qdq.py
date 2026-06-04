import pytest
import torch
import torch.nn as nn

from aimet_torch.fixed_point import ExecutionMode, quant_execution_mode
from aimet_torch.fixed_point.metrics import (
    DEFAULT_COMPARE_MODES,
    compare_modes,
    compute_pair_metrics,
)
from aimet_torch.fixed_point.encoding import FixedScaleEncoding
from aimet_torch.fixed_point.encoding_export import (
    fixed_scale_encoding_from_dict,
    fixed_scale_encoding_to_dict,
)
from aimet_torch.fixed_point.fixed_scale_qdq import (
    dequantize_with_fixed_scale,
    quantize_dequantize_with_fixed_scale,
    quantize_with_fixed_scale,
)
from aimet_torch.fixed_point.offline.scale_fixed import quantize_scale_to_m_rshift
from aimet_torch.v2.quantization.affine.backends import _derive_qmin_qmax
from aimet_torch.v2.quantization.affine.backends.torch_builtins import quantize_dequantize

pytest.importorskip("onnxscript")
from aimet_torch.v2.nn import QuantizedLinear, QuantizedReLU  # noqa: E402
from aimet_torch.v2.quantization.affine import Quantize  # noqa: E402


def teardown_function():
    from aimet_torch.fixed_point import set_quant_execution_mode

    set_quant_execution_mode(ExecutionMode.FP32_QDQ)


def test_quantize_scale_to_m_rshift_one_eighth():
    m, r = quantize_scale_to_m_rshift(0.125)
    approx = float(m.item()) / (2 ** int(r.item()))
    assert abs(approx - 0.125) / 0.125 < 1e-3


def test_fixed_scale_quantize_dequant_roundtrip():
    m, r = quantize_scale_to_m_rshift(0.125)
    enc = FixedScaleEncoding(
        m_int16=m,
        rshift=r,
        zero_point=torch.tensor(0, dtype=torch.int32),
        qmin=-128,
        qmax=127,
    )
    x = torch.tensor(0.5, dtype=torch.float32)
    q = quantize_with_fixed_scale(x, enc)
    y = dequantize_with_fixed_scale(q, enc)
    assert abs(float(y.item()) - 0.5) < 1e-3


def test_v2_quantize_dequantize_fixed_scale_mode_matches_fp32():
    tensor = torch.randn(4, 8, dtype=torch.float32)
    scale = torch.tensor(0.05, dtype=torch.float32)
    offset = torch.tensor(0.0, dtype=torch.float32)

    with quant_execution_mode(ExecutionMode.FP32_QDQ):
        y_ref = quantize_dequantize(tensor, scale, offset, -128, 127)
    with quant_execution_mode(ExecutionMode.FIXED_SCALE_QDQ):
        y_fix = quantize_dequantize(tensor, scale, offset, -128, 127)

    max_abs = (y_ref - y_fix).abs().max().item()
    assert max_abs < 1e-4 * max(float(tensor.abs().max().item()), 1.0)


def test_fixed_scale_encoding_json_roundtrip():
    enc = FixedScaleEncoding(
        m_int16=torch.tensor([1, 2], dtype=torch.uint16),
        rshift=torch.tensor([3, 4], dtype=torch.int8),
        zero_point=torch.tensor([0, 1], dtype=torch.int32),
        qmin=-128,
        qmax=127,
        scale_fp_legacy=torch.tensor([0.125, 0.0625], dtype=torch.float32),
    )
    data = fixed_scale_encoding_to_dict(enc)
    restored = fixed_scale_encoding_from_dict(data)
    assert restored.m_int16.dtype == torch.uint16
    assert torch.equal(restored.m_int16.to(torch.int64), enc.m_int16.to(torch.int64))
    assert torch.equal(restored.rshift, enc.rshift)


def test_fixed_scale_encoding_from_dict_missing_fields_raises():
    with pytest.raises(ValueError, match="m_int16"):
        fixed_scale_encoding_from_dict({"qmin": 0, "qmax": 255, "zero_point": 0})


def test_fixed_scale_qdq_runtime_derives_m_r_not_float_scale_path():
    """Runtime Q/DQ uses (M,r); calibrated float scale only seeds offline conversion."""

    from aimet_torch.fixed_point.fixed_scale_qdq import quantize_dequantize_from_float_encoding

    tensor = torch.tensor([0.12, -0.37, 0.51], dtype=torch.float32)
    scale = torch.tensor(0.05, dtype=torch.float32)
    offset = torch.tensor(0.0, dtype=torch.float32)

    with quant_execution_mode(ExecutionMode.FP32_QDQ):
        y_fp32 = quantize_dequantize(tensor, scale, offset, -128, 127)
    with quant_execution_mode(ExecutionMode.FIXED_SCALE_QDQ):
        y_derived = quantize_dequantize(tensor, scale, offset, -128, 127)
        m, r = quantize_scale_to_m_rshift(float(scale.item()))
        enc = FixedScaleEncoding(
            m_int16=m,
            rshift=r,
            zero_point=torch.tensor(0, dtype=torch.int32),
            qmin=-128,
            qmax=127,
        )
        y_explicit = quantize_dequantize_with_fixed_scale(tensor, enc)

    assert torch.allclose(
        y_derived.float(), y_explicit.float(), atol=1e-5, rtol=1e-5
    )
    assert torch.allclose(
        y_fp32.float(),
        y_derived.float(),
        atol=1e-4 * max(float(tensor.abs().max().item()), 1.0),
    )


def _assert_fp32_vs_fixed_scale_close(
    y_ref: torch.Tensor,
    y_fix: torch.Tensor,
    *,
    min_cosine: float = 0.9999,
    max_rel_err: float = 0.02,
):
    metrics = compute_pair_metrics(y_ref, y_fix)
    assert metrics["cosine_similarity"] >= min_cosine, (
        f"cosine={metrics['cosine_similarity']:.6f} < {min_cosine}"
    )
    denom = max(float(y_ref.abs().max().item()), 1e-6)
    rel = float(metrics["max_abs_error"]) / denom
    assert rel <= max_rel_err, (
        f"rel max_abs={rel:.6f} > {max_rel_err}, max_abs={metrics['max_abs_error']}"
    )


@pytest.mark.parametrize(
    "bitwidth,symmetric,scale_shape,offset_value",
    [
        pytest.param(4, True, (), 0.0, id="W4_sym_per_tensor"),
        pytest.param(8, True, (), 0.0, id="W8_sym_per_tensor"),
        pytest.param(8, False, (), 3.0, id="A8_asym_per_tensor"),
        pytest.param(8, True, (8,), 0.0, id="W8_sym_per_channel"),
        pytest.param(8, False, (8,), 2.0, id="A8_asym_per_channel"),
        pytest.param(16, False, (), 128.0, id="A16_asym_per_tensor"),
    ],
)
def test_qdq_fp32_vs_fixed_scale_wa_variants(
    bitwidth, symmetric, scale_shape, offset_value
):
    """Q/DQ boundary: fixed_scale matches fp32 across bitwidth / symmetry / scale layout."""

    qmin, qmax = _derive_qmin_qmax(bitwidth=bitwidth, signed=symmetric)
    torch.manual_seed(bitwidth * 10 + int(symmetric) + len(scale_shape))

    if scale_shape:
        scale = torch.linspace(0.02, 0.15, scale_shape[0], dtype=torch.float32)
        tensor = torch.randn(4, scale_shape[0], dtype=torch.float32)
        offset = torch.full(scale_shape, offset_value, dtype=torch.float32)
    else:
        scale = torch.tensor(0.04 + bitwidth * 0.002, dtype=torch.float32)
        tensor = torch.randn(4, 8, dtype=torch.float32)
        offset = torch.tensor(offset_value, dtype=torch.float32)

    with quant_execution_mode(ExecutionMode.FP32_QDQ):
        y_ref = quantize_dequantize(tensor, scale, offset, qmin, qmax)
    with quant_execution_mode(ExecutionMode.FIXED_SCALE_QDQ):
        y_fix = quantize_dequantize(tensor, scale, offset, qmin, qmax)

    # U16 grid has coarser steps; allow slightly looser bound
    min_cos = 0.9995 if bitwidth >= 16 else 0.9999
    max_rel = 0.03 if bitwidth >= 16 else 0.02
    _assert_fp32_vs_fixed_scale_close(y_ref, y_fix, min_cosine=min_cos, max_rel_err=max_rel)


def _init_quantizer(q: Quantize, *, lo: float, hi: float):
    q.min = nn.Parameter(torch.tensor(lo, dtype=torch.float32))
    q.max = nn.Parameter(torch.tensor(hi, dtype=torch.float32))


def _init_unary(m: QuantizedReLU, *, in_lo, in_hi, out_lo, out_hi):
    m.input_quantizers[0] = Quantize((), 16, symmetric=False)
    m.output_quantizers[0] = Quantize((), 16, symmetric=False)
    _init_quantizer(m.input_quantizers[0], lo=in_lo, hi=in_hi)
    _init_quantizer(m.output_quantizers[0], lo=out_lo, hi=out_hi)


@pytest.mark.parametrize(
    "weight_bw,act_bw,act_symmetric",
    [
        pytest.param(4, 8, True, id="W4_A8_sym"),
        pytest.param(8, 16, False, id="W8_A16_asym"),
    ],
)
def test_quantized_linear_fp32_vs_fixed_scale_mixed_wa(
    weight_bw, act_bw, act_symmetric
):
    """Tiny layer: mixed weight/activation bitwidth through real v2 quantizers."""

    out_features = 6
    model = QuantizedLinear(8, out_features)
    model.input_quantizers[0] = Quantize((), act_bw, symmetric=act_symmetric)
    model.param_quantizers["weight"] = Quantize(
        (out_features, 1), weight_bw, symmetric=True
    )
    model.output_quantizers[0] = Quantize((), act_bw, symmetric=act_symmetric)

    _init_quantizer(model.input_quantizers[0], lo=-1.5, hi=1.5)
    _init_quantizer(
        model.param_quantizers["weight"], lo=-0.4, hi=0.4
    )
    _init_quantizer(model.output_quantizers[0], lo=-2.0, hi=2.0)

    nn.init.uniform_(model.weight, -0.3, 0.3)
    nn.init.uniform_(model.bias, -0.05, 0.05)

    x = torch.randn(3, 8)
    with torch.no_grad():
        with quant_execution_mode(ExecutionMode.FP32_QDQ):
            y_ref = model(x)
        with quant_execution_mode(ExecutionMode.FIXED_SCALE_QDQ):
            y_fix = model(x)

    min_cos = 0.9995 if act_bw >= 16 else 0.9999
    _assert_fp32_vs_fixed_scale_close(y_ref, y_fix, min_cosine=min_cos, max_rel_err=0.025)


def test_quantize_scale_to_m_rshift_handles_tiny_scale():
    """Very small scales need r>31 before normalize; must not raise."""

    tiny = 1e-9
    m, r = quantize_scale_to_m_rshift(tiny)
    assert int(r.item()) <= 31
    approx = float(m.item()) / (2 ** int(r.item()))
    # After folding r to 31, very tiny scales have coarser M; still order-of-magnitude.
    assert abs(approx - tiny) / tiny < 0.5


def test_mlp_w4_u16_fp32_vs_fixed_scale():
    """Two-layer MLP: W4 per-channel weights + U16 asymmetric activations."""

    model = nn.Sequential(
        QuantizedLinear(12, 8),
        QuantizedReLU(),
        QuantizedLinear(8, 4),
    )
    for layer in (model[0], model[2]):
        layer.input_quantizers[0] = Quantize((), 16, symmetric=False)
        layer.param_quantizers["weight"] = Quantize(
            (layer.out_features, 1), 4, symmetric=True
        )
        layer.output_quantizers[0] = Quantize((), 16, symmetric=False)
        _init_quantizer(layer.input_quantizers[0], lo=0.0, hi=4.0)
        _init_quantizer(layer.param_quantizers["weight"], lo=-0.35, hi=0.35)
        _init_quantizer(layer.output_quantizers[0], lo=0.0, hi=4.0)
    _init_unary(model[1], in_lo=0.0, in_hi=4.0, out_lo=0.0, out_hi=4.0)

    for layer in model:
        if isinstance(layer, QuantizedLinear):
            nn.init.uniform_(layer.weight, -0.25, 0.25)
            nn.init.uniform_(layer.bias, -0.02, 0.02)

    x = torch.rand(5, 12) * 3.5
    with torch.no_grad():
        with quant_execution_mode(ExecutionMode.FP32_QDQ):
            y_ref = model(x)
        with quant_execution_mode(ExecutionMode.FIXED_SCALE_QDQ):
            y_fix = model(x)

    _assert_fp32_vs_fixed_scale_close(y_ref, y_fix, min_cosine=0.9995, max_rel_err=0.03)


def test_default_compare_modes_includes_fixed_scale_qdq():
    assert ExecutionMode.FP32_QDQ == DEFAULT_COMPARE_MODES[0]
    assert ExecutionMode.FIXED_SCALE_QDQ in DEFAULT_COMPARE_MODES


def test_compare_modes_fp32_vs_fixed_scale_smoke():
    model = QuantizedLinear(6, 4)
    model.input_quantizers[0] = Quantize((), 8, symmetric=True)
    model.param_quantizers["weight"] = Quantize((4, 1), 4, symmetric=True)
    model.output_quantizers[0] = Quantize((), 8, symmetric=True)
    _init_quantizer(model.input_quantizers[0], lo=-1.0, hi=1.0)
    _init_quantizer(model.param_quantizers["weight"], lo=-0.3, hi=0.3)
    _init_quantizer(model.output_quantizers[0], lo=-1.5, hi=1.5)
    nn.init.uniform_(model.weight, -0.2, 0.2)

    x = torch.randn(2, 6)
    report = compare_modes(
        model,
        x,
        [ExecutionMode.FP32_QDQ, ExecutionMode.FIXED_SCALE_QDQ],
        metrics=("max_abs_error", "cosine_similarity"),
    )
    pair = report["pairwise"]["fp32_qdq_vs_fixed_scale_qdq"]
    assert pair["cosine_similarity"] >= 0.9999
    assert pair["max_abs_error"] >= 0.0


def test_quantize_dequantize_with_fixed_scale_supports_grad():
    enc = FixedScaleEncoding(
        m_int16=torch.tensor(32767, dtype=torch.int16),
        rshift=torch.tensor(12, dtype=torch.int8),
        zero_point=torch.tensor(0, dtype=torch.int32),
        qmin=-128,
        qmax=127,
    )
    x = torch.randn(3, requires_grad=True)
    y = quantize_dequantize_with_fixed_scale(x, enc)
    y.sum().backward()
    assert x.grad is not None

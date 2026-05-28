import pytest

pytest.importorskip("onnxscript")
import torch
from torch import nn

from aimet_torch.fixed_point import ExecutionMode, quant_execution_mode
from aimet_torch.fixed_point.metrics import assert_fp16_vs_fp32_reference
from aimet_torch.quantizable_batchnorm import QuantizableBatchNorm2d
from aimet_torch.v2.nn import QuantizedLinear
from aimet_torch.v2.nn.modules.custom import QuantizedReshape
from aimet_torch.v2.nn.true_quant import (
    QuantizationMixin,
    _quantize_dequantize_if_applicable,
)
from aimet_torch.v2.quantization.affine.backends.torch_builtins import (
    quantize_dequantize,
)
from aimet_torch.v2.quantization.affine.quantizer import QuantizeDequantize


def test_v2_quantize_dequantize_fp16_mode_returns_half():
    tensor = torch.tensor([-0.25, 0.0, 0.25], dtype=torch.float32)
    scale = torch.tensor(0.125, dtype=torch.float32)
    offset = torch.tensor(0.0, dtype=torch.float32)

    with quant_execution_mode(ExecutionMode.FP16_QDQ):
        output = quantize_dequantize(tensor, scale, offset, -128, 127)

    assert output.dtype == torch.float16


def test_v2_quantize_dequantize_default_dtype_unchanged():
    tensor = torch.tensor([-0.25, 0.0, 0.25], dtype=torch.float32)
    scale = torch.tensor(0.125, dtype=torch.float32)
    offset = torch.tensor(0.0, dtype=torch.float32)

    with quant_execution_mode(ExecutionMode.FP32_QDQ):
        output = quantize_dequantize(tensor, scale, offset, -128, 127)

    assert output.dtype == torch.float32


def test_fp16_qdq_matches_fp32_qdq_within_cosine_threshold():
    tensor = torch.randn(4, 8, dtype=torch.float32)
    scale = torch.tensor(0.05, dtype=torch.float32)
    offset = torch.tensor(0.0, dtype=torch.float32)

    with quant_execution_mode(ExecutionMode.FP32_QDQ):
        y_fp32 = quantize_dequantize(tensor, scale, offset, -128, 127)
    with quant_execution_mode(ExecutionMode.FP16_QDQ):
        y_fp16 = quantize_dequantize(tensor, scale, offset, -128, 127)

    assert_fp16_vs_fp32_reference(y_fp16.float(), y_fp32)


def test_fp16_qdq_reshape_preserves_shape_tensor_as_metadata():
    module = QuantizedReshape()
    module.input_quantizers = nn.ModuleList([None, None])
    x = torch.randn(2, 3, dtype=torch.float32)
    shape = torch.tensor([3, 2], dtype=torch.int64)

    with quant_execution_mode(ExecutionMode.FP32_QDQ):
        y_fp32 = module(x, shape)
    with quant_execution_mode(ExecutionMode.FP16_QDQ):
        y_fp16 = module(x, shape)

    assert y_fp32.shape == (3, 2)
    assert y_fp32.dtype == torch.float32
    assert y_fp16.shape == (3, 2)
    assert y_fp16.dtype == torch.float16


def test_fp16_qdq_helper_preserves_integer_metadata_tensor():
    metadata = torch.tensor([3, 2], dtype=torch.int64)

    with quant_execution_mode(ExecutionMode.FP16_QDQ):
        output = _quantize_dequantize_if_applicable(metadata, None)

    assert output.dtype == torch.int64
    assert torch.equal(output, metadata)


def _make_int8_qdq():
    return QuantizeDequantize(shape=(), qmin=-128, qmax=127, symmetric=False)


def _make_int8_sym_qdq():
    return QuantizeDequantize(shape=(), qmin=-128, qmax=127, symmetric=True)


def test_fp16_qdq_batchnorm_promotes_affine_params_without_param_quantizer():
    """Repro for MRNN failure: input quantizer casts activations to fp16 but the
    BN affine params remained fp32, raising
    'Input type (HalfTensor) and weight type (FloatTensor) should be the same'.
    """
    bn = QuantizableBatchNorm2d(8, affine=True).eval()
    qbn = QuantizationMixin.from_module(bn)
    qbn.input_quantizers[0] = _make_int8_qdq()
    qbn.output_quantizers[0] = _make_int8_qdq()
    with qbn.compute_encodings():
        qbn(torch.randn(2, 8, 4, 4))

    with quant_execution_mode(ExecutionMode.FP16_QDQ):
        y = qbn(torch.randn(2, 8, 4, 4))

    assert y.dtype == torch.float16
    # Outside the wrapped forward, parameters must remain fp32 (patch_attr restores).
    assert qbn.weight.dtype == torch.float32
    assert qbn.bias.dtype == torch.float32


def test_fp16_qdq_linear_without_weight_quantizer_runs():
    """Activation-only quantization: weight has no param_quantizer; under
    FP16_QDQ the dispatch wrapper must still cast it to fp16 to match input.
    """
    qlinear = QuantizedLinear(4, 8)
    qlinear.input_quantizers[0] = _make_int8_qdq()
    qlinear.output_quantizers[0] = _make_int8_qdq()
    with qlinear.compute_encodings():
        qlinear(torch.randn(2, 4))

    with quant_execution_mode(ExecutionMode.FP16_QDQ):
        y = qlinear(torch.randn(2, 4))

    assert y.dtype == torch.float16
    assert qlinear.weight.dtype == torch.float32  # untouched outside forward


def test_fp16_qdq_linear_with_weight_quantizer_runs():
    """Both activation and weight quantizers present; result must still be fp16."""
    qlinear = QuantizedLinear(4, 8)
    qlinear.input_quantizers[0] = _make_int8_qdq()
    qlinear.output_quantizers[0] = _make_int8_qdq()
    qlinear.param_quantizers["weight"] = _make_int8_sym_qdq()
    with qlinear.compute_encodings():
        qlinear(torch.randn(2, 4))

    with quant_execution_mode(ExecutionMode.FP16_QDQ):
        y = qlinear(torch.randn(2, 4))

    assert y.dtype == torch.float16

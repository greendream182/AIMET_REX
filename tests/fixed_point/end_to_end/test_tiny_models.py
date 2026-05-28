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
from aimet_torch.fixed_point.metrics import assert_int16_vs_fp32_reference  # noqa: E402
from aimet_torch.v2.nn import (  # noqa: E402
    QuantizedConv2d,
    QuantizedFlatten,
    QuantizedGELU,
    QuantizedLinear,
    QuantizedMaxPool2d,
    QuantizedReLU,
    QuantizedSigmoid,
    QuantizedTanh,
)
from aimet_torch.v2.nn.modules.custom import QuantizedAdd, QuantizedMultiply  # noqa: E402
from aimet_torch.v2.quantization.affine import Quantize  # noqa: E402


def _dequantize(value):
    return value.dequantize() if hasattr(value, "dequantize") else value


def _init_linear(m: QuantizedLinear, in_range=2.0, weight_range=0.5, out_range=2.0):
    m.input_quantizers[0] = Quantize((), 8, symmetric=True)
    m.param_quantizers["weight"] = Quantize((m.out_features, 1), 8, symmetric=True)
    m.output_quantizers[0] = Quantize((), 8, symmetric=True)
    m.input_quantizers[0].min = nn.Parameter(torch.tensor(-in_range))
    m.input_quantizers[0].max = nn.Parameter(torch.tensor(in_range))
    m.param_quantizers["weight"].min = nn.Parameter(
        torch.full((m.out_features, 1), -weight_range)
    )
    m.param_quantizers["weight"].max = nn.Parameter(
        torch.full((m.out_features, 1), weight_range)
    )
    m.output_quantizers[0].min = nn.Parameter(torch.tensor(-out_range))
    m.output_quantizers[0].max = nn.Parameter(torch.tensor(out_range))


def _init_conv2d(m: QuantizedConv2d, in_range=2.0, weight_range=0.5, out_range=2.0):
    oc = m.out_channels
    m.input_quantizers[0] = Quantize((), 8, symmetric=True)
    m.param_quantizers["weight"] = Quantize((oc, 1, 1, 1), 8, symmetric=True)
    m.output_quantizers[0] = Quantize((), 8, symmetric=True)
    m.input_quantizers[0].min = nn.Parameter(torch.tensor(-in_range))
    m.input_quantizers[0].max = nn.Parameter(torch.tensor(in_range))
    m.param_quantizers["weight"].min = nn.Parameter(
        torch.full((oc, 1, 1, 1), -weight_range)
    )
    m.param_quantizers["weight"].max = nn.Parameter(
        torch.full((oc, 1, 1, 1), weight_range)
    )
    m.output_quantizers[0].min = nn.Parameter(torch.tensor(-out_range))
    m.output_quantizers[0].max = nn.Parameter(torch.tensor(out_range))


def _init_unary(m: nn.Module, in_range=2.0, out_min=-2.0, out_max=2.0):
    m.input_quantizers[0] = Quantize((), 8, symmetric=True)
    m.output_quantizers[0] = Quantize((), 8, symmetric=True)
    m.input_quantizers[0].min = nn.Parameter(torch.tensor(-in_range))
    m.input_quantizers[0].max = nn.Parameter(torch.tensor(in_range))
    m.output_quantizers[0].min = nn.Parameter(torch.tensor(out_min))
    m.output_quantizers[0].max = nn.Parameter(torch.tensor(out_max))


def _init_binary(m: nn.Module, in_range=2.0, out_min=-2.0, out_max=2.0):
    m.input_quantizers[0] = Quantize((), 8, symmetric=True)
    m.input_quantizers[1] = Quantize((), 8, symmetric=True)
    m.output_quantizers[0] = Quantize((), 8, symmetric=True)
    for q in m.input_quantizers:
        q.min = nn.Parameter(torch.tensor(-in_range))
        q.max = nn.Parameter(torch.tensor(in_range))
    m.output_quantizers[0].min = nn.Parameter(torch.tensor(out_min))
    m.output_quantizers[0].max = nn.Parameter(torch.tensor(out_max))


def test_tiny_linear_relu_linear_preserves_int16_carrier():
    model = nn.Sequential(
        QuantizedLinear(3, 4),
        QuantizedReLU(),
        QuantizedLinear(4, 2),
    )
    _init_linear(model[0], in_range=2.0, weight_range=0.5, out_range=2.0)
    _init_unary(model[1], in_range=2.0, out_min=0.0, out_max=2.0)
    _init_linear(model[2], in_range=2.0, weight_range=0.5, out_range=2.0)
    nn.init.constant_(model[0].weight, 0.1)
    nn.init.constant_(model[0].bias, 0.01)
    nn.init.constant_(model[2].weight, 0.08)
    nn.init.constant_(model[2].bias, -0.02)

    x = torch.tensor([[0.5, -0.25, 0.1]], dtype=torch.float32)

    with quant_execution_mode(ExecutionMode.FP32_QDQ):
        y_ref = _dequantize(model(x))
    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y_int = model(x)

    assert isinstance(y_int, Int16QuantizedTensor)
    assert_int16_vs_fp32_reference(y_int, y_ref)


def test_tiny_sigmoid_pwl_activation_preserves_int16_carrier():
    model = nn.Sequential(
        QuantizedLinear(3, 3),
        QuantizedSigmoid(),
    )
    _init_linear(model[0], in_range=2.0, weight_range=0.5, out_range=4.0)
    _init_unary(model[1], in_range=4.0, out_min=0.0, out_max=1.0)
    nn.init.constant_(model[0].weight, 0.2)
    nn.init.constant_(model[0].bias, 0.0)

    x = torch.tensor([[0.5, -0.25, 0.1]], dtype=torch.float32)

    with quant_execution_mode(ExecutionMode.FP32_QDQ):
        y_ref = _dequantize(model(x))
    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y_int = model(x)

    assert isinstance(y_int, Int16QuantizedTensor)
    assert_int16_vs_fp32_reference(y_int, y_ref)


def test_tiny_conv_pool_flatten_linear_preserves_int16_carrier():
    model = nn.Sequential(
        QuantizedConv2d(1, 2, kernel_size=2),
        QuantizedReLU(),
        QuantizedMaxPool2d(kernel_size=2),
        QuantizedFlatten(),
        QuantizedLinear(2, 2),
    )
    _init_conv2d(model[0], in_range=2.0, weight_range=0.5, out_range=2.0)
    _init_unary(model[1], in_range=2.0, out_min=0.0, out_max=2.0)
    _init_unary(model[2], in_range=2.0, out_min=0.0, out_max=2.0)
    _init_unary(model[3], in_range=2.0, out_min=0.0, out_max=2.0)
    _init_linear(model[4], in_range=2.0, weight_range=0.5, out_range=2.0)
    nn.init.constant_(model[0].weight, 0.2)
    nn.init.constant_(model[0].bias, 0.01)
    nn.init.constant_(model[4].weight, 0.1)
    nn.init.constant_(model[4].bias, 0.0)

    x = torch.tensor(
        [[[[0.5, -0.25, 0.1], [0.0, 0.3, -0.2], [0.1, 0.1, 0.0]]]],
        dtype=torch.float32,
    )

    with quant_execution_mode(ExecutionMode.FP32_QDQ):
        y_ref = _dequantize(model(x))
    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y_int = model(x)

    assert isinstance(y_int, Int16QuantizedTensor)
    assert_int16_vs_fp32_reference(y_int, y_ref)


def test_tiny_gelu_tanh_pwl_activations_preserve_int16_carrier():
    gelu = nn.Sequential(QuantizedLinear(3, 3), QuantizedGELU(), QuantizedLinear(3, 2))
    _init_linear(gelu[0], in_range=2.0, weight_range=0.5, out_range=3.0)
    _init_unary(gelu[1], in_range=3.0, out_min=-1.0, out_max=3.0)
    _init_linear(gelu[2], in_range=3.0, weight_range=0.5, out_range=2.0)
    nn.init.constant_(gelu[0].weight, 0.15)
    nn.init.constant_(gelu[0].bias, 0.01)
    nn.init.constant_(gelu[2].weight, 0.1)
    nn.init.constant_(gelu[2].bias, 0.0)

    tanh = nn.Sequential(QuantizedLinear(3, 3), QuantizedTanh())
    _init_linear(tanh[0], in_range=2.0, weight_range=0.5, out_range=3.0)
    _init_unary(tanh[1], in_range=3.0, out_min=-1.0, out_max=1.0)
    nn.init.constant_(tanh[0].weight, 0.1)
    nn.init.constant_(tanh[0].bias, 0.0)

    x = torch.tensor([[0.5, -0.25, 0.1]], dtype=torch.float32)
    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y_gelu = gelu(x)
        y_tanh = tanh(x)

    assert isinstance(y_gelu, Int16QuantizedTensor)
    assert isinstance(y_tanh, Int16QuantizedTensor)


def test_tiny_add_and_multiply_preserve_int16_carrier():
    left = QuantizedLinear(3, 3)
    right = QuantizedLinear(3, 3)
    add = QuantizedAdd()
    mul = QuantizedMultiply()
    _init_linear(left, in_range=2.0, weight_range=0.5, out_range=2.0)
    _init_linear(right, in_range=2.0, weight_range=0.5, out_range=2.0)
    _init_binary(add, in_range=2.0, out_min=-4.0, out_max=4.0)
    _init_binary(mul, in_range=4.0, out_min=-4.0, out_max=4.0)
    nn.init.constant_(left.weight, 0.12)
    nn.init.constant_(right.weight, -0.08)

    x = torch.tensor([[0.5, -0.25, 0.1]], dtype=torch.float32)
    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y_left = left(x)
        y_right = right(x)
        y_add = add(y_left, y_right)
        y_mul = mul(y_add, y_left)

    assert isinstance(y_add, Int16QuantizedTensor)
    assert isinstance(y_mul, Int16QuantizedTensor)


def test_tiny_conv_relu_supergroup_tail_eval_and_qat_sim_bit_exact():
    """ReLU with ``input_quantizers[0]=None`` (super-group tail) must match eval in qat_sim."""

    model = nn.Sequential(
        QuantizedConv2d(1, 2, kernel_size=2, padding=0),
        QuantizedReLU(),
    )
    _init_conv2d(model[0], in_range=2.0, weight_range=0.5, out_range=2.0)
    _init_unary(model[1], in_range=2.0, out_min=0.0, out_max=2.0)
    model[1].input_quantizers[0] = None
    nn.init.constant_(model[0].weight, 0.2)
    nn.init.constant_(model[0].bias, 0.01)

    x = torch.tensor(
        [[[[0.5, -0.25], [0.1, 0.0]]]],
        dtype=torch.float32,
    )

    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y_eval = model(x)
    with quant_execution_mode(ExecutionMode.INT16_FIXED_QAT_SIM):
        y_qat = model(x)

    assert isinstance(y_eval, Int16QuantizedTensor)
    assert isinstance(y_qat, torch.Tensor)
    assert torch.equal(y_eval.to_float(torch.float32), y_qat.detach())
    y_qat_int = Int16QuantizedTensor.from_affine_encoding(
        y_qat.detach(),
        model[1].output_quantizers[0].get_encodings(),
    )
    assert torch.equal(y_eval.int_repr, y_qat_int.int_repr)


def test_tiny_linear_eval_and_qat_sim_forward_bit_exact():
    """Spec 11: INT16_FIXED_EVAL and INT16_FIXED_QAT_SIM must match on forward ints."""

    model = QuantizedLinear(3, 2)
    _init_linear(model, in_range=2.0, weight_range=0.5, out_range=2.0)
    nn.init.constant_(model.weight, 0.1)
    nn.init.constant_(model.bias, 0.02)

    x = torch.tensor([[0.5, -0.25, 0.1]], dtype=torch.float32)
    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y_eval = model(x)
    with quant_execution_mode(ExecutionMode.INT16_FIXED_QAT_SIM):
        y_qat = model(x)

    assert isinstance(y_eval, Int16QuantizedTensor)
    assert isinstance(y_qat, torch.Tensor)
    # QAT-SIM forward must preserve the same dequantized grid as eval (spec 11).
    assert torch.equal(y_eval.to_float(torch.float32), y_qat.detach())
    y_qat_int = Int16QuantizedTensor.from_affine_encoding(
        y_qat.detach(),
        model.output_quantizers[0].get_encodings(),
    )
    assert torch.equal(y_eval.int_repr, y_qat_int.int_repr)


def test_tiny_linear_int16_qat_sim_backward_uses_surrogate_gradient():
    model = QuantizedLinear(3, 2)
    _init_linear(model, in_range=2.0, weight_range=0.5, out_range=2.0)
    nn.init.constant_(model.weight, 0.1)
    nn.init.constant_(model.bias, 0.0)

    x = torch.tensor([[0.5, -0.25, 0.1]], dtype=torch.float32, requires_grad=True)
    with quant_execution_mode(ExecutionMode.INT16_FIXED_QAT_SIM):
        y = model(x)
        loss = y.sum()
    loss.backward()

    assert isinstance(y, torch.Tensor)
    assert x.grad is not None
    assert model.weight.grad is not None

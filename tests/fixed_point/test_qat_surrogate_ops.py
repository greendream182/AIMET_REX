# -*- mode: python -*-
"""QAT surrogate coverage for MRNN frontend ops."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from aimet_torch._base.nn.modules import custom
from aimet_torch.fixed_point.qat.carrier import (
    clear_int16_carriers,
    consume_int16_carrier,
    publish_int16_carrier,
)
from aimet_torch.fixed_point.qat_train import release_qat_grads, select_qat_trainable_parameters
from aimet_torch.fixed_point.tensor import Int16QuantizedTensor
from aimet_torch.v2.quantization.affine.fixed_point.adapter import (
    _qat_surrogate_float,
    _run_qat_surrogate,
)


def test_qat_surrogate_matmul_sqrt_sign():
    a = torch.randn(2, 3)
    b = torch.randn(3, 4)
    x = torch.tensor([0.25, 1.0, 4.0])

    matmul_y = _qat_surrogate_float(custom.MatMul, custom.MatMul, [a, b], {}, {})
    sqrt_y = _qat_surrogate_float(custom.Sqrt, custom.Sqrt, [x], {}, {})
    sign_y = _qat_surrogate_float(
        custom.ElementwiseUnarySign,
        custom.ElementwiseUnarySign,
        [x],
        {},
        {},
    )

    assert matmul_y is not None
    assert sqrt_y is not None
    assert sign_y is not None
    torch.testing.assert_close(matmul_y, torch.matmul(a, b))
    torch.testing.assert_close(sqrt_y, torch.sqrt(x))
    torch.testing.assert_close(sign_y, torch.sign(x))


def test_select_qat_trainable_excludes_quantizer():
    class _Wrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(4, 2)
            self.input_quantizers_0_scale = torch.nn.Parameter(torch.tensor(1.0))

    wrap = _Wrap()
    selected_ids = {id(p) for p in select_qat_trainable_parameters(wrap, scope="weights")}
    names = {n for n, p in wrap.named_parameters() if id(p) in selected_ids}
    assert names == {"fc.weight", "fc.bias"}


def test_release_qat_grads_clears_unconsumed_int16_carriers():
    x = torch.tensor([1.0])
    carrier = Int16QuantizedTensor(
        int_repr=torch.tensor([1], dtype=torch.int32),
        scale=torch.tensor(0.5),
        zero_point=torch.tensor(0, dtype=torch.int32),
    )

    clear_int16_carriers()
    publish_int16_carrier(x, carrier)
    release_qat_grads()

    assert consume_int16_carrier(x) is None


def test_qat_surrogate_checkpoint_preserves_value_and_gradient():
    class _LinearWrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([[0.2, -0.1], [0.05, 0.3]]))
            self.bias = torch.nn.Parameter(torch.tensor([0.01, -0.02]))

    wrap = _LinearWrap()
    x = torch.tensor([[0.5, -0.25]], requires_grad=True)
    direct = _qat_surrogate_float(wrap, torch.nn.Linear, [x], {}, {})
    via_ckpt = _run_qat_surrogate(wrap, torch.nn.Linear, [x], {}, {})
    assert direct is not None and via_ckpt is not None
    torch.testing.assert_close(via_ckpt, direct)

    loss = via_ckpt.sum()
    loss.backward()
    assert x.grad is not None
    assert wrap.weight.grad is not None

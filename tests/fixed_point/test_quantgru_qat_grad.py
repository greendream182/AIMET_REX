# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
"""QAT gradient boundary protocol tests for QuantGRU black-box adapter (plan §2.4)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxscript")

from aimet_torch.fixed_point import ExecutionMode, quant_execution_mode  # noqa: E402
from aimet_torch.fixed_point.gradient_helpers import (  # noqa: E402
    stop_grad_dequantize,
    wrap_int_tensor_with_meta,
)
from aimet_torch.fixed_point.quantgru_adapter import (  # noqa: E402
    aimet_configure,
    dispatch_quantgru_blackbox,
)
from aimet_torch.fixed_point.tensor import Int16QuantizedTensor  # noqa: E402


def _has_quant_gru() -> bool:
    try:
        import quant_gru  # noqa: F401

        return True
    except ImportError:
        return False


requires_quant_gru_cuda = pytest.mark.skipif(
    not (_has_quant_gru() and torch.cuda.is_available()),
    reason="quant_gru + CUDA required",
)


def test_stop_grad_dequantize_participates_in_autograd():
    """Boundary dequant fp proxy must participate in autograd (§2.4 identity path)."""
    meta = {"scale": 0.01, "zp": 0, "bitwidth": 16, "is_symmetric": True}
    ints = torch.tensor([[3, -2]], dtype=torch.int32)
    carrier = wrap_int_tensor_with_meta(ints, meta)
    fp = stop_grad_dequantize(carrier)
    weight = torch.ones_like(fp, requires_grad=True)
    loss = (fp + weight).sum()
    loss.backward()
    assert weight.grad is not None
    assert torch.allclose(weight.grad, torch.ones_like(weight))


def test_stop_grad_dequantize_upstream_identity_ste():
    """Upstream fp leaf receives grad via identity STE (§2.4)."""
    x = torch.randn(2, 3, requires_grad=True)
    scale = torch.tensor(0.01)
    zp = torch.tensor(0, dtype=torch.int32)
    carrier = Int16QuantizedTensor.from_float(x.detach(), scale=scale, zero_point=zp)
    fp = stop_grad_dequantize(carrier)
    y = fp + (x - x.detach())
    y.sum().backward()
    assert x.grad is not None
    assert torch.allclose(x.grad, torch.ones_like(x))


def test_wrap_int_tensor_downstream_grad_to_fp_leaf():
    meta = {"scale": 0.02, "zp": 0, "bitwidth": 16, "is_symmetric": True}
    fp = torch.tensor([[1.0, 2.0]], requires_grad=True)
    ints = torch.round(fp.detach() / meta["scale"]).to(torch.int32)
    carrier = wrap_int_tensor_with_meta(ints, meta)
    out_fp = carrier.to_float() + (fp - fp.detach())
    out_fp.sum().backward()
    assert fp.grad is not None


def _calibrate_quantgru(gru, x: torch.Tensor) -> None:
    with gru._aimet_unlock_ctx():
        aimet_configure(gru, "calibrating")
        with torch.no_grad():
            gru(x)
        gru.finalize_calibration(verbose=False)


@requires_quant_gru_cuda
def test_qat_sim_dispatch_returns_float_with_grad_path():
    from aimet_torch.v2.nn.modules.custom import QuantizedQuantGRU

    gru = QuantizedQuantGRU(8, 8, batch_first=True).cuda().train()
    x = torch.randn(2, 4, 8, device="cuda", requires_grad=True)
    _calibrate_quantgru(gru, x.detach())

    with quant_execution_mode(ExecutionMode.INT16_FIXED_QAT_SIM):
        out_fp, hn_fp = dispatch_quantgru_blackbox(gru, x)

    assert out_fp.dtype == torch.float32
    assert hn_fp.dtype == torch.float32
    assert out_fp.shape == x.shape
    assert out_fp.requires_grad


@requires_quant_gru_cuda
def test_qat_grad_matches_native_forward_quant():
    """AIMET QAT_SIM dispatch must match native ``forward()`` + ``backward_quant``."""
    from aimet_torch.v2.nn.modules.custom import QuantizedQuantGRU

    torch.manual_seed(0)
    gru = QuantizedQuantGRU(8, 8, batch_first=True).cuda().train()
    x = torch.randn(2, 4, 8, device="cuda")
    _calibrate_quantgru(gru, x)

    x_ref = x.detach().clone().requires_grad_(True)
    from aimet_torch.v2.nn.modules.custom import _OptionalQuantGRU

    with gru._aimet_unlock_ctx():
        aimet_configure(gru, "int16_fixed_qat_sim")
    out_ref, _ = _OptionalQuantGRU.forward(gru, x_ref)
    out_ref.sum().backward()
    ref_x_grad = x_ref.grad.detach().clone()
    ref_w_grad = gru.weight_ih_l0.grad.detach().clone()

    gru.zero_grad(set_to_none=True)
    x_aimet = x.detach().clone().requires_grad_(True)
    with quant_execution_mode(ExecutionMode.INT16_FIXED_QAT_SIM):
        out_aimet, _ = dispatch_quantgru_blackbox(gru, x_aimet)
    out_aimet.sum().backward()

    assert torch.allclose(x_aimet.grad, ref_x_grad, atol=1e-5, rtol=0)
    assert torch.allclose(gru.weight_ih_l0.grad, ref_w_grad, atol=1e-5, rtol=0)


@requires_quant_gru_cuda
def test_qat_grad_with_int16_upstream_carrier():
    """INT16 upstream carrier: identity dequant at GRU boundary; weight grads match native.

    The A path feeds the GRU an ``Int16QuantizedTensor`` (upstream INT16
    carrier); inside ``dispatch_quantgru_blackbox`` it is dequantized via
    ``stop_grad_dequantize`` and forwarded as float. The B path takes the
    *same* dequantized carrier as float input (rather than the raw ``x_fp``)
    so that both paths see bit-identical float activations at the GRU
    boundary. Without this, quant-gru-pytorch v1.0.5's affine ``scale_x_``
    leaks ~0.5 LSB of round-trip error into the input, which the GRU
    backward path amplifies into ~1e-4 grad drift — that drift is *expected*
    (it is the carrier's own quantization error, not a dispatch bug).
    """
    from aimet_torch.v2.nn.modules.custom import QuantizedQuantGRU, _OptionalQuantGRU

    torch.manual_seed(1)
    gru = QuantizedQuantGRU(8, 8, batch_first=True).cuda().train()
    x_fp = torch.randn(2, 4, 8, device="cuda")
    _calibrate_quantgru(gru, x_fp)
    meta = gru.get_io_quant_meta()

    scale_t = torch.tensor(meta["input"]["scale"], device="cuda")
    zp_t = torch.tensor(meta["input"]["zp"], dtype=torch.int32, device="cuda")
    x_q = Int16QuantizedTensor.from_float(x_fp, scale=scale_t, zero_point=zp_t)
    # Use the same dequantize path that ``dispatch_quantgru_blackbox`` takes
    # internally (``stop_grad_dequantize`` → ``Int16QuantizedTensor.to_float``)
    # so the B baseline sees bit-identical float activations at the GRU input.
    from aimet_torch.fixed_point.gradient_helpers import stop_grad_dequantize
    x_q_dq = stop_grad_dequantize(x_q).detach().clone()

    with quant_execution_mode(ExecutionMode.INT16_FIXED_QAT_SIM):
        out_aimet, _ = dispatch_quantgru_blackbox(gru, x_q)
    out_aimet.sum().backward()
    aimet_w_grad = gru.weight_ih_l0.grad.detach().clone()

    gru.zero_grad(set_to_none=True)
    x_ref = x_q_dq.requires_grad_(True)
    with gru._aimet_unlock_ctx():
        aimet_configure(gru, "int16_fixed_qat_sim")
    out_ref, _ = _OptionalQuantGRU.forward(gru, x_ref)
    out_ref.sum().backward()

    assert aimet_w_grad is not None
    assert torch.allclose(aimet_w_grad, gru.weight_ih_l0.grad, atol=1e-5, rtol=0)


@requires_quant_gru_cuda
def test_qat_grad_bidirectional_matches_native():
    """BiGRU: QAT_SIM dispatch grads match native forward + backward_quant."""
    from aimet_torch.v2.nn.modules.custom import QuantizedQuantGRU, _OptionalQuantGRU

    torch.manual_seed(2)
    gru = QuantizedQuantGRU(8, 8, batch_first=True, bidirectional=True).cuda().train()
    x = torch.randn(2, 5, 8, device="cuda")
    _calibrate_quantgru(gru, x)

    x_ref = x.detach().clone().requires_grad_(True)
    with gru._aimet_unlock_ctx():
        aimet_configure(gru, "int16_fixed_qat_sim")
    out_ref, _ = _OptionalQuantGRU.forward(gru, x_ref)
    out_ref.sum().backward()
    ref_x = x_ref.grad.detach().clone()
    ref_w_f = gru.weight_ih_l0.grad.detach().clone()
    ref_w_r = gru.weight_ih_l0_reverse.grad.detach().clone()

    gru.zero_grad(set_to_none=True)
    x_aimet = x.detach().clone().requires_grad_(True)
    with quant_execution_mode(ExecutionMode.INT16_FIXED_QAT_SIM):
        out_aimet, _ = dispatch_quantgru_blackbox(gru, x_aimet)
    out_aimet.sum().backward()

    assert out_aimet.shape[-1] == 16  # 2 * hidden
    assert torch.allclose(x_aimet.grad, ref_x, atol=1e-5, rtol=0)
    assert torch.allclose(gru.weight_ih_l0.grad, ref_w_f, atol=1e-5, rtol=0)
    assert torch.allclose(gru.weight_ih_l0_reverse.grad, ref_w_r, atol=1e-5, rtol=0)

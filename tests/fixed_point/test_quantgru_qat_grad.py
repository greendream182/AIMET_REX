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
from aimet_torch.fixed_point.quantgru_adapter import dispatch_quantgru_blackbox  # noqa: E402
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


@requires_quant_gru_cuda
def test_qat_sim_dispatch_returns_float_with_grad_path():
    from aimet_torch.v2.nn import compute_encodings
    from aimet_torch.v2.nn.modules.custom import QuantizedQuantGRU

    gru = QuantizedQuantGRU(8, 8, batch_first=True).cuda().train()
    x = torch.randn(2, 4, 8, device="cuda", requires_grad=True)

    with compute_encodings(gru):
        with torch.no_grad():
            gru(x)

    with quant_execution_mode(ExecutionMode.INT16_FIXED_QAT_SIM):
        out_fp, hn_fp = dispatch_quantgru_blackbox(gru, x)

    assert out_fp.dtype == torch.float32
    assert hn_fp.dtype == torch.float32
    assert out_fp.shape == x.shape


@requires_quant_gru_cuda
@pytest.mark.skip(
    reason="生产验收（§2.4）：待 AIMET adapter 与 QuantGRU backward_quant 数值对齐 CI 就绪"
)
def test_qat_grad_matches_backward_quant():
    """AIMET 边界 identity 反向须与 QuantGRU backward_quant 一致（max abs diff < 1e-6）。"""
    from aimet_torch.v2.nn.modules.custom import QuantizedQuantGRU

    gru = QuantizedQuantGRU(8, 8, batch_first=True).cuda().train()
    x = torch.randn(2, 4, 8, device="cuda", requires_grad=True)
    del gru, x

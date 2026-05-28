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
"""CPU smoke 测试：INT16_FIXED_EVAL dispatch 在 Conv2d / Linear / Mean / ReLU 链上能完整 forward。

本测试**不依赖** quant_gru 或 CUDA，专门固化 ``dispatch_int16_fixed`` 的 fp32
round-trip fallback 在 CPU 上不退化（PyTorch CPU ``torch.matmul`` 支持 int 张量
的范围有限；任何破坏 fallback 路径的改动都会让本测试报错）。

与 ``tests/fixed_point/test_mrnn_int16_e2e.py`` 的 backbone 测试互补：
  - 那边需要 quant_gru + CUDA 全栈环境；
  - 这边只用 PyTorch + AIMET v2，CI 任何机器都跑得动。
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxscript")
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import aimet_torch.fixed_point.kernels  # noqa: F401, E402  # 触发 kernel 注册
from aimet_torch.fixed_point import (  # noqa: E402
    ExecutionMode,
    Int16QuantizedTensor,
    ensure_output_quantizers_for_int16_eval,
    quant_execution_mode,
)
from aimet_torch.fixed_point.tensor import FixedPointSimTensor  # noqa: E402
from aimet_torch.v2.quantization.tensor import QuantizedTensorBase  # noqa: E402


def _build_calibrated_sim(model: nn.Module, dummy: torch.Tensor):
    from aimet_torch import model_preparer
    from aimet_torch.v2.quantsim import quantsim

    prepared = model_preparer.prepare_model(model)
    sim = quantsim.QuantizationSimModel(
        prepared,
        dummy_input=dummy,
        quant_scheme="tf",
        default_output_bw=8,
        default_param_bw=8,
    )
    ensure_output_quantizers_for_int16_eval(sim)
    sim.model.eval()

    def _calib(m: nn.Module) -> None:
        with torch.no_grad():
            for _ in range(4):
                m(torch.randn_like(dummy))

    sim.compute_encodings(_calib)
    return sim


class _LinearOnly(nn.Module):
    def __init__(self, in_dim: int = 8, out_dim: int = 4):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class _ConvReluMeanLinear(nn.Module):
    """MRNN backbone 的纯 CPU 替身（去掉 QuantGRU 子段）。"""

    def __init__(self, hidden: int = 8, num_classes: int = 4):
        super().__init__()
        self.conv_in = nn.Conv2d(1, hidden, kernel_size=3, stride=(1, 2), padding=1)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv_in(x))
        x = torch.mean(x, dim=(2, 3))
        return self.fc(x)


def _shape_of(y) -> torch.Size:
    """统一取 forward 输出的张量形状，兼容 plain Tensor / FixedPointSimTensor / QuantizedTensorBase。"""
    if isinstance(y, FixedPointSimTensor):
        return y.int_repr.shape
    if isinstance(y, QuantizedTensorBase):
        return y.shape
    return y.shape


def _to_dense_outside_int16_ctx(y) -> torch.Tensor:
    """在 INT16 上下文之外把 carrier 反量化成 fp32（避免 to_float 在 INT16 mode 内的 warning）。"""
    if isinstance(y, FixedPointSimTensor):
        return y.to_float(torch.float32)
    if isinstance(y, QuantizedTensorBase):
        return y.dequantize()
    return y


# ---------------------------------------------------------------------------
# Smoke：dispatch_int16_fixed 覆盖关键算子（CPU 上）
# ---------------------------------------------------------------------------


def test_int16_eval_linear_cpu_smoke():
    torch.manual_seed(0)
    model = _LinearOnly(in_dim=8, out_dim=4).eval()
    dummy = torch.randn(2, 8)
    sim = _build_calibrated_sim(model, dummy)

    with torch.no_grad(), quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y = sim.model(dummy)

    assert _shape_of(y) == (2, 4)
    assert isinstance(y, (FixedPointSimTensor, QuantizedTensorBase, torch.Tensor))


def test_int16_eval_conv_relu_mean_linear_cpu_smoke():
    torch.manual_seed(1)
    model = _ConvReluMeanLinear(hidden=8, num_classes=4).eval()
    dummy = torch.randn(2, 1, 8, 6)
    sim = _build_calibrated_sim(model, dummy)

    with torch.no_grad(), quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y = sim.model(dummy)

    assert _shape_of(y) == (2, 4)
    assert isinstance(y, (FixedPointSimTensor, QuantizedTensorBase, torch.Tensor))


def test_int16_eval_matches_fp32_qdq_within_tolerance():
    """INT16 fallback 输出应当与 FP32_QDQ 输出在量化误差量级内一致。"""
    torch.manual_seed(2)
    model = _ConvReluMeanLinear(hidden=8, num_classes=4).eval()
    dummy = torch.randn(2, 1, 8, 6)
    sim = _build_calibrated_sim(model, dummy)

    with torch.no_grad():
        with quant_execution_mode(ExecutionMode.FP32_QDQ):
            y_fp32_qdq = sim.model(dummy)
        with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
            y_carrier = sim.model(dummy)

    # 退出 INT16 上下文后再做 dequant，避免 to_float 的 INT16 warning。
    y_int16 = _to_dense_outside_int16_ctx(y_carrier)
    diff = (y_int16 - y_fp32_qdq).abs().max().item()
    # FP32_QDQ vs INT16_FIXED_EVAL 在 8-bit param + 16-bit activation 下保守上限 0.5。
    assert diff <= 0.5, f"INT16 vs FP32_QDQ max abs diff = {diff} 异常超出预期"


def test_int16_eval_no_exception_on_backbone_substitute():
    """CPU 替身 backbone 在 INT16 dispatch 下不应抛任何异常（包括 KernelNotFoundError）。"""
    torch.manual_seed(3)
    model = _ConvReluMeanLinear(hidden=8, num_classes=4).eval()
    dummy = torch.randn(2, 1, 8, 6)
    sim = _build_calibrated_sim(model, dummy)

    with torch.no_grad(), quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        sim.model(dummy)  # 抛任何异常都让 pytest 失败，从而暴露 dispatch 退化


# ---------------------------------------------------------------------------
# FIXED_SCALE_QDQ：boundary scale 走 (m_int16, rshift) 网格；
# v2 sim 内置 quantizer 在该 mode 下自动切到 fixed-scale Q/DQ
# (aimet_torch/v2/quantization/affine/backends/torch_builtins.py:230)。
# 验证 backbone 替身（不含 GRU）在该 mode 下能完整 forward。
# ---------------------------------------------------------------------------


def test_fixed_scale_qdq_conv_relu_mean_linear_cpu_smoke():
    torch.manual_seed(4)
    model = _ConvReluMeanLinear(hidden=8, num_classes=4).eval()
    dummy = torch.randn(2, 1, 8, 6)
    sim = _build_calibrated_sim(model, dummy)

    with torch.no_grad(), quant_execution_mode(ExecutionMode.FIXED_SCALE_QDQ):
        y = sim.model(dummy)

    assert _shape_of(y) == (2, 4)


def test_fixed_scale_qdq_matches_fp32_qdq_within_tolerance():
    """FIXED_SCALE_QDQ 输出在 POT scale 校准下应与 FP32_QDQ 数值近似一致。"""
    torch.manual_seed(5)
    model = _ConvReluMeanLinear(hidden=8, num_classes=4).eval()
    dummy = torch.randn(2, 1, 8, 6)
    sim = _build_calibrated_sim(model, dummy)

    with torch.no_grad():
        with quant_execution_mode(ExecutionMode.FP32_QDQ):
            y_fp32 = sim.model(dummy)
        with quant_execution_mode(ExecutionMode.FIXED_SCALE_QDQ):
            y_fs = sim.model(dummy)

    diff = (y_fs - y_fp32).abs().max().item()
    # (m_int16, rshift) 网格逼近原始 float scale；典型差异 ≤ 量化步长量级
    assert diff <= 0.5, (
        f"FIXED_SCALE_QDQ vs FP32_QDQ max abs diff = {diff} 超出 (m_int16, rshift) 近似预期"
    )

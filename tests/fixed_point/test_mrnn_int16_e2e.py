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
"""QuantGRU + MRNN 集成测试（生产标准对齐 QuantGRU_INT16接入计划 §4.2）。

当前可验收项（Phase 2–4）：
  - sim 包装 / 校准 hook / missing-oq 扫描跳过 QuantGRU
  - ``diagnose_int16_readiness`` 校准后无 uncalibrated_quantgru
  - MRNN 主干分段：FP32_QDQ 全 forward + GRU 边界 INT16 dispatch

生产阻塞项（quant-gru 侧，契约 1.4）：
  - MRNN 前端算子（STFT / BandConverter 等）无 INT16 kernel（R3）
  - QAT 反向梯度匹配 ``backward_quant``（R1）
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxscript")
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import aimet_torch.fixed_point.kernels  # noqa: F401, E402
from aimet_torch.fixed_point import (  # noqa: E402
    ExecutionMode,
    Int16QuantizedTensor,
    diagnose_int16_readiness,
    ensure_output_quantizers_for_int16_eval,
    iter_missing_output_quantizers,
    quant_execution_mode,
)


def _has_quant_gru() -> bool:
    try:
        import quant_gru  # noqa: F401

        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not (_has_quant_gru() and torch.cuda.is_available()),
    reason="QuantGRU integration requires quant_gru + CUDA (use quant-gru-cuda128 container)",
)


class MiniGrnSimModel(nn.Module):
    """仅 QuantGRU，用于最小黑盒验收。"""

    def __init__(self, hidden: int = 16):
        super().__init__()
        from quant_gru import QuantGRU

        self.gru = QuantGRU(
            input_size=hidden,
            hidden_size=hidden,
            batch_first=True,
            num_layers=1,
            bidirectional=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return out[:, -1, :]


class Rnn2DBlock(nn.Module):
    """对齐 quick_start.RNN2D 结构（去掉 BN/CLN），保留 QuantGRU + Conv1x1。"""

    def __init__(self, hidden: int):
        super().__init__()
        from quant_gru import QuantGRU

        self.hidden = hidden
        self.gru = QuantGRU(
            input_size=hidden,
            hidden_size=hidden,
            batch_first=True,
            num_layers=1,
            bidirectional=False,
        )
        self.conv_t = nn.Conv2d(hidden, hidden, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, f = x.shape
        seq = x.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        seq, _ = self.gru(seq)
        seq = seq.view(b, f, t, self.hidden).permute(0, 3, 2, 1).contiguous()
        return self.conv_t(seq)


class MrnnBackboneSegment(nn.Module):
    """MRNN 分段 INT16 主干：conv_in → RNN2D → global mean → fc（无 STFT 前端）。"""

    def __init__(self, hidden: int = 16, num_classes: int = 4):
        super().__init__()
        self.conv_in = nn.Conv2d(1, hidden, kernel_size=3, stride=(1, 2), padding=1)
        self.block = Rnn2DBlock(hidden)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv_in(x))
        x = self.block(x)
        x = torch.mean(x, dim=(2, 3))
        return self.fc(x)


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
    sim.model.to(dummy.device).eval()

    def _calib_random(m):
        with torch.no_grad():
            for _ in range(4):
                m(torch.randn_like(dummy))

    sim.compute_encodings(_calib_random)
    return sim


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda")


@pytest.fixture(scope="module")
def gru_only_sim(device):
    hidden = 16
    model = MiniGrnSimModel(hidden=hidden).to(device).eval()
    dummy = torch.randn(2, 8, hidden, device=device)
    sim = _build_calibrated_sim(model, dummy)
    return {"sim": sim, "dummy": dummy, "hidden": hidden}


@pytest.fixture(scope="module")
def backbone_sim(device):
    hidden = 16
    model = MrnnBackboneSegment(hidden=hidden, num_classes=4).to(device).eval()
    dummy = torch.randn(2, 1, 8, 6, device=device)
    sim = _build_calibrated_sim(model, dummy)
    return {"sim": sim, "dummy": dummy, "hidden": hidden}


def test_sim_wraps_quantized_quantgru(gru_only_sim):
    from aimet_torch.v2.nn.modules.custom import QuantizedQuantGRU

    quant_grus = [
        m for m in gru_only_sim["sim"].model.modules() if isinstance(m, QuantizedQuantGRU)
    ]
    assert len(quant_grus) == 1


def test_quantgru_skipped_by_missing_oq_scan(gru_only_sim):
    missing = list(iter_missing_output_quantizers(gru_only_sim["sim"]))
    assert all(class_name != "QuantizedQuantGRU" for _, class_name in missing)


def test_int16_fixed_eval_forward_on_quantgru(gru_only_sim):
    from aimet_torch.v2.nn.modules.custom import QuantizedQuantGRU

    sim = gru_only_sim["sim"]
    x = gru_only_sim["dummy"]
    gru = next(m for m in sim.model.modules() if isinstance(m, QuantizedQuantGRU))
    assert gru.is_calibrated()

    with torch.no_grad(), quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        out_q, hn_q = gru(x)

    assert isinstance(out_q, Int16QuantizedTensor)
    assert isinstance(hn_q, Int16QuantizedTensor)


def test_int16_fixed_eval_no_kernel_not_found(gru_only_sim):
    from aimet_torch.v2.nn.modules.custom import QuantizedQuantGRU

    gru = next(
        m for m in gru_only_sim["sim"].model.modules() if isinstance(m, QuantizedQuantGRU)
    )
    with torch.no_grad(), quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        gru(gru_only_sim["dummy"])


def test_backbone_has_quantized_quantgru(backbone_sim):
    from aimet_torch.v2.nn.modules.custom import QuantizedQuantGRU

    quant_grus = [
        m for m in backbone_sim["sim"].model.modules() if isinstance(m, QuantizedQuantGRU)
    ]
    assert len(quant_grus) == 1


def test_backbone_quantgru_int16_submodule(backbone_sim):
    from aimet_torch.v2.nn.modules.custom import QuantizedQuantGRU

    sim = backbone_sim["sim"]
    gru = next(m for m in sim.model.modules() if isinstance(m, QuantizedQuantGRU))
    assert gru.is_calibrated()

    # RNN2D 内 GRU 输入：(B*F, T, C)；用与 block 一致的 shape 构造
    b, t, c, f = 2, 8, 16, 3
    gru_in = torch.randn(b * f, t, c, device=backbone_sim["dummy"].device)

    with torch.no_grad(), quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        out_q, hn_q = gru(gru_in)

    assert isinstance(out_q, Int16QuantizedTensor)
    assert isinstance(hn_q, Int16QuantizedTensor)


def test_backbone_fp32_qdq_forward(backbone_sim):
    """分段 INT16 前置：校准后主干在 FP32_QDQ 下可完整 forward。"""
    sim = backbone_sim["sim"]
    x = backbone_sim["dummy"]

    with torch.no_grad(), quant_execution_mode(ExecutionMode.FP32_QDQ):
        y = sim.model(x)

    assert y.shape == (2, 4)
    assert y.dtype == torch.float32


def test_backbone_staged_int16_at_gru_boundary(backbone_sim):
    """分段 INT16：FP32_QDQ 跑到 GRU 边界，再对 QuantGRU 单独 INT16 dispatch。"""
    from aimet_torch.v2.nn.modules.custom import QuantizedQuantGRU

    sim = backbone_sim["sim"]
    x = backbone_sim["dummy"]
    gru = next(m for m in sim.model.modules() if isinstance(m, QuantizedQuantGRU))
    captured: dict[str, torch.Tensor] = {}

    def _hook(_mod, inp, _out):
        captured["gru_in"] = inp[0].detach()

    handle = gru.register_forward_hook(_hook)
    try:
        with torch.no_grad(), quant_execution_mode(ExecutionMode.FP32_QDQ):
            sim.model(x)
    finally:
        handle.remove()

    assert "gru_in" in captured
    gru_in = captured["gru_in"]
    assert gru_in.ndim == 3

    with torch.no_grad(), quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        out_q, hn_q = gru(gru_in)

    assert isinstance(out_q, Int16QuantizedTensor)
    assert isinstance(hn_q, Int16QuantizedTensor)
    assert out_q.int_repr.shape == gru_in.shape


def test_backbone_diagnose_int16_ready(backbone_sim):
    """生产 readiness：校准后 QuantGRU 不应出现在 uncalibrated 列表。"""
    report = diagnose_int16_readiness(backbone_sim["sim"])
    assert report["uncalibrated_quantgru"] == []


def test_backbone_segment_int16_e2e_forward(backbone_sim):
    """全图 INT16 e2e（含 QuantGRU bit-exact 边界 + Conv/Linear 浮点 fallback 路径）。"""
    sim = backbone_sim["sim"]
    x = backbone_sim["dummy"]

    with torch.no_grad(), quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y = sim.model(x)

    assert y.shape == (2, 4)

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
"""MRNN 整图 3 档 QDQ 主路径 smoke：fp32_qdq / fp16_qdq / fixed_scale_qdq.

考核口径（设计 §10.1 / 用户验收）：
  - 校准始终在 ``fp32_qdq``（设计 §4.2 / §4.4）；
  - 切 mode 后 forward 不挂，logits 与 ``fp32_qdq`` 的 cosine ≥ 0.99（占位 smoke，
    实际 < 3 pp Top1 阈值需 SpeechCommands 数据集，由 ``examples/quick_start.py``
    与 ``examples/quick_start_int16_metric.py`` 端到端跑出）；
  - QAT (fp32_qdq) 训几步后切 mode 评估同样不挂。

依赖：``quant_gru + CUDA``，与 ``test_mrnn_int16_*`` 系列一致。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxscript")

_EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

import aimet_torch.fixed_point.kernels  # noqa: F401, E402
from aimet_torch.fixed_point import (  # noqa: E402
    ExecutionMode,
    ensure_output_quantizers_for_int16_eval,
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
    reason="MRNN 3-mode smoke requires quant_gru + CUDA",
)


def _build_mrnn_sim(dummy: torch.Tensor):
    from aimet_torch import model_preparer
    from aimet_torch.utils_rx import apply_mixed_precision_bitwidth
    from aimet_torch.v2 import nn as v2nn
    from aimet_torch.v2 import quantsim

    from quick_start import (  # noqa: E402
        BITWIDTH_CONFIG_FILE,
        CONFIG_FILE,
        DEFAULT_BW,
        MRNN,
        NUM_CLASSES,
        PERCENTILE_VALUE,
        QUANT_SCHEME,
    )

    model = MRNN(output_dim=NUM_CLASSES).to(dummy.device).eval()
    prepared = model_preparer.prepare_model(model)
    sim = quantsim.QuantizationSimModel(
        prepared,
        dummy_input=dummy,
        quant_scheme=QUANT_SCHEME,
        config_file=str(CONFIG_FILE),
        default_output_bw=DEFAULT_BW,
        default_param_bw=DEFAULT_BW,
    )
    sim.set_percentile_value(PERCENTILE_VALUE)
    apply_mixed_precision_bitwidth(
        sim.model, config_file=str(BITWIDTH_CONFIG_FILE), verbose=False,
    )
    ensure_output_quantizers_for_int16_eval(sim)
    sim.model.to(dummy.device).eval()

    with torch.no_grad(), v2nn.compute_encodings(sim.model):
        for _ in range(2):
            sim.model(torch.randn_like(dummy))
    return sim


@pytest.fixture(scope="module")
def mrnn_three_mode_bundle():
    device = torch.device("cuda")
    dummy = torch.randn(2, 16000, 1, device=device)
    sim = _build_mrnn_sim(dummy)
    return {"sim": sim, "dummy": dummy, "device": device}


_THREE_MODES = (
    ExecutionMode.FP32_QDQ,
    ExecutionMode.FP16_QDQ,
    ExecutionMode.FIXED_SCALE_QDQ,
)


def _to_dense(y) -> torch.Tensor:
    if hasattr(y, "to_float"):
        return y.to_float()
    return y


def test_mrnn_three_modes_forward_smoke(mrnn_three_mode_bundle):
    """3 档主路径 forward 不挂，输出形状一致。"""

    sim = mrnn_three_mode_bundle["sim"]
    x = mrnn_three_mode_bundle["dummy"]

    sim.model.eval()
    outs: dict[ExecutionMode, torch.Tensor] = {}
    with torch.no_grad():
        for mode in _THREE_MODES:
            with quant_execution_mode(mode):
                y = _to_dense(sim.model(x))
            assert y.shape[0] == x.shape[0], f"{mode.value} batch dim mismatch"
            assert torch.isfinite(y).all(), f"{mode.value} produced non-finite logits"
            outs[mode] = y.float().detach()

    base_shape = outs[ExecutionMode.FP32_QDQ].shape
    for mode in _THREE_MODES:
        assert outs[mode].shape == base_shape, f"{mode.value} shape != fp32_qdq"


def test_mrnn_three_modes_logits_cosine_close_to_fp32(mrnn_three_mode_bundle, capsys):
    """3 档主路径 logits 与 fp32_qdq 的 cosine ≥ 0.9995；max_abs / mean_abs / cosine 显式打印。

    阈值取自设计 §10.1 参考表（``fixed_scale_qdq`` cosine ≈ 0.99985+；
    ``fp16_qdq`` cosine ≈ 0.9999）。这里取保守下限 0.9995，让回归足够严格。
    """

    sim = mrnn_three_mode_bundle["sim"]
    x = mrnn_three_mode_bundle["dummy"]
    sim.model.eval()

    outs: dict[ExecutionMode, torch.Tensor] = {}
    with torch.no_grad():
        for mode in _THREE_MODES:
            with quant_execution_mode(mode):
                outs[mode] = _to_dense(sim.model(x)).float().detach()

    fp_ref = outs[ExecutionMode.FP32_QDQ]
    fp_ref_flat = fp_ref.flatten()
    fp_ref_norm = max(float(fp_ref.abs().max().item()), 1e-6)

    rows: list[str] = ["mode | cosine | max_abs | rel_max_abs | mean_abs"]
    for mode in (ExecutionMode.FP16_QDQ, ExecutionMode.FIXED_SCALE_QDQ):
        cand = outs[mode]
        cand_flat = cand.flatten()
        cos = torch.nn.functional.cosine_similarity(
            fp_ref_flat.unsqueeze(0), cand_flat.unsqueeze(0)
        ).item()
        diff = (cand - fp_ref).abs()
        max_abs = float(diff.max().item())
        mean_abs = float(diff.mean().item())
        rel_max = max_abs / fp_ref_norm
        rows.append(
            f"{mode.value} | {cos:.6f} | {max_abs:.3e} | {rel_max:.3e} | {mean_abs:.3e}"
        )
        assert cos >= 0.9995, (
            f"{mode.value} vs fp32_qdq logits cosine = {cos:.6f} < 0.9995"
        )

    # 通过 capsys 注入 stdout，pytest -s 时直接可见；
    # 失败 / -s 模式下都会成为接入证据。
    print("\n[3-mode numerical proof]")
    for r in rows:
        print(" ", r)
    capsys.disabled() if False else None  # 只是显式持有 capsys，避免 lint 警告


# QAT 反传 smoke 由 ``tests/fixed_point/test_mrnn_int16_qat_smoke.py`` 覆盖
# （full-graph INT16_FIXED_QAT_SIM forward + backward + multi-step）。本文件不重复，
# 真实数据集 QAT 后多模式 ΔTop1 由 ``examples/quick_start.py`` 在 SpeechCommands 上跑出。

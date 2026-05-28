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
"""Unit tests for QuantGRU black-box AIMET adapter (Phase 2 skeleton)."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Optional, Tuple

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxscript")
import torch.nn as nn  # noqa: E402

from aimet_torch.fixed_point import ExecutionMode, quant_execution_mode  # noqa: E402
from aimet_torch.fixed_point.errors import (  # noqa: E402
    IncompatibleAdapterVersionError,
    QuantGRUFlagLockedError,
)
from aimet_torch.fixed_point.gradient_helpers import (  # noqa: E402
    is_quantized_activation,
    stop_grad_dequantize,
    wrap_int_tensor_with_meta,
)
from aimet_torch.fixed_point.quantgru_adapter import (  # noqa: E402
    aimet_capabilities,
    aimet_configure,
    check_adapter_version,
    dispatch_quantgru_blackbox,
    get_io_quant_meta,
    stub_aimet_configure,
    stub_forward_quantized,
    stub_get_io_quant_meta,
)
from aimet_torch.fixed_point.sim_utils import (  # noqa: E402
    ensure_output_quantizers_for_int16_eval,
)
from aimet_torch.fixed_point.tensor import Int16QuantizedTensor  # noqa: E402


def _has_quant_gru() -> bool:
    try:
        import quant_gru  # noqa: F401

        return True
    except ImportError:
        return False


HAS_QUANT_GRU = _has_quant_gru()
HAS_CUDA = torch.cuda.is_available()
requires_quant_gru = pytest.mark.skipif(
    not HAS_QUANT_GRU,
    reason="quant_gru / gru_interface_binding not installed",
)
requires_quant_gru_cuda = pytest.mark.skipif(
    not (HAS_QUANT_GRU and HAS_CUDA),
    reason="quant_gru with CUDA required",
)


class FakeQuantParams:
    shift_x_ = 8
    zp_x_ = 0
    x_ = 16
    x_symmetric_ = True
    x_unsigned_ = False
    shift_h_ = 8
    zp_h_ = 0
    h_ = 16
    h_symmetric_ = True
    h_unsigned_ = False


class FakeQuantGRU(nn.Module):
    """Minimal stand-in for adapter-level tests (not a real GRU)."""

    def __init__(self):
        super().__init__()
        self.use_quantization = False
        self.calibrating = False
        self.export_mode = False
        self.export_format = "float"
        self.quant_params = None
        self._aimet_lock = False

    def is_calibrated(self) -> bool:
        return self.quant_params is not None

    @contextmanager
    def _aimet_unlock_ctx(self):
        prev = self._aimet_lock
        self._aimet_lock = False
        try:
            yield
        finally:
            self._aimet_lock = prev

    def forward(self, input: torch.Tensor, hx=None):
        del hx
        return input + 0.1, input[:, -1:, :]


@pytest.fixture
def fake_quant_params():
    return FakeQuantParams()


@pytest.fixture
def calibrated_fake_gru(fake_quant_params):
    gru = FakeQuantGRU()
    gru.quant_params = fake_quant_params
    return gru


def test_gradient_helpers_wrap_and_dequant():
    meta = {"scale": 0.01, "zp": 0, "bitwidth": 16, "is_symmetric": True}
    ints = torch.tensor([[10, -5]], dtype=torch.int32)
    carrier = wrap_int_tensor_with_meta(ints, meta)
    assert isinstance(carrier, Int16QuantizedTensor)
    assert is_quantized_activation(carrier)

    fp = stop_grad_dequantize(carrier)
    assert fp.dtype == torch.float32
    assert fp.shape == ints.shape
    # Identity STE: values match dequant, backward does not attach to int_repr.
    assert torch.allclose(fp, carrier.to_float())


def test_stub_get_io_quant_meta_schema(calibrated_fake_gru):
    meta = stub_get_io_quant_meta(calibrated_fake_gru)
    assert set(meta.keys()) == {"input", "output", "hidden"}
    for key in ("input", "output", "hidden"):
        entry = meta[key]
        assert {"scale", "zp", "bitwidth", "is_symmetric"} <= set(entry.keys())
        assert entry["scale"] == pytest.approx(1.0 / 256.0)


def test_stub_get_io_quant_meta_uncalibrated_raises():
    gru = FakeQuantGRU()
    with pytest.raises(RuntimeError, match="not calibrated"):
        stub_get_io_quant_meta(gru)


@pytest.mark.parametrize(
    "mode,use_quant,calibrating",
    [
        ("fp32", False, False),
        ("fp32_qdq", False, False),
        ("int16_fixed_eval", True, False),
        ("calibrating", False, True),
    ],
)
def test_stub_aimet_configure(mode, use_quant, calibrating):
    gru = FakeQuantGRU()
    stub_aimet_configure(gru, mode)
    assert gru.use_quantization is use_quant
    assert gru.calibrating is calibrating
    assert gru.export_mode is False


def test_check_adapter_version_ok(calibrated_fake_gru):
    check_adapter_version(calibrated_fake_gru)


def test_check_adapter_version_raises(calibrated_fake_gru, monkeypatch):
    monkeypatch.setattr(
        "aimet_torch.fixed_point.quantgru_adapter.stub_aimet_capabilities",
        lambda _module: {"adapter_version": "2.0"},
    )
    with pytest.raises(IncompatibleAdapterVersionError):
        check_adapter_version(calibrated_fake_gru)


def test_dispatch_quantgru_blackbox_int16_eval(calibrated_fake_gru, monkeypatch):
    def _fake_forward_quantized(module, input, hx=None):
        del module, hx
        meta = stub_get_io_quant_meta(calibrated_fake_gru)
        out = torch.round(input / meta["output"]["scale"]).to(torch.int32)
        hn = torch.round(input[:, -1:, :] / meta["hidden"]["scale"]).to(torch.int32)
        return out, hn

    monkeypatch.setattr(
        "aimet_torch.fixed_point.quantgru_adapter.forward_quantized",
        _fake_forward_quantized,
    )

    x = torch.randn(2, 4, 8)
    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        out_q, hn_q = dispatch_quantgru_blackbox(calibrated_fake_gru, x)

    assert isinstance(out_q, Int16QuantizedTensor)
    assert isinstance(hn_q, Int16QuantizedTensor)
    assert out_q.int_repr.shape == x.shape
    assert hn_q.int_repr.shape == (2, 1, 8)


def test_dispatch_quantgru_blackbox_qat_returns_float(calibrated_fake_gru, monkeypatch):
    def _fake_forward_quantized(module, input, hx=None):
        del module, hx
        meta = stub_get_io_quant_meta(calibrated_fake_gru)
        out = torch.round(input / meta["output"]["scale"]).to(torch.int32)
        hn = torch.round(input[:, -1:, :] / meta["hidden"]["scale"]).to(torch.int32)
        return out, hn

    monkeypatch.setattr(
        "aimet_torch.fixed_point.quantgru_adapter.forward_quantized",
        _fake_forward_quantized,
    )

    x = torch.randn(2, 4, 8)
    with quant_execution_mode(ExecutionMode.INT16_FIXED_QAT_SIM):
        out_fp, hn_fp = dispatch_quantgru_blackbox(calibrated_fake_gru, x)

    assert out_fp.dtype == torch.float32
    assert hn_fp.dtype == torch.float32


def test_aimet_capabilities_contract(calibrated_fake_gru):
    caps = aimet_capabilities(calibrated_fake_gru)
    assert caps["adapter_version"] == "1.0"
    assert caps["supports_forward_quantized"] is True
    assert "int16_fixed_eval" in caps["supported_modes"]


@requires_quant_gru
def test_quantized_quantgru_flag_lock():
    from aimet_torch.v2.nn.modules.custom import QuantizedQuantGRU

    gru = QuantizedQuantGRU(8, 8, batch_first=True).cuda()
    with pytest.raises(QuantGRUFlagLockedError, match="use_quantization"):
        gru.use_quantization = True


@requires_quant_gru
def test_quantized_quantgru_unlock_ctx_allows_flag_change():
    from aimet_torch.v2.nn.modules.custom import QuantizedQuantGRU

    gru = QuantizedQuantGRU(4, 4, batch_first=True).cuda()
    with gru._aimet_unlock_ctx():
        gru.use_quantization = False
    assert gru.use_quantization is False


@requires_quant_gru
def test_sim_utils_skips_quantized_quantgru_for_output_quantizer_patch():
    from aimet_torch.v2.nn.modules.custom import QuantizedQuantGRU

    gru = QuantizedQuantGRU(4, 4, batch_first=True)
    sim = SimpleNamespace(model=nn.ModuleDict({"gru": gru}))
    patched = ensure_output_quantizers_for_int16_eval(sim)
    assert patched == []
    assert gru.output_quantizers[0] is None


@requires_quant_gru_cuda
def test_compute_encodings_hooks_toggle_calibrating():
    from aimet_torch.v2.nn import compute_encodings
    from aimet_torch.v2.nn.modules.custom import QuantizedQuantGRU

    gru = QuantizedQuantGRU(4, 4, batch_first=True).cuda()
    model = nn.Sequential(gru)
    x = torch.randn(2, 5, 4, device="cuda")

    assert gru.calibrating is False
    with compute_encodings(model):
        assert gru.calibrating is True
        with torch.no_grad():
            model(x)
    assert gru.calibrating is False


# ---------------------------------------------------------------------------
# Plan §3.1.1 G: FP32_QDQ / FP16_QDQ wrapper 修复后的五条用例
# 这些测试需要真实 QuantGRU + CUDA；本地无该依赖时按 skip 处理。
# 测试以 **手工注入** quantizer 的方式直接验证
# `_builtin_torch_fn_helper` + forward 路由；
# 不依赖完整 QuantizationSimModel.realize 链路。
# ---------------------------------------------------------------------------


def _inject_qdq_quantizers(gru, *, bitwidth: int = 8, symmetric: bool = False):
    """手工注入 input/output QuantizeDequantize 槽（sim builder 路径的近似替身）。"""
    from aimet_torch.v2.quantization.affine import QuantizeDequantize

    device = next(gru.parameters()).device

    def make() -> "QuantizeDequantize":
        return QuantizeDequantize(shape=(), bitwidth=bitwidth, symmetric=symmetric).to(device)

    with gru._aimet_unlock_ctx():
        gru.input_quantizers[0] = make()
        gru.input_quantizers[1] = make()
        gru.output_quantizers[0] = make()
        gru.output_quantizers[1] = make()


def _inject_fp16_quantizers(gru):
    """手工注入 fp16 FloatQuantizeDequantize 槽。"""
    from aimet_torch.v2.quantization.float import FloatQuantizeDequantize

    device = next(gru.parameters()).device

    def make() -> "FloatQuantizeDequantize":
        return FloatQuantizeDequantize(dtype=torch.float16).to(device)

    with gru._aimet_unlock_ctx():
        gru.input_quantizers[0] = make()
        gru.input_quantizers[1] = make()
        gru.output_quantizers[0] = make()
        gru.output_quantizers[1] = make()


@requires_quant_gru_cuda
def test_fp32_qdq_calibrated_forward_matches_nn_gru_with_edge_qdq():
    """§3.1.1 G-1: FP32_QDQ 已校准 forward 数值 ≈ nn.GRU + 边界 QuantizeDequantize。"""
    import quant_gru as _qg
    from aimet_torch.v2.nn import compute_encodings
    from aimet_torch.v2.nn.modules.custom import QuantizedQuantGRU

    torch.manual_seed(0)
    gru = QuantizedQuantGRU(4, 4, batch_first=True).cuda()
    _inject_qdq_quantizers(gru, bitwidth=8, symmetric=False)

    x = torch.randn(2, 5, 4, device="cuda")

    # 校准期 + FP32_QDQ：encoding 由 input/output 的 QuantizeDequantize 自我收集。
    with quant_execution_mode(ExecutionMode.FP32_QDQ):
        with compute_encodings(nn.Sequential(gru)):
            with torch.no_grad():
                _ = gru(x)

        with torch.no_grad():
            out_aimet, _ = gru(x)

    assert out_aimet.dtype == torch.float32
    assert out_aimet.shape == (2, 5, 4)
    # 经过双端 fake-quant，输出与原始 fp32 GRU forward 应该「相近但不等」。
    # ExecutionMode 没有"纯 FP32"枚举；reference 直接走底层 QuantGRU.forward，
    # 此时 use_quantization=False, calibrating=False，等同于 fp32 推理。
    assert gru.use_quantization is False and gru.calibrating is False
    with torch.no_grad():
        out_fp32, _ = _qg.QuantGRU.forward(gru, x, None)
    diff = (out_aimet - out_fp32).abs().max().item()
    # 8-bit fake-quant 的典型误差量级 ≤ 5e-2（保守上限）
    assert diff <= 5e-2, f"FP32_QDQ vs FP32 max abs diff = {diff} exceeds budget"


@requires_quant_gru_cuda
def test_fp32_qdq_uncalibrated_input_quantizer_engaged():
    """§3.1.1 G-2: 未校准时 input_quantizers[0].is_initialized() 必须为 False，
    且 forward 路径会真实调到 input quantizer（不是静默走 FP32）。"""
    from aimet_torch.v2.nn.modules.custom import QuantizedQuantGRU

    gru = QuantizedQuantGRU(4, 4, batch_first=True).cuda()
    _inject_qdq_quantizers(gru, bitwidth=8, symmetric=False)

    iq = gru.input_quantizers[0]
    oq = gru.output_quantizers[0]
    assert iq is not None and oq is not None
    assert iq.is_initialized() is False
    assert oq.is_initialized() is False

    called = {"n": 0}

    def _hook(module, args, kwargs):
        called["n"] += 1

    handle = iq.register_forward_pre_hook(_hook, with_kwargs=True)
    try:
        x = torch.randn(2, 3, 4, device="cuda")
        with quant_execution_mode(ExecutionMode.FP32_QDQ):
            try:
                with torch.no_grad():
                    gru(x)
            except Exception:
                # 未初始化 quantizer 抛错是预期的；不影响"被调用过"的断言。
                pass
    finally:
        handle.remove()

    assert called["n"] >= 1, "input_quantizer[0] 未被 forward 调用，FP32_QDQ 路由失效"


@requires_quant_gru_cuda
def test_fp16_qdq_forward_close_to_fp16_cast():
    """§3.1.1 G-3: FP16_QDQ forward 数值与 fp32 forward 在 fp16 cast 后逐元素 close。"""
    import quant_gru as _qg
    from aimet_torch.v2.nn.modules.custom import QuantizedQuantGRU

    torch.manual_seed(1)
    gru = QuantizedQuantGRU(4, 4, batch_first=True).cuda()
    _inject_fp16_quantizers(gru)

    x = torch.randn(2, 4, 4, device="cuda")

    with quant_execution_mode(ExecutionMode.FP16_QDQ):
        with torch.no_grad():
            out_fp16, _ = gru(x)
    # ExecutionMode 没有"纯 FP32"枚举；reference 直接走底层 QuantGRU.forward。
    # 上一行 FP16_QDQ forward 退出后，wrapper 已把 flag 设为 (False, False)，
    # 等同于 fp32 推理。
    assert gru.use_quantization is False and gru.calibrating is False
    with torch.no_grad():
        out_fp32, _ = _qg.QuantGRU.forward(gru, x, None)

    out_fp32_as_fp16 = out_fp32.to(torch.float16).to(torch.float32)
    diff = (out_fp16.to(torch.float32) - out_fp32_as_fp16).abs().max().item()
    # fp16 ULP 上限保守取 1e-2（典型 < 1e-3）
    assert diff <= 1e-2, f"FP16_QDQ vs fp32→fp16 max abs diff = {diff}"


@requires_quant_gru_cuda
def test_fp32_qdq_compute_encodings_initializes_boundary_quantizers():
    """§3.1.1 G-4: compute_encodings 退出后 input/output_quantizers[0/1] 都已初始化。"""
    from aimet_torch.v2.nn import compute_encodings
    from aimet_torch.v2.nn.modules.custom import QuantizedQuantGRU

    torch.manual_seed(2)
    gru = QuantizedQuantGRU(4, 4, batch_first=True).cuda()
    _inject_qdq_quantizers(gru, bitwidth=8, symmetric=False)

    x = torch.randn(2, 5, 4, device="cuda")
    with quant_execution_mode(ExecutionMode.FP32_QDQ):
        with compute_encodings(nn.Sequential(gru)):
            with torch.no_grad():
                gru(x)

    assert gru.input_quantizers[0].is_initialized() is True
    assert gru.output_quantizers[0].is_initialized() is True
    # hidden 端独立 quantizer 也应被前向触发并收 stats（§2.5 后续会要求与 input/output[1] 共享）
    assert gru.input_quantizers[1].is_initialized() is True
    assert gru.output_quantizers[1].is_initialized() is True


@requires_quant_gru_cuda
def test_hidden_quantizer_shared_after_first_forward():
    """§2.5: 第一次 FP*_QDQ forward 后 hx/h_n quantizer 必须是同一对象引用。"""
    from aimet_torch.v2.nn.modules.custom import QuantizedQuantGRU

    gru = QuantizedQuantGRU(4, 4, batch_first=True).cuda()
    _inject_qdq_quantizers(gru, bitwidth=8, symmetric=False)

    # sim builder 注入后两端独立
    assert gru.input_quantizers[1] is not gru.output_quantizers[1]

    x = torch.randn(2, 3, 4, device="cuda")
    with quant_execution_mode(ExecutionMode.FP32_QDQ):
        try:
            with torch.no_grad():
                gru(x)
        except Exception:
            # 未校准时 forward 可能抛错；本测试只关心 share 是否在 forward 入口生效。
            pass

    # forward 后必须是同一对象引用
    assert gru.input_quantizers[1] is gru.output_quantizers[1], (
        "plan §2.5 违反：hx 端与 h_n 端 quantizer 未共享同一 EncodingBase"
    )


@requires_quant_gru_cuda
def test_hidden_share_does_not_apply_to_int16_path():
    """§2.5 + INT16: INT16 路径下不会进入 _ensure_hidden_quantizer_shared，
    sim builder 注入的两端独立 quantizer 维持原状（dispatch 不读这些 quantizer）。"""
    from aimet_torch.v2.nn.modules.custom import QuantizedQuantGRU

    gru = QuantizedQuantGRU(4, 4, batch_first=True).cuda()
    _inject_qdq_quantizers(gru, bitwidth=8, symmetric=False)

    iq1_id = id(gru.input_quantizers[1])
    oq1_id = id(gru.output_quantizers[1])
    assert iq1_id != oq1_id

    x = torch.randn(2, 3, 4, device="cuda")
    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        try:
            with torch.no_grad():
                gru(x)
        except Exception:
            pass  # 未校准 / kernel 等错误不影响本断言

    # INT16 不触发 share；两端仍是 sim builder 注入的独立实例
    assert id(gru.input_quantizers[1]) == iq1_id
    assert id(gru.output_quantizers[1]) == oq1_id
    assert gru.input_quantizers[1] is not gru.output_quantizers[1]


@requires_quant_gru_cuda
def test_fixed_scale_qdq_forward_via_wrapper_alias():
    """§0.2.1: FIXED_SCALE_QDQ 在 wrapper 层别名为 FP32_QDQ；forward 路径与 FP32_QDQ 一致。

    QuantGRU 内部 use_quantization=False (浮点 forward)；
    boundary Q/DQ 由 AIMET quantizer 在 ExecutionMode.FIXED_SCALE_QDQ 下自动切到 (m_int16, rshift)。
    """
    from aimet_torch.v2.nn import compute_encodings
    from aimet_torch.v2.nn.modules.custom import QuantizedQuantGRU

    torch.manual_seed(10)
    gru = QuantizedQuantGRU(4, 4, batch_first=True).cuda()
    _inject_qdq_quantizers(gru, bitwidth=8, symmetric=False)

    x = torch.randn(2, 5, 4, device="cuda")

    # 校准在 FP32_QDQ 下完成（fixed-scale 网格在 forward 时自动切换）。
    with quant_execution_mode(ExecutionMode.FP32_QDQ):
        with compute_encodings(nn.Sequential(gru)):
            with torch.no_grad():
                gru(x)

    # FIXED_SCALE_QDQ forward 不应抛 ValueError("Unsupported aimet_configure mode")。
    with quant_execution_mode(ExecutionMode.FIXED_SCALE_QDQ):
        with torch.no_grad():
            out_fs, hn_fs = gru(x)

    # QuantGRU 内部 flag 应被映射到 fp32_qdq 语义（use_quantization=False）。
    assert gru.use_quantization is False
    assert gru.calibrating is False

    # 输出 shape 与 FP32_QDQ 一致。
    with quant_execution_mode(ExecutionMode.FP32_QDQ):
        with torch.no_grad():
            out_fp, hn_fp = gru(x)

    assert out_fs.shape == out_fp.shape == x.shape
    assert hn_fs.shape == hn_fp.shape

    # FIXED_SCALE_QDQ vs FP32_QDQ 数值差异应在 (m_int16, rshift) 近似量级（POT 时 = 0）。
    diff = (out_fs - out_fp).abs().max().item()
    assert diff <= 0.5, (
        f"QuantGRU FIXED_SCALE_QDQ vs FP32_QDQ max abs diff = {diff} 超出预期"
    )


@requires_quant_gru_cuda
def test_compute_encodings_inside_fixed_scale_qdq_context_does_not_raise():
    """§0.2.1 + bug fix: 在 FIXED_SCALE_QDQ 上下文里跑 compute_encodings 必须不抛.

    回归用例：早期 _aimet_compute_encodings_exit 直接用 mode.value 调
    aimet_configure，FIXED_SCALE_QDQ 不在 contract v1 _SUPPORTED_MODES 内
    会触发 ValueError("Unsupported aimet_configure mode: 'fixed_scale_qdq'")。
    修复后 exit hook 先经 resolve_quantgru_mode_str 别名为 'fp32_qdq'。
    """
    from aimet_torch.v2.nn import compute_encodings
    from aimet_torch.v2.nn.modules.custom import QuantizedQuantGRU

    torch.manual_seed(11)
    gru = QuantizedQuantGRU(4, 4, batch_first=True).cuda()
    _inject_qdq_quantizers(gru, bitwidth=8, symmetric=False)

    x = torch.randn(2, 5, 4, device="cuda")

    with quant_execution_mode(ExecutionMode.FIXED_SCALE_QDQ):
        with compute_encodings(nn.Sequential(gru)):
            with torch.no_grad():
                gru(x)
        # 退出 compute_encodings 后 exit hook 已被触发；下一次 forward 也应正常
        with torch.no_grad():
            out, hn = gru(x)

    assert out.shape == x.shape
    assert hn.shape[-1] == 4
    # exit hook 别名映射后 QuantGRU 内部应停留在 fp32_qdq 语义。
    assert gru.use_quantization is False
    assert gru.calibrating is False


@requires_quant_gru_cuda
def test_int16_path_does_not_trigger_boundary_quantizers():
    """§3.1.1 G-5: 切到 INT16 后 boundary quantizer 不被调用（dispatch 全权接管）。"""
    from aimet_torch.v2.nn.modules.custom import QuantizedQuantGRU

    gru = QuantizedQuantGRU(4, 4, batch_first=True).cuda()
    _inject_qdq_quantizers(gru, bitwidth=8, symmetric=False)

    boundary_calls = {"n": 0}

    def _hook(module, args, kwargs):
        boundary_calls["n"] += 1

    handles = [
        gru.input_quantizers[0].register_forward_pre_hook(_hook, with_kwargs=True),
        gru.input_quantizers[1].register_forward_pre_hook(_hook, with_kwargs=True),
        gru.output_quantizers[0].register_forward_pre_hook(_hook, with_kwargs=True),
        gru.output_quantizers[1].register_forward_pre_hook(_hook, with_kwargs=True),
    ]
    try:
        x = torch.randn(2, 4, 4, device="cuda")
        with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
            try:
                with torch.no_grad():
                    gru(x)
            except Exception:
                # 未校准 / kernel 缺失等 INT16 错误属于非 R0 范围；
                # 本测试只关心 boundary quantizer 是否被调用。
                pass
    finally:
        for h in handles:
            h.remove()

    assert boundary_calls["n"] == 0, (
        f"INT16 路径意外触发 boundary quantizer ({boundary_calls['n']} 次); "
        "应由 dispatch_quantgru_blackbox 全权接管"
    )

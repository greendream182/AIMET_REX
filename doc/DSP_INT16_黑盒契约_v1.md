# MRNN DSP 前端 INT16 黑盒契约 v1（草案）

> 状态：草案 v1 / 配合文档：[QuantGRU_INT16接入计划.md](./QuantGRU_INT16接入计划.md) 附 R3
> 目标：让 STFT / BandConverter 等前端与 QuantGRU 一样，可被 AIMET 当作**黑盒**接入 INT16_FIXED_EVAL。

## 0. 边界原则（与 QuantGRU §0.2.1 对齐）

| 算子 | INT16 bit-exact 责任方 | AIMET 责任方 |
|---|---|---|
| **STFT** (`transform_cpx`) | v1 **decomposed**（Pad + Conv1d INT16）；bit-exact 黑盒见 §2 | `model_preparer` conv1d + INT16 kernel；非独立黑盒 |
| **BandConverter** | 默认走 **MatMul INT16 kernel**（固定 ERB 矩阵）；可选升级为独立黑盒 | `ensure_output_quantizers` + MatMul dispatch |
| **PowerCompress / HypotFun / CLN** | v1 **decomposed**（Sqrt/Square **PWL/CLZ**；**Abs PWL**；**Sign 整数精确**；**Divide 整数**；Mean reference） | 无 preserve；``clamp(min=EPS)`` 代替 ``+EPS`` |
| **QuantizableBatchNorm2d** | v1 **decomposed**（Sub/Add/Sqrt/Divide INT16） | metric 已验收 |

## 1. 已落地（2026-05-28）

| 项 | 落点 |
|---|---|
| MatMul INT16 kernel | `aimet_torch/fixed_point/kernels/conv_linear.py::MatMulInt16Kernel` |
| MatMul dispatchable | `aimet_torch/fixed_point/sim_utils.py` |
| MatMul output scale | `adapter.dispatch_int16_fixed` 双输入 scale 推导 |
| Divide INT16 整数 kernel | `kernels/eltwise.py::DivideInt16Kernel`（``M,rshift`` + 整数除 + ``eps`` 防零） |
| STFT decomposed | `model_preparer` conv1d + `DynamicConv1d`；`examples/common/torch_stft.py::forward` |
| Abs INT16 PWL kernel | `kernels/lut.py::AbsInt16Kernel` + adapter 自动 `pwl_lut` |
| Sign INT16 整数精确 kernel | `kernels/eltwise.py::SignInt16Kernel` |
| 全图 metric | `quick_start_int16_metric.py`：无 FP32 preserve；Δ=0 pp |
| shape 标量旁路 | `aimet_torch/fixed_point/shape_meta.py`（`b*f` 等 layout 元数据） |
| FixedPointSimTensor layout | `aimet_torch/fixed_point/tensor.py` |
| 前端分段 smoke | `tests/fixed_point/test_mrnn_int16_frontend_segment.py`（BandConverter + Conv，无 STFT） |

## 2. STFT 黑盒契约（可选升级；v1 已走 decomposed INT16）

> **v1 现状**：STFT 经 ``model_preparer`` 分解为 Pad + Conv1d，走 INT16 kernel（非黑盒）。
> 本节描述 **bit-exact 黑盒**升级路径，供 DSP 侧后续实现。

参照 QuantGRU contract v1，STFT 黑盒需暴露：

```python
def get_io_quant_meta(self) -> dict: ...
def forward_quantized(self, input: Tensor) -> Tensor: ...  # bit-exact INT16 输出
def aimet_configure(self, mode: str) -> None: ...
def aimet_capabilities(self) -> dict: ...
```

**不变量**：
- `forward` 入口/出口 dtype = float32（与 QuantGRU §1.1 一致）
- `forward_quantized` 输出 int32 张量 + 侧车 meta（scale/zp/bitwidth）
- AIMET wrapper 仅在 INT16 模式调 `forward_quantized`；FP32_QDQ 走 boundary helper

## 3. BandConverter

**当前策略（v1 最小）**：不单独黑盒；`model_preparer` 追踪 `torch.matmul` → `QuantizedMatMul` → `MatMulInt16Kernel`。

**升级路径（v1.1）**：若 ERB 矩阵需 offline 整数量化或融合 rescale，再抽 `QuantizedBandConverter` 黑盒（类似 QuantGRU）。

## 4. 验收

| 阶段 | 验收 |
|---|---|
| R3-min（当前） | `test_mrnn_int16_frontend_segment.py` 全绿；metric Δ≤0.5 pp；`diagnose` 全绿 |
| R3-full | 全图 decomposed INT16 metric Δ≤0.5 pp（已完成） |
| R3-bit-exact | Abs/Sign/Divide ✅；Pow0.5；STFT 黑盒 |

## 5. 阻塞

- **bit-exact LUT**（可选）：Pow0.5 等待；Abs/Sign/Divide 已落地
- **STFT bit-exact 黑盒**（可选）：integer FFT/im2col 语义定义

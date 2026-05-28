# 07 Conv / Linear / MatMul INT16 kernel

Status: implemented

> **硬件对齐（ADR-015）**：默认 `torch.matmul` 在 int32 上可能绕回。严格仿真：`AIMET_RX_ACC_INT32_SAT=1` 或 `AIMET_RX_HW_REF=1` 时在 **int64** 上 MAC，经 `saturate_mac_accumulator` 钳到 INT32 再 `requantize_int`。
>
> **Conv3d**：无原生整数 `conv3d`；默认 float32 MAC + round。HW ref 下 float64 MAC（int64 操作数精确可表）+ INT32 累加器饱和。

## 1. 目标

实现 Conv1d / Conv2d / Conv3d / Linear / MatMul / BMM 的 INT16 定点 kernel，全程整数计算，使用 multiplier + rshift requantize。

## 2. 范围

### 2.1 新增文件

- [aimet_torch/fixed_point/kernels/conv_linear.py](../../aimet_torch/fixed_point/kernels/conv_linear.py)

### 2.2 修改文件

- [aimet_torch/fixed_point/kernels/__init__.py](../../aimet_torch/fixed_point/kernels/__init__.py)：触发 kernel 注册。

### 2.3 不在范围

- BatchNorm folding（应在 Sim 准备阶段完成）。
- 离线 multiplier / bias 生成（spec 10）。
- 自写 CUDA kernel（视性能需要再单列 spec）。

## 3. 前置依赖

- spec 04 / 05 / 06 完成。

## 4. 数据契约

输入：

- `inputs[0]`：`Int16QuantizedTensor`，shape 与 op 维度一致
- `params["weight"]`：`Int16QuantizedTensor`，per-tensor 或 per-channel
- `params["bias"]`（可选）：`int32` tensor（已离线生成）
- `output_encoding`：含 `multiplier` (int16) 与 `rshift` (int8)

输出：

- `Int16QuantizedTensor`，shape 由 op 计算规则决定，dtype `int16`

不变式：

- weight per-channel 时，`output_encoding.multiplier` 与 `rshift` 也必须 per-channel（沿输出通道）
- bias 的 scale = `x_scale * w_scale`（per-channel 时与 weight 对齐）
- 累加器使用 int32；中间 `acc * multiplier` 使用 int64
- saturate 到 int16 范围

## 5. API 签名

```python
# aimet_torch/fixed_point/kernels/conv_linear.py
from ..registry import register_fixed_kernel, FixedKernel
from ..tensor import Int16QuantizedTensor
from ..requantize import requantize_int
import torch
import torch.nn as nn

@register_fixed_kernel(nn.Conv2d)
class Conv2dInt16Kernel:
    module_type = nn.Conv2d
    def __call__(self, inputs, params, output_encoding, extra) -> Int16QuantizedTensor: ...

# 同模式：Conv1d / Conv3d / Linear / MatMul / BMM
```

`extra` dict 字段：

- Conv：`stride`, `padding`, `dilation`, `groups`
- Linear：无
- MatMul / BMM：`transpose_a`, `transpose_b`

## 6. 算法 / 伪代码

Conv2d：

```text
x = inputs[0].int_repr.to(int32)
x_zp = inputs[0].zero_point.to(int32)
w = params["weight"].int_repr.to(int32)
w_zp = params["weight"].zero_point.to(int32)
bias_int32 = params.get("bias", None)        # 已是 int32

# zero-point 折入：(x - x_zp) * (w - w_zp)
# 实现可分四项展开以提升效率，初版按朴素展开：
x_centered = x - x_zp
w_centered = w - w_zp
# 调用 PyTorch int conv (CPU): F.conv2d 不支持 int32，需要 fallback：
acc_int32 = conv2d_int32_reference(x_centered, w_centered, stride, padding, dilation, groups)

if bias_int32 is not None:
    acc_int32 = acc_int32 + bias_int32.view(1, -1, 1, 1)

# 整数 requantize
y_int16 = requantize_int(
    acc_int32,
    output_encoding.multiplier,
    output_encoding.rshift,
    output_encoding.zero_point,
    output_encoding.qmin,
    output_encoding.qmax,
)
return Int16QuantizedTensor(int_repr=y_int16, scale=output_encoding.scale, ...)
```

`conv2d_int32_reference` 实现策略：

- CPU：用 `torch.int32` GEMM-like 实现（im2col + matmul）。
- GPU：临时 cast 到 int32 计算 matmul（PyTorch CUDA 支持 int32 matmul），再 reshape。
- 性能不达 Tier C 时再考虑自写 CUDA。

边界：

- 空 input：返回空 `Int16QuantizedTensor`。
- 大尺寸：检查中间 `int32` 是否溢出（求和元素个数 × 最大乘积 < 2^31）；超出抛 `RuntimeError` 且建议升级到 int64 累加。
- groups > 1：按组循环。

Linear：

```text
acc_int32 = (x - x_zp) @ (w - w_zp).T
# 后续与 Conv 相同
```

MatMul / BMM：参考 Linear，处理批次维。

## 7. 实施步骤

1. 实现 `conv2d_int32_reference`，提供 CPU + CUDA 两个 path。
2. 实现 `Conv2dInt16Kernel`，注册到 registry。
3. 复用同套逻辑覆盖 Conv1d / Conv3d / Linear / MatMul / BMM。
4. 单元测试每个 op 的 golden case：手写整数参考与 kernel 输出 bit-exact 一致。
5. 与 `fp32_qdq` 输出对比 cosine_similarity ≥ 0.999。

## 8. 验收标准

### 8.1 单元测试

```python
def test_conv2d_golden_per_tensor():
    x = build_int16_tensor([[1, 2], [3, 4]], scale=0.1, zp=0)
    w = build_int16_tensor([[1, 0], [0, 1]], scale=0.1, zp=0)
    bias = torch.tensor([0], dtype=torch.int32)
    out_enc = build_output_encoding(scale=0.01, zp=0, multiplier=32767, rshift=15)
    kernel = get_fixed_kernel(nn.Conv2d)
    y = kernel([x], {"weight": w, "bias": bias}, out_enc, {"stride": 1, ...})
    assert y.int_repr.dtype is SIM_TENSOR_DTYPE  # torch.int32 (ADR-013)
    assert y.int_repr.tolist() == expected_reference

def test_linear_per_channel_weight():
    ...

def test_overflow_raises():
    ...
```

### 8.2 必须通过的现有测试

- 默认模式 Conv/Linear 行为不变。

### 8.3 性能阈值

- Tier A：64×64 Conv 在 GPU 上 < 100ms（reference 实现）。

## 9. 不允许做的事

- 不允许在 kernel 内调用 `to_float()` 或任何 `.float()`。
- 不允许在 kernel 内构造 `multiplier` / `rshift`（必须从 `output_encoding` 取）。
- 不允许 wrap-around 溢出（必须 saturate）。
- 不允许 silent fallback 到 `nn.functional.conv2d(float)`。

## 10. 参考

- spec 05 requantize_int。
- `quant-gru-pytorch-main/src/gru_forward_gpu_quant.cu` int64 累加示例。
- TFLite `optimized_ops::Conv` 中的 quantized 实现。

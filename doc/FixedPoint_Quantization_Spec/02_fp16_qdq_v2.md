# 02 fp16_qdq v2 实现

Status: implemented

## 1. 目标

在 AIMET v2 中实现 `fp16_qdq` 模式：QDQ 输出张量与模块计算均使用 `torch.float16`，输出仍可携带整数网格 encoding。

## 2. 范围

### 2.1 修改文件

- [aimet_torch/v2/quantization/affine/backends/torch_builtins.py](../../aimet_torch/v2/quantization/affine/backends/torch_builtins.py)：`QuantDequantFunc` 增加 mode-aware dtype 处理。
- [aimet_torch/v2/quantization/affine/quantizer.py](../../aimet_torch/v2/quantization/affine/quantizer.py)：`QuantizeDequantize.forward()` 在 `fp16_qdq` 下输出 `float16`。
- [aimet_torch/v2/nn/base.py](../../aimet_torch/v2/nn/base.py)：`_patch_quantized_parameters()` 支持参数转 `float16`。
- [aimet_torch/v2/nn/true_quant.py](../../aimet_torch/v2/nn/true_quant.py)：`_quantize_dequantize_if_applicable()` 输出按 mode 保持 half。

### 2.2 新增文件

无（仅扩展 v2 现有文件）。

### 2.3 不在范围

- v1 fp16（见 spec 03）。
- int16 路径（见 spec 04+）。
- BatchNorm/LayerNorm fp32 fallback 配置（列入未来扩展）。

## 3. 前置依赖

- spec 01 完成（execution mode API 可用）。

## 4. 数据契约

- 输入 tensor dtype：可为 `float32` / `float16` / `bfloat16`。
- `fp16_qdq` 模式下，QDQ 输出 dtype = `torch.float16`。
- scale / offset 内部计算保持原 dtype（通常 `float32`），仅在最终结果阶段 cast 到 half，避免 round 误差放大。
- 参数（weight / bias）经 param quantizer 后 dtype = `torch.float16`。
- 输出 quantizer 输出 dtype = `torch.float16`。
- 默认 `fp32_qdq` 行为保持不变。

不变式：

- 模式判断必须每次 forward 时读取（不能缓存到模块属性），保证 context manager 立即生效。
- 不允许在 fp16 模式下静默 fallback 到 fp32（除非 op 完全不支持 half，且必须显式 warn）。

## 5. API 签名

无新增 public 符号。修改现有方法签名保持兼容。

## 6. 算法 / 伪代码

`QuantDequantFunc.forward`：

```text
mode = get_quant_execution_mode()

x_round = round(tensor.to(scale.dtype) / scale) - offset
x_quant = clamp(x_round, qmin, qmax)
x_dequant = (x_quant + offset) * scale     # 仍为 scale.dtype，通常 float32

if mode == FP16_QDQ:
    return x_dequant.to(torch.float16)
return x_dequant
```

`_patch_quantized_parameters`：

```text
quantized_param = param_quantizer(orig_param)
if get_quant_execution_mode() == FP16_QDQ:
    quantized_param = quantized_param.to(torch.float16)
ctx = patch_attr(self, param_name, quantized_param)
```

边界：

- 空 tensor / 0-d tensor：保持原行为。
- scale 为 `float64`：cast 路径仍只在最终输出做 half cast。
- per-channel scale：广播规则不变。

## 7. 实施步骤

1. 在 `torch_builtins.py` 顶部 import `get_quant_execution_mode`。
2. 修改 `QuantDequantFunc` 的 forward 与 inplace 变体，在返回前按 mode 添加 dtype cast。
3. `quantizer.py` 中 `QuantizeDequantize.forward()` 若已经包装在 mixed precision 路径，确保 dtype 与 backend 输出一致。
4. `nn/base.py` 中 `_patch_quantized_parameters()` 参数 cast。
5. `nn/true_quant.py` 中 `_quantize_dequantize_if_applicable()` 输出 cast。
6. 编写测试覆盖：
   - `test_fp16_qdq_output_dtype`
   - `test_fp16_qdq_param_dtype`
   - `test_fp16_qdq_default_unchanged`
   - `test_fp16_qdq_with_per_channel`
7. 跑现有 `tests/v2/` 全量回归确认默认路径未受影响。

## 8. 验收标准

### 8.1 单元测试

```python
def test_fp16_qdq_output_dtype():
    q = QuantizeDequantize(...).cuda()
    x = torch.randn(2, 3, 4, 4, dtype=torch.float32, device="cuda")
    with quant_execution_mode("fp16_qdq"):
        y = q(x)
    assert y.dtype == torch.float16

def test_fp16_qdq_default_unchanged():
    q = QuantizeDequantize(...)
    x = torch.randn(2, 3, 4, 4)
    y = q(x)
    assert y.dtype == torch.float32
```

数值验收：

- `fp16_qdq` 输出与 `fp32_qdq` 输出在 `cosine_similarity >= 0.9999` 范围内（per-tensor，per-channel 均测）。
- tiny Conv-BN-ReLU 模型端到端 logits cosine_similarity ≥ 0.9995。

### 8.2 必须通过的现有测试

- `tests/v2/test_affine_quantizer.py` 全部通过。
- `tests/v2/test_true_quant.py` 全部通过。

### 8.3 性能阈值

- `fp16_qdq` 单层 forward 时间 ≤ `fp32_qdq` 的 2×（CPU 仅做正确性测试，性能以 GPU 为准）。

## 9. 不允许做的事

- 不允许把 scale 本身 cast 成 `float16`（会丢精度）。
- 不允许在 fp16 模式下回退到 fp32 计算而不告知用户。
- 不允许修改 `QuantizationSimModel` 公共 API。
- 不允许在 forward 中调用 `set_quant_execution_mode`（只读）。

## 10. 参考

- 现状代码：`torch_builtins.py` 第 95-99 行 `QuantDequantFunc` 核心 QDQ 公式。
- INTERFACE.md 第 3 节 ExecutionMode。
- 顶层文档 ADR-002（半精度路径无运行时 fallback 默认）。

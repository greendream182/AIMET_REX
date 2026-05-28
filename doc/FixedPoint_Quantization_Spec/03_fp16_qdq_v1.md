# 03 fp16_qdq v1 实现

Status: implemented

## 1. 目标

为 AIMET v1 兼容路径实现 `fp16_qdq` 模式：wrapper forward 中 QDQ 输出与参数转 `float16`，wrapped 模块以 half 计算。

## 2. 范围

### 2.1 修改文件

- [aimet_torch/v1/quantsim_straight_through_grad.py](../../aimet_torch/v1/quantsim_straight_through_grad.py)：STE QDQ 输出按 mode 输出 half。
- [aimet_torch/v1/tensor_quantizer.py](../../aimet_torch/v1/tensor_quantizer.py)：`StaticGridTensorQuantizer.quantize_dequantize()` 与 learned grid 路径接入 dtype 控制。
- [aimet_torch/v1/qc_quantize_op.py](../../aimet_torch/v1/qc_quantize_op.py)：wrapper forward 在调用 wrapped 模块前 cast 输入 / 参数到 half。

### 2.2 不在范围

- 共享 fixed kernel registry（见 spec 06）。
- v1 INT16 适配（见 spec 04+）。

## 3. 前置依赖

- spec 01 完成。

## 4. 数据契约

- 输入：v1 wrapper 接收的 tensor，dtype 通常为 `float32`。
- `fp16_qdq` 下：
  - QDQ 输出 dtype = `float16`
  - 参数经 param quantizer cast 到 `float16`
  - wrapped 模块以 `float16` 执行
  - output quantizer QDQ 后保持 `float16`
- 默认 `fp32_qdq` 行为完全不变。

不变式：

- 模式状态读取必须每次 forward；不缓存。
- v1 与 v2 在相同 mode 下输出 dtype 一致。

## 5. API 签名

无新增 public 符号。

## 6. 算法 / 伪代码

`calculate_forward_pass`（STE）：

```text
x_round = round(tensor / delta) - offset
x_quant = clamp(x_round, zero, num_steps)
x_dequant = (x_quant + offset) * delta

if mode == FP16_QDQ:
    return x_dequant.to(torch.float16)
return x_dequant.to(orig_dtype)
```

`QcQuantizeWrapper.forward`：

```text
quantized_inputs = self._quantize_dequantize(self.input_quantizers, inputs)

for name, param in self._module_to_wrap.named_parameters():
    pq = self.param_quantizers[name]
    if pq.enabled:
        new_param = pq.quantize_dequantize(param)
        if mode == FP16_QDQ:
            new_param = new_param.to(torch.float16)
        setattr(self._module_to_wrap, name, torch.nn.Parameter(new_param, requires_grad=True))

if mode == FP16_QDQ:
    quantized_inputs = tuple(t.to(torch.float16) if torch.is_tensor(t) else t for t in quantized_inputs)

wrapped_output = self._module_to_wrap(*quantized_inputs, **kwargs)
output = self._quantize_dequantize(self.output_quantizers, wrapped_output)
```

边界：

- 部分 wrapped 模块（如 `BatchNorm` 在 CPU 上）对 half 支持有限：第一阶段允许在 CPU 上 raise `RuntimeError`，附带模块名提示。
- 非 tensor 类输入（int / list）保持原样。

## 7. 实施步骤

1. 在 `quantsim_straight_through_grad.py` import `get_quant_execution_mode`；修改 `calculate_forward_pass` 与 learned grid 函数返回值 cast。
2. `tensor_quantizer.py` 中 `quantize_dequantize` 与 `quantize` 输出按 mode cast。
3. `qc_quantize_op.py` 中 `QcQuantizeWrapper.forward` 完成输入 cast、参数 cast、调用、输出 QDQ。
4. 单元测试：
   - `test_v1_fp16_wrapper_output_dtype`
   - `test_v1_fp16_param_dtype`
   - `test_v1_default_unchanged`
5. 与 v2 对比测试：相同 tiny model + 相同输入，v1/v2 在 `fp16_qdq` 下端到端 cosine_similarity ≥ 0.999。

## 8. 验收标准

### 8.1 单元测试

```python
def test_v1_fp16_wrapper_output_dtype():
    wrapper = build_v1_wrapper(nn.Conv2d(3, 8, 3))
    x = torch.randn(1, 3, 8, 8, device="cuda")
    with quant_execution_mode("fp16_qdq"):
        y = wrapper(x)
    assert y.dtype == torch.float16

def test_v1_default_unchanged():
    wrapper = build_v1_wrapper(nn.Conv2d(3, 8, 3))
    x = torch.randn(1, 3, 8, 8)
    y = wrapper(x)
    assert y.dtype == torch.float32
```

### 8.2 必须通过的现有测试

- `tests/v1/` 默认路径全部通过。

### 8.3 性能阈值

- 与 spec 02 一致。

## 9. 不允许做的事

- 不允许修改 v1 公共 API。
- 不允许在 wrapper 内部隐式 fallback 到 fp32。
- 不允许修改 `QcQuantizeWrapper` 的属性结构（`input_quantizers` / `param_quantizers` / `output_quantizers` 命名与类型保持不变）。

## 10. 参考

- 现状代码：`qc_quantize_op.py` wrapper forward；`quantsim_straight_through_grad.py` 第 130-137 行 STE QDQ。
- spec 02（v2 等价实现）。

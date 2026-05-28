# 11 int16_fixed_qat_sim 反向与 STE

Status: implemented

## 1. 目标

为 `int16_fixed_qat_sim` 模式提供可训练的反向路径。前向模拟硬件整数行为；反向通过 STE / surrogate gradient 在浮点域反传，使模型可在 QAT 下学习适配 INT16 + multiplier/rshift 量化误差。

## 2. 范围

### 2.1 新增文件

- [aimet_torch/fixed_point/qat/__init__.py](../../aimet_torch/fixed_point/qat/__init__.py)
- [aimet_torch/fixed_point/qat/ste.py](../../aimet_torch/fixed_point/qat/ste.py)
- [aimet_torch/fixed_point/qat/surrogate.py](../../aimet_torch/fixed_point/qat/surrogate.py)

### 2.2 修改文件

- spec 07 / 08 / 09 中 kernel 在 `int16_fixed_qat_sim` 模式下走 STE 包装路径。

### 2.3 不在范围

- 训练循环本身（用户负责）。
- 优化器特殊处理（暂用普通 SGD/AdamW）。

## 3. 前置依赖

- spec 04 / 05 / 06 / 07 / 10 完成。

## 4. 数据契约

模式区分：

- `int16_fixed_eval`：前向调用 fixed kernel；输入/输出/参数全为整数；不支持 backward。
- `int16_fixed_qat_sim`：前向语义与 eval 一致（bit-exact 同样的整数运算），但所有不可导节点（round / clamp / shift / saturate / LUT）通过 STE / surrogate gradient 注入可导反向。

不变式：

- 反向梯度的 dtype 为 fp32（默认）或 fp16（用户配置）。
- 不破坏前向 bit-exact 性质：eval 与 qat_sim 在相同输入与参数下，前向输出必须 bit-exact 一致。
- STE 的实现使用 `torch.autograd.Function`；前向产出整数，反向直通梯度到对应输入。

## 5. API 签名

```python
# qat/ste.py
class FakeQuantInt16STE(torch.autograd.Function):
    """
    forward: 调用 INT16 量化（round + saturate）
    backward: 直通 + clamp gradient mask
    """
    @staticmethod
    def forward(ctx, x_float, scale, zero_point, qmin, qmax) -> torch.Tensor: ...

    @staticmethod
    def backward(ctx, grad_output): ...

class RoundSTE(torch.autograd.Function): ...
class RShiftSTE(torch.autograd.Function): ...
class SaturateSTE(torch.autograd.Function): ...
class LookupLutSurrogate(torch.autograd.Function):
    """
    forward: LUT 整数查表
    backward: 真函数（torch.sigmoid / torch.tanh）的解析梯度作为 surrogate
    """
    ...
```

```python
# qat/surrogate.py
def fake_quantize_int16_qat(
    x_float: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    qmin: int = -32768,
    qmax: int = 32767,
) -> torch.Tensor:
    """
    QAT 训练入口：内部走 FakeQuantInt16STE。
    返回浮点张量（值落在量化网格上），方便后续浮点反向。
    """
    ...
```

## 6. 算法 / 伪代码

### 6.1 量化输入与参数

```text
class FakeQuantInt16STE.forward:
    x_q = round(x_float / scale + zero_point)
    x_q = clamp(x_q, qmin, qmax)
    keep = (x_float / scale + zero_point) is in [qmin, qmax]   # 用于 backward mask
    ctx.save(keep)
    return ((x_q - zero_point) * scale)

class FakeQuantInt16STE.backward:
    keep, = ctx.saved
    grad_x = grad_output * keep
    return grad_x, None, None, None, None
```

### 6.2 kernel 适配

`int16_fixed_qat_sim` 下，kernel 不再直接接受 `Int16QuantizedTensor`，而是接受浮点张量并在内部通过 `FakeQuantInt16STE` 模拟整数化。这样反向可以打通。

```text
def conv2d_kernel_qat_sim(x_float, w_float, bias_float, encodings):
    # 输入 / 参数：fake quant
    x_sim = FakeQuantInt16STE.apply(x_float, x_scale, x_zp, ...)
    w_sim = FakeQuantInt16STE.apply(w_float, w_scale, w_zp, ...)

    # 浮点 conv，但数学结果与整数 conv 等价（因为输入已在整数网格上）
    y_real = conv2d(x_sim, w_sim, bias_float)

    # 模拟整数 requantize：在 float 上施加同样的 round / shift / saturate
    y_sim = SimulatedRequantizeSTE.apply(y_real, multiplier, rshift, y_scale, y_zp, qmin, qmax)
    return y_sim
```

`SimulatedRequantizeSTE` 的前向必须与 spec 05 的整数 `requantize_int` 数学等价（bit-exact）：

```text
forward:
    acc_int32 = round(y_real / (x_scale * w_scale))
    y_int = requantize_int(acc_int32, multiplier, rshift, y_zp, qmin, qmax)
    return ((y_int - y_zp) * y_scale).to(float32)

backward:
    grad_y_real = grad_output * keep_mask
```

### 6.3 LUT surrogate

```text
class LookupLutSurrogate(torch.autograd.Function).forward:
    y_int = lookup_lut(x_int16, lut_int16, index_shift)
    ctx.save(x_float)
    return ((y_int - y_zp) * y_scale).to(float32)

class LookupLutSurrogate.backward:
    x_float, = ctx.saved
    # 用真函数的解析梯度作为 surrogate
    grad_x = grad_output * sigmoid_derivative(x_float)
    return grad_x, None, None, None
```

## 7. 实施步骤

1. 实现 `FakeQuantInt16STE` / `RoundSTE` / `SaturateSTE`。
2. 实现 `SimulatedRequantizeSTE`，前向调用 spec 05 的 `requantize_int`。
3. 实现 `LookupLutSurrogate`，反向使用解析梯度。
4. 修改 spec 07 / 08 / 09 kernel：在 `int16_fixed_qat_sim` 模式分流到 fake quant 浮点路径。
5. 单元测试：
   - eval vs qat_sim 前向 bit-exact 一致
   - 反向梯度数值合理（与 fp32 fake quant 相比 cosine similarity ≥ 0.99）
   - 简单 QAT 训练 loss 可下降

## 8. 验收标准

### 8.1 单元测试

```python
def test_eval_and_qat_sim_forward_bitexact():
    model = build_tiny_int16_model()
    x = build_input()
    with quant_execution_mode("int16_fixed_eval"):
        y_eval = model(x)
    with quant_execution_mode("int16_fixed_qat_sim"):
        y_qat = model(x)
    assert torch.allclose(y_eval.to_float(), y_qat, atol=0)

def test_qat_sim_backward_runs():
    model = build_tiny_int16_model()
    x = build_input(requires_grad=True)
    with quant_execution_mode("int16_fixed_qat_sim"):
        y = model(x)
        y.sum().backward()
    assert x.grad is not None
```

### 8.2 必须通过的现有测试

- 默认模式 QAT 路径不受影响。

### 8.3 性能阈值

- `int16_fixed_qat_sim` 单步耗时 ≤ `fp32_qdq` QAT 的 5×（Tier C）。

## 9. 不允许做的事

- 不允许 qat_sim 前向数值与 eval 不一致。
- 不允许 LUT 反向用数值微分（必须解析）。
- 不允许跳过 saturate mask（否则梯度泄漏）。
- 不允许 backward 中调用整数 kernel（PyTorch 整数张量无 grad）。

## 10. 参考

- 顶层文档 ADR-006。
- 经典 STE：Bengio et al. "Estimating or Propagating Gradients Through Stochastic Neurons"。
- TFLite QAT 文档。
- `quant-gru-pytorch-main` 的 forward kernels 与 surrogate 思想。

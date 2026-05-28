# 08 Eltwise / Pool / Shape ops INT16 kernel

Status: implemented

> **硬件对齐（ADR-015）**：Add/Sub/Mul 与 AvgPool/Mean 累加在 ``AIMET_RX_ACC_INT32_SAT=1`` 或 ``AIMET_RX_HW_REF=1`` 下经 int64 运算后 ``saturate_int32`` / ``saturate_mac_accumulator``（与 spec 07 一致）。MaxPool 为比较操作，无 MAC 饱和。

## 1. 目标

实现 Add / Sub / Mul / ReLU / Clamp / MaxPool / AvgPool / AdaptiveAvgPool / Concat / Reshape / Flatten / Transpose / Permute 的 INT16 定点 kernel。

## 2. 范围

### 2.1 新增文件

- [aimet_torch/fixed_point/kernels/eltwise.py](../../aimet_torch/fixed_point/kernels/eltwise.py)
- [aimet_torch/fixed_point/kernels/pool.py](../../aimet_torch/fixed_point/kernels/pool.py)
- [aimet_torch/fixed_point/kernels/shape_ops.py](../../aimet_torch/fixed_point/kernels/shape_ops.py)

### 2.2 不在范围

- 非线性激活（spec 09）。
- BatchNorm（应在准备阶段折叠到 Conv）。

## 3. 前置依赖

- spec 04 / 05 / 06 完成。

## 4. 数据契约

按 op 区分，下方逐项说明。

## 5. API 签名

每个 kernel 以 `register_fixed_kernel(module_type)` 注册：

```python
@register_fixed_kernel(nn.ReLU)
class ReLUInt16Kernel:
    module_type = nn.ReLU
    def __call__(self, inputs, params, output_encoding, extra) -> Int16QuantizedTensor: ...
```

覆盖类型清单：

- `aimet_torch.elementwise_ops.Add` / `Subtract` / `Multiply`
- `nn.ReLU` / `nn.ReLU6` / `nn.Hardtanh` / `nn.Identity`
- `nn.MaxPool1d` / `2d` / `3d`
- `nn.AvgPool1d` / `2d` / `3d` / `nn.AdaptiveAvgPool2d`
- `aimet_torch.elementwise_ops.Concat`
- `nn.Flatten` / `nn.Identity`（shape ops 通常无独立 module，按 module 形态注册）

## 6. 算法 / 伪代码

### 6.1 Add

```text
x1, x2 = inputs[0].int_repr.to(int32), inputs[1].int_repr.to(int32)
x1_zp = inputs[0].zero_point.to(int32)
x2_zp = inputs[1].zero_point.to(int32)

# 离线已生成两路 multiplier / rshift 用于把 x1/x2 对齐到 output scale
m1, s1 = output_encoding.extra["m1_int16"], output_encoding.extra["s1_int8"]
m2, s2 = output_encoding.extra["m2_int16"], output_encoding.extra["s2_int8"]

x1_aligned = requantize_int(x1 - x1_zp, m1, s1, 0)
x2_aligned = requantize_int(x2 - x2_zp, m2, s2, 0)
y = x1_aligned + x2_aligned + output_encoding.zero_point
return Int16QuantizedTensor(int_repr=saturate_sim_tensor(y, qmin, qmax), ...)
```

注意：`extra` 中的两路 multiplier 由离线 pipeline 生成（spec 10 扩展支持）。

### 6.2 Mul

```text
x1, x2 = inputs[0].int_repr.to(int32), inputs[1].int_repr.to(int32)
acc_int32 = (x1 - x1_zp) * (x2 - x2_zp)
y_int16 = requantize_int(acc_int32, output_encoding.multiplier, output_encoding.rshift,
                         output_encoding.zero_point, output_encoding.qmin, output_encoding.qmax)
return Int16QuantizedTensor(...)
```

### 6.3 ReLU / Clamp

```text
relu_min_int = round(0.0 / output_encoding.scale) + output_encoding.zero_point
y = clamp(inputs[0].int_repr, relu_min_int, output_encoding.qmax)
return Int16QuantizedTensor(int_repr=y, ...)
```

`relu_min_int` 必须由离线生成或在 kernel 调用前缓存；运行时不允许除法。

### 6.4 MaxPool

```text
y = max_pool_int(inputs[0].int_repr, kernel_size, stride, padding)
return Int16QuantizedTensor(int_repr=y, scale=inputs[0].scale, zero_point=inputs[0].zero_point, ...)
```

输出 scale = 输入 scale，无需 requantize。

### 6.5 AvgPool

```text
acc_int32 = sum_window(inputs[0].int_repr.to(int32), kernel_size)
# 1/kernel_size 已离线转换为 multiplier + rshift（写入 output_encoding）
y_int16 = requantize_int(acc_int32, output_encoding.multiplier, output_encoding.rshift,
                         output_encoding.zero_point, output_encoding.qmin, output_encoding.qmax)
return ...
```

### 6.6 Concat

要求所有输入 encoding 与 output encoding 一致：

```text
for i, inp in enumerate(inputs):
    if not encoding_equal(inp, output_encoding):
        # 离线已生成对齐 multiplier / rshift
        m_i, s_i = output_encoding.extra[f"m{i}"], output_encoding.extra[f"s{i}"]
        inp = requantize_int(inp.int_repr - inp.zero_point, m_i, s_i, output_encoding.zero_point)
y = cat([inp.int_repr for inp in aligned_inputs], dim=axis)
return Int16QuantizedTensor(int_repr=y, ...)
```

### 6.7 Shape ops

```text
y = reshape(inputs[0].int_repr, new_shape)   # / flatten / transpose / permute
return Int16QuantizedTensor(int_repr=y, scale=inputs[0].scale, zero_point=inputs[0].zero_point, ...)
```

## 7. 实施步骤

1. 实现 `eltwise.py` 中的 Add / Sub / Mul / ReLU / Clamp。
2. 实现 `pool.py` 中的 Max/AvgPool 三维变体。
3. 实现 `shape_ops.py` 中的 Reshape / Flatten / Transpose / Permute / Concat。
4. 把这些注册写入 `kernels/__init__.py`，确保 import 自动触发。
5. 单元测试覆盖每个 op 至少一个 golden case。

## 8. 验收标准

### 8.1 单元测试

```python
def test_relu_int16_clamps_below_zero():
    x = build_int16_tensor([-100, 0, 100], scale=0.01, zp=0)
    out_enc = build_output_encoding(scale=0.01, zp=0, ...)
    y = get_fixed_kernel(nn.ReLU)([x], {}, out_enc, {})
    assert y.int_repr.tolist() == [0, 0, 100]

def test_concat_requires_aligned_encoding():
    ...

def test_avgpool_uses_multiplier_shift():
    ...
```

### 8.2 必须通过的现有测试

- 默认模式 op 行为不变。

### 8.3 性能阈值

- Tier A 标准。

## 9. 不允许做的事

- 不允许调用 `nn.functional.avg_pool2d(float)`。
- 不允许在 kernel 中把 int 临时转 float 求 mean。
- 不允许 Concat 在 kernel 内静默 requantize 而不消费离线参数。

## 10. 参考

- spec 05 requantize_int。
- spec 10 离线参数（Add / Concat / AvgPool 多路 multiplier 生成）。

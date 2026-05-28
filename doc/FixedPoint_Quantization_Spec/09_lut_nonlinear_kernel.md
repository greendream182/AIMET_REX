# 09 LUT 非线性 kernel

Status: implemented

> **硬件对齐（ADR-015）**：生产路径为 `evaluate_pwl_lut_int16` + `extra['pwl_lut']`（16 段 PWL，与 [PE_BxC_LUT_Design](../../abc_lut-shuai/lut_int_po2/docs/PE_BxC_LUT_Design(1).md) 一致）。离线拟合使用 `(m_int16, rshift)` 反演的 effective scale。上游 op 与 LUT 拟合 scale 不一致时，运行时通过 `align_op_quant_grid_to_lut_quant_grid`（§3.0）对齐。
>
> **仿真精度开关（默认关，保 e2e）**：
> - `AIMET_RX_PWL_HW_MAC_SAT=1`：乘积 INT32 饱和（INT16 中心化操作数）。
> - `AIMET_RX_PWL_HW_REF=1` 或 `AIMET_RX_HW_REF=1`：完整 `lut_int_general` tap 点（乘/移位/加后 INT32 饱和 + half-up 右移）；§3.0 对齐用 `HALF_UP`；与 abc **bit-exact**，见 `test_lut_abc_reference.py`。
> - `AIMET_RX_REQUANTIZE_INT32_SAT=1`：全局 `requantize_int` 乘后 INT32 饱和（spec 05）。

## 1. 目标

为 INT16 定点路径提供 Sigmoid / Tanh / Softmax 等非线性函数的查表（LUT）实现，运行时全程整数。

## 2. 范围

### 2.1 新增文件

- [aimet_torch/fixed_point/kernels/lut.py](../../aimet_torch/fixed_point/kernels/lut.py)
- [aimet_torch/fixed_point/offline/lut_gen.py](../../aimet_torch/fixed_point/offline/lut_gen.py)

### 2.2 不在范围 / 遗留路径

- **生产路径**：16 段 PWL（`extra['pwl_lut']`），见 ADR-015。
- **遗留路径**：均匀 `lookup_lut_int16` + `extra['lut_int16']`（单测/调试，adapter 不注入）。
- 多项式近似不在范围。

## 3. 前置依赖

- spec 04 / 05 / 06 完成。

## 4. 数据契约

LUT 表存储：

- dtype `torch.int16`
- shape `(table_size,)`，默认 256
- 索引方式：把 `int16` 输入映射到 `[0, table_size)` 区间

每个 op（生产路径均为 16 段 `pwl_lut`）：

- Sigmoid / Tanh / GELU / SiLU / Mish / Softplus / Hardsigmoid / Hardswish / LeakyReLU / PReLU：`_LutInt16Kernel` + `generate_pwl_lut_for_export`
- Softmax：`SoftmaxInt16Kernel`（PWL `exp` + 整数归一化；`legacy_float_softmax` 仅调试）

## 5. API 签名

```python
# aimet_torch/fixed_point/kernels/lut.py
@register_fixed_kernel(nn.Sigmoid)
class SigmoidInt16Kernel:
    module_type = nn.Sigmoid
    def __call__(self, inputs, params, output_encoding, extra) -> Int16QuantizedTensor: ...

@register_fixed_kernel(nn.Tanh)
class TanhInt16Kernel:
    module_type = nn.Tanh
    def __call__(self, inputs, params, output_encoding, extra) -> Int16QuantizedTensor: ...
```

`extra`（生产）必须包含：

- `pwl_lut`：`thresholds` / `q_b` / `n_bx_total` / `term_c` / `input_zero_point` / `output_qmin` / `output_qmax`
- `pwl_input_encoding`：可选；与上游 op 网格不一致时由 kernel 调用 `align_op_quant_grid_to_lut_quant_grid`

`extra`（遗留单表）：

- `lut_int16`：一维 `torch.int16`；`input_qmin` / `input_qmax` 映射索引（见 `lookup_lut_int16`）

## 6. 算法 / 伪代码

**运行时 PWL（生产）**：

```text
segment = searchsorted(thresholds, q_x, right=True) - 1   # 等价 PE: max i where q_x >= s[i]
centered = q_x - zp_x
prod = q_b[segment] * centered          # 可选 AIMET_RX_PWL_HW_MAC_SAT=1: INT32 乘积饱和
term_bx = round_shift(prod, n_bx[segment], HALF_AWAY_FROM_ZERO)  # n_bx<0 则左移
q_y = saturate(term_bx + term_c[segment], output_qmin, output_qmax)
```

**离线**：`generate_pwl_lut` / `generate_pwl_lut_for_export`（16 段、effective scale）；JSON 导出含 `fmin`/`fmax` 以兼容 `lut_int_general.infer_with_lut`。

**遗留单表**：`lookup_lut_int16` + `generate_lut_int16`（均匀索引，非 adapter 默认）。

## 7. 实施步骤

1. 实现 `generate_lut_int16` 离线函数。
2. 实现 `lookup_lut` 运行时整数索引。
3. 注册 Sigmoid / Tanh kernel。
4. 单元测试 LUT 与浮点 reference 误差 < 1.5 LSB。
5. Softmax：PWL `exp` + 整数 sum（见 `softmax_int16_pwl`）。
6. 与 `lut_int_general` 对齐：见 `tests/fixed_point/kernels/test_lut_abc_reference.py`。

## 8. 验收标准

### 8.1 单元测试

```python
def test_sigmoid_lut_matches_reference_within_lsb():
    x_float = torch.linspace(-6, 6, 1000)
    x_int16 = quantize(x_float, scale=12/32768, zp=0)
    lut = generate_lut_int16(torch.sigmoid, in_enc, out_enc, table_size=256)
    y = SigmoidInt16Kernel()([x_int16_tensor], {}, out_enc, {"lut_int16": lut, "index_shift": 8})
    y_float = y.to_float()
    err = (y_float - torch.sigmoid(x_float)).abs().max()
    assert err < out_enc.scale * 1.5  # 不超过 1.5 LSB
```

### 8.2 必须通过的现有测试

- 默认模式 Sigmoid / Tanh 行为不变。

### 8.3 性能阈值

- LUT lookup 耗时 ≤ Conv2d 的 1%。

## 9. 不允许做的事

- 不允许在 runtime 调用 `torch.sigmoid(...)` 或 `torch.tanh(...)`。
- 不允许在 runtime 重新生成 LUT。
- 不允许 LUT 表 dtype 为 float。

## 10. 参考

- `quant-gru-pytorch-main/include/quantize_ops_helper.h` 中 `LUT` 与 `clamp_by_bitwidth`。
- 顶层文档 ADR-007。
- abc 函数覆盖对照：[09a_lut_coverage_vs_abc.md](09a_lut_coverage_vs_abc.md)（对照 `abc_lut-shuai/lut_int_po2/docs/LUT_NONLINEAR_FUNCTIONS.md`）。

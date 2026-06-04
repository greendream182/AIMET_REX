# PE General-Scale LUT / CLZ_LUT 二进制存储格式

本文定义当量化 scale 是任意正浮点数时，PE 侧 LUT/CLZ_LUT 的参数计算与二进制存储约定。

本文以硬件给出的 general_lut 二进制 **head 存储规约**为准；`lut` 数据区的逐段参数格式等待硬件进一步确认后补充。

重要说明：硬件给出的表格是**高地址在左、低地址在右**。本文统一改写为软件/文档更常用的 **低地址到高地址** 顺序：

```text
槽位 0 = 最低地址 = 硬件原表最右侧的有效参数
槽位 10 = 最高地址 = 硬件原表最左侧的有效参数
```

硬件原表每行最右侧的 `lut(...)` 用作表类型/后续 LUT 数据区标记，不计入下面 11 个 head 槽位。

---

## 1. General-Scale 的核心变化

po2 版只保存一个 shift：

```text
scale = 2^-n
```

General 版把 scale 表示为 multiplier + shift：

```text
scale ≈ M / 2^n_s
```

硬件执行缩放时使用：

```text
scaled = round_shift(x * M, n_s)
```

因此 general_lut 的 head 里会出现多组：

```text
q_*      # multiplier
n_*      # shift
```

例如：

```text
q_r / n_r              # 普通 LUT 输入 scale adapter
q_inv_2pi / n_inv_2pi  # 周期折叠中除以 2π 的近似
q_norm / n_norm        # CLZ: m_int -> q_m
r_q / r_shift          # CLZ 后处理反归一化
```

---

## 2. Head 槽位总览（低地址到高地址）

硬件给出的 head 表按 11 个槽位组织。本文按**低地址到高地址**排列后，槽位位宽如下：

| 槽位 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 位宽 | 16 | 16 | 16 | 16 | 16 | 16 | 8 | 8 | 8 | 16 | 16 |

不同函数族复用这 11 个槽位，但每个槽位的语义不同。

---

## 3. 饱和类和特殊类 LUT Head

适用函数：

```text
sigmoid, tanh, exp/exponential, softplus, silu/swish, gelu, mish,
relu6, hard_sigmoid, hard_swish, leaky_relu, prelu, power_0_3 等。
```

硬件原表从左到右是高地址到低地址；下表已按低地址到高地址重排：

| 槽位 | 位宽 bit | 参数名 | 说明 |
|---|---:|---|---|
| 0 | 16 | `input.zp` | 输入零点 |
| 1 | 16 | `q_r` | 输入 scale adapter 的 multiplier |
| 2 | 16 | reserved | 未使用，写 0 |
| 3 | 16 | `output_min` | 输出饱和下界 |
| 4 | 16 | `out_max` | 输出饱和上界 |
| 5 | 16 | reserved | 未使用，写 0 |
| 6 | 8 | `inl_accu_width` | 内部累加器位宽 |
| 7 | 8 | `n_r` | 输入 scale adapter 的 shift |
| 8 | 8 | reserved | 未使用，写 0 |
| 9 | 16 | reserved | 未使用，写 0 |
| 10 | 16 | reserved | 未使用，写 0 |

对应计算：

```text
x_offset_op = q_x_op - zp_op
x_offset_lut = round_shift(x_offset_op * q_r, n_r)
q_x_lut = x_offset_lut + input.zp
```

---

## 4. 周期类 LUT Head

适用函数：

```text
sin, cos
```

`cos` 默认复用 `sin` 表。

硬件原表从左到右是高地址到低地址；下表已按低地址到高地址重排：

| 槽位 | 位宽 bit | 参数名 | 说明 |
|---|---:|---|---|
| 0 | 16 | `input.zp` | 输入零点 |
| 1 | 16 | `q_r` | 输入 scale adapter 的 multiplier |
| 2 | 16 | `q_inv_2pi` | `1 / Q_2PI` 乘倒数近似的 multiplier |
| 3 | 16 | `output_min` | 输出饱和下界 |
| 4 | 16 | `out_max` | 输出饱和上界 |
| 5 | 16 | `q_2pi` | `2π` 周期常量 |
| 6 | 8 | `inl_accu_width` | 内部累加器位宽 |
| 7 | 8 | `n_r` | 输入 scale adapter 的 shift |
| 8 | 8 | `n_inv_2pi` | `1 / Q_2PI` 乘倒数近似的 shift |
| 9 | 16 | `q_halfpi` | `π/2` 相位偏移常量，用于 cos 复用 sin |
| 10 | 16 | reserved | 未使用，写 0 |

周期折叠：

```text
# cos 分支
q_x_phase = q_x + q_halfpi

# k ≈ round(q_x_phase / q_2pi)
k = round_shift(q_x_phase * q_inv_2pi, n_inv_2pi)
q_fold = q_x_phase - k * q_2pi

# 再做输入 scale adapter
x_offset_lut = round_shift((q_fold - zp_op) * q_r, n_r)
q_x_lut = x_offset_lut + input.zp
```

---

## 5. CLZ 类 LUT Head

适用函数：

```text
reciprocal, sqrt, rsqrt, power_2, log
```

硬件原表从左到右是高地址到低地址；下表已按低地址到高地址重排：

| 槽位 | 位宽 bit | 参数名 | 说明 |
|---|---:|---|---|
| 0 | 16 | `input.zp` | 原始输入零点 |
| 1 | 16 | `q_norm` | `m_int -> q_m` 的 multiplier |
| 2 | 16 | `r_q` | CLZ 后处理 R 的 multiplier |
| 3 | 16 | `output_min` | 输出饱和下界 |
| 4 | 16 | `out_max` | 输出饱和上界 |
| 5 | 16 | `q_ln2` | `ln(2)` 在输出空间的常量，log 使用 |
| 6 | 8 | `inl_accu_width` | 内部累加器位宽 |
| 7 | 8 | `n_norm` | `m_int -> q_m` 的 shift |
| 8 | 8 | `r_shift` | CLZ 后处理 R 的 shift |
| 9 | 16 | `q_ln_sx` | `ln(s_x)` 在输出空间的常量，log 使用 |
| 10 | 16 | `output_zp` | 最终输出零点 |

CLZ 前处理：

```text
x_offset = q_x - input.zp
m_int, e_offset = clz_normalize(x_offset)

q_m = round_shift(m_int * q_norm, n_norm)
```

CLZ 后处理：

```text
y_offset = q_y_norm - z_y_norm
val = y_offset * r_q
val = round_shift(val, r_shift_adjusted_by_function)
q_y = val + output_zp
```

其中 `r_shift_adjusted_by_function` 由 `reciprocal/sqrt/rsqrt/power_2/log` 的恢复公式决定。

log 特有：

```text
q_y = round_shift(y_offset * r_q, r_shift) + E * q_ln2 + q_ln_sx + output_zp
```

---

## 6. LUT 数据区与 head_lut 边界区

硬件补充的 LUT 数据区分为两部分：

- `lut`：需要通过 `id` 查表，存放每段 Bx+C 计算参数。
- `head_lut`：不需要通过 `id` 查表，存放段边界 `threshold_quantized[i]`。

### 6.1 LUT 区域

| 区域 | 函数类型 | 参数 | 位宽 | 总位宽 | 偏移地址 | 说明 |
|---|---|---|---:|---:|---|---|
| `lut` | 都需要 | `term_c_precomputed[i]`，16 个 | 每个 32 bit | 512 bit | `0x00` | 每段截距项 |
| `lut` | 都需要 | `n_bx_total[i]`，16 个 | 每个 8 bit | 128 bit | `0x40` | 每段融合移位 |
| `lut` | 都需要 | `q_b[i]`，16 个 | 每个 16 bit | 256 bit | `0x50` | 每段斜率 |

地址规则：

```text
term_c_precomputed_addr(i) = 0x00 + 4 * i
n_bx_total_addr(i)         = 0x40 + i
q_b_addr(i)                = 0x50 + 2 * i
```

覆盖范围：

```text
0x00 .. 0x3f    term_c_precomputed[16]
0x40 .. 0x4f    n_bx_total[16]
0x50 .. 0x6f    q_b[16]
```

### 6.2 head_lut 边界区

| 区域 | 函数类型 | 参数 | 位宽 | 总位宽 | 偏移地址 | 说明 |
|---|---|---|---:|---:|---|---|
| `head_lut` | 边界（都需要） | `threshold_quantized[i]`，16 个 | 每个 17 bit | 272 bit | `0x70` | 16 个段起始边界，连续 bit-pack |

`threshold_quantized[i]` 从 `0x70` 开始连续 bit-pack：

```text
threshold_base_bit = 0x70 * 8
threshold_i_bit    = threshold_base_bit + 17 * i
threshold_i_width  = 17
```

16 个 17-bit 阈值共占：

```text
16 * 17 bit = 272 bit = 34 Bytes
```

覆盖范围：

```text
0x70 .. 0x91    threshold_quantized[16]
```

段查找语义：

```text
segment_id = max{i | q_input >= threshold_quantized[i]}
```

### 6.3 查表说明

```text
lut      需要 id 去查表
head_lut 不需要 id 去查表
```

也就是说，B×C 参数数组 `term_c_precomputed/n_bx_total/q_b` 通过 `lut id` 选择；段边界 `threshold_quantized` 位于 `head_lut`，不走 `lut id` 查表路径。

---

## 7. 与 po2 版存储的关键差异

| 项目 | po2 LUT | general LUT |
|---|---|---|
| 普通输入 scale adapter | `n_diff` | `q_r / n_r` |
| 周期折叠除以 `2π` | 可用除法或 po2 近似 | `q_inv_2pi / n_inv_2pi` |
| CLZ `m_int -> q_m` | `shift_m` | `q_norm / n_norm` |
| CLZ 后处理 | `scale_shift` | `r_q / r_shift` |
| log 额外项 | `Q_LN2` | `q_ln2 / q_ln_sx` |
| Bx+C LUT 数据区 | 旧实现可能按 segment AoS | `lut` 区域按数组存储，`head_lut` 单独存边界 |

---

## 8. 验收与 MRNN 集成注意点

> 详细对照见 `aimet_rx-main/doc/MRNN_SingleOp_LUT_Validation.md`。

### 8.1 Head 字段必查项

| 函数族 | 必查槽位 | 常见 MRNN 问题 |
|--------|----------|----------------|
| 饱和类 | `output_min` / `out_max` (3–4) | square 输出饱和在 127，需扩 `out_max` 或升 bitwidth |
| CLZ 类 | `q_norm/n_norm`, `r_q/r_shift` (1–2, 7–8) | sqrt 在 Q_in 上 cos≥0.999，但 vs float 可低 |
| CLZ 类 | `output_min/out_max` (3–4) | 正域函数：`x_offset≤0` 须走特殊路径，勿进 CLZ |
| `power_2` / square | `out_max` (4) | 须按 **输入 peak²** 标定，勿沿用 x 的 8-bit 127 上界 |
| `reciprocal` / div 分母 | `input.zp`, 正域 | MRNN `module_div` 分母=std；**unsigned + min≥eps** |
| 全部 | `input.zp` (0) | div 分母 Q→0 → reciprocal 零值路径 |

### 8.2 不在 LUT head 内的 op

`sign` / 纯整数 `mul` / `add` / `sub` **不占用** §3–§5 的 LUT head 模板。其验收：

- **禁止**仅用 cos vs `float_native` 判 FAIL；
- `sign` 用 **sign_agreement(Q_in)**；
- 双输入 `div` 须同时校验两路 input quantizer。

### 8.3 Po2 导出到 general head

Po2 workflow 后 `scale = 2^-n`，可映射为 `q=1, n=n` 写入 `q_r/n_r` 或 CLZ 的 `r_q/r_shift`。导出工具应保证与 `convert_encodings_to_fixed_scale` 的 `(M,r)` 一致，避免 head 与 runtime QDQ 漂移。


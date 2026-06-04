# Int-LUT 设计与复用（General-Scale 版）

> **姊妹文档**：`lut/doc/lut_int_design_po2.md`（Power-of-Two 版）
>
> 本文档与 `lut_int_design_0513_26.md` **章节一一对应**，把里面所有
> 「`scale = 2^-n`、所有缩放/反缩放只用 shift」的位置统一改造为：
>
> $$
> s \;\approx\; \frac{M}{2^{n_s}}\quad(\text{M 是大整数，}n_s\text{ 是算术右移位数})
> $$
>
> 缩放操作从「`x >> n`」改为「`(x * M + round) >> n_s`」（即 TFLite /
> ONNX-Runtime / AIMET v2 的 `quantize_multiplier` 配方）。除此之外
> 分类、段查找、JSON 字段、ROM 形态、四类函数的总体流程**全部保持
> 不变**，可以与 po2 版共用同一硬件 ROM 解析器，只在两处 flag 控制是否
> 走 multiplier 路径。

---

## 1. 基础知识

### 1.1 基本思想

```text
输入区间 [x_min, x_max]
  -> 量化到 int (scale = M_x / 2^{n_x},  zero_point)        # ← 通用 scale
  -> 划分为 N 段
  -> 每段拟合 y = b*x + c
       -> 合并 M = b · s_x / s_y           (real, 段独有)
       -> 合并 T = c / s_y + zp_y          (real, 段独有)
       -> M -> (q_b, n_bx_total)            (每段独立 best-shift)
       -> T -> term_c                       (饱和到 term_c 位宽)
  -> 推理时只做：
       1) 找段                               (int 比较)
       2) 读取 q_b, n_bx_total, term_c       (ROM 读)
       3) 定点乘加 + 算术 shift              (int 乘、加、移)
       4) 输出饱和到 int
       5) (可选) scale 反量化为 float        (一次 fixed-point mul + shift)
```

整体公式（与 po2 版**完全相同**）：

$$
q_y = \mathrm{SAT}\!\left(\big(q_b \cdot (q_x - zp_x)\big) \gg n_{bx\_total} + \mathrm{term\_c}\right)
$$

> **与 po2 版的关键差异**：在 po2 版里 `n_bx_total = n_b + n_x - n_y` 是
> 几个独立 shift 字段的复合，并依赖「`scale` 必须是 `2^-n`」的前提。
> 在本版里 `n_bx_total` 由**每段独立**的 `quantize_multiplier(M, bw_b)`
> 直接得到（见 §4），不再需要全局 `2^-n` 约束。

### 1.2 定点量化格式（General Quantizer）

- **`scale ≈ M / 2^{n_s}`**（M 为大整数，称为 *quantized multiplier*；
  `n_s` 称为 *shift*）→ 所有缩放/反缩放都用「一次 int 乘 + 一次算术
  shift」，没有浮点乘法。

`quantize_multiplier(s, bw_M)` 的标准实现：

```python
def quantize_multiplier(s: float, bw_M: int = 16) -> tuple[int, int]:
    """
    把任意正浮点 s 编成 (M, n_s)，满足 M / 2^{n_s} ≈ s
    且 |M| 几乎用满 bw_M-bit 有符号范围（高位归一化，最大化有效位）。
    """
    assert s > 0
    max_M = 2**(bw_M-1) - 1
    n_s   = floor(log2(max_M / s))   # 最大的 n_s 使 |s · 2^{n_s}| ≤ max_M
    M     = round(s * 2**n_s)
    return M, n_s
```

性质：

- `M ∈ [2^{bw_M-2}, 2^{bw_M-1})`，约保留 `bw_M-1` bit 有效精度
  （16-bit `M` ≈ 5 位十进制有效位，足够覆盖 LUT 系数级的实数）。
- `n_s` 在常用 16-bit 量化下大约落在 `[10, 30]`，硬件上用 5~6 bit 存。

通用乘移配方（**贯穿 §3 / §4 / §5**）：

```text
rounded_mul_shift(x, M, n_s):
    prod = x * M                              # int 乘 (data_bw × M_bw)
    if n_s > 0:
        return SAT( (prod + (1 << (n_s-1))) >> n_s )   # 算术右移带 round
    elif n_s == 0:
        return SAT( prod )
    else:
        return SAT( prod << (-n_s) )          # 算术左移
```

**与 po2 版对照**：

| 维度                       | po2 版               | **General 版（本版）**                                       |
| -------------------------- | -------------------- | ------------------------------------------------------------ |
| `scale` 形式               | 强制 `2^-n`          | 任意正浮点（一般是 `(fmax-fmin)/(qmax-qmin)`）               |
| 单个 quantizer 存储        | 1 个 `n`（int）      | `(M, n_s)`：一个大整数 + 一个 shift                          |
| 缩放操作                   | `x << n` 或 `x >> n` | `rounded_mul_shift(x, M, n_s)`                               |
| `prefer_larger_range` 开关 | 必需                 | **不需要**，scale 直接由 fmax/fmin 解析得到                  |
| 段内反量化                 | 全是 shift           | 推理段内仍可全是 shift（只需把 `M_b·M_x/M_y` 烘到段内 `(q_b, n_bx_total)`） |
| 跨 op 复用路径             | `n_diff` 一次 shift  | `(q_r, n_r)` 一次 multiplier+shift                           |

### 1.3 16-bit 硬件配置参考表（建表脚本 → 硬件实现）

定点 LUT 的所有位宽都是建表阶段就锁死的常数，**硬件按这张表分配
寄存器/ROM/MAC 位宽即可**。General 版相比 po2 版多了两类「multiplier」
字段（`M` 大整数 + `n_s` shift），均在 ROM 表头里：

| 参数                             | 含义                                                         | 非规格化函数<br>（饱和 / 周期 / 单独定制） | CLZ 规格化函数<br>（reciprocal / sqrt / rsqrt / log / power_2） | 硬件实体                             |
| -------------------------------- | ------------------------------------------------------------ | ------------------------------------------ | ------------------------------------------------------------ | ------------------------------------ |
| `NUM_SEGMENTS`                   | 段数                                                         | **16**                                     | **16**                                                       | 段比较器树（4 级）+ ROM 深度         |
| `INPUT_BIT_WIDTH`                | 全局输入/输出位宽                                            | **16**                                     | **16**                                                       | 输入端口、输出端口                   |
| `ZP_BIT_WIDTH`                   | 零点位宽                                                     | **16**                                     | **16**                                                       | 输入端口、输出端口                   |
| `INTERNAL_BIT_WIDTH`             | 累加器位宽<br>（`bx`, `term_bx`, `y_acc`, `prod` 等中间量）  | **32**                                     | **32**                                                       | MAC 累加器、SAT 单元                 |
| `MULTIPLIER_BIT_WIDTH`           | 通用 multiplier `M` 的位宽<br>（用于 §3.1.1.2 跨 op 适配、§3.4 CLZ 归一化、§5 反归一化） | **16**                                     | **16**                                                       | ROM 一列 + 32-bit 整数乘法器一份     |
| `SHIFT_BIT_WIDTH`                | 通用 shift `n_s` 的位宽                                      | **6**（带符号）                            | **6**（带符号）                                              | ROM 一列：`6 · K_multipliers` 比特   |
| `COEFF_B_BIT_WIDTH`              | 系数 `q_b` 的 ROM 宽度                                       | **16**                                     | **16**                                                       | ROM 一列：`bw_b · NUM_SEGMENTS` 比特 |
| `COEFF_C_BIT_WIDTH`              | 系数 `q_c` 的 ROM 宽度（中间量，烘焙到 `term_c` 后不再单独存） | **16**                                     | **16**                                                       | 仅建表时使用                         |
| `TERM_C_BIT_WIDTH`               | `term_c_precomputed` 的 ROM 宽度                             | **32**                                     | **32**                                                       | ROM 一列：`32 · NUM_SEGMENTS` 比特   |
| `CLZ_NORMALIZED_SPACE_BIT_WIDTH` | CLZ 归一化空间的 LUT 输入/输出位宽<br>（`m_int → q_m`、`q_y_norm` 的位宽） | — （N/A）                                  | **16**                                                       | CLZ 子模块内部数据通路               |
| `n_bx_total`                     | 段内 / CLZ 全表共用的移位常量（位宽见 §1.3.1）               | 段内 6-bit                                 | 段内 6-bit                                                   | 移位寄存器预设值                     |

**与建表脚本的对应关系（便于 RTL 团队对照）：**

```text
test_lut_generation.py (非规格化函数, General-Scale):
  NUM_SEGMENTS         = 16
  INPUT_BIT_WIDTH      = 16
  INTERNAL_BIT_WIDTH   = 32
  MULTIPLIER_BIT_WIDTH = 16          # ← 新增
  SHIFT_BIT_WIDTH      = 6           # ← 新增 (带符号)
  COEFF_B_BIT_WIDTH    = 16
  COEFF_C_BIT_WIDTH    = 16
  TERM_C_BIT_WIDTH     = 32

test_clz_integration.py (规格化函数, General-Scale):
  上述 8 项 + CLZ_NORMALIZED_SPACE_BIT_WIDTH = 16
```

> **新增字段 `MULTIPLIER_BIT_WIDTH` / `SHIFT_BIT_WIDTH` 的硬件含义**：
> 整张 LUT 表头存放若干 `(M, n_s)` 对——
> 跨 op 适配用 1 对（§3.1.1.2）、CLZ 归一化用 1 对（§3.4）、CLZ
> 反归一化的 `R` 用 1 对（§5.0）。所有 `M` 都共用同一份
> 16x32 整数乘法器；所有 `n_s` 都共用同一份带符号桶形移位器。
> 段内系数 `q_b`、`term_c` 的存储格式与 po2 版**完全相同**。

#### 1.3.2 8-bit 配置（**仅非规格化函数**）

```text
test_lut_generation.py (8-bit, 仅非规格化, General-Scale):
  NUM_SEGMENTS         = 16
  INPUT_BIT_WIDTH      = 8       # 输入/输出位宽
  INTERNAL_BIT_WIDTH   = 16      # 内部累加器位宽
  MULTIPLIER_BIT_WIDTH = 8       # 8-bit M，覆盖 ~6 bit 有效位
  SHIFT_BIT_WIDTH      = 5
  COEFF_B_BIT_WIDTH    = 8
  COEFF_C_BIT_WIDTH    = 8
  TERM_C_BIT_WIDTH     = 16
```

**重要约束**（与 po2 版一致 + 一条本版补充）：

1. **8-bit 配置只能用于非规格化函数**。规格化路径
   （`reciprocal / sqrt / rsqrt / log / power_2`）必须保留
   16-bit + 32-bit 累加器，理由见 po2 文档 §6.1。
2. **本版补充**：8-bit 配置下，`MULTIPLIER_BIT_WIDTH = 8` 时
   通用 multiplier 的精度只有 ~1.5 位十进制，**比 po2 单 shift 更糟**。
   若上游 op 的 `s_op` 与 `s_lut` 比值不在 `[0.5, 2)` 内，建议直接走
   po2 路径，General 在 8-bit 不划算。

---

## 2. 函数的四种分类

**与 po2 版完全一致**，General 版在分类层面没有调整：

### 2.1 饱和类（有限输入 → 直接建表）

| 函数           | 饱和行为                                |
| -------------- | --------------------------------------- |
| `sigmoid`      | 两端 → 0 / 1                            |
| `tanh`         | 两端 → ±1                               |
| `exp`          | 一端接近 0                              |
| `hard_sigmoid` | 两端 → 0 / 1，`min(max(0, x+3), 6) / 6` |

### 2.2 周期类（相位折叠 → 共享 sin 表）

- `sin`
- `cos`

### 2.3 规格化类（CLZ 输入归一化 → 恢复）

代表函数：

- `reciprocal` — `1/x`
- `sqrt` — `√x`
- `rsqrt` — `1/√x`
- `log` — `ln(x)`
- `power_2` — `x²`

### 2.4 单独定制类（无法复用）

- `leaky_relu` — `x if x≥0 else α·x`
- `prelu` — `leaky_relu` 的可学习版本（α 由网络学出）
- `power_0_3` — `|x|^0.3`

| 函数                                                 | 左端行为                   | 右端行为                      |
| ---------------------------------------------------- | -------------------------- | ----------------------------- |
| `softplus` = `log(1 + exp(x))`                       | → 0                        | → `x`                         |
| `silu` / `swish` = `x · sigmoid(x)`                  | → 0                        | → `x`                         |
| `gelu` ≈ `0.5 x (1 + tanh(√(2/π)(x + 0.044715 x³)))` | → 0                        | → `x`                         |
| `mish` = `x · tanh(softplus(x))`                     | → 0                        | → `x`                         |
| `hard_swish` = `x · hard_sigmoid(x)`                 | → 0（`x ≤ -3` 后严格为 0） | → `x`（`x ≥ 3` 后严格为 `x`） |

---

## 3. 四种函数的前处理（定点）

### 3.1 饱和类和周期类

### 3.1.1 痛点

#### 3.1.1.1 痛点1：输入类型 int / uint 不同导致多张表（>2）

**问题场景**：与 po2 版相同—— `term_c_precomputed` 在建表时已经把
`zero_point_x`、`zero_point_y` 融合进常量，zp 不一样表就不一样。

**设计：把 LUT 输入空间统一到 (bw + 1)-bit signed**

> **核心**：把 `bw`-bit unsigned 减去零点后转成 `(bw+1)`-bit signed
> 输入，硬件只保留一张 `(bw+1)`-bit signed 表。
>
> 该处理**与 scale 形式无关**，General 版不做改动。

#### 3.1.1.2 痛点：网络输入 scale ≠ LUT 输入 scale  ⭐ **本版主要差异**

**问题场景**：

LUT 的（`scale_lut, zp_lut`），由 LUT 自己的 `[x_min_lut, x_max_lut]`
决定。但实际部署时，**调用 LUT 的算子（上游 op）的输出 quantizer 是
quantsim 在网络上跑校准算出来的**，两者一般不一致。

**po2 版的解决**：因为 `s_op = 2^-n_op`、`s_lut = 2^-n_lut`，比值
`s_op / s_lut = 2^{n_lut - n_op}` 一定是 2 的幂，所以只要存一个
4-bit `n_diff = n_op - n_lut` 就能用一次算术 shift 完成适配。

**General 版的解决**：`s_op / s_lut` 是任意正浮点，纯 shift 不够用。
使用 §1.2 的通用配方烘焙：

$$
\frac{s_{op}}{s_{lut}} \;\approx\; \frac{q_r}{2^{n_r}}
\quad\text{(预烘焙在每个上游 op 的配置里)}
$$

存储：`(q_r : MULTIPLIER_BIT_WIDTH, n_r : SHIFT_BIT_WIDTH)`，
典型 (16 bit, 5~6 bit)。

转换流程（**1 次 int 减 + 1 次 int 乘 + 1 次带 round 的算术 shift

+ 1 次 int 加 + 1 次 clip**）：

```text
x_offset_op = q_x_op - zp_op                              # int 减
prod        = q_r * x_offset_op                           # int 乘 (16x16 → 32)
if n_r >= 0:
    x_offset_lut = (prod + (1 << (n_r-1))) >> n_r         # 算术右移带四舍五入
else:
    x_offset_lut = prod << (-n_r)                         # 算术左移
q_x_lut     = clip(x_offset_lut + zp_lut, q_min_lut, q_max_lut)
```

> 网络训练 / 校准完成后，`(q_r, n_r)` 是定值；
> 与 po2 版「n_diff = 0」时退化为 1 次 shift 自然衔接：
> `q_r ≈ 2^{bw_M-1}`、`n_r ≈ bw_M-1` 时整体接近恒等。


### 3.1.2 饱和类前处理

进行 3.1.1 的处理。

**`exp` 这类单边衰减函数的特殊处理 for softmax**（与 po2 完全一致）：

```text
if x = x_left_sat :  output = 0
else :               按 LUT 正常查
```

### 3.1.3 周期类前处理

先进行周期折叠：把任意 `q_x` 折叠到主周期
`[-π, π)` 对应的整数区间。再进行 3.1.1 的处理

```text
常数:
  Q_2PI    = round(2π / s_x)            # 2π   在 op 自身 scale 下的整数
  Q_HALFPI = round(π/2 / s_x)           # π/2  在 op 自身 scale 下的整数
                                        # (与 po2 一致，s_x 此时是任意浮点)

cos 分支:
  q_x' = q_x + Q_HALFPI                 # 相位平移（int 加法）

周期折叠（int 除法 + 乘法 + 减法）:
  k    = round_to_nearest(q_x' / Q_2PI) # 也可以用 (q_x' * Q_INV_2PI) >> n_inv 近似
  q_fold = q_x' - k * Q_2PI             # int 减
  # q_fold ∈ [-Q_2PI/2, +Q_2PI/2) ≈ [-π, π) 的整数表示

查共享 sin LUT:
  segment_id <- 在 q_fold 上找段
  q_y <- 按 §4 的定点乘加
```

> **顺序很关键**：周期类必须**先在 op 自身 scale 下折叠**，再做
> §3.1.1.2 的 multiplier+shift 适配进入 LUT 空间。否则 sin/cos 的
> 「非渐近常数」性质会让端点 clip 出灾难性误差。
>
> 对 `1 / Q_2PI` 的近似除法，General 版统一用 `(q_inv_2pi, n_inv_2pi)`
> 一对常量（同 §1.2 配方）替代，保持「无浮点」属性。

### 3.2 单独定制类前处理

将 `bw`-bit unsigned 减去零点后转成 `(bw+1)`-bit signed 输入，
硬件只保留一张 `(bw+1)`-bit signed 表。同 po2 版。

### 3.4 规格化类前处理（CLZ）

目标：把正整数 `x_offset` 分解为 `m_int · 2^e`，其中
`m_int ∈ [2^{bw-1}, 2^bw)`。

**核心：CLZ（Count Leading Zeros）指令**

硬件上仍然是两个操作（与 po2 完全一致）：

1. CLZ 指令 → `leading_zeros`（1 cycle）
2. 左移 → `m_int`（1 cycle）

**示例**（`bit_width = 8`）：

```text
x_offset = 100  (01100100)
bit_length = 7, leading_zeros = 1
m_int    = 100 << 1 = 200  (0b11001000)   ∈ [128, 256)   ✓
e_offset = 6
验证:   (200/256) · 2^(6+1) = 0.78125 · 128 = 100        ✓
```

**指数修正（General 版）**：

```text
E = e_offset + 1            # 2 进制指数，与 s_x 无关
```

> 对比 po2 版的 `exponent = e_offset + 1 - n_in`：po2 把
> `s_x = 2^-n_in` 直接吸进了 `n_in` 这个整数；General 版里 `s_x` 不是
> `2^-n_in`，所以 `s_x` 的影响**改成在 §5 的 `R` 常数里吸收**，
> `E` 单纯表示「mantissa 的 2 进制指数」。这一处归一化的「职责切分」
> 是本版唯一的语义改变。

**`m_int → q_m` 的对齐（每个采样点 1 次 int 乘 + 1 次带 round 的 shift）**：

po2 用的是「一次 shift」：`q_m = m_int >> (bw - n_norm)`。

General 版里 `s_m`（归一化输入 quantizer 的 scale）是任意浮点：

```text
# 表头烘焙 (整张表一份)
(q_norm, n_norm) = quantize_multiplier(1 / (s_m · 2^bw), MULTIPLIER_BIT_WIDTH)
                   # 含义：把 m_int 映射到 q_m 的整体倍率

# 推理时（每个采样点）
prod    = m_int * q_norm                        # int 乘
if n_norm >= 0:
    q_m = ((prod + (1 << (n_norm-1))) >> n_norm)
else:
    q_m =  (prod << (-n_norm)) 
```

> `q_m` 永远是正值（`m ∈ [0.5, 1)`），归一化 LUT 输入空间统一用
> symmetric / unsigned 表，`zp_m ≡ 0`，故公式里**不含**零点项——
> 与 po2 版 `q_m = m_int >> (bw - n_norm)` 完全一致。

**⚠️ 关于负值**（与 po2 完全一致）：

- CLZ 只能处理正整数。
- `reciprocal` 想处理负值：先 `abs`，最后再补符号位。
- `rsqrt`：`x ≤ 0` 无定义 → 直接输出饱和最大值。
- `sqrt` / `log`：`x < 0` 无定义 → 输出 `zero_point`。
- `power_2`：用 sign-magnitude 分离，`x² = |x|²`。

---

## 4. 查表（定点乘加 + 饱和）

### 4.1 分段线性拟合的总体思路

对每一段 `[x_start, x_end]`，用最小二乘法拟合：

```text
y = b * x + c
```

**General 版的关键改造**：不再把 `b`、`c` 单独量化到独立 quantizer 里，
而是**把 `s_x`、`s_y` 一起合并进系数**：

$$
\boxed{\,
M = b \cdot \frac{s_x}{s_y},\qquad
T = \frac{c}{s_y} + zp_y\,}
$$

- `M` 是「在 `q_x` 整数空间下，每加 1 个 `q_x_offset` 应当让 `q_y`
  增加多少」的**真实倍率**。
- `T` 是「`q_x_offset = 0` 时段内贡献的 `q_y` 截距」。

### 4.2 `q_b` / `n_bx_total` / `term_c` 的来历

#### 4.2.1 `q_b` 与 `n_bx_total`：每段独立的 best-shift

把实数倍率 `M` 编成 `(q_b, n_bx_total)`，使得
`q_b · 2^{-n_bx_total} ≈ M`，且 `|q_b|` 几乎用满
`COEFF_B_BIT_WIDTH` 的有符号范围：

```python
def quantize_multiplier(M_real, bw_b):
    max_q = 2**(bw_b-1) - 1
    if abs(M_real) == 0:
        return 0, 0
    n_bx_total = floor(log2(max_q / abs(M_real)))   # 最大 n 使 |M · 2^n| ≤ max_q
    q_b        = round(M_real * 2**n_bx_total)      # 在该 n 下做四舍五入
    return q_b, n_bx_total
```

性质（与 po2 版对照）：

- `n_bx_total` 与符号无关，只与 `|M|` 的数量级有关；段间动态范围越大，
  每段 `n_bx_total` 自动适配 → **等价于 po2 版「归一化斜率」
  (`fit_single_function_quantized_normalized`) 的副作用，但不需要
  单独走一条代码路径**。
- 全表共享 `COEFF_B_BIT_WIDTH`、`INTERNAL_BIT_WIDTH`、`TERM_C_BIT_WIDTH`
  三个全局位宽；`n_bx_total` 在 `[-acc_bw, +acc_bw]` 范围内动态变化，
  硬件上仅是 ROM 里多一个 6-bit shift 字段（po2 版「归一化斜率」模式
  下完全相同的存储格式）。

#### 4.2.2 `term_c` 的预烘焙

把推理公式完全展开（用通用 scale 表示）：

$$
y_{float} = b \cdot x_{float} + c
\quad\Leftrightarrow\quad
(q_y - zp_y)\cdot s_y \;=\; b \cdot (q_x - zp_x)\cdot s_x + c
$$

> `q_x` 此时减去 `zp_x` 后我们把它视为对称有符号；与 po2 版相同。

移项（保持 int）：

$$
q_y \;=\;
\underbrace{q_b \cdot (q_x - zp_x) \gg n_{bx\_total}}_{\text{term\_bx}}
\;+\;
\underbrace{\mathrm{SAT}\!\left(\mathrm{round}\left(\frac{c}{s_y} + zp_y\right),\ \mathrm{TERM\_C\_BW}\right)}_{\text{term\_c (pre-computed)}}
$$

要点：

- `term_c` 在拟合阶段就烘焙进 JSON，推理时 ROM 直接读，**零运算**。
- 所有 `zero_point` 也都吸进 `term_c`。
- **没有 po2 版的 `n_yc` 移位环节**——`c` 早就在浮点空间里被合并进了
  `T`，因此 `term_c` 是一步到位的结果。
- 副作用：`term_c` 在大动态范围下需要更宽的位宽（推荐 32-bit）。

### 4.3 Int-LUT 的定点推理流程

定点链路（与 po2 字节对齐，只在第一拍多/少一次 multiplier）：

```text
输入 q_x  (int, bw_in)
  -> §3.1.1.2 (可选) 跨 op scale 适配                      (1 拍 mul + shift)
  -> 段查找: 在 segments[].threshold_quantized 上做 int 比较 (1 拍)
  -> 读取:   q_b, n_bx_total, term_c                        (ROM)
  -> x_offset = q_x - zp_x                                  (int 减)
  -> bx      = SAT(q_b * x_offset, internal_acc_bw)         (int 乘, 大位宽)
  -> term_bx = SAT(bx >> n_bx_total, internal_acc_bw)       (算术右移; 负 n 改左移)
  -> q_y     = SAT(term_bx + term_c, internal_acc_bw)       (int 加)
  -> q_y     = clip(q_y, q_min_out, q_max_out)              (输出饱和)
  -> (可选) y_float = s_y * (q_y - zp_y)                     (下游反量化)
```

**所有步骤都是 int，没有一次浮点运算。** 与 po2 唯一的 ROM 字段差异：
**只剩一个 `n_bx_total`**，不再拆分为 `n_bx / n_yb / n_yc`。

---

## 5. 后处理（只有规格化类有）

后处理只存在于**规格化类**。饱和类和周期类、单独定制类的 `q_y`
从 LUT 查到的就是最终结果。

**重要说明（硬件视角）**：

> 与 po2 版「`· × 2^e` 永远是算术移位」不同，**General 版的反归一化
> 必然包含 1 次定点常数乘法**——用来吸收 `s_x`、`s_y` 在 `√·`、`1/·`、
> `(·)²`、`ln(·)` 下产生的实数残差。
>
> **唯一例外：`log`** —— `log` 的指数恢复仍然是**加法**（`log(m·2^E)
> = log(m) + E·ln 2`），但 `log(s_x)` 这一项是常量，**烘焙在表头**，
> 硬件上还是「乘+加」。

### 5.0 先算输出反归一化的 multiplier：`(R_q, R_shift)`

在 fitter 阶段，按函数类别预先算出一个**实数 `R`**，再用
`quantize_multiplier(R, MULTIPLIER_BIT_WIDTH)` 编成 `(R_q, R_shift)`：

| 函数         | `R` 表达式                         | 物理意义                                                     |
| ------------ | ---------------------------------- | ------------------------------------------------------------ |
| `reciprocal` | `R = s_{norm\_y} / (s_x · s_y)`    | 把 `1/m` 从归一化空间挪到全局输出空间，并吸收 `1/s_x`，再换算到 `1/s_y` 单位 |
| `sqrt`       | `R = √{s_x} · s_{norm\_y} / s_y`   | `√x = √m · 2^{E/2} · √{s_x}` 中 `√{s_x}` 的吸收              |
| `rsqrt`      | `R = s_{norm\_y} / (√{s_x} · s_y)` | `1/√x` 中 `1/√{s_x}` 的吸收                                  |
| `power_2`    | `R = s_x² · s_{norm\_y} / s_y`     | `x² = m² · 2^{2E} · s_x²` 中 `s_x²` 的吸收                   |
| `log`        | `R = s_{norm\_y} / s_y`            | 把 `q_y_norm`（在 `s_{norm\_y}` 单位下的 `ln m`）挪到输出空间 |

> `R` 在 `s_x ≈ 2^-12`、`s_y ≈ 2^-13` 的常见 16-bit 配置下大概在
> `1e-4 ~ 1e+2`，`quantize_multiplier` 自动选 `R_shift ∈ [10, 30]`。

`E` 是 §3.4 给出的 `e_offset + 1`，对每个采样点都算一遍。

### 5.1 `reciprocal`

$$
\frac{1}{x} \;=\;
\frac{1}{m\cdot 2^{E}\cdot s_x}
\quad\Rightarrow\quad
q_y - zp_y \;=\; (q_{y,\text{norm}} - zp_{y,\text{norm}})\cdot R \cdot 2^{-E}
$$

定点后处理（**1 次 int 乘 + 1 次有符号 shift + 1 次加 zp**）：

```text
y_offset    = q_y_norm - zp_y_norm
prod        = y_offset * R_q                          # ← 唯一 1 次 int 乘
net_shift   = R_shift + E                             # 把 2^-E 也合并到 shift
if net_shift >= 0:
    q_y_raw = (prod + (1 << (net_shift-1))) >> net_shift
else:
    q_y_raw = prod << (-net_shift)
q_y         = SAT(q_y_raw + zp_y, acc_bw)
```

硬件实现：1 次 int 乘 + 1 次有符号 shift，比 po2 多 1 次乘
（po2 是纯 shift）。

### 5.2 `sqrt`

$$
\sqrt{x} \;=\; \sqrt{m} \cdot 2^{E/2} \cdot \sqrt{s_x}
$$

把 `E` 拆为 `E_int = E // 2`、`parity = E & 1`：

```text
y_offset = q_y_norm - zp_y_norm
val      = y_offset * R_q                             # int 乘 1
if parity == 1:
    val  = SAT(val, acc_bw)
    val  = (val * SQRT2_Q16 + (1 << 15)) >> 16        # int 乘 2 (Q16 √2)
net_shift = R_shift - E_int
if net_shift >= 0:
    val  = (val + (1 << (net_shift-1))) >> net_shift
else:
    val  = val << (-net_shift)
q_y      = SAT(val + zp_y, acc_bw)
```

常量：`SQRT2_Q16 = round(√2 · 2^16) = 92682`。

> 也可以把 `R_q · √2` 预先合并成 `R_q_odd`，避免推理时的 `* SQRT2_Q16`
> 这一步，代价是 ROM 多一对 `(R_q_odd, R_shift_odd)`。

硬件实现：**偶数 `E` 时 1 次 int 乘（`R_q`）**；
**奇数 `E` 时多 1 次 √2 乘**（与 po2 同；但 po2 偶数情况是纯 shift）。

### 5.3 `rsqrt`

$$
\frac{1}{\sqrt{x}} \;=\; \frac{1}{\sqrt{m}} \cdot 2^{-E/2} \cdot \frac{1}{\sqrt{s_x}}
$$

和 `sqrt` 对称，常量换成
`INV_SQRT2_Q16 = round(1/√2 · 2^16) = 46341`：

```text
y_offset = q_y_norm - zp_y_norm
val      = y_offset * R_q                             # int 乘 1
if parity == 1:
    val  = SAT(val, acc_bw)
    val  = (val * INV_SQRT2_Q16 + (1 << 15)) >> 16    # int 乘 2 (Q16 1/√2)
net_shift = R_shift + E_int
if net_shift >= 0:
    val  = (val + (1 << (net_shift-1))) >> net_shift
else:
    val  = val << (-net_shift)
q_y      = SAT(val + zp_y, acc_bw)
```



### 5.4 `log` ✅

$$
\ln(x) \;=\; \ln(m) + E \cdot \ln 2 + \ln(s_x)
$$

**和其它规格化函数不同：不是 shift，而是 int 乘加**。

预烘焙三个常量（均在表头烘焙一份）：

```text
(R_q, R_shift) = quantize_multiplier(s_norm_y / s_y, MULTIPLIER_BIT_WIDTH)
Q_LN2          = round(ln(2)  / s_y)         # ln 2     在输出空间的整数
Q_LN_SX        = round(ln(s_x) / s_y)        # ln(s_x)  在输出空间的整数 (常量)
```

定点后处理：

```text
# y_offset 已经是 "ln(m) 在归一化输出空间的量化值"
y_tmp = y_offset * R_q
if R_shift >= 0:
    y_tmp = (y_tmp + (1 << (R_shift-1))) >> R_shift   # 挪到全局输出空间
else:
    y_tmp =  y_tmp << (-R_shift)
add   = E * Q_LN2 + Q_LN_SX                 # int 乘 (E 通常 ≤ 5-bit) + int 加
q_y   = SAT(y_tmp + add, internal_acc_bw) + zp_out
```

关键点：

- `E` 是规格化得到的小整数（8-bit 输入 `E ∈ [-7, +24]`，
  16-bit 输入 `E ∈ [-15, +16]`），**只要几 bit**。
- `E · Q_LN2` 是「小位宽 × 常量」的 int 乘，延迟很低；也可以查一张
  `E → Q_LN2·E` 的小表（深度仅 ~32）彻底省掉乘法。
- `Q_LN_SX` 是纯加数常量，硬件代价为 0。
- **负输入 `x ≤ 0` 时 `log(x)` 无定义** → 返回饱和值 / 标记 NaN 位。

### 5.5 `power_2`

$$
x^2 \;=\; (|x|)^2 \;=\; m^2 \cdot 2^{2E} \cdot s_x^2
$$

```text
y_offset    = q_y_norm - zp_y_norm
val         = y_offset * R_q                  # int 乘 1
net_shift   = R_shift - 2 * E                 # 2·E == E << 1
if net_shift >= 0:
    val     = (val + (1 << (net_shift-1))) >> net_shift
else:
    val     = val << (-net_shift)
q_y         = SAT(val + zp_y, acc_bw)
```

硬件实现：**1 次 int 乘 + 1 次算术 shift**（po2 是纯 shift，本版多 1 次乘）。

---

## Question

Q: 在前处理和后处理会有数据的计算和移位信息（CLZ 用于后处理）的多次计算，这些计算的应该在什么位宽下进行？前处理应该输出 16bit 的数据来进行 LUT 查表？

A:

- `E = e_offset + 1`：与 `s_x` 无关，`E` 占用约 5~6 bit（位宽列在 §5.0 的描述里）。
- `m_int → q_m` 的 multiplier `q_norm` 为 16-bit，shift `n_norm` 为 6-bit；
  乘积 `m_int * q_norm` 在 16+16 = 32-bit 累加器里完成，再 right-shift 落回 16-bit。
- 16-bit 的 CLZ LUT 表：`n_in = 8 bit`、`e_offset, E`：5~6 bit、`R_shift`：6-bit、`R_q`：16-bit。

---

## 7. LUT 字段的「表头共享 vs 段内独有」切分

> 对照 `lut.py` 中 `infer_with_lut` 与 `infer_with_clz_normalized_lut`
> 的实测读取路径切分；与 po2 版字段一一对应。

### 7.1 表头字段（整张表只存一份，循环外读）

#### A. I/O 量化参数（推理流水线必需）

`infer_with_lut` 在循环外只读一次，`x_offset = q_x − zp_x` 用
`input.zero_point`，输出饱和 `clip(y, output.min, output.max)`
用 `output` 的范围：

| 字段                             | 含义                                                   |
| -------------------------------- | ------------------------------------------------------ |
| `input.zero_point`               | 段查找前减零点；对于同类型函数的多个实例 `zp` 需要更新 |
| `output.min max`                 | 输出饱和 `[min, max]`                                  |
| `internal_accumulator_bit_width` | 中间饱和宽度，16/32-bit                                |

#### B. 前处理常量

| 字段                     | 含义                                                         |
| ------------------------ | ------------------------------------------------------------ |
| `(q_r, n_r)`             | 网络上游 op 与 LUT 的 scale 比值，§3.1.1.2；存储 (MULTIPLIER_BW + SHIFT_BW) ≈ 16 + 6 = 22 bit |
| `Q_2PI`、`Q_HALFPI`      | 周期类（`sin`/`cos`）共用，§3.1.3                            |
| `(q_inv_2pi, n_inv_2pi)` | 周期折叠中 `1/Q_2PI` 的乘倒数近似，§3.1.3                    |

> **对照 po2**：po2 这里存的是 `n_diff`（4-bit）。General 版用
> `(q_r, n_r)` 替代，多出约 18 bit / 每个 op 实例。

#### C. CLZ 规格化类专属表头

JSON 里独立成一个 `clz_params` 块，整张表一份：

| 字段                             | 含义                                                         |
| -------------------------------- | ------------------------------------------------------------ |
| `bit_width`                      | CLZ 输入位宽                                                 |
| `(q_norm, n_norm)`               | `m_int → q_m` 对齐的 multiplier+shift，§3.4                  |
| `(R_q, R_shift)`                 | 反归一化常数 `R = f(s_x, s_y, s_norm_y)` 的 multiplier+shift，§5.0 |
| `Q_LN2 = round(ln 2 / s_y)`      | 仅 `log` 用                                                  |
| `Q_LN_SX = round(ln(s_x) / s_y)` | 仅 `log` 用                                                  |

> **对照 po2**：po2 这里存的是 `n_in`、`shift_m`、`scale_shift`
> 三个纯 shift 字段。General 版统一改为两组 multiplier+shift
> （`(q_norm, n_norm)` + `(R_q, R_shift)`），数据通路上多一个
> 16x32 整数乘法器。

#### D. 查表移位常量（标准量化版下的核心收益）

| 字段         | 推导                                                         | 是否做表头                                                   |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `n_bx_total` | 每段独立由 `quantize_multiplier(M, COEFF_B_BW)` 给出（M = b·s_x/s_y） | ❌ **段内独有**（吸收逐段 `M` 的数量级，相当于 po2 「归一化斜率」模式的默认行为） |

> **对照 po2**：po2 版「饱和/周期/定制类」可以把 `n_bx_total = n_b + n_x − n_y` 上提到表头；
> General 版**所有函数族**统一走「逐段 best-shift」，`n_bx_total` 永远段内，
> 字段统一性更强（po2 的「半独有」状态消失）。

#### E. 后处理常量（函数族级）

§5.x 里的：

- `SQRT2_Q16 = 92682`（仅 `sqrt`）
- `INV_SQRT2_Q16 = 46341`（仅 `rsqrt`）
- `Q_LN2 = round(ln 2 / s_y)`（仅 `log`）
- `Q_LN_SX = round(ln(s_x) / s_y)`（仅 `log`，General 版新增）

---

### 7.2 段内字段（每段独有，循环内逐段读，必须在 16 行 ROM 里）

`infer_with_lut` 循环内每一拍只读这 3 类：

| 字段                 | JSON 字段                    | 用途                                                         |
| -------------------- | ---------------------------- | ------------------------------------------------------------ |
| 边界 `s`             | `threshold_quantized[i]`     | 段比较器树，16 段共 17 个边界                                |
| `q_b`                | `coefficients_quantized.q_b` | 核心斜率，参与 `bx = q_b * x_offset`                         |
| `n_bx_total`         | `shift_bits.n_bx_total`      | 段内 `(q_b, n_bx_total)` 对的 shift 分量                     |
| `term_c_precomputed` | `term_c_precomputed`         | 核心常数，已烘进 `round(c/s_y + zp_y)`（CLZ log 路径再 `+ E · Q_LN2 + Q_LN_SX`） |

> **对照 po2**：po2 段内字段是 `(q_b, term_c)` + 半独有的 `n_bx_total`；
> General 版段内字段恒为 `(q_b, n_bx_total, term_c)` 这 3 项，
> ROM 列固定、解析逻辑统一。

---

## 8. 与 Power-of-Two 版的对照清单

| 维度                   | `lut_int_design_0513_26.md`（po2 版）                        | **`lut_int_design_general_0522_26.md`（本版）**              |
| ---------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `scale` 形式           | 强制 `2^-n`                                                  | **任意正浮点**（TFLite/ONNX-RT/AIMET v2 默认）               |
| `quantizer` 存储       | 单个整数 `n`                                                 | `(M, n_s)` 一对：MULTIPLIER_BW + SHIFT_BW ≈ 16+6 bit         |
| `n_bx_total` 推导      | `n_b + n_x − n_y`（多字段组合，全表共享或逐段）              | 每段独立 `quantize_multiplier(M, COEFF_B_BW)`（始终段内）    |
| `term_c` 推导          | `q_c >> n_yc + zp_y`（需要单独的 `q_c`）                     | `round(c/s_y + zp_y)` 一步到位，没有 `q_c` 中间量            |
| ROM 段字段             | `q_b, term_c, [n_bx_total]`                                  | `q_b, n_bx_total, term_c`（恒为 3 项）                       |
| 「归一化斜率」分支     | 需 `fit_single_function_quantized_normalized` 单独路径       | **不需要**，每段 best-shift 默认就是它                       |
| §3.1.1.2 跨 op 适配    | 1 次 shift（`n_diff` 4-bit）                                 | 1 次 multiplier+shift（`q_r` 16-bit + `n_r` 6-bit）          |
| §3.4 `m_int → q_m`     | 1 次 shift（`bw - n_norm`）                                  | 1 次 multiplier+shift（`q_norm`, `n_norm`）                  |
| §5 CLZ 反归一化        | 纯 shift；sqrt/rsqrt 奇 E 多 1 次 √2 乘；log 多 1 次 `E·Q_LN2` | **必有 1 次 R-mul**；sqrt/rsqrt 奇 E 再加 1 次 √2 乘；log 再加 1 次 `E·Q_LN2 + Q_LN_SX` |
| 周期类 cos 复用 sin 表 | 文档规划                                                     | **同样支持**，借 `Q_HALFPI` 实现                             |
| 跨 op 复用饱和类 LUT   | 文档规划，部分支持                                           | **完整支持**（`(q_r, n_r)` 表头烘焙）                        |

---

## 10. MRNN 分解图与 §2.3 规格化类映射

MRNN 经 `prepare_model` 后，CLN/BN 等 stateless 模块拆成 §2.3 规格化 primitive 链：

```text
CLN:  x → power_2 → mean → sqrt → reciprocal(1/std) → (× x)
BN:   (x-μ) / σ  →  sub, square, mean, add, sqrt, reciprocal(σ), mul
```

| 分解节点 | §2.3 函数 | 说明 |
|----------|-----------|------|
| `module_square*` | `power_2` | 输出须按 x² 动态范围标定 `out_max` |
| `module_sqrt*` | `sqrt` | 正域 + clamp(EPS) 后进 CLZ |
| `module_div*`（除 std/σ） | **`reciprocal(b)`** | 图上是 `QuantizedDivide(a,b)`；部署为 `a × reciprocal(b)` |
| `module_sign*` | （非 §2.3） | bypass/finer input Q，无 LUT head |

Encoding 工具：`aimet_rx-main/examples/common/mrnn_clz_encoding.py`（Po2 后 reciprocal 分母 + power_2 输出重标定）。

---

- **优先用 General 版的场景**：
  - 上游 quantsim 是 AIMET v2 默认 / TFLite / ONNX-RT，scale 不强制 `2^-n`。
  - 一份 LUT 要被多个不同 calibration 范围的 op 复用（饱和类）。
  - `sin` / `cos` 想共享一张表。
  - 16-bit 及以上位宽下追求更小的 MAE（每段 `n_bx_total` 自动 best-shift）。
- **优先用 po2 版的场景**：
  - 上游 quantsim 强制 `PowerOfTwoQuantizer`（AIMET v1 或某些 NPU 工具链），整条流水都是纯 shift，本版 §3.1.1.2 + §5 的 multiplier 是多余开销。
  - 8-bit 紧位宽配置（系数也 8-bit），此时 `q_r / R_q` 8-bit 的精度劣化已经压过 po2 的纯 shift 误差。
- **共用部分**：JSON 顶层结构、ROM 段字段、四类函数分类、CLZ 数学等均对齐；
  两版可以**串联部署**（同一硬件 ROM 解析器，配置文件里只切换
  「是否启用 §3.1.1.2 mul」与「是否启用 §5 R-mul」两个 flag 即可）。
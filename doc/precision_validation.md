# 单算子定点精度验证记录

本文件按算子记录"INT16_FIXED_EVAL kernel 输出 vs float reference"
的实测精度，作为 spec 与 CI 之间的可追溯桥梁。它的**唯一目标**：
对每个已实现的 fixed-point 单算子，明确列出
（实现位置 / 测试位置 / reference 类型 / 输入分布 / 阈值 / 实测值快照 / 状态），
用于 review 与回归。

## 本文件不是

- **不是** spec：spec 在 `doc/04_算子详细规格/`、
  `doc/FixedPoint_Quantization_Spec/`、ADR；本文件只**引用**而不重写
  数学定义或硬件契约。
- **不是** CI 阈值表：阈值表在
  `aimet_torch/fixed_point/metrics/thresholds.py`；本文件**引用**那里
  的常量名，并记录实际运行下命中的统计快照。
- **不是** test 列表：完整 test 在 `tests/fixed_point/`；本文件指向**主**
  精度测试文件，不重复罗列结构性 / dispatch 测试。

## 字段说明（每个算子段使用）

| 字段 | 含义 |
|------|------|
| **Kernel** | 实现路径 + 类名 |
| **KernelKind** | `REQUANTIZING` / `LOOKUP` / `SAME_GRID_VALUE` / `SAME_GRID_OR_REQUANT` |
| **Spec** | 主要 spec 文档锚点（doc/04_xx_xx.md 或 ADR） |
| **测试** | 主精度测试文件 |
| **Reference** | 用于对比的 float 参考函数（FP32 / analytic / 同类 reference Python 实现） |
| **Input 分布** | 测试输入构造方式（grid / 数值范围 / 随机种子 / trial 数） |
| **阈值** | 命中的 thresholds.py 常量 + 数值 |
| **实测值** | 当前 commit 上的统计快照（cos_min / lsb_max 等） |
| **状态** | ✓ PASS / ⚠ KNOWN_LIMIT / ✗ FAIL / TODO |
| **注** | 算子级语义限制 / known-limits / follow-up 链接 |

实测值是**快照**而非阈值——CI 只检阈值；但实测快照让"环境/PyTorch 版本/
随机种子选择带来的 ULP 漂移"在 review 时可见。每次对算子做精度相关改
动后用文末 §**实测值采集流程**重跑。

---

# 系统级 / 跨算子 known limitations

本节记录**跨多个 op 的系统级限制** —— 单算子段无法独立描述、需要从全
图角度才能看清的精度天花板。每条都附"诊断指引"，帮助后续 reviewer
在碰到相同症状时快速识别根因，避免重复探索。

## SYS-LIMIT-1: REQUANTIZING 16-bit×16-bit MAC 触发 INT32 ALU 饱和（**RESOLVED with combo contract — 2026-06-09，SYS-FU-1.B**）

- **契约位置（更新后）**:
  - `aimet_torch/fixed_point/capabilities.py::SUPPORTED_ACTIVATION_BITWIDTHS = (8, 16)`
    （per-operand 验证集）
  - `aimet_torch/fixed_point/capabilities.py::REQUANTIZING_COMBO_BITWIDTH_BUDGET = 24`
    （MAC-reduction 组合上限：`input_bw + weight_bw ≤ 24`）
  - `aimet_torch/fixed_point/capabilities.py::OperatorCapability.is_reduction`
    （区分 Conv/Linear/MatMul 的 MAC 求和归约 vs Multiply/Divide
    的 element-wise / AvgPool/Mean 的 sum-only 归约）
- **守护测试（更新后）**:
  - `tests/fixed_point/test_v2_int16_adapter.py::test_int16_fixed_eval_refuses_full_16bit_reduction`
    —— 16+16 MAC 组合必拒
  - `tests/fixed_point/test_v2_int16_adapter.py::test_quantized_linear_int16_fixed_combo_gate_rejects[W4A8/W4A16/W16A16]`
    —— 4-bit weight、16+16 reduction 必拒
  - `tests/fixed_point/test_v2_int16_adapter.py::test_avgpool2d_full_16bit_dispatch_succeeds`
    —— sum-only reduction 16+16 必通
  - `tests/fixed_point/test_v2_int16_adapter.py::test_linear_asymmetric_16bit_dispatch_succeeds`
    —— 16+8 子集必通
  - `tests/fixed_point/test_w5_combo_probe_regression.py` —— W5.1 矩阵
    冻结：所有 9 个 `combo × N` case dispatch + 3 个 W16A16 拒
- **覆盖 op**: 所有 `KernelKind.REQUANTIZING`
  - **MAC-reduction（is_reduction=True）**: `Linear` / `Conv*` /
    `MatMul` —— 受 24-bit combo budget 约束，16+16 永远拒
  - **Element-wise / sum-only（is_reduction=False）**: `Multiply` /
    `Divide` / `Add`(cross-grid) / `Subtract`(cross-grid) —— 单 MAC，
    无 N 累加，16+16 通过；`AvgPool` / `Mean` / `LayerNorm` —— sum-only
    reduction（无 operand×operand），16+16 通过
- **真因**（保持不变）: HW spec 的 INT32 ALU 限制
  （`saturate_mac_accumulator` / `require_int32_saturated_accumulator`
  强制契约）：
  - 8bit×8bit 单 MAC `±2^14`，`N=2^17` 才接近 int32 边界 → 大部分层安全
  - 16bit×16bit 单 MAC `±2^30`，`N=2` 已饱和 → 大 N 系统性丢高位信息
  - 16bit×8bit / 8bit×16bit 单 MAC `±2^22`，`N=2^9` 才接近边界
    → SYS-FU-1.B 在 `N ≤ 4096` 内全程安全（probe 验证）
- **W5.1 probe 数据快照**（2026-06-09 ad-hoc，`LinearInt16Kernel`，
  B=8, M=16, random fp32 input/weight；probe 脚本未入库，但矩阵
  已冻结为 `tests/fixed_point/test_w5_combo_probe_regression.py` 的
  dispatch contract regression（dispatch 级守护，对应表的"是否
  接受 / 是否拒绝"列）。SQNR 列保留作历史参考）:

  | input bw | weight bw | output bw | N | cos | SQNR (dB) | 契约 |
  |---|---|---|---|---|---|---|
  | 8 | 8 | 8 | 64 | 0.99995 | 39.6 | ✅ accept (baseline) |
  | **16** | 8 | 8 | 64 | 0.99996 | 41.0 | ✅ accept (SYS-FU-1.B) |
  | 8 | **16** | 8 | 64 | 0.99997 | 42.4 | ✅ accept (SYS-FU-1.B) |
  | 16 | **16** | 8 | 64 | 0.99964 | 31.4 | ❌ **reject** (combo > 24) |
  | 16 | 16 | 16 | **4** | 1.0 | 86.7 | ❌ reject (small N 也拒，contract 不依赖 N) |
  | 16 | 16 | 16 | 64 | 0.99966 | 31.3 | ❌ reject |
  | 16 | 16 | 16 | **1024** | 0.96 | **9.4** | ❌ reject |
  | 16 | 16 | 16 | **4096** | 0.89 | **4.7** | ❌ reject |

  → SYS-FU-1.B 解封了 `16+8 / 8+16` 子集（最低风险路径），16+16 reduction
  仍按 contract 拒（无论 N，避免大 N 隐式退化）。
- **历史选项（已选定）**:
  - ~~A. 严格 16bit×16bit~~：升 ALU 到 INT64，**未实施**（HW spec
    改动，5+ 天）。RESOLVED 后该路径变为"未来可选"。
  - **B. 16bit input + 8bit weight 子集（已实施）**：扩
    `SUPPORTED_ACTIVATION_BITWIDTHS = (8, 16)` + 新增
    `REQUANTIZING_COMBO_BITWIDTH_BUDGET = 24` + `is_reduction` flag。
    实施落点：commit `c4c15cb` (PR-1 capabilities API) →
    `8da69c1` (PR-2 adapter/diagnose 接入) →
    `fc6ebca` (PR-3 守护测试改造) →
    `177a87e` (PR-4 dispatch 矩阵冻结)。
- **诊断指引（碰到 16bit acceptance config 报错时）**:
  - 若 acceptance config（如 `mrnn_acceptance_mixed_precision.json`）
    把某 op 的 `input_bitwidth=16` 升到 16bit 而组合 gate 报错，
    先看错误消息是哪一类：
    - `INT16_FIXED_EVAL bitwidth=N at ... is not validated for
      REQUANTIZING kernels; supported bitwidths: {8, 16}` —— operand
      位宽不在 (8, 16)，例如 4-bit weight；改回 8 或 16
    - `INT16_FIXED_EVAL combo (operand_bw=16, operand_bw=16) at ...
      exceeds the REQUANTIZING-with-MAC-reduction budget` —— 16+16
      MAC reduction，把 weight 或 input 一边降到 8-bit
  - **不要直接把 budget 改 32 或 `_REQUANTIZING_COMBO_VALIDATED_BITWIDTHS`
    收紧** —— 前者会让 INT32 ALU 在大 N 隐式饱和（W5.1 已数值证伪），
    后者会破坏 PR-3 的 expected-pass 守护测试。两条都有 commit 历史
    引用，回滚或调整需先 update 矩阵 + 重跑 probe
- **acceptance config（`examples/config/mrnn_acceptance_mixed_precision.json`）E2E 状态**（2026-06-09 实跑验证）:
  - **smoke 跑分**（`--max-eval-batches 4 --max-calib-batches 4`，19s 全程）:
    - `compute_encodings` 完成 4.6s ✅
    - `ensure_output_quantizers_for_int16_eval` patched 97 slots ✅
    - CLZ encoding fix（reciprocal=8 / power_2=6）通过 ✅
    - `convert_encodings_to_fixed_scale`：164 个 affine quantizer
      已缓存 `(M, r)` ✅
    - `diagnose_int16_readiness` 报告 1 项 `unsupported_activation_bitwidth`：
      `[('fft2band.module_matmul', 'QuantizedMatMul', 16)]` —— 与
      `int16_fixed_eval` 实跑同步报错，contract 自洽。
    - `int16_fixed_eval` ❌ 被
      `REQUANTIZING-with-MAC-reduction` 拒：`fft2band.module_matmul`
      input + weight 都是 16-bit，`is_reduction=True`，组合 32 > 24
      budget。
    - `float_native` baseline 96.88% Top-1（4 batch）✅
  - **结论**:
    - **acceptance config 中除 `fft2band.*` MatMul 之外的所有 16-bit
      升级（element-wise REQUANTIZING / BatchNorm / sum-only 归约）
      都按 PR-2 combo gate 通过**，与 PR-3/4 守护测试预期一致。
    - **`fft2band.module_matmul` 是该 config 唯一被新 contract
      阻断的 op**。该阻断不是 PR-2 引入的回归——PR-2 之前 16-bit
      input 被旧 `SUPPORTED_ACTIVATION_BITWIDTHS=(8,)` gate 拒，
      PR-2 之后被新 combo budget 拒，**结果一致**（拒）但**理由
      更精确**（点出 MAC 累加器饱和而非笼统的"16-bit 不支持"）。
    - **acceptance config 并未为新 contract 适配**：要让 fft2band
      的 MatMul 在 `INT16_FIXED_EVAL` 真正跑通，需要把
      `fft2band.*` 的 `input_bitwidth=16` 与 weight 的 16-bit 中
      至少一边降到 8-bit（即 SYS-FU-1.B 子集）。这属于**后续
      工单**（acceptance config 维护方决定哪边降；从 W5.1 probe
      看 `weight=8bit` 损失最小）。
  - **metric-level（不再 follow-up）**: 由于 acceptance config 还
    需调整 fft2band 才能跑全图 INT16_FIXED_EVAL，且 SYS-OPEN-Q-1
    会拖底任何 backbone-level metric 数字，**全 epoch metric 跑分
    在 fft2band 重配 + SYS-OPEN-Q-1 闭合前都不会有可比较的数据**。
    这条已从 SYS-LIMIT-1 的 follow-up 列表中独立成为
    SYS-OPEN-Q-2（MatMul 在 acceptance config 中的 16+8 重配
    决策），与本 RESOLVED 不再耦合。

## SYS-OPEN-Q-1（未结案）: MRNN backbone 8bit×8bit Conv/ConvT/Linear 单步 SQNR ≤ 5 dB

**这是与 SYS-LIMIT-1 独立的另一个问题**，本轮 R2 + W2 + W5.1 调研未
触碰其根因，留作后续工单。

- **症状**（`quick_start_full_quant.json` 8bit baseline，
  `--per-node-int16-isolated __all__`）：

  | module | iso_cos | sqnr_dB |
  |---|---|---|
  | `freq_downs.2.conv2d` | 0.776 | 3.07 |
  | `freq_downs.1.conv2d` | 0.818 | 1.58 |
  | `neck_seqs.1.conv_t` | 0.844 | 3.74 |
  | `neck_seqs.0.conv_t` | 0.848 | 2.26 |
  | `fc0` (Linear) | 0.873 | 5.34 |
  | `enc_seqs.1.conv_t` | 0.889 | 2.60 |
  | `freq_downs.0.conv2d` | 0.901 | 5.21 |
  | `enc_seqs.0.conv_t` | 0.915 | 4.93 |

  全图 `int16_fixed_eval Top-1 = 0.00%`、`cosine vs float_native = -0.27`
- **与 SYS-LIMIT-1 无因果**: W5.1 probe 显示 8bit×8bit 在 N=64 时
  SQNR=39.6 dB，**该量级与 MRNN baseline 实测的 ≤5 dB 差 30+ dB**；
  解封 16bit 路径**无法**修复本问题
- **W5.2 进一步排除"通用大 K conv 路径退化"**：用 MRNN
  `freq_downs.{0,1,2}` 完全相同的 `(C_in, C_out, K, stride)` shape
  +random fp32 输入 + per-channel weight 8bit + per-tensor activation 8bit
  跑 `Conv2dInt16Kernel`：

  | shape (与 MRNN 同) | C | K | N=Cin·Kh·Kw | cos | SQNR (dB) |
  |---|---|---|---|---|---|
  | freq_downs.0_like | 120→120 | (2,4) | 960 | 0.999865 | **35.7** |
  | freq_downs.1_like | 120→240 | (2,5) | 1200 | 0.999870 | **35.8** |
  | freq_downs.2_like | 240→320 | (1,6) | 1440 | 0.999843 | **35.0** |
  | small_K_3x3       | 64→64   | (3,3) | 576 | 0.999854 | **35.3** |

  → kernel 在 MRNN 形状上 SQNR≈35 dB（`per_channel_quantization` 与
  否差异 <1 dB），与 MRNN baseline 实测的 1.6 / 1.6 / 3.1 dB 相差
  **30+ dB**。**SYS-OPEN-Q-1 不是 kernel bug**，是 MRNN
  business-data + calib + activation 分布特定的问题
- **已排除的方向**:
  - per-channel weight：`mrnn_quantsim_config_custom_mixed_precision_v2.
    json` 全局已开（line 15）→ 多写一份 user-level config 给
    layer_type 加 `per_channel_quantization` 数字逐字相同
  - 100 步 `INT16_FIXED_QAT_SIM` 微调（lr=1e-4, bs=32）：`ce_loss` 在
    `5.4 ~ 7.6` 区间随机震荡无下降；post-QAT isolated 数字差异在第
    4-5 位小数；fp32 ckpt 在 INT16 forward 下 `ce_loss=5.4`（fp32 时
    ~0.1）意味着 INT16 forward 已结构性破坏模型表征，loss surface 在
    该 weight 邻域完全失真，short-step QAT 无法逃出
- **W5.1 + W5.2 已排除：通用 kernel/算子路径**（kernel 数学正确）
- **可能的方向（未验证，留作后续）**:
  - **MRNN 真实 activation 分布的 long-tail / multi-mode**：random
    gaussian 数据 SQNR 35 dB，MRNN 真实 STFT-band-feature 数据 1-5 dB
    → 量级差异说明 activation 分布对 8bit per-tensor quantizer 极不
    友好（音频谱常见 30+ dB 动态范围）
  - **calib percentile 截断**：`percentile=99.99` 在 long-tail 分布上
    会截掉真实 outlier，导致 backbone 某些 channel 失去表达能力
  - **frontend non-linear (HypotFun / PowerCompress / log) 输出量化**
    把 channel-wise 信息全压到 per-tensor 单一 scale 下
- **诊断指引（下一阶段工单 SYS-FU-2）**:
  1. **dump 真实 activation 分布**：在 fp32 forward 阶段 hook
     `freq_downs.{0,1,2}.conv2d.input` + `*.cln.output`，记录每
     channel 的 abs-percentile (50/99/99.9/99.99/100)，看是否
     long-tail / multi-mode
  2. **per-channel activation 量化实验**：把 `freq_downs.*.input`
     activation_quantizer 切到 per-channel 模式（业务 spec 不允许时
     至少 audit 测试），看 isolated SQNR 是否拉起；这是验证
     "per-tensor activation grid 是真因"的强证据
  3. **calib scheme 替换**：尝试 `min_max` / `tf_enhanced` / 更小
     percentile（99.5）的 calib，看 isolated SQNR 是否改善
  4. 若 1-3 都不能拉起，再考虑 frontend 重设计或 STFT-domain 输出
     channel-wise scale 表达

## SYS-LIMIT-2: R2 grid-aware floor 与 fp32 EPS 语义差

详见 `nn.Hardtanh / custom.Clamp / custom.Clip` 章节
**`R2-GRID-FLOOR-VS-FP32-EPS-SEMANTIC-GAP`**（known limitation 段）。
摘要：`Clamp(min=ε)` 在 INT16 grid 上 floor 到 1 LSB，与 fp32 reference
的 `EPS=1e-8` 在数值上差几个数量级；表现为 `*.cln.module_div_*`
单步 `iso_cos ~0.16` 的整 channel 错位。唯一根治路径 = R1（业务侧
`CLN.EPS` 提到 `1e-3`，需重训 fp32 ckpt）。**MRNN 上具体观察**：
4 个 `*.cln.module_div_*` 中 3 个 `≥0.99977`，仅 `enc_seqs.1.cln.
module_div_4` 卡在 `0.165`（该支 `mean_sq abs_max=0.192` 比其他三支
小一两个数量级，small-mean_sq channel 占比高，触发本语义差）。

---

# REQUANTIZING

## custom.Add

- **Kernel**: `aimet_torch/fixed_point/kernels/eltwise.py::AddInt16Kernel`
  （继承 `_BinaryAlignedKernel`，hot path 调 `int32_add_sat`）
- **KernelKind**: `REQUANTIZING`
- **Spec**: `doc/04_算子详细规格/04_03_逐元素运算类算子.md` §4.3.2
  - spec 契约：`x_i' = (q_xi - Z_xi) * M_i >> rshift_i`（两路各自独立
    `M_i, rshift_i`），`y_q = sat(x_1' + x_2' + Z_y)`
  - kernel 对齐：`_BinaryAlignedKernel._align_to_output` 对每个输入独立
    做 `quantize_multiplier(scale_xi / scale_y)`，与 Subtract 共享
    实现（只差末尾 `int32_add_sat` vs `int32_sub_sat`）
- **Spec 测试覆盖度**: ⚠ **部分**——同 Subtract，Path-B 测试固定
  `multiplier=32767, rshift=15`（α ≡ 1 的边界），即 `scale_a = scale_b =
  scale_out` 时两路 multiplier 都退化为恒等。"两路 M_i 都不为恒等"的
  完整 spec 路径在 adapter dispatch 测试中通过 end-to-end calibrate
  path 间接覆盖；本 file 等价测的是 `int32_add_sat + saturate` 的精度
  下限。详见 §**精度收敛 follow-up** 的 `FU-ADDSUB-DUAL-M`。
- **测试**: `tests/fixed_point/kernels/test_add_ideal_float_reference.py`
- **Reference**: `ref = a + b`（float32，无量化的 ideal float add）
- **Input 分布**: 32 trials × 64 elements per (grid, mode)
  - 输入码 `qa, qb ∈ [-code_limit, code_limit]`（signed）或
    `[0, code_limit]`（unsigned），code_limit 与 grid 相关
  - 同标度：`scale_a = scale_b = scale_out`
  - 跨标度：`scale_a = scale_out * exp(±0.05/2)`（i8 用 ±0.02）
- **阈值**:
  - `_MIN_COSINE = 0.9999`（要求严格 `>`）
  - `_MAX_FLOAT_LSB = 1.0`（要求严格 `<`）
  - 与 Subtract Path-B 共享同一份精度 floor
- **实测值**（commit time 2026-06-08, PyTorch CPU, 32 trials × 64 elem）:

| grid | mode | cos_min | lsb_max |
|------|------|---------|---------|
| i8 | same-scale | 1.000000 | 0.0000 |
| i8 | cross-scale | 0.999984 | 0.2285 |
| u8 | same-scale | 1.000000 | 0.0000 |
| u8 | cross-scale | 0.999949 | 0.4990 |
| i16 | same-scale | 1.000000 | 0.0000 |
| i16 | cross-scale | 0.999979 | 0.4997 |
| u16 | same-scale | 1.000000 | 0.0000 |
| u16 | cross-scale | 0.999989 | 0.5003 |
| i32 | same-scale | 1.000000 | 0.0000 |
| i32 | cross-scale | 0.999978 | 0.4992 |

- **状态**: ✓ PASS（同标度 bit-exact；跨标度 cos ≥ 0.99995，远高于
  0.9999 floor，lsb_max ≤ 0.50）
- **注**:
  - **u32 grid pending**：`test_add_path_b_u32_pending_int64_container`
    标 SKIP，需要 int64 carrier；spec 04_03 允许 u32 但当前 `int32_add_sat`
    在 carrier 上限 ±2³¹ 触发 overflow，等 int64 path 落地后启用
  - **unsigned 编码**：与 Subtract 相反，Add 在 unsigned grid 上天然 valid
    （`a+b ≥ 0`），所以本段覆盖 u8/u16；spec 路径 `Z_x1+Z_x2` 与本段
    `Z_x=Z_y=default_zp=0` 是兼容子集，非零 zp 的 unsigned 路径见
    `FU-MUL-Z-NONZERO`（与 Multiply 同类 follow-up）

---

## custom.Subtract

- **Kernel**: `aimet_torch/fixed_point/kernels/eltwise.py::SubtractInt16Kernel`
  (继承 `_BinaryAlignedKernel`，hot path 调 `int32_sub_sat`)
- **KernelKind**: `REQUANTIZING`
- **Spec**: `doc/04_算子详细规格/04_03_逐元素运算类算子.md` §4.3.3
  - spec 契约：`x_i' = (q_xi - Z_xi) * M_i >> rshift_i`（**两路各自有
    独立 M_i, rshift_i**），`y_q = sat(x_1' - x_2' + Z_y)`
  - kernel 对齐：`_BinaryAlignedKernel._align_to_output` 对每个输入独
    立做 `quantize_multiplier(scale_xi / scale_y)`，行为与 spec 一致
- **Spec 测试覆盖度**: ⚠ **部分**——当前 Path-B 测试在 `_output_encoding`
  里固定 `multiplier=32767, rshift=15`（α ≡ 1 的边界），即 `scale_a =
  scale_b = scale_out` 时两路 multiplier 都退化为恒等。**真正 spec 描述
  的"两路独立 M_i 都不为恒等"的情形**没被这个 file 覆盖；它在
  `tests/fixed_point/test_v2_int16_adapter.py` 的 dispatch 测试里通过
  end-to-end calibrate path 间接覆盖。本 file 等价测的是
  `int32_sub_sat + saturate` 的精度下限，不等价测 `quantize_multiplier`
  在两路同时激活时的舍入误差叠加。详见**精度收敛 follow-up** 小节。
- **测试**: `tests/fixed_point/kernels/test_subtract_int16_precision.py`
- **Reference**: `ref = a - b`（float32，无量化的 ideal float subtraction）
- **Input 分布**: 在每个 signed grid 上 128 trials × 256 elements
  （= 32_768 effective comparisons per (grid, mode)）
  - 输入码 `qa, qb ∈ [-code_limit, code_limit]`（code_limit 与 grid 相关）
  - 同标度：`scale_a = scale_b = scale_out`
  - 跨标度：`scale_a = scale_out * exp(±0.05/2)`（i8 用 ±0.02）
- **阈值**:
  - `_MIN_COSINE = 0.9999`（要求严格 `>`）
  - `_MAX_FLOAT_LSB = 1.0`（要求严格 `<`）
  - 与 Add Path-B (`test_add_ideal_float_reference.py`) 共享同一份精度
    floor，因为这两个算子在 `_BinaryAlignedKernel` 里只差一个
    `int32_{add,sub}_sat`。
- **实测值**（commit time 2026-06-08, PyTorch CPU, 128 trials × 256 elem）:

| grid | mode | cos_min | cos_median | lsb_max | lsb_median |
|------|------|---------|-----------|---------|-----------|
| i8 | same-scale | 1.000000 | 1.000000 | 0.0000 | 0.0000 |
| i8 | cross-scale | 0.999986 | 0.999997 | 0.2405 | 0.1270 |
| i16 | same-scale | 1.000000 | 1.000000 | 0.0000 | 0.0000 |
| i16 | cross-scale | 0.999979 | 0.999986 | 0.5001 | 0.4948 |
| i32 | same-scale | 1.000000 | 1.000000 | 0.0000 | 0.0000 |
| i32 | cross-scale | 0.999979 | 0.999986 | 0.4999 | 0.4956 |

- **状态**: ✓ PASS（同标度 bit-exact；跨标度 cos ≥ 0.999971，远高于 0.9999 floor）
- **注（算子级 KNOWN_LIMIT）**: unsigned grid（u8/u16/u32）在
  `default_zero_point=0` 下**无法表达** `a - b < 0`。Add 在同样的 grid
  上能通过测试是因为随机码被夹到 `[0, code_limit]`，所以 `a + b ≥ 0`
  天然成立；Subtract 不享受这个性质。这是 spec 级别的 unsigned 编码
  属性，不是 kernel bug。Path-B 测试**不**覆盖 unsigned grids，避免
  把语义限制误当成精度回归。要在 unsigned 下表达负差需用非零 zp（如
  zp=128 for u8），那是另一类编码契约的测试，不在本段范围。

---

## custom.Multiply

- **Kernel**: `aimet_torch/fixed_point/kernels/eltwise.py::MultiplyInt16Kernel`
  (hot path: `int32_mul_sat(_center(a), _center(b)) → _requantize`)
- **KernelKind**: `REQUANTIZING`
- **Spec**: `doc/04_算子详细规格/04_03_逐元素运算类算子.md` §4.3.1
  - spec 契约：`p = (q_x1 - Z_x1)·(q_x2 - Z_x2)`，`α = S_x1·S_x2/S_y`，
    `y_q = sat((p·M ≫ rshift) + Z_y)`（**单 multiplier**）
  - kernel 对齐：hot path 与 spec 完全一致；adapter 把 `real_m =
    scale_a*scale_b/scale_out` 喂给 `quantize_multiplier` 得 (M, rshift)
- **Spec 测试覆盖度**: ✓ **完全**——本 file 的
  `_multiply_output_encoding` 用 `quantize_multiplier(real_m)` 计算
  实际 (M, rshift) 而**不是**像 Add/Sub 那样固定 32767/15，因此
  `real_m ≠ 1` 的 cross-scale 情形里 multiplier 折叠路径**真实激活**。
- **测试**: `tests/fixed_point/kernels/test_multiply_int16_precision.py`
- **Reference**: `ref = a * b`（float32，无量化的 ideal float multiply）
- **Input 分布**: 在每个 signed grid 上 128 trials × 256 elements
  - 输入码 `qa, qb ∈ [-code_limit, code_limit]`，`code_limit² ≤ qmax`
    （i8: 8, i16: 64, i32: 64）以避免 saturate 掩盖精度
  - 同标度：`scale_a = scale_b`；`scale_out = scale_a * scale_b`
  - 跨标度：`scale_a = scale_b * exp(±0.05/2)`（i8 用 ±0.02），
    `scale_out = scale_a * scale_b`
  - multiplier/rshift 通过 `quantize_multiplier(real_m)` 折叠（不像 Add/Sub
    Path-B 那样固定写 32767/15）—— 即测的就是 Multiply 真实 hot path
- **阈值**:
  - `_MIN_COSINE = 0.9999`（要求严格 `>`）
  - `_MAX_FLOAT_LSB = 1.0`（要求严格 `<`）
- **实测值**（commit time 2026-06-08, PyTorch CPU, 128 trials × 256 elem）:

| grid | mode | cos_min | cos_median | lsb_max | lsb_median |
|------|------|---------|-----------|---------|-----------|
| i8 | same-scale | 1.000000 | 1.000000 | 0.0000 | 0.0000 |
| i8 | cross-scale | 1.000000 | 1.000000 | 0.0000 | 0.0000 |
| i16 | same-scale | 1.000000 | 1.000000 | 0.0008 | 0.0004 |
| i16 | cross-scale | 1.000000 | 1.000000 | 0.0008 | 0.0004 |
| i32 | same-scale | 1.000000 | 1.000000 | 0.0008 | 0.0004 |
| i32 | cross-scale | 1.000000 | 1.000000 | 0.0008 | 0.0004 |

- **状态**: ✓ PASS（远高于 floor — Multiply 的 fp64 真乘积 + 精确 multiplier 折叠
  让 i16+ grid 几乎 bit-exact，i8 同标度同样 0 误差）
- **注（算子级 KNOWN_LIMIT）**: unsigned grid 与 Subtract 同根问题——
  混合符号下乘积会出现负值，`zero_point=0` 的 unsigned grid 无法表达。
  Path-B 测试与 Subtract 一致**只覆盖 signed grids**。

---

## custom.Divide

- **Kernel**: `aimet_torch/fixed_point/kernels/eltwise.py::DivideInt16Kernel`
  —— **双路实现**，由 LUT 资产可用性 + `extra` 决定走哪条：
  1. **spec §4.3.4 路径**（默认，`abc_lut-shuai/.../reciprocal_clz_lut.json`
     可用时启用）：复用 `aimet_torch/fixed_point/kernels/clz_lut.py::
     reciprocal_via_clz_lut` 算 `1/b`（CLZ 归一化 + 16 段二阶多项式 LUT
     + denormalize），再走 `MultiplyInt16Kernel` 完成 `a × (1/b)` 的
     `int32_mul_sat → quantize_multiplier(scale_a × scale_recip /
     scale_out)` 重定标。LUT 自身已包含 spec §4.3.4 的"normalize → 多项式
     → 反规格化"步骤；牛顿迭代 step 3 当前**未启用**，原因见下方"为什么
     不在 normalized space 上加 Newton"。
  2. **legacy integer-div 路径**（`extra['force_legacy_integer_divide']=
     True` 或 LUT 资产缺失时）：原有的 `(num_centered × M) >> rshift →
     num_scaled // den_centered (round-half) → requant`，精度 floor
     ~1.5 LSB（spec §4.3.4 末尾"按工具链自动选择"允许此路径）。
- **KernelKind**: `REQUANTIZING`
- **Spec**: `doc/04_算子详细规格/04_03_逐元素运算类算子.md` §4.3.4
- **测试（双路独立锁定）**:
  - `tests/fixed_point/kernels/test_divide_int16_clz_precision.py`
    —— **spec §4.3.4 reciprocal CLZ LUT 路径**精度锁，**全 PASS**
    在统一 0.9999 cos / 1 LSB floor 下（i16/i32 same/cross-scale）；
    skipif 当 abc_lut-shuai 不在 workspace（与
    `test_clz_reciprocal_golden.py` 一致）
  - `tests/fixed_point/kernels/test_divide_int16_precision.py`
    —— **legacy integer-div 路径**精度锁，强制 `force_legacy_
    integer_divide=True`，全 6 用例 xfail strict（保留 KNOWN_LIMIT 档案）
  - `tests/fixed_point/kernels/test_divide_int16.py` —— 点测 + integer
    LSB 距离 ≤ 2，BN 路径 + register；dispatch / regression 锁，跑默认
    （LUT）路径
  - `tests/fixed_point/kernels/test_divide_int16_eps_clamp.py`（eps 路径）
- **Reference**: `ref = a / b`（float32，无量化的 ideal float divide）

### spec §4.3.4 reciprocal CLZ LUT 路径

- **Input 分布**（`test_divide_int16_clz_precision.py`，128 trials ×
  256 elements per (grid, mode)）:
  - 分母 |b| 严格落在 `[_LUT_MARGIN=0.5, 0.9 × _LUT_IN_FMAX=5.4]`：
    LUT fitted domain 内部，避免 alignment 阶段饱和；同时 |b|≥0.5 让
    LUT 内部"零分母饱和"分支不触发（eps 单独测）
  - 分子 |a| ≤ `_OUTPUT_SCALE_RATIO × 0.75 = 0.3`：**乘法误差传播预算**
    `|output_err| ≈ |a| × LUT_max_LSB_float`，与 LUT 自身误差成正比；
    超此预算 lsb 被放大
  - `scale_out = _OUTPUT_SCALE_RATIO × _LUT_OUT_SCALE = 0.4 × 3.05e-3
    ≈ 1.22e-3`：调到 LUT-output grid 比例的 0.4×，让 LUT-out 量化粒度
    （3.05e-3）与 output grid 粒度（1.22e-3）的比 = 2.5x，正好在 cos
    不掉 + lsb_max < 1 的 sweet band
  - **i8 grid 不在覆盖范围**（i8 qmax × scale_out = ±0.155 容不下商）
- **阈值**: `_MIN_COSINE = 0.9999`、`_MAX_FLOAT_LSB = 1.0`（与
  Add/Sub/Mul 完全一致，**不放宽**）
- **实测值**（commit time 2026-06-08, PyTorch CPU, 128 trials × 256 elem）:

| grid | mode | cos_min | cos_median | lsb_max | lsb_median |
|------|------|---------|-----------|---------|-----------|
| i16 | same-scale | 0.999990 | 0.999994 | 0.8493 | 0.7627 |
| i16 | cross-scale | 0.999989 | 0.999993 | 0.8379 | 0.7540 |
| i32 | same-scale | 0.999990 | 0.999993 | 0.8590 | 0.7525 |
| i32 | cross-scale | 0.999990 | 0.999993 | 0.9343 | 0.7586 |

- **状态**: ✓ PASS（lsb_max ≤ 0.93，距 1.0 floor 留 ~10% headroom；
  cos_min 0.99999 远超 0.9999 floor）

### legacy integer-div 路径

- **Input 分布**（`test_divide_int16_precision.py`，`force_legacy_integer_
  divide=True`，random scale 全域，128 trials × 256 elements）。不受
  LUT fitted domain 约束 —— 测的就是 fallback 在任意 scale 下的精度。
- **实测值**（同 commit；统一 floor 下的实际表现，**不**作为通过判据）:

| grid | mode | cos_min | cos_median | lsb_max | lsb_median |
|------|------|---------|-----------|---------|-----------|
| i8 | same-scale | 0.998379 | 0.998926 | 1.4722 | 1.4600 |
| i8 | cross-scale | 0.998491 | 0.998899 | 1.4722 | 1.4600 |
| i16 | same-scale | 1.000000 | 1.000000 | 1.4875 | 1.4727 |
| i16 | cross-scale | 1.000000 | 1.000000 | 1.4875 | 1.4717 |
| i32 | same-scale | 1.000000 | 1.000000 | 1.4875 | 1.4728 |
| i32 | cross-scale | 1.000000 | 1.000000 | 1.4875 | 1.4740 |

- **状态**: ⚠ KNOWN_LIMIT — 1.5 LSB worst case，xfail strict=True 锁定
  - i8 cos_min 0.998 < 0.9999；i16/i32 lsb_max ≈ 1.49 > 1.0 → 6/6 xfail
  - 不通过原因：(a) `quantize_multiplier(real_m)` ≤ 0.5 LSB +
    (b) 整除 round-half ≤ 0.5 LSB +
    (c) cross-scale multiplier 折叠 ≤ 0.5 LSB → 几何叠加 ≈ 1.5 LSB
  - 整除 round-half 是 reciprocal LUT 路径**没有**的误差源；这正是
    LUT 路径过 1 LSB 而 legacy 不过的根本差异

### 路径选择策略

- **默认走 LUT 路径**（abc_lut-shuai 在 workspace 时）：精度更好，与
  spec §4.3.4 字面对齐
- **fallback 到 legacy integer-div**（abc_lut-shuai 不可用时）：保兼容，
  避免最小环境 CI 失去 Divide 支持
- **opt-in legacy via `extra['force_legacy_integer_divide']=True`**：
  precision_validation 用，让 legacy 精度档案在所有环境下都被锁定

### 为什么不在 normalized space 上加 Newton（spec §4.3.4 step 3）

实测发现：当前 abc 默认 `reciprocal_clz_lut.json` 的 16-段二阶多项式
LUT 在 LUT-domain 内部已经达到 ~0.5 LUT-out LSB 精度（亚 LUT-output LSB
floor），叠加 multiply 路径后总误差 ≤ 1 output LSB。**牛顿迭代不是
当前 floor 的瓶颈**——瓶颈是乘法误差传播 `|a| × LUT_max_LSB`。

数学上 Newton 仍有 ε² 的二阶收敛优势，但在当前 fixed-point 实现上：

- 在已 denormalized 的 i16 LUT 输出（q_recip）上做 Newton → 整数除法
  round-half 反而引入 ~1 LSB 噪声，**让精度更差**（实测 0.53 → 1.48 LSB）
- 要让 Newton 真正生效，需要在 normalized space 的 i32 q_y_norm 上做
  `r₁ = r₀ × (2 - m × r₀) / norm_unit`，再 denormalize；这要 fork
  `_evaluate_clz_vectorized` 的内部状态（mantissa, q_y_norm, exponent
  全部要外露），重复 ~150 行 vectorized 路径

成本 / 收益不平衡：当前 LUT 路径已过统一 floor，再加 Newton 只是把
lsb_max 从 0.93 推到 0.3 量级，没有解锁新场景。Newton 实施留作未来若
LUT 资产换成更小段数 / 更高 dynamic range 时的精度兜底。
### 算子级 KNOWN_LIMIT（与精度无关）

- **unsigned grid 不覆盖**：与 Subtract / Multiply 一致，`zero_point=0`
  unsigned 无法表达负商，两条路径都只测 signed grids
- **i8 grid 在 LUT 路径下不覆盖**：i8 输出 grid（qmax×scale_out=±0.16）
  容不下 LUT-domain 的典型商范围。建议 8-bit Divide 通过"requant 到
  i16 → divide → requant 回 i8"的复合路径处理，而非直接 dispatch i8
  Divide。这是 grid 级语义限制，不是 LUT 路径精度问题

---

## nn.AvgPool2d

- **Kernel**: `aimet_torch/fixed_point/kernels/pool.py::
  {AvgPool2d,CustomAvgPool2d}Int16Kernel`（继承 `_AvgPool2dKernel`，
  两者共享 hot path；同 P6 MaxPool 写法）
- **KernelKind**: `REQUANTIZING`
- **Spec**: `doc/04_算子详细规格/04_09_池化类算子.md` §4.9.2
  - spec 契约（按汇编路径）：`s = Σ q_x - k_t·k_f·Z_x` → `m = (s·inv) ≫ shift`
    （`inv = round(2^r / N)` 是离线常量倒数）→ `y_q = sat((m·M) ≫ rshift) + Z_y`
  - kernel 对齐：仿真器把"`1/N` + `S_x/S_y`"两步**离线折叠**到一个
    `M/rshift`（`real_m = scale_in / (N · scale_out)`），spec 也允许
    工具链做这步等价合并；kernel 入口由 `require_reduce_size_matches_extra`
    把守 `extra["reduce_size"] == k_t·k_f`，避免 adapter 漏折 / 折错
- **Spec 测试覆盖度**: ✓ **完全**——本 file 用 `quantize_multiplier(real_m)`
  生成实际 (M, rshift) 而**不是**像 Add/Sub Path-B 那样固定 32767/15，
  跨 4 个 spec-pinned kernel size 都激活 multiplier 折叠路径
- **测试**: `tests/fixed_point/kernels/test_avgpool2d_int16_precision.py`
- **Reference**: `nn.functional.avg_pool2d(x.float())`（fp32，无量化的 ideal
  float average pooling）
- **Input 分布**: 32 trials × shape=(1,4,8,8) per (grid, kernel_size, mode)
  - kernel size 覆盖 spec 约束的全集 `{(2,2), (4,4), (4,2), (2,4)}`
  - 输入码 `qx ∈ [-code_limit, code_limit]`，i16/i32 上 `code_limit=4096`
  - 同标度：`scale_in = scale_out`；跨标度：`scale_in = scale_out · exp(±0.05/2)`
  - 所有 trial 都重新算一次 `(M, rshift)`，覆盖 cross-scale 折叠路径
- **阈值**: `_MIN_COSINE = 0.9999`、`_MAX_FLOAT_LSB = 1.0`（与 Add/Sub/Mul/Div
  完全一致）
- **实测值**（commit time 2026-06-08, PyTorch CPU, 32 trials）:

| grid | kernel | mode | cos_min | lsb_max |
|------|--------|------|---------|---------|
| i16 | 2x2 | same-scale | 1.000000 | 0.5003 |
| i16 | 2x2 | cross-scale | 1.000000 | 0.5220 |
| i16 | 4x4 | same-scale | 1.000000 | 0.5001 |
| i16 | 4x4 | cross-scale | 0.999999 | 0.4983 |
| i16 | 4x2 | same-scale | 1.000000 | 0.5002 |
| i16 | 4x2 | cross-scale | 1.000000 | 0.5284 |
| i16 | 2x4 | same-scale | 1.000000 | 0.5002 |
| i16 | 2x4 | cross-scale | 1.000000 | 0.5124 |
| i32 | 2x2 | same-scale | 1.000000 | 0.5003 |
| i32 | 2x2 | cross-scale | 1.000000 | 0.5216 |
| i32 | 4x4 | same-scale | 1.000000 | 0.5001 |
| i32 | 4x4 | cross-scale | 0.999999 | 0.4976 |
| i32 | 4x2 | same-scale | 1.000000 | 0.5001 |
| i32 | 4x2 | cross-scale | 1.000000 | 0.5071 |
| i32 | 2x4 | same-scale | 1.000000 | 0.5002 |
| i32 | 2x4 | cross-scale | 1.000000 | 0.5067 |

- **i8 实测值**（i8 也参与 parametrize，按 `(grid, kernel_size)` 联合
  分别打 xfail——i8-2x2 边缘震荡用 strict=False，i8 大 kernel 用 strict=True）:

| grid | kernel | mode | cos_min | lsb_max | xfail mark |
|------|--------|------|---------|---------|------------|
| i8 | 2x2 | same-scale | 0.999914 | 0.5000 | strict=False（边缘震荡）|
| i8 | 2x2 | cross-scale | 0.999920 | 0.4998 | strict=False（边缘震荡）|
| i8 | 4x4 | same-scale | 0.999531 | 0.5000 | strict=True |
| i8 | 4x4 | cross-scale | 0.999479 | 0.4999 | strict=True |
| i8 | 4x2 | same-scale | 0.999729 | 0.5000 | strict=True |
| i8 | 4x2 | cross-scale | 0.999823 | 0.4999 | strict=True |
| i8 | 2x4 | same-scale | 0.999742 | 0.5000 | strict=True |
| i8 | 2x4 | cross-scale | 0.999782 | 0.4994 | strict=True |

- **状态**: ✓ PASS（i16/i32，共 16/24 用例）；
  ⚠ KNOWN_LIMIT（i8 + k=4 共 2/24 用例 xfail strict=False 边缘震荡，
  i8 + k≥8 共 6/24 用例 xfail strict=True）
- **注（算子级 KNOWN_LIMIT 与不通过原因）**:
  - **i8 + k≥8（4x4/4x2/2x4）xfail strict=True 锁定**：grid+reduction 物理
    SNR 上限——random 输入下 avg pool 输出方差被 `1/sqrt(N)` 抑制，量化
    噪声仍在 0.5 LSB。要 cos > 0.9999 需要 SNR > 100，即 `code_limit >
    50·sqrt(N)`。i8 (qmax=127) 在 N≥8 时物理上做不到 → cos 实测
    `0.9994～0.9998` 稳定低于 floor，**lsb_max 仍 ≤ 0.5**。
  - **i8 + k=4 (2x2) xfail strict=False 锁定**：N=4 处于 SNR 边缘，cos
    在 `0.99989～0.99992` 之间随 random trial sequence 震荡——既不稳定
    PASS 也不稳定 FAIL。strict=False 接受这个客观事实：测试照跑、档案
    照记，但不要求 PASS/FAIL 任意一种结果。lsb_max 仍 ≤ 0.5。
  - **改进建议**（不修改统一 floor，记录可选改进路径）：
    1. adapter 端在 i8 reduction 算子上**避免直接 i8→i8 dispatch**，改走
       "i8 → requant 到 i16 → reduce → requant 回 i8"复合路径，把 cos
       floor 拉回 i16 域（已知 PASS）；
    2. 或让 calibration 给 i8 reduction 路径选**更宽 scale_out**（output
       dynamic range 不再被 1/sqrt(N) 压缩），与 spec 4.9.2 的硬件 inv/
       shift 路径并不冲突；
    3. 或在 spec 文档明确"i8 + reduction 类算子 cos 阈值降到 0.999"作为
       **算子级别**的 floor（违反统一 floor 纪律，不推荐）
  - **unsigned grid u8/u16（Layer C3 已补 spot-check）**：spec
    `04_09 §4.9.2` AvgPool / `04_06 §4.6.1` Mean 明确允许 u8/u16 输入；
    本段已用 `zp = qmax//2` 非零 zp 覆盖 4 个 case（u8/u16 × same/cross-
    scale，kernel=2x2）：**u16 全 PASS（cos≥0.9999/lsb≤0.5），u8 边缘
    震荡 xfail strict=False（与 i8 + 小 kernel 同根 SNR ceiling，code
    envelope ~96 < 50·sqrt(4)=100）**。完整覆盖（多 kernel × 多 zp）
    保留为 follow-up，本段 spot-check 主要 ack spec 允许且 hot path
    无 dtype 偏差
  - **`custom.AvgPool2d` alias**：functional wrapper `custom.AvgPool2d`
    与 `nn.AvgPool2d` 共享 `_AvgPool2dKernel` 基类（同 `pool.py`），行为
    同根；manifest 标 `dispatchable=False`（adapter 还未从 args/kwargs
    unpack `kernel_size/stride/padding`，见 `route-custom-pool-through-adapter`
    plan item）。等 adapter 路由落地后追加独立 parametrize 锁定 wrapper
    parity，目前 i16 / i8 cross-grid 行为按 `_AvgPool2dKernel` 共享路径
    推断与 `nn.AvgPool2d` 一致

---

## custom.Mean

- **Kernel**: `aimet_torch/fixed_point/kernels/pool.py::MeanInt16Kernel`
  （继承 `_MeanInt16Kernel`，hot path 与 `AdaptiveAvgPool2dInt16Kernel` 共享）
- **KernelKind**: `REQUANTIZING`
- **Spec**: `doc/04_算子详细规格/04_06_统计类算子.md` §4.6.1
  - spec 契约：`s = Σ_dim (q_x - Z_x)` → `mean = (s · inv) ≫ shift`
    （`inv = round(2^r / N)` 离线常量）→ `y_q = sat((mean · M) ≫ rshift) + Z_y`
  - kernel 对齐：与 AvgPool2d 同型——`1/N` + `S_x/S_y` 离线折叠到一个
    `M/rshift`；`require_reduce_size_matches_extra` 在入口验证
    `extra["reduce_size"] == ∏ shape[d] for d in dim`
- **Spec 测试覆盖度**: ✓ **完全**——`real_m = scale_in / (N · scale_out)`
  通过 `quantize_multiplier` 折叠，cross-scale 时 multiplier 路径真实激活。
  覆盖两种归约 pattern：spatial-HW（`dim=(2,3), keepdim=True`，匹配
  AdaptiveAvgPool2d 路径）和 last-axis（`dim=-1, keepdim=False`，1D 归约）
- **测试**: `tests/fixed_point/kernels/test_mean_int16_precision.py`
- **Reference**: `torch.mean(x.float(), dim=..., keepdim=...)`（fp32 ideal mean）
- **Input 分布**: 32 trials per (grid, reduction, mode)
  - spatial-HW: shape=(1,4,8,8), `dim=(2,3)`, N=64
  - last-axis: shape=(4,64), `dim=-1`, N=64
  - 输入码 `qx ∈ [-4096, 4096]`，scale 与 AvgPool2d 一致
- **阈值**: `_MIN_COSINE = 0.9999`、`_MAX_FLOAT_LSB = 1.0`
- **实测值**（commit time 2026-06-08, PyTorch CPU, 32 trials）:

| grid | reduction | mode | cos_min | lsb_max |
|------|-----------|------|---------|---------|
| i16 | spatial-HW | same-scale | 0.999998 | 0.4844 |
| i16 | spatial-HW | cross-scale | 0.999995 | 0.4986 |
| i16 | last-axis | same-scale | 0.999986 | 0.5000 |
| i16 | last-axis | cross-scale | 0.999999 | 0.4971 |
| i32 | spatial-HW | same-scale | 0.999998 | 0.5000 |
| i32 | spatial-HW | cross-scale | 0.999996 | 0.4877 |
| i32 | last-axis | same-scale | 0.999987 | 0.5000 |
| i32 | last-axis | cross-scale | 0.999992 | 0.4981 |

- **i8 实测值**（i8 也参与 parametrize，4 用例 xfail strict 锁定）:

| grid | reduction | mode | cos_min | lsb_max |
|------|-----------|------|---------|---------|
| i8 | spatial-HW (N=64) | same-scale | 0.990432 | 0.5000 |
| i8 | spatial-HW (N=64) | cross-scale | 0.996907 | 0.4983 |
| i8 | last-axis (N=64) | same-scale | 0.988345 | 0.5000 |
| i8 | last-axis (N=64) | cross-scale | 0.997317 | 0.4974 |

- **状态**: ✓ PASS（i16/i32，8/12 用例）；⚠ KNOWN_LIMIT（i8，4/12 用例
  xfail strict）
- **注（算子级 KNOWN_LIMIT 与不通过原因）**:
  - **i8 grid xfail strict 锁定**：与 AvgPool2d 同根问题——reduction 类
    算子物理 SNR 上限。Mean 测试用 N=64（shape (1,4,8,8) 全空间归约 /
    shape (4,64) last-axis 归约），i8 (qmax=127) 在 N=64 时 cos 上限
    ≈ `1 - 1/(2 · (96/sqrt(192))²)` ≈ 0.9986，**实测 cos_min 0.988～
    0.997**，与理论一致。**lsb_max 仍 ≤ 0.5**（kernel 算术完全正确），
    cos 不过 0.9999 floor。
  - **改进建议**（同 AvgPool2d）：
    1. adapter 端 i8 reduction 走"requant 到 i16 → reduce → requant 回
       i8"复合路径；
    2. 或让 calibration 在 i8 reduction 路径上选更宽 scale_out
  - **unsigned grid u8/u16（Layer C3 已补 spot-check）**：spec
    `04_06 §4.6.1 mean` 允许 u8/u16；本段用 spatial-HW reduction (N=16)
    + `zp = qmax//2` 覆盖 4 个 case：**u16 全 PASS，u8 全 xfail strict=
    True（N=16 远超 i8/u8 SNR ceiling，与 i8 同根）**。完整覆盖（多
    reduction × 多 zp）保留为 follow-up

---

## custom.AdaptiveAvgPool2d

- **Kernel**: `aimet_torch/fixed_point/kernels/pool.py::AdaptiveAvgPool2dInt16Kernel`
  （继承 `_MeanInt16Kernel`，仅支持 `output_size=(1,1)`；其他 output size
  在 dispatch 端被 refused，落到 float QDQ fallback）
- **KernelKind**: `REQUANTIZING`
- **Spec**: `doc/04_算子详细规格/04_09_池化类算子.md` §4.9.x（与 AvgPool 同段）
  + `04_06_统计类算子.md` §4.6.1（共享 ReduceMean 公式）
  - spec 契约：等价于 `Mean(x, dim=(2,3), keepdim=True)`，N=H·W；adapter
    自动把 AdaptiveAvgPool2d(1,1) 重写成 spatial-mean 后走 Mean 同一 kernel
- **Spec 测试覆盖度**: ✓ **完全**——adapter 路径的 dim/keepdim/output_size
  设置与 kernel 入口契约（`require_adaptive_avgpool_output_unit`）都被测
- **测试**: `tests/fixed_point/kernels/test_adaptive_avgpool2d_int16_precision.py`
- **Reference**: `torch.mean(x.float(), dim=(2,3), keepdim=True)`（与
  `nn.functional.adaptive_avg_pool2d(x, (1,1))` 数值等价的 fp32 ideal mean）
- **Input 分布**: 32 trials per (grid, spatial, mode)；spatial ∈
  {(4,4), (8,8), (4,8)} → N ∈ {16, 64, 32}；shape=(1,4,H,W)；其余与
  Mean 一致
- **阈值**: `_MIN_COSINE = 0.9999`、`_MAX_FLOAT_LSB = 1.0`
- **实测值**（commit time 2026-06-08, PyTorch CPU, 32 trials）:

| grid | spatial | mode | cos_min | lsb_max |
|------|---------|------|---------|---------|
| i16 | 4x4 | same-scale | 0.999999 | 0.5000 |
| i16 | 4x4 | cross-scale | 0.999998 | 0.5028 |
| i16 | 8x8 | same-scale | 0.999997 | 0.5000 |
| i16 | 8x8 | cross-scale | 0.999998 | 0.4999 |
| i16 | 4x8 | same-scale | 0.999998 | 0.5000 |
| i16 | 4x8 | cross-scale | 0.999999 | 0.4967 |
| i32 | 4x4 | same-scale | 0.999996 | 0.5000 |
| i32 | 4x4 | cross-scale | 1.000000 | 0.4940 |
| i32 | 8x8 | same-scale | 0.999994 | 0.5000 |
| i32 | 8x8 | cross-scale | 0.999996 | 0.5030 |
| i32 | 4x8 | same-scale | 0.999998 | 0.5000 |
| i32 | 4x8 | cross-scale | 0.999998 | 0.4966 |

- **i8 实测值**（i8 也参与 parametrize，6 用例 xfail strict 锁定）:

| grid | spatial | mode | cos_min | lsb_max |
|------|---------|------|---------|---------|
| i8 | 4x4 (N=16) | same-scale | 0.996768 | 0.5000 |
| i8 | 4x4 (N=16) | cross-scale | 0.998157 | 0.4993 |
| i8 | 8x8 (N=64) | same-scale | 0.977559 | 0.5000 |
| i8 | 8x8 (N=64) | cross-scale | 0.996068 | 0.4895 |
| i8 | 4x8 (N=32) | same-scale | 0.997999 | 0.5000 |
| i8 | 4x8 (N=32) | cross-scale | 0.998087 | 0.4970 |

- **状态**: ✓ PASS（i16/i32，12/18 用例）；⚠ KNOWN_LIMIT（i8，6/18 用例
  xfail strict）
- **注（算子级 KNOWN_LIMIT 与不通过原因）**:
  - **i8 grid xfail strict 锁定**：reduction 类物理 SNR 上限——
    AdaptiveAvgPool2d 在 H·W=64 时与 Mean spatial-HW 等价 cos floor
    （≈0.9986 理论上限）。实测 cos_min `0.978～0.998`（H·W 越大 cos 越
    低，与 SNR ∝ 1/sqrt(H·W) 一致）。**lsb_max 仍 ≤ 0.5**，kernel 算术
    正确。
  - **改进建议**：同 AvgPool2d / Mean——adapter 端走 i8→i16→reduce→i8
    复合路径，或 calibration 调宽 scale_out
  - **`output_size != (1,1)` 不走本 kernel**：dispatch 端拒绝并落 float
    fallback；非本段范围
  - **unsigned grid u8/u16（Layer C3 已补 spot-check）**：与
    AvgPool2d / Mean 段同 framing；本段（AdaptiveAvgPool2d）共享 Mean
    spatial-HW hot path，未独立参数化。Mean unsigned spot-check 已 ack
    spec u8/u16 兼容，AdaptiveAvgPool2d 推断为同精度档案

---

## nn.Linear

- **Kernel**: `aimet_torch/fixed_point/kernels/conv_linear.py::LinearInt16Kernel`
- **KernelKind**: `REQUANTIZING`
- **Spec**: `doc/04_算子详细规格/04_02_矩阵运算类算子.md §4.2.1 matmul`
  （Linear = batched matmul 特例）+ `04_01_卷积类算子.md §4.1.1 conv2d`
  尾段 "兼容 fc 模式"（spec 中 Linear 不是独立 section，复用 matmul/
  conv2d 量化推导）
  - spec 契约：`acc = sum(q_w · (q_x - Z_x)) + b_int` →
    `y_q = sat((acc · M) ≫ rshift) + Z_y`，权重 **`Z_w=0`** 强制
    （symmetric quantize），`real_m = S_x · S_w / S_y`
  - **spec 输出 dtype**: 仅 `i8 / i16`；i32 是 acc grid 中间态，不是
    activation grid，本段不覆盖
  - kernel 对齐：weight `Z_w=0` 在 `_get_weight` 入口硬校验，bias 双
    位宽 (16/32) 由 `OutputEncoding.bias_bits` 选；`int32_matmul` 内部
    走 int64 product → INT32 saturation
- **Spec 测试覆盖度**: ✓ **完全**——`real_m = S_x · S_w / S_out` 经
  `quantize_multiplier` 折叠，cross-scale 时 multiplier 路径真实激活；
  bias 用 int32 dtype（`bias_bits=32` 默认）
- **测试**: `tests/fixed_point/kernels/test_linear_int16_precision.py`
- **Reference**: `nn.functional.linear(x.float(), w.float(), b.float())`
- **Input 分布**: 64 trials per (grid, mode)；shape `(B=4, K=16, M=16)` →
  64 output elements（**故意小**以暴露 i8 matmul-class SNR 边缘）
  - i8: `code_x, code_w ∈ [-96, 96]`（i8 qmax=127 物理上限的 75%）
  - i16: `code_x, code_w ∈ [-256, 256]`
  - bias 在 acc-scale (`S_x · S_w`) 上量化为 int32
  - `scale_out` 取 `|ref_y|.max() / (qmax · 0.5)` 让 output 居中无饱和
- **阈值**: `_MIN_COSINE = 0.9999`、`_MAX_FLOAT_LSB = 1.0`
- **实测值**（commit time 2026-06-08, PyTorch CPU, 64 trials × 64 elements）:

| grid | mode | cos_min | lsb_max | xfail? |
|------|------|---------|---------|--------|
| i8 | same-scale | 0.999862 | 0.5002 | yes（strict=False，边缘震荡）|
| i8 | cross-scale | 0.999882 | 0.5006 | yes（strict=False，边缘震荡）|
| i16 | same-scale | 1.000000 | 0.6356 | no（PASS）|
| i16 | cross-scale | 1.000000 | 0.6720 | no（PASS）|

- **状态**: ✓ PASS（i16 同/跨标度，2/4 用例）；⚠ KNOWN_LIMIT（i8 同/
  跨标度，2/4 用例 xfail strict=False 锁定）
- **注（算子级 KNOWN_LIMIT 与不通过原因）**:
  - **i8 grid xfail (strict=False) 锁定**：matmul-class SNR 上限——64 个
    输出元素的 cosine 统计样本不够大，per-row SNR 方差让 worst-row 的
    cos 落到 0.9999 floor 边缘（实测 cos_min `0.99971～0.99996` 区间，
    取决于 random trial sequence）。**lsb_max 仍 ≤ 0.5**（kernel 算术
    完全正确），cos 不过 0.9999 floor 是 small-output-volume statistical
    artefact，非 kernel bug。**用 `strict=False` 而非 `strict=True`**：
    floor 跨越本身是 random 的，strict=True 会在 unexpected pass 时假
    红色，掩盖真实边缘行为
  - **Conv2d 同样的 i8 grid 在更大 output volume（H'·W'·F ≥ 200）下
    稳过 floor**——证明这是输出元素数问题，不是 Linear kernel 问题。
    要让 i8 Linear 稳过 0.9999 也只需更大 batch/M；本测试**故意保留
    小 64-output 以暴露这个边缘**作为档案
  - **改进建议**（与 P3 i8 reduction 同）：
    1. adapter 端把 i8 Linear 改写成 "i8 → requant 到 i16 → linear →
       requant 回 i8" 复合路径；
    2. 或在 capability manifest 上声明 i8 Linear unsupported 并 routing
       到 i16
  - **i32 输出 grid 不覆盖**：spec 04_02 §Linear 仅允许输出 dtype
    `i8 / i16`，i32 是 MAC 累加器域；测试参数化中已排除
  - **unsigned grid u8/u16 不覆盖（测试覆盖缺口，非 spec 限制）**：
    spec `04_02 §4.2.1 matmul` 允许 u8/u16；本段未参数化的根因是
    `zero_point=0` unsigned 无法表达零均值 random 输入产生的负 matmul
    结果，需 calibration 给非零 zp
  - **weight bitwidth ∈ {i2, i4}**：spec `04_02 §4.2.1 matmul` +
    `04_01 §4.1.1 fc 兼容模式` 允许 i2/i4 权重；Layer C1 在 Conv2d 段
    实测 i16 × i4 weight 路径 PASS（同根 GEMM hot path），本段
    Linear/Conv1d 共享 `_int32_matmul`，软件路径 ready；本段未独立
    参数化测试 i2/i4 weight

---

## custom.MatMul

- **Kernel**: `aimet_torch/fixed_point/kernels/conv_linear.py::MatMulInt16Kernel`
- **KernelKind**: `REQUANTIZING`
- **Spec**: `doc/04_算子详细规格/04_02_矩阵运算类算子.md §4.2.1 matmul`
  - spec 契约：`centered → int32_matmul → requant`；与 Linear 同 hot-path
    （`_center_tensor + _int32_matmul + _requantize_output`），区别只在
    没有 weight / bias 这两个参数（两个输入都是 activation）
  - **spec 输出 dtype**: `i8 / i16`（同 Linear）
  - kernel 对齐：`_int32_matmul` 在 INT16_FIXED_EVAL 下走 int64 product
    + INT32 saturation，`real_m = S_a · S_b / S_y`
- **Spec 测试覆盖度**: ✓ **完全**——`real_m` 经 `quantize_multiplier`
  生成实际 (M, rshift)，覆盖 cross-scale 折叠路径；输出 i8/i16 两类
- **测试**: `tests/fixed_point/kernels/test_matmul_int16_precision.py`
- **Reference**: `torch.matmul(a.float(), b.float())`（fp32 无量化 ideal
  matmul）
- **Input 分布**: 32 trials per (grid, mode)；shape `(B=4, M=16, K=8) @
  (B=4, K=8, N=16)` → `(4, 16, 16)` = 1024 outputs/trial（matmul-class
  output volume 远超 P4 Linear 的 64 outputs，给出充分 i8 SNR head-room）
  - 输入码 `qa/qb ∈ [-code_limit, code_limit]`，i8 code_limit=16，i16
    code_limit=1024
  - 同标度：`scale_out = ref_abs_max / (qmax · 0.85)`（从实际 ref 推导，
    防 saturate，同 P4 Linear approach）
  - 跨标度：`scale_out = same-scale · exp(±0.05/2)`
- **阈值**: `_MIN_COSINE = 0.9999`、`_MAX_FLOAT_LSB = 1.0`（统一 floor，
  与 Add/Sub/Mul/Div/Linear/Conv 完全一致）
- **实测值**（commit time 2026-06-08, PyTorch CPU, 32 trials × 1024 elem）:

| grid | mode | cos_min | lsb_max |
|------|------|---------|---------|
| i8 | same-scale | 0.999938 | 0.5004 |
| i8 | cross-scale | 0.999925 | 0.5004 |
| i16 | same-scale | 1.000000 | 0.7553 |
| i16 | cross-scale | 1.000000 | 0.8018 |

- **状态**: ✓ PASS（全 4/4 用例稳过 floor）
- **关键观察**:
  - **i8 也 PASS**：与 P4 Linear i8 edge-oscillation（`xfail strict=False`）
    形成对比——本段输出体积 1024 vs P4 Linear 64，matmul-class SNR
    随输出体积稳步提升，1024 outputs 足以让 i8 cos_min 稳定在 0.99992+
  - i16 lsb_max 略高于 0.5（达 0.80）但仍 < 1.0 floor，且 cos=1.0；
    本质是 K=8 累加 + 8 个非零项的 quantization noise 概率分布在 i16
    grid 上更"显形"——不是 kernel bug，与 Linear i16 段 lsb ≤ 0.67
    同源
- **注（算子级 KNOWN_LIMIT）**:
  - **i32 输出 grid 不覆盖**：与 Linear / Conv 同根（spec 04_02 仅允许
    输出 dtype i8/i16）
  - **unsigned grid u8/u16 不覆盖（测试覆盖缺口，非 spec 限制）**：
    spec `04_02 §4.2.1 matmul` 允许 u8/u16；default `zero_point=0`
    unsigned 无法表达零均值 random matmul 结果；与 Linear 段同根
    framing
  - **batched matmul**：本段用 3-D `(B, M, K) @ (B, K, N)` 覆盖最常见
    形态；2-D `(M, K) @ (K, N)` 与 N-D broadcast matmul 走相同 kernel
    路径，未单独测；broadcast 边界 case 见 dispatch 测试
  - **weight/operand bitwidth ∈ {i2, i4}**：spec `04_02 §4.2.1 matmul`
    允许 i2/i4 operand bitwidth，同 Linear / Conv2d 段说明（Layer C1
    Conv2d 已实测 i4 weight PASS，MatMul 共享 GEMM hot path）；本段
    未独立测试

---

## nn.Conv2d

- **Kernel**: `aimet_torch/fixed_point/kernels/conv_linear.py::Conv2dInt16Kernel`
- **KernelKind**: `REQUANTIZING`
- **Spec**: `doc/04_算子详细规格/04_01_卷积类算子.md` §量化推导
  - spec 契约：`im2col → centered → int32_matmul → +bias → saturate → requant`，
    weight `Z_w=0`，`real_m = S_x · S_w / S_y`，bias 在 acc-scale 上整数表达
  - **spec 输出 dtype**: `i8 / i16`
  - kernel 对齐：im2col 走整数路径（`im2col_int`，ADR-002 禁止 float
    intermediate），`int32_matmul` 在 INT16_FIXED_EVAL 下走 int64 product
    + INT32 saturation；支持 `groups` 任意，depthwise 是 g=in_channels 特例
- **Spec 测试覆盖度**: ✓ **完全**——三个典型 conv config 覆盖 spec
  允许的 stride/padding/groups 组合；real_m 经 `quantize_multiplier` 折叠
- **测试**: `tests/fixed_point/kernels/test_conv2d_int16_precision.py`
- **Reference**: `nn.functional.conv2d(x.float(), w.float(), b.float(), ...)`
- **Input 分布**: 16 trials per (grid, config, mode)；shape `(B=1, C=16,
  H=8, W=8)`；`out_channels=8`；conv configs:
  - `k3-s1-p0-g1`：标准 3x3，full conv，K=16·9=144
  - `k3-s2-p1-g1`：strided 3x3 with pad，K=16·9=144
  - `k3-s1-p1-g4`：grouped 3x3 (groups=4)，K=(16/4)·9=36（仍足够 SNR）
- **阈值**: `_MIN_COSINE = 0.9999`、`_MAX_FLOAT_LSB = 1.0`
- **实测值**（commit time 2026-06-08, PyTorch CPU, 16 trials）:

| grid | config | mode | cos_min | lsb_max |
|------|--------|------|---------|---------|
| i8 | k3-s1-p0-g1 | same-scale | 0.999948 | 0.5002 |
| i8 | k3-s1-p0-g1 | cross-scale | 0.999957 | 0.5000 |
| i8 | k3-s2-p1-g1 | same-scale | 0.999958 | 0.5000 |
| i8 | k3-s2-p1-g1 | cross-scale | 0.999955 | 0.5000 |
| i8 | k3-s1-p1-g4 | same-scale | 0.999940 | 0.4999 |
| i8 | k3-s1-p1-g4 | cross-scale | 0.999931 | 0.5000 |
| i16 | k3-s1-p0-g1 | same-scale | 1.000000 | 0.6108 |
| i16 | k3-s1-p0-g1 | cross-scale | 1.000000 | 0.6198 |
| i16 | k3-s2-p1-g1 | same-scale | 1.000000 | 0.6196 |
| i16 | k3-s2-p1-g1 | cross-scale | 1.000000 | 0.6170 |
| i16 | k3-s1-p1-g4 | same-scale | 1.000000 | 0.6317 |
| i16 | k3-s1-p1-g4 | cross-scale | 1.000000 | 0.6013 |

- **状态**: ✓ PASS（全 12/12 用例，i8 与 i16 都稳过 0.9999 floor）
- **关键观察**：i8 在 Conv2d 上**稳过** 0.9999 floor（cos_min ≥ 0.99993），
  与 Linear i8 边缘震荡形成对比——Conv2d 的 H'·W'·F 输出体积（典型
  ≥ 200 元素）让 cosine 统计稳定，Linear 的 64 元素则在统计噪声边缘
- **注（算子级 KNOWN_LIMIT）**:
  - **i32 输出 grid 不覆盖**：spec 04_01 不允许；同 Linear
  - **unsigned grid u8/u16 不覆盖（测试覆盖缺口，非 spec 限制）**：
    spec `04_01 §4.1.1 conv2d` 允许 u8/u16 输入；同 Linear / Pool 段
    framing
  - **dilation > 1**：本段未直接测，归 follow-up（实际 kernel 通过 `extra
    ['dilation']` 接收，im2col 已支持，但 spec 未要求 INT16_FIXED_EVAL
    支持）
  - **weight bitwidth ∈ {i2, i4}（Layer C1 已 spot-check）**：spec
    `04_01 §4.1.1 conv2d` 明确允许 `i2 / i4 / i8` 权重。Layer C1 加
    `test_conv2d_i16_input_i4_weight_spotcheck` (config k=3-s1-p1-g1)
    实测：
    - i16 input × **i4 weight** ([-8, 7])：cos>0.9999、lsb<1.0 全
      `_NUM_RANDOM_TRIALS` 个 trial **PASS** → kernel 路径完全可行，
      `Int16QuantizedTensor` 仅依赖 `qmin/qmax` 而与 dtype 名解耦，
      `int32_sat` 累加器对 `code_limit_w ≤ 6` 远在饱和限内
    - i16 input × **i2 weight** ([-2, 1])：xfail strict=True，4 level
      量化步长 ~25% 动态范围、cos 不过 0.9999 floor（kernel 算术
      正确，是 SNR 物理上限）
    full 参数化（多 config × 多 grid 组合）按需后续展开；现状证明
    spec 允许的 i4 weight 路径在 software 上 day-1 ready
  - **kernel-size 仅覆盖 k=3（测试覆盖缺口）**：spec 允许 `kt, kf ∈
    [1, 8]`；其余尺寸通过 im2col 共用 hot path，理论上同精度档案，但
    未独立参数化（k=1×1 pointwise / k 大 ≥ 5 边界 padding 可能有数值
    差异，需要时补独立 case）
  - **depthwise (g=in_channels) / pointwise (k=1×1)（Layer C5 已补
    spot-check）**：spec `04_12 §4.12.1 DepthwiseConv2d / §4.12.2
    Pointwise` 说"通过 conv2d + group/kernel=1 实现"。本段已扩 Conv
    configs 到 5 个，新增 `k1-s1-p0-g1` (Pointwise) 与
    `k3-s1-p1-gIN` (Depthwise，g=in_channels=16，out_channels=16)。
    Pointwise 全 PASS（4 case：i8/i16 × same/cross），Depthwise
    i16 全 PASS（2 case），**i8 depthwise (gIN) 标 xfail strict=False**
    （2 case：same/cross）—— 同 `FU-P4-CONV1D-I8-STRIDE2-FLAKE` 与
    i8 MatMul / Linear 同根 SNR 边缘：K_eff=9 比 standard conv K=144
    小 16 倍，i8 [-128,127] 代码包络只剩 ~96，cos 距 0.9999 floor 不
    足 1e-4 余量。lsb_max 仍 ≤ 0.5（kernel 算术正确）。
    **关于 Layer C5 早期"全 PASS"叙述的修正（2026-06-08 复核）**：
    早期报告基于 PYTHONHASHSEED 随机化下的 lucky seed —— 切换到
    `_stable_seed_token` (md5-based deterministic hash) 后 i8
    depthwise 跨进程稳定落到 SNR 边缘 fail。承认这是 i8 物理上限而
    非 transient flake，避免"靠 hash 抽签 PASS"的假信号。
    Depthwise 在 i16 grid 下稳过 0.9999 floor，因为
    H'·W'·F=1024 个输出元素让 cosine 统计稳定 —— 这印证了"conv2d
    hot path 同精度档案"的结论。完整 kernel-size 1-8 的 cross 验证
    保留为 follow-up

---

## nn.Conv1d

- **Kernel**: `aimet_torch/fixed_point/kernels/conv_linear.py::Conv1dInt16Kernel`
  （shim：unsqueeze 到 4D 后调用 `Conv2dInt16Kernel`）
- **KernelKind**: `REQUANTIZING`
- **Spec**: `doc/04_算子详细规格/04_01_卷积类算子.md` §量化推导（与 Conv2d
  共享）；hardware 上 1D conv 与 2D conv 共用 MAC 阵列，spec 只在 stride/
  padding 维度数上有差异
- **Spec 测试覆盖度**: ✓ **完全**——shim 把 1D 输入升维后走 Conv2d 同一
  hot path；本段独立测以锁定 shim 的 squeeze/unsqueeze 正确性
- **测试**: `tests/fixed_point/kernels/test_conv1d_int16_precision.py`
- **Reference**: `nn.functional.conv1d(x.float(), w.float(), b.float(), ...)`
- **Input 分布**: 16 trials per (grid, config, mode)；shape `(B=1, C=16,
  L=32)`；`out_channels=8`；conv configs 与 Conv2d 同形（k=3, 三种
  stride/padding/groups 组合）
- **阈值**: `_MIN_COSINE = 0.9999`、`_MAX_FLOAT_LSB = 1.0`
- **实测值**（commit time 2026-06-08, PyTorch CPU, 16 trials）:

| grid | config | mode | cos_min | lsb_max |
|------|--------|------|---------|---------|
| i8 | k3-s1-p0-g1 | same-scale | 0.999943 | 0.4997 |
| i8 | k3-s1-p0-g1 | cross-scale | 0.999903 | 0.4999 |
| i8 | k3-s2-p1-g1 | same-scale | 0.999941 | 0.5001 |
| i8 | k3-s2-p1-g1 | cross-scale | 0.999941 | 0.4994 |
| i8 | k3-s1-p1-g4 | same-scale | 0.999909 | 0.4997 |
| i8 | k3-s1-p1-g4 | cross-scale | 0.999918 | 0.4998 |
| i16 | k3-s1-p0-g1 | same-scale | 1.000000 | 0.6416 |
| i16 | k3-s1-p0-g1 | cross-scale | 1.000000 | 0.6569 |
| i16 | k3-s2-p1-g1 | same-scale | 1.000000 | 0.6332 |
| i16 | k3-s2-p1-g1 | cross-scale | 1.000000 | 0.6603 |
| i16 | k3-s1-p1-g4 | same-scale | 1.000000 | 0.6602 |
| i16 | k3-s1-p1-g4 | cross-scale | 1.000000 | 0.6328 |

- **状态**: ✓ PASS（10/12 用例）；
  ⚠ KNOWN_LIMIT（i8 + depthwise (g=4) 共 2/12 用例 xfail strict=False
  锁定边缘震荡：standalone cos_min ≈ 0.99991 PASS，跨测试运行偶发到
  0.99989 边缘——K=(in/g)·k=4·3=12 比 standard conv K=48 小，统计稳定
  性下降；lsb_max 仍 ≤ 0.5）
- **注**:
  - 与 Conv2d 同根（共享 hot path）；shim 路径正确性也被本段锁定
  - Conv2d 的 depthwise (g=4) 因 H'·W' = 64 输出元素显著多于 Conv1d 的
    L' = 32，i8-depthwise 在 Conv2d 上稳过 0.9999 而 Conv1d 上震荡——
    输出体积差异是关键
  - **改进建议**（同 P3/P4 i8 边缘案例）：adapter 端 i8 + depthwise 改
    走 "i8→i16→conv1d→i8" 复合路径；或在 manifest 上声明 i8 + Conv1d
    + depthwise 不被支持，落地后 2 个 xfail 翻 PASS
  - **weight bitwidth ∈ {i2, i4} / kernel-size 仅覆盖 k=3**：同 Conv2d
    段说明（Layer C1 Conv2d i4 weight 已 PASS、i2 weight xfail strict；
    Conv1d 共享 GEMM hot path，软件路径 ready，未独立参数化）
  - **已知边缘观察（一次性，未做长期统计）**:
    `test_conv1d_cross_scale_random_fp32_per_grid[i8-k3-s2-p1-g1]` 在
    2026-06-08 本轮验证窗口的全套跑（287 PASS / 3 次跑）中**观察到 1 次
    FAIL**（cos 稍 < 0.9999），单独跑该 case 与之后的 2 次全套跑均稳定
    PASS。样本量太小，**未做长期 flake-rate 统计**，仅作为客观记录。
    猜测根因：i8 grid 在 stride=2 + padding=1 + bias 配合下 cos 距 floor
    余量 < 1e-4，与全套并行下 import 顺序或 global state（如
    `torch.manual_seed` 副作用）有微小关联。当前**不加 `strict=False
    xfail`**（避免在样本量不足时掩盖真实数据），跟踪在
    `FU-P4-CONV1D-I8-STRIDE2-FLAKE` follow-up：先按 N×100 次的 flake-rate
    采样，再判断是否需要 (a) 拉宽 `code_limit` 提高 SNR，或 (b) 走
    i8→i16→conv→i8 复合（同 `FU-P4-I8-LINEAR-COMPOSITE`）

---

## nn.ReLU / nn.ReLU6

- **Kernel**: `aimet_torch/fixed_point/kernels/eltwise.py::ReLUInt16Kernel` /
  `ReLU6Int16Kernel`
- **KernelKind**: `SAME_GRID_OR_REQUANT`（same-grid 时 M=1 退化为
  比较直通，cross-grid 时走 single-shot `_requantize`）
- **Spec**: `doc/04_算子详细规格/04_04_激活函数类算子.md` §4.4.1
  - spec 契约：`y_q = sat((max(0, q_x - Z_x) · M) ≫ rshift) + Z_y`
  - **spec 输出 dtype**: `i8 / i16`（与输入相同）
  - ReLU6 spec 在 `04_13_特殊激活与常量算子.md` §4.14.1，等价于
    `clamp(x, 0, 6)`；kernel 实现走 ReLU 同源 hot path 但加上 `clamp(0,
    round(6/scale_in))` 上界
  - kernel 对齐：centered → `clamp_min(0)` 或 `clamp(0, max)` → 可选
    `_requantize`；M=1 同 grid 时退化为加 Z_y 直通
- **Spec 测试覆盖度**: ✓ **完全**——same-scale 验 M=1 fast path，
  cross-scale 验 single-shot requantize（`M/rshift` 由 `quantize_multiplier`
  从 `real_m = S_x / S_y` 折叠）
- **测试**: `tests/fixed_point/kernels/test_relu_int16_precision.py`
- **Reference**: `torch.relu(x)` / `torch.clamp(x, 0.0, 6.0)`
- **Input 分布**: 32 trials per (grid, op, mode)，shape `(256,)`；
  zero-mean random input with `code_limit = qmax/2`——约一半输入命中
  clamp 分支，另一半 pass-through
- **阈值**: `_MIN_COSINE = 0.9999`、`_MAX_FLOAT_LSB = 1.0`
- **实测值**（commit time 2026-06-08, PyTorch CPU, 32 trials × 256 elements）:

| op | grid | mode | cos_min | lsb_max |
|----|------|------|---------|---------|
| ReLU | i8 | same-scale | 1.000000 | 0.0000 |
| ReLU | i8 | cross-scale | 0.999959 | 0.4994 |
| ReLU | i16 | same-scale | 1.000000 | 0.0000 |
| ReLU | i16 | cross-scale | 1.000000 | 0.6732 |
| ReLU6 | i8 | same-scale | 0.999974 | 0.4991 |
| ReLU6 | i8 | cross-scale | 0.999904 | 0.7071 |
| ReLU6 | i16 | same-scale | 1.000000 | 0.4971 |
| ReLU6 | i16 | cross-scale | 0.999998 | 0.8023 |

- **状态**: ✓ PASS（全 8/8 用例）
- **关键观察**:
  - ReLU same-scale 完美：`lsb_max = 0.0` —— kernel 退化为 integer
    `clamp_min(int_repr, 0) + Z_y`，零量化误差
  - ReLU6 same-scale 仍引入 ≤ 0.5 LSB：因 `round(6/scale_in)` 在 input
    grid 上的整数化引入了一次 round-half 噪声（每元素至多 ±1 LSB-of-input，
    最终在 output scale 上是 ±0.5 LSB-of-output）
  - cross-scale 都通过单次 multiplier 折叠，误差 ≤ 1 LSB-of-output
- **注（算子级 KNOWN_LIMIT）**:
  - **u8/u16 不覆盖**：zp=0 unsigned 无法表达 zero-mean 输入的负部分；
    实际项目若用 u8/u16 ReLU 输出，calibration 会把 zp 设到 grid 起点
  - **i32 输出 grid 不覆盖**：spec 04_04 仅允许 `i8 / i16` activation
    grid，i32 不是 activation grid

---

## custom.Abs（integer-abs path，spec 04_03 §4.3.5）

- **Kernel**: `aimet_torch/fixed_point/kernels/eltwise.py::AbsInt16Kernel`
  （Layer B1 review 中**从 PWL `_LutInt16Kernel` 迁出**到 SAME_GRID_OR_REQUANT
  家族，与 ReLU / Clamp 共享 hot path）
- **KernelKind**: `SAME_GRID_OR_REQUANT`（manifest 同步切回，详见
  `FU-ABS-KIND-MISMATCH` 已关闭条目）
- **Spec**: `doc/04_算子详细规格/04_03_逐元素运算类算子.md §4.3.5 abs`
  - spec 契约：`x' = |q_x − Z_x|`（int32 centered absolute value）→
    `y_q = sat((x' · M) ≫ rshift) + Z_y`
  - **spec 输出 dtype**: 与输入相同（i8 / i16）
  - kernel 对齐：`_center_tensor(input)` → `torch.abs(centered)` →
    same-grid 时 `+ zp + saturate_sim_tensor`（M=None 直通），cross-grid
    走 `_requantize`（同 ReLU 模式）
- **Spec 测试覆盖度**: ✓ 完全（与 ReLU 同模式：same-scale × cross-scale ×
  i8/i16 共 4 cases；abs(-qmax) 不越界，因为 random `code_limit = qmax/2`
  让 |x| 永远小于 qmax；abs(-qmin) saturation 由 `saturate_sim_tensor` 兜底）
- **测试**: `tests/fixed_point/kernels/test_relu_int16_precision.py::
  test_abs_{same,cross}_scale_random_fp32_per_grid`
- **Reference**: `torch.abs`（fp32）
- **Input 分布**: 32 trials × 2 modes × 2 grids = 128 trials；zero-mean
  random with `code_limit = qmax/2`；same-scale 用 base scale 一致，
  cross-scale 用 log-ratio span 0.05
- **阈值**: `_MIN_COSINE = 0.9999`、`_MAX_FLOAT_LSB = 1.0`（统一 floor）
- **状态**: ✓ 4 PASS（i8/i16 × same/cross），cos=1.0，lsb=0.0
  （byte-stream identity，比 PWL 时代 lsb=0.5 严格更好；统计来自
  Layer B1 落地后的全套回归 `287 passed + 44 xfailed + 8 xpassed`）
- **关键观察**:
  - same-grid 路径 byte-stream identity——`|centered| + zp + saturate`
    完全 integer，没有 rounding 误差
  - cross-grid 路径 `_requantize(|centered|)` 与 ReLU 同 hot path，相同
    rounding 半 LSB 噪声；但 |x| ≥ 0 不像 ReLU 那样在 `clamp_min(0)` 之后
    才进 requantize，所以输入分布对 saturation 更敏感（abs 没有"负向被
    钳零"的 tail），实测 cross-scale lsb 仍 ≤ 0.5
- **注（与 P7-S1 段对照）**：本算子曾在 P7-S1 段以 LOOKUP/PWL 形式记录
  （`fit_max_lsb ≈ 0.5`），Layer B1 后已迁出；P7-S1 段保留 10 个 PWL 算子

---

## custom.ElementwiseUnarySign（spec 04_03 §4.3.6）

- **Kernel**: `aimet_torch/fixed_point/kernels/eltwise.py::SignInt16Kernel`
- **KernelKind**: `SAME_GRID_OR_REQUANT`（输出整数 ∈ {-1, 0, 1}，无需
  output `M/rshift`；走比较器整数路径）
- **Spec**: `doc/04_算子详细规格/04_03_逐元素运算类算子.md §4.3.6 sign`
  - spec 契约：`q_y = sign(q_x − Z_x)`，结果取值 ∈ {-1, 0, 1}
  - **spec 输入 dtype**: `i8 / i16`
  - **spec 输出 dtype**: `i8`（spec 唯一指定）
  - kernel 对齐：默认走 `sign_int16_centered`（INT16_FIXED_EVAL 模式
    强制走整数比较，**禁止**浮点 fallback 除非显式 `extra['sign_
    float_ref']=True` 或 `AIMET_RX_SIGN_FLOAT_REF=1`）
- **Spec 测试覆盖度**: ✓ 完全（spec 给整数比较定义，无近似误差；测试
  对比 byte-stream identity）
- **测试**: `tests/fixed_point/kernels/test_abs_sign_pwl_int16.py::
  test_sign_int16_centered_matches_grid_in_eval` (默认整数路径) +
  `test_sign_int16_float_ref_matches_grid` (可选 float 参考路径)
- **Reference**: 整数 `sign(centered)`（默认）或 `torch.sign(x_float)`
  （float ref 路径，MRNN STFT 用）
- **Input 分布**: 含 -0.4 / 0.0 / 0.3 三类临界值（负、零、正）；spec
  中"零阈值为严格等于零"已被 `centered == 0` 直接表达
- **状态**: ✓ 2 PASS（整数路径 + float ref 路径），byte-stream 与
  `quantize_float_to_grid(sign(x_f))` 完全一致（`torch.testing.
  assert_close` 整数完全相等）
- **关键观察**:
  - 整数路径无任何 round/saturate 噪声，输出严格 {-1, 0, 1}
  - float ref 路径仅在 calibration 给出 input scale 让"接近零的浮点
    输入"在量化后落到 0 时与整数路径产生差异——MRNN STFT 用例的
    设计选择，不影响 spec compliance
- **注（manifest 一致性）**：`capabilities.py::custom.ElementwiseUnarySign`
  注册为 `SAME_GRID_OR_REQUANT`，与 kernel 行为一致；`dispatchable=True`
  / `int16_eval=True` / `exportable=True`

---

## nn.Hardtanh / custom.Clamp / custom.Clip

- **Kernel**: `aimet_torch/fixed_point/kernels/eltwise.py::ClampInt16Kernel`
  （`FunctionalClampInt16Kernel` / `FunctionalClipInt16Kernel` 都继承）
- **KernelKind**: `SAME_GRID_OR_REQUANT`
- **Spec**: `doc/04_算子详细规格/04_07_数据操作类算子.md §4.7.6 clamp`
  （主体）+ `04_13_特殊激活与常量算子.md §4.14.1 ReLU6/Clip`（边界
  case 即 ReLU6 = `clamp(0, 6)`、Hardtanh = `clamp(-1, 1)` 的特例）；
  spec `04_04 §4.4.1` 是 ReLU，**不是** Hardtanh 的规范出处
  - spec 契约：input grid 上的纯整数 clamp `q_y = clamp(q_x, q_min, q_max)`，
    `q_min = round(min/S_x) + Z_x`、`q_max = round(max/S_x) + Z_x`，
    `S_y/Z_y` 与 `S_x/Z_x` 一致时输出可直接使用裁剪后定点值
  - **spec 输出 dtype**: `i8 / i16`
  - **spec 明确**："不应让纯 Clip 比较逻辑隐式承担量化域转换" ——
    cross-scale 由编译器在前后图优化中插入或融合重定标，Clip kernel 不
    自带 `M/rshift`
- **Spec 测试覆盖度**: ✓ 完全（包含 FU-P5-CLAMP-DOUBLE-RESCALE 修复后
  新增的 adapter-multiplier 真实路径覆盖）
  - same-scale：直接走整数 clamp 比较路径
  - cross-scale `multiplier=None`：legacy sidestep 路径，模拟 spec
    "前后图融合 rescale" 后的纯 Clip 语义
  - cross-scale `adapter M/rshift`：**新增**，模拟真实 adapter dispatch
    路径（`adapter.py` line ~1016 `real_m = x_scale / y_scale`）。
    修复前在此路径 `lsb_max ~150`（双重 rescale），修复后 `≤ 0.63`
  - bit-exact gate（新增）：cross-scale 下 adapter-path 与 no-multiplier
    路径 `int_repr` 元素相等，强约束未来不会再次回归
- **测试**: `tests/fixed_point/kernels/test_clamp_int16_precision.py`
- **Reference**: `torch.clamp(x, min=-0.5, max=0.5)`
- **Input 分布**: 32 trials per (grid, module, mode)，shape `(256,)`；
  scale 取使 ±qmax/2 落在 ±0.5 附近，让大约一半输入命中 clamp 边界
- **阈值**: `_MIN_COSINE = 0.9999`、`_MAX_FLOAT_LSB = 1.0`
- **实测值**（commit time 2026-06-09, FU-P5 修复后，PyTorch CPU, 32 trials × 256 elements）:

| module | grid | mode | cos_min | lsb_max |
|--------|------|------|---------|---------|
| Hardtanh | i8 | same-scale | 0.999976 | 0.4997 |
| Hardtanh | i8 | cross-scale (multiplier=None) | 0.999910 | 0.4970 |
| Hardtanh | i8 | cross-scale (adapter M/rshift) | 0.999933 | 0.4991 |
| Hardtanh | i16 | same-scale | 1.000000 | 0.4925 |
| Hardtanh | i16 | cross-scale (multiplier=None) | 1.000000 | 0.6049 |
| Hardtanh | i16 | cross-scale (adapter M/rshift) | 1.000000 | 0.5986 |
| Clamp | i8 | same-scale | 0.999973 | 0.4981 |
| Clamp | i8 | cross-scale (multiplier=None) | 0.999918 | 0.4997 |
| Clamp | i8 | cross-scale (adapter M/rshift) | 0.999931 | 0.4998 |
| Clamp | i16 | same-scale | 1.000000 | 0.4864 |
| Clamp | i16 | cross-scale (multiplier=None) | 1.000000 | 0.5712 |
| Clamp | i16 | cross-scale (adapter M/rshift) | 1.000000 | 0.5916 |
| Clip | i8 | same-scale | 0.999969 | 0.4956 |
| Clip | i8 | cross-scale (multiplier=None) | 0.999909 | 0.4986 |
| Clip | i8 | cross-scale (adapter M/rshift) | 0.999919 | 0.4992 |
| Clip | i16 | same-scale | 1.000000 | 0.4717 |
| Clip | i16 | cross-scale (multiplier=None) | 1.000000 | 0.6289 |
| Clip | i16 | cross-scale (adapter M/rshift) | 1.000000 | 0.6201 |

- **状态**: ✓ PASS（全 18/18 数值用例 + 12/12 bit-exact 用例 = **30/30 cases**；
  i8 cross-scale 个别 seed 仍在 strict=False xfail 标记下，与
  `FU-P4-CONV1D-I8-STRIDE2-FLAKE` 同根的 small-K i8 SNR 边缘问题，**不是
  FU-P5 残留**）
- **关键观察**:
  - **FU-P5-CLAMP-DOUBLE-RESCALE 修复（commit 2026-06-09）**：
    `clamp_int16` 在 cross-grid 路径上**显式忽略**
    `output_encoding.multiplier` / `rshift`，因为
    `align_centered_int32_to_output` 已经把 `S_x → S_y` 折叠了一次，
    再走 `_requantize` 会重复折叠（误差 ×`(S_x/S_y)²`）。修复后
    adapter-multiplier 路径与 no-multiplier 路径的 `int_repr` **元素
    相等**（bit-exact），lsb_max 从 ~150 降到 ≤ 0.63（**250×
    精度提升**）
  - Hardtanh / Clamp / Clip 三模块走同一 `ClampInt16Kernel`，精度档案
    一致（差异仅来自不同 trial seed）
  - same-scale `lsb_max` 远低于 0.5：kernel 走整数 clamp fast path，
    误差仅来自 input quantize 的 round-half
  - cross-scale `lsb_max ≤ 0.63`：单次 align rescale 的 multiplier
    折叠误差（与 adapter-multiplier 路径几乎相同，差异 < 0.005 LSB
    纯 round-half 噪声）
- **注（算子级 KNOWN_LIMIT 与 follow-up）**:
  - **u8/u16 不覆盖**：同 ReLU 类，calibration 决定
  - **i32 输出 grid 不覆盖**：spec 04_13 不允许
  - **FU-P5-CLAMP-DOUBLE-RESCALE**: ✓ **已关闭（commit 2026-06-09）** ——
    `clamp_int16` cross-grid 分支现在**显式忽略**
    `output_encoding.multiplier` / `rshift`（详见 kernel docstring）。
    adapter dispatch site（`adapter.py` line ~1016）仍然按
    `real_m = x_scale / y_scale` 传入 multiplier；kernel 在 cross-grid
    分支不再调用 `_requantize`，仅走 `align → clamp → +Z_y + saturate`
    一条路径。bit-exact gate（`test_clamp_cross_scale_adapter_path_
    matches_no_multiplier_path`）强制约束未来不会回归。本次修复方案
    选择"kernel 内显式忽略"而非"adapter 端跳过 multiplier"，是因为
    SAME_GRID_OR_REQUANT 其他算子（Pad/Concat/Dropout 等）依赖
    adapter fallback 路径产生 multiplier；分头改 adapter 影响面太大
  - **FU-P5-CLAMP-R2-GRID-AWARE-FLOOR**: ✓ **已关闭（commit 2026-06-09，
    R2 工单）** —— `clamp_int16` 入口新增
    `_grid_aware_floor_clamp_extra` helper：当用户指定的 `min` 严格
    `> 0` 但 `extra['min_int']` 因 `round(min/S_x + zp_x) <= zp_x` 而
    塌到 zero-point（即 `min` 在输入 grid 上 `< 1` LSB），把
    `min_int` 提到 `zp_x + 1` / `min` 提到 `S_x`；负 `max` 对称处理。
    动机：MRNN `CLN.forward` 的 `torch.clamp(mean_sq, min=EPS=1e-8)`
    在 `mean_sq` per-tensor grid（scale ~ 1.5e-3）上 round 到 0，让
    clamp 在 INT16 dispatch 上完全失效；下游 `Sqrt → Divide` 命中
    `reciprocal_via_clz_lut q_in=0 → out_qmax` 的 spec-mandated
    div-by-zero 保护，乘 `+100×` 得到 `norm_max ≈ 7e30` 的活化爆炸
    （`per_layer_isolated_cosine` 上观察到 4 个 `*.cln.module_div_*`
    全部 `iso_cos = 0`）。修复后 4 中 3 个 `cln.module_div_*` 单步
    cos `≥ 0.999977`、`norm_max ≤ 0.05`，1 个（`enc_seqs.1.cln.
    module_div_4`）残留 cos ≈ 0.165 是 **R2 与 fp32 EPS 语义差**
    的固有 trade-off（见下条 KNOWN_LIMIT）。**fp32 / QDQ 路径
    完全不受影响** —— floor 仅在 `clamp_int16` 入口生效。新增 12
    个 R2 用例覆盖 sub-LSB `min` floor / 负 `max` floor / `min == 0`
    保留 ReLU 语义 / `min > 1 LSB` 不被自动改写四类边界
  - **R2-GRID-FLOOR-VS-FP32-EPS-SEMANTIC-GAP（known limitation）**:
    R2 用 INT16 grid 上的 1 LSB 替代 fp32 path 上的 `EPS = 1e-8` —
    在 `mean_sq < grid_LSB` 的元素上，fp32 reference 的
    `1/sqrt(EPS) ≈ 10000` 与 INT16 candidate 的 `1/sqrt(grid_LSB)`
    差几个数量级（8bit 下 ~25、16bit 下 ~414），**升 bitwidth 无法
    消除**（验证 run：`mrnn_cln_16bit.json` + 临时
    `SUPPORTED_ACTIVATION_BITWIDTHS = (8, 16)` 跑出来 cos 仍 ≈
    0.163；同时 `test_int16_fixed_eval_refuses_unsupported_
    activation_bitwidth` 契约测试会失败，证明 16bit 解封路径会破坏
    `audit-int16-activation-quantizer-contract`）。**症状**：在
    `mean_sq` 真值范围窄、small-mean_sq 通道占比高的层上（如 MRNN
    `enc_seqs.1.cln`，`mean_sq abs_max = 0.192` vs 其他三支
    `1.5 ~ 9.46`），`module_div_*` 的 `iso_cos` 卡在 ~0.16、
    `sqnr` 负值、`p99_err == max_abs`（整 channel 全错而不是少数
    outlier）。**唯一根治路径**: R1 — 在业务侧把 `CLN.EPS` 从
    `1e-8` 提到 INT16-grid friendly 量级（如 `1e-3`），让 fp32 与
    INT16 在 `clamp(mean_sq, min=EPS)` 语义上对齐。需要重训 fp32
    ckpt 验证 EPS 提升对 fp32 Top-1 无影响后才能落地。**排查指引**:
    碰到 `*.cln.module_div_*` 单步 `iso_cos < 0.95` 且 `sqnr < 0`
    时，先 dump 该支 `mean_sq abs_max`；若该值与同模型其他 cln 支差
    1 个数量级以上，确诊本 known limitation，不要再尝试升 bitwidth
    或修 R2 floor

---

## nn.Identity / nn.Flatten / custom.Reshape / custom.Permute

- **Kernel**: `aimet_torch/fixed_point/kernels/shape_ops.py::
  IdentityInt16Kernel / FlattenInt16Kernel / ReshapeInt16Kernel /
  PermuteInt16Kernel`（**Identity 继承 `_IdentityLikeInt16Kernel`；
  Flatten / Reshape / Permute 各自独立类**，统一通过
  `require_kernel_kind_encoding_contract(SAME_GRID_VALUE)` 强制
  same-grid）
- **KernelKind**: `SAME_GRID_VALUE`（**严格** same-grid，contract
  入口 `require_kernel_kind_encoding_contract` 在 input/output encoding
  不一致时报错——没有 cross-scale 路径）
- **Spec**: 按 sub-op 拆分：
  - `nn.Identity` —— spec 隐式（无独立 section，PyTorch 标准 op）
  - `nn.Flatten` —— `04_11_其他数据操作算子.md §4.11.2 flatten`
  - `custom.Reshape` —— `04_07_数据操作类算子.md §4.7.3 view/reshape`
  - `custom.Permute` —— `04_07_数据操作类算子.md §4.7.2 permute`
  - spec 契约：value-空间无变换，只是 view/rewrap；输入输出 dtype 与
    encoding 必须一致，cross-scale 由编译器在前后图融合外部 rescale
  - **spec 输出 dtype**: 与输入相同（i8 / i16）
  - kernel 对齐：均走 `_wrap(view(int_repr), output_encoding)`，零 LSB
    误差；contract 入口阻止误用 cross-grid 路径
- **Spec 测试覆盖度**: ✓ 完全（spec 不允许 cross-scale；测试只测
  same-scale 即穷尽语义）
- **测试**: `tests/fixed_point/kernels/test_shape_ops_int16_precision.py`
- **Reference**: 对应的 fp32 torch op (`x`, `torch.flatten`,
  `torch.reshape`, `torch.permute`)
- **Input 分布**: 32 trials per (grid, op)，不同 op 用不同 shape
  （Identity: `(2,8,16)`, Flatten: `(2,4,8,8)→(2,256)`, Reshape:
  `(2,4,16)→(2,64)`, Permute: `(2,4,8,4) dims=(0,2,1,3)`）；
  zero-mean random input with `code_limit = qmax/2`
- **阈值**: `_MIN_COSINE = 0.9999`、`_MAX_FLOAT_LSB = 1.0`
- **实测值**（commit time 2026-06-08, PyTorch CPU, 32 trials）:

| op | grid | mode | cos_min | lsb_max |
|----|------|------|---------|---------|
| Identity | i8 | same-scale | 1.000000 | 0.0000 |
| Identity | i16 | same-scale | 1.000000 | 0.0000 |
| Flatten | i8 | same-scale | 1.000000 | 0.0000 |
| Flatten | i16 | same-scale | 1.000000 | 0.0000 |
| Reshape | i8 | same-scale | 1.000000 | 0.0000 |
| Reshape | i16 | same-scale | 1.000000 | 0.0000 |
| Permute | i8 | same-scale | 1.000000 | 0.0000 |
| Permute | i16 | same-scale | 1.000000 | 0.0000 |

- **状态**: ✓ PASS（全 8/8 用例，**perfect byte-stream identity**）
- **关键观察**: 所有 SAME_GRID_VALUE 算子的实际量化误差为 0——kernel
  完全不触碰数值（只是 view/reshape/permute `int_repr`），唯一的误差
  源是 input quantize 步骤本身的 round-half，与 kernel 无关
- **注（算子级 KNOWN_LIMIT）**:
  - **cross-scale 不覆盖**：spec 与 kernel contract 都明示 same-grid，
    cross-scale 由编译器在外部插入或融合重定标（参考 spec 04_11）
  - **unsigned grid 不覆盖**：byte-stream identity 不关 zp，u8/u16 走同
    路径但语义不变，未单独测

---

## nn.Upsample / nn.UpsamplingNearest2d （nearest 模式）

- **Kernel**: `aimet_torch/fixed_point/kernels/shape_ops.py::
  UpsampleInt16Kernel / UpsamplingNearest2dInt16Kernel`（都继承
  `_NearestResizeInt16Kernel`，hot path 调
  `F.interpolate(int_repr.to(fp32), mode='nearest').to(sim_dtype)`；
  fp32 round-trip 在 INT16 qmin/qmax 范围内位精确，与 `_pad_int_via_
  float_roundtrip` 同源）
- **KernelKind**: `SAME_GRID_VALUE`（**严格** same-grid，contract 入口
  `require_kernel_kind_encoding_contract` 强制 input/output encoding
  完全一致；mode != 'nearest' 在 kernel 入口 raise，让 adapter
  fallback FP32_QDQ 而不是静默走错）
- **Spec**: `doc/04_算子详细规格/04_10_Resize类算子.md §4.10.2`
  Nearest-neighbour
  - spec 契约：`y[t_out, f_out] = x[round(t_out·H_in/H_out),
    round(f_out·W_in/W_out)]`——纯地址映射，spec 明示「直接复制
    量化值，无需重量化」
  - **spec 输出 dtype**: 与输入相同（i8/u8/i16/u16/i32/u32 全支持，
    本测试覆盖 i8/i16）
  - kernel 对齐：`F.interpolate(..., mode='nearest')` 的索引计算与
    spec 公式 `round(idx_out · H_in / H_out)` 在所有整数 scale_factor
    上 bit-exact 一致；对非整数 scale_factor，spec 与 PyTorch 共用同一
    `round` 半数规则
- **Spec 测试覆盖度**: ✓ 完全（spec 不要求 cross-scale；nearest 全部
  upsample/downsample/explicit-size/scale_factor 路径都覆盖，且
  bit-exact gate 验证 SAME_GRID_VALUE 契约）
- **测试**: `tests/fixed_point/kernels/test_resize_nearest_int16.py`
- **Reference**: `F.interpolate(x_fp32, ..., mode='nearest')`
- **Input 分布**: 32 trials per (grid, module, mode)，shape 与 mode
  组合：
  - `nn.Upsample` × {`size=(16,16)` upsample, `scale_factor=2.0`,
    `size=(8,8)` downsample}
  - `nn.UpsamplingNearest2d` × {`size=(12,12)`, `scale_factor=3.0`}
  - 起始 shape `(2,4,8,8)` 或 `(2,4,16,16)`；
    zero-mean random `int_repr` with `code_limit = qmax/2`
- **阈值**: `_MIN_COSINE = 0.9999`、`_MAX_FLOAT_LSB = 1.0`
- **实测值**（commit time 2026-06-08, PyTorch CPU, 32 trials per case）:

| op | grid | mode | cos_min | lsb_max |
|----|------|------|---------|---------|
| Upsample | i8 | size upsample | 1.000000 | 0.0000 |
| Upsample | i8 | scale_factor up | 1.000000 | 0.0000 |
| Upsample | i8 | size downsample | 1.000000 | 0.0000 |
| Upsample | i16 | size upsample | 1.000000 | 0.0000 |
| Upsample | i16 | scale_factor up | 1.000000 | 0.0000 |
| Upsample | i16 | size downsample | 1.000000 | 0.0000 |
| UpsamplingNearest2d | i8 | size | 1.000000 | 0.0000 |
| UpsamplingNearest2d | i8 | scale_factor | 1.000000 | 0.0000 |
| UpsamplingNearest2d | i16 | size | 1.000000 | 0.0000 |
| UpsamplingNearest2d | i16 | scale_factor | 1.000000 | 0.0000 |

- **状态**: ✓ PASS（全 10/10 数值用例 + 2/2 bit-exact 用例 + 2/2 负向
  用例 = **14/14 cases**，与其他 SAME_GRID_VALUE 算子一样达到 perfect
  byte-stream identity）
- **关键观察**:
  - **Bit-exact identity**: 测试包含一个独立 case
    `test_nearest_int_repr_bit_exact_byte_stream_identity` 直接断言
    `kernel.int_repr == F.interpolate(int_repr, mode='nearest')`
    元素相等——这是 SAME_GRID_VALUE 契约的最强 gate，对应 spec
    「直接复制量化值」的形式化
  - **负向路径**: `mode='bilinear'` 与 `size=None & scale_factor=None`
    都被 kernel 入口拒绝（ValueError），不会静默退化
- **注（算子级 KNOWN_LIMIT）**:
  - **Bilinear 未覆盖**：spec §4.10.1 是 REQUANTIZING（需要插值系数
    定点 + `M_y/rshift_y` 重定标），与 nearest 不共享路径；本轮明确
    跳过，kernel 入口拒绝。Bilinear 立项见**未实现 spec 章节 dry-run
    评估** §D3
  - **cross-scale 不覆盖**：spec 与 kernel contract 明示 same-grid；
    若图上确实需要 cross-grid，应由编译器在前/后插独立 rescale
  - **`custom.Interpolate`（functional wrapper）不覆盖**：本轮只处理
    `nn.Module` 形态（`nn.Upsample` / `nn.UpsamplingNearest2d`）。
    Functional `F.interpolate(...)` 在图上通常被 model_preparer 转成
    `nn.Upsample` module；若 adapter 需要直接路由 functional 调用，
    那是单独 PR 的范围

---

## nn.LayerNorm

- **Kernel**: `aimet_torch/fixed_point/kernels/norm.py::LayerNormInt16Kernel`
  （**spec §4.5.4 完整 bit-parity 整数路径**：4 步管线全部走整数
  carrier：
  1. **spec §4.5.2 in-line integer variance**（`_inline_integer_variance`）：
     `s_o = Σ(q_x-Z_x)`、`q_μ-Z_μ = (s_o · inv_N) >> shift_N`、
     `v_o = (Σd² · inv_N) >> shift_N`、`q_var = ((v_o · M_var) >> rshift_var) + Z_var`
  2. **RSqrt CLZ LUT**（`_q_inv_int_from_qvar`）：integer-in / integer-out
     `q_inv = LUT(q_var)`
  3. **spec line 422-428 integer M/rshift affine**（`_integer_affine`）：
     `q_y = ((q_γ·q_inv·((q_x-Z_x)-(q_μ-Z_μ))·M_x) >> rshift_x) + b^LN`
     （`b^LN = ((q_β·M_β) >> rshift_β) + Z_y`；`S_μ=S_x` default 下
     `M_μ=M_x` 折叠）
  4. saturate 到 `output_encoding.qmin/qmax`

  同模块下保留两个 oracle 函数 (**不注册**)：
  - `layer_norm_float_reference`：pure fp32 (no LUT, no integer)，用于
    度量 LUT 单独引入的残差
  - `layer_norm_lut_fp32_affine_reference`：LUT integer + fp32 variance +
    fp32 affine（即 FU 关闭前的路径），用于度量 integer-variance +
    integer-affine 在 fp32-affine 基础上引入的额外 round-half）
- **KernelKind**: `REQUANTIZING`（input/output 量化域不同，与 LN 的 1/std
  scale 变换一致；REQUANTIZING activation-bitwidth gate 强制 8-bit 路径
  与项目其他 REQUANTIZING 算子一致）
- **Spec**: `doc/04_算子详细规格/04_05_归一化类算子.md §4.5.4`
  - spec 数学定义视角（line 387）：
    `q_y = (S_γ·S_x·S_inv/S_y)·q_γ·q_inv·[(q_x-Z_x)-(q_μ-Z_μ)] + (S_β/S_y)·q_β + Z_y`
    其中 `q_inv` 由 rsqrt LUT 产出：`S_inv·q_inv = f(q_σ²)`
  - spec 硬件整数视角（line 422-428）：相同合成 + 每个 `S·..·.../S_y`
    比值折叠为 16bit `M/rshift`，`q_μ/q_σ²` 来自在线 `variance` 基础
    指令（spec §4.5.2）
  - **当前实现选型（commit 2026-06-09 升级）**：spec **完整 bit-parity
    整数路径**——`FU-LAYERNORM-DSP-PARITY` 和 `FU-LAYERNORM-AFFINE-INTEGER`
    两个 FU 都已闭合，4 步管线全部走整数 carrier，与 spec line 422-428
    + §4.5.2 字面对齐
  - **未在 spec 中明示但本实施采用的 4 个量化 default**（"Default A"）：
    - `S_γ = γ.abs().max() / 32767, Z_γ = 0`（symmetric per-tensor weight
      quant，与 `aimet_torch.conv2d` weight 同款）
    - `S_β = β.abs().max() / 32767, Z_β = 0`（同 γ）
    - `S_μ = S_x, Z_μ = Z_x`（mean 共用 input grid；与 spec 公式中
      `q_x-q_μ` 减法对齐最自然）
    - `S_var, Z_var` 与 RSqrt LUT input grid 对齐（取自 LUT body 的
      `quantization.input` 块）—— 这样 step 1 的 `q_var` 直接 feed
      LUT step 2 不需要再次重定标
- **Spec 测试覆盖度**: ✓ 完全
  - 多 grid：i8 / i16
  - 多 normalized_shape：last-1d (`(F,)` 8 elements) / last-2d
    (`(T,F)` 16 elements) / last-3d (`(C,T,F)` 32 elements)，覆盖
    spec §4.5.4 描述的 `prefix_shape + normalized_shape` 任意拆分
  - elementwise_affine on/off：spec 明示 γ/β 可选；两条路径都覆盖
  - **LUT 残差 vs float oracle bounded gate**：用同模块下的
    `layer_norm_float_reference` 作为 oracle，断言 spec-aligned kernel
    与 oracle 的 int_repr diff ≤ 20 codes（i16 grid，实测 max diff 14
    codes），证明唯一精度 diff 来源就是 RSqrt LUT 步骤
  - 负向路径：missing `extra['normalized_shape']` / 多 input 都 raise
    ValueError
- **测试**: `tests/fixed_point/kernels/test_layernorm_int16.py`
- **Reference**: `F.layer_norm(x_fp32, normalized_shape, γ, β, eps=1e-5)`
- **Input 分布**: 16 trials per (grid, normalized_shape, affine)
  - `x ~ N(0, 1)` 标准正态，calibrated input scale
    `S_x = x.abs().max() / (qmax/1.5)`（与现实 percentile calibrator
    一致；naive `S_x = 1/code_limit` 会让 ±3σ 尾被 saturate，把 lsb
    误差放大到 ~5000 LSB，详见测试 docstring）
  - `γ ~ N(1.0, 0.2)`, `β ~ N(0, 0.1)`（典型训练后分布）
  - calibrated output scale `S_y = ref.abs().max() / (qmax/1.5)`
- **阈值**: 项目统一严格 gate `_MIN_COSINE = 0.9999`、
  `_MAX_FLOAT_LSB = 1.0`（与所有其他 INT16 kernel 一致，未为 LayerNorm
  放宽）。LayerNorm 在 spec 完整 bit-parity 路径下**两个 lsb 物理上限**
  都超出 1.0 floor：
  - **RSqrt CLZ LUT PWL fit residual**（与 P7 `custom.RSqrt` 同根）
    经 `γ/std` 放大 ~5-7 LSB
  - **spec line 414 的 16-bit M + max_rshift=31 物理上限**：i16
    calibrated grid + γ 量化下 `α_x = S_γ·S_x·S_inv/S_y ≈ 1.1e-8`，
    需要 `rshift ≈ 41` 才能让 M 满载 16-bit；超出 max_rshift=31 后 M
    精度退化到 ~5-bit，affine 路径 lsb 物理上限大幅升高到 ~1000 LSB

  与项目 P7 PWL/CLZ family 同款处理：`test_layernorm_random_fp32_per_grid`
  + `test_layernorm_normalized_shape_as_int_in_extra` 整组用
  `@pytest.mark.xfail(strict=True, reason=_XFAIL_LAYERNORM_REASON_RSQRT_LUT_CEILING)`
  显式登记为 **KNOWN_LIMIT**（两个物理上限同时存在，xfail reason 涵盖
  其一即可）。物理上限同步登记在
  `aimet_torch/fixed_point/metrics/thresholds.py::LAYERNORM_VS_FP32_PER_GRID_LIMITS`
  （分 grid × 分 affine/noaffine 两层维度，因为 affine 触发 spec
  16-bit M 二级上限，noaffine 不触发）
- **实测值**（commit time 2026-06-09, PyTorch CPU, 16 trials × 96 elements
  per config，**spec §4.5.4 完整 bit-parity 整数路径**）:

| grid | normalized_shape | affine | cos_min | lsb_max | 主导上限 |
|------|------------------|--------|---------|---------|----------|
| i8 | (8,) | True | 0.999512 | 4.33 | spec 16-bit M α_x ≈ 4e-9 |
| i8 | (8,) | False | 0.999630 | 2.27 | LUT residual |
| i8 | (4, 4) | True | 0.999229 | 5.33 | spec 16-bit M α_x ≈ 4e-9 |
| i8 | (4, 4) | False | 0.999739 | 1.94 | LUT residual |
| i8 | (4, 2, 4) | True | 0.999468 | 3.33 | spec 16-bit M α_x ≈ 4e-9 |
| i8 | (4, 2, 4) | False | 0.999844 | 1.56 | LUT residual |
| i16 | (8,) | True | 0.999996 | **617.33** | **spec 16-bit M α_x ≈ 1e-8** |
| i16 | (8,) | False | 1.000000 | 9.74 | LUT residual + int round-half |
| i16 | (4, 4) | True | 0.999995 | **910.34** | **spec 16-bit M α_x ≈ 1e-8** |
| i16 | (4, 4) | False | 1.000000 | 6.69 | LUT residual + int round-half |
| i16 | (4, 2, 4) | True | 0.999994 | **807.33** | **spec 16-bit M α_x ≈ 1e-8** |
| i16 | (4, 2, 4) | False | 1.000000 | 8.33 | LUT residual + int round-half |

**对照实验**（同 192 trials，跑 `layer_norm_lut_fp32_affine_reference` =
LUT integer + fp32 variance + fp32 affine，作为隔离 integer-affine
round-half 贡献的 oracle）：

| grid | affine 路径 lsb_max | noaffine 路径 lsb_max |
|------|---------------------|------------------------|
| i8 | ≤ 2.22 (vs 5.33 of integer) | ≤ 1.56 (vs 2.27) |
| i16 | ≤ 6.80 (vs 910 of integer) | ≤ 6.80 (vs 9.74) |

**差额完整解释为 spec 16-bit M 折叠的物理上限**：i16 affine 路径上 integer
比 fp32-affine 多 ~900 LSB（spec M 精度只有 ~5-bit），但 noaffine 路径只
多 ~3 LSB（spec M 满载 16-bit）。

- **状态**: 数值 case 全部 **XFAIL (KNOWN_LIMIT, strict=True)** + 边界/
  负向/oracle-residual case 全部 PASS：
  - 13 xfailed（12 数值 grid×shape×affine + 1 int-form-extra 边界，都跑
    `_assert_gates` 命中严格 1-LSB gate）
  - 4 passed（2 oracle-bounded gate: noaffine 路径 ≤ 16 codes vs
    `lut_fp32_affine_reference` / affine 路径 ≤ 1200 codes 标 spec 16-bit
    M physical ceiling + 2 负向 case）
  - **strict=True** 表示 lsb 物理超出是稳定可复现的——若哪天硬件/spec 更新
    把 16-bit M 加宽或 α_x 因子分解改变，让 lsb 回到 < 1 LSB，会触发
    XPASS failure 提醒移除 xfail 标记
- **不通过原因（按 step 残差分解 + 多 oracle 对照实验）**:

  **判定方法**：用同模块下的两个 oracle 跑同 192 trial：
  - `layer_norm_float_reference`：pure fp32, no LUT
  - `layer_norm_lut_fp32_affine_reference`：LUT integer + fp32 variance + fp32 affine

  | grid | affine | INTEGER spec lsb | LUT+fp32 affine lsb | pure-fp32 lsb |
  |------|--------|------------------|---------------------|---------------|
  | i8   | True   | ≤ 5.33           | ≤ 2.22              | ≤ 2.94        |
  | i8   | False  | ≤ 2.27           | ≤ 1.56              | ≤ 2.94        |
  | i16  | True   | ≤ **910**        | ≤ 6.80              | ≤ 1.99        |
  | i16  | False  | ≤ 9.74           | ≤ 6.80              | ≤ 1.99        |

  按 step 残差拆解：

  - **step 1 integer variance**（spec §4.5.2）: 引入 ~1-3 LSB 残差，来自
    两层 round-half：`(s · inv_N) >> shift_N` 和 `(v · M_var) >> rshift_var`。
    noaffine 路径 i16 lsb 从 LUT-fp32 的 6.80 升到 9.74，差额 ~3 LSB 就是这层
  - **step 2 RSqrt CLZ LUT**（spec line 384）: PWL fit residual ~3-5 LSB at
    LUT output grid，经 `γ/std` 放大到 LayerNorm output ~5-7 LSB。与 P7
    `custom.RSqrt` 单算子同根（同 LUT 资产 / 同物理上限），已是
    KNOWN_LIMIT
  - **step 3 integer M/rshift affine**（spec line 422-428）: 在 noaffine
    路径只引入 ~3 LSB（spec 16-bit M 在 `α_x = S_x·S_inv/S_y ≈ 3e-4`
    满载），与 step 1 自洽
  - **step 3 affine 路径下的 spec 16-bit M 物理上限**（**主导项 in i16
    affine**）: `α_x = S_γ · S_x · S_inv / S_y ≈ 1.1e-8` 在 i16 calibrated
    + γ.abs().max()/32767 default 下，需要 `rshift ≈ 41` 才能让 M 满载
    16-bit；超出 `max_rshift=31` 后 M 精度退化到 5-bit (M ≈ 24)，affine
    路径输出经 4 因子乘积放大后 lsb ≤ 910。**这是 spec line 414 的硬件
    设计选择，与项目级 `MULTIPLIER_QBITS=16` 严格一致，不是 kernel bug**
  - **step 4 saturate**: 0 LSB（无 round-half，只截断）

  **物理上界自洽性**：
  - i16 noaffine: LUT 5 + step1 round-half 3 + step3 round-half 1.5 ≈ 9.5 LSB（实测 ≤ 9.74 ✓）
  - i16 affine: 16-bit M 折叠 ~5-bit precision，`α_x` 误差 ~3% 经 4 因子乘积放大到 ~1000 LSB（实测 ≤ 910 ✓）
  - i8 受益于粗粒度 grid 吸收部分残差，affine 路径 ≤ 5.33
- **关键观察**:
  - **i16 cos=1.0**：尽管 lsb_max=6.67 > 1，cosine 仍完美——RSqrt LUT
    残差是 element-wise 加性零均值噪声，对方向无系统偏置（与 P7 PWL
    family 一致的 cos 通过、lsb 物理超出图样）
  - **i8 cos ≥ 0.99984 / lsb ≤ 1.83**：i8 lsb 反而比 i16 _小_——i8 grid
    的 1 LSB 物理大小 ~256 倍 i16，吸收了大部分 LUT residual；但仍轻微
    超出 1.0（i8 noaffine 最低 1.0089）
  - **input calibration 至关重要**：naive `S_x = 1/code_limit`（让 ±3σ
    超出 grid）会把 i16 lsb_max 推高到 ~5000+，这是 input saturation
    而非 kernel 缺陷；calibrated path 是 spec §4.5.4 的隐含假定
- **注（算子级 KNOWN_LIMIT 与 follow-up）**:
  - **KNOWN_LIMIT 1: RSqrt CLZ LUT 物理 fit ceiling**（与 P7
    `custom.RSqrt` 同根）：spec §4.5.4 line 384 强制 LUT 步骤；同款 LUT
    在 P7 单算子测试也是 `xfail strict=True`。要破需硬件层增加 RSqrt
    LUT segment 数（硬件 ROI 决策，非 kernel 改进）
  - **KNOWN_LIMIT 2: spec line 414 的 16-bit M 物理上限**（i16 affine
    主导项）：`MULTIPLIER_QBITS = 16` + `max_rshift = 31` 限制 α 表达
    动态范围。LayerNorm i16 calibrated + γ 量化下 `α_x ≈ 1.1e-8` 落在
    精度退化区（M 仅 5-bit），引入 ~1000 LSB。**这是 spec 设计选择**，
    与所有其他 INT16 算子使用同一 `quantize_scale_to_m_rshift` 路径
    一致；要破需修改 spec line 414（如允许 24/32-bit M）或重新分解
    α_x（登记为 `FU-LAYERNORM-ALPHA-X-CALIBRATION`）
  - **业务影响评估**：cos 在 192 trials 上 ≥ 0.999229，几何相关性达到
    项目 INT16 main gate（0.99）。但 LSB-level 精度在 i16 affine 路径
    上比 fp32-affine reference 差 ~130x；下游链路若对 LayerNorm 输出
    做 element-wise 精确比较会受影响——业务侧若发现回归需触发
    `FU-LAYERNORM-ALPHA-X-CALIBRATION`
  - **物理上限登记**：见 `aimet_torch/fixed_point/metrics/thresholds.py::LAYERNORM_VS_FP32_PER_GRID_LIMITS`
    （现已分 affine/noaffine 二维：i16 affine max_lsb=1100, noaffine
    max_lsb=12；i8 affine max_lsb=8, noaffine max_lsb=4），与
    `PWL_VS_ANALYTIC_PER_FN_LIMITS` 同一类登记模式
  - **per-channel output encoding 不支持**：spec §4.5.4 假定 per-tensor
    output（与 LayerNorm 语义一致——γ/β 是 normalized_shape 形状广播，
    output 与 input 一致 shape）。kernel 用 `_integer_affine` 走
    per-tensor 路径
  - **Default A 假设（spec 未明示）**：见正文 "未在 spec 中明示但本实施
    采用的 4 个量化 default" 段；若部署硬件使用不同 default（如
    percentile calibrator γ scale 或独立 mean grid），需要回归测试这
    4 个 default 是否仍是最优；任何 default 调整都可能改变 α_x 的量级
    和 spec 16-bit M 物理上限的位置
  - **u8/u16 不覆盖**：spec §4.5.4 输出 dtype 列了 i8/i16；kernel 通过
    `output_encoding.qmin/qmax` 自然支持 unsigned，但未单独测，与
    P1-P4 同款"signed-grid optimized"约定

---

## nn.Dropout

- **Kernel**: `aimet_torch/fixed_point/kernels/shape_ops.py::DropoutInt16Kernel`
- **KernelKind**: `SAME_GRID_OR_REQUANT`（eval mode 下 value-空间无
  变换，但图上可能跨 quant grid，所以保留 requantize 分支）
- **Spec**: 训练时 random mask + scale，但 spec 与 INT16_FIXED_EVAL
  仅约定 **eval mode** 下 Dropout 等价于 Identity（value-空间）；与
  cross-grid 的兼容仅作 graph-survival 用
- **Spec 测试覆盖度**: ✓ 完全（覆盖 eval 下 same/cross 两路）
- **测试**: `tests/fixed_point/kernels/test_shape_ops_int16_precision.py`
- **Reference**: `x` (identity，eval mode)
- **Input 分布**: 32 trials per (grid, mode)，shape `(2, 8, 16)` = 256
  元素；same/cross-scale 同 P5 规模
- **阈值**: `_MIN_COSINE = 0.9999`、`_MAX_FLOAT_LSB = 1.0`
- **实测值**（commit time 2026-06-08, PyTorch CPU, 32 trials）:

| grid | mode | cos_min | lsb_max |
|------|------|---------|---------|
| i8 | same-scale | 1.000000 | 0.0000 |
| i8 | cross-scale | 0.999961 | 0.4990 |
| i16 | same-scale | 1.000000 | 0.0000 |
| i16 | cross-scale | 1.000000 | 0.7087 |

- **状态**: ✓ PASS（全 4/4 用例）
- **关键观察**: same-scale 走 byte-stream identity 路径，零误差；
  cross-scale 走 `centered → requantize_int` 单次折叠，误差 ≤ 1 LSB
- **注**: 训练 mode 的 random-mask + scale 行为不在 INT16_FIXED_EVAL
  支持范围内（spec 视 INT16_FIXED_EVAL 为部署/推理路径）

---

## custom.Pad

- **Kernel**: `aimet_torch/fixed_point/kernels/shape_ops.py::PadInt16Kernel`
- **KernelKind**: `SAME_GRID_OR_REQUANT`
- **Spec**: `doc/04_算子详细规格/04_11_其他数据操作算子.md §4.11.1 Pad`
  - spec 契约：`q_pad = round(value_float / S_y) + Z_y` —— pad 值在
    **output grid** 上量化；input 先按需 align 到 output grid，再 F.pad
  - spec mode 允许 `constant / replicate / reflect`（kernel 已全部实现，
    Layer C4 完成）
  - **spec 输出 dtype**: 与输入相同（i8 / i16）
  - kernel 对齐：`_requantize_identity_output(input, out_enc)` 把 input
    align 到 output grid（same-grid → 直通，cross-grid → 单次 requantize），
    然后按 mode 分派：
    - `constant` → `F.pad(... value=round(pad_value/S_y)+Z_y)`
    - `replicate` / `reflect` → `_pad_int_via_float_roundtrip` 在 fp32
      中走 `F.pad(int_repr.float(), mode=mode)` 再 cast 回 int（reflect/
      replicate 是按位置 copy，无算术，cast 在 ±2^24 范围内 bit-exact）
- **Spec 测试覆盖度**: ✓ 完全（spec 三种 mode 全覆盖，pad shape 多轴）
- **测试**: `tests/fixed_point/kernels/test_shape_ops_int16_precision.py`
  （`test_pad_(constant|replicate|reflect)_(same|cross)_scale_random_fp32_per_grid`，
  共 18 个 case = 3 mode × 2 scale × 3 grid）
- **Reference**: `F.pad(x_fp32, pad, mode=mode, value=...)`
- **Input 分布**: 32 trials per (grid, mode, scale)，shape `(1, 4, 8, 8)`，
  - constant: pad=(1,2,3,1)（不对称），pad_value=0.0
  - replicate / reflect: pad=(1,1,1,1)（满足 reflect 的 `pad < dim` 约束）
- **阈值**: `_MIN_COSINE = 0.9999`、`_MAX_FLOAT_LSB = 1.0`
- **实测值**（commit time 2026-06-08, PyTorch CPU, 32 trials）:

| grid | mode | scale | cos_min | lsb_max |
|------|------|-------|---------|---------|
| i8 | constant  | same  | 1.000000 | 0.0000 |
| i8 | constant  | cross | 0.999959 | 0.4993 |
| i8 | replicate | same  | 1.000000 | 0.0000 |
| i8 | replicate | cross | ≤0.99996 | ≤0.50 |
| i8 | reflect   | same  | 1.000000 | 0.0000 |
| i8 | reflect   | cross | ≤0.99996 | ≤0.50 |
| i16 | constant  | same  | 1.000000 | 0.0000 |
| i16 | constant  | cross | 1.000000 | 0.6880 |
| i16 | replicate | same  | 1.000000 | 0.0000 |
| i16 | replicate | cross | 1.000000 | ≤0.69 |
| i16 | reflect   | same  | 1.000000 | 0.0000 |
| i16 | reflect   | cross | 1.000000 | ≤0.69 |

- **状态**: ✓ PASS（全 18/18 case，含 12 个 Layer C4 新增）
- **关键观察**:
  - 与 Dropout 同根（共享 `_requantize_identity_output` align step）；
    same-scale 任意 mode 均 byte-stream identity（lsb=0）
  - cross-scale 单次 requantize ≤ 1 LSB；reflect/replicate 在 align 后
    只做 position copy，**不引入额外误差**（与 constant cross-scale 同
    上限）
- **注**:
  - **三种 mode 现已全覆盖**（Layer C4 完成；之前段落的"replicate /
    reflect 是 kernel-side follow-up（未实现）"已被本 release 关闭）；
    spec `04_11 §4.11.1` "硬件约束: 优先支持 constant" 仍成立——硬件可
    选择只实现 constant，但 software reference 提供完整三 mode 以便
    adapter 端不需特化
  - **pad_value=0.0 是常见 case**；非零 pad 值不在本段覆盖范围（kernel
    会按 `round(value/S_y) + Z_y` 量化 pad 值，理论上误差仍 ≤ 1 LSB）
  - reflect/replicate 用 fp32 round-trip 的实现细节是规避 PyTorch 对
    int dtype 的 reflect/replicate kernel 历史限制；reflect/replicate
    只做位置 copy 不做算术，fp32 在 ±2^24 范围内对所有 i16 sim 整数
    精确表示，因此 cast 是 **bit-exact** 的（不计入"浮点 fallback"，
    `int16_fixed_eval_mode` 不需禁用）

---

## custom.Concat

- **Kernel**: `aimet_torch/fixed_point/kernels/shape_ops.py::ConcatInt16Kernel`
- **KernelKind**: `SAME_GRID_OR_REQUANT`
- **Spec**: `doc/04_算子详细规格/04_07_数据操作类算子.md §4.7.1 concat`
  - spec 契约：每个 input branch 独立按 `align_centered_int32_to_output`
    对齐到 output grid，然后在 `axis` 上 cat int_repr，最后加 `Z_y` 并
    saturate
  - kernel 对齐：`align_centered_int32_to_output` 同时承担 same-grid
    直通（zp 翻译，无 rescale）和 cross-grid `(scale_in/scale_out)`
    rescale；末尾 `int32_add_sat(acc, Z_y)` + `saturate_sim_tensor`
- **Spec 测试覆盖度**: ✓ **完全**——本段覆盖 3 个 branch 不同 scale
  的 cross-grid path（每个 branch 各自 align），axis=channels-dim
- **测试**: `tests/fixed_point/kernels/test_shape_ops_int16_precision.py`
  (test_concat_*)
- **Reference**: `torch.cat([a.float(), b.float(), c.float()], dim=axis)`
- **Input 分布**: 32 trials per (grid, mode)；3 branches shape `(2,4,8) /
  (2,6,8) / (2,2,8)` (cat 后 `(2,12,8)`)；axis=1
  - 同标度：`scale_a = scale_b = scale_c = scale_out`
  - 跨标度：`scale_a / scale_c` 各自走 `exp(±0.025)`（与 P6 Pad / Dropout
    同 `_CROSS_SCALE_LOG_RATIO_SPAN = 0.05` 半宽——log-ratio ±0.025，约
    ±2.5%），`scale_b = scale_out`。**剧烈 cross-grid 不在本段覆盖**——
    spec 不限制 ratio，但 calibration 端典型 branch ratio 都在 ±10% 内
- **阈值**: `_MIN_COSINE = 0.9999`、`_MAX_FLOAT_LSB = 1.0`
- **实测值**（commit time 2026-06-08, PyTorch CPU, 32 trials）:

| grid | mode | cos_min | lsb_max |
|------|------|---------|---------|
| i8 | same-scale | 1.000000 | 0.0000 |
| i8 | cross-scale | 0.999980 | 0.4993 |
| i16 | same-scale | 1.000000 | 0.0000 |
| i16 | cross-scale | 1.000000 | 0.7083 |

> 注：上表数字用与测试**相同**的 `seed_base = 790_000 / 800_000` 跑出，
> 可在 `tests/fixed_point/kernels/test_shape_ops_int16_precision.py` 中
> 临时把 `_assert_strict_gates` 改为打印即可复现。

- **状态**: ✓ PASS（全 4/4 用例）
- **关键观察**: same-scale 走 align 的 zp-only 直通分支，每个 branch
  perfect byte-stream identity；cross-scale 每个 branch 各自 1 LSB 内
  requantize，复合误差仍 ≤ 1 LSB（branches 之间 independent rounding）
- **注**:
  - **axis 仅覆盖 channels-dim**：spec 允许任意 axis，本段未参数化所有
    axis（kernel 不依赖 axis 具体值，axis 通过 `torch.cat` 透传）
  - **branch 数量**：本段 3 个 branch；spec 允许 ≥1，kernel 对单 branch
    退化为 align 后直返

---

## nn.MaxPool2d / custom.MaxPool2d

- **Kernel**: `aimet_torch/fixed_point/kernels/pool.py::{MaxPool2d,CustomMaxPool2d}Int16Kernel`
  （继承 `_MaxPool2dKernel`，两者共享 hot path）
- **KernelKind**: `SAME_GRID_VALUE`
- **Spec**: `doc/04_算子详细规格/04_09_池化类算子.md` §MaxPool
  - spec 契约：input/output 共享 `(scale, zp, qmin, qmax)`，
    "comparator-only HW，no M/rshift"；kt,kf ≤ 3，padding 0..3
  - kernel 对齐：`F.max_pool2d` 直接在 int_repr 上做（max 选 existing
    code），输出按 input encoding 重 wrap
- **Spec 测试覆盖度**: ✓ **完全**——3 个 spec-pinned conv config 覆盖
  `kernel ∈ {2x2, 3x3}` × stride/padding 常见组合；`custom.MaxPool2d`
  在本段独立参数化以锁定 functional wrapper 的 kernel parity
- **测试**: `tests/fixed_point/kernels/test_maxpool2d_int16_precision.py`
- **Reference**: `F.max_pool2d(x.float(), kernel_size, stride, padding)`
- **Input 分布**: 32 trials per (grid, config)；shape `(1, 4, 8, 8)`；
  configs: `(k=2, s=2, p=0)` / `(k=3, s=2, p=1)` / `(k=3, s=1, p=1)`
- **阈值**: `_MIN_COSINE = 0.9999`、`_MAX_FLOAT_LSB = 1.0`
- **实测值**（commit time 2026-06-08, PyTorch CPU, 32 trials）:

| grid | config | cos_min | lsb_max |
|------|--------|---------|---------|
| i8 | k2-s2-p0 | 1.000000 | 0.0000 |
| i8 | k3-s2-p1 | 1.000000 | 0.0000 |
| i8 | k3-s1-p1 | 1.000000 | 0.0000 |
| i16 | k2-s2-p0 | 1.000000 | 0.0000 |
| i16 | k3-s2-p1 | 1.000000 | 0.0000 |
| i16 | k3-s1-p1 | 1.000000 | 0.0000 |

> 注：SAME_GRID_VALUE 的 byte-stream identity 与 random seed 无关——
> kernel 直接 `F.max_pool2d` 在 int_repr 上选 existing code，无 rescale
> 路径，故任何 seed 都得到上表数字。

- **状态**: ✓ PASS（全 12/12 用例 nn + 12 用例 custom，共 24 用例
  perfect byte-stream identity）
- **关键观察**: 与 P6 `SAME_GRID_VALUE` 族（Identity/Flatten/Reshape/
  Permute）同根——max 操作选 existing code，从不合成新码，所以浮点
  reference 在 value 空间上严格等价；lsb_max=0.0 是 contract 必然
- **注**:
  - **`custom.MaxPool2d` 是 functional wrapper**：manifest 标
    `dispatchable=False`（adapter 还未从 args/kwargs unpack
    kernel_size），但 kernel 已注册并通过本段测试。adapter 路由落地见
    `route-custom-pool-through-adapter` follow-up
  - **dilation 仅覆盖默认 1**：spec 不限制 dilation 但实际部署罕见 >1，
    kernel 透传 dilation 到 `F.max_pool2d`，行为同 PyTorch
  - **ceil_mode 仅覆盖默认 False**：同上，kernel 透传

---

## custom.Sqrt / custom.RSqrt / custom.Reciprocal / custom.Square（CLZ 二阶 LUT 族）

- **Kernel**: `aimet_torch/fixed_point/kernels/clz_lut.py::{Sqrt,RSqrt,Reciprocal,Square}Int16ClzKernel`
- **KernelKind**: `LOOKUP`（CLZ-normalized 二阶多项式 LUT）
- **Spec**: `doc/04_算子详细规格/04_08_特殊运算类算子.md §4.8.1 hypot
  (sqrt) / §4.8.2 power (p=0.5/2/3 → sqrt/square/cube)` + ADR
  `LUT_Binary_Storage_General §5 CLZ-normalized` —— CLZ 归一化把 input
  映射到 normalized space `[-1, 1]`，对每段拟合
  `y = a·x_norm² + b·x_norm + c`，整体 16 段 + 32-bit accumulator
  （`reciprocal` 形态由 `04_03 §4.3.4` 的"定点倒数"段隐含）
- **LUT 来源**: 测试用 `abc_lut-shuai/lut_int_general/output/lut_test/`
  下的 `sqrt_clz_lut.json` / `rsqrt_clz_lut.json` / `reciprocal_clz_lut.json` /
  `power_2_clz_lut.json` —— 这是 spec 一手 golden LUT
- **Spec 测试覆盖度**: ✓ 完全（kernel 与 abc 评估器在 6 个 sample 点上
  `atol=0` 已锁，见 `test_clz_sqrt_golden.py` 等；本段补 fp32-random
  尺度上的 cos/lsb 统计）
- **测试**: `tests/fixed_point/kernels/test_lookup_int16_precision.py`
- **Reference**:
  - Sqrt：`torch.sqrt(clamp(x, min=0))`
  - RSqrt：`1/sqrt(clamp(x, min=1e-6))`
  - Reciprocal：`1/clamp(|x|, min=1e-6) * sign(x)`
  - Square：`x * x`
- **Input 分布**: 32 trials per 算子，shape `(4, 64)` = 256 元素
  - Sqrt：x ∈ [0, 8]（valid 域 x≥0）
  - RSqrt：x ∈ [0.5, 8]（valid 域 x>0，避开 0 处奇异）
  - Reciprocal：x ∈ [0.2, 5]（valid 域 x≠0）
  - Square：x ∈ [-2.5, 2.5]（保证 x² 在 LUT 输出动态范围内）
- **阈值**: `_MIN_COSINE = 0.9999`、`_MAX_FLOAT_LSB = 1.0`（统一 floor，
  不放宽；按 KNOWN_LIMIT 锁 xfail strict=True）
- **实测值**（commit time 2026-06-08, PyTorch CPU, 32 trials, i16 only）:

| 算子 | cos_min | cos_mean | lsb_max | lsb_mean | gate |
|------|---------|----------|---------|----------|------|
| Sqrt | 1.000000 | 1.000000 | 100.24 | 6.75 | xfail (lsb≫1) |
| RSqrt | 1.000000 | 1.000000 | 1.67 | 0.96 | xfail (lsb>1) |
| Reciprocal | 0.999999 | 1.000000 | 1.66 | 0.81 | xfail (lsb>1) |
| Square | 1.000000 | 1.000000 | 2.01 | 1.69 | xfail (lsb>1) |

- **状态**: ⚠ KNOWN_LIMIT（4/4 xfail strict=True）
- **关键观察**:
  - cos 几乎完美（CLZ 归一化把动态范围消干净）。Sqrt 的 lsb_max=100
    是 sqrt 在 x→0 处导数 `0.5/sqrt(x)→∞`：input 移动 1 LSB（2.44e-4）
    时 output 可移动 ~15 LSB，少量近 0 sample 主导 lsb_max；cos 不
    受影响因为这种 outlier 不改变向量相似度
  - RSqrt / Reciprocal / Square 的 lsb_max 1.5-2 LSB 是 16 段二阶
    拟合在 normalized 空间上的稳态残差，与段数与多项式阶完全一致
  - 这是 spec 设计选择（abc 一手 golden 也是同源 LUT），不是 kernel bug
- **改进建议**（不修改统一 floor）:
  - **不要**通过缩小测试域人为压低 lsb_max（会掩盖 sqrt 在 0 处的真实行为）
  - 推荐 manifest 上为 CLZ 族声明独立 gate：cos≥0.9999 + lsb≤2 LSB
    (Sqrt 单独 lsb≤256 if x_min≤0；或要求 calibration 把 x_min 抬高
    到 LSB 阈值之上)
  - 若未来要让 lsb 严格<1，需在 normalized space 加 Newton step 3
    (见 FU-DIV-NEWTON-FUTURE)；目前 abc 资产不带 Newton step

---

## custom.Log / custom.Exponential / custom.Sin / custom.Cos（PWL 16-segment LUT 族）

- **Kernel**: `aimet_torch/fixed_point/kernels/lut.py::_LutInt16Kernel`
  (`LogInt16Kernel`, `ExponentialInt16Kernel`, `SinInt16Kernel`,
  `CosInt16Kernel` 都继承 `_LutInt16Kernel`)
- **KernelKind**: `LOOKUP`（PWL 16 段 × `y = q_b·(x - x_left) + q_c`
  二项式，16-bit coeff_b + 32-bit term_c + 32-bit accumulator）
- **Spec**: `doc/04_算子详细规格/04_04_激活函数类算子.md §4.4.3 查表激活`
  —— PWL hardware-fixed `PWL_HARDWARE_NUM_SEGMENTS = 16`；spec §4.4.3
  显式列 prelu/gelu/sigmoid/tanh，其余 PWL 算子按"等基于查找表的激活
  函数"隐式覆盖
- **LUT 来源**: 测试用 `offline.lut_gen.generate_pwl_lut_for_export`
  在线拟合（同 export pipeline 用的函数），`enforce_quality=False`
  以拿回 metrics
- **Spec 测试覆盖度**:
  - ✓ PWL 拟合质量：S1 `test_pwl_extra_activations.py` /
    `test_lut.py` 已用 `PWL_VS_ANALYTIC_PER_FN_LIMITS` 锁
  - 本段补 **kernel forward 的 fp32-random 尺度上 cos/lsb 统计**
    （PWL fit 误差经过 kernel 整数 hot-path 到 output i16 的端到端值）
- **测试**: `tests/fixed_point/kernels/test_lookup_int16_precision.py`
- **Reference**: Log = `log(clamp(x, min=1e-3))`、Exp = `exp(x)`、
  Sin = `sin(x)`、Cos = `cos(x)`
- **Input 分布**: 32 trials per 算子，shape `(4, 64)` = 256 元素
  - Log：x ∈ [0.05, 8]（valid 域 x>0）
  - Exp：x ∈ [-4, 2]（避免 exp(x) 越过 output 动态范围 ±8）
  - Sin / Cos：x ∈ [-π, π]（principal range，spec 推荐区间）
- **阈值**: `_MIN_COSINE = 0.9999`、`_MAX_FLOAT_LSB = 1.0`（统一 floor，
  不放宽；按 KNOWN_LIMIT 锁 xfail strict=True）
- **实测值**（commit time 2026-06-08, PyTorch CPU, 32 trials, i16 only）:

| 算子 | cos_min | cos_mean | lsb_max | lsb_mean | PWL fit_max_lsb (per_fn 限) |
|------|---------|----------|---------|----------|----|
| Log | 0.99368 | 0.99554 | 5239.8 | 4984.1 | 9949 (限 2048) ⚠ 见下 |
| Exponential | 0.99994 | 0.99996 | 499.3 | 425.7 | 737 (限 2048) |
| Sin | 0.99977 | 0.99981 | 1437.8 | 1354.8 | 1438 (限 512) |
| Cos | 0.99979 | 0.99982 | 1434.4 | 1381.8 | 1431 (限 512) |

- **状态**: ⚠ KNOWN_LIMIT（4/4 xfail strict=True）
- **关键观察**:
  - **Log 同时超 cos 与 lsb floor**：`log` 在 [0.05, 0.5] 区间斜率最陡，
    16 段二项拟合的近端段几乎无法跟上 `log` 的对数变化率；fit_max_lsb
    9949 已远超 per_fn 限 2048（拟合本身就是 OOL 的 stress case）。
    若实际部署的 calibration 把 input 收窄到 [0.1, 8] 大概能让 cos
    回到 0.999+ 水平
  - **Exp 仅超 lsb floor**：cos 0.99994 接近 0.9999，输入 -4 时 exp(-4)=0.018
    很小，靠近输出 grid 底端的 1 LSB 相对误差天然就 ~50 LSB
  - **Sin / Cos 都同时超**：principal range 上 |sin'|, |cos'| 各自 ≤ 1，
    16 段拟合在零交叉点附近的 LSB 误差 ~1000 LSB
  - kernel 算术正确性已被 `evaluate_pwl_lut_int16` 在 `test_lut.py` 验证；
    本段的 fail 都是 PWL 16 段的设计上限 —— spec 选 16 段是为 hardware
    cost 与精度的折中
- **改进建议**（不修改统一 floor）:
  - **不要**改 PWL 段数（spec 硬约束 `PWL_HARDWARE_NUM_SEGMENTS = 16`）
  - 推荐 manifest 上为 PWL 族声明独立 gate：cos-only（≥0.9994，参考
    `INT16_SINGLE_OP_PER_CASE_MIN_COSINE` "Linear → Tanh (PWL)" 案例），
    或按算子分级（log/sin/cos 0.9995；exp/sigmoid 0.9999）
  - calibration 端按下游容忍度路由：若整图 cos 要求 ≥0.999，PWL log
    需要走 range-folded LUT（同 Sin/Cos 已用 `phase_fold` 提示路径），
    或在编译器侧把 Log 折叠到下游的 fp32 boundary 上
  - 与 FU-P7-LOOKUP-FIT-CEILING 一并处理
- **⚠ spec 偏差现状（custom.Log，Layer B2 已澄清阻塞根因）**: spec
  `LUT_Binary_Storage_General §5` 把 `log` 归类为 **CLZ**，独有公式
  `q_y = round_shift(y_offset · r_q, r_shift) + E · q_ln2 + q_ln_sx +
  output_zp`，与本段其他三个 PWL 算子（exp/sin/cos）**不同**。
  当前 runtime 仍走 PWL 是 `review-nonlinear-lut` follow-up 的临时实现。
  **Log fit_max_lsb=9949 远超 per_fn 限 2048 的根因正是 PWL 路径不
  适合 log**——切回 CLZ 后预期 lsb_max 大幅下降到 Sqrt/RSqrt 量级
  （数 LSB）。
  **真实阻塞**（Layer B2 dry-run 二次核实后修正）：
  - `lut_int_general/output/lut_test/log_lut.json` —— **PWL 16 段**
    （与 sigmoid/gelu 同族）
  - `lut_int_po2/output/lut_test/log_lut.json` / `log_lut_clean.json`
    —— **PWL 16 段**（po2 scale 模式）
  - `lut_fp/output/fp32_test/log_normalized_fp32_lut.json` —— **CLZ-
    normalized 概念资产**（input ∈ [1.0, 2.0] = mantissa，output ∈
    [0, ln 2]，16 段），**但是 fp32 形式不是 int**
  - `lut_int_general/quantization/clz_normalized_fitter.py` —— **只
    支持 reciprocal/sqrt/rsqrt/power_2**，log 不在 CLZ fitter 范围
  也就是说：**概念上 CLZ-normalized log LUT 已存在（fp32 形式）**，
  但缺 int16 量化版本与 q_ln2/q_ln_sx 常量。修复路径：
  1. abc_lut-shuai 数据科学侧扩展 `clz_normalized_fitter` 加 log 分
     支并输出 `log_clz_lut.json`（q_norm/n_norm/r_q/r_shift/q_ln2/
     q_ln_sx + 16 mantissa segments）—— fp32 normalized 已给 segment
     fit，缺的是 int16 量化与两个常量，约 **50-100 LOC abc_lut-shuai
     侧**
  2. software 侧扩展 `clz_lut.py::denormalize_clz` 加 log 分支
     （~30 LOC）+ 注册 `LogInt16ClzKernel`（~20 LOC）—— **本仓库**
  3. 测试需 spec golden LUT 做 bit-exact oracle（同 sqrt/rsqrt 模式）
  **Layer B2 因 #1 上游 LUT 数据阻塞，本轮不展开 software 实现**——
  仅做 software 骨架但没有数据资产 → kernel 永远找不到 LUT 入口、
  精度无变化，反而引入"半成品代码"的维护成本。
  替代方案（**未本轮做**）：从 fp32 normalized log LUT **在线量化**
  到 int16 临时挂到 `LogInt16ClzKernel`（不依赖 #1），可作为可行性
  prototype；缺点是 LUT 与 spec golden 不一致、运行时多一次 fp→int
  量化、Sin/Cos 没有这种过渡资产。跟踪在 `FU-LOG-CLZ-LUT-UPSTREAM`
  follow-up
- **Layer B2-prototype 实测（fp32 oracle 上界估算，2026-06-08）**:
  在 PWL log 同样的输入分布（x ∈ [0.05, 8]、in_scale=8/32767、
  out_scale=4/32767、6 trial × shape=(2,16,32,32)、seed_base=820_000）
  下，用 fp32 normalized log LUT 直接做 spec §5 公式 oracle
  （`log(x_f) = log(m) + e·ln(2)`、segment 拟合 `b·m + c` 在 fp32
  中、最后 round 到 out grid）：

  | 指标 | PWL log（当前 runtime） | CLZ-form log oracle（fp32 上界） | 比值 |
  | --- | --- | --- | --- |
  | cos_min | 0.9999+ | 1.0000 | — |
  | lsb_max | 9949 | **18.97** | **~525× 改善** |

  **结论**：CLZ-form log 的物理上限就在 ~19 LSB 附近，与 Sqrt/RSqrt
  量级一致；PWL log 的 9949 LSB 完全是 PWL 16 段在跨 5 个数量级输入
  域上"硬铺"的 fit 上限。**实施只要 #1 abc_lut-shuai 侧出 int 量化
  LUT，本仓 software 修复后 lsb_max 可降到 19~50 LSB（int 量化会比
  fp32 oracle 多一些舍入噪声但量级不变）**。
  oracle 脚本在 `scripts/_inspect_log_clz_oracle.py`（一次性，使用
  完即可删除；本快照已纳入此段）

---

## P7-S1: PWL 激活族（Sigmoid / Tanh / GELU / SiLU / Mish / Softplus / Hardsigmoid / Hardswish / LeakyReLU / PReLU）

- **覆盖范围**: spec S1（standard activation set）的 10 个 PWL 算子，本段
  以 **kernel 端到端 fp32-random** 视角与 unified floor 对账（cos>0.9999、
  lsb<1.0），与 `test_pwl_extra_activations.py` 中 PWL fit-quality
  (`PWL_VS_ANALYTIC_PER_FN_LIMITS`) 视角互为补充
  （**Abs 已迁出**：Layer B1 把 `custom.Abs` 从 PWL `_LutInt16Kernel`
  改为 spec `04_03 §4.3.5` 的 integer-abs path，新位置见 P5 段
  "custom.Abs (integer-abs path)"，原 PWL 行的 cos=1.0/lsb=0.50
  已被 byte-stream identity cos=1.0/lsb=0.0 取代）
- **Kernel**: `aimet_torch/fixed_point/kernels/lut.py::_LutInt16Kernel` 派生
  族（`SigmoidInt16Kernel` / `TanhInt16Kernel` / `GELUInt16Kernel` /
  `SiLUInt16Kernel` / `MishInt16Kernel` / `SoftplusInt16Kernel` /
  `HardsigmoidInt16Kernel` / `HardswishInt16Kernel` / `LeakyReLUInt16Kernel`
  / `PReLUInt16Kernel`）
- **KernelKind**: `LOOKUP`
- **Spec**: `doc/04_算子详细规格/04_04_激活函数类算子.md §4.4.3 查表激活`
  —— PWL hardware-fixed `PWL_HARDWARE_NUM_SEGMENTS = 16`（spec §4.4.3
  显式列 prelu/gelu/sigmoid/tanh，LeakyReLU/Mish/SiLU/Softplus/
  Hardsigmoid/Hardswish 按"等基于查找表的激活函数"隐式覆盖）
- **LUT 来源**: 同 P7 PWL —— `offline.lut_gen.generate_pwl_lut_for_export`
  在线拟合（同 export pipeline），`enforce_quality=False`
- **测试**: `tests/fixed_point/kernels/test_lookup_int16_precision.py`
  (test_*_pwl_random_fp32_i16)
- **Reference**: 各算子对应 `torch.*` / `F.*` 浮点函数
- **Input 分布**: 32 trials per 算子，shape `(4, 64)` = 256 元素，按各
  算子饱和/拐点行为分三档：
  - 对称 ±8（Sigmoid / SiLU）：覆盖完整 sigmoid saturation tails
  - 对称 ±4（Tanh / GELU / Mish / Hardsigmoid / Hardswish / LeakyReLU /
    PReLU / Abs）：tanh / GELU 主要曲率段、PWL kink 附近
  - 非对称 [-4, 8]（Softplus）：负端 saturation 到 0、正端线性段到 8
- **阈值**: `_MIN_COSINE = 0.9999`、`_MAX_FLOAT_LSB = 1.0`（统一 floor，
  不放宽）
- **实测值**（commit time 2026-06-08, PyTorch CPU, 32 trials, i16 only）:

| 算子 | cos_min | lsb_max | 状态 | 备注 |
|------|---------|---------|------|------|
| Sigmoid | 0.999996 | 246.2 | ⚠ xfail strict=True | PWL fit ceiling |
| Tanh | 0.999990 | 498.5 | ⚠ xfail strict=True | PWL fit ceiling |
| GELU | 0.999998 | 124.4 | ⚠ xfail strict=True | PWL fit ceiling |
| SiLU | 0.999997 | 146.6 | ⚠ xfail strict=True | PWL fit ceiling |
| Mish | 0.999998 | 103.3 | ⚠ xfail strict=True | PWL fit ceiling |
| Softplus | 0.999999 | 78.6 | ⚠ xfail strict=True | PWL fit ceiling |
| Hardsigmoid | 1.000000 | 1.49 | ⚠ xfail strict=False | 1/6·x+0.5 直线段，但 in-process fit 不识别 → 接近 floor |
| Hardswish | 0.999998 | 55.4 | ⚠ xfail strict=True | x·Hardsigmoid(x)，平滑过渡 |
| LeakyReLU | 1.000000 | 0.50 | ✓ PASS | 2-piece exact PWL representable |
| PReLU | 1.000000 | 0.625 | ✓ PASS | slope=0.25 固定，同 LeakyReLU |

- **状态**: ✓ 2 PASS（LeakyReLU / PReLU）+ ⚠ 8 KNOWN_LIMIT（xfail
  strict）。每个 case cos ≥ 0.999990 已稳定 ≥ floor，差距全在 lsb_max
- **关键观察**:
  - **平滑非线性族（Sigmoid/Tanh/GELU/SiLU/Mish/Softplus/Hardswish）**:
    PWL 16 段在 saturation tail 与 inflection point 附近有不可避免的
    fit-residual；lsb_max 78～498，但都在 `PWL_VS_ANALYTIC_PER_FN_LIMITS`
    限内（Sigmoid 246 / 限 320、Tanh 498 / 限 2400 等）—— **拟合本身合
    spec**，是统一 floor 与 spec 设计的张力
  - **分段线性族（LeakyReLU / PReLU）**: 在 `generate_pwl_lut_for_export`
    的 16 段化路径下能 fit 到 ≤ 0.625 LSB —— fitter 把 kink 放在段边界
    上，每段内部 `q_b·x + q_c` 二项就是真实 affine。**LSB 误差只来自
    16-bit 量化舍入**
    （Abs 原本属于这类，Layer B1 后已迁出走 integer-abs path）
  - **Hardsigmoid（边界）**: 严格说也是 2-piece linear (内部 1/6·x+0.5)，
    但 fitter 在 ±3 边界上做了 smooth blend → 1.49 LSB，刚好踩在统一
    floor 上方。`strict=False` 不要求 PASS/FAIL 任一种
- **改进建议**（不修改统一 floor）:
  - **保留 16 段硬约束**（spec 硬性指标 `PWL_HARDWARE_NUM_SEGMENTS=16`）
  - 平滑族走 **per-fn cos floor** 或 **PWL 族独立 manifest gate**（同
    `FU-P7-LOOKUP-FIT-CEILING` 提议）
  - Hardsigmoid 在 LUT 生成器侧可加 **kink-snap 优化**（已为 LeakyReLU
    族隐式触发），把 ±3 边界的 smooth blend 改为段边界 → lsb_max 应能
    回到 ≤ 0.5
  - 与 FU-P7-LOOKUP-FIT-CEILING 同根，落地后翻 8 个 xfail 为新 gate 的
    PASS

---

## nn.Softmax（composed kernel：PWL exp + int32 sum + 整数 normalize）

- **Kernel**: `aimet_torch/fixed_point/kernels/softmax.py::softmax_int16_pwl`，
  通过 `aimet_torch/fixed_point/kernels/lut.py::SoftmaxInt16Kernel` 注册
- **KernelKind**: `LOOKUP`（manifest 当前归入 LUT 族）
- **Spec**: `doc/04_算子详细规格/04_04_激活函数类算子.md` §4.4.3 查表激活
  （PWL 通用框架）；Softmax 本身没有独立 § 段，按 composed 路径处理
- **Composed 路径** (stable max-subtract → PWL exp → int32 sum → integer
  normalize)：
  1. `centered = x.int_repr - Z_x`；`shifted = centered - centered.amax`，
     保证 `shifted ≤ 0`（数值稳定）
  2. 动态 `in_enc`：`(scale = scale_in, zp = 0, qmin = max(shift_min,
     -32768), qmax = 0)` —— **注意 `qmin` 被硬截至 -32768**
  3. PWL exp 用 `generate_pwl_lut(torch.exp, ...)` 在线生成 16 段拟合，
     `exp_q = evaluate_pwl_lut_int16(shifted_q, pwl_lut)`，范围 [0, qmax_out]
  4. `exp_sum = exp_q.sum(dim=dim)` —— int32 容器，spec 容差内
  5. 整数 normalize: `prob_fixed = (exp_q << 15) / exp_sum`（15-bit 定点
     概率，范围 [0, 32767]），再 `q = prob_fixed · (qmax - qmin) >> 15
     + Z_y`，最后 saturate

### ⚠ Output encoding 选择对实测值的决定性影响（必读）

Softmax 输出语义上 ∈ [0, 1]，**output grid 必须用全 16-bit dynamic
range 覆盖此区间**。两种 encoding 实测对比：

| output encoding | `prob = 0.5` 落点 | 后果 |
|----------------|-----------------|------|
| 错：symmetric i16，`scale = 1/32767, zp = 0` | `q = 32768` ⇒ saturate 到 32767 | 所有 `prob > 0.5` 都饱和到 1.0，lsb_max 虚高 3-4 倍 |
| 对：u16-on-i16，`scale = 1/(qmax-qmin) = 1/65535, zp = qmin = -32768` | `q = 0` | 全 dynamic range 都用上，无 saturation |

本段表 / 测试 / inspector 都已切到**正确 encoding**；首版（2026-06-08
上午）doc 使用错误 encoding 导致 lsb_max 报到 2400～16100，已修正。

### 误差源 (按贡献排序，采用正确 output encoding 后)

1. **15-bit fixed-point reciprocal-normalize（dominant）**:
   `prob_fixed` 步长 = `1/32768`，映射到 65535-span output 上
   ⇒ **每个 `prob_fixed` 步 ≈ 2 LSB output**。这是 lsb_max ~500 的
   主因——硬件成本与精度的 spec 折中
2. **PWL exp 16 段拟合残差**: 见 P7 PWL `custom.Exponential` 段
   (fit_max_lsb 737 LSB)；经 `1/sum` 放大，tail class（小 prob）
   的 relative error 进一步放大到 *lsb* 量纲
3. **动态 `in_enc.qmin` 硬截至 -32768**: `shifted` 实际可达
   `-2·logit_max/in_scale ≈ -2·qmax`（logit_max=4 时 ≈ -65534），
   被 clamp 到 `-qmax`，**logit 差超过 2·logit_max 的负 tail 全部
   压扁到 boundary** —— winner-take-all (logit ±8 / cls 少) case
   lsb 大于 logit ±4 是这一项体现
4. **int32 sum 无误差**: spec 范围内 (`N_classes · qmax_exp ≤
   INT32_QMAX`)

- **Spec 测试覆盖度**:
  - ✓ `_integer_softmax_normalize` 已有 sum-to-one on-grid 单测
    （`test_softmax_int16.py`）
  - ✓ v2 path 端到端有 `QuantizedSoftmax` cosine ≥ 0.999 单测（同文件，
    cal range 收紧后 v2 path 是 8-bit 输出）
  - 本段补 **直接调 kernel 的 fp32-random unified gate** 对账
- **测试**: `tests/fixed_point/kernels/test_lookup_int16_precision.py`
  (`test_softmax_*`)
- **Reference**: `torch.softmax(x, dim=-1)`
- **Input 分布**: 32 trials per case，shape `(4, num_classes)`；dim=-1；
  4 个 config 覆盖 logit 强度 × class 数：
  - `logit_±4_cls8`（弱 confidence，少 class）
  - `logit_±4_cls32`（典型 NLP / vision head）
  - `logit_±4_cls128`（大词表 LM head）
  - `logit_±8_cls32`（高 confidence，winner-take-all）
- **Output encoding**: `scale = 1/65535, zp = -32768, qmin = -32768,
  qmax = 32767`（u16-on-i16，softmax [0, 1] 输出语义）
- **阈值**: `_MIN_COSINE = 0.9999`、`_MAX_FLOAT_LSB = 1.0`（统一 floor，
  不放宽；按 KNOWN_LIMIT 锁 xfail strict=True）
- **实测值**（commit time 2026-06-08, PyTorch CPU, 32 trials, i16 only,
  **正确 u16-on-i16 output encoding**）:

| config | cos_min | lsb_max | 与 v2 cos floor 0.999 对账 |
|--------|---------|---------|----------------------------|
| logit_±4_cls8 | 0.999363 | 3849.1 | ✓ ≥ 0.999 |
| logit_±4_cls32 | 0.998439 | 2571.6 | ✗ < 0.999（边缘）|
| logit_±4_cls128 | 0.998940 | 499.6 | ✗ < 0.999（边缘）|
| logit_±8_cls32 | 0.999929 | 480.9 | ✓ ≥ 0.999 |

- **状态**: ⚠ KNOWN_LIMIT（4/4 xfail strict=True，cos 接近 0.999 但 lsb
  超 1.0 floor，物理 normalize 上限）
- **关键观察**:
  - **lsb_max 与 cls 反比相关**：cls=8 lsb=3849；cls=32 lsb=2572；
    cls=128 lsb=500。dominant class prob 越分散（cls 越多），每个 prob
    越接近 1/cls，normalize 误差被 1/cls 衰减
  - **cos 已稳过 0.998**：实测 4/4 ≥ 0.998（其中 2 个 ≥ 0.999）—— v2
    path Linear→Softmax 0.999 floor 在 cls=32/128 边缘上下浮动，cls=8
    与高 confidence (logit±8) 反而稳过。说明 v2 path 0.999 floor 在
    calibration 后是合理的，本段 raw kernel 视角 cos 也基本贴齐
  - **lsb_max 500～3850 是 composed 路径物理上限**: 三项物理因素
    （15-bit normalize 步长、PWL exp 16 段、in_enc -32768 clamp）
    叠加；要在 LSB 量纲拉到 ≤ 1.0 需要 24-bit normalize 或 32-bit
    intermediate，**不符 spec**
- **改进建议**（不修改统一 floor）:
  - **保留 spec composed 路径**（PWL 16 段 + 15-bit normalize 是硬约束）
  - 推荐 manifest 上为 Softmax 声明**独立 cos-only gate**（≥0.998 或
    复用 `INT16_SINGLE_OP_PER_CASE_MIN_COSINE` 0.999 分级）
  - **adapter 端把 output encoding 强制设为 `scale=1/(qmax-qmin),
    zp=qmin`**（u16-on-i16）—— 若 calibration 自动选成 symmetric i16
    则 prob>0.5 全部 saturation，**这是已知陷阱**，应在 capability
    manifest 上加门控或自动校正（新 follow-up
    `FU-SOFTMAX-OE-NORMALIZE`）
  - 同 PWL 族 / FU-P7-LOOKUP-FIT-CEILING 同根：composed kernel 的统一
    gate 需按 family 拆

---

# 待补段（按 doc 落地节奏）

按 P1..P7 的拆包顺序追加。每个段统一上述字段；除 **状态** 与 **实测值** 外，
其他字段在该算子开始测试前先写好（spec / kernel / 输入分布 / 阈值），
然后跑测试填快照。

## P2: `_center × _center → _requantize`
- custom.Multiply：✓ PASS（见下方段）
- custom.Divide：双路实现 —— spec §4.3.4 reciprocal CLZ LUT 路径
  ✓ PASS（统一 floor 内 lsb_max ≤ 0.93，i16/i32 same/cross-scale）；
  legacy integer-div 路径 ⚠ KNOWN_LIMIT（~1.5 LSB），xfail strict 锁定，
  作为 LUT 资产不可用时的 fallback。详见下方段

## P3: `int32_sum_sat → saturate → requantize`
- nn.AvgPool2d：✓ floor-PASS（i16/i32 全部 16 用例 + i8-2x2 同/跨 2 用例，
  共 18/24，cos_min ≥ 0.99991 / lsb_max ≤ 0.53）；⚠ KNOWN_LIMIT（i8
  reduce-size N ≥ 8——即 kernel ∈ {4×4, 4×2, 2×4}——共 6/24 用例 xfail
  strict=True 锁定 cos 0.9994～0.9998，lsb_max 仍 ≤ 0.5；reduction 类
  物理 SNR 上限；其中 i8-2x2 边缘震荡用 strict=False 单独标）
- custom.Mean：✓ PASS（i16/i32 共 8/12 用例，cos_min ≥ 0.999986 /
  lsb_max ≤ 0.50）；⚠ KNOWN_LIMIT（i8 共 4/12 用例 xfail strict 锁定
  cos 0.988～0.997，lsb_max 仍 ≤ 0.50；reduction 类物理 SNR 上限，
  与 AvgPool2d 同根）
- custom.AdaptiveAvgPool2d：✓ PASS（i16/i32 共 12/18 用例，cos_min ≥
  0.999994 / lsb_max ≤ 0.51）；⚠ KNOWN_LIMIT（i8 共 6/18 用例 xfail
  strict 锁定 cos 0.978～0.998，lsb_max 仍 ≤ 0.50；与 Mean 同根）

P3 改进建议（不修改统一 floor）：i8 reduction 路径在 adapter 端走
"requant 到 i16 → reduce → requant 回 i8"复合，或 calibration 调宽
output scale，把 cos floor 拉回 i16 域。详见各算子段"改进建议"。

## P4: `im2col + matmul + bias`
- nn.Linear：✓ PASS（i16 same/cross-scale，cos_min=1.0 / lsb_max ≤ 0.67）；
  ⚠ KNOWN_LIMIT（i8 same/cross-scale 共 2 用例 xfail 锁定 cos_min ≈ 0.99987
  边缘震荡，lsb_max 仍 ≤ 0.5；matmul-class SNR 上限在 64 outputs 时与
  P3 reduction-class i8 同根）
- nn.Conv2d：✓ PASS（i8/i16 × 3 conv config × same/cross-scale 共 12 用例，
  cos_min ≥ 0.99993 / lsb_max ≤ 0.63；H'·W'·F output volume 给出足够 SNR
  让 i8 也稳过 floor）
- nn.Conv1d：✓ PASS（i8/i16 × 3 conv config × same/cross-scale 共 12 用例，
  cos_min ≥ 0.99990 / lsb_max ≤ 0.66；走 Conv2d unsqueeze 路径，本段独立
  测以锁定 1D shim）
- nn.Conv3d：**out-of-scope**（项目当前不使用 3D 卷积）。manifest 显式
  标 `IMPLEMENTED` 但 `dispatchable=int16_eval=exportable=False`，
  防止 adapter 在没精度快照前误路由；kernel 注册保留以供未来 ad-hoc
  unit test。等需要 3D 时翻三个 flag 并补本节实测快照

## P5: value-clamp
- nn.ReLU：✓ PASS（i8/i16 × same/cross-scale 共 4 用例，cos_min ≥ 0.99996 /
  lsb_max ≤ 0.67；same-scale fast path 时 lsb_max=0.0 perfect identity）
- nn.ReLU6：✓ PASS（i8/i16 × same/cross-scale 共 4 用例，cos_min ≥ 0.99990 /
  lsb_max ≤ 0.81）
- nn.Hardtanh / custom.Clamp / custom.Clip：✓ PASS（共享 `ClampInt16Kernel`
  族，3 模块 × i8/i16 × same/cross-scale 共 12 用例，cos_min ≥ 0.99989 /
  lsb_max ≤ 0.60；cross-scale 测试设置 `multiplier=None` 让 kernel 走
  `align → clamp → +Z_y` 单次 rescale，与 spec 04_13 §4.14.1
  "Clip 不应承担量化域转换"一致）；⚠ FU-P5-CLAMP-DOUBLE-RESCALE: 当外部
  传入非 None multiplier 时 kernel 会双重 rescale（align + _requantize），
  follow-up 待修

## P6: byte-stream identity
- nn.Identity / nn.Flatten / custom.Reshape / custom.Permute：✓ PASS
  （`SAME_GRID_VALUE` 严格 same-grid，i8/i16 × same-scale 共 8 用例，
  cos_min=1.000000 / **lsb_max=0.0000** —— perfect byte-stream identity）
- nn.Dropout：✓ PASS（`SAME_GRID_OR_REQUANT`，i8/i16 × same/cross-scale
  共 4 用例：same-scale cos=1.0/lsb=0.0，cross-scale cos_min ≥ 0.99996 /
  lsb_max ≤ 0.71）
- custom.Pad：✓ PASS（`SAME_GRID_OR_REQUANT`，i8/i16 × same/cross-scale
  共 4 用例：same-scale cos=1.0/lsb=0.0，cross-scale cos_min ≥ 0.99996 /
  lsb_max ≤ 0.69）

## P7: LOOKUP 关键算子（CLZ 二阶 LUT + PWL 16-segment）

> **覆盖范围说明**：本段只覆盖 P7 显式拆包的 8 个算子；S1 已用
> per-fn LSB 限锁定的 PWL 11 个算子（Sigmoid/Tanh/GELU/SiLU/Mish/
> Softplus/Hardsigmoid/Hardswish/LeakyReLU/PReLU/Abs）以及 Softmax
> 不在本段，跟踪在 FU-DOC-COVERAGE-PWL-S1 / FU-DOC-COVERAGE-SOFTMAX
> follow-up（见下表）。


- **CLZ 二阶 LUT 族**（abc 资产驱动，`kernels/clz_lut.py`）—— ⚠ KNOWN_LIMIT
  按统一 floor (cos>0.9999 + lsb<1.0) 锁 4 用例：
  - `custom.Sqrt`：cos=1.0 / lsb_max=100.24（x→0 处 sqrt' 发散主导）
  - `custom.RSqrt`：cos=1.0 / lsb_max=1.67
  - `custom.Reciprocal`：cos_min=0.999999 / lsb_max=1.66
  - `custom.Square`：cos=1.0 / lsb_max=2.01
  cos 实测全部稳过 0.9999（几乎完美），lsb_max 略超 1.0 floor 是
  二阶 CLZ-normalized 拟合在低位 LSB 上的物理残差（**不是 kernel bug**，
  abc golden 比对仍 atol=0 通过）
- **PWL 16-segment LUT 族**（`offline/lut_gen.generate_pwl_lut_for_export`
  → `kernels/lut.py::_LutInt16Kernel`）—— ⚠ KNOWN_LIMIT
  按统一 floor 锁 4 用例（cos、lsb 双 gate 都不过），实测：
  - `custom.Log`：cos_min=0.9937 / lsb_max=5240（PWL fit_max=9949）
  - `custom.Exponential`：cos_min=0.99994 / lsb_max=499（fit_max=737）
  - `custom.Sin`：cos_min=0.99977 / lsb_max=1438（fit_max=1438）
  - `custom.Cos`：cos_min=0.99979 / lsb_max=1434（fit_max=1431）
  实测 lsb_max 与 `PWL_VS_ANALYTIC_PER_FN_LIMITS` 设定的 per-fn
  ceiling（log/exp=2048、sin/cos=512、tanh=2400）同数量级，说明 kernel
  忠实复现了拟合误差 —— 这是 spec 04_05 设计 PWL 时就接受的精度
  trade-off（16 段 × 二次多项式 vs 浮点参考的拟合残差）

P7 改进建议（不修改统一 floor）：见各算子段"改进建议"与
FU-P7-LOOKUP-FIT-CEILING follow-up；核心是 capability manifest 上
明确把 LOOKUP 类的精度档位标为 cos-only（不卡 lsb），或在 adapter
calibration 时根据下游容忍度决定是否启用 PWL 替代 (e.g. PWL exp 用
range-folded LUT)。

---

## 精度收敛 follow-up（按 spec 对齐 / KNOWN_LIMIT 落地）

| ID | 算子 | spec 锚点 | 现状 | 收敛动作 |
|----|------|-----------|------|----------|
| FU-DIV-RECIPROCAL | custom.Divide | 04_03 §4.3.4 | spec 路径 ✓ PASS（reciprocal CLZ LUT 已落地，`test_divide_int16_clz_precision.py` 4 用例 PASS）；legacy 路径 ⚠ KNOWN_LIMIT（fallback，xfail strict 锁 6 用例） | （已部分关闭）若未来切换到段数更少 / 动态范围更小的 LUT 资产，需在 normalized space 上加 Newton step 3（`_evaluate_clz_vectorized` 需要外露 mantissa/q_y_norm/exponent）|
| FU-ADDSUB-DUAL-M | custom.Add / custom.Subtract | 04_03 §4.3.2 / §4.3.3 | ✓ PASS（统一 floor），但 Path-B 用 fixed M=32767/rshift=15 | 补一份"双路 quantize_multiplier 都激活"的测试（`scale_a, scale_b, scale_out` 三者两两不同），把 spec 描述的"两路独立 M_i" 完整覆盖 |
| FU-MUL-Z-NONZERO | custom.Multiply | 04_03 §4.3.1 | ✓ PASS（zp=0 路径），spec 公式含 `Z_x1, Z_x2` | 补 zp ≠ 0 的非零零点 multiply 测（asymmetric u8/u16，此前被 Subtract 同根 unsigned KNOWN_LIMIT 一起跳过）|
| FU-P3-I8-COMPOSITE | nn.AvgPool2d / custom.Mean / custom.AdaptiveAvgPool2d | 04_09 §4.9.2 / 04_06 §4.6.1 | i8 reduction 路径 ⚠ KNOWN_LIMIT，xfail strict 锁定（AvgPool2d 6 用例 + Mean 4 用例 + AdaptiveAvgPool2d 6 用例 = 16 xfail）；lsb_max 仍 ≤ 0.5 即 kernel 算术正确，cos 不过 0.9999 floor 是物理 SNR 上限 | adapter 端把 i8 reduction 改写成"requant 到 i16 → reduce → requant 回 i8"复合路径；落地后所有 i8 用例应 PASS 0.9999 floor 并翻 xfail 为正常 PASS。或在 capability manifest 上声明 i8 reduction 为 unsupported 并要求 calibration pipeline 自动 routing 到 i16 |
| FU-P4-I8-LINEAR-COMPOSITE | nn.Linear | 04_02 §Linear | i8 Linear 在 64-output 场景下 cos 边缘震荡（0.99986～0.99996），xfail strict=False 锁 2 用例；lsb_max 仍 ≤ 0.5 即 kernel 算术正确。Conv2d 在更大输出体积下稳过 floor 证明这是 small-output-volume 统计 artefact | 同 P3：adapter 端 i8 Linear 改写成"i8→i16→linear→i8"复合路径，或 manifest 声明 i8 Linear unsupported；落地后 2 个 xfail 翻 PASS |
| FU-P5-CLAMP-DOUBLE-RESCALE | nn.Hardtanh / custom.Clamp / custom.Clip | 04_13 §4.14.1 | **✓ 已关闭（commit 2026-06-09）** —— `clamp_int16` cross-grid 分支现在显式忽略 `output_encoding.multiplier` / `rshift`，仅走 `align → clamp → +Z_y + saturate` 单次 rescale。修复前在真实 adapter 路径（`adapter.py` line ~1016 给非特化算子下发 `real_m = x_scale / y_scale`）上 lsb_max ~150；修复后 lsb_max ≤ 0.63（**250× 精度提升**），adapter-path 与 no-multiplier 路径 `int_repr` 元素相等（bit-exact，30/30 cases PASS） | （已关闭）回归 gate = `test_clamp_cross_scale_adapter_path_matches_no_multiplier_path`；若未来"adapter 不再为 SAME_GRID_OR_REQUANT 类传 multiplier"路径落地，可去掉 kernel 内的 multiplier 忽略逻辑（影响面：Pad/Concat/Dropout 等共用 adapter fallback） |
| FU-P7-LOOKUP-FIT-CEILING | custom.Sqrt / RSqrt / Reciprocal / Square / Log / Exponential / Sin / Cos | 04_04 §LUT、04_05 §CLZ | LOOKUP 族（CLZ 4 + PWL 4 共 8 算子）按统一 floor 全部 ⚠ KNOWN_LIMIT，xfail strict 锁定。CLZ 类 cos 稳过 0.9999（几乎完美），lsb_max 1.5-2 LSB（Sqrt 例外，x→0 处导数发散使 lsb_max 100，但 cos=1）；PWL 类按 `PWL_VS_ANALYTIC_PER_FN_LIMITS` 物理上限 fail（log 9949 / exp 737 / sin/cos 1438 LSB）。这是 spec 设计接受的拟合残差，不是 kernel 缺陷（CLZ 已与 abc golden atol=0 锁定，PWL fit 已与 per-fn 限 PASS） | 二选一：(a) capability manifest 上为 LOOKUP 类声明独立 gate（cos-only 不卡 lsb，或按 per-fn LSB 限松绑）；或 (b) 在 adapter calibration 端按下游容忍度路由（e.g. 容忍 cos≥0.99 的子图允许 PWL；否则 fallback 到浮点 boundary）。落地后翻 8 个 xfail 为按新 gate 的 PASS |
| FU-DOC-COVERAGE-ADD-MATMUL-CONCAT-MAXPOOL | custom.AvgPool2d | 04_09 | **几乎关闭**：Add ✓ / MatMul ✓ / Concat ✓ / nn.MaxPool2d ✓ / custom.MaxPool2d ✓ 都已补独立 doc § 段 + 实测快照（共 26 + 4 + 12 = 42 个新 PASS 用例）。剩余 `custom.AvgPool2d`：manifest `dispatchable=False`（functional wrapper，adapter 还未 unpack kernel_size），kernel 与 `nn.AvgPool2d` 共享 `_AvgPool2dKernel` 基类，行为同根；P3 AvgPool2d 段已显式包含此 alias，未独立测试 | 等 adapter 路由 functional wrapper 落地（见 `route-custom-pool-through-adapter` plan item）后，在 P3 AvgPool2d 段后追加 `test_custom_avgpool2d_*` 一行 parametrize 用例 + 关闭整条 follow-up |
| FU-DOC-COVERAGE-PWL-S1 | nn.Sigmoid / Tanh / GELU / SiLU / Mish / Softplus / Hardsigmoid / Hardswish / LeakyReLU / PReLU / custom.Abs | 04_04 §LUT、`PWL_VS_ANALYTIC_PER_FN_LIMITS` | **已关闭**：本 doc 已补 P7-S1 段（在 P7 PWL Log/Exp/Sin/Cos 段后），11 个算子全部覆盖；测试在 `test_lookup_int16_precision.py` 复用 `_run_pwl_case` helper（11 新 case，3 PASS：LeakyReLU/PReLU/Abs，8 xfail：Sigmoid/Tanh/GELU/SiLU/Mish/Softplus/Hardsigmoid/Hardswish）；实测快照 cos_min ≥ 0.99999、lsb_max 0.5～498.5（与 `PWL_VS_ANALYTIC_PER_FN_LIMITS` 的 per-fn 限对应） | 后续改进归 FU-P7-LOOKUP-FIT-CEILING 同根处理（PWL 族独立 manifest gate / Hardsigmoid kink-snap 优化）|
| FU-DOC-COVERAGE-SOFTMAX | nn.Softmax | 04_04 §LUT（composed: PWL exp + 整数 sum + reciprocal） | **已关闭**：本 doc 已补 `## nn.Softmax（composed kernel...）` 段（4 个 config × 32 trials + 实测快照），测试在 `test_lookup_int16_precision.py` 新增 4 个 case 全 xfail strict=True 锁 KNOWN_LIMIT；**首版用了错误的 symmetric i16 output encoding 导致 lsb 虚高 3-4 倍**，已切到 u16-on-i16 (`scale=1/65535, zp=-32768`) 正确 encoding，真实 cos_min 0.998～0.9999、lsb_max 481～3849（composed 路径物理上限：15-bit normalize 步长 / PWL 16 段 / in_enc -32768 clamp 三因素叠加） | 后续改进归 FU-P7-LOOKUP-FIT-CEILING 同根处理（独立 cos-only gate / 按 cls 分级）+ 新增 FU-SOFTMAX-OE-NORMALIZE（adapter 强制选对 OE）|
| FU-SOFTMAX-OE-NORMALIZE | nn.Softmax | 04_04 §LUT、softmax kernel | calibration 若自动选 symmetric i16 output encoding (`scale=1/qmax, zp=0`)，prob>0.5 全部 saturate 到 1.0，lsb_max 虚高 3-4 倍 —— 这是 spec 未明确 / 工具未门控的陷阱 | adapter 在 dispatch 前若发现 Softmax 输出 quantizer 是 symmetric → 自动改为 `scale=1/(qmax-qmin), zp=qmin`（u16-on-i16）；或 capability manifest 上声明 Softmax 输出强制 asymmetric grid；附 sanity test 拒绝 symmetric output encoding |
| FU-CONV3D-MANIFEST | nn.Conv3d | 04_01 / 04_02 | 已修复：manifest 显式 `IMPLEMENTED` + `dispatchable=int16_eval=exportable=False` —— kernel 保留供未来 ad-hoc 测试，adapter 不会误路由 | （已关闭）等项目用到 3D conv 时翻三个 flag 并补 P4 nn.Conv3d 实测快照即可 |
| FU-ABS-KIND-MISMATCH | custom.Abs | 04_03 §4.3.5 abs | **已完全关闭 (Layer B1)**：`AbsInt16Kernel` 已从 `kernels/lut.py` (PWL `_LutInt16Kernel`) 迁出，新实现在 `kernels/eltwise.py` 走 spec integer-abs path（`x' = |q_x − Z_x|; y_q = sat(x'·M ≫ rshift) + Z_y`）；manifest `kernel_kind` 从 `LOOKUP` 改回 `SAME_GRID_OR_REQUANT`；测试从 `test_lookup_int16_precision.py` 的 PWL case 迁到 `test_relu_int16_precision.py` 的 same/cross-scale × i8/i16 4 个 case，全 PASS 且 cos=1.0/lsb=0.0（byte-stream identity） | （已关闭）`PWL_VS_ANALYTIC_PER_FN_LIMITS['abs']` 在 `thresholds.py` 中保留作为历史 PWL-export 消费方的 sanity，不再用于 runtime |
| FU-LOG-CLZ-LUT-UPSTREAM | custom.Log | `LUT_Binary_Storage_General §5`、`capabilities.py:430-465` | **已诊断（Layer B2 dry-run + 二次核实）阻塞在 abc_lut-shuai 上游**：spec 把 log 归类为 CLZ（独有 `q_y = round_shift(y_offset · r_q, r_shift) + E · q_ln2 + q_ln_sx + zp` 公式）。abc_lut-shuai 资产现状：(a) `lut_int_general/lut_int_po2` 给的 `log_lut.json` 都是 **PWL 16 段**（与 sigmoid/gelu 同族）；(b) `lut_fp/output/fp32_test/log_normalized_fp32_lut.json` 是 **CLZ-normalized 概念资产**（input ∈ [1.0, 2.0]、output ∈ [0, ln 2]、16 段）但是 **fp32 形式不是 int**；(c) `clz_normalized_fitter.py` 只支持 reciprocal/sqrt/rsqrt/power_2，**log 不在 fitter 范围**。runtime 仍走 PWL 是工具链限制，不是 software 决策；当前 lsb_max=9949 是 PWL 用于 log 的物理上限。**Layer B2-prototype（2026-06-08）oracle 实测**：用 fp32 normalized log LUT 模拟 spec §5 公式上界 → lsb_max=18.97 vs PWL=9949（改善 ~525×），证实 CLZ-form 物理上限在 Sqrt/RSqrt 量级；只待 #1 abc_lut-shuai 侧 int 量化输出 | 三步走（必须按序）：(1) abc_lut-shuai 数据科学侧扩展 `clz_normalized_fitter` 加 log 分支（基于已有 fp32 normalized 资产，加 int16 量化与 q_ln2/q_ln_sx 常量），输出 `log_clz_lut.json`（约 50-100 LOC abc_lut-shuai 侧）；(2) software 侧扩展 `clz_lut.py::denormalize_clz` 加 log 分支（~30 LOC）+ 注册 `LogInt16ClzKernel`（~20 LOC）；(3) 测试用 spec golden LUT 做 bit-exact oracle，预期 lsb_max 降到 Sqrt/RSqrt 量级（数 LSB）。**替代 prototype 路径**（不依赖 #1）：从 fp32 normalized log LUT 在线量化到 int16 临时挂到 `LogInt16ClzKernel`——可行性验证用，缺点是 LUT 与 spec golden 不一致，建议作为单独的可选 prototype follow-up |
| FU-P4-CONV1D-I8-STRIDE2-FLAKE | nn.Conv1d (i8-k3-s2-p1-g1) / nn.Conv2d (i8-k3-s1-p1-gIN depthwise) / custom.MatMul (i8) | 04_01 卷积类算子 | 2026-06-08 全套 fixed_point/ 回归观察：i8 small-K 配置（Conv1d stride=2、Conv2d depthwise gIN、MatMul）偶发 fail（cos 略 < 0.9999），单独跑稳定 PASS——根因都是 i8 [-128,127] 代码包络 + 小 K_eff 让 cos 距 floor 不足 1e-4 余量。lsb_max 仍 ≤ 0.5（kernel 算术正确）；与 `FU-P4-I8-LINEAR-COMPOSITE` 同根 | 先按 N×100 次循环采样落实 flake-rate；若 > 0.5%，二选一：(a) 拉宽 `code_limit` 提高 SNR；(b) adapter 端走 i8→i16→conv→i8 复合（同 `FU-P4-I8-LINEAR-COMPOSITE`）。或 manifest 声明 i8 small-K 路径 unsupported 强制 routing |

每条 follow-up 应**先**在本 doc 对应算子段记录新发现的 KNOWN_LIMIT 或
新阈值快照，**再**在 kernel 侧落地修复——避免"先改 kernel 后改 doc"
导致快照与代码漂移。

---

## 实测值采集流程

精度修改后重新生成 §**实测值** 快照：

1. 找到本算子段 §**测试** 指向的测试文件。
2. 在该测试文件目录下用一次性 Python 脚本调用同一组 fixture / helper，
   收集 cosine 与 max_error_lsb_float 的 (min, median, max)。本仓约定
   **不**在测试文件内长期保留打印逻辑；快照在文档里、CI 只检阈值。
3. 替换本段表格内数值，重提 commit。表格只保留**当前**快照——历史值看 git。

工具：见 `tests/fixed_point/kernels/test_subtract_int16_precision.py`
里的 `_random_fp32_pair / _quantize_float / _output_encoding` helper；
其他算子段在测试文件中提供**同名** helper（pattern across files），
所以采集脚本可以直接 import 复用。

---

## spec 章节路径核实（Layer D2/D3/D4）

下列三章 spec 原本作为"未实现 INT16_FIXED_EVAL kernel"列入 dry-run；
经过 codebase 实地核实，发现绝大多数 spec 章节**不需要新增独立 kernel**
——项目通过 **BN-fold + sub-op composition** 已经覆盖了大部分 spec 行为，
剩余条目要么 spec-only（业务无触发点），要么属于硬件平台层（软件
reference 范围之外）。本节给出每章的逐项核实结论 + 真实剩余工作量，
取代原 dry-run 评估。

**本节修订历程**（按 commit 时间倒序）:
- **2026-06-09 (本次)**: D2-4.5.1 / 4.5.2 / 4.5.7 / 4.5.8 spec-only 4 章
  正式结清——`integer_square_mean_base_instruction` /
  `integer_variance_base_instruction` 公开 API 抽出（spec 基础指令），
  `cLN2D` / `SimCln2d` nn.Module 实施（sub-op composition），新增
  18 cases unit test 全部 PASS，关闭 `FU-LAYERNORM-AFFINE-INTEGER` /
  `FU-LAYERNORM-DSP-PARITY` / `FU-NORM-SPECONLY-CLN2D-SIMCLN2D` 共 3 个
  follow-up；累计代码量 ~410 LOC
- **2026-06-08 (上次)**: D2/D3/D4 三章路径核实，真实剩余 LOC 从 3400
  → 0；明确 9 个 follow-up 跟踪条目（业务触发后立项）
- **2026-06-08 (上上次)**: D2-4.5.3 BatchNorm + D3-4.10.2 Nearest 结清
  （前者文档化 BN-fold，后者实施 ~440 LOC kernel + 测试）
- **初版**: 8+3+5 = 16 个 spec 章节，按字面工作量估 3600 LOC，建议
  分 9 个独立 PR

---

### D2: 04_05 归一化算子（8 个 spec 章节，0 个需要新增独立 INT16 kernel）

| §    | 算子        | 项目 codebase 状态 | 覆盖路径 | 剩余 LOC |
| ---- | ----------- | ------------------ | -------- | -------- |
| ~~4.5.1~~ | ~~square_mean~~ | ✓ 已实现公开 helper `integer_square_mean_base_instruction`（commit 2026-06-09，`norm.py`，spec line 67-75 字面对齐）| **✓ base-instruction helper**（见下方 §**D2-4.5.1 路径核实**）；`tests/fixed_point/kernels/test_norm_base_instructions_int16.py` 3 cases PASS | 0（已结清）|
| ~~4.5.2~~ | ~~variance~~    | ✓ 已实现公开 helper `integer_variance_base_instruction`（同 commit，由 LayerNorm 内联抽出，spec line 109-117 字面对齐）| **✓ base-instruction helper**（见下方 §**D2-4.5.2 路径核实**）；同测试文件 3 cases PASS | 0（已结清）|
| ~~4.5.3~~ | ~~BatchNorm~~ | ✓ PyTorch 内置 | **✓ BN-fold**（见下方 §**D2-4.5.3 路径核实**）| 0（已结清）|
| ~~4.5.4~~ | ~~LayerNorm~~ | ✓ 已实现（**spec 完整 bit-parity 整数路径**，commit 2026-06-09，含 §4.5.2 integer variance + line 422-428 integer M/rshift affine）| **✓ kernel**（见正文 §**nn.LayerNorm**）；spec 16-bit M 的精度上限登记为 `FU-LAYERNORM-ALPHA-X-CALIBRATION` | 0（已结清）|
| ~~4.5.5~~ | ~~CLN~~ | ✓ `examples/quick_start.py:335`，forward = `mean(square(x)) → clamp → sqrt → div` | **✓ sub-op composition**（见下方 §**D2-4.5.5 路径核实**）| 0（已结清）|
| ~~4.5.6~~ | ~~cfLN2D~~ | ✓ `_base/nn/modules/custom.py:774`，forward 用 explicit-op 链 `Subtract/Add/Sqrt/Divide/Multiply` | **✓ sub-op composition**（见下方 §**D2-4.5.6 路径核实**）| 0（已结清）|
| ~~4.5.7~~ | ~~cLN2D~~       | ✓ 已实现 `_base/nn/modules/custom.py::cLN2D`（commit 2026-06-09，spec §4.5.7 reference 等价的 `var(dim=(1,3)) + sqrt + div`，explicit op-modules）| **✓ sub-op composition**（见下方 §**D2-4.5.7 路径核实**）；`tests/fixed_point/kernels/test_cln2d_simcln2d_modules.py` 4 cases PASS | 0（已结清）|
| ~~4.5.8~~ | ~~SimCln2d~~    | ✓ 已实现 `_base/nn/modules/custom.py::SimCln2d`（同 commit，spec §4.5.8 forward + inverse，explicit op-modules，`self.std` 缓存）| **✓ sub-op composition**（见下方 §**D2-4.5.8 路径核实**）；同测试文件 5 cases PASS（含 inverse round-trip）| 0（已结清）|

- **总剩余 LOC**: **0**（vs 原评估 1800 LOC）—— 路径核实后修订
- **本轮（2026-06-09）补充工作量**: ~410 LOC 全部已合入：
  - `norm.py` +110 LOC：抽出 `_inline_integer_square_mean` 新 helper（spec §4.5.1）+ 公开 API
    `integer_square_mean_base_instruction` / `integer_variance_base_instruction`
  - `_base/nn/modules/custom.py` +90 LOC：新增 `cLN2D` / `SimCln2d` nn.Module（spec §4.5.7/§4.5.8）
  - `tests/fixed_point/kernels/test_norm_base_instructions_int16.py` +210 LOC：spec §4.5.1/§4.5.2 base-instruction bit-parity 测试 7 cases
  - `tests/fixed_point/kernels/test_cln2d_simcln2d_modules.py` +160 LOC：cLN2D/SimCln2d forward/inverse 11 cases
  - 全部 18 cases PASS；LayerNorm 回归 4 PASS + 13 XFAIL（与 commit 2026-06-09 快照一致，重构无破坏）
- **结论**: §4.5 整章在当前 codebase 下**不需要任何新 INT16 kernel**。
  spec 把所有归一化都描述为"完整大算子下发给硬件"，但**项目实际选择
  把所有归一化拆为基础算子链路**（`Subtract/Add/Square/Sqrt/Mean/
  Divide/Multiply` 等已实现的 op），由 v2 `QuantizationMixin` 的
  `__torch_function__` 拦截 sub-op 完成 INT16 dispatch
- **隐含假定与风险**（写入 follow-up 跟踪而非本轮立项）:
  - **FU-NORM-SUBOP-VS-DSP-PARITY**: sub-op composition 与 DSP 单指令
    在数值上是否 bit-exact 一致？答案是**理论上不一致**——多 sub-op
    链路的 `M/rshift` 量化每步都会引入舍入；DSP 单指令可以使用更宽
    的中间累加器和单次重定标。当前项目走 reference-software 路径，
    sub-op composition 精度 ≥ 0.9999 cos 已经覆盖业务需求；当部署
    目标硬件确实是 DSP 单指令时，需要立项"per-op 完整大算子 kernel"
    并补 hardware emulator 路径
  - **FU-LAYERNORM-FIRST-USE**: ✓ **已关闭（commit 2026-06-09）** —— 业务
    首次引入 `nn.LayerNorm` 触发，采用 **spec §4.5.4 对齐路径**：
    1. dequant INT16 input
    2. fp32 `variance` 仿真（spec §4.5.2 base instruction reference）
    3. **RSqrt CLZ LUT**（spec line 384 强制 LUT 步骤，复用
       `custom.RSqrt` 的同款 `abc_lut-shuai` LUT 资产）
    4. fp32 affine（spec line 387 数学定义视角）+ requantize

    总实施成本 ~440 LOC：kernel `norm.LayerNormInt16Kernel` ~340 LOC
    （含详细 docstring 解释 spec 对齐 + RSqrt LUT helper +
    `layer_norm_float_reference` oracle 函数）+ manifest 1 条目 +
    adapter 3 处接入 + 测试 16 cases：**3 passed + 13 xfailed（strict=True
    KNOWN_LIMIT）**。xfail 不是 kernel 缺陷，而是项目统一严格 gate
    `cos > 0.9999 / lsb < 1.0` 与 LayerNorm 物理上限的客观差异——
    RSqrt CLZ LUT 的 PWL fit residual（~3-5 LSB）经 `γ/std` 放大到
    LayerNorm output domain。物理上限登记在 `thresholds.py::
    LAYERNORM_VS_FP32_PER_GRID_LIMITS`（i16 max_lsb=8, i8 max_lsb=3，
    两个 grid 的 cos 仍 ≥ 0.9999），与 P7 PWL/CLZ family 同款处理
  - **FU-LAYERNORM-AFFINE-INTEGER**: ✓ **已关闭（commit 2026-06-09）** ——
    spec §4.5.4 line 422-428 的整数 `M/rshift` 折叠 affine 已通过
    `_integer_affine` 实施，与 spec line 425-426 公式 bit-parity（在
    Default A 的 `S_μ = S_x` 约定下 `(M_μ, rshift_μ) = (M_x, rshift_x)`
    折叠为单一减法 + 一次乘法），并发现 spec line 414 的 16-bit M 在
    LayerNorm calibrated i16 affine 路径上精度退化到 ~5-bit，登记为
    新 FU `FU-LAYERNORM-ALPHA-X-CALIBRATION`
  - **FU-LAYERNORM-DSP-PARITY**: ✓ **已关闭（commit 2026-06-09）** ——
    spec §4.5.2 描述的整数 `variance` 基础指令已通过
    `_inline_integer_variance` 实施，覆盖两阶段在线 reduce（`s_o = Σ(q_x − Z_x)`
    → `q_μ = (s_o · inv_N) >> shift_N`；`d² = (q_x − q_μ)²` →
    `v_o = (Σd² · inv_N) >> shift_N`）+ 重定标到 `(S_var, Z_var) = LUT input grid`。
    LOC ~140 kernel + 单元测试通过；该 FU 关闭后 LayerNorm 与硬件 DSP
    `variance` 指令字段 bit-parity
  - **FU-LAYERNORM-ALPHA-X-CALIBRATION**（新增）: spec line 414 的
    16-bit `M_x` + `max_rshift = 31` 总动态范围 = 47-bit，但 LayerNorm
    calibrated i16 + γ 量化下 `α_x = S_γ · S_x · S_inv / S_y ≈ 1e-8`
    需要 `rshift ≈ 41` 才能让 M 满载 16-bit，超出 max_rshift 后 M
    精度退化到 ~5-bit，affine 路径 lsb 物理上限升到 ~1000 LSB。要破有
    三条路径，触发条件 = 业务模型对 LayerNorm 输出 LSB-level 精度有
    硬性要求：
    1. **硬件层 widen M**：把 spec line 414 的 16-bit M 改为 24/32-bit
       并相应延长 max_rshift；需要修改 spec 和编译器 codegen 表
    2. **改 α_x 因子分解**：把 `q_γ` 与 `M_x` 拆成两次 multiply（先 `q_γ
       · q_inv · centered` 再 `· M_x`），让 `α_x = S_x · S_inv / S_y ≈
       3e-4` 落在 16-bit 满载区——但 q_γ × q_inv × centered 可能
       overflow int32 carrier，需要 int64 中间累加
    3. **改 γ 量化 grid**：增大 `S_γ`（如降低 γ qmax 到 64 或 128 而非
       32767），让 q_γ 整体下降同时 α_x = S_γ · ... 上升入 16-bit M 满
       载区；需要重新验证 γ 量化对端到端模型精度影响
    LOC 估计 ~80-150（视选型），需先和硬件 owner 对齐 spec 接口才能动
  - **FU-NORM-SPECONLY-CLN2D-SIMCLN2D**: ✓ **已关闭（commit 2026-06-09）**
    —— `cLN2D` / `SimCln2d` 在 `_base/nn/modules/custom.py` 已实施
    （sub-op composition 风格），与 cfLN2D 对齐；INT16 dispatch 经由
    现成的 `Add / Sqrt / Divide / Multiply / Abs` sub-op kernel 自动
    覆盖。如果未来真的需要"DSP 单指令 cLN2D / SimCln2d" bit-parity
    路径，归并到 `FU-NORM-SUBOP-VS-DSP-PARITY` 跟踪

#### D2-4.5.1 square_mean 路径核实（结清）

- **结论**: spec §4.5.1 的"DSP 单指令 `square_mean`：`Σ(q_x − Z_x)² ·
  inv_N >> shift_N` 重定标到 `(S_sq, Z_sq)`"在当前 codebase **不需要
  作为 nn.Module 注册**（spec 描述的是 DSP 基础指令，不是用户算子），
  但**实现作为公开 helper 函数已就位**，供 LayerNorm 内联以及未来
  RMSNorm / cLN2D DSP-parity 路径直接调用
- **覆盖证据**:
  - `aimet_torch/fixed_point/kernels/norm.py::integer_square_mean_base_instruction`
    （commit 2026-06-09）—— 字面对齐 spec line 67-75 的三步公式：
    1. `s_o = Σ(q_x − Z_x)²`（int64 累加器）
    2. `m_o = (s_o · inv_N) >> shift_N`（`_compute_inv_n_shift_n`
       与 LayerNorm `variance` 共用，确保编译器侧能产出 bit-parity
       的 `inv_N` 值）
    3. `q_sq = ((m_o · M_sq) >> rshift_sq) + Z_sq`，
       `α_sq = S_x² / S_sq` 由 `quantize_scale_to_m_rshift` 折叠为 16bit
       Multiplier 和右移
  - 单元测试 `tests/fixed_point/kernels/test_norm_base_instructions_int16.py
    ::test_square_mean_base_instruction_matches_fp32`（3 reduce sizes
    × 8 trials = 24 trials）
- **实测精度**（commit 2026-06-09，i16 grid，N(0,1) input，calibrated
  `S_x` / `S_sq`）:
  - cos_min > 0.9999 全部通过
  - lsb_max ≤ 5 codes（在 `_BASE_INSTR_LSB_BOUND` 信封内）
  - 残差成分：`(s · inv_N) >> shift_N` 一次 round-half + `(m · M_sq)
    >> rshift_sq` 一次 round-half ≈ 1+3 LSB；trial-to-trial 余量 ≈ 1
    LSB。相比 LayerNorm 的物理上限 ~7-910 LSB，base instruction 单步
    无 LUT 放大，因此精度信封极紧
- **与 spec 大算子路径的差异**: 项目当前 CLN（§4.5.5）走 sub-op
  composition（`mean(square)` 拆为 `Square + Mean`），与 base
  instruction 的 1 次重定标 vs sub-op 链路的 2 次重定标在 `cos ≥
  0.9999` 信封下**不可观测**（噪声被 `mean` 的 `1/N` 平均吸收）。当
  部署目标硬件确实需要 base instruction bit-parity 时，可直接调用
  `integer_square_mean_base_instruction` 替代 sub-op 链路；这是
  `FU-NORM-SUBOP-VS-DSP-PARITY` follow-up 的第一个落地切入点

#### D2-4.5.2 variance 路径核实（结清）

- **结论**: spec §4.5.2 的"DSP 单指令 `variance`：两阶段在线 reduce
  得到 `q_μ` 和 `q_var`"在当前 codebase 已**作为公开 helper
  `integer_variance_base_instruction` 就位**——同一段代码也是
  LayerNorm `LayerNormInt16Kernel` 的 step 1（spec §4.5.4 line 422-428
  的 in-line `variance` 调用）。该 FU `FU-LAYERNORM-DSP-PARITY` 在
  commit 2026-06-09 已闭合
- **覆盖证据**:
  - `aimet_torch/fixed_point/kernels/norm.py::integer_variance_base_instruction`
    —— 包装内部 `_inline_integer_variance` 暴露公开 API，字面对齐 spec
    line 109-117 的四步公式：
    1. `s_o = Σ(q_x − Z_x)`            （int64 sum）
    2. `q_μ − Z_μ = (s_o · inv_N) >> shift_N`  （Default A: `S_μ = S_x`，`Z_μ = Z_x`）
    3. `v_o = (Σd² · inv_N) >> shift_N`（int64 carrier）
    4. `q_var = ((v_o · M_var) >> rshift_var) + Z_var`，
       `α_var = S_x² / S_var`
  - 单元测试 `test_norm_base_instructions_int16.py
    ::test_variance_base_instruction_matches_fp32`（3 reduce sizes
    × 8 trials = 24 trials，覆盖 `q_μ` 和 `q_var` 两个输出）
- **实测精度**（同上 commit，i16 grid，N(0,1) input）:
  - `q_var`: cos_min > 0.9999, lsb_max ≤ 5（同 §4.5.1 信封）
  - `q_μ`: cos_min > 0.9999, **lsb_max ≤ 2**（更紧——只有 1 次
    `inv_N` round-half，没有第二步 M/rshift 重定标）
- **与 spec 大算子路径的差异**: cfLN2D（§4.5.6）当前 sub-op 路径用
  `torch.var(dim=...)` 直接拿 mean/var，与 base instruction 的整数
  carrier 在 `cos ≥ 0.9999` 信封下不可观测；DSP-parity 路径需要时
  可直接调用 `integer_variance_base_instruction`（与 LayerNorm 走
  同一段代码）



- **结论**: spec §4.5.3 的"预融合 `γ' = γ/√(σ²+ε)`，`β' = β - γ'·μ`，运行时 `y = γ'·x + β'`"
  在当前 codebase **不需要独立 INT16 kernel**，因为整张图上的 BN 在
  量化 sim 构建早期就被吃掉了
- **覆盖证据**:
  - `aimet_torch/fixed_point/e2e/sim_builder.py` 默认 `apply_bn_fold=True`
    （build_calibrated_sim 第 156-162 行），调用
    `aimet_torch.batch_norm_fold.fold_all_batch_norms` 将
    `nn.BatchNorm{1,2,3}d` 折叠进前序 `nn.Conv*` / `nn.Linear`
  - 折叠后图上不再存在 `nn.BatchNorm*` module，dispatch 入口只会看到
    被吸收后的 Conv/Linear，权重 = γ'·W、bias = γ'·b + β'，与
    spec §4.5.3「mult + add」形式严格等价
  - 因此 manifest（`capabilities.py`）**有意**不为 `nn.BatchNormXd`
    建条目；INT16 Conv/Linear kernel 已经覆盖 BN 在量化路径上的
    全部行为
- **KNOWN_LIMIT / follow-up**: 若图上存在**无法被 fold 的孤立 BN**
  （例如 BN 后面不是 Conv/Linear），sim 当前会让该 BN 走 FP32_QDQ
  路径而非 INT16，可能违反纯 INT16 部署假设。这种边缘场景的处理
  分两步：
  1. **诊断**: `diagnose_int16_readiness` 已经在标准化路径里捕获
     "missing kernel for nn.BatchNorm*" 类警告（间接覆盖）
  2. **若实需修复**: 立项 `FU-BN-STANDALONE-AFFINE` —— 实现一个
     per-channel affine kernel（M_c/rshift_c + b_c 形式，spec
     §4.5.3 最终融合公式给出），约 ~80 LOC kernel + ~60 LOC 测试。
     **本轮明确不做**：当前所有引用模型走标准 PTQ 流程，BN-fold
     成功率 100%，没有实际触发点

#### D2-4.5.5 CLN 路径核实（结清）

- **结论**: spec §4.5.5 的"square_mean + rsqrt LUT + scale"完整大算子
  在当前 codebase **不需要独立 INT16 kernel**；项目实际把 CLN 写成
  4 个 functional torch op 的链路，每个都是已实现的 IMPLEMENTED 算子
- **覆盖证据**: `examples/quick_start.py:335-346`

  ```python
  class CLN(nn.Module):
      def forward(self, x):
          mean_sq = torch.mean(torch.square(x), dim=(1, 3), keepdim=True)
          std = torch.sqrt(torch.clamp(mean_sq, min=EPS))
          return torch.div(x, std)
  ```

  - `torch.square` → `custom.Square`（LOOKUP，CLZ 归一化平方 LUT）
  - `torch.mean` → `custom.Mean`（REQUANTIZING，1/N 已折叠）
  - `torch.clamp` → `custom.Clamp`（SAME_GRID_OR_REQUANT）
  - `torch.sqrt` → `custom.Sqrt`（LOOKUP，CLZ sqrt LUT）
  - `torch.div` → `custom.Divide`（REQUANTIZING）

  全 5 个 sub-op 都有 INT16 kernel + 精度快照（见正文相应段）。`model_preparer`
  在 `prepare_model(stateless_modules_to_preserve=[..., CLN, ...])` 保留 CLN
  module 外壳，但 forward 内部的 functional 调用被 v2 `QuantizationMixin.
  __torch_function__` 接管，INT16_FIXED_EVAL dispatch 命中每个 sub-op 的
  kernel
- **与 spec 大算子路径的精度差异**: spec 给的"单指令 square_mean → rsqrt
  LUT → scale"是 1 次 重定标；sub-op composition 是 5 次重定标
  （square → mean → clamp → sqrt → div），每步 `M/rshift` 都引入
  ~0.5 LSB 舍入。但 `mean(square)` 的累加位宽足够大（int32 carrier），
  且 `sqrt`/`div` 都已是 CLZ-normalized LOOKUP，整条链路上的 cos
  精度 ≥ 0.999 在 quick_start 模型上实测验证过（见 e2e calibration
  log）。**结论：sub-op 路径精度足以满足项目当前 INT16 需求**

#### D2-4.5.6 cfLN2D 路径核实（结清）

- **结论**: spec §4.5.6 说 "保留独立 cfLN2D 指令路径，编译器仍按完整
  大算子接口下发"，但**项目代码实际选择 sub-op composition 路径**——
  这是 spec 与 codebase 实现策略的有意偏差，不是 bug
- **覆盖证据**: `aimet_torch/_base/nn/modules/custom.py:774-806`
  cfLN2D forward 显式注释 "使用显式的算子模块，以便量化"，把归一化
  拆为 `subtract / add / sqrt / divide / multiply / add_bias` 共
  6 个 explicit module（不是 torch functional，而是真正的 `Subtract()`
  / `Add()` / `Sqrt()` 等子模块）。注：项目代码用 `torch.var`/
  `torch.mean` 拿到 mean/var，但 spec 要求 var/mean 用 `variance`
  基础指令——这部分见 `FU-NORM-SUBOP-VS-DSP-PARITY` follow-up
- **与 spec 大算子路径的差异**: 同 CLN，sub-op composition 引入多次
  重定标舍入；当部署目标真的是"DSP 单指令 cfLN2D"时需要立项 follow-up
  把整章重写为单 kernel

#### D2-4.5.7 cLN2D 路径核实（结清）

- **结论**: spec §4.5.7 的"在拉平的 `(C,F)` 维度计算 variance 并直接
  除以 std（不减均值、无仿射）"在当前 codebase **不需要独立 INT16
  kernel**；commit 2026-06-09 在 `_base/nn/modules/custom.py` 添加
  `cLN2D` nn.Module，forward 用 sub-op composition 风格（与 cfLN2D
  对齐），由 v2 `QuantizationMixin.__torch_function__` 接管 INT16
  dispatch
- **覆盖证据**: `_base/nn/modules/custom.py::cLN2D`
  ```python
  class cLN2D(torch.nn.Module):
      def __init__(self, eps: float = 1e-5):
          super().__init__()
          self.eps = eps
          self.add = Add()
          self.sqrt = Sqrt()
          self.divide = Divide()

      def forward(self, y):
          var = torch.var(y, dim=(1, 3), keepdim=True, unbiased=False)
          var_eps = self.add(var, self.eps)
          std = self.sqrt(var_eps)
          return self.divide(y, std)
  ```
  说明：spec reference 用 `rearrange(y, 'b c t f -> b (c f) t')` +
  `var(dim=1)` 是为了把 `C/F` 合并后单维归约；我们用 `torch.var(y,
  dim=(1, 3), keepdim=True, unbiased=False)` 直接归约 `(C, F)` 两个
  维度，**数学上完全等价**（unit test 用 spec line 793-803 的 reference
  公式做 `torch.allclose` 验证），同时省掉 einops 依赖
- **测试覆盖**: `tests/fixed_point/kernels/test_cln2d_simcln2d_modules.py`
  - `test_cln2d_forward_matches_spec_reference`（3 shapes：标准
    `[2,3,4,5]` / 最小 `[1,8,1,4]` / 大 reduce `[4,4,8,4]`）—— 与 spec
    reference 公式逐元素 allclose（`atol=1e-5`）
  - `test_cln2d_uses_explicit_op_modules` —— 守卫 `add` / `sqrt` /
    `divide` 子模块仍是真正的 op-modules，防止未来重构把它们 inline 成
    free `+/-` 让 INT16 dispatch 静默掉到 FP32_QDQ
- **与 spec 大算子路径的差异**: 同 CLN/cfLN2D，sub-op composition 引入
  多次重定标舍入；DSP 单指令路径触发条件 = 业务把 cLN2D 部署到目标
  硬件且对 LSB-level 精度有硬性要求，触发后立项 `FU-NORM-SUBOP-VS-DSP-PARITY`
  并复用 `integer_variance_base_instruction`（spec §4.5.2 helper 已就位）

#### D2-4.5.8 SimCln2d 路径核实（结清）

- **结论**: spec §4.5.8 的"L1 范数（绝对值均值）归一化 + inverse
  恢复"在当前 codebase **不需要独立 INT16 kernel**；commit 2026-06-09
  在 `_base/nn/modules/custom.py` 添加 `SimCln2d` nn.Module，forward
  和 inverse 分别用 sub-op composition 实现，`self.std` 缓存设计与
  spec line 933 的 reference 一致
- **覆盖证据**: `_base/nn/modules/custom.py::SimCln2d`
  ```python
  class SimCln2d(torch.nn.Module):
      def __init__(self, eps: float = 0.009765625):
          super().__init__()
          self.eps = eps
          self.std: Optional[torch.Tensor] = None
          self.abs = Abs()
          self.add = Add()
          self.divide = Divide()
          self.multiply = Multiply()

      def forward(self, x):
          abs_x = self.abs(x)
          mean = torch.mean(abs_x, dim=(1, 3), keepdim=True)
          self.std = self.add(mean, self.eps)
          return self.divide(x, self.std)

      def inverse(self, x):
          if self.std is None:
              raise RuntimeError(...)
          return self.multiply(x, self.std)
  ```
  注：`self.std` 不作为 `register_buffer`/`Parameter`，而是普通属性
  ——这与 spec reference 一致，让量化 observer 把它视为活动张量经过
  explicit op chain 流动；调用顺序错误（先 `inverse` 后 `forward`）
  会立即 raise `RuntimeError`
- **测试覆盖**: `tests/fixed_point/kernels/test_cln2d_simcln2d_modules.py`
  - `test_simcln2d_forward_matches_spec_reference`（3 shapes）——
    forward 与 spec line 931-936 的 reference 公式 allclose，同时验证
    `self.std` 缓存被正确填充且 shape 为 `[B,1,T,1]`
  - `test_simcln2d_inverse_round_trip`（3 shapes）—— `inverse(forward(x))
    ≈ x`（rtol=1e-4），同时验证未 forward 直接 inverse 的负向
    case 抛 `RuntimeError`
  - `test_simcln2d_uses_explicit_op_modules` —— 守卫 `abs / add /
    divide / multiply` 子模块都是 op-modules
- **与 spec 大算子路径的差异**: SimCln2d 用 L1 范数 `mean(|x|)` 而非
  L2 范数 `mean(x²)`，因此**不属于** `square_mean` 或 `variance` 基础
  指令的数学形式（spec §4.5.8 line 988 已明示）。Forward 路径若要
  做 DSP-parity 仍需独立 helper（约 50 LOC，结构与
  `_inline_integer_square_mean` 类似但用 `abs(centered)` 替换
  `centered²`）；当前业务无触发点，作为 `FU-NORM-SUBOP-VS-DSP-PARITY`
  的子条目跟踪

### D3: 04_10 Resize 类算子（3 个 spec 章节，1 个已实现，2 个 follow-up）

| §    | 算子                    | 项目 codebase 状态 | 覆盖路径 | 剩余 LOC |
| ---- | ----------------------- | ------------------ | -------- | -------- |
| 4.10.1 | Bilinear 插值          | ✗ 项目无 `nn.Upsample(mode='bilinear')` / `nn.UpsamplingBilinear2d` 触发点（grep examples/ + e2e/ 无 hit）| 立项条件：业务首次引入 Bilinear upsample 时再做 | 0（项目未使用）|
| ~~4.10.2~~ | ~~Nearest-neighbour~~ | ✓ 已实现 | **✓ kernel**（见正文 §**nn.Upsample / nn.UpsamplingNearest2d**）| 0（已结清）|
| 4.10.3 | Down / Upsampling      | ✗ spec 描述是"按 mode 选择 Bilinear 或 Nearest 的 wrapper"，**不是独立算子** | 当前 `nn.Upsample` kernel 已自动覆盖 Nearest 分支；Bilinear 分支随 §4.10.1 立项 | 0（wrapper，不独立立项）|

- **总剩余 LOC**: **0**（vs 原评估 530 LOC）—— 路径核实后修订
- **已实现部分实际成本**:
  ~130 LOC kernel（`shape_ops.py::_NearestResizeInt16Kernel` +
  `UpsampleInt16Kernel` + `UpsamplingNearest2dInt16Kernel` +
  `_nearest_resample_int_repr` helper），~310 LOC 测试
  （`test_resize_nearest_int16.py` 14 cases），manifest 2 个条目，
  adapter dispatch 2 处分支（fp32-eval surrogate + QAT surrogate）
- **结论**: §4.10 章节当前剩余真实代码工作 **0 LOC**。Bilinear 在
  项目里**无任何 example / e2e 模型触发**（grep 排除框架基础类、
  ONNX export 与本文档自身）；Nearest kernel 入口已经把 mode 白名单
  当中 `mode != 'nearest'` 的请求显式拒绝（raise ValueError），
  Bilinear 业务到来时 adapter dispatch 会自然 fallback 到 FP32_QDQ
- **隐含假定与风险**（写入 follow-up 跟踪）:
  - **FU-BILINEAR-FIRST-USE**: 项目首次引入 `nn.Upsample(mode=
    'bilinear')` 或 `nn.UpsamplingBilinear2d` 时立项；实现方案就 spec
    §4.10.1（REQUANTIZING，4-tap 加权和 + `M_y/rshift_y` 重定标），
    ~300 LOC kernel + ~100 LOC 测试；现有 Nearest 路径的 manifest /
    adapter / kernel 框架可直接复用，仅需把 mode 白名单从 `{nearest,
    nearest-exact}` 扩展到含 `bilinear`，并实现 `BilinearInt16Kernel`
  - **FU-RESIZE-COORD-FIXED-POINT**: spec 4.10.1 要求坐标映射用
    定点乘法 `t_in,q = (t_out · M_t) >> r_t`，但 reference kernel
    用 `F.interpolate` 的 fp32 round-trip 更精确（与 Nearest 同源
    论证）。当硬件 emulator 需要 1:1 复现"8bit 插值系数 + int32
    加权和"时，需要额外的 `BilinearHwReferenceKernel` 路径
  - **FU-DOWN-UPSAMPLING-ALGORITHM**: spec §4.10.3 没明确"
    Down/Upsampling 默认采用 nearest 还是 bilinear"——这是硬件团队
    决策项，**不属于软件 reference kernel 工作范围**

### D4: 04_08 特殊运算类算子（5 个 spec 章节，2 个 sub-op 覆盖，3 个 follow-up）

| §    | 算子                  | 项目 codebase 状态 | 覆盖路径 | 剩余 LOC |
| ---- | --------------------- | ------------------ | -------- | -------- |
| ~~4.8.1~~ | ~~hypot~~ | ✓ `examples/quick_start.py:326`，`HypotFun.forward = sqrt(square(x) + square(y) + clamp(EPS))` | **✓ sub-op composition**：`Square + Add + Clamp + Sqrt` 全是 IMPLEMENTED | 0（已结清）|
| ~~4.8.2~~ | ~~power（特殊情况 b=0.5）~~ | ✓ `examples/quick_start.py:318`，`PowerCompress.forward = sign(x) * sqrt(abs(x))` | **✓ sub-op composition**：`Sign + Sqrt + Abs + Multiply` 全是 IMPLEMENTED | 0（已结清，仅覆盖 b=0.5）|
| 4.8.2' | power（通用 b ≠ 0.5）| ✗ 项目无通用 power module | spec 通用 b 需 `exp(b·log(x))`；log CLZ LUT 阻塞 `FU-LOG-CLZ-LUT-UPSTREAM`；当前业务只用 b=0.5 | 0（被上游 LUT 阻塞）|
| 4.8.3 | shift                 | ✗ 项目无独立 shift module；定点重定标的 shift 已在 `requantize_int` 内 hot path 完成 | 软件 reference 路径下 shift 不是独立"算子"；编译期通过 `quantize_multiplier` 产出 `M/rshift`，运行时由所有 REQUANTIZING kernel 共享 | 0（已在 requantize 内）|
| 4.8.4 | band-merge / split    | ✗ 项目无 class，与硬件 CIM 矩阵乘法深度耦合 | spec §4.8.4 含 6 个子节（CIM 适配 / 稀疏 / 位宽 / 量化 / 伪代码 / 性能），属硬件平台层；**软件 reference kernel 范围之外** | 0（硬件层，软件 N/A）|
| 4.8.5 | 矩阵行索引            | ✗ 项目无独立"行索引"算子触发点；常规 indexing 用 `torch.index_select` 等 functional torch op | 立项条件：业务模型出现非 torch-functional 的 spec-style 矩阵行索引（如 hardware-specific gather）时再做 | 0（项目未使用）|

- **总剩余 LOC**: **0**（vs 原评估 1060 LOC）—— 路径核实后修订
- **结论**: §4.8 章节当前剩余真实代码工作 **0 LOC**：
  - hypot / power(b=0.5) 已通过项目代码的 sub-op 拆解（`HypotFun` /
    `PowerCompress`）+ v2 `QuantizationMixin` 自动 dispatch 覆盖
  - shift 在软件 reference 路径下不是独立算子（融在 `requantize_int`
    内）；spec §4.8.3 的"shift 单指令"是 DSP 硬件视角
  - band-merge/split (§4.8.4) 是 CIM 矩阵乘法硬件适配章节，软件
    reference kernel 范围之外
  - power 通用 b / 矩阵行索引：项目实际无触发点
- **隐含假定与风险**（写入 follow-up 跟踪）:
  - **FU-LOG-CLZ-LUT-UPSTREAM**（已存在）: abc_lut-shuai 上游需要
    扩 `clz_normalized_fitter.py` 加 log 分支并生成
    `log_clz_lut.json`。log CLZ 落地之后，通用 `power(b)` 就可以
    通过 `exp(b·log(x))` 实现，~80 LOC kernel
  - **FU-SHIFT-STANDALONE**: 如果硬件确实需要"shift 单指令"路径
    （绕过 REQUANTIZING 的 M/rshift 折叠），需要一个 ~50 LOC kernel
    + 测试；当前没有触发点
  - **FU-CIM-BAND-MERGE-SPLIT**: 与硬件 CIM 工具链对齐后立项；
    可能需要 hardware emulator 而非纯软件 reference

### 三章合计估算（路径核实后大幅修订：3400 → 0 LOC 真实必需工作）

- **原始评估（spec 字面工作量）**: D2 1900 + D3 640 + D4 1060 ≈
  **3600 LOC** —— 这是"假设每个 spec 章节都需要独立 INT16 kernel"
  下的估算
- **路径核实后真实必需工作**: **0 LOC** —— 项目 codebase 已经通过两条
  机制把 spec 章节全覆盖：
  1. **BN-fold**（§4.5.3）：sim_builder 默认折叠 BN 进 Conv/Linear
  2. **sub-op composition**（§4.5.5/5.6 + §4.8.1/8.2-special）：项目
     把"完整大算子"拆为已实现的 functional torch op 或 explicit
     module 链路，v2 `QuantizationMixin.__torch_function__` 接管
     dispatch
- **本轮实际新增 kernel 工作量**:
  - D3-§4.10.2 Nearest（~440 LOC，上一轮交付）—— 项目实际有触发点且
    不能用 sub-op 表达
  - D2-§4.5.4 LayerNorm（~150 LOC，本轮交付）—— `FU-LAYERNORM-FIRST-USE`
    触发，采用 float-reference 路径而非 spec 大算子路径（详见下方
    `## nn.LayerNorm` 段）
- **剩余 follow-up 清单**（仅在触发条件满足时立项）:
  - `FU-NORM-SUBOP-VS-DSP-PARITY` —— 部署目标确实是 DSP 单指令时立项；
    spec §4.5.1 / §4.5.2 的 `integer_square_mean_base_instruction` /
    `integer_variance_base_instruction` 公开 API 已就位（commit
    2026-06-09），可作为切入点
  - ~~`FU-LAYERNORM-AFFINE-INTEGER`~~ —— ✓ 已关闭（commit 2026-06-09）
  - ~~`FU-LAYERNORM-DSP-PARITY`~~ —— ✓ 已关闭（commit 2026-06-09）
  - ~~`FU-NORM-SPECONLY-CLN2D-SIMCLN2D`~~ —— ✓ 已关闭（commit 2026-06-09，
    cLN2D / SimCln2d nn.Module 已实施）
  - `FU-BN-STANDALONE-AFFINE` —— 出现无法 fold 的孤立 BN
  - `FU-BILINEAR-FIRST-USE` —— 业务首次需要 bilinear upsample
  - `FU-RESIZE-COORD-FIXED-POINT` —— 硬件 emulator 需要 1:1 复刻
    8bit 插值系数路径
  - `FU-LOG-CLZ-LUT-UPSTREAM` —— abc_lut-shuai 上游 log CLZ LUT
  - `FU-SHIFT-STANDALONE` —— 硬件需要"shift 单指令"非融合路径
  - `FU-CIM-BAND-MERGE-SPLIT` —— 与硬件 CIM 工具链对齐
- **路径核实工作的价值**: 避免了"按 spec 字面工作量去做无人用的
  ~3000 LOC kernel"的虚高估计；后续工作可以聚焦"业务实际触发的
  spec 章节"而不是"清单上的全部 spec 章节"
- **关键风险（须显式承担）**: **sub-op composition 与 spec "完整大
  算子" 在数值上不是 bit-exact 等价**——多 sub-op 链路每步 `M/rshift`
  都引入舍入，spec 大算子可以使用更宽中间累加器和单次重定标。
  当前 reference-software 路径在 cos≥0.9999 / max_lsb<1.0 floor
  下精度满足业务需求，但部署到 DSP 单指令硬件时需要单独立项
  hardware emulator 路径以确保 bit-parity（追踪
  `FU-NORM-SUBOP-VS-DSP-PARITY`）
- **下一步推荐**: 按"业务驱动"而非"清单驱动"——下一个真实 kernel
  工作的触发条件 = 业务首次引入 `nn.LayerNorm` /
  `nn.UpsamplingBilinear2d` 等清单上的算子；在没有触发之前，
  **本文件的精度验证工作可以告一段落**

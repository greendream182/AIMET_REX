# MRNN 单算子验收与 LUT/CLZ 设计对照

Status: 与 `int16_single_op_vs_float_native.py` / `int16_unary_op_diagnosis.py` 同步（2026-06）

配套文档：

- [LUT_Binary_Storage_General.md](./LUT_Binary_Storage_General.md) — PE general-scale LUT head / 数据区
- `abc_lut-shuai/doc/lut_int_design_general.md` — Int-LUT 算法与 §3.0 scale 适配

---

## 1. 三种验收口径（勿混用）

| 口径 | 参考 | 候选 | 用途 |
|------|------|------|------|
| **A. Kernel** | 同模块 `fp32_qdq` 输出 | `int16_fixed_eval` / LUT 路径 | quality report / pytest |
| **B. QDQ 插入损失** | `float op(Q_in(x))` | `fp32_qdq` / `fixed_scale_qdq` 模块输出 | 边界 QDQ 是否合理 |
| **C. 总量化损失** | `float op(x)`（prepared 浮点图） | 各 ExecutionMode | 整图 vs `float_native` 上界 |

当前 MRNN 脚本默认测 **C**；LUT 文档 §6 测的是 **孤立 LUT + 可控输入范围** 的 MAE/cos，更接近 **A/B 的子集**，不能直接套用到 C。

---

## 2. MRNN 分解节点 → LUT §2.3 规格化类

CLN/BN 分解后由 **规格化类 primitive 链** 组成（非另一套「层归一化」分类）：

```text
x → power_2 → mean → sqrt → reciprocal(1/std) → mul → y
      §2.3          (归约)  §2.3      §2.3
```

| MRNN 模块名模式 | LUT §2.3 规格化类 | 部署形态 | 单算子主指标 |
|-----------------|-------------------|----------|--------------|
| `module_square*` | `power_2` | unary CLZ | cos(B) + saturation_rate |
| `module_sqrt*` | `sqrt` / `rsqrt` | unary CLZ | cos(B) ≥ 0.999 |
| `module_div*`（BN/CLN 除 std） | **`reciprocal(b)`** | `a/b = a × reciprocal(std)`；AIMET 图为双输入 div | cos(B) + zero_denom_rate |
| `module_sign*` | **不在 §2.3** | 无 LUT head | sign_agreement(Q_in) |
| `module_mul/add/sub` | 饱和类整数 op | — | cos(B) ≥ 0.999 |
| `conv*` / `conv_t` | 有参 Conv | — | cos(B) ≥ 0.999 |

**Encoding 修复**（`examples/common/mrnn_clz_encoding.py`）：

| 节点 | 修复 | 说明 |
|------|------|------|
| sign | bypass input Q | §3.1；非 LUT |
| div 分母 | `fix_reciprocal_denom_encodings` | unsigned + min≥eps，禁止 Q(b)→0 |
| square | `fix_power2_output_encodings` | out_max ≈ max(\|x_dequant\|)² |
| 诊断 bypass | `--div-denom-input-bypass` / `--square-output-bypass` | 仅隔离 kernel，非生产路径 |

---

## 3. 已观测问题与设计项对照

### 3.1 `module_sign`（PowerCompress 前端）

**现象**（calib batch0，`power_compress_1.module_sign`）：

```text
input scale ≈ 0.25 (8-bit symmetric)
x:    neg≈50%, zero≈0.2%, pos≈50%
Q(x): neg≈2.7%, zero≈94.5%, pos≈2.7%
sign(x) vs sign(Q(x)) 一致率 ≈ 5.6%
sign(Q(x)) vs QuantizedSign 输出 cos ≈ 0.999992
```

**根因**：\(|x| < scale/2\) 被 round 到 0，符号信息在 **input Q** 阶段丢失；非 sign 公式错误。

**设计动作**（不在 LUT head 内）：

1. sign 输入 **finer scale** 或 **bypass input Q**（仅 sign 前驱）。
2. 验收改用 **sign_agreement(Q_in)**，阈值建议 ≥ 0.99；禁止用 cos vs float_native 判 FAIL。
3. Po2 后 scale=0.25 与 `|x|∈[-24,19]` 不匹配，需 mixed-precision 或 per-op encoding 审查。

### 3.2 `module_sqrt` / `hypot_fun.module_sqrt_*`

**现象**：

| 模块 | cos(C) vs float | cos(B) vs float(sqrt(Q_in)) |
|------|-----------------|------------------------------|
| `power_compress_1.module_sqrt` | 0.919 | **0.99988** |
| `hypot_fun.module_sqrt_1` | 0.979 | **0.99995** |

**对照 lut_int_design_general §6.2**：孤立 `sqrt` LUT MAE≈5e-5，cos≈1.0（输入 ∈ [0,8]）。

**设计动作**：

1. MRNN 最终 INT16 路径应对齐 **CLZ head**（`q_norm/n_norm`, `r_q/r_shift`, `output_min/out_max`），见 [LUT_Binary_Storage_General.md §5](./LUT_Binary_Storage_General.md#5-clz-类-lut-head)。
2. 单算子验收报告须同时输出 **cos_total** 与 **cos_on_qinput**；后者对齐 LUT 文档精度预期。
3. 输入须 **x_offset > 0** 再进 CLZ；`hypot` 已用 clamp(min=EPS)，与 §3.3「负值/零值特殊路径」一致。

### 3.3 `module_square_*`（CLN / BN 分解）

**现象**：`enc_seqs.0.cln.module_square_2` — float max≈375.9，quant max≈**127**，max_abs≈249，cos(C)≈0.967。

**对照 binary head**：槽位 3/4 为 `output_min` / `out_max`；当前 8-bit symmetric 无法覆盖 \(x^2\) 动态范围。

**设计动作**（power_2 / CLZ head `out_max`）：

1. 用 `collect_power2_float_out_fmax()` 在校准 batch 上观测 **float x² fmax**（勿仅用 input quantizer peak²）。
2. `fix_power2_output_encodings(..., float_out_fmax=...)` 写 16-bit unsigned out_max。
3. 报告 **saturation_rate**；仍 FAIL 时查 Po2 是否把 out_max 再次压窄。

### 3.4 `module_div_*`（BN / CLN → reciprocal 分母）

**语义**：`a/b` 中 **b = std > 0**；LUT 侧为 **`reciprocal(b)`**（§2.3），完整式 `a × reciprocal(b)`。

**现象**：

| 模块 | cos(C) | 备注 |
|------|--------|------|
| `pre_bn.module_div` | 0.994 | 分母 b 无大量零 |
| `enc_seqs.0.rnn2d_bn.module_div_1` | 0.145 | `b_q` 约 11.7% 为 0 |
| 部分 div | cos=**nan** | ref 近零向量，非 div 输出 NaN |

**设计动作**（reciprocal 正域，§3.4 CLZ + §5.1）：

1. 分母 encoding：**unsigned + dequant_min ≥ EPS**，禁止 Q(b)→0。
2. 实现：`fix_reciprocal_denom_encodings()`（Po2 后）；勿长期用 `--div-denom-input-bypass`。
3. cos=nan → 标 **DEGEN**；teacher-force **(a,b)** 两路。

---

## 4. 推荐验收阈值（MRNN teacher-forced）

| 类别 | 主指标 | 阈值 | 备注 |
|------|--------|------|------|
| Conv / Linear | cos(B) | ≥ 0.999 | vs float on Q_in |
| mul / add / sub / clamp | cos(B) | ≥ 0.999 | |
| sign | `sign_agreement_output` + `sign_agreement`(Q_in) | output≥0.99；input≥0.99 否则 encoding FAIL |
| sqrt / rsqrt | cos(B) | ≥ 0.999 | CLZ 路径 |
| square / power_2 | cos(B) + saturation_rate | cos≥0.99, sat≤1% | `fix_power2_output_encodings` |
| reciprocal / div | cos(B) + zero_denom_rate | cos≥0.99, zero_denom=0 | `fix_reciprocal_denom_encodings` |
| 任意 | ref_norm | > 1e-6 | 否则 DEGEN |

**cos(C) vs float_native** 仅作 informational，不作为 kernel/LUT PASS 门。

---

## 5. 脚本与报告

```bash
cd aimet_rx-main/examples
PYTHONPATH=../..:../../quant-gru-pytorch/pytorch:$PYTHONPATH \
  python3 int16_single_op_vs_float_native.py \
  --clz-encoding-fix \
  --output output/int16_single_op_clz_encoding_fix_v2.json

PYTHONPATH=../..:../../quant-gru-pytorch/pytorch:$PYTHONPATH \
  python3 int16_unary_op_diagnosis.py

PYTHONPATH=../..:../../quant-gru-pytorch/pytorch:$PYTHONPATH \
  python3 int16_single_op_input_ablation.py \
  --output output/int16_single_op_input_ablation.json
```

报告字段（完善后）：

- `cos_total` — 口径 C
- `cos_on_qinput` — 口径 B（与 LUT §6 可比）
- `sign_agreement` / `saturation_rate` / `zero_denom_rate`
- `status`: PASS | FAIL | DEGEN | ERROR | SKIP

---

## 7. 整图 path B（``int16_whole_graph_vs_float_native.py``）

基准：**未包装 MRNN** test Top-1（与单算子脚本 `float_native_top1` 同口径）。

```bash
PYTHONPATH=../..:../../quant-gru-pytorch/pytorch:$PYTHONPATH \
  python3 int16_whole_graph_vs_float_native.py \
  --max-calib-batches 100 --output output/int16_whole_graph_vs_float_native_clz.json
```

**2026-06 实测**（``model_fp.pth`` 95.56%，PTQ skip QAT，100 calib batch）：

整图脚本默认 **`--no-apply-po2`**（Ada200 主线：校准 → ``convert_encodings_to_fixed_scale`` → ``fixed_scale_qdq``）；旧数含 Po2 时用 ``--apply-po2`` 复现。

| 配置 | fp32_qdq | fp16_qdq | fixed_scale_qdq |
|------|----------|----------|-----------------|
| 无 CLZ fix | 15.20% | 16.78% | 1.50% |
| **`--clz-encoding-fix` + `--apply-po2`（旧）** | **57.34%** | **57.05%** | 1.50% |
| **`--clz-encoding-fix`（默认无 Po2）** | **67.43%** | **67.74%** | 1.50% |

无 Po2 相对 Po2+CLZ：**+10 pp**（fp32_qdq）；与 Design v2「校准 → M,rshift」主线一致。报告：``output/int16_whole_graph_no_po2_clz.json``。

**QAT 扫参**（``int16_whole_graph_qat_sweep.py``，无 Po2，``clz_post_qat``，100 calib）：

| case | sign bypass | lr | epoch | fp32_qdq | QAT loss | 备注 |
|------|-------------|-----|-------|----------|----------|------|
| PTQ 基线（skip QAT） | 开 | — | — | 67.43% | — | |
| ``no_sign_bypass_lr1e4_e1`` | **关** | 1e-4 | 1 | **67.81%** | 3.77 | 最佳，+0.4 pp（QAT 未真收敛，checkpoint 回滚） |
| ``no_sign_bypass_lr1e4_e1_no_ckpt`` | 关 | 1e-4 | 1 | **4.04%** | 3.70 | 无 checkpoint → 崩盘 |
| 其余 lr/epoch/b200 | 关/开 | 1e-5~1e-4 | 1~3 | 67.33~67.71% | ~3.5 | loss 不降，val≈3.7% |

结论：当前 path B 上 **QAT 超参/关 sign-bypass 均无法像旧 85% ckpt 那样拉到 ~88%**；val checkpoint 必须保留。

**QAT 1 epoch（同配置，`--no-skip-qat`）**：

| 阶段 | fp32_qdq | 说明 |
|------|----------|------|
| PTQ + CLZ（QAT 前） | 57.34% | 与上表一致 |
| QAT 1 epoch 后（**旧顺序**：CLZ post-calib → QAT） | **4.04%** | train loss ~3.68 几乎不下降；**QAT 与手工 reciprocal/power_2 encoding 冲突** |
| 诊断 | — | ``convert_encodings`` 前后均为 ~4%；根因在 QAT 本身，非 convert |

报告：``examples/output/int16_whole_graph_qat1_clz.json``

**缓解（已实现）**：

1. ``qat_finetune``：显式 ``fp32_qdq`` + **val Top-1 checkpoint**（不低于 QAT 前 PTQ）。
2. ``--clz-post-qat``：reciprocal/power_2 改在 **QAT 之后** 再 ``apply_mrnn_clz_encoding_fixes_post_calib``。

```bash
# 推荐 QAT 命令
python3 int16_whole_graph_vs_float_native.py --max-calib-batches 100 \
  --no-skip-qat --qat-epochs 1 --clz-post-qat \
  --output output/int16_whole_graph_qat1_clz_post.json
```

结论：

1. §2.3 encoding fix 把整图 fp32_qdq 从 ~15% 拉到 ~**57%**（+42 pp），与单算子 v2 全 PASS 一致方向。
2. **旧顺序 QAT 会崩溃**；需 ``--clz-post-qat`` 或 val checkpoint 后再评估。
3. 距 float_native 95.56% 仍差 ~**38 pp** → QAT 收敛 + 串联误差 / GRU carrier 继续 ablation。
4. ``fixed_scale_qdq`` 整图仍 ~1.5%（与 INT16 carrier / convert 路径相关，非单点 encoding）。

报告：``examples/output/int16_whole_graph_vs_float_native_clz.json``

---

MRNN 经 `apply_power_of_2_workflow` 后 scale 变为 \(2^{-n}\)，general LUT head 中 `q_* / n_*` 应可由 \((M, rshift)\) 导出（见 binary doc §1）。**验收顺序建议**：

1. Po2 + calib 后导出各 op encodings → 核对 CLZ head 槽位 0–10。
2. 跑单算子 **cos_on_qinput** 门禁。
3. 再跑整图 QAT / INT16 ablation。

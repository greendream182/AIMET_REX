# AIMET RX 定点量化技术实现说明

| 字段 | 内容 |
|------|------|
| 适用项目 | `aimet_rx` |
| 目标读者 | 模型量化、编译器/Runtime 对接、硬件对拍、测试与维护人员 |
| 推荐配合阅读 | `FixedPoint_Quantization_Design_v2.md`、`FixedPoint_Quantization_Spec/`、`INTERFACE.md` |
| 关注重点 | Ada200 定点 scale、G2/G3 双路径、硬件最大位宽对齐、sidecar 导出、软件验收 |

---

## 1. 一句话概括

本项目在 AIMET 的 QuantSim 体系上增加一套面向 Ada200 NPU 的定点量化扩展：校准和 QAT 继续复用 AIMET，部署所需的 scale 统一表示为 `(M_int16, rshift)`，同时提供一条可选的整数 kernel 仿真路径，用于在主机上尽量复现硬件整数计算、requantize、饱和和 LUT 行为。

最重要的设计边界有三条：

1. `fixed_scale_qdq` 是主路径：Q/DQ 边界使用 `(M, rshift)`，模块内部仍是 PyTorch float op，适合 PTQ/QAT、导出和与 fp32 baseline 对比。
2. `int16_fixed_eval` 是硬件整数仿真路径：张量以整数载体在层间传递，Conv/Linear/Add/Pool/LUT 等走 fixed-point kernel，适合硅前整数语义对拍。
3. 两条路径数值不等价，不能用 `fixed_scale_qdq` 通过来替代 `int16_fixed_eval` 或板端 sign-off。

---

## 2. 为什么要做这套扩展

标准 AIMET QuantSim 的核心能力是“模拟量化误差”：在 quantizer 边界做 Q/DQ，算子内部仍执行浮点计算。这对训练、校准和一般精度评估很有用，但还不能完整回答 Ada200 部署中的几个问题：

- Ada200 侧希望每个 quantizer 的 scale 能用硬件可消费的 `(M_int16, rshift)` 表达，而不是只保留 float scale。
- 部署整数图中，Conv/Linear 的累加、层间 requantize、Add/Concat 的输入对齐、AvgPool 的平均缩放、非线性 LUT 都是整数语义。
- 硬件实现存在明确位宽边界：累加器、乘后结果、输出值域都可能饱和，而不是 Python/PyTorch 默认的无限精度数学。
- 导出不仅要有 ONNX 和 AIMET encodings，还需要 sidecar 把层间 multiplier、rshift、bias_int32、PWL/CLZ LUT 等部署参数带出去。

因此项目拆成两个层级：

- **G2：定点 scale QDQ**，解决 scale 表示和导出问题。
- **G3：硬件整数仿真**，解决整数算子、requantize、饱和和 LUT 对拍问题。

---

## 3. 核心术语

| 术语 | 含义 |
|------|------|
| `scale` | quantizer 步长，AIMET 校准后原本是 float |
| `M_int16, rshift` | 定点 scale 表示，`scale ≈ M / 2^rshift` |
| `qmin, qmax` | 量化整数网格范围，由 bitwidth 和 signed/unsigned 决定 |
| `zero_point` | 非对称量化零点 |
| `multiplier, rshift` | G3 层间 requant 参数，近似 `s_x * s_w / s_y` 或 `s_in / s_out` |
| `FixedScaleEncoding` | G2 的定点 scale encoding，字段包括 `m_int16`、`rshift`、`zero_point`、`qmin/qmax` |
| `OutputEncoding` | G3 kernel 的输出 encoding，除 scale/zp/qmin/qmax 外，还含层间 `multiplier` / `rshift` |
| `FixedPointSimTensor` | G3 段间整数仿真载体，历史兼容名为 `Int16QuantizedTensor` |
| `SIM_TENSOR_DTYPE` | G3 载体 `int_repr` 的容器 dtype，目前统一为 `torch.int32` |
| sidecar | `*.int16.json`，部署辅助 JSON，记录整数 kernel 所需的额外参数 |

一个容易误解的点是：`Int16QuantizedTensor` 这个名字不表示所有权重/激活语义上都是 16 bit。现在的设计中，容器 dtype 是 `torch.int32`，语义值域由 `qmin/qmax` 表达，例如 U8、S8、U16、S16 都可以用同一个载体承载。

---

## 4. 总体架构

```text
用户模型
  |
  | prepare_model / BN fold / CLE / AdaRound / calibration
  v
AIMET QuantizationSimModel
  |
  +-- fp32_qdq / fp16_qdq
  |     标准 QDQ，校准和 QAT 主入口
  |
  +-- fixed_scale_qdq (G2)
  |     Q/DQ scale 改用 (M_int16, rshift)，算子仍为 Op_float
  |
  +-- int16_fixed_eval / int16_fixed_qat_sim (G3)
        TrueQuant.forward -> dispatch_int16_fixed
          -> boundary quantize
          -> fixed kernel registry
          -> integer op / requantize / LUT
          -> FixedPointSimTensor
```

代码分层如下：

| 层级 | 代表文件 | 职责 |
|------|----------|------|
| 模式控制 | `aimet_torch/fixed_point/execution_mode.py` | 定义 `ExecutionMode`，支持 API 和环境变量切换 |
| G2 scale | `fixed_scale_qdq.py`、`offline/scale_fixed.py` | float scale 到 `(M,r)`，并在 Q/DQ 中使用 |
| G3 载体 | `tensor.py`、`encoding.py` | 定义整数仿真载体和 encoding 数据结构 |
| G3 分派 | `v2/quantization/affine/fixed_point/adapter.py` | 在 `int16_fixed_*` 下把 v2 Quantized module 分派到整数 kernel |
| kernel registry | `registry.py`、`kernels/*` | 注册 Conv/Linear/Add/Pool/LUT/Softmax 等整数 kernel |
| requantize | `requantize.py`、`rounding.py` | 整数缩放、舍入、饱和、硬件位宽对齐 |
| 离线 freeze | `offline/pipeline.py`、`multiplier.py`、`bias.py`、`lut_gen.py` | 生成部署所需 multiplier/rshift/bias/LUT/sidecar |
| 导出 | `export/sidecar.py`、`export/v2_collect.py`、`encoding_export.py` | 生成和加载 `*.int16.json` |
| 验证 | `metrics/*`、`scripts/fixed_point/*`、`tests/fixed_point/*` | 多模式对比、质量报告、CI 和 golden 测试 |

---

## 5. ExecutionMode 如何工作

| Mode | 主要用途 | 计算语义 |
|------|----------|----------|
| `fp32_qdq` | 默认、校准、PTQ baseline | Q/DQ 后 float op |
| `fp16_qdq` | FP16 实验 | Q/DQ 后 FP16 op |
| `fixed_scale_qdq` | G2 主路径、导出门禁 | Q/DQ 使用 `(M,r)`，op 仍 float |
| `int16_fixed_eval` | G3 硬件整数仿真 | 段间整数 tensor + fixed kernel |
| `int16_fixed_qat_sim` | 可选整数约束 QAT | 前向整数语义，反向 STE |

模式通过两种方式生效：

```python
from aimet_torch.fixed_point import ExecutionMode, set_quant_execution_mode, quant_execution_mode

set_quant_execution_mode(ExecutionMode.FIXED_SCALE_QDQ)

with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
    y = sim.model(x)
```

或进程启动前设置：

```bash
export AIMET_RX_QUANT_EXECUTION_MODE=int16_fixed_eval
```

代码入口主要有两个：

- `torch_builtins.quantize_dequantize(...)`：当 mode 是 `fixed_scale_qdq` 时，改走 `quantize_dequantize_from_float_encoding`。
- `TrueQuant.forward(...)`：当 mode 是 `int16_fixed_eval` / `int16_fixed_qat_sim` 时，优先调用 `dispatch_int16_fixed`。

---

## 6. G2：fixed_scale_qdq 的实现过程

### 6.1 目标

G2 的目标是让每个 quantizer 的 scale 不再只依赖 float，而能以硬件部署认可的 `(M_int16, rshift)` 作为权威表示：

\[
scale \approx \frac{M}{2^{rshift}}
\]

其中 `M` 是 int16 系数，`rshift` 是非负右移位数。Ada200 不要求 scale 必须是 Power-of-2，Po2 只是 `(M,r)` 的一个子集。

### 6.2 生成 `(M, rshift)`

校准完成后，每个 AIMET `AffineEncoding` 已有 float `scale`。G2 通过 `offline/scale_fixed.py` 做转换：

```text
convert_encodings_to_fixed_scale(sim)
  -> 遍历 input/output/param quantizers
  -> quantizer.get_encodings()
  -> get_or_create_fixed_scale_encoding(encoding)
  -> quantize_scale_to_m_rshift(scale)
  -> 缓存在 encoding._aimet_rx_fixed_scale_encoding
```

这一步不会改变默认 `fp32_qdq` 行为。缓存的意义是：后续 `fixed_scale_qdq` 和 G3 边界量化可以复用同一套 `(M,r)`。

### 6.3 Q/DQ 运行公式

G2 的 Q/DQ 位于 quantizer 边界：

```text
q = clamp(round(x * 2^rshift / M) - offset, qmin, qmax)
x_dequant = (q + offset) * M / 2^rshift
```

注意这里的 `offset` 来自 AIMET v2 affine encoding，等价于 `-zero_point`。代码中会小心保留 `offset` 的精度，避免 bf16/fp16 下大 zero point 被舍入。

### 6.4 G2 的边界

G2 只改 Q/DQ 的 scale 表示，不把 Conv/Linear 变成整数 MAC：

```text
fp input
  -> Q/DQ with (M,r)
  -> PyTorch float Conv/Linear/ReLU/...
  -> Q/DQ with (M,r)
  -> fp output
```

因此 G2 更接近 AIMET 传统 QuantSim，适合做 PTQ/QAT 与导出门禁，但不能证明整数硬件路径 bit-exact。

---

## 7. G3：硬件整数仿真的实现过程

### 7.1 目标

G3 的目标是在主机上执行“像硬件一样”的整数路径：

```text
fp input
  -> boundary Q
  -> FixedPointSimTensor
  -> fixed kernel: int MAC / int add / int pool / int LUT
  -> requantize
  -> FixedPointSimTensor
  -> final DQ for debug/metrics
```

### 7.2 G3 dispatch 主链路

在 `int16_fixed_eval` 下，v2 TrueQuant module 的 `forward` 会走：

```text
TrueQuant.forward
  -> dispatch_int16_fixed(qmodule, *args, **kwargs)
     -> resolve base_cls
     -> get_fixed_kernel(base_cls)
     -> input: quantize_boundary_from_affine 或复用上游 FixedPointSimTensor
     -> weight: quantize_boundary_from_affine
     -> derive output real_multiplier
     -> build OutputEncoding
     -> collect op extra: stride/padding/axis/dim/LUT 等
     -> kernel(inputs_int, params, out_enc, extra)
     -> 返回 FixedPointSimTensor
```

如果没有 output quantizer，或者没有注册对应 fixed kernel，dispatch 返回 `None`，上层会抛出明确错误，不允许静默 fallback 到 float。这样做是为了避免“看起来跑通但其实不是硬件语义”的风险。

### 7.3 边界量化

G3 的输入和权重通过 `boundary_quantize.py` 从 float 进入整数域：

```text
quantize_boundary_from_affine(tensor, encoding)
  if encoding 已缓存 FixedScaleEncoding
     或 AIMET_RX_INT16_BOUNDARY_USE_M_R=1:
       使用 (M,r) 量化
  else:
       使用原 affine float scale 量化
```

这解释了 G2/G3 的一个细节：G2 总是用 `(M,r)` 做 Q/DQ；G3 的边界默认兼容历史 float scale，只有预转换或打开环境变量时才强制使用 `(M,r)`。但 G3 的导出 sidecar 会包含 `m_int16/rshift`。

### 7.4 层间 requantize

对于 Conv/Linear，整数累加的自然 scale 是：

\[
s_{acc} = s_x \cdot s_w
\]

但输出 quantizer 期望 scale 是 \(s_y\)，所以需要层间 requantize：

\[
q_y = requant(acc; M_{req}, r_{req}, z_y)
\]

其中：

\[
\frac{M_{req}}{2^{r_{req}}} \approx \frac{s_x \cdot s_w}{s_y}
\]

这套 `M_req/r_req` 与 G2 quantizer 边界的 `m_int16/rshift` 不是同一个参数，不能混用：

| 参数 | 作用 | 生成位置 |
|------|------|----------|
| `m_int16/rshift` | 近似单个 quantizer 的 `scale` | `scale_fixed.py` |
| `multiplier/rshift` | 近似层间 `s_in/s_out` 或 `s_x*s_w/s_y` | `multiplier.py` / adapter / freeze |

---

## 8. 硬件最大位宽对齐：为什么要 int64，又何时饱和到 int32

这一节来自 “Alignment for hardware maximum bit width” 相关讨论，是理解 `requantize.py` 的关键。

### 8.1 `acc * multiplier` 为什么不能用 int32

`requantize_int` 的输入约束是：

| 量 | 类型/范围 | 位宽含义 |
|----|-----------|----------|
| `acc` | `torch.int32` | Conv/Linear/Add/Pool 的整数累加结果 |
| `multiplier` | `torch.int16`，范围 `[0, 32767]` | Q15 非负定点乘子 |
| `rshift` | `torch.int8`，范围 `[0, 31]` | 右移位数 |

最坏情况下：

\[
|acc \cdot multiplier| \approx 2^{31} \cdot 2^{15} = 2^{46}
\]

乘积需要约 47 bit（含符号），远超 int32 的 31 bit 正数范围。如果直接用 int32，结果会发生 wrap-around，PyTorch 通常不会替你报错，后续舍入和饱和都会建立在错误值上。

因此代码先提升到 int64：

```python
prod = acc.to(torch.int64) * multiplier.to(torch.int64)
```

这不是为了“模拟硬件有 int64 输出寄存器”，而是为了在软件中保留精确数学乘积，再按硬件定义的位置进行截断或饱和。许多硬件指令也有类似的 widened multiply 语义，例如 32x32 乘法产生更宽的中间结果。

### 8.2 为什么不用 float

不能用 float 代替 int64，原因有两个：

- `float32` 只有 24 bit 有效尾数，无法精确表达 47 bit 乘积。
- 即使用 `float64` 可以覆盖这一级乘积，浮点舍入规则也可能与整数硬件不同，破坏 bit-exact 对拍。

定点仿真的目标是“整数路径可复现”，所以中间量必须保持整数语义。

### 8.3 默认路径和严格硬件参考路径

当前实现区分两种需求：

| 模式 | 行为 | 目的 |
|------|------|------|
| 默认 | `acc * multiplier` 在 int64 中完成，再右移、加 zp、按输出值域饱和 | 保留较稳定的 e2e 精度和数学近似 |
| 严格硬件参考 | 在指定位置额外 `saturate_int32` | 对齐 Ada200 最大位宽/ALU 饱和语义 |

相关环境变量：

| 环境变量 | 作用 |
|----------|------|
| `AIMET_RX_REQUANTIZE_INT32_SAT=1` | requantize 乘后执行 INT32 饱和 |
| `AIMET_RX_ACC_INT32_SAT=1` | Conv/Linear/Add/Pool 累加器执行 INT32 饱和 |
| `AIMET_RX_HW_REF=1` | 打开严格硬件参考语义，覆盖 requantize、PWL、MAC 等子路径 |
| `AIMET_RX_PWL_HW_REF=1` | 仅对 PWL/相关 requant 子路径打开硬件参考语义 |

这体现了项目的位宽对齐原则：

1. **数学中间量先用足够宽的整数类型保存**，防止软件自身溢出污染结果。
2. **硬件规定的最大位宽边界必须显式建模**，例如 INT32 累加器饱和、乘后饱和、最终 qmin/qmax 饱和。
3. **默认精度路径与严格硬件路径可切换**，用于定位“算法误差”和“硬件饱和误差”各自的影响。

### 8.4 `requantize_int` 的微观流程

```text
requantize_int(acc, multiplier, rshift, y_zp, qmin, qmax):
  1. 校验 dtype 和范围
  2. prod = int64(acc) * int64(multiplier)
  3. 如果打开 REQUANTIZE_INT32_SAT/HW_REF:
       prod = saturate_int32(prod)
  4. rounded = round_shift(prod, rshift, rounding_mode)
  5. shifted = rounded + y_zp
  6. return saturate_sim_tensor(shifted, qmin, qmax)
```

输出 dtype 是 `SIM_TENSOR_DTYPE`，当前为 `torch.int32`；输出数值一定被裁剪到 `[qmin, qmax]`。这就是“容器 dtype”和“语义 bitwidth”分离。

---

## 9. 离线 freeze 与 sidecar

### 9.1 为什么需要 freeze

G3 运行时不应该在每个 forward 中重新推导部署参数。`freeze_int16_fixed` 负责把 calibration / QAT 后的 float encoding 固化成可部署的整数参数：

```text
freeze_int16_fixed(sim, output_path)
  -> collect_v2_int16_layers(model)
  -> derive_int16_real_multiplier
  -> quantize_multiplier(real_multiplier)
  -> quantize_bias_int32
  -> generate PWL / CLZ LUT
  -> build_int16_sidecar_document
  -> 写出 *.int16.json 和可选 *.bias_int32.bin
```

### 9.2 sidecar 里有什么

sidecar 顶层格式是 `aimet_rx_int16_fixed_sidecar`，每层包含：

- op 类型；
- output encoding：`scale`、`zero_point`、`qmin/qmax`、`multiplier`、`rshift`；
- `input_requants`：Add/Concat 等多输入对齐参数；
- `pwl`：PWL 非线性段表；
- `clz`：sqrt/rsqrt/reciprocal/square 等 CLZ LUT；
- `phase_fold`：sin/cos 周期折叠信息；
- 可选 ONNX tensor name hints；
- 可选 bias int32 二进制路径。

运行时可通过：

```python
from aimet_torch.fixed_point.export.sidecar_loader import attach_int16_sidecar_to_model

attach_int16_sidecar_to_model(model, "/path/to/model.int16.json")
```

或设置：

```bash
export AIMET_RX_INT16_SIDECAR_PATH=/path/to/model.int16.json
```

加载后，`dispatch_int16_fixed` 会优先使用 sidecar 中的 PWL/CLZ 表，避免在线重新拟合。

---

## 10. kernel 实现拆解

### 10.1 Conv / Linear

Conv/Linear 的典型路径：

```text
input FixedPointSimTensor
weight FixedPointSimTensor
  -> centered_int32: int_repr - zero_point
  -> int32 MAC
  -> + bias_int32
  -> saturate_mac_accumulator
  -> requantize_int(multiplier, rshift, y_zp, qmin, qmax)
  -> FixedPointSimTensor
```

Conv2d 通过 `im2col_int` 保持整数路径，避免 `F.unfold` 在某些后端把数据带回浮点或不支持 int16/int32。Conv1d 可委托到 Conv2d 形态。Conv3d 有参考实现或受限路径，需要结合硬件对拍确认。

### 10.2 Add / Sub / Concat

多输入 op 的难点是每个输入可能有不同 scale/zp。实现会先把各输入对齐到输出网格：

```text
aligned_i = requantize_int(
  input_i.int_repr - input_i.zero_point,
  multiplier_i,
  rshift_i,
  0,
  output_qmin,
  output_qmax
)
```

然后执行 Add/Sub/Concat。Add/Sub 在严格硬件参考模式下会通过 `int32_add_sat` / `int32_sub_sat` 建模 INT32 ALU 饱和。

### 10.3 Multiply

Multiply 先做两个 centered int 的乘法，再用 output encoding 的 `multiplier/rshift` requantize。由于 `int16 * int16` 以及后续 scale 乘法都可能扩大位宽，实现中会使用 int32/int64 中间量，并在指定边界饱和。

### 10.4 Pool / Mean

MaxPool 是比较操作，通常不需要 requantize；AvgPool/Mean 需要对窗口求和后乘以 `1 / kernel_area` 或 `1 / reduce_size`，因此会生成对应 `real_m` 并调用 `requantize_int`。

### 10.5 LUT / 非线性

Sigmoid、Tanh、SiLU、GELU、Mish、Softplus、Hardsigmoid、Hardswish、exp/log、sin/cos 等走 PWL LUT。sqrt、rsqrt、reciprocal、square 走 CLZ LUT。关键点：

- PWL/CLZ 尽量离线生成并写入 sidecar；
- sin/cos 使用 `phase_fold` 把周期输入折到主值域；
- `AIMET_RX_HW_REF` / `AIMET_RX_PWL_HW_REF` 下使用更严格的硬件参考舍入和饱和；
- 无 CLZ 依赖时默认 soft-fail，可用 `AIMET_RX_REQUIRE_CLZ_LUT=1` 强制失败。

### 10.6 Shape ops

Reshape、Flatten、Transpose、Permute 等不改变数值，只重排 `int_repr` 并保留 scale/zp/qmin/qmax 元数据。

---

## 11. 端到端工作流

推荐主线如下：

```text
1. 准备 float 模型
   -> prepare_model
   -> BN fold / CLE / Bias Correction / AdaRound 可选

2. 建 QuantizationSimModel
   -> 默认 fp32_qdq

3. Calibration
   -> compute_encodings
   -> 得到 float AffineEncoding

4. 补 G3 所需 output quantizers
   -> ensure_output_quantizers_for_int16_eval(sim)
   -> 重跑/继续 compute_encodings

5. G2 验证
   -> convert_encodings_to_fixed_scale(sim)
   -> fixed_scale_qdq vs fp32_qdq

6. 导出
   -> ONNX / AIMET encodings
   -> freeze_int16_fixed 输出 *.int16.json

7. G3 验证
   -> int16_fixed_eval vs fp32_qdq
   -> 如有硬件 golden，再对比板端/仿真器

8. 可选 QAT
   -> int16_fixed_qat_sim 训练
   -> int16_fixed_eval 验收
```

这套流程已经在 MobileNet V2 e2e 中落地。共享骨架位于 `aimet_torch/fixed_point/e2e/sim_builder.py`，模型族 wrapper 位于 `e2e/mobilenet_v2.py`、`attention.py`、`yolo.py`、`audio.py` 等。

---

## 12. 验证体系

### 12.1 测试分层

| 层级 | 内容 | 代表文件 |
|------|------|----------|
| L1 单函数 | rounding、requantize、scale 转换 | `tests/fixed_point/test_requantize.py` |
| L2 单算子 | Conv/Linear/Add/Pool/LUT | `tests/fixed_point/kernels/*` |
| L3 模块 | tiny model / adapter / sidecar replay | `tests/fixed_point/end_to_end/*` |
| L4 e2e | MobileNet、ImageNet smoke/full | `tests/fixed_point/end_to_end/test_mobilenet_v2*.py` |
| 质量报告 | 多模式 cosine/SQNR/LSB/饱和统计 | `scripts/fixed_point/run_report*.sh` |

### 12.2 常用命令

```bash
cd aimet_rx
export PYTHONPATH="$(pwd)"

# 快速 fixed_point pytest，约 30s
bash scripts/fixed_point/run_fixed_point_fast.sh

# 全量 fixed_point，不含真实 ImageNet，约 5min
bash scripts/fixed_point/run_fixed_point_full.sh

# 严格 LUT / CLZ / requantize 硬件参考子集
bash scripts/fixed_point/run_hw_ref_checks.sh

# 质量报告 + baseline
bash scripts/fixed_point/run_report.sh
```

### 12.3 主要门槛

- `fixed_scale_qdq` vs `fp32_qdq`：作为 G2 主路径门禁，要求非常接近 baseline。
- `int16_fixed_eval` vs `fp32_qdq`：报告型对比或设宽松门槛，用于观察整数路径损失。
- `int16_fixed_eval` vs Ada200 golden：最终以硬件团队定义为准。
- 覆盖率：CI 对 `aimet_torch/fixed_point/` 有覆盖率门禁。

---

## 13. 当前实现状态与已知边界

### 13.1 已实现主能力

| 能力 | 状态 |
|------|------|
| ExecutionMode 框架 | 已实现 |
| fp16_qdq | 已实现 |
| fixed_scale_qdq | 已实现 |
| v1 StaticGrid fixed_scale 适配 | 已实现 |
| FixedPointSimTensor / encoding | 已实现 |
| requantize / rounding / saturate | 已实现 |
| Conv/Linear/Add/Sub/Mul/Pool/Shape/LUT/Softmax kernel | 已实现 |
| freeze_int16_fixed / sidecar | 已实现 |
| int16_fixed_qat_sim | 已实现 |
| compare_modes / quality report / CI | 已实现 |

### 13.2 需注意的边界

| 项 | 说明 |
|----|------|
| ADR-008 舍入 | 默认 half-to-even，硬件最终 bit-exact 规则仍需持续对齐 |
| 默认 HW_REF | 严格硬件参考默认不全链路打开，需通过环境变量或脚本显式启用 |
| PE `n_BX` 物理打包 | sidecar 保留逻辑语义，物理打包属于下游格式转换/硬件集成 |
| 板端 RTL golden | 不属于纯软件验收本身，需要硬件/仿真器接入 |
| 部分 op 变体 | 未覆盖配置应显式失败，不允许静默 float fallback |
| PWL per-channel | 离线 PWL 主要按 per-tensor encoding 路径实现 |

---

## 14. 排障思路

### 14.1 fixed_scale_qdq 差

优先检查：

1. 是否完成 calibration；
2. 是否调用 `convert_encodings_to_fixed_scale`；
3. scale 是否极小导致 `(M,r)` fold 或饱和；
4. `qmin/qmax/zero_point` 是否与 mixed precision 配置一致；
5. 是否误把 Po2 当成 Ada200 必需步骤。

### 14.2 int16_fixed_eval 差

优先检查：

1. 是否调用 `ensure_output_quantizers_for_int16_eval(sim)` 并重跑 `compute_encodings`；
2. 是否有未注册 kernel；
3. `freeze_int16_fixed` 是否生成 multiplier/rshift/bias/LUT；
4. saturation 统计是否异常；
5. Add/Concat 多输入 scale 是否已对齐；
6. PWL/CLZ 是否使用 sidecar 表；
7. 是否需要打开 `AIMET_RX_HW_REF` 定位硬件饱和差异。

### 14.3 板端差但主机仿真好

优先检查：

1. sidecar 字段是否被编译器/Runtime 完整消费；
2. ONNX tensor name 与 PyTorch layer name 映射是否一致；
3. 输入预处理、layout、padding、rounding 是否一致；
4. bias_int32、LUT、input_requants 是否同步；
5. 硬件是否采用与主机相同的饱和和舍入点。

---

## 15. 给新同学的阅读顺序

如果只想快速理解项目：

1. 读本文第 1-8 节，先建立 G2/G3 和位宽对齐的整体模型。
2. 读 `FixedPoint_Quantization_Design_v2.md` 的 §1-5、§8-10。
3. 读 `FixedPoint_Quantization_Spec/15_fixed_scale_qdq.md`，理解 G2。
4. 读 `FixedPoint_Quantization_Spec/05_requantize_int_kernel.md`、`07_conv_linear_kernel.md`、`08_eltwise_pool_concat_kernel.md`，理解 G3 kernel。
5. 跟一条代码调用链：`true_quant.py` -> `adapter.py` -> `conv_linear.py` -> `requantize.py`。
6. 跑 `bash scripts/fixed_point/run_fixed_point_fast.sh`，看测试覆盖哪些假设。

如果要接入新模型：

1. 先看 `aimet_torch/fixed_point/e2e/README.md`。
2. 图像/attention 类模型优先复用 image sampler 和 classification evaluator。
3. YOLO/音频类模型复用 PTQ skeleton，但 evaluator 和 QAT loss 需要按任务自定义。

---

## 16. 关键结论

- AIMET RX 的定点扩展不是替代 AIMET，而是在 AIMET 校准/QAT/导出体系上补齐 Ada200 所需的定点 scale 和整数仿真能力。
- G2 `fixed_scale_qdq` 解决“scale 怎样以 `(M,r)` 表示和导出”的问题。
- G3 `int16_fixed_eval` 解决“整数图怎样计算、requantize、饱和、查表”的问题。
- `FixedPointSimTensor` 的 `int_repr` 使用 int32 容器是工程选择，真正的语义 bitwidth 由 `qmin/qmax` 管控。
- `acc * multiplier` 必须先用 int64 保存精确乘积；硬件最大位宽通过显式的 `saturate_int32`、`saturate_sim_tensor` 和环境变量开关建模。
- sidecar 是部署闭环的关键产物，承载了 ONNX/encodings 之外的整数 kernel 参数。
- 验收必须分层：G2 对 fp32 baseline、G3 对 fp32 或 hardware golden、板端最终 sign-off 不可省略。

---

## 17. 参考索引

| 主题 | 路径 |
|------|------|
| 顶层设计 | `doc/FixedPoint_Quantization_Design_v2.md` |
| 实施 spec 总览 | `doc/FixedPoint_Quantization_Spec/00_overview.md` |
| API 契约 | `aimet_torch/v2/quantization/affine/fixed_point/INTERFACE.md` |
| G2 fixed scale | `aimet_torch/fixed_point/fixed_scale_qdq.py`、`offline/scale_fixed.py` |
| G3 dispatch | `aimet_torch/v2/quantization/affine/fixed_point/adapter.py` |
| 整数载体 | `aimet_torch/fixed_point/tensor.py` |
| Requantize 和硬件位宽 | `aimet_torch/fixed_point/requantize.py` |
| Multiplier 生成 | `aimet_torch/fixed_point/offline/multiplier.py` |
| 离线 freeze | `aimet_torch/fixed_point/offline/pipeline.py` |
| Sidecar | `aimet_torch/fixed_point/export/sidecar.py` |
| Kernel | `aimet_torch/fixed_point/kernels/` |
| 测试 | `tests/fixed_point/` |
| 脚本 | `scripts/fixed_point/README.md` |


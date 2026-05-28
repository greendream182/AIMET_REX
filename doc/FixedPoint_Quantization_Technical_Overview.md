# AIMET RX 定点量化实现概览

> 面向内部分享和新人快速理解。更完整的实现细节见 `FixedPoint_Quantization_Technical_Implementation.md`、`FixedPoint_Quantization_Design_v2.md` 和 `FixedPoint_Quantization_Spec/`。

## 1. 项目解决什么问题

`aimet_rx-main` 在 AIMET QuantSim 之上，补齐 Ada200 NPU 部署需要的定点量化能力：

- 继续复用 AIMET 做 PTQ、QAT、混合精度和 encodings 管理。
- 把每个 quantizer 的 float `scale` 转成硬件可消费的 `(M_int16, rshift)`。
- 导出 ONNX / AIMET encodings 之外，额外生成 `*.int16.json` sidecar。
- 提供一条整数 kernel 仿真路径，用于主机侧对齐硬件整数语义。

一句话：**AIMET 负责训练和量化仿真基础设施，AIMET RX 负责把 scale、整数 kernel、requantize、LUT 和导出参数对齐到 Ada200。**

---

## 2. 最重要的设计：G2 / G3 双路径

项目不是把所有路径都改成整数，而是明确分成两条用途不同的路径。

| 路径 | ExecutionMode | 做什么 | 适合什么 |
|------|---------------|--------|----------|
| G2 主路径 | `fixed_scale_qdq` | Q/DQ 使用 `(M,r)`，算子内部仍是 float op | PTQ/QAT 后验证、导出门禁、与 fp32 baseline 对比 |
| G3 整数仿真 | `int16_fixed_eval` | 段间传整数载体，Conv/Add/Pool/LUT 等走 fixed kernel | 硬件整数语义对拍、分析饱和和 requantize 误差 |

两条路径**数值不等价**：

```text
G2 fixed_scale_qdq:
  fp -> Q/DQ(M,r) -> fp Op -> Q/DQ(M,r) -> fp

G3 int16_fixed_eval:
  fp -> Q -> int tensor -> int kernel + requantize -> int tensor -> DQ/debug
```

因此：

- `fixed_scale_qdq` 通过，不能替代 `int16_fixed_eval` 或板端 sign-off。
- `int16_fixed_eval` 不是 PTQ 校准模式，校准仍应在 `fp32_qdq` / `fp16_qdq` 下完成。

---

## 3. 核心数据结构

| 名称 | 作用 |
|------|------|
| `FixedScaleEncoding` | G2 使用，保存 `m_int16`、`rshift`、`zero_point`、`qmin/qmax` |
| `OutputEncoding` | G3 kernel 使用，除 scale/zp/qmin/qmax 外，还保存层间 `multiplier/rshift` |
| `FixedPointSimTensor` | G3 段间整数载体，历史兼容名是 `Int16QuantizedTensor` |
| `SIM_TENSOR_DTYPE` | G3 的 `int_repr` 容器 dtype，目前是 `torch.int32` |
| sidecar | `*.int16.json`，记录部署所需 multiplier、rshift、bias、LUT 等 |

注意：`Int16QuantizedTensor` 这个历史名字不表示所有激活/权重都是 16 bit。当前实现中，容器 dtype 是 `torch.int32`，实际量化值域由 `qmin/qmax` 表达，可以承载 U8、S8、U16、S16 等语义。

---

## 4. 端到端流程

典型流程如下：

```text
1. 准备 float 模型
   prepare_model / BN fold / CLE / Bias Correction / AdaRound 可选

2. 构建 AIMET QuantizationSimModel
   默认模式是 fp32_qdq

3. Calibration / QAT
   compute_encodings 得到 float AffineEncoding

4. 生成定点 scale
   convert_encodings_to_fixed_scale(sim)
   scale -> (M_int16, rshift)

5. G2 验证
   fixed_scale_qdq vs fp32_qdq

6. 离线 freeze 和导出
   freeze_int16_fixed(sim, "model.int16.json")
   生成 multiplier/rshift、bias_int32、PWL/CLZ LUT、sidecar

7. G3 验证
   int16_fixed_eval vs fp32_qdq
   如有硬件 golden，再对比板端/仿真器
```

核心代码入口：

| 阶段 | 入口 |
|------|------|
| 模式切换 | `aimet_torch/fixed_point/execution_mode.py` |
| G2 Q/DQ | `fixed_scale_qdq.py`、`offline/scale_fixed.py` |
| G3 dispatch | `v2/quantization/affine/fixed_point/adapter.py` |
| 整数载体 | `fixed_point/tensor.py` |
| Requantize | `fixed_point/requantize.py` |
| Kernel | `fixed_point/kernels/` |
| Freeze | `fixed_point/offline/pipeline.py` |
| Sidecar | `fixed_point/export/sidecar.py` |

---

## 5. G2 是怎样实现的

G2 的关键是把 float scale 转成定点形式：

\[
scale \approx \frac{M}{2^{rshift}}
\]

实现链路：

```text
convert_encodings_to_fixed_scale(sim)
  -> 遍历所有已初始化 quantizer
  -> get_encodings()
  -> quantize_scale_to_m_rshift(scale)
  -> 缓存 FixedScaleEncoding
```

运行时，当 mode 是 `fixed_scale_qdq`：

```text
torch_builtins.quantize_dequantize(...)
  -> quantize_dequantize_from_float_encoding(...)
  -> Q/DQ 使用 (M,r)
```

重要边界：

- G2 只改变 Q/DQ 的 scale 表示。
- Conv/Linear/ReLU 等算子内部仍是 PyTorch float 计算。
- Ada200 不要求 scale 是 Power-of-2；Po2 只是可选后处理，不是部署前置条件。

---

## 6. G3 是怎样实现的

G3 在 `int16_fixed_eval` 下接管 v2 TrueQuant module 的 forward：

```text
TrueQuant.forward
  -> dispatch_int16_fixed
     -> 找到原始 op 类型
     -> get_fixed_kernel(op)
     -> 输入/权重量化成 FixedPointSimTensor
     -> 生成 OutputEncoding
     -> 调用整数 kernel
     -> 返回 FixedPointSimTensor
```

以 Conv/Linear 为例：

```text
input_int, weight_int
  -> centered_int32 = int_repr - zero_point
  -> int32 MAC
  -> + bias_int32
  -> requantize_int(multiplier, rshift, output_zp)
  -> saturate 到 qmin/qmax
  -> FixedPointSimTensor
```

没有注册 fixed kernel 或缺 output quantizer 时，代码会显式失败，避免静默走 float fallback。

---

## 7. 硬件最大位宽对齐

`requantize.py` 是理解硬件位宽对齐的核心。

层间 requantize 的输入是：

| 量 | 类型/范围 |
|----|-----------|
| `acc` | `torch.int32` 累加器 |
| `multiplier` | `torch.int16`，范围 `[0, 32767]` |
| `rshift` | `torch.int8`，范围 `[0, 31]` |

为什么 `acc * multiplier` 必须先用 `int64`？

\[
2^{31} \times 2^{15} = 2^{46}
\]

乘积最多需要约 47 bit，int32 装不下，会静默回绕。float 也不合适，因为会破坏整数 bit-exact 语义。因此软件里先用 int64 保存精确乘积，再按硬件规则决定在哪里饱和或截断。

`requantize_int` 的微观流程：

```text
prod = int64(acc) * int64(multiplier)
if AIMET_RX_REQUANTIZE_INT32_SAT or AIMET_RX_HW_REF:
    prod = saturate_int32(prod)
rounded = round_shift(prod, rshift)
shifted = rounded + output_zero_point
out = saturate_sim_tensor(shifted, qmin, qmax)
```

相关开关：

| 环境变量 | 作用 |
|----------|------|
| `AIMET_RX_REQUANTIZE_INT32_SAT=1` | requantize 乘后 INT32 饱和 |
| `AIMET_RX_ACC_INT32_SAT=1` | MAC / Add / Pool 等累加器 INT32 饱和 |
| `AIMET_RX_HW_REF=1` | 打开严格硬件参考语义 |
| `AIMET_RX_PWL_HW_REF=1` | 对 PWL/LUT 相关路径打开硬件参考语义 |

默认路径偏向稳定 e2e 评估；HW_REF 路径用于定位和对齐硬件最大位宽行为。

---

## 8. sidecar 承载什么

`freeze_int16_fixed` 会生成 `*.int16.json`，用于把运行时整数参数带给下游编译器或仿真器。

sidecar 主要包含：

- 每层 op 类型；
- output encoding；
- 层间 `multiplier/rshift`；
- Conv/Linear 的 `bias_int32`；
- Add/Concat 的多输入 requant 参数；
- PWL LUT；
- CLZ LUT；
- sin/cos 的 `phase_fold`；
- 可选 ONNX tensor name hints。

运行时可以通过：

```python
attach_int16_sidecar_to_model(model, "model.int16.json")
```

或：

```bash
export AIMET_RX_INT16_SIDECAR_PATH=/path/to/model.int16.json
```

加载 sidecar 后，G3 dispatch 会优先使用冻结好的 LUT 和参数，不再在线重新拟合。

---

## 9. 测试和验收

常用命令：

```bash
cd /home/llq/workspace/aimet_rx-main
export PYTHONPATH="$(pwd)"

# 快速 fixed_point 回归
bash scripts/fixed_point/run_fixed_point_fast.sh

# 全量 fixed_point，不含真实 ImageNet
bash scripts/fixed_point/run_fixed_point_full.sh

# 严格硬件参考子集
bash scripts/fixed_point/run_hw_ref_checks.sh

# 质量报告和 baseline
bash scripts/fixed_point/run_report.sh
```

验收分层：

| 对比 | 用途 |
|------|------|
| `fixed_scale_qdq` vs `fp32_qdq` | G2 主路径精度门禁 |
| `int16_fixed_eval` vs `fp32_qdq` | G3 整数仿真误差观察 |
| `int16_fixed_eval` vs Ada200 golden | 硬件语义对拍 |
| sidecar replay vs online INT16 | 导出一致性检查 |

---

## 10. 已知边界

- `fixed_scale_qdq` 不能代替板端 sign-off。
- `int16_fixed_eval` 不是 PTQ 校准模式。
- 舍入规则默认 half-to-even，硬件最终 bit-exact 规则需持续确认。
- 严格 HW_REF 默认不全链路开启，需要显式设置环境变量或运行专用脚本。
- PE `n_BX` 物理打包和板端 RTL golden 属于下游硬件集成。
- 未覆盖的 op 或配置应显式失败，不应静默 fallback 到 float。

---

## 11. 推荐阅读顺序

快速理解：

1. 本文档。
2. `FixedPoint_Quantization_Design_v2.md` 的 §1-5、§8-10。
3. `FixedPoint_Quantization_Spec/15_fixed_scale_qdq.md`。
4. `FixedPoint_Quantization_Spec/05_requantize_int_kernel.md`。
5. `aimet_torch/v2/quantization/affine/fixed_point/adapter.py`。
6. `aimet_torch/fixed_point/requantize.py`。

需要更完整细节时，再读：

- `FixedPoint_Quantization_Technical_Implementation.md`
- `FixedPoint_Quantization_Spec/`
- `aimet_torch/v2/quantization/affine/fixed_point/INTERFACE.md`


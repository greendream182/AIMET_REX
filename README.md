# AIMET RX

**AIMET RX** 在 Qualcomm [AIMET](https://github.com/quic/aimet) 之上，为面向 **Ada200 系列 NPU** 的模型部署提供定点量化仿真、整数 kernel 对拍与 sidecar 导出能力。仓库同时保留 AIMET 的 PTQ / QAT、混合精度、Po2 scale 与 ONNX + encodings 导出等完整能力。

| 项 | 说明 |
|---|---|
| GitLab | [ai-software/rxmet](http://192.168.30.203/ai-software/rxmet) |
| 开发分支 | `aimet-rx`（合并至 `main` 前在此提交） |
| 克隆目录名 | 建议 `aimet_rx` |
| 当前版本 | **v1.3.7**（见 [CHANGELOG.md](./CHANGELOG.md)） |

> 文档入口：[`doc/README.md`](doc/README.md) · 建议先读 [`doc/项目总览与验收核实.md`](doc/项目总览与验收核实.md)

---

## 核心能力

| 能力 | 说明 |
|------|------|
| AIMET 量化基础设施 | `aimet_torch` / `aimet_onnx` / `aimet_common`：QuantSim、PTQ、QAT、混合精度 |
| 定点扩展 | `aimet_torch/fixed_point/`：执行模式切换、`(M_int16, rshift)` Q/DQ、整数 kernel、requantize |
| Sidecar 导出 | `*.int16.json`：multiplier / rshift / bias / LUT 等部署参数 |
| 质量门禁 | `scripts/fixed_point/` + `baseline.json`：cosine、LSNR、LSB 等自动化报告 |
| CI | [`.github/workflows/fixed_point_ci.yml`](.github/workflows/fixed_point_ci.yml) |

一句话：**AIMET 负责量化仿真基础设施，AIMET RX 负责把 scale、整数 kernel、requantize、LUT 与导出参数对齐到 Ada200。**

---

## G2 与 G3 双路径（核心设计）

AIMET RX **不是**把整条推理链都改成整数运算，而是刻意拆成两条**用途不同、数值不等价**的路径。读文档或验收时请先分清 G2 / G3。

### 对照表

| | **G2 主路径** | **G3 整数仿真** |
|---|----------------|-----------------|
| **ExecutionMode** | `fixed_scale_qdq` | `int16_fixed_eval`（训练仿真：`int16_fixed_qat_sim`） |
| **边界 Q/DQ** | 用 **`(M_int16, rshift)`** 做 quantizer 边界量化/反量化 | 段间传递 **整数仿真载体**（`FixedPointSimTensor` / 历史名 `Int16QuantizedTensor`） |
| **算子内部** | 仍是 **float op**（PyTorch 原模块） | **整数 kernel**（Conv / Add / Pool / LUT 等）+ 层间 **`requantize_int`** |
| **主要用途** | PTQ/QAT 后验证、**导出门禁**、与 `fp32_qdq` baseline 对比 | 硅前 **硬件整数语义对拍**、分析饱和 / requantize 误差 |
| **能否做 PTQ 校准** | 否（校准在 `fp32_qdq` / `fp16_qdq` 完成） | **否**——G3 是校准 **之后** 的对拍模式，不是 PTQ 模式 |
| **能否替代板端 sign-off** | **否** | **否** |

### 数据流（为何数值不等价）

```text
G2  fixed_scale_qdq（部署主路径）:
  fp ──Q(M,r)──► fp ── Op_float ──► fp ──Q(M,r)──► fp
  （只在 quantizer 边界用定点 scale；卷积/线性等仍在 float 域计算）

G3  int16_fixed_eval（整数仿真）:
  fp ──Q──► int 载体 ── Op_fix + requant ──► int 载体 ──► DQ（调试）
  （段间传整数；MAC、饱和、舍入、层间 requant 均走 fixed kernel）
```

因此：

- **`fixed_scale_qdq` 通过 ≠ `int16_fixed_eval` 通过**——两条路径都要各自验收。
- **G2 通过不能代替 G3 或板端 golden**；G3 通过也不能代替上板 sign-off。
- Sidecar（`*.int16.json`）记录 multiplier / rshift / bias / LUT 等部署参数；G2 导出与 G3 离线 freeze 共用同一套 offline 管线，但**运行时仿真语义**仍由上述两条路径分别承担。

### 推荐工作流

```text
1. fp32_qdq（或 fp16_qdq）     ← PTQ / QAT 校准，得到 float AffineEncoding
2. convert_encodings_to_fixed_scale   ← scale → (M_int16, rshift)
3. fixed_scale_qdq            ← G2：主路径验证、与 fp32 对比、导出门禁
4. freeze_int16_fixed         ← 生成 *.int16.json sidecar
5. int16_fixed_eval           ← G3（可选）：整数 kernel 对拍
6. int16_fixed_qat_sim        ← G3（可选）：在整数行为约束下微调权重；验收仍用 int16_fixed_eval
```

### 关键数据结构

| 名称 | 所属路径 | 作用 |
|------|----------|------|
| `FixedScaleEncoding` | G2 | 保存 quantizer 的 `m_int16`、`rshift`、`zero_point`、`qmin/qmax` |
| `OutputEncoding` | G3 kernel | 层间 requant 用的 `multiplier` / `rshift` 等 |
| `FixedPointSimTensor` | G3 | 段间整数载体（历史别名 `Int16QuantizedTensor`；容器 dtype 为 `int32`，值域由 `qmin/qmax` 决定，可表达 U8/S8/U16/S16 语义） |
| `*.int16.json` | 导出 | 部署 sidecar，供编译器 / runtime 消费 |

更完整的设计说明见 [doc/FixedPoint_Quantization_Design_v2.md](doc/FixedPoint_Quantization_Design_v2.md) §1.4 / §3.7。

---

## 量化执行模式

通过 API 或环境变量 `AIMET_RX_QUANT_EXECUTION_MODE` 切换（未设置时等同标准 AIMET `fp32_qdq`）：

| 模式 | 路径 | 含义 | 典型用途 |
|------|------|------|----------|
| `fp32_qdq` | — | 默认伪量化，计算用 FP32 | **PTQ/QAT 校准**；与上游 AIMET 100% 兼容 |
| `fp16_qdq` | — | 伪量化后 FP16 参与算子（experimental） | FP16 实验 |
| **`fixed_scale_qdq`** | **G2** | Q/DQ 用 `(m_int16, rshift)`，段内仍 float op | **Ada200 部署主路径**、导出门禁 |
| `int16_fixed_eval` | **G3** | 整数 MAC + 层间 requant | 硬件整数语义对拍（**非 PTQ**） |
| `int16_fixed_qat_sim` | **G3** | 前向同 eval，反向 STE | 整数行为约束下的 QAT；验收仍用 `int16_fixed_eval` |


## 仓库结构

```text
aimet_rx/
├── aimet_common/          # AIMET 通用模块
├── aimet_onnx/            # ONNX QuantSim 与优化
├── aimet_torch/           # PyTorch QuantSim；定点扩展在 fixed_point/
├── doc/                   # 设计文档、spec、验收说明
├── examples/              # quick_start、fixed_point_minimal、freeze_int16_fixed 等
├── export_onnx_and_encodings/  # ONNX + encodings 导出工具
├── scripts/fixed_point/   # 质量报告、compare_modes、baseline
├── tests/fixed_point/     # 定点 kernel / E2E / 回归测试
├── Makefile               # test-fixed-point-fast / coverage / baseline
└── pyproject.toml         # 包名 aimet-rx
```

---

## 快速开始

### 环境

```bash
git clone ssh://git@192.168.30.203:2222/ai-software/rxmet.git aimet_rx
cd aimet_rx
git checkout aimet-rx          # 开发分支

pip install -r requirements.txt
export PYTHONPATH="$(pwd)"
```

E2E（MobileNet 等）测试额外需要：`torchvision`、`onnxscript`。

### 标准 AIMET 量化（PTQ / QAT）

```python
from aimet_torch.v2 import quantsim
from aimet_torch.utils_rx import apply_mixed_precision_bitwidth

sim = quantsim.QuantizationSimModel(
    model,
    dummy_input=sample_input,
    quant_scheme="percentile",
    default_output_bw=8,
    default_param_bw=8,
)

apply_mixed_precision_bitwidth(sim.model, config_file="bitwidth_config.json")

import aimet_torch.v2 as aimet
with aimet.nn.compute_encodings(sim.model):
    for inputs, _ in calib_loader:
        sim.model(inputs)
```

完整 SpeechCommands + MRNN 示例见 [`examples/quick_start.py`](examples/quick_start.py)。

最小 G2/G3 对比（tiny `QuantizedLinear`）：

```bash
python examples/fixed_point_minimal.py
```

详见 [`examples/README.md`](examples/README.md)。

### 切换执行模式（G2 / G3 示例）

```python
from aimet_torch.fixed_point import (
    ExecutionMode,
    ensure_output_quantizers_for_int16_eval,
    quant_execution_mode,
    set_quant_execution_mode,
)

# G2：部署主路径验证
set_quant_execution_mode(ExecutionMode.FIXED_SCALE_QDQ)
out_g2 = model(x)

# G3：整数 kernel 对拍（需先 ensure_output_quantizers + 离线 multiplier/bias/LUT）
ensure_output_quantizers_for_int16_eval(sim)
with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
    out_g3 = model(x)

# 或环境变量：export AIMET_RX_QUANT_EXECUTION_MODE=int16_fixed_eval
```

### 冻结 sidecar（G2/G3 共用部署参数）

校准完成并生成 `(M, rshift)` 后，导出 `*.int16.json` 供编译器 / runtime 使用：

```python
from aimet_torch.fixed_point.offline import freeze_int16_fixed

sim.compute_encodings(forward_pass_callback)
report = freeze_int16_fixed(sim, "exports/model.int16.json")
```

CLI 演示：

```bash
python examples/freeze_int16_fixed.py --demo --output exports/demo.int16.json
```

---

## 测试与质量验收

```bash
make test-fixed-point-fast    # 快速回归（跳过 slow / imagenet）
make coverage                 # 覆盖率报告
make baseline                 # 对照 baseline.json
make report-fast              # 快速质量报告
bash scripts/fixed_point/run_software_signoff.sh   # 软件 sign-off 套件
```

模式对比 CLI：

```bash
python scripts/fixed_point/compare_quant_modes.py --model dual_linear \
  --report scripts/fixed_point/reports/quant_mode_report.json
```

---

## 文档索引

| 文档 | 说明 |
|------|------|
| [doc/项目总览与验收核实.md](doc/项目总览与验收核实.md) | 设计目的、实现地图、验收结论（**建议先读**） |
| [doc/FixedPoint_Quantization_Design_v2.md](doc/FixedPoint_Quantization_Design_v2.md) | 定点量化顶层设计（当前有效） |
| [doc/FixedPoint_Quantization_Technical_Implementation.md](doc/FixedPoint_Quantization_Technical_Implementation.md) | 完整技术实现说明 |
| [doc/FixedPoint_Quantization_Spec/](doc/FixedPoint_Quantization_Spec/00_overview.md) | 分模块实施 spec（M1–M6；M2.5 见 spec 15） |
| [INTERFACE.md](aimet_torch/v2/quantization/affine/fixed_point/INTERFACE.md) | 公共接口契约 |
| [doc/Quant_config.md](doc/Quant_config.md) | QuantSim 双层 JSON 配置 |
| [scripts/fixed_point/README.md](scripts/fixed_point/README.md) | 质量报告与 CI 说明 |

---

## 构建与安装

```bash
python -m pip install build
python -m build
pip install dist/aimet_rx-1.3.7-py3-none-any.whl
```

---

## 外部依赖

**QuantGRU** 来自上游 [**CX9898/quant-gru-pytorch**](https://github.com/CX9898/quant-gru-pytorch)（含 CUDA/C++ 扩展，需从源码编译）。本仓库不打包该模块，说明见 [`quant-gru-pytorch/`](./quant-gru-pytorch/)。

---

## AIMET RX 工具函数（`aimet_torch.utils_rx`）

| 函数 | 用途 |
|------|------|
| `apply_mixed_precision_bitwidth()` | 应用混合精度位宽 JSON 配置 |
| `setup_percentile_calibration()` | Percentile 校准 |
| `apply_power_of_2_workflow()` | Power-of-2 scale 对齐（NPU 友好） |
| `freeze_quantizer_parameters()` | 冻结量化器参数（QAT 准备） |

混合精度配置优先级（高 → 低）：`layer_name_config`（精确名）→ `layer_name_config`（通配符 `*`）→ `layer_type_config` → `default_bitwidth`。

---

## 许可证

BSD-3-Clause · Copyright (c) 2024–2026, Qualcomm Innovation Center, Inc.

本包基于 AIMET 官方版本定制，增加了 Ada200 定点量化扩展、便捷工具函数与 bug 修复。

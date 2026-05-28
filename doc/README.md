# AIMET RX 文档目录

## 快速入口

| 文档 | 说明 |
|------|------|
| [**项目总览与验收核实**](项目总览与验收核实.md) | 设计目的、原理、实现地图、使用方式、验收结论（建议先读） |
| [**FixedPoint_Quantization_Design_v2.md**](FixedPoint_Quantization_Design_v2.md) | **定点量化与仿真顶层设计**（当前唯一推荐阅读） |
| [FixedPoint_Quantization_Technical_Overview.md](FixedPoint_Quantization_Technical_Overview.md) | 面向内部分享的精简实现概览（推荐分享入口） |
| [FixedPoint_Quantization_Technical_Implementation.md](FixedPoint_Quantization_Technical_Implementation.md) | 面向内部分享的完整技术实现说明（宏观流程、代码路径、硬件位宽对齐） |
| [FixedPoint_Quantization_Design.md](FixedPoint_Quantization_Design.md) | 历史归档（已废弃，请勿用于新需求） |
| [FixedPoint_Quantization_Spec/](FixedPoint_Quantization_Spec/00_overview.md) | 分模块实施 spec（M1–M6；M2.5 已实现，见 spec 15） |
| [../aimet_torch/v2/quantization/affine/fixed_point/INTERFACE.md](../aimet_torch/v2/quantization/affine/fixed_point/INTERFACE.md) | 公共接口契约（与代码同步维护） |
| [Quant_config.md](Quant_config.md) | QuantSim 双层 JSON 配置说明 |
| [../scripts/fixed_point/reports/quality_report.md](../scripts/fixed_point/reports/quality_report.md) | 定点质量自动化报告 |
| [../scripts/fixed_point/README.md](../scripts/fixed_point/README.md) | 质量报告、**CI**、`compare_modes` CLI |
| [../Makefile](../Makefile) | `make test-fixed-point-fast` / `coverage` / `baseline` |

## 目录结构

```text
doc/
├── README.md                               # 本文件
├── FixedPoint_Quantization_Design_v2.md    # 定点量化与仿真设计（当前）
├── FixedPoint_Quantization_Technical_Overview.md # 精简实现概览
├── FixedPoint_Quantization_Technical_Implementation.md # 技术实现说明
├── FixedPoint_Quantization_Design.md       # 历史归档（废弃）
├── Quant_config.md                         # 通用量化配置
├── 项目总览与验收核实.md                   # 项目总览与验收结论
└── FixedPoint_Quantization_Spec/         # 01–15 实施 spec（M2.5 见 15）
```

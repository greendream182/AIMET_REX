# Spec 集索引与统一模板

本目录是 AIMET RX 定点量化与计算改造的实施 spec 集。每个 `NN_xxx.md` 描述一个可独立交付的子任务。顶层叙事文档见 **[../FixedPoint_Quantization_Design_v2.md](../FixedPoint_Quantization_Design_v2.md)**。所有 public API 签名以 [INTERFACE.md](../../aimet_torch/v2/quantization/affine/fixed_point/INTERFACE.md) 为准。

---

## 1. 阅读顺序

按里程碑：

- M1：[01_execution_mode_api.md](01_execution_mode_api.md)
- M2：[02_fp16_qdq_v2.md](02_fp16_qdq_v2.md) → [03_fp16_qdq_v1.md](03_fp16_qdq_v1.md)
- **M2.5**：[15_fixed_scale_qdq.md](15_fixed_scale_qdq.md)（**已实现**）
- M3：[04_int16_tensor_and_encoding.md](04_int16_tensor_and_encoding.md) → [05_requantize_int_kernel.md](05_requantize_int_kernel.md) → [06_fixed_kernel_registry.md](06_fixed_kernel_registry.md) → [07_conv_linear_kernel.md](07_conv_linear_kernel.md)
- M4：[08_eltwise_pool_concat_kernel.md](08_eltwise_pool_concat_kernel.md) → [09_lut_nonlinear_kernel.md](09_lut_nonlinear_kernel.md)
- M5：[10_offline_param_pipeline.md](10_offline_param_pipeline.md) → [11_qat_sim_backward.md](11_qat_sim_backward.md)
- M6：[12_metrics_and_compare.md](12_metrics_and_compare.md) → [13_tests_acceptance.md](13_tests_acceptance.md) → [14_ci_and_baseline.md](14_ci_and_baseline.md)

---

## 2. 统一模板

每个 spec 文件按以下章节组织。agent 实施时按章节顺序读取即可获得完整上下文。

```text
# 任务标题

## 1. 目标 (Why)
一句话说明业务/技术目的。

## 2. 范围 (Scope)
2.1 修改 / 新增的文件清单（带路径）
2.2 不在本 spec 范围内的事

## 3. 前置依赖
依赖的 spec / 库 / 配置。

## 4. 数据契约
输入 / 输出的 dtype、shape、device、约束。
不变式 (invariants)。

## 5. API 签名
Python 类型签名。docstring 模板。

## 6. 算法 / 伪代码
逐步可执行伪代码。关键边界（溢出、零除、空 tensor）。

## 7. 实施步骤
具体修改哪些文件、新增什么类、按顺序的子步骤。

## 8. 验收标准
8.1 单元测试用例（输入 → 期望输出）
8.2 必须通过的现有测试
8.3 性能阈值（如适用）

## 9. 不允许做的事 (Do NOT)
明确禁区。

## 10. 参考
代码引用、参考论文、参考项目。
```

---

## 3. spec 编写规则

- 每份 spec 自包含。读完一份就能开工。
- API 签名必须与 INTERFACE.md 一致；如发现冲突，以 INTERFACE.md 为准。
- 涉及现有代码改动时必须给出文件路径与函数名（最好附行号）。
- 所有伪代码使用项目统一符号（见顶层文档第 2 节词汇表）。
- 验收标准必须包含至少一个具体数值用例。
- 禁止在 spec 中固化时间排期、人员分工、商业敏感信息。

---

## 4. 维护

- spec 改动需同步更新顶层文档第 12 节索引。
- 接口字段变更同步更新 INTERFACE.md，并在 ADR 增补条目。
- 任何 spec 标注 `Status: draft / accepted / implemented / deprecated`，本目录全部初始为 `accepted`。

## 5. 实现状态（2026-05-19）

> 详细验收数据与使用说明见 [项目总览与验收核实](../项目总览与验收核实.md)。

| 里程碑 | 实现 | 备注 |
|--------|------|------|
| M1 模式框架 | 是 | `execution_mode.py` |
| M2 fp16_qdq | 是 | v2/v1 测试 + 质量报告 |
| M2.5 fixed_scale_qdq | 是 | spec 15；v2 + **v1 StaticGrid**（`v1/fixed_scale_qdq.py`）；`test_fixed_scale_qdq.py`、`test_v1_fixed_scale_qdq.py` |
| M3 INT16 核心 | 是 | hardware_sim；tensor / requantize / conv-linear |
| M4 算子覆盖 | 是 | pool / eltwise / LUT / softmax / **Conv3d** |
| M5 离线 + QAT | 是 | `offline/pipeline.freeze_int16_fixed`、`int16_fixed_qat_sim` |
| M6 指标与 CI | 是 | `compare_quant_modes.py` CLI、`baseline.json`、CI workflow |
| ImageNet e2e | 可选 | `test_imagenet_mobilenet_v2.py`；默认 `variant=torchvision`（ImageNet 预训练） |
| spec 13 golden / 覆盖率 | 是 | `tests/fixed_point/data/*.npz`、`.coveragerc`（≥85%，CI 门禁） |

**建议**：将已交付 spec 的 `Status` 从 `accepted` 改为 `implemented` 时，由 owner 逐份 PR 更新，避免与 INTERFACE 漂移不同步。

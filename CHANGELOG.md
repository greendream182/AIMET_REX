# Changelog

本文档记录本仓库所有重要变更。

格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本 SemVer](https://semver.org/lang/zh-CN/)。

## [Unreleased]

<!-- 在下次发版前，把新增条目写到这里 -->

## [1.3.7] - 2026-04-29

### 🐛 Bug 修复

- **修复 `export_onnx_and_encodings` 子模块的 import 路径错误（继承自 1.3.6）**
  - 1.3.6 wheel 中的 `export_onnx_and_encodings/{export_onnx_json, custom_gru_onnx, custom_bn_onnx}.py`
    使用了错误的 import 路径 `from aimet_rx.aimet_torch.xxx import ...`，导致用户侧
    `from export_onnx_and_encodings.export_onnx_json import export_onnx_json` 抛出
    `ModuleNotFoundError: No module named 'aimet_rx'`
  - 仓库源码已在 commit `b2a5d06 "Update Import"`（2026-04-28）改回正确的
    `from aimet_torch.xxx import ...`；本版本随之重新打 wheel 发布该修复
  - 影响范围：所有依赖 `export_onnx_json` / `custom_gru_onnx` / `custom_bn_onnx` 的下游代码

### 🔧 改进

- **打包配置统一到 `pyproject.toml`（PEP 621 单一真相源）**
  - 删除 `setup.py`（此前与 `pyproject.toml` 双源、版本号一度脱节为 `1.0.2`/`1.3.6`）
  - 把 `packages` / `package_data` / `include_package_data` / `zip_safe` 等
    setuptools 配置迁移到 `[tool.setuptools]` 子表
  - 运行时依赖改为 `dynamic`，由 `pyproject.toml` 直接读 `requirements.txt`，
    避免 `pyproject` `dependencies` 字段与 `requirements.txt` 双重维护

### 📦 发布与升级

```bash
pip install --upgrade --force-reinstall aimet_rx-1.3.7-py3-none-any.whl

# 安装后自检
python -c "from export_onnx_and_encodings.export_onnx_json import export_onnx_json; print('OK')"
```

## [1.3.6] - 2026-04-21

### ✨ 新增功能

- **添加 QuantGRU 的源码级集成支持**
  （`aimet_torch/v2/nn/modules/custom.py`、`aimet_torch/model_preparer.py`）
  - 在 AIMET v2 自定义量化模块注册表中增加 `QuantGRU` 的透传包装器
  - `QuantGRU` 继续使用其自身的内部量化实现，不额外叠加 AIMET 输入/输出量化器

### 🔧 改进

- 增强 `prepare_model` 对 `QuantGRU` 的处理逻辑：
  - 自动将 `QuantGRU` 识别为 leaf module，避免 FX 展开其内部实现
  - 使用 `QuantGRU` 时，无需在业务脚本中额外手动配置 `module_classes_to_exclude=[QuantGRU]`
  - 降低 `quant-gru-pytorch` 接入 AIMET 的样板代码和使用门槛

### ⚠️ 已知问题

- 本版本 wheel 包含错误的 import 路径，导致 `export_onnx_and_encodings` 子模块在使用时报
  `ModuleNotFoundError: No module named 'aimet_rx'`，详见 [1.3.7] 的修复说明。
  **请直接升级到 1.3.7。**

## [1.2.2] - 2024-12-24

### 🐛 Bug 修复

- **修复对称量化配置问题**（`aimet_torch/utils_rx.py`）
  - 修复 `apply_mixed_precision_bitwidth` 在设置对称量化时，`qmin` / `qmax` 没有正确更新的问题
  - 修复前：设置 `symmetric=True` 后，`qmin` / `qmax` 仍然保持非对称值（如 `0, 255`）
  - 修复后：正确更新为对称范围（如 8-bit: `-128, 127`；2-bit: `-2, 1`）
  - 影响范围：所有通过 JSON 配置文件设置 `input_symmetric` / `output_symmetric` 的量化器

### 🔧 改进

- 增强量化参数设置逻辑：
  - 位宽变更时自动根据对称性更新 `qmin` / `qmax`
  - 对称性变更时自动根据当前位宽更新 `qmin` / `qmax`
  - 确保量化器状态始终一致

## [1.2.1] - 2024-12-23

### ✨ 新增功能

- 添加混合精度位宽配置工具函数
- 支持通过 JSON 配置文件灵活控制量化参数

## [1.2.0] - 2024-12-20

### ✨ 初始版本

- 基于 AIMET 官方版本构建
- 提供 PyTorch 和 ONNX 量化支持

<!--
版本对比链接（在使用 GitHub/GitLab 时填入对应仓库 URL，例如：

[Unreleased]: https://github.com/<org>/aimet-rx/compare/v1.3.7...HEAD
[1.3.7]: https://github.com/<org>/aimet-rx/compare/v1.3.6...v1.3.7
[1.3.6]: https://github.com/<org>/aimet-rx/compare/v1.2.2...v1.3.6
[1.2.2]:  https://github.com/<org>/aimet-rx/compare/v1.2.1...v1.2.2
[1.2.1]:  https://github.com/<org>/aimet-rx/compare/v1.2.0...v1.2.1
[1.2.0]:  https://github.com/<org>/aimet-rx/releases/tag/v1.2.0
-->

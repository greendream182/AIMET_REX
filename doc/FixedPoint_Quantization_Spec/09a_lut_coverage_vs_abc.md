# 09a LUT 覆盖对照（AIMET RX vs abc_lut `LUT_NONLINEAR_FUNCTIONS.md`）

Status: reference

## 生产路径（16 段 PWL + `evaluate_pwl_lut_int16`）

| 函数类 | abc 文档章节 | AIMET RX kernel | 备注 |
|--------|-------------|-----------------|------|
| Sigmoid | §3.1 | `SigmoidInt16Kernel` | PWL |
| Tanh | §2 基础 | `TanhInt16Kernel` | PWL |
| SiLU | §3.2 | `SiLUInt16Kernel` | PWL |
| GELU | §2 | `GELUInt16Kernel` | PWL |
| Mish / Softplus | §2 | `MishInt16Kernel` / `SoftplusInt16Kernel` | PWL |
| Hardsigmoid / Hardswish | §2 | 已注册 | PWL |
| LeakyReLU / PReLU | §2 | 已注册 | PWL |
| exp / log | §2 饱和 | `ExponentialInt16Kernel` / `LogInt16Kernel` | PWL；log 拟合域钳到正数 |
| Softmax | §5 | `SoftmaxInt16Kernel` | 内置 PWL `exp` + 整数归一化 |
| ReLU / ReLU6 | §2 | `eltwise` | 非 LUT |
| sin / cos | §3.2 周期 PWL | **已接** | `SinInt16Kernel` / `CosInt16Kernel` + `phase_fold`；cos 复用 sin 表 |
| sqrt / rsqrt / reciprocal (CLZ) | §1 CLZ | **已接** | 三 kernel + `generate_clz_lut_for_export`；reciprocal 负输入绕 `zp` 取反 |
| power_2 / Square | §1 CLZ | **已接** | `SquareInt16ClzKernel`；负输入按 `|x|²` 走 CLZ 正域 |
| 均匀 `lut_int16` | 早期查表 | `lookup_lut_int16` | 遗留，adapter 不用 |

## Sidecar 导出 / 加载（`*.int16.json`）

| 字段 | 内容 |
|------|------|
| `pwl` | `lut_int_general` 段表 + `quality`；sin/cos 含 `phase_fold`、`pwl_input_encoding` |
| `clz` | CLZ 段表（sqrt/rsqrt/reciprocal/power_2）；`export_metrics` 含拟合域 |
| `phase_fold` | 层顶冗余字段（与 `pwl.sin.phase_fold` 一致） |

导出：`derive_int16_pwl_json` / `derive_int16_clz_json` / `collect_v2_int16_layers` / `export_int16_sidecar_json`。

加载（推理）：

- `attach_int16_sidecar_to_model(model, path_or_doc)` → 各层 `_int16_sidecar_extra`
- 或设环境变量 ``AIMET_RX_INT16_SIDECAR_PATH=/path/to/model.int16.json``（`build_calibrated_sim` 校准后自动 attach）
- 层名不一致时：用 sidecar 内 ``onnx_tensor_names`` 或 PyTorch 路径后缀匹配（``resolve_sidecar_layer_to_module_names``）

`dispatch_int16_fixed` 优先使用 sidecar 中的 `pwl_lut` / `clz_lut`，**不再**在线重新拟合。

## 严格仿真

与 `lut_int_general` bit-exact：`AIMET_RX_HW_REF=1` 或 `AIMET_RX_PWL_HW_REF=1`（见 `test_lut_abc_reference.py`）。

## §3.0 / 周期类

| 能力 | API | 状态 |
|------|-----|------|
| 运行时对齐 | `align_op_quant_grid_to_lut_quant_grid` | 已实现 |
| ROM 烘焙（缓解 2） | `bake_op_scale_adapter_into_pwl_lut` + `scale_adapter_baked` | 已实现 |
| sin/cos 相位折叠 | `extra['phase_fold']` + `fold_periodic_input_to_principal_range` | 已实现 |
| CLZ 输出 grid 重映射 | `_remap_lut_q_to_op_grid` in `clz_lut.py` | 已实现 |

## 未实现（硬件/后续）

- PE `n_BX` 物理打包（JSON 为有符号 `n_bx_total`）
- 板端 RTL golden
- 默认全链路 `HW_REF`（见 `09b_lut_env_and_remaining.md`）

冻结 golden 验收：`tests/fixed_point/export/test_sidecar_clz_golden.py`（sidecar `clz` → kernel，与 `lut_int_general` atol≤3）。

环境变量与 sidecar 加载详见 **`09b_lut_env_and_remaining.md`**。

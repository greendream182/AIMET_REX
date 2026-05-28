# 09b LUT / INT16 环境变量与剩余项

Status: reference

## 环境变量（LUT / sidecar / 严格仿真）

| 变量 | 作用 |
|------|------|
| `AIMET_RX_HW_REF=1` | 全链路严格 INT32 饱和 + PWL half-up（与 `lut_int_general` bit-exact） |
| `AIMET_RX_PWL_HW_REF=1` | 仅 PWL 乘后 INT32 饱和 |
| `AIMET_RX_PWL_HW_MAC_SAT=1` | 仅 PWL MAC 饱和 |
| `AIMET_RX_REQUANTIZE_INT32_SAT=1` | requantize 乘后饱和 |
| `AIMET_RX_ACC_INT32_SAT=1` | Conv/eltwise/pool 累加器饱和 |
| `AIMET_RX_ABC_LUT_ROOT` | 离线 CLZ 拟合：`lut_int_general` 路径 |
| `AIMET_RX_INT16_SIDECAR_PATH` | 指向 `*.int16.json`；`build_calibrated_sim` 后自动 attach |
| `AIMET_RX_IMAGENET_VAL` | 真实 ImageNet val；可指向 ImageFolder 目录，也可指向 `imagenet_val.zip` |
| `AIMET_RX_IMAGENET_VAL_ZIP` | 真实 ImageNet val zip；当目录权限不可读时可直接读 ImageFolder-style zip |
| `AIMET_RX_BAKE_PWL_SCALE=1` | adapter 生成 PWL 后执行 §3.0 `bake_op_scale_adapter_into_pwl_lut` |
| `AIMET_RX_REQUIRE_CLZ_LUT=1` | 无 abc 时 CLZ 拟合失败则硬错误；默认 **soft-fail**（`dispatch` 返回 `None`，不进入 kernel） |

## CLZ 算子覆盖

| `custom` 模块 | CLZ 函数名 | Kernel |
|---------------|-----------|--------|
| Sqrt | sqrt | `SqrtInt16ClzKernel` |
| RSqrt | rsqrt | `RSqrtInt16ClzKernel` |
| Reciprocal | reciprocal | `ReciprocalInt16ClzKernel` |
| Square | power_2 | `SquareInt16ClzKernel` |

## 刻意未默认开启

- **默认 HW_REF**：e2e cosine 回归代价大；用 `run_hw_ref_checks.sh` 做严格子集。
- **真实 ImageNet**：需 `AIMET_RX_IMAGENET_VAL` 指向本地 ImageFolder val，或 `AIMET_RX_IMAGENET_VAL_ZIP` 指向 ImageFolder-style zip；纯软件 signoff 可用 `run_software_signoff.sh` 自动检测并运行。

## 非阻塞下游硬件集成项

- **PE `n_BX` 物理打包**：当前软件 sidecar 保留 `n_bx_total` 逻辑语义，供下游格式转换消费；不阻塞纯软件量产前验收。
- **板端 RTL golden**：属于硬件/仿真器接入阶段；当前 signoff 只覆盖软件 INT16 kernel、sidecar replay、abc 正域 golden 与 ImageNet 指标。

## 近期补齐

- **CLZ 拟合域**：`generate_clz_lut_for_export` 支持 `fit_float_min/max`（来自 quantizer `min`/`max`），避免对称量化负半轴导致 reciprocal 全局 output quant 爆炸。
- **`QuantizedSquare`**：v2 模块已启用；离线函数名 `power_2`。
- **Sidecar**：`Reciprocal` / `Square` 导出 `clz` 块；`test_sidecar_loader` 校验 float 校准 → online INT16 与 sidecar（含 JSON 落盘 reload）`int_repr` bit-exact（需先 `forward` 初始化 quantizer），并对错误 format/version/layers、缺失 module、空 PWL/CLZ payload fail-fast。

验收清单（勾选状态）：**`09c_lut_int16_acceptance.md`**。

## 单测锚点

| 区域 | 文件 |
|------|------|
| CLZ 离线拟合 | `tests/fixed_point/offline/test_clz_gen.py` |
| Sidecar 导出/加载 | `tests/fixed_point/export/test_sidecar_export.py`, `test_sidecar_loader.py` |
| CLZ golden | `tests/fixed_point/kernels/test_clz_*_golden.py` |
| 冻结 CLZ sidecar → kernel | `tests/fixed_point/export/test_sidecar_clz_golden.py` |
| MobileNet v2 smoke e2e | `tests/fixed_point/end_to_end/test_mobilenet_v2.py`（约 8s；含 sidecar 导出回放） |
| MobileNet PTQ/QAT e2e | `tests/fixed_point/end_to_end/test_mobilenet_v2_ptq_qat.py`（约 4–5min，含 CLE/AdaRound/QAT/AutoQuant） |
| 真实 ImageNet 软件验收 | `scripts/fixed_point/run_imagenet_validation.sh`（需 `AIMET_RX_IMAGENET_VAL` 或 `AIMET_RX_IMAGENET_VAL_ZIP`） |

## 验证命令

```bash
# 严格 LUT / CLZ / requantize 子集
bash scripts/fixed_point/run_hw_ref_checks.sh

# 快回归（约 30s，不含 mobilenet PTQ/QAT）
bash scripts/fixed_point/run_fixed_point_fast.sh

# 全量 fixed_point（约 5min，含 PTQ/QAT；不含 ImageNet）
bash scripts/fixed_point/run_fixed_point_full.sh

# 纯软件量产前 signoff：full + HW_REF + sidecar + ImageNet（若配置）
bash scripts/fixed_point/run_software_signoff.sh

# 强制要求真实 ImageNet 路径；缺失则失败
AIMET_RX_SIGNOFF_IMAGENET=required \
  bash scripts/fixed_point/run_software_signoff.sh

# 目录权限不可读时，直接使用 ImageFolder-style zip
AIMET_RX_IMAGENET_VAL_ZIP=/path/to/imagenet_val.zip \
  bash scripts/fixed_point/run_imagenet_validation.sh --source zip
```

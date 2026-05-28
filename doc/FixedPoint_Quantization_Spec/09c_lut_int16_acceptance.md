# 09c INT16 LUT / sidecar 软件量产前验收清单（非 PR）

Status: reference · 最后全量回归：271 passed（含 PTQ/QAT + MobileNet sidecar 回放）

## 运行时

- [x] PWL 16 段：`evaluate_pwl_lut_int16` + HW_REF 可选
- [x] CLZ：sqrt / rsqrt / reciprocal / power_2（Square）
- [x] sin/cos：`phase_fold` + 主值域拟合
- [x] adapter 在线拟合 + `AIMET_RX_REQUIRE_CLZ_LUT`
- [x] reciprocal 拟合域：`fit_float_min/max`（quantizer min/max）
- [x] CLZ 有符号域：`power_2` 用 `|x|`；`reciprocal` 负输入符号反射（**非** abc 负 `q` bit-exact；见 `test_clz_*_golden` 负域用例）
- [x] CLZ 缺 abc 时 adapter soft-fail；`AIMET_RX_REQUIRE_CLZ_LUT=1` 硬失败
- [x] sidecar `detach` 后 env 可重新 attach

## Sidecar

- [x] 导出 `pwl` / `clz` / `phase_fold`
- [x] `attach_int16_sidecar_to_model` / `AIMET_RX_INT16_SIDECAR_PATH`
- [x] `build_calibrated_v2_sim` 校准后自动 attach
- [x] 冻结 golden：`test_sidecar_clz_golden.py`
- [x] MobileNet 导出回放：`test_mobilenet_v2_sidecar_export_matches_online_int_repr`
- [x] float quantizer → sidecar JSON → reload：`test_sidecar_loader`（sqrt/reciprocal/square + JSON 落盘）
- [x] sidecar fail-fast：错误 format/version/layers、缺失 module、空 PWL/CLZ payload 明确失败

## 软件 signoff

- [x] 一键软件验收脚本：`scripts/fixed_point/run_software_signoff.sh`
- [x] full fixed_point：`run_fixed_point_full.sh`
- [x] 严格软件 golden 子集：`run_hw_ref_checks.sh`
- [x] 真实 ImageNet：`run_imagenet_validation.sh`，由 `AIMET_RX_IMAGENET_VAL` 提供本地 ImageFolder val，或由 `AIMET_RX_IMAGENET_VAL_ZIP` 直接读取 ImageFolder-style zip

## 回归脚本

| 脚本 | 约耗时 |
|------|--------|
| `run_hw_ref_checks.sh` | &lt;10s |
| `run_fixed_point_fast.sh` | ~30s |
| `run_fixed_point_full.sh` | ~5min |
| `run_imagenet_toy_smoke.sh` | ~10s |
| `run_imagenet_validation.sh` | 取决于样本数 |
| `run_software_signoff.sh` | full + HW_REF + sidecar + ImageNet（若配置） |

## 未纳入默认

- [ ] 默认 `AIMET_RX_HW_REF=1`（e2e 代价）
- [ ] 真实 ImageNet val 默认强制执行（需 `AIMET_RX_SIGNOFF_IMAGENET=required`；目录或 zip 均可）

## 非阻塞下游硬件集成项

- [ ] PE `n_BX` 物理打包（当前 sidecar 保留逻辑语义）
- [ ] 板端 RTL golden（纯软件量产前验收不依赖 RTL/PE 仿真器）

详见 `09a_lut_coverage_vs_abc.md`、`09b_lut_env_and_remaining.md`。

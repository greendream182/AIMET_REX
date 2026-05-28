# 12 模式间精度对比与 profiler

Status: implemented

## 1. 目标

提供多种执行模式（`fp32_qdq` / `fp16_qdq` / **`fixed_scale_qdq`** / `int16_fixed_eval` / `int16_fixed_qat_sim`）下的逐层与端到端误差对比工具，输出可机读 JSON 报告。同时提供运行时统计 hook，输出 saturation / acc 占用 / multiplier 取值等信息。

**扩展（M2.5）**：必须支持 `fixed_scale_qdq vs fp32_qdq`；`fixed_scale_qdq vs int16_fixed_eval` 为报告型对比（不设等价阈值）。

## 2. 范围

### 2.1 新增文件

- [aimet_torch/fixed_point/metrics/__init__.py](../../aimet_torch/fixed_point/metrics/__init__.py)
- [aimet_torch/fixed_point/metrics/compare.py](../../aimet_torch/fixed_point/metrics/compare.py)
- [aimet_torch/fixed_point/metrics/profiler.py](../../aimet_torch/fixed_point/metrics/profiler.py)
- [examples/compare_quant_execution_modes.py](../../examples/compare_quant_execution_modes.py)

### 2.2 不在范围

- CI 集成（spec 14）。
- 业务模型评估（属于业务方任务）。

## 3. 前置依赖

- spec 01 / 02 / 04 / 06 完成。

## 4. 数据契约

`compare_modes` 返回结构：

```json
{
  "model": "tiny_conv",
  "modes": ["fp32_qdq", "fp16_qdq", "fixed_scale_qdq", "int16_fixed_eval"],
  "global": {
    "fp16_qdq vs fp32_qdq": {"max_abs_error": ..., "rmse": ..., "cosine_similarity": ..., "sqnr_db": ...},
    "fixed_scale_qdq vs fp32_qdq": {...},
    "int16_fixed_eval vs fp32_qdq": {...},
    "fixed_scale_qdq vs int16_fixed_eval": {...}
  },
  "per_layer": {
    "conv1": {"fp16_qdq vs fp32_qdq": {...}, "int16_fixed_eval vs fp32_qdq": {...}},
    ...
  },
  "saturation": {
    "conv1": 0.012,
    "conv2": 0.0,
    ...
  }
}
```

`FixedPointProfiler` JSON 与 INTERFACE.md 第 10 节一致。

## 5. API 签名

```python
# metrics/compare.py
DEFAULT_COMPARE_MODES = (
    ExecutionMode.FP32_QDQ,
    ExecutionMode.FP16_QDQ,
    ExecutionMode.FIXED_SCALE_QDQ,
    ExecutionMode.INT16_FIXED_EVAL,
)

def compare_modes(
    model: torch.nn.Module,
    input_data,
    modes: list | None = None,  # None → DEFAULT_COMPARE_MODES
    metrics: tuple = ("max_abs_error", "rmse", "cosine_similarity", "sqnr_db"),
) -> dict: ...

def write_compare_report(report: dict, output_path: str) -> None: ...
```

```python
# metrics/profiler.py
class FixedPointProfiler:
    def __init__(self, model: torch.nn.Module): ...
    def __enter__(self): ...
    def __exit__(self, exc_type, exc, tb): ...
    def to_dict(self) -> dict: ...
    def to_json(self, path: str) -> None: ...
```

CLI：

**默认模式列表**（`modes=None` 或与 CLI 未传 `--modes` 时）：

```text
fp32_qdq → fp16_qdq → fixed_scale_qdq → int16_fixed_eval
```

第一项为参考基准；`pairwise` 键为 `{mode}_vs_{reference}`（参考为 `fp32_qdq`）。

```bash
python examples/compare_quant_execution_modes.py \
    --model tiny_conv \
    --modes fp32_qdq fp16_qdq fixed_scale_qdq int16_fixed_eval \
    --report output/quant_mode_report.json \
    --per-layer-csv output/per_layer.csv
```

CLI 实现：[`scripts/fixed_point/compare_quant_modes.py`](../../scripts/fixed_point/compare_quant_modes.py)（`dual_linear` / `mock_mobilenet`）。

## 6. 算法 / 伪代码

`compare_modes`：

```text
results = {}
for mode in modes:
    with quant_execution_mode(mode):
        with FixedPointProfiler(model) as prof:
            y, layer_outputs = run_with_layer_hooks(model, input_data)
    results[mode] = (y, layer_outputs, prof.to_dict())

baseline = results["fp32_qdq"]
report = {"global": {}, "per_layer": {}, "saturation": {}}
for mode, (y, lo, prof) in results.items():
    if mode == "fp32_qdq":
        continue
    key = f"{mode} vs fp32_qdq"
    report["global"][key] = compute_metrics(y, baseline[0])
    for name in lo:
        report["per_layer"].setdefault(name, {})[key] = compute_metrics(lo[name], baseline[1][name])
    report["saturation"].update(prof["saturation"])
return report
```

`FixedPointProfiler`：

```text
hooks = []
for module in model.modules():
    if is_int16_kernel_module(module):
        hooks.append(module.register_forward_hook(record_stats))

def record_stats(module, inputs, output):
    self.stats[name(module)] = {
        "saturation_ratio": compute_sat_ratio(output.int_repr),
        "acc_max_abs": ...,
        "multiplier": ...,
        "rshift": ...,
        "bit_utilization": ...,
    }
```

## 7. 实施步骤

1. 实现 `compute_metrics`（max_abs_error / rmse / cosine_similarity / sqnr_db）。
2. 实现 `compare_modes` 与 `write_compare_report`。
3. 实现 `FixedPointProfiler` hook。
4. 写 CLI `examples/compare_quant_execution_modes.py`。
5. 单元测试：合成数据下指标值与手算一致。

## 8. 验收标准

### 8.1 单元测试

```python
def test_compute_metrics_known_values():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([1.0, 2.1, 3.0])
    m = compute_metrics(a, b)
    assert abs(m["max_abs_error"] - 0.1) < 1e-6

def test_compare_modes_returns_required_keys():
    report = compare_modes(model, x)  # default includes fixed_scale_qdq
    assert "fp32_qdq_vs_fixed_scale_qdq" in report["pairwise"]
```

### 8.2 必须通过的现有测试

- 默认 `examples/quick_start.py` 行为不变。

### 8.3 性能阈值

- profiler 引入开销 < 10%。

## 9. 不允许做的事

- 不允许 profiler 修改 forward 数值。
- 不允许 compare 工具自动改变全局执行模式后不恢复（必须使用 context manager）。
- 不允许把 profiler 状态泄漏到下次模型 forward。

## 10. 参考

- INTERFACE.md 第 10 节。
- 顶层文档第 9 节成功标准。

# 14 CI 集成与回归基线

Status: implemented

## 1. 目标

把 spec 13 定义的测试矩阵接入 CI，建立回归基线模型与精度/性能阈值，提供可复现的 nightly 对比报告。

## 2. 范围

### 2.1 新增文件

- [.github/workflows/fixed_point_ci.yml](../../.github/workflows/fixed_point_ci.yml)
- [scripts/fixed_point/run_baseline.py](../../scripts/fixed_point/run_baseline.py)
- [scripts/fixed_point/baseline.json](../../scripts/fixed_point/baseline.json)
- [scripts/fixed_point/check_thresholds.py](../../scripts/fixed_point/check_thresholds.py)

### 2.2 不在范围

- 部署 / 发布流水线。

## 3. 前置依赖

- spec 12 / 13 完成。

## 4. 数据契约

CI 触发规则：

- PR：仅跑 L1 + L2（unit + op）+ tiny e2e；GPU 可选。
- nightly：跑 L1-L4 + 性能基线 + 全模式对比报告。
- release：人工触发，跑全套 + 业务模型。

`baseline.json` 字段：

```json
{
  "model": "tiny_conv",
  "metrics": {
    "fp16_qdq vs fp32_qdq": {"top1_drop_max": 0.005, "cosine_similarity_min": 0.9995},
    "int16_fixed_eval vs fp32_qdq": {"top1_drop_max": 0.01, "cosine_similarity_min": 0.999}
  },
  "perf": {
    "tier_a_max_seconds": 1.0,
    "tier_b_max_seconds": 10.0
  },
  "updated_at": "2026-05-12"
}
```

## 5. API 签名

```python
# scripts/fixed_point/check_thresholds.py
def check_against_baseline(report_path: str, baseline_path: str) -> int:
    """
    返回非零退出码若任一阈值未达标。
    打印逐项对比表。
    """
    ...
```

## 6. 算法 / 伪代码

CI workflow 流程：

```text
- checkout
- setup python + cuda
- install dependencies
- run L1 unit tests
- run L2 op tests
- run tiny e2e (compare_modes)
- run check_thresholds against baseline.json
- upload report.json + per_layer.csv as CI artifact
```

`check_thresholds`：

```text
report = json.load(report_path)
baseline = json.load(baseline_path)
failures = []
for key, expected in baseline["metrics"].items():
    actual = report["global"].get(key, {})
    if actual["cosine_similarity"] < expected["cosine_similarity_min"]:
        failures.append((key, "cosine_similarity", actual, expected))
    ...
if failures:
    print_failures(failures)
    sys.exit(1)
sys.exit(0)
```

## 7. 实施步骤

1. 建立 GitHub Actions workflow：`.github/workflows/fixed_point_ci.yml`。
2. 提供 baseline 初始模型与阈值（先以 tiny_conv 为例，业务模型逐步加入）。
3. nightly job：运行完整 compare 报告并对比 baseline。
4. PR job：运行最小集合 + threshold 检查。
5. 提供失败时下载 artifact 的说明（README）。

## 8. 验收标准

### 8.1 CI 自检

- 工作流在 GitHub Actions 上正确解析与运行。
- baseline 阈值未达标时 PR 阻塞。
- 工作流总耗时 < 30min（非 nightly）。

### 8.2 报告

- 每次 nightly 自动产出 `quant_mode_report.json` 与 `per_layer.csv`，并归档至 artifact。

### 8.3 阈值更新流程

- 任何阈值变更必须 PR 标注 `baseline-update` 并经 owner approve。

## 9. 不允许做的事

- 不允许在 CI 中放宽默认阈值不留 PR 痕迹。
- 不允许 nightly 报告失败被 silent 忽略。
- 不允许在 baseline.json 中存储任何业务敏感信息。

## 10. 参考

- spec 12 compare_modes。
- spec 13 测试矩阵。
- 顶层文档第 9 节成功标准、第 11 节里程碑。

# 13 测试与验收

Status: implemented

## 1. 目标

定义本项目所有 spec 共用的测试矩阵、命名规范、验收阈值与 golden 数据生成方法。

## 2. 范围

### 2.1 新增文件

- [tests/fixed_point/__init__.py](../../tests/fixed_point/__init__.py)
- [tests/fixed_point/test_execution_mode.py](../../tests/fixed_point/test_execution_mode.py)
- [tests/fixed_point/test_int16_tensor.py](../../tests/fixed_point/test_int16_tensor.py)
- [tests/fixed_point/test_requantize.py](../../tests/fixed_point/test_requantize.py)
- [tests/fixed_point/test_registry.py](../../tests/fixed_point/test_registry.py)
- [tests/fixed_point/kernels/test_conv_linear.py](../../tests/fixed_point/kernels/test_conv_linear.py)
- [tests/fixed_point/kernels/test_eltwise.py](../../tests/fixed_point/kernels/test_eltwise.py)
- [tests/fixed_point/kernels/test_pool.py](../../tests/fixed_point/kernels/test_pool.py)
- [tests/fixed_point/kernels/test_lut.py](../../tests/fixed_point/kernels/test_lut.py)
- [tests/fixed_point/offline/test_pipeline.py](../../tests/fixed_point/offline/test_pipeline.py)
- [tests/fixed_point/qat/test_ste_backward.py](../../tests/fixed_point/qat/test_ste_backward.py)
- [tests/fixed_point/end_to_end/test_tiny_models.py](../../tests/fixed_point/end_to_end/test_tiny_models.py)
- [tests/fixed_point/data/](../../tests/fixed_point/data/)：golden npz / json

### 2.2 不在范围

- 业务模型测试。

## 3. 前置依赖

- 各 spec 至少完成接口骨架。

## 4. 数据契约

测试分层：

- L1 unit：单个函数 / 类，纯数据驱动。
- L2 op：单算子 + 真实 encoding，与浮点 reference 对比。
- L3 module：多算子小模型（tiny Conv-BN-ReLU、tiny Linear、tiny Add）。
- L4 e2e：业务对齐 tiny model（含完整 PTQ + freeze + eval）。

阈值统一引用顶层文档第 9 节。

Golden 数据：

- 存储为 `.npz`（numpy）+ `.json`（meta）。
- 用纯整数 reference（参考实现）生成；CI 不允许动态依赖任何浮点 kernel 行为。

## 5. API 签名

测试辅助：

```python
# tests/fixed_point/_helpers.py
def build_int16_tensor(int_list, scale, zp, axis=None) -> Int16QuantizedTensor: ...
def build_input_encoding(scale, zp, qmin=-32768, qmax=32767) -> InputEncoding: ...
def build_output_encoding(scale, zp, multiplier, rshift, qmin=-32768, qmax=32767) -> OutputEncoding: ...
def assert_int_tensor_equal(a, b, msg=""): ...
def assert_close_qdq(a, b, atol=None, rtol=None, sqnr_db_min=None): ...
def reference_int_conv2d(x, w, bias_int32, ...) -> torch.Tensor: ...
def reference_int_linear(x, w, bias_int32) -> torch.Tensor: ...
```

## 6. 算法 / 伪代码

`reference_int_conv2d` 用纯 Python `int` 累加，作为最权威 reference：

```text
for each output position:
    acc = 0
    for each c, kh, kw:
        acc += int(x_centered[..., n, c, h+kh, w+kw]) * int(w_centered[..., oc, c, kh, kw])
    if bias is not None:
        acc += int(bias_int32[oc])
    out[..., oc, h, w] = acc
return tensor(out, dtype=int32)
```

`assert_int_tensor_equal`：

- 同 dtype 同 shape；逐元素相等才通过。
- 不允许 `atol`（整数比较）。

## 7. 实施步骤

1. 建测试目录与 `_helpers.py`。
2. 各 spec 实现完成后，对应 test 文件由该 spec owner 同步落地。
3. golden 数据集中放在 `tests/fixed_point/data/`，命名 `<op>_<case>.npz`。
4. 端到端测试覆盖：
   - tiny Conv-BN-ReLU
   - tiny Linear
   - tiny residual Add
   - QuantGRU 的最小 cell（M5+ 完成）
5. 配置 `pytest` marker：`@pytest.mark.gpu`、`@pytest.mark.slow`、`@pytest.mark.golden`。

## 8. 验收标准

### 8.1 单元测试

- 全部 L1 测试在 PR 提交时强制运行，覆盖率（lines）≥ 85%（仅统计 `aimet_torch/fixed_point/`）。

### 8.2 op / module 测试

- `int16_fixed_eval` 输出与纯 Python int reference bit-exact 相等。
- `fp16_qdq` vs `fp32_qdq` cosine similarity ≥ 0.9995。
- `int16_fixed_eval` vs `fp32_qdq` cosine similarity ≥ 0.999。

### 8.3 ADR-013/014 byte-stream parity（容器迁移守门）

容器自 v2.1 起为 `torch.int32`（ADR-013），但导出/sidecar 仍按 `int16` byte-stream 序列化。强制守门测试：

- `tests/fixed_point/test_golden_data.py`：committed golden `.npz` 仍保留 v2.0 int16 byte-stream；运行时输出经 `.to(torch.int16)` 截位后必须与 golden bit-equal。
- `tests/fixed_point/test_sim_tensor_dtype_guard.py`：源码级 grep 守门，禁止 kernels 包内出现以下反模式：
  - `F.unfold(...)` 调用（用 `kernels._im2col.im2col_int` 替代）。
  - `saturate_int16(...)` 调用（用 `saturate_sim_tensor` 替代）。
  - `int_repr.to(torch.int16)`、`int_repr=...dtype=torch.int16`（容器固定为 `SIM_TENSOR_DTYPE`）。
- `tests/fixed_point/test_int16_tensor.py::test_kernels_emit_sim_tensor_dtype`：抽样多种 kernel（ReLU / Add / MaxPool / Identity），运行时输出 `int_repr.dtype` 必须为 `SIM_TENSOR_DTYPE`。

### 8.3 e2e

- tiny model top1 跌落满足顶层文档第 9 节阈值。

### 8.4 性能

- 全测试套（不含 slow）在 CI 单卡 GPU 上 < 15min。

## 9. 不允许做的事

- 不允许测试中调用 `to_float()` 进行整数计算的等价比较（除非作为 metric 使用）。
- 不允许跳过 golden 数据校验。
- 不允许在 CI 中无声 retry。

## 10. 参考

- 顶层文档第 9 节成功标准、第 11 节里程碑。
- spec 12 metrics 定义。

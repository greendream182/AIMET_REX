# 01 执行模式 API

Status: implemented

## 1. 目标

为 AIMET RX 提供统一的 execution mode API，使 `fp32_qdq` / `fp16_qdq` / `fixed_scale_qdq` / `int16_fixed_eval` / `int16_fixed_qat_sim` 可以通过一套显式接口切换，且默认行为与现状完全一致。

> **扩展**：`fixed_scale_qdq` 的语义与实现见 [15_fixed_scale_qdq.md](15_fixed_scale_qdq.md)；本 spec 仅要求枚举与环境变量支持该取值。

## 2. 范围

### 2.1 新增文件

- [aimet_torch/fixed_point/__init__.py](../../aimet_torch/fixed_point/__init__.py)
- [aimet_torch/fixed_point/execution_mode.py](../../aimet_torch/fixed_point/execution_mode.py)

### 2.2 修改文件

- [aimet_torch/__init__.py](../../aimet_torch/__init__.py)：导出 `set_quant_execution_mode` / `quant_execution_mode` / `ExecutionMode`。
- [aimet_torch/v2/quantization/affine/quantizer.py](../../aimet_torch/v2/quantization/affine/quantizer.py)：`QuantizeDequantize.forward()` 读取当前 mode 并分派。
- [aimet_torch/v1/qc_quantize_op.py](../../aimet_torch/v1/qc_quantize_op.py)：wrapper forward 同理。

### 2.3 不在本 spec 范围内

- 实际 fp16 / int16 路径实现（见 spec 02 / 03 / 04+）。
- encoding sidecar 扩展（见 spec 04）。
- profile 与对比工具（见 spec 12）。

## 3. 前置依赖

无。这是 M1 第一份 spec。

## 4. 数据契约

- `set_quant_execution_mode(mode)` 必须线程安全，使用 `threading.RLock`。
- `quant_execution_mode(mode)` 必须支持嵌套（保存/恢复栈）。
- 进程启动时若环境变量 `AIMET_RX_QUANT_EXECUTION_MODE` 存在则覆盖默认。
- 默认值 `ExecutionMode.FP32_QDQ`。

## 5. API 签名

```python
# aimet_torch/fixed_point/execution_mode.py
from contextlib import contextmanager
from enum import Enum
from typing import Iterator, Union

class ExecutionMode(str, Enum):
    FP32_QDQ = "fp32_qdq"
    FP16_QDQ = "fp16_qdq"
    FIXED_SCALE_QDQ = "fixed_scale_qdq"
    INT16_FIXED_EVAL = "int16_fixed_eval"
    INT16_FIXED_QAT_SIM = "int16_fixed_qat_sim"

ModeLike = Union[ExecutionMode, str]

def set_quant_execution_mode(mode: ModeLike) -> None: ...
def get_quant_execution_mode() -> ExecutionMode: ...

@contextmanager
def quant_execution_mode(mode: ModeLike) -> Iterator[ExecutionMode]: ...
```

## 6. 算法 / 伪代码

```text
state = thread_local()
state.stack = []
state.current = ExecutionMode(env_or_default())

def set_quant_execution_mode(mode):
    with lock:
        state.current = ExecutionMode(mode)

def get_quant_execution_mode():
    with lock:
        return state.current

@contextmanager
def quant_execution_mode(mode):
    prev = get_quant_execution_mode()
    set_quant_execution_mode(mode)
    try:
        yield get_quant_execution_mode()
    finally:
        set_quant_execution_mode(prev)
```

错误处理：

- 未知字符串 → `ValueError`，错误信息列出有效值。
- 非 str / ExecutionMode 类型 → `TypeError`。

## 7. 实施步骤

1. 新增 `aimet_torch/fixed_point/__init__.py`，从 `execution_mode` 导出三个公共符号。
2. 新增 `execution_mode.py` 实现枚举、全局状态、context manager，遵从 INTERFACE.md 第 3 节。
3. 在 `aimet_torch/__init__.py` 顶层导出符号，使 `import aimet_torch as a; a.set_quant_execution_mode("fp16_qdq")` 可用。
4. v2 `QuantizeDequantize.forward()` 添加最小骨架：`mode = get_quant_execution_mode()`，并在 `mode != FP32_QDQ` 时 `NotImplementedError`（占位，由后续 spec 替换）。
5. v1 wrapper forward 同步加占位。
6. 编写单元测试 `tests/fixed_point/test_execution_mode.py`，覆盖默认值、字符串/枚举混用、context manager 嵌套、环境变量覆盖、未知值报错、线程安全。

## 8. 验收标准

### 8.1 单元测试

```python
def test_default_mode():
    assert get_quant_execution_mode() is ExecutionMode.FP32_QDQ

def test_set_with_str():
    set_quant_execution_mode("fp16_qdq")
    assert get_quant_execution_mode() is ExecutionMode.FP16_QDQ
    set_quant_execution_mode(ExecutionMode.FP32_QDQ)

def test_context_manager_nested():
    with quant_execution_mode("fp16_qdq"):
        assert get_quant_execution_mode() is ExecutionMode.FP16_QDQ
        with quant_execution_mode("int16_fixed_eval"):
            assert get_quant_execution_mode() is ExecutionMode.INT16_FIXED_EVAL
        assert get_quant_execution_mode() is ExecutionMode.FP16_QDQ
    assert get_quant_execution_mode() is ExecutionMode.FP32_QDQ

def test_unknown_value_raises():
    with pytest.raises(ValueError):
        set_quant_execution_mode("int8_fixed")
```

### 8.2 必须通过的现有测试

- `tests/` 目录下默认 `fp32_qdq` 路径全部通过；任何回归视为 P0 阻塞。

### 8.3 性能阈值

- `get_quant_execution_mode()` 调用开销 < 1us（每 forward 调用次数有限，可粗测）。

## 9. 不允许做的事

- 不允许在本 spec 内修改任何 QDQ 数学逻辑。
- 不允许默认开启 `fp16_qdq` 或 `int16_fixed_*`。
- 不允许把 mode 状态挂到 `QuantizationSimModel` 实例上（必须是进程级）。

## 10. 参考

- 顶层文档第 4-5 节系统分层与状态机。
- INTERFACE.md 第 3 节。

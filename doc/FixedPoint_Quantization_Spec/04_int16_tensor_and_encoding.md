# 04 Int16QuantizedTensor 与 encoding 扩展

Status: implemented

> **命名说明**：`Int16QuantizedTensor` 为历史类名，表示 **G3 整数仿真段载体**，**不是**「语义 bitwidth 必为 INT16 / 全网 U16」。语义量化域由 `qmin`/`qmax` 决定。推荐新文档使用别名 `FixedPointSimTensor`（同一类型）。见 [Design v2 §3.7](../FixedPoint_Quantization_Design_v2.md#37-g2-与-g3-的-encoding-分工避免误读-adr-009)。
>
> **容器 dtype 更新（ADR-013，v2.1）**：`int_repr` 容器自 v2.1 起统一为 **`torch.int32`**（`aimet_torch.fixed_point.requantize.SIM_TENSOR_DTYPE`），值域仍受 `(qmin, qmax)` 与 `bitwidth` 约束（默认 INT16：`qmin=-32768, qmax=32767`）。**INT16 仅指值域**而**非容器**（ADR-014）。byte-stream / sidecar / ONNX 导出仍按 `int16` 截位序列化，硬件契约 byte-equal。`__post_init__` 会自动把传入的 `torch.int16` 输入升格到 `SIM_TENSOR_DTYPE` 以兼容旧调用方；新代码请直接构造 `SIM_TENSOR_DTYPE` 容器。

## 1. 目标

为 **硬件整数仿真路径**（`int16_fixed_eval` / `int16_fixed_qat_sim`）提供段间张量载体与 encoding sidecar 扩展：承载 `int_repr`、边界 `scale`/`zero_point`、以及层间离线生成的 `multiplier` / `rshift`。

## 2. 范围

### 2.1 新增文件

- [aimet_torch/fixed_point/tensor.py](../../aimet_torch/fixed_point/tensor.py)
- [aimet_torch/fixed_point/encoding.py](../../aimet_torch/fixed_point/encoding.py)

### 2.2 修改文件

- [aimet_torch/fixed_point/__init__.py](../../aimet_torch/fixed_point/__init__.py)：导出 `Int16QuantizedTensor` / `OutputEncoding` / `InputEncoding`。
- [aimet_torch/v2/quantization/affine/encoding.py](../../aimet_torch/v2/quantization/affine/encoding.py)：扩展 encoding 序列化，写入 `multiplier` / `rshift` / `bias_int32_path` / `lut_path`。

### 2.3 不在范围

- requantize_int 实现（见 spec 05）。
- multiplier / bias / LUT 离线生成实现（见 spec 10）。
- 现有 ONNX 导出不动。

## 3. 前置依赖

- spec 01 完成。

## 4. 数据契约

### 4.1 `Int16QuantizedTensor`（别名 `FixedPointSimTensor`）

字段约束（与 INTERFACE.md 第 4 节一致）：

- `int_repr.dtype is SIM_TENSOR_DTYPE`（v2.1 起为 `torch.int32`；ADR-013/014）；值域见 `qmin`/`qmax`，sidecar/导出按 `int16` byte-stream 截位
- `zero_point.dtype == torch.int32`
- `scale.dtype == torch.float32`，仅离线计算 multiplier 时使用
- `axis is None` → per-tensor；非 None → per-channel，且 `scale.shape[0] == int_repr.shape[axis]`
- 实例不可变；变换返回新实例

### 4.2 Encoding 扩展

新字段：

- `multiplier` int16 张量
- `rshift` int8 张量
- `bias_int32_path` 可选字符串（外部 sidecar）
- `lut_path` 可选字符串（外部 sidecar）

向后兼容：

- 旧 encodings 文件无新字段时按 `multiplier=None, rshift=None` 加载，运行时若需 INT16 路径但未提供则抛 `ValueError`。

## 5. API 签名

```python
# aimet_torch/fixed_point/tensor.py
import torch
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class Int16QuantizedTensor:
    int_repr: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor
    qmin: int = -32768
    qmax: int = 32767
    axis: Optional[int] = None

    @classmethod
    def from_float(
        cls,
        tensor: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        axis: Optional[int] = None,
    ) -> "Int16QuantizedTensor": ...

    def to_float(self, debug_dtype: torch.dtype = torch.float32) -> torch.Tensor: ...
    def quantized_repr(self) -> torch.Tensor: ...
    def saturate(self) -> "Int16QuantizedTensor": ...
    def to(self, device: torch.device) -> "Int16QuantizedTensor": ...
```

```python
# aimet_torch/fixed_point/encoding.py
@dataclass(frozen=True)
class InputEncoding:
    scale: torch.Tensor
    zero_point: torch.Tensor
    qmin: int
    qmax: int
    axis: Optional[int] = None

@dataclass(frozen=True)
class OutputEncoding(InputEncoding):
    multiplier: torch.Tensor       # int16, Q15
    rshift: torch.Tensor           # int8, 非负
```

## 6. 算法 / 伪代码

`from_float`：

```text
y_round = round(tensor / scale) + zero_point
y_clamped = clamp(y_round, qmin, qmax)
# ADR-013: 容器 SIM_TENSOR_DTYPE (torch.int32)；值域受 qmin/qmax 约束
return FixedPointSimTensor(int_repr=saturate_sim_tensor(y_round, qmin, qmax),
                           scale, zero_point, ...)
```

`to_float`：

```text
return ((int_repr.to(int32) - zero_point).to(debug_dtype)) * scale.to(debug_dtype)
```

注意 `to_float` 仅供 metrics / 调试使用。`int16_fixed_eval` 模式下若被 forward 调用应通过 profiler 报警。

`saturate`：

```text
return Int16QuantizedTensor(int_repr.clamp(qmin, qmax), ...)
```

边界：

- 输入 tensor 含 NaN：`from_float` 抛 `ValueError`。
- scale 含 0：抛 `ValueError`。
- per-channel 时 zero_point 形状必须广播到 scale。

## 7. 实施步骤

1. 新增 `tensor.py`，使用 `frozen=True` dataclass + 自定义 `__post_init__` 校验。
2. 新增 `encoding.py`，`InputEncoding` / `OutputEncoding` dataclass。
3. 实现 `from_float` / `to_float` / `saturate` / `to`。
4. 在 `aimet_torch/v2/quantization/affine/encoding.py` 增加序列化字段：
   - `to_dict()` 写入 `multiplier_uint16`, `rshift_int8`, `bias_int32_path`, `lut_path`。
   - `from_dict()` 读取并填充对应字段，缺失时为 None。
5. 单元测试覆盖 dtype / shape / 不可变性 / 序列化兼容。

## 8. 验收标准

### 8.1 单元测试

```python
def test_int16_tensor_from_float_per_tensor():
    x = torch.tensor([0.0, 0.5, 1.0])
    s = torch.tensor(0.01)
    zp = torch.tensor(0, dtype=torch.int32)
    q = FixedPointSimTensor.from_float(x, s, zp)
    assert q.int_repr.dtype is SIM_TENSOR_DTYPE  # torch.int32 (ADR-013)
    assert q.int_repr.tolist() == [0, 50, 100]

def test_int16_tensor_saturate():
    x = torch.tensor([40000, -40000, 100], dtype=SIM_TENSOR_DTYPE)
    q = FixedPointSimTensor(int_repr=x, ...).saturate()
    assert q.int_repr.tolist() == [32767, -32768, 100]

def test_encoding_backward_compatible_load():
    enc_dict = {"scale": 0.01, "offset": 0, "qmin": 0, "qmax": 255}
    enc = OutputEncoding.from_dict(enc_dict)
    assert enc.multiplier is None
```

### 8.2 必须通过的现有测试

- 现有 encoding 序列化测试全部通过；旧 encodings 文件 round-trip 无变化。

## 9. 不允许做的事

- 不允许在 `int_repr` 上挂 `requires_grad=True`（整数张量本身不支持）。
- 不允许在 `int16_fixed_eval` runtime kernel 中调用 `to_float`。
- 不允许将 `multiplier` 存储为 float（必须 int16）。

## 10. 参考

- INTERFACE.md 第 4 节、第 6.3 节。
- 顶层文档第 2.2 节符号表。

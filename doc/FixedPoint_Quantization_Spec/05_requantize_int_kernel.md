# 05 requantize_int 与饱和工具

Status: implemented

> **硬件对齐（ADR-015）**：默认 `requantize_int` 在 int64 上完成 ``acc * multiplier`` 再移位（保 e2e）。严格仿真：
> - `AIMET_RX_REQUANTIZE_INT32_SAT=1` 或 `AIMET_RX_HW_REF=1` / `AIMET_RX_PWL_HW_REF=1`：乘后 `saturate_int32`。
> - `AIMET_RX_ACC_INT32_SAT=1` 或 `AIMET_RX_HW_REF=1`：Conv/Linear MAC 累加器在 `requantize_int` 前 `saturate_mac_accumulator`（spec 07）。
> - `RoundingMode.HALF_UP`：LUT §3.0 网格对齐在 HW_REF 下使用（见 `align_op_quant_grid_to_lut_quant_grid`）。

## 1. 目标

提供 `int16_fixed` 全链路所需的核心整数 requantize 与饱和工具，确保运行时无浮点参与。

## 2. 范围

### 2.1 新增文件

- [aimet_torch/fixed_point/requantize.py](../../aimet_torch/fixed_point/requantize.py)
- [aimet_torch/fixed_point/rounding.py](../../aimet_torch/fixed_point/rounding.py)

### 2.2 修改文件

- [aimet_torch/fixed_point/__init__.py](../../aimet_torch/fixed_point/__init__.py)：导出 `requantize_int` / `saturate_int32` / `saturate_sim_tensor` / `saturate_to_range` / `SIM_TENSOR_DTYPE` / `RoundingMode`。`saturate_int16` 自 v2.1 起为兼容 alias（一个 release 后退役）。

### 2.3 不在范围

- 算子 kernel 实现（见 spec 07 / 08）。
- multiplier 离线生成（见 spec 10）。

## 3. 前置依赖

- spec 04 完成。

## 4. 数据契约

输入：

- `acc.dtype == torch.int32`
- `multiplier.dtype == torch.uint16`，`0 <= multiplier <= 65535`
- `rshift.dtype == torch.int8`，`0 <= rshift <= 31`
- `y_zp.dtype == torch.int32`
- `qmin`, `qmax`：int

输出：

- `dtype is SIM_TENSOR_DTYPE`（v2.1 起为 `torch.int32`；ADR-013/014）
- 数值范围严格在 `[qmin, qmax]` 内（INT16 / U8 / U16 等仅由 `(qmin, qmax)` 表达）
- byte-stream 导出仍按 `int16` 截位序列化（与硬件契约 byte-equal）

不变式：

- 中间乘法使用 `int64` 防溢出
- 严禁出现任何 `float` 中间张量
- 舍入策略由 `RoundingMode` 决定，默认 `HALF_TO_EVEN`
- 输入 broadcast：multiplier / rshift / y_zp 支持 per-tensor 与 per-channel 沿指定轴广播

## 5. API 签名

```python
# aimet_torch/fixed_point/rounding.py
from enum import Enum

class RoundingMode(str, Enum):
    HALF_TO_EVEN = "half_to_even"
    HALF_AWAY_FROM_ZERO = "half_away_from_zero"
    TRUNCATE = "truncate"
```

```python
# aimet_torch/fixed_point/requantize.py
import torch
from .rounding import RoundingMode

INT16_QMIN = -32768
INT16_QMAX = 32767

def requantize_int(
    acc: torch.Tensor,
    multiplier: torch.Tensor,
    rshift: torch.Tensor,
    y_zp: torch.Tensor,
    qmin: int = INT16_QMIN,
    qmax: int = INT16_QMAX,
    rounding_mode: RoundingMode = RoundingMode.HALF_TO_EVEN,
) -> torch.Tensor: ...

def saturate_int16(x: torch.Tensor) -> torch.Tensor: ...
def saturate_to_range(x: torch.Tensor, qmin: int, qmax: int) -> torch.Tensor: ...

def round_shift(
    x: torch.Tensor,             # int64
    rshift: torch.Tensor,        # int8
    rounding_mode: RoundingMode,
) -> torch.Tensor: ...           # int64
```

## 6. 算法 / 伪代码

```text
def requantize_int(acc, multiplier, rshift, y_zp, qmin, qmax, mode):
    assert acc.dtype == torch.int32
    assert multiplier.dtype == torch.uint16
    assert rshift.dtype == torch.int8

    prod = acc.to(torch.int64) * multiplier.to(torch.int64)
    rounded = round_shift(prod, rshift, mode)
    y = rounded + y_zp.to(torch.int64)
    # ADR-013/014: 容器为 SIM_TENSOR_DTYPE (torch.int32)，值域受 [qmin, qmax] 约束
    return saturate_sim_tensor(y, qmin, qmax)
```

```text
def round_shift(x, rshift, mode):
    if mode == TRUNCATE:
        return x >> rshift

    if mode == HALF_AWAY_FROM_ZERO:
        offset = (1 << (rshift - 1))
        # signed: 对负数 x，offset 取负
        offset_signed = where(x >= 0, offset, -offset)
        return (x + offset_signed) >> rshift

    if mode == HALF_TO_EVEN:
        truncated = x >> rshift
        remainder = x & ((1 << rshift) - 1)
        half = 1 << (rshift - 1)
        # remainder > half -> +1
        # remainder == half and truncated 奇数 -> +1
        bump = (remainder > half) | ((remainder == half) & ((truncated & 1) == 1))
        return truncated + bump.to(torch.int64)
```

边界：

- `rshift == 0`：直接返回 `x + y_zp`，不做位移与舍入。
- `multiplier == 0`：结果恒为 `y_zp`（saturate 后）。
- `acc` 接近 int32 上下限：检查 `prod` 不溢出 int64（`abs(acc) * 32767 < 2^63` 恒成立）。

## 7. 实施步骤

1. 新增 `rounding.py`，实现三种舍入。
2. 新增 `requantize.py`，实现 `requantize_int` / `saturate_*` / `round_shift`。
3. 在 `__init__.py` 导出公共符号。
4. 单元测试 `tests/fixed_point/test_requantize.py`，覆盖：
   - 三种舍入模式数值用例
   - 正负 acc
   - per-tensor / per-channel multiplier 广播
   - 饱和上下界
   - rshift=0 / rshift=31 边界
   - dtype 错误抛 `TypeError`

## 8. 验收标准

### 8.1 单元测试

```python
def test_requantize_basic_half_to_even():
    acc = torch.tensor([100, -100, 32766], dtype=torch.int32)
    mul = torch.tensor(16384, dtype=torch.int16)   # 0.5 * 2^15
    rsh = torch.tensor(15, dtype=torch.int8)
    zp = torch.tensor(0, dtype=torch.int32)
    y = requantize_int(acc, mul, rsh, zp, -32768, 32767)
    assert y.dtype is SIM_TENSOR_DTYPE  # torch.int32 (ADR-013)
    assert y.tolist() == [50, -50, 16383]

def test_requantize_saturation():
    acc = torch.tensor([2_000_000_000], dtype=torch.int32)
    mul = torch.tensor(32767, dtype=torch.int16)
    rsh = torch.tensor(15, dtype=torch.int8)
    zp = torch.tensor(0, dtype=torch.int32)
    y = requantize_int(acc, mul, rsh, zp, -32768, 32767)
    assert y.item() == 32767  # 饱和

def test_requantize_per_channel():
    acc = torch.zeros(4, dtype=torch.int32)
    mul = torch.tensor([16384, 8192, 4096, 0], dtype=torch.int16)
    rsh = torch.tensor([15, 15, 15, 15], dtype=torch.int8)
    zp = torch.tensor([10, 20, 30, 40], dtype=torch.int32)
    y = requantize_int(acc, mul, rsh, zp, -32768, 32767)
    assert y.tolist() == [10, 20, 30, 40]
```

### 8.2 必须通过的现有测试

无（新文件）。

### 8.3 性能阈值

- 1M 元素 requantize 在 GPU 上 < 5ms（reference 实现）。

## 9. 不允许做的事

- 不允许使用 `float` / `double` 中间张量。
- 不允许使用 `numpy` 在运行时计算（仅测试可用作 reference）。
- 不允许使用 Python `for` 循环逐元素处理（必须 vectorized）。
- 不允许在 saturate 后再做未受控的 cast。

## 10. 参考

- INTERFACE.md 第 5 节。
- TFLite reference: `MultiplyByQuantizedMultiplier`。
- `quant-gru-pytorch-main/include/quantize_ops_helper.h` 中的 `rshift_round`。

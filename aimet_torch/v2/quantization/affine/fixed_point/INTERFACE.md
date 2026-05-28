# Fixed Point 模块公共接口契约（INTERFACE.md）

本文件是 `aimet_torch/fixed_point/...` 与 `aimet_torch/v2/quantization/affine/fixed_point/...` 全部新增 public 符号的**单一事实源**。

任何 spec、代码实现、测试都必须以本文件为准。如发现 spec 与本文件冲突，以本文件为准；变更接口必须先改本文件并通知所有 spec 维护者。

未列入本文件的符号视为 private，禁止跨模块依赖。

---

## 1. 包结构

```text
aimet_torch/fixed_point/
    __init__.py              # 导出公共符号
    execution_mode.py        # ExecutionMode 与全局状态
    tensor.py                # Int16QuantizedTensor / FixedPointSimTensor
    boundary_quantize.py     # quantize_boundary_from_affine (G3 边界 Q)
    requantize.py            # requantize_int + saturate
    rounding.py              # 舍入策略常量与函数
    registry.py              # FixedKernel 协议与 registry
    kernels/
        __init__.py
        conv_linear.py
        eltwise.py
        pool.py
        shape_ops.py
        lut.py
    offline/
        __init__.py
        multiplier.py        # quantize_multiplier
        bias.py              # quantize_bias_int32
        lut_gen.py           # generate_lut_int16
        scale_fixed.py       # quantize_scale_to_m_rshift, convert_encodings_to_fixed_scale
        pipeline.py          # freeze_int16_fixed
    metrics/
        __init__.py
        profiler.py          # FixedPointProfiler
        compare.py           # mode 间对比工具
    export/
        __init__.py
        v2_collect.py        # 从 v2 QuantizationMixin 推导 OutputEncoding / PWL
        sidecar.py           # INT16 sidecar JSON 导出 / 加载 / 一致性校验
```

v2 适配层位于：

```text
aimet_torch/v2/quantization/affine/fixed_point/
    __init__.py
    adapter.py               # v2 quantizer / module 与 fixed_point 的桥接
```

v1 适配层位于：

```text
aimet_torch/v1/fixed_point_adapter.py
```

---

## 2. 枚举与常量

### 2.1 `ExecutionMode`

```python
from enum import Enum

class ExecutionMode(str, Enum):
    FP32_QDQ = "fp32_qdq"
    FP16_QDQ = "fp16_qdq"
    FIXED_SCALE_QDQ = "fixed_scale_qdq"
    INT16_FIXED_EVAL = "int16_fixed_eval"
    INT16_FIXED_QAT_SIM = "int16_fixed_qat_sim"
```

不变式：

- 字符串值与枚举值一一对应，对外 API 同时接受字符串和枚举。
- 默认值 `ExecutionMode.FP32_QDQ`。
- 任何未知字符串必须抛 `ValueError`。

### 2.2 `RoundingMode`

```python
class RoundingMode(str, Enum):
    HALF_TO_EVEN = "half_to_even"        # 默认
    HALF_AWAY_FROM_ZERO = "half_away_from_zero"
    TRUNCATE = "truncate"
```

默认 `HALF_TO_EVEN`，由 ADR-008 规定。

### 2.3 数据类型常量

```python
INT16_QMIN: int = -32768
INT16_QMAX: int = 32767
INT32_QMIN: int = -2_147_483_648
INT32_QMAX: int = 2_147_483_647
MULTIPLIER_QBITS: int = 15            # Q15 multiplier
MULTIPLIER_MAX: int = (1 << MULTIPLIER_QBITS) - 1   # 32767

# ADR-013/014: 仿真张量统一容器 dtype；INT16/U8/U16 仅指值域，由 (qmin, qmax) 表达
SIM_TENSOR_DTYPE: torch.dtype = torch.int32  # in aimet_torch.fixed_point.requantize
```

---

## 3. 全局执行模式

模块路径：`aimet_torch.fixed_point.execution_mode`

### 3.1 全局开关

```python
def set_quant_execution_mode(mode: ExecutionMode | str) -> None: ...
def get_quant_execution_mode() -> ExecutionMode: ...
```

### 3.2 上下文管理器

```python
from contextlib import contextmanager

@contextmanager
def quant_execution_mode(mode: ExecutionMode | str) -> Iterator[ExecutionMode]: ...
```

### 3.3 环境变量

```text
AIMET_RX_QUANT_EXECUTION_MODE=fp32_qdq | fp16_qdq | fixed_scale_qdq | int16_fixed_eval | int16_fixed_qat_sim
```

进程启动时若环境变量存在则覆盖默认值。

**LUT / sidecar（INT16 非线性）** — 详见 `doc/FixedPoint_Quantization_Spec/09b_lut_env_and_remaining.md`：

```text
AIMET_RX_INT16_SIDECAR_PATH=<model.int16.json>   # 冻结 PWL/CLZ，跳过在线拟合
AIMET_RX_ABC_LUT_ROOT=<abc_lut-shuai>             # CLZ 离线拟合
AIMET_RX_BAKE_PWL_SCALE=1                         # §3.0 烘焙进 PWL
AIMET_RX_HW_REF=1 | AIMET_RX_PWL_HW_REF=1         # 严格 abc/PE tap 点
AIMET_RX_REQUIRE_CLZ_LUT=1                        # CLZ 拟合失败即报错
```

**CLZ 负输入（相对 abc golden）**：`power_2` 在负 `q` 上按 `|x|²` 推理（abc 正域 LUT 对负 `q` 常为 0）；`reciprocal` 在负 `q` 上绕 `output_zp` 取反（abc 常饱和到 `out_qmax`）。正域 `q` 仍与 abc 对齐（容差见 `09c`）。

不变式：

- 模式切换是进程级全局状态，但读写需线程安全（`threading.local` 或 `RLock`）。
- 同一进程内多个 `QuantizationSimModel` 共享当前模式。
- DataParallel/DDP 下，模式必须在 forward 之前在每个 rank 上一致设置。
- `FIXED_SCALE_QDQ`：Q/DQ 仅用 `(m_int16, rshift)`（可从已校准 float `scale` 即时生成，或经 `convert_encodings_to_fixed_scale` 缓存）。**Sidecar 加载**缺 `m_int16`/`rshift` 时抛 `ValueError`；禁止在 Q/DQ 路径用 float `scale` 作乘子。

---

## 4. `FixedScaleEncoding` 与定点 scale

模块路径：`aimet_torch.fixed_point.encoding`（或独立 `fixed_scale_encoding.py`）

用于 **QDQ 主路径**（`ExecutionMode.FIXED_SCALE_QDQ`）。Ada200 规范：**任意**合法 `(M, rshift)`，不强制 Power-of-2 scale。

```python
from dataclasses import dataclass
from typing import Optional
import torch

@dataclass(frozen=True)
class FixedScaleEncoding:
  m_int16: torch.Tensor       # dtype=torch.int16
  rshift: torch.Tensor        # dtype=torch.int8, 非负
  zero_point: torch.Tensor    # dtype=torch.int32
  qmin: int
  qmax: int
  axis: Optional[int] = None
  scale_fp_legacy: Optional[torch.Tensor] = None  # 只读；运行时禁止用于 Q/DQ
```

不变式：

- \(\text{scale} \approx m\_\text{int16} / 2^{\text{rshift}}\)（离线生成；`scale_fp_legacy` 仅作对照）。
- `0 <= m_int16 <= 32767`（有符号 INT16 时允许负 M，按硬件文档；默认非负 scale 对应非负 M）。
- `0 <= rshift <= 31`。
- per-channel 时 `m_int16.shape[0] == rshift.shape[0]` 且与 `zero_point` 广播轴一致。

---

## 5. `fixed_scale` Q/DQ 算子

模块路径：`aimet_torch.fixed_point.fixed_scale_qdq`

仅在 `FIXED_SCALE_QDQ` 下由 v2/v1 adapter 调用。中间算子仍为 float（Op_float）。

```python
def quantize_with_fixed_scale(
    tensor: torch.Tensor,
    encoding: FixedScaleEncoding,
    rounding_mode: RoundingMode = RoundingMode.HALF_TO_EVEN,
) -> torch.Tensor:
  """
  q = clamp(round(tensor * 2^rshift / m_int16) - zero_point, qmin, qmax)
  返回 float 载体（与 AIMET QDQ 一致），值为整数网格。
  禁止读取 encoding.scale_fp_legacy。
  """
  ...

def dequantize_with_fixed_scale(
    tensor: torch.Tensor,
    encoding: FixedScaleEncoding,
) -> torch.Tensor:
  """return (tensor + zero_point) * m_int16 / 2^rshift"""
  ...

def quantize_dequantize_with_fixed_scale(
    tensor: torch.Tensor,
    encoding: FixedScaleEncoding,
    rounding_mode: RoundingMode = RoundingMode.HALF_TO_EVEN,
) -> torch.Tensor:
  """STE 反向与 torch_builtins.QuantDequantFunc 一致。"""
  ...
```

### 5.1 离线 scale → (M, rshift)

模块路径：`aimet_torch.fixed_point.offline.scale_fixed`（或 `multiplier.quantize_scale_to_m_rshift` 别名）

```python
def quantize_scale_to_m_rshift(
    scale: float | torch.Tensor,
    multiplier_bits: int = MULTIPLIER_QBITS,
    max_rshift: int = 31,
) -> tuple[torch.Tensor, torch.Tensor]:
  """
  对正 scale 张量逐元素生成 (m_int16, rshift_int8)。
  算法与 quantize_multiplier 相同（frexp + Q15）。
  scale==0 → (0, 0)。
  """
  ...

def convert_encodings_to_fixed_scale(sim) -> int:
  """遍历 sim 上已初始化 quantizer，缓存 FixedScaleEncoding；返回处理数量。"""
  ...
```

### 5.2 G3 边界量化

模块路径：`aimet_torch.fixed_point.boundary_quantize`

```python
def quantize_boundary_from_affine(tensor: torch.Tensor, encoding: AffineEncoding) -> Int16QuantizedTensor:
    """
    G3 入口/权重量化：默认 float scale 网格；
    若 encoding 已 convert 或 AIMET_RX_INT16_BOUNDARY_USE_M_R=1，则用 (M,r) 与 fixed_scale_qdq 对齐。
    """

FixedPointSimTensor = Int16QuantizedTensor  # 推荐别名，见 Design v2 §3.7
```

---

## 6. `Int16QuantizedTensor`（别名 `FixedPointSimTensor`）

模块路径：`aimet_torch.fixed_point.tensor`

> **命名**：历史类名 `Int16QuantizedTensor` 表示 **G3 整数仿真段载体**，**不表示**语义 bitwidth 必为 16；`qmin`/`qmax` 由 quantizer 决定（可为 U8 等）。
>
> **容器 dtype（ADR-013/014, v2.1）**：`int_repr` 容器统一为 `SIM_TENSOR_DTYPE`（`torch.int32`），值域受 `(qmin, qmax)` 与 `bitwidth` 约束；INT16 仅指值域非容器。`__post_init__` 自动把 `torch.int16` 输入升格到 `SIM_TENSOR_DTYPE`（向后兼容窗口）。byte-stream/sidecar 导出仍按 int16 截位。

```python
import torch
from typing import Optional
from aimet_torch.fixed_point.requantize import SIM_TENSOR_DTYPE  # = torch.int32

class Int16QuantizedTensor:
    int_repr: torch.Tensor         # dtype is SIM_TENSOR_DTYPE (torch.int32)
    scale: torch.Tensor            # 边界/对齐元数据，dtype=torch.float32
    zero_point: torch.Tensor       # dtype=torch.int32
    qmin: int                      # 语义网格下界（可非 INT16 范围）
    qmax: int                      # 语义网格上界
    axis: Optional[int]            # per-channel 时为通道轴

    @classmethod
    def from_float(...) -> "Int16QuantizedTensor": ...

    @classmethod
    def from_affine_encoding(cls, tensor, encoding: AffineEncoding) -> "Int16QuantizedTensor": ...

    @classmethod
    def from_fixed_scale_encoding(cls, tensor, encoding: FixedScaleEncoding) -> "Int16QuantizedTensor": ...

    def to_float(self, debug_dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """仅供 debug / metrics；int16_fixed_eval 下受 profiler / 环境变量约束。"""

    def quantized_repr(self) -> torch.Tensor: ...
    def saturate(self) -> "Int16QuantizedTensor": ...

FixedPointSimTensor = Int16QuantizedTensor
```

不变式：

- `int_repr.dtype is SIM_TENSOR_DTYPE`（v2.1 起为 `torch.int32`；旧 `torch.int16` 输入会被 `__post_init__` 自动升格，禁止手动构造其他整数容器）
- `int_repr.device == zero_point.device`
- 当 `axis is not None` 时，`scale.shape[0] == int_repr.shape[axis]`
- 任何调用 `to_float` 的代码路径在 `int16_fixed_eval` 模式下必须被 `FixedPointProfiler` 捕获并报警。
- 实例不可变（行为类似 frozen dataclass）；变换返回新实例。
- 禁止对 carrier 裸用 `+`/`-`/`*`；须走 `QuantizedAdd` 等模块（见 `tensor.py`）。

---

## 7. `requantize_int` 与饱和工具

模块路径：`aimet_torch.fixed_point.requantize`

### 7.1 主接口

```python
def requantize_int(
    acc: torch.Tensor,            # dtype=torch.int32
    multiplier: torch.Tensor,     # dtype=torch.int16, Q15
    rshift: torch.Tensor,         # dtype=torch.int8, 非负
    y_zp: torch.Tensor,           # dtype=torch.int32
    qmin: int = INT16_QMIN,
    qmax: int = INT16_QMAX,
    rounding_mode: RoundingMode = RoundingMode.HALF_TO_EVEN,
) -> torch.Tensor:
    """
    返回 dtype=torch.int16 张量。

    实现：
        prod = acc.to(int64) * multiplier.to(int64)
        rounded = round_shift(prod, rshift, rounding_mode)
        y = rounded + y_zp
        return saturate_to_range(y, qmin, qmax).to(int16)
    """
    ...
```

不变式：

- `acc.dtype == torch.int32`
- `0 <= multiplier <= MULTIPLIER_MAX`（per-tensor 或 per-channel 广播）
- `rshift >= 0` 且 `rshift <= 31`
- 内部中间量必须使用 `int64`，禁止 `float`。
- 输出必须经 saturate；wrap-around 视为缺陷。

### 7.2 辅助接口

```python
def saturate_int16(x: torch.Tensor) -> torch.Tensor: ...
def saturate_to_range(x: torch.Tensor, qmin: int, qmax: int) -> torch.Tensor: ...

def round_shift(
    x: torch.Tensor,             # dtype=torch.int64
    rshift: torch.Tensor,        # dtype=torch.int8
    rounding_mode: RoundingMode,
) -> torch.Tensor:               # 返回 int64
    ...
```

---

## 8. `FixedKernel` 协议与 registry

模块路径：`aimet_torch.fixed_point.registry`

### 8.1 协议

```python
from typing import Protocol, runtime_checkable, Any

@runtime_checkable
class FixedKernel(Protocol):
    """
    所有定点 kernel 必须实现该协议。

    inputs / params 均为 Int16QuantizedTensor 列表。
    output_encoding 描述输出 scale/zp/qmin/qmax。
    extra 携带 op 特定的形状、stride、padding 等。
    """

    module_type: type

    def __call__(
        self,
        inputs: list[Int16QuantizedTensor],
        params: dict[str, Int16QuantizedTensor],
        output_encoding: "OutputEncoding",
        extra: dict[str, Any],
    ) -> Int16QuantizedTensor:
        ...
```

### 8.2 注册接口

```python
def register_fixed_kernel(
    module_type: type,
    *,
    overwrite: bool = False,
) -> Callable[[type], type]:
    """
    装饰器。将 FixedKernel 子类注册到全局 registry。

    重复注册同一 module_type 时若 overwrite=False 则抛错。
    """
    ...

def get_fixed_kernel(module_type: type) -> FixedKernel:
    """未注册时抛 KernelNotFoundError，禁止静默 fallback 到 float。"""
    ...

def list_registered_kernels() -> list[type]: ...

class KernelNotFoundError(RuntimeError): ...
```

### 8.3 OutputEncoding 数据结构

```python
@dataclass(frozen=True)
class OutputEncoding:
    scale: torch.Tensor              # 仅离线生成 multiplier 时使用
    zero_point: torch.Tensor
    qmin: int
    qmax: int
    multiplier: torch.Tensor         # int16, Q15
    rshift: torch.Tensor             # int8, 非负
    axis: Optional[int] = None
```

不变式：`multiplier` 与 `rshift` 必须由离线 pipeline 生成并固化；运行时只读。

---

## 9. 离线参数生成

模块路径：`aimet_torch.fixed_point.offline`

### 9.1 multiplier 生成

```python
def quantize_multiplier(
    real_multiplier: float | torch.Tensor,   # 仅离线允许
    multiplier_bits: int = MULTIPLIER_QBITS,
    max_rshift: int = 31,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    返回 (multiplier_int16, rshift_int8)。

    若 real_multiplier 为张量则按元素生成（per-channel 支持）。
    若 real_multiplier == 0 则返回 (0, 0)。
    若超出可表达范围则抛 ValueError。
    """
    ...
```

### 9.2 bias 生成

```python
def quantize_bias_int32(
    bias_float: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
) -> torch.Tensor:
    """
    返回 dtype=torch.int32。
    bias_int32 = round(bias_float / (x_scale * w_scale))
    溢出抛 ValueError。
    """
    ...
```

### 9.3 LUT 生成

```python
def generate_lut_int16(
    fn: Callable[[torch.Tensor], torch.Tensor],   # 浮点参考函数
    input_encoding: "InputEncoding",
    output_encoding: OutputEncoding,
    table_size: int = 256,
) -> torch.Tensor:
    """
    返回 dtype=torch.int16, shape=(table_size,)。
    """
    ...
```

### 9.4 INT16 freeze pipeline

模块路径：`aimet_torch.fixed_point.offline.pipeline`

```python
def freeze_int16_fixed(
    sim_model,
    output_path: str,
    *,
    write_binaries: bool = True,
    error_threshold: float = 0.005,
    aimet_encoding_path: str | None = None,
    metadata: dict | None = None,
) -> dict[str, Any]:
    """
    写入 INT16 sidecar JSON + 逐层报告；Conv/Linear 可写 bias_int32.bin。
    返回 {"sidecar_path", "layers", "skipped", ...}。
    """

def freeze_int16_fixed_report_only(sim_model, *, error_threshold: float = 0.005) -> dict[str, Any]: ...
```

### 9.5 INT16 sidecar 导出（Phase B）

模块路径：`aimet_torch.fixed_point.export`

```python
def export_int16_sidecar_json(model: torch.nn.Module, path: str, ...) -> dict[str, Any]: ...
def load_int16_sidecar_json(path: str) -> dict[str, Any]: ...
def compare_sidecar_with_model(model: torch.nn.Module, sidecar: Mapping[str, Any], ...) -> dict[str, Any]: ...
```

Sidecar 文档字段：

- `version`: `1.0.0-int16-fixed`
- `format`: `aimet_rx_int16_fixed_sidecar`
- `layers[<module_name>]`: `fixed_point_tensor_bundle`（`output_encoding` + 可选 `pwl` + 可选 **`input_requants`**）
- `layers[<module_name>].bias_int32_path`（可选）：相对 sidecar 目录的 bias 二进制路径
- `layers[<module_name>].onnx_tensor_names`（可选）：由 AIMET `.encodings` 推导的 ONNX 张量名提示
- `aimet_encoding_reference`：对应 `.encodings` 路径

部署导出：`export_onnx_and_encodings.export_onnx_json(..., export_int16_sidecar=True)` 默认写出
``{prefix}.int16.json``，返回 ``(onnx_path, enc_path, int16_path)``。

不变式：导出推导必须与 `dispatch_int16_fixed` 使用同一套 `OutputEncoding` / PWL 生成逻辑。

---

## 10. 可观测性

模块路径：`aimet_torch.fixed_point.metrics`

### 8.1 Profiler

```python
class FixedPointProfiler:
    def __init__(self, model: torch.nn.Module): ...
    def __enter__(self) -> "FixedPointProfiler": ...
    def __exit__(self, exc_type, exc, tb) -> None: ...

    def to_dict(self) -> dict[str, Any]:
        """
        返回结构：
        {
          "<layer_name>": {
            "saturation_ratio": float,
            "acc_max_abs": int,
            "multiplier": int,
            "rshift": int,
            "bit_utilization": float,
          },
          ...
        }
        """
        ...

    def to_json(self, path: str) -> None: ...
```

### 8.2 模式间对比

```python
DEFAULT_COMPARE_MODES = (
    ExecutionMode.FP32_QDQ,
    ExecutionMode.FP16_QDQ,
    ExecutionMode.FIXED_SCALE_QDQ,
    ExecutionMode.INT16_FIXED_EVAL,
)

def compare_modes(
    model: torch.nn.Module,
    input_data: torch.Tensor | tuple[torch.Tensor, ...],
    modes: list[ExecutionMode] | None = None,
    metrics: list[str] = ("max_abs_error", "rmse", "cosine_similarity", "sqnr_db"),
) -> dict[str, Any]:
    """If ``modes`` is None, uses ``DEFAULT_COMPARE_MODES`` (``fp32_qdq`` first)."""
    ...
```

---

## 11. v1 / v2 适配层

### 9.1 v2 适配

```python
# aimet_torch/v2/quantization/affine/fixed_point/adapter.py

def patch_v2_module_for_int16(module: "QuantizationMixin") -> None: ...
def patch_v2_module_for_fp16(module: "QuantizationMixin") -> None: ...
```

### 9.2 v1 适配

```python
# aimet_torch/v1/fixed_point_adapter.py

def patch_v1_wrapper_for_int16(wrapper: "QcQuantizeWrapper") -> None: ...
def patch_v1_wrapper_for_fp16(wrapper: "QcQuantizeWrapper") -> None: ...
```

不变式：

- 适配层只调用本文件定义的 public 符号；不允许穿透到 `aimet_torch.fixed_point` 内部 private 模块。
- 适配层不持有量化数学逻辑；逻辑在 `aimet_torch.fixed_point` 中。

---

## 12. 错误模型

| 错误类 | 触发条件 |
| --- | --- |
| `KernelNotFoundError` | 未注册 fixed kernel |
| `ValueError` | encoding/multiplier/bias/LUT 越界 |
| `RuntimeError` | int16_fixed 路径检测到 float tensor 计算 |
| `NotImplementedError` | spec 中标注为 future 的功能 |

约束：

- `int16_fixed_eval` 与 `int16_fixed_qat_sim` 不允许任何 `try/except` 静默吞掉上述异常；必须传播或显式记录。
- 所有错误信息必须包含模块名、张量 shape、dtype，方便调试。

---

## 13. 版本与变更政策

- 本文件版本随 `aimet_rx` 主版本号变化；任何破坏性接口变更必须 bump 主版本。
- 新增枚举值、新增 kernel、新增 metric 视为兼容变更，bump minor 版本。
- 文件顶部应保留 `Last updated:` 行；本文件由设计 owner 维护。

Last updated: 2026-05-19

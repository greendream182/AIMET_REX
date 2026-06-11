# 15 fixed_scale_qdq（定点 scale + QDQ 主路径）

Status: implemented

## 1. 目标

实现 **G2 / `fixed_scale_qdq`**：每个 quantizer 的 encoding scale 以 **`(m_int16, rshift)`** 为权威表示（`scale = M / 2^rshift`），前向在 quantizer 边界用定点 scale 做 Q/DQ，模块内仍为 **Op_float**。满足 Ada200「任意 M,r」规范，**不**依赖 Power-of-2 scale。

顶层叙事见 [FixedPoint_Quantization_Design_v2.md](../FixedPoint_Quantization_Design_v2.md) §3.2、§5。

## 2. 范围

### 2.1 新增文件

- `aimet_torch/fixed_point/fixed_scale_qdq.py` — `quantize_with_fixed_scale` / `dequantize_with_fixed_scale` / `quantize_dequantize_with_fixed_scale`
- `aimet_torch/fixed_point/offline/scale_fixed.py` — `quantize_scale_to_m_rshift`（或由 `multiplier.py` 复用并导出别名）
- `tests/fixed_point/test_fixed_scale_qdq.py`

### 2.2 修改文件

- [execution_mode.py](../../aimet_torch/fixed_point/execution_mode.py) — 增加 `FIXED_SCALE_QDQ`
- [encoding_export.py](../../aimet_torch/fixed_point/encoding_export.py) — sidecar 字段 `m_int16`、`rshift`
- [adapter.py](../../aimet_torch/v2/quantization/affine/fixed_point/adapter.py) — `fixed_scale_qdq` 分派
- [quantizer.py](../../aimet_torch/v2/quantization/affine/quantizer.py) — `QuantizeDequantize.forward` 读取 mode
- [affine/encoding.py](../../aimet_torch/v2/quantization/affine/encoding.py) — 可选缓存 `FixedScaleEncoding`
- [fixed_point/__init__.py](../../aimet_torch/fixed_point/__init__.py) — 导出新符号
- [INTERFACE.md](../../aimet_torch/v2/quantization/affine/fixed_point/INTERFACE.md) — 本 spec 的 API 契约

### 2.3 不在范围

- v1 legacy：薄适配见 `aimet_torch/v1/fixed_scale_qdq.py`（StaticGrid + STE 前向；非本 spec 主交付）
- 硬件整数仿真（`int16_fixed_*`，spec 04–09）
- Power-of-2 对齐逻辑改动（保持 `power_of_2_quantization.py` 可选独立）
- 自动修改 AIMET 默认 `fp32_qdq` 行为

## 3. 前置依赖

- spec 01（execution mode API）已实现
- spec 02（fp16）可选；与 fp16 正交
- 离线 `quantize_multiplier` 算法可复用（[multiplier.py](../../aimet_torch/fixed_point/offline/multiplier.py)）

## 4. 数据契约

### 4.1 `FixedScaleEncoding`（每个 quantizer 一份）

| 字段 | dtype | 约束 |
|------|-------|------|
| `m_int16` | `torch.int16` | 与 scale 同 shape（per-tensor 标量或 per-channel） |
| `rshift` | `torch.int8` | 非负，≤ 31；与 `m_int16` 可广播 |
| `zero_point` | `torch.int32` | 与 AIMET offset 一致 |
| `qmin`, `qmax` | `int` | 由 bitwidth / 对称性决定 |
| `axis` | `Optional[int]` | per-channel 时为通道轴 |
| `scale_fp_legacy` | `Optional[float Tensor]` | 只读；离线校准值，**运行时不用** |

不变式：

- `scale ≈ m_int16.to(float64) / 2^rshift`（离线生成误差在配置阈值内）
- `m_int16 == 0` 时仅当 `scale == 0`；此时 Q/DQ 输出全为 dequant 零点
- **不**要求 `scale` 为 \(1/2^n\)

### 4.2 Q/DQ 输入输出（`fixed_scale_qdq` mode）

| 算子 | 输入 | 输出 |
|------|------|------|
| `quantize_with_fixed_scale` | `x: float Tensor` | `q: float Tensor`（整数值，dtype 与 backend 一致，落在 qmin..qmax） |
| `dequantize_with_fixed_scale` | `q` | `x̃: float Tensor` |
| `quantize_dequantize_with_fixed_scale` | `x` | `x̃`（与 AIMET QDQ 语义一致，scale 来自 M,r） |

- 权/活 **bitwidth 不同** 时，各自独立的 `FixedScaleEncoding` 与 `qmin/qmax`。
- 中间模块仍接收 **float** `x̃`（与 `fp32_qdq` 相同 carrier）。

### 4.3 sidecar JSON（扩展）

在现有 encoding 导出上增加（向后兼容）：

```json
{
  "scale": 0.0078125,
  "offset": 0,
  "bitwidth": 8,
  "m_int16": 32767,
  "rshift": 12
}
```

- **Sidecar 反序列化**（`fixed_scale_encoding_from_dict`）无 `m_int16`/`rshift` 时 **必须** `ValueError`，提示先 `convert_encodings_to_fixed_scale`。
- **运行时 Q/DQ**（v2 `quantize_dequantize` + `FIXED_SCALE_QDQ`）：从已校准的 float `scale` **即时生成** `(M,r)` 并仅用 M,r 做乘除（**禁止**用 float scale 作 Q/DQ 乘子）；部署前仍应 `convert_encodings_to_fixed_scale` 固化 sidecar。

## 5. API 签名

以 [INTERFACE.md](../../aimet_torch/v2/quantization/affine/fixed_point/INTERFACE.md) §4–§5 为准。摘要：

```python
# execution_mode.py
class ExecutionMode(str, Enum):
    ...
    FIXED_SCALE_QDQ = "fixed_scale_qdq"

# fixed_point/encoding.py 或 dataclass 模块
@dataclass(frozen=True)
class FixedScaleEncoding:
    m_int16: torch.Tensor
    rshift: torch.Tensor
    zero_point: torch.Tensor
    qmin: int
    qmax: int
    axis: Optional[int] = None
    scale_fp_legacy: Optional[torch.Tensor] = None

# offline/scale_fixed.py
def quantize_scale_to_m_rshift(
    scale: Union[float, torch.Tensor],
    multiplier_bits: int = 16,
    max_rshift: int = 31,
) -> Tuple[torch.Tensor, torch.Tensor]: ...

# fixed_point/fixed_scale_qdq.py
def quantize_with_fixed_scale(
    tensor: torch.Tensor,
    encoding: FixedScaleEncoding,
    rounding_mode: RoundingMode = RoundingMode.HALF_TO_EVEN,
) -> torch.Tensor: ...

def dequantize_with_fixed_scale(
    tensor: torch.Tensor,
    encoding: FixedScaleEncoding,
) -> torch.Tensor: ...

def quantize_dequantize_with_fixed_scale(
    tensor: torch.Tensor,
    encoding: FixedScaleEncoding,
    rounding_mode: RoundingMode = RoundingMode.HALF_TO_EVEN,
) -> torch.Tensor: ...
```

## 6. 算法 / 伪代码

### 6.1 离线：float scale → (M, rshift)

```text
# 与 quantize_multiplier 相同：对 real_value = scale 做 frexp + Q15
(m, r) = quantize_scalar_multiplier(scale, multiplier_bits=15, max_rshift=31)
# 验证: |scale - m/2^r| / scale < tol（默认 tol=1e-4，可配置）
```

对 per-channel `scale` 张量逐元素调用。

### 6.2 量化（运行时，无 float scale）

```text
function quantize_with_fixed_scale(x, enc):
    M, r, zp, qmin, qmax = enc.m_int16, enc.rshift, enc.zero_point, enc.qmin, enc.qmax
    # 广播 M, r, zp 到 x 的 shape
    if M == 0:
        return full_tensor(zp, like=x)   # 或按项目约定返回 clamped zp
  x64 = x.to(int64)   # 或 float 路径见实现注
    q = round(x * 2^r / M) - zp          # 实际实现用 int64 避免溢出
    q = clamp(q, qmin, qmax)
    return q_as_float_carrier(q)         # 与 torch_builtins QDQ 一致：float 载体存整数网格
```

实现注：允许用 `float64` 中间量计算 `round(x * 2^r / M)`，但 **禁止** 读取 `scale_fp_legacy` 作为乘子；若用纯整数路径，需处理 `x * 2^r` 溢出（大 feature map 用分块或 int64）。

### 6.3 反量化

```text
function dequantize_with_fixed_scale(q, enc):
    return (q + zp) * M / 2^r    # 输出 float，与 enc 广播一致
```

### 6.4 QDQ（STE 反向）

```text
function quantize_dequantize_with_fixed_scale(x, enc):
    q = quantize_with_fixed_scale(x, enc)
    return dequantize_with_fixed_scale(q, enc)
```

反向：与现有 `QuantDequantFunc` 相同 STE（mask 在 qmin..qmax 内）。

### 6.5 v2 adapter 分派

```text
forward QuantizeDequantize(x):
    mode = get_quant_execution_mode()
    if mode == FP32_QDQ:
        return existing_float_scale_qdq(x)
    if mode == FP16_QDQ:
        return fp16_float_scale_qdq(x)
    if mode == FIXED_SCALE_QDQ:
        enc = load_fixed_scale_encoding(self)  # 含 M,r,zp,qmin,qmax
        return quantize_dequantize_with_fixed_scale(x, enc)
    ...
```

## 7. 实施步骤

1. 更新 INTERFACE.md 与 `ExecutionMode`。
2. 实现 `quantize_scale_to_m_rshift`（可 thin wrap `quantize_multiplier`）。
3. 实现 `fixed_scale_qdq.py` 三个算子 + 单元测试 golden。
4. 扩展 `encoding_export` 读写 `m_int16`/`rshift`。
5. 提供 `convert_encodings_to_fixed_scale(sim)` 遍历所有 quantizer 写 sidecar。
6. v2 `adapter` / `QuantizeDequantize` 分派 `FIXED_SCALE_QDQ`。
7. 扩展 `compare_modes` 增加 `fixed_scale_qdq vs fp32_qdq`（spec 12 协同）。
8. 文档：Design v2 §12 M2.5 标为 implemented。

## 8. 验收标准

### 8.1 单元测试

| 用例 | 输入 | 期望 |
|------|------|------|
| scale=1/8 | `M=1, rshift=3`, x=0.5 | q=4（U8 网格示例），dequant≈0.5 |
| scale≈0.00390625 | 离线 (M,r) 重建误差 | relative error < 1e-3 |
| per-channel | scale shape [C] | M,r shape [C] 广播正确 |
| vs fp32_qdq | 同一 encodings，tiny linear | output max_abs < 1e-5 * range（或 cosine ≥ 0.9999） |
| 缺 M,r | mode=fixed_scale_qdq | `ValueError` 含 convert 提示 |

### 8.2 回归

- 默认 `fp32_qdq`：`tests/fixed_point/` 现有用例全部通过。
- `int16_fixed_*` 行为不变。

### 8.3 性能

- 单 quantizer QDQ forward（CPU，1M 元素）< 10ms（Tier A 子集）。

## 9. 不允许做的事 (Do NOT)

- 在 `fixed_scale_qdq` 下静默回退到 float `scale`。
- 将 Po2 对齐设为进入 `fixed_scale_qdq` 的前置条件。
- 在 `fixed_scale_qdq` 前向中调用 `Int16QuantizedTensor` 或 integer conv kernel。
- 假设 `fixed_scale_qdq` 与 `int16_fixed_eval` 数值等价。

## 10. 参考

- [FixedPoint_Quantization_Design_v2.md](../FixedPoint_Quantization_Design_v2.md)
- [multiplier.py](../../aimet_torch/fixed_point/offline/multiplier.py)
- [torch_builtins.py](../../aimet_torch/v2/quantization/affine/backends/torch_builtins.py) `QuantDequantFunc`

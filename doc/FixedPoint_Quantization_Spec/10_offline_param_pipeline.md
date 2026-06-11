# 10 离线参数固化 pipeline

Status: implemented

## 1. 目标

把 PTQ / QAT 完成后的 encoding 转成 `int16_fixed_eval` 运行时所需的整数参数：`multiplier` / `rshift` / `bias_int32` / LUT，并写回 encoding sidecar。**这是浮点数学唯一被允许出现的阶段**。

## 2. 范围

### 2.1 新增文件

- [aimet_torch/fixed_point/offline/multiplier.py](../../aimet_torch/fixed_point/offline/multiplier.py)
- [aimet_torch/fixed_point/offline/bias.py](../../aimet_torch/fixed_point/offline/bias.py)
- [aimet_torch/fixed_point/offline/lut_gen.py](../../aimet_torch/fixed_point/offline/lut_gen.py)
- [aimet_torch/fixed_point/offline/pipeline.py](../../aimet_torch/fixed_point/offline/pipeline.py)
- [examples/freeze_int16_fixed.py](../../examples/freeze_int16_fixed.py)

### 2.2 修改文件

- 必要时扩展 `aimet_torch/v2/quantization/affine/encoding.py` 序列化字段（若 spec 04 未完成）。

### 2.3 不在范围

- 运行时 kernel（spec 05 / 07 / 08 / 09）。
- QAT 反向（spec 11）。

## 3. 前置依赖

- spec 04 / 05 完成。
- 模型已完成 PTQ calibration 或 QAT 训练，encoding 已 freeze。

## 4. 数据契约

入口：

```text
freeze_int16_fixed(sim_model, output_path) -> path/to/encodings_int16.json
```

输出 sidecar JSON 字段（在原 encoding 上扩展）：

- 每层：`multiplier_uint16` / `rshift_int8`
- Conv/Linear 层：附 `bias_int32_path`
- Add/Concat：每路输入对应一组 `m_i` / `s_i`
- AvgPool：`multiplier` 含 `1/kernel_size` 的近似
- Sigmoid/Tanh：`lut_path` 指向 LUT 二进制文件

不变式：

- 浮点 scale 进入 pipeline；整数 multiplier / rshift 出 pipeline。
- PWL 离线拟合（`generate_pwl_lut`）使用 `(m_uint16, rshift)` 反演的 **effective scale**（`m/2^r`），与板端 scale 权威一致（ADR-015）。
- 同一 op 的 multiplier 与 rshift 必须满足 `0 <= multiplier <= 65535`，`0 <= rshift <= 31`。
- 误差超阈值时（`|real_multiplier - multiplier/2^rshift| / real_multiplier > 0.5%`）输出 warning 并写入报告。
- 二进制 LUT / bias 文件按 little-endian 写入，与 sidecar JSON 路径关联。

## 5. API 签名

```python
# offline/multiplier.py
def quantize_multiplier(
    real_multiplier: float | torch.Tensor,
    multiplier_bits: int = 16,
    max_rshift: int = 31,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 (multiplier_uint16, rshift_int8)。"""
    ...

# offline/bias.py
def quantize_bias_int32(
    bias_float: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
) -> torch.Tensor: ...

# offline/lut_gen.py
def generate_lut_int16(
    fn: Callable[[torch.Tensor], torch.Tensor],
    input_encoding: InputEncoding,
    output_encoding: OutputEncoding,
    table_size: int = 256,
) -> torch.Tensor: ...

# offline/pipeline.py
def freeze_int16_fixed(
    sim_model,
    output_path: str,
    *,
    write_binaries: bool = True,
    error_threshold: float = 0.005,
) -> dict[str, Any]:
    """
    遍历 sim_model 的每个量化模块，根据 op 类型生成 (multiplier, rshift, bias, lut)，
    写入 encoding sidecar。返回每层报告字典。
    """
    ...
```

## 6. 算法 / 伪代码

`quantize_multiplier`（per-tensor 示例）：

```text
if real_multiplier == 0:
    return 0, 0

# 归一化到 [0.5, 1)
mantissa, exponent = frexp(real_multiplier)   # mantissa in [0.5, 1)
shift = -exponent + (multiplier_bits)          # 例：rshift = 15 - exponent
multiplier = round(mantissa * (1 << multiplier_bits))
if multiplier == (1 << multiplier_bits):
    multiplier //= 2
    shift -= 1

if shift < 0 or shift > max_rshift:
    raise ValueError(...)

return int16(multiplier), int8(shift)
```

per-channel：对张量元素逐个调用并保留为张量。

`freeze_int16_fixed`：

```text
report = {}
for module in sim_model.modules():
    if not is_quant_module(module):
        continue

    op_type = type(unwrap(module))
    enc_in = collect_input_encodings(module)
    enc_w = collect_param_encodings(module).get("weight")
    enc_out = collect_output_encoding(module)

    if op_type in (Conv, Linear, MatMul):
        real_m = (enc_in.scale * enc_w.scale / enc_out.scale)
        m, s = quantize_multiplier(real_m)
        bias_int32 = quantize_bias_int32(module.bias, enc_in.scale, enc_w.scale)
        attach_to_encoding(enc_out, multiplier=m, rshift=s, bias=bias_int32)
    elif op_type in (Add,):
        for i, ei in enumerate(enc_in):
            real_m_i = ei.scale / enc_out.scale
            m_i, s_i = quantize_multiplier(real_m_i)
            attach_to_encoding(enc_out, key=f"m{i}", value=(m_i, s_i))
    elif op_type in (AvgPool2d,):
        real_m = enc_in.scale / (enc_out.scale * kernel_area)
        m, s = quantize_multiplier(real_m)
        attach_to_encoding(enc_out, multiplier=m, rshift=s)
    elif op_type in (Sigmoid, Tanh):
        lut = generate_lut_int16(reference_fn, enc_in, enc_out)
        attach_to_encoding(enc_out, lut_path=write_binary(lut))
    else:
        report[name] = "skipped: no fixed-point rule"

    report[name] = compute_error_diagnostics(real_m, m, s)

write_sidecar(output_path, sim_model.encodings, extras=report)
return report
```

边界：

- `real_multiplier` 极小或极大：触发 `ValueError`，提示需要重做 calibration。
- bias 越 int32 上下界：抛 `ValueError`。
- LUT 生成时 input range 估计来自 `input_encoding.qmin/qmax * scale`。

## 7. 实施步骤

1. 实现 `quantize_multiplier` per-tensor / per-channel。
2. 实现 `quantize_bias_int32`。
3. 实现 `generate_lut_int16`。
4. 实现 `freeze_int16_fixed`，按 op type 分派。
5. CLI 入口 `examples/freeze_int16_fixed.py`：`--sim-model checkpoint.pth --encoding xxx.json --output xxx_int16.json --report report.json`。
6. 单元测试覆盖每种 op type 的 multiplier 生成与误差报告。

## 8. 验收标准

### 8.1 单元测试

```python
def test_quantize_multiplier_known_value():
    m, s = quantize_multiplier(0.1234)
    approx = m / (1 << s)
    assert abs(approx - 0.1234) / 0.1234 < 1e-3

def test_quantize_multiplier_per_channel():
    real = torch.tensor([0.1, 0.05, 0.01])
    m, s = quantize_multiplier(real)
    assert m.dtype == torch.uint16
    assert s.dtype == torch.int8

def test_freeze_pipeline_writes_all_keys():
    sim = build_tiny_calibrated_sim()
    report = freeze_int16_fixed(sim, "out.json")
    assert "multiplier_uint16" in report["conv1"]
    assert "bias_int32_path" in report["conv1"]
```

### 8.2 必须通过的现有测试

- 默认 encoding 序列化测试通过。

### 8.3 性能阈值

- 中等模型（~50 层）freeze 时间 < 30s。

## 9. 不允许做的事

- 不允许 pipeline 输出包含浮点 scale 字段（必须替换为 multiplier + rshift）。
- 不允许默认 silently 跳过未支持 op；必须出现在 report 中并显式标注。
- 不允许 LUT 写入 float 表。

## 10. 参考

- INTERFACE.md 第 7 节。
- TFLite `QuantizeMultiplier`。
- 顶层文档 ADR-001。

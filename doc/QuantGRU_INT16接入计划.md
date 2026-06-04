# QuantGRU 黑盒算子在 INT16_FIXED_EVAL 下的 AIMET 接入计划

> 状态：草案 v1 / 主笔：AIMET 侧 / 配合方：quant-gru-pytorch
> 配套文档：[quant-gru-pytorch/docs/Rescale_M_int16_r_shift_实施计划.md](../../quant-gru-pytorch/docs/Rescale_M_int16_r_shift_实施计划.md)（QuantGRU 内部 rescale 改造，由 quant-gru-pytorch 同事负责）

## 实施进度（更新于 2026-05-28）

| 项 | 状态 | 落点 |
|---|---|---|
| R0 wrapper FP32_QDQ / FP16_QDQ 修复 | done | `aimet_torch/v2/nn/modules/custom.py::QuantizedQuantGRU._builtin_torch_fn_helper` + forward 路由；测试 G-1..G-5 |
| R0+ §2.5 hidden 端 quantizer 共享 | done | `_ensure_hidden_quantizer_shared()`；测试 G-6 / G-7 |
| R0++ FIXED_SCALE_QDQ wrapper 别名 | done | `resolve_quantgru_mode_str()`（`quantgru_adapter.py`）；forward + `_aimet_compute_encodings_exit` 统一映射；测试 G-8 / G-9 |
| R0 审计 bugfix（ExecutionMode.FP32） | done | 删除 wrapper 中不存在的 `ExecutionMode.FP32` 分支；G-1/G-3 改走底层 `QuantGRU.forward` 作 reference |
| 本机 CI 真环境验收 | done | quant_gru + RTX 5090：**47 passed, 0 skipped**（`tests/fixed_point/conftest.py` 自动发现 sibling `quant-gru-pytorch/pytorch`） |
| R2 全图 INT16 MRNN backbone 分段 | done | e2e 全绿；`shape_meta.py` 通用旁路 shape 标量运算；`FixedPointSimTensor` layout API |
| R1 QAT 反向梯度匹配 `backward_quant` | done | `quantgru_adapter`：QAT_SIM 走 `_OptionalQuantGRU.forward`（`GRUFunction`+`backward_quant`）；eval 仍 `forward_quantized`；`test_quantgru_qat_grad.py` |
| R3 MRNN DSP 前端 INT16 适配 | done (decomposed) | 全图 decomposed INT16；metric Δ=0；Abs/Sign/Divide reference kernels |

> 各条详细行动 / 验收 / 回滚见附 C。本节只做"截至当下"的状态摆渡，更细的 PR 拆分以附 C 为准。

## 0. 概览

### 0.1 目标

让 AIMET 在所有现有 ExecutionMode（FP32 / FP32_QDQ / FP16_QDQ / INT16_FIXED_EVAL / INT16_FIXED_QAT_SIM）下都能正确驱动 `quant_gru.QuantGRU`，把它当作**黑盒算子**对待：AIMET 只管输入张量到达 QuantGRU 之前与离开 QuantGRU 之后的"形态切换"，不复刻 QuantGRU 的内部定点计算。

### 0.2 非目标

- 不在 AIMET 侧重新实现 GRU 整数语义。
- 不解析 QuantGRU 内部参数字段（`shift_*` / `quant_params` / `LUT` 等）。
- 不接管 QuantGRU 的 ONNX 子图；导出仍走 QuantGRU 自带 `export_mode + export_format`。
- 不阻塞 quant-gru-pytorch 同事的 rescale 改造；他们改 POT → (M_int16, r_shift) 期间，本接入逻辑保持不变。
- **不抽 `OpaqueFusedOpAdapter` 通用 Mixin**：所有 AIMET 适配代码留在 `QuantizedQuantGRU` 单点；待出现第二个 OpaqueFusedOp（如 QuantLSTM）再 refactor。当前抽象会与现有 `QuantizationMixin` / `_DispatchMixin` 职责重叠。
- **不立独立 ABI 文档**：契约即本文档第 1 章，避免「AIMET plan / rescale plan / ABI」三向同步成本；待出现第三方算子或契约跨多版本演进时再拆分。

### 0.2.1 边界原则（QuantGRU 与 AIMET 的责任分割）

> 此原则用于阻止「为 QuantGRU 在每种 ExecutionMode 都写一份内部实现」的设计漂移。所有路由表与 wrapper 行为必须先满足本节，再展开。

| 模式 | 核心 forward 责任方 | 边界 Q/DQ 责任方 |
|---|---|---|
| FP32 | QuantGRU（标准浮点 forward） | 无 |
| FP32_QDQ | QuantGRU（标准浮点 forward） | **AIMET wrapper**（input/output_quantizers 走通用 fake-quant） |
| FP16_QDQ | QuantGRU（fp32 forward + 外层 fp16 cast） | **AIMET wrapper**（边界 fp16 quantizer） |
| INT16_FIXED_EVAL | **QuantGRU**（`forward_quantized` bit-exact 整数） | AIMET dispatch（identity 反向 + Int16 容器，§2.4） |
| INT16_FIXED_QAT_SIM | 同上 | 同上（§2.4 协议） |
| 校准期 | QuantGRU（`_forward_with_calibration` 收 GRU 内部 stats） | 通用 quantizer 同步收 boundary stats |

强制约束（任何后续修改必须保持）：

- **QuantGRU 仓库只内化 INT16 bit-exact**：因为 INT16 数值必须与部署 CUDA kernel 严格一致，AIMET 无法在 fp32 域复刻；这是 quant-gru-pytorch 必须独占的能力。
- **QuantGRU 不内化 FP32_QDQ / FP16_QDQ 边界 Q/DQ**：交由 AIMET 通用 dispatch 处理，避免 quant-gru-pytorch 仓库为每种 mode 再长出一份独立实现，也避免 AIMET 端的 mode 路由失去意义。
- **AIMET wrapper 必须把 FP32_QDQ / FP16_QDQ 接到通用 `_DispatchMixin.forward` 路径**（而不是直接调 `_OptionalQuantGRU.forward`），否则 `input_quantizers` / `output_quantizers` 形同虚设，与 MobileNet 等其它算子在两种 mode 下的行为不对齐。具体修复见 §3.1.1。
- **INT16 路径下 wrapper 的 `input_quantizers` / `output_quantizers` 维持 None**，由 `dispatch_int16_fixed → _dispatch_quantgru_blackbox` 全权接管（避免与 §2.4 identity 反向叠加）。

### 0.3 验收

- MRNN/含 QuantGRU 的模型在 5 个 ExecutionMode 下 forward 通过、无 KernelNotFoundError、无 missing output quantizer。
- INT16_FIXED_EVAL 下端到端任务 metric 与 quant-gru-pytorch 自带量化推理 **bit-exact 一致**（通过 1.4 `forward_quantized` 保证）。
- INT16_FIXED_QAT_SIM 下 backward 与 QuantGRU 自身 `backward_quant` 梯度一致（max abs diff < 1e-6）。
- compute_encodings 上下文同步切换 QuantGRU 校准状态。
- rescale 改造（POT → M_int16）合入后，AIMET 侧无需修改任何代码（契约 v1 稳定性验证）。
- 双向 CI conformance（§4.4）红绿守护两侧 PR。

### 0.4 范围

| 模块 | 是否改 | 说明 |
|---|---|---|
| `aimet_torch/v2/nn/modules/custom.py::QuantizedQuantGRU` | 是 | 主要改动点：模式路由 + 校准 hook + 元数据 getter |
| `aimet_torch/v2/nn/true_quant.py::_DispatchMixin.forward` | 否 | 走 dispatch 时 QuantGRU 自身决定 forward |
| `aimet_torch/v2/quantization/affine/fixed_point/adapter.py::dispatch_int16_fixed` | 是 | 增加 QuantGRU 的 dispatch 分支：直接调 `forward_quantized` + 元数据打包 |
| `aimet_torch/fixed_point/sim_utils.py::ensure_output_quantizers_for_int16_eval` | 是 | 跳过 QuantGRU 类型 |
| `aimet_torch/fixed_point/sim_utils.py::INT16_DISPATCHABLE_MODULES` | 是 | 把 QuantGRU 加入清单（让 dispatch 路由命中） |
| 新增：readiness 诊断 | 是 | QuantGRU `is_calibrated()` 检查 |
| 测试 | 是 | tests/fixed_point 下增 5 个 mode + 校准协同回归 |

## 1. 接口契约 v1（冻结）

> 本章为双方对齐契约 **v1，已冻结**。未经 major 版本协商不得修改任何方法签名、IO dtype/device 协议、flag 名称与语义。
>
> 兼容规则见 1.8。修订时新增方法/字段为 minor +1（向后兼容），破坏性变更为 major +1（双方对齐）。本章冻结后双方各自落地，互不阻塞。

### 1.1 forward dtype/device 协议（不变量）

| 边界 | 约束 |
|---|---|
| 输入 `input` | `torch.float32`、CUDA、shape 与 `nn.GRU` 一致 |
| 输入 `hx` | 同上，可为 `None` |
| 输出 `output` | `torch.float32`、CUDA、与 `nn.GRU` 输出 shape 一致 |
| 输出 `h_n` | 同上 |

> AIMET 在调用 QuantGRU 前若上游是 `Int16QuantizedTensor` / `FP16QuantizedTensor`，必须 dequantize 成 fp32；这是 AIMET 内部责任。
> quant-gru-pytorch 不得改 `forward(input, hx) -> (output, h_n)` 的签名与 dtype；直吐整数张量走 1.4 节 `forward_quantized` 专用方法，**不得**替换 `forward`。

### 1.2 公开稳定 flag（运行时切换）

| flag | 类型 | 含义 |
|---|---|---|
| `use_quantization` | bool | True=走定点 forward；False=走浮点 |
| `calibrating` | bool | True=forward 时收集校准统计；与 `use_quantization` 互斥 |
| `export_mode` | bool | True=走纯 PyTorch 实现以便 ONNX 追踪 |
| `export_format` | str | `"float"` / `"qdq"` |

> 这些 flag 是公开稳定 API。`quant_gru.QuantGRU` 增减这些 flag 必须先与 AIMET 商量。

### 1.3 元数据 getter（quant-gru-pytorch 新增、AIMET 强依赖）

> **本接口是阻塞项**：AIMET 在 INT16_FIXED_EVAL 下打包输出张量必须依赖它。

```python
class QuantGRU(nn.Module):
    def get_io_quant_meta(self) -> dict:
        """
        Returns:
          {
            "input":  {"scale": float, "zp": int, "bitwidth": int, "is_symmetric": bool},
            "output": {"scale": float, "zp": int, "bitwidth": int, "is_symmetric": bool},
            "hidden": {"scale": float, "zp": int, "bitwidth": int, "is_symmetric": bool},
          }
        Notes:
          - scale 永远是连续浮点 scale（POT 与 (M_int16, r_shift) 模式下都返回连续值）
          - zp 是整数（对称量化时为 0）
          - 未校准时抛 RuntimeError("QuantGRU not calibrated")
        """
```

设计要点：

- 不论 QuantGRU 内部走 POT (`scale = 2^(-shift)`) 还是 (M_int16, r_shift) (`scale = M / 2^(15+r_shift)`)，本接口永远返回连续 `scale`。AIMET 侧不感知差异。
- `bitwidth` / `is_symmetric` 用于 AIMET 选择 `Int16QuantizedTensor` / `Int8QuantizedTensor` 等容器。
- `hidden` 的 scale 与 `output` 的 scale 在 QuantGRU 设计中一致；保留独立字段为未来扩展。
- 返回的 scale/zp 必须与 1.4 `forward_quantized` 的整数输出严格一致（bit-exact 依赖）。

### 1.4 `forward_quantized` 专用方法（INT16 bit-exact 边界）

> **本接口是 INT16 mode 阻塞项**：标准 `forward` 在末尾会反量化，AIMET 若再 requant 成 int16 会和部署 kernel 存在 ±1 误差。`forward_quantized` 直出内部整数张量，保证 sim 与部署 bit-exact。

```python
class QuantGRU(nn.Module):
    def forward_quantized(
        self,
        input: torch.Tensor,        # fp32 / CUDA
        hx: Optional[torch.Tensor] = None,  # fp32 / CUDA
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        仅 INT16_FIXED_EVAL / INT16_FIXED_QAT_SIM mode 下由 AIMET adapter 调用。

        Returns:
          (output_q, h_n_q): 两个张量元素 dtype 为 int16 (或 bitwidth 对应整数类型)，
                            shape 与标准 forward 一致。
                            数值即内部定点结果，未做反量化。

        Constraints:
          - 必须与部署 CUDA kernel bit-exact（实现要点：跳过现 _forward_cuda 末尾的 dequantize 步骤）。
          - 输出 scale/zp 必须等于 get_io_quant_meta()["output"] / ["hidden"] 的值。
          - 标准 forward(input, hx) -> (fp32, fp32) 签名保持不变。
          - 未校准（use_quantization=False 或 quant_params is None）时抛 RuntimeError。
        """
```

设计要点：

- **不污染标准 forward**：`nn.GRU` 协议兼容性保持，第三方代码不受影响。
- AIMET adapter 在 INT16_* mode 下**只**调 `forward_quantized`，绕过 fp32 round-trip。
- **禁用** "fp32 out + ±1 容忍" 的过渡方案（生产标准要求 bit-exact）。
- BiGRU 两个方向输出 stack/concat 在 `forward_quantized` 内部完成，对外仍是 2 个张量。

### 1.5 校准生命周期 hook + `aimet_configure` 单一入口

> AIMET 侧实现路由；quant-gru-pytorch 仅需实现 `aimet_configure(mode_str)`，**不需要懂 AIMET ExecutionMode 概念**。

```python
class QuantGRU(nn.Module):
    def aimet_configure(self, mode: str) -> None:
        """
        AIMET 侧用 ExecutionMode 字符串调用本方法，QuantGRU 内部映射到 (use_quantization, calibrating, export_mode)。

        合法 mode 字符串（须与 aimet_capabilities()["supported_modes"] 一致）：
          "fp32"               → use_quantization=False, calibrating=False, export_mode=False
          "fp32_qdq"           → use_quantization=False, calibrating=False, export_mode=False
          "fp16_qdq"           → use_quantization=False, calibrating=False, export_mode=False
          "int16_fixed_eval"   → use_quantization=True,  calibrating=False, export_mode=False
          "int16_fixed_qat_sim"→ use_quantization=True,  calibrating=False, export_mode=False
                                 （上层须 model.train()；本仓库不感知 train/eval）
          "calibrating"        → use_quantization=False, calibrating=True,  export_mode=False
        """
```

AIMET 侧调用范式：

```python
# 由 AIMET 在 sim.compute_encodings 上下文 enter/exit 调用
with module._aimet_unlock_ctx():       # 第 8 节
    gru.aimet_configure("calibrating")   # enter
# 用户传入校准数据做 forward
with module._aimet_unlock_ctx():
    gru.finalize_calibration()
    gru.aimet_configure(get_quant_execution_mode().value)  # exit
```

关键约束：

- `aimet_configure` **不引入**新内部计算路径，仅复用现有 `use_quantization / calibrating / export_mode` 已有语义。
- `finalize_calibration()` 由 quant-gru-pytorch 已有实现，AIMET 不接管。
- 校准期内 forward 仍按当前实现走 `_forward_with_calibration`。

### 1.6 sidecar JSON

`quant_params.json` 由 quant-gru-pytorch 自身导出/加载（`gru.export_quant_params(path)` / `gru.load_quant_params(path)`）。AIMET **不解析** sidecar 内部字段，仅持有路径与作用域。

### 1.7 ONNX 导出

AIMET 通用 export 路径碰到 QuantGRU 时**不递归拆解**，直接调用：

```python
gru.export_mode = True
gru.export_format = "qdq"  # 或 "float"
torch.onnx.export(gru, dummy_input, path, dynamo=False)
gru.export_mode = False
```

### 1.8 `adapter_version` 兼容规则

QuantGRU 必须提供 capability 报告：

```python
class QuantGRU(nn.Module):
    def aimet_capabilities(self) -> dict:
        """
        Returns:
          {
            "adapter_version": "1.0",       # major.minor 字符串
            "supported_modes": ["fp32", "fp32_qdq", "fp16_qdq",
                                "int16_fixed_eval", "int16_fixed_qat_sim",
                                "calibrating"],
            "forward_io_dtype": "float32",
            "forward_io_device": "cuda",
            "requires_calibration_for": ["int16_fixed_eval", "int16_fixed_qat_sim"],
            "supports_forward_quantized": True,
          }
        """
```

兼容规则：

| 变更类型 | 版本号 | AIMET 行为 |
|---|---|---|
| 新增 mode / 新增 capability 字段 / 实现层优化（如 POT → M16 rescale） | minor +1 | **不动** |
| `forward` 签名变 / IO dtype 变 / Q/DQ 权责变 / 删除已有 mode | major +1 | 双方对齐，AIMET adapter 同步升级 |

启动期检查（在 `QuantizedQuantGRU.__quant_init__` 或 sim 构建期执行）：

```python
caps = gru.aimet_capabilities()
EXPECTED_MAJOR = 1
MIN_SUPPORTED_MINOR = 0

major, minor = map(int, caps["adapter_version"].split("."))
if major != EXPECTED_MAJOR or minor < MIN_SUPPORTED_MINOR:
    raise IncompatibleAdapterVersionError(
        f"QuantGRU adapter_version={caps['adapter_version']} not supported. "
        f"AIMET requires major == {EXPECTED_MAJOR} and minor >= {MIN_SUPPORTED_MINOR}."
    )
```

不满足版本约束 → sim 构建期报错，不进 forward。

## 2. ExecutionMode 路由表

| AIMET ExecutionMode | use_quantization | calibrating | 调用方法 | 输入端处理 | 输出端处理 |
|---|---|---|---|---|---|
| FP32 | False | False | `forward()` | 上游 fp32 直传 | fp32 直传 |
| FP32_QDQ | False（默认）| False | `forward()` | dequant 上游 → fp32 | fp32 输出，可选追加 fake-quant |
| FP16_QDQ | False | False | `forward()` | dequant 上游 → fp32（QuantGRU 不支持 fp16） | fp32 → cast fp16 → fake-quant |
| INT16_FIXED_EVAL | True | False | **`forward_quantized()`** | dequant 上游 → fp32 | 直接整数张量 → 用 `get_io_quant_meta()` 配 scale/zp 包成 `Int16QuantizedTensor` |
| INT16_FIXED_QAT_SIM | True | False | **`forward_quantized()`** | 同上；要求 `model.train()` 已设 | 同上；反向遵守 2.4 节梯度协议 |
| 校准期 | False | True | `forward()` | 上游 fp32 直传 | fp32 输出（不做量化包装） |

设计原则：

- **入口统一 fp32**：所有模式下 QuantGRU 入口张量都是 fp32，避免内部因 dtype 多分支。
- **出口按 mode 二选一**：INT16_* 走 `forward_quantized` 直出整数（bit-exact）；其它 mode 走标准 `forward` 输出 fp32。
- **FP32_QDQ 下默认走浮点**：避免与 INT16 路径混淆；如果将来要"用定点输出当 fake quant"，作为可选项再加 flag。
- **禁止** fp32 round-trip 后 requant 的过渡方案（详见 1.4 节）。

### 2.4 QAT 梯度边界协议（INT16_FIXED_QAT_SIM 子规范）

这是 INT16_FIXED_QAT_SIM mode 下最容易翻车的盲点：QuantGRU 内部已有 STE，AIMET 边界若再触发一次 STE，**梯度被截断两次**。Loss 会静默异常，调试极慢。

反向规则（强制）：

| 边界 | 反向行为 |
|---|---|
| 上游量化张量 → QuantGRU `input` | dequant 反向 = **identity**（无 STE，不重复量化噪声） |
| QuantGRU `output` → 下游 | 整数 pack 反向 = **identity**（无 STE） |
| QuantGRU 内部 `_forward_cuda` | **自带 STE**（`backward_quant` clamp mask 处梯度置零） |

实现要点（AIMET 侧 `_dispatch_quantgru_blackbox` 内）：

```python
def _stop_grad_dequantize(qt: Int16QuantizedTensor) -> torch.Tensor:
    """量化张量 → fp32，反向用 identity 而非 Quantize.apply 反向。"""
    fp = qt.dequantize().detach()
    fp = fp + (qt.dequantize() - qt.dequantize().detach())  # straight-through identity
    return fp

def _stop_grad_pack(fp: torch.Tensor, meta: dict) -> Int16QuantizedTensor:
    """fp32 → 整数容器，反向 identity（不再加 STE 噪声）。"""
    int_tensor = _quantize_with_meta(fp, meta).detach()
    int_tensor = int_tensor + (fp - fp.detach())  # straight-through identity to fp
    return Int16QuantizedTensor.from_int(int_tensor, scale=meta["scale"], zp=meta["zp"])
```

**严禁**在边界 dequant / pack 处调 `Quantize.apply` / `QuantizeDequantize.apply`——它们的 backward 已含 STE。

单元测试要求（4.1 节会补）：

- 给定固定输入和已校准 GRU，比较 AIMET adapter 反向梯度 vs QuantGRU 自身 `backward_quant` 直接反向梯度，**数值需一致**（max abs diff < 1e-6）。
- 测试 BiGRU 双向场景。

### 2.5 hx/h_n scale 一致性约束

事实：`h_n[t]` 即下一次循环的 `hx[t+1]`，**是同一张量**。两端 quantizer scale 不一致会引入循环误差，BiGRU 双向各一份会放大。

约束（强制）：

```
output_quantizers[1].scale  ==  input_quantizers[1].scale
                            ==  get_io_quant_meta()["hidden"]["scale"]
output_quantizers[1].zp     ==  input_quantizers[1].zp
                            ==  get_io_quant_meta()["hidden"]["zp"]
```

实现要点：

- 校准期 AIMET **不重新算** hidden scale，直接读 QuantGRU 自报值（来自 `get_io_quant_meta()`）。
- `QuantizedQuantGRU.__quant_init__` 在初始化 hidden 端 quantizer 时，让 `input_quantizers[1]` 与 `output_quantizers[1]` **共享同一 `EncodingBase`** 引用（不是 deepcopy）。
- 反向也共享，不可拆成两个独立 quantizer。
- BiGRU：两个方向各自独立一组 `(hidden_fwd_in, hidden_fwd_out)` 和 `(hidden_bwd_in, hidden_bwd_out)`，每组内部共享，组间独立。
- 启动期 sanity check：如果 sim 用户手动覆盖了任一端 quantizer，校验失败时 raise。

> 实现细节：QuantGRU 当前设计中 `get_io_quant_meta()["hidden"]["scale"] == ["output"]["scale"]`；若未来分离，约束按 hidden 自身的 scale 走。

## 3. AIMET 侧改动详细清单

### 3.1 `aimet_torch/v2/nn/modules/custom.py`

`QuantizedQuantGRU` 重写（含 flag lock、版本校验、hidden 端 quantizer 共享）：

```python
if _OptionalQuantGRU is not None:
    @QuantizationMixin.implements(_OptionalQuantGRU)
    class QuantizedQuantGRU(_DispatchMixin, QuantizationMixin, _OptionalQuantGRU):
        """Black-box AIMET wrapper for QuantGRU.

        Routes forward by ExecutionMode; never re-implements GRU integer math.
        Bit-exact in INT16 modes via forward_quantized.
        """
        _builtin_torch_fn = None  # 不走通用 dispatch 的 builtin 路径

        EXPECTED_ADAPTER_MAJOR = 1
        MIN_SUPPORTED_MINOR    = 0

        def __quant_init__(self):
            super().__quant_init__()

            # 1.8 启动期版本校验
            self._check_adapter_version()

            # 2.5 hx/h_n scale 一致性：两端共享同一 EncodingBase（hidden quantizer）
            # input_quantizers[0] = 由 sim 构建期按 mode 决定（fp32_qdq 启用，int16 留 None）
            # input_quantizers[1] / output_quantizers[1] 共享同一 quantizer 引用
            self.input_quantizers  = nn.ModuleList([None, None])
            self.output_quantizers = nn.ModuleList([None, None])

            # 第 8 节：flag 锁
            self._aimet_lock = False

        # --- 1.8 适配版本校验 ---
        def _check_adapter_version(self):
            caps = self.aimet_capabilities()
            major, minor = map(int, caps["adapter_version"].split("."))
            if major != self.EXPECTED_ADAPTER_MAJOR or minor < self.MIN_SUPPORTED_MINOR:
                raise IncompatibleAdapterVersionError(
                    f"QuantGRU adapter_version={caps['adapter_version']} not supported. "
                    f"AIMET requires major == {self.EXPECTED_ADAPTER_MAJOR} "
                    f"and minor >= {self.MIN_SUPPORTED_MINOR}."
                )

        # --- 1.5 校准生命周期 hook（由 sim.compute_encodings 调用）---
        def _aimet_compute_encodings_enter(self):
            with self._aimet_unlock_ctx():
                self.aimet_configure("calibrating")
            self._aimet_lock = True

        def _aimet_compute_encodings_exit(self):
            self._aimet_lock = False
            with self._aimet_unlock_ctx():
                self.finalize_calibration()
                self.aimet_configure(get_quant_execution_mode().value)
            self._aimet_lock = True

        # --- 主 forward：mode 路由 ---
        def forward(self, input, hx=None):
            mode = get_quant_execution_mode().value

            if mode in ("int16_fixed_eval", "int16_fixed_qat_sim"):
                # 走 _DispatchMixin → dispatch_int16_fixed → _dispatch_quantgru_blackbox
                # 内部调 forward_quantized 直出整数（bit-exact，2.4 梯度协议生效）
                return super().forward(input, hx)

            # FP32 / FP32_QDQ / FP16_QDQ / 校准期：走标准 forward
            with self._aimet_unlock_ctx():
                self.aimet_configure(mode)

            return super().forward(input, hx)
```

要点：

- 不引入新 `__init__` 参数；模式切换全部由 `aimet_configure(mode_str)` 单一入口完成。
- INT16_* mode 通过 `_DispatchMixin` 走 `dispatch_int16_fixed`，里面会调 `forward_quantized`。
- 其它 mode 直接走标准 `forward`（fp32 IO），由 `input_quantizers` / `output_quantizers` 在 `_DispatchMixin.forward` 通用路径中处理边界 Q/DQ。
- `_aimet_unlock_ctx()` 与 `_aimet_lock` 见第 8 节。

### 3.1.1 wrapper FP32_QDQ / FP16_QDQ 修复方案

> 现状（截至本修订）：`QuantizedQuantGRU.__quant_init__` 把 `input_quantizers = [None, None]`、`output_quantizers = [None, None]`，且 `forward` 在 INT16 之外的所有模式下直接调 `_OptionalQuantGRU.forward(...)`，**绕过 `_DispatchMixin.forward` 通用路径**。结果 FP32_QDQ / FP16_QDQ 数值与 FP32 完全相同，与 §0.2.1 表中"边界 Q/DQ 由 AIMET wrapper 负责"的约定矛盾。本节是把 wrapper 接回通用路径的修复设计；契约 v1（第 1 章）不变，只调整 AIMET 内部实现。

#### A. 现状问题逐条

1. `input_quantizers[0/1]` / `output_quantizers[0/1]` 始终为 `None`，sim 构建期不会注入 fake-quant 实例，FP32_QDQ / FP16_QDQ 也无 boundary 行为。
2. `forward` 跳过 `_DispatchMixin`，意味着即使外部把 input/output quantizer 显式塞回，也不会被调用。
3. 校准期通过 `_aimet_compute_encodings_enter` 切到 QuantGRU 自身 `_forward_with_calibration`，**未同时**驱动 AIMET 通用 quantizer 收 stats；FP32_QDQ 的 boundary encoding 始终未被生成。
4. FP16_QDQ 路径完全没有 fp16 cast / dequant 链路；与其它 op 在该 mode 下的行为不一致。

#### B. quantizer 配置矩阵

`__quant_init__` 不再硬置 None；保留长度 2 的 `ModuleList`（与 `nn.GRU` IO 对齐：`[input, hx]` / `[output, h_n]`），具体实例由 sim 构建期按 mode 注入：

| 模式 | `input_quantizers[0]` (input) | `input_quantizers[1]` (hx) | `output_quantizers[0]` (output) | `output_quantizers[1]` (h_n) |
|---|---|---|---|---|
| FP32 | None | None | None | None |
| FP32_QDQ | `QuantizeDequantize`（bitwidth 由配置决定） | 共享 hidden quantizer（§2.5） | `QuantizeDequantize` | 共享 hidden quantizer（§2.5） |
| FP16_QDQ | fp16 fake-quant | 共享 hidden fp16 quantizer | fp16 fake-quant | 共享 hidden fp16 quantizer |
| INT16_FIXED_EVAL / INT16_FIXED_QAT_SIM | None（dispatch 接管） | None（dispatch 接管） | None（dispatch 接管） | None（dispatch 接管） |
| 校准期 | 共用 FP32_QDQ 配置（收 stats） | 同上 | 同上 | 同上 |

> hidden 端共享同一 `EncodingBase` 的约束见 §2.5；本表只补 FP*_QDQ 行。

#### C. forward 路由修订

```python
def forward(self, input, hx=None):
    mode = get_quant_execution_mode()
    # ExecutionMode 枚举无 "纯 FP32"；FP32 透传 = FP32_QDQ + 全 None boundary quantizer。
    if mode in (ExecutionMode.INT16_FIXED_EVAL, ExecutionMode.INT16_FIXED_QAT_SIM):
        # INT16：走 dispatch（_dispatch_quantgru_blackbox）；boundary quantizer 已置 None
        return _DispatchMixin.forward(self, input, hx)

    # FP32_QDQ / FP16_QDQ / FIXED_SCALE_QDQ / 校准期：走通用 _DispatchMixin 路径
    self._ensure_hidden_quantizer_shared()
    if not self.calibrating:
        with self._aimet_unlock_ctx():
            _quantgru_aimet_configure(self, resolve_quantgru_mode_str(mode))
    return _DispatchMixin.forward(self, input, hx)
```

#### D. `_builtin_torch_fn_helper`（QuantGRU 专用模板）

QuantGRU 无原子 `torch._builtin_fn`，`_builtin_torch_fn = None`。直接在 `QuantizedQuantGRU` 内复写边界 Q/DQ helper，沿用 `_DispatchMixin` 通用模板：

```python
def _builtin_torch_fn_helper(self, fn):
    def helper(input, hx):
        x_q = _quantize_dequantize_if_applicable(input, self.input_quantizers[0])
        h_q = (
            _quantize_dequantize_if_applicable(hx, self.input_quantizers[1])
            if hx is not None else None
        )
        out, h_n = _OptionalQuantGRU.forward(self, x_q, h_q)
        out = _quantize_dequantize_if_applicable(out, self.output_quantizers[0])
        h_n = _quantize_dequantize_if_applicable(h_n, self.output_quantizers[1])
        return out, h_n
    return helper
```

> hidden 端 `input_quantizers[1]` 与 `output_quantizers[1]` 共享同一对象（§2.5）；helper 调用顺序保证 hx 输入端与 h_n 输出端 scale 一致。

#### E. 校准期协同（修订）

`_aimet_compute_encodings_enter` / `_exit` 不再仅切 QuantGRU 自身 `calibrating` flag；上层 `QuantizationSimModel.compute_encodings`（§3.4）继续递归驱动通用 `input/output_quantizers` 收 stats。两套 stats 互不干扰：

- QuantGRU 自身 `quant_ranges` / `hist_collectors`：用于 INT16 路径的 `forward_quantized` 计算 `(M_int16, r_shift)`。
- AIMET 通用 quantizer：用于 FP32_QDQ / FP16_QDQ 边界 fake-quant。

#### F. `_LOCKED_FLAGS` 不变

`use_quantization` / `calibrating` / `export_mode` / `export_format` 仍然由 AIMET 管控；`input_quantizers` / `output_quantizers` 不进入锁定列表（与其它 `Quantized*` 一致）。

#### G. 测试要求（§4.1 扩展）

| 用例 | 期望 |
|---|---|
| FP32_QDQ 已校准 forward | 数值 = `nn.GRU.forward` 加边界 `QuantizeDequantize`（max abs diff ≤ 5e-3，含 STE 容差） |
| FP32_QDQ 未校准 forward | 自动跳过 fake-quant 或抛清晰提示（"input/output quantizer not initialized"） |
| FP16_QDQ forward | 数值与 fp32 forward 在 fp16 cast 后逐元素 close（atol 与 fp16 ULP 对齐） |
| FP32_QDQ encoding 收集 | `compute_encodings` 退出后 `input_quantizers[0].is_initialized() is True` |
| 切换 ExecutionMode FP32_QDQ → INT16 | input/output_quantizers 自动停用，不影响 INT16 bit-exact |

> 修复完成后，§4.1 表中"FP32_QDQ forward 不校准"用例的"数值与 FP32 一致"应改为"数值与 nn.GRU + Q/DQ 一致"。

#### H. 与 §3.1 当前实现的差异点（合并指引）

| 项 | §3.1（当前实现） | §3.1.1（修复后） |
|---|---|---|
| `input_quantizers` / `output_quantizers` 初值 | 全 None | 长度 2 的 ModuleList，按 mode 注入 |
| `forward` 非 INT16 分支 | 直接调 `_OptionalQuantGRU.forward` | 走 `_DispatchMixin.forward`（仅 FP32 例外） |
| `_builtin_torch_fn_helper` | 未定义（依赖 `_builtin_torch_fn`）| 显式定义 helper（包 `_OptionalQuantGRU.forward`） |
| FP32_QDQ encoding | 不生成 | 由通用 `compute_encodings` 生成 |
| FP16_QDQ 行为 | 等同 FP32 | 边界 fp16 fake-quant |

落地顺序：先合 §3.1.1 修复（PR 不破坏 INT16 路径，仅扩 FP*_QDQ）→ 再补 §G 五条用例 → §4.1 表项措辞同步更新。

### 3.2 `aimet_torch/v2/quantization/affine/fixed_point/adapter.py::dispatch_int16_fixed`

新增 QuantGRU 分支：

```python
def dispatch_int16_fixed(module, *args, **kwargs):
    ...
    if isinstance(module, QuantizedQuantGRU):
        return _dispatch_quantgru_blackbox(module, *args, **kwargs)
    ...

def _dispatch_quantgru_blackbox(module, input, hx=None):
    # 1. 入口：reset/锁住 QuantGRU 内部 flag 为 int16 路径
    with module._aimet_unlock_ctx():
        module.aimet_configure(get_quant_execution_mode().value)

    # 2. 上游量化张量 → fp32（2.4 梯度协议：identity 反向）
    fp_input = _stop_grad_dequantize(input) if _is_quantized_tensor(input) else input
    fp_hx    = _stop_grad_dequantize(hx)    if (hx is not None and _is_quantized_tensor(hx)) else hx

    # 3. 调用 QuantGRU 专用 INT16 入口（1.4 forward_quantized，bit-exact）
    int_out, int_hn = module.forward_quantized(fp_input, fp_hx)

    # 4. 整数张量 → Int16QuantizedTensor 容器（不再量化，仅附 meta；2.4 identity 反向）
    meta = module.get_io_quant_meta()
    out_q = _wrap_int_tensor_with_meta(int_out, meta["output"])
    hn_q  = _wrap_int_tensor_with_meta(int_hn,  meta["hidden"])
    return out_q, hn_q
```

要点：

- **使用 `forward_quantized` 直出整数**，与 QuantGRU 部署 CUDA kernel **bit-exact**。
- `_stop_grad_dequantize` / `_wrap_int_tensor_with_meta` 实现见 2.4 节梯度协议（边界 identity 反向，禁用 STE 叠加）。
- `_is_quantized_tensor` 容忍上游既是 fp32 也是 `Int16QuantizedTensor`。
- `module.get_io_quant_meta()` 是契约 1.3，未校准时抛错。
- FP16_QDQ 路径**不进** `dispatch_int16_fixed`；由 `_DispatchMixin` 通用路径走 fp32 forward + 外层 fp16 quantizer 处理。

### 3.3 `aimet_torch/fixed_point/sim_utils.py`

```python
INT16_DISPATCHABLE_MODULES = (
    ...,
    QuantizedQuantGRU,   # 新增
)

def ensure_output_quantizers_for_int16_eval(sim, ...):
    for name, module in sim.model.named_modules():
        if isinstance(module, QuantizedQuantGRU):
            continue   # QuantGRU 自带 IO 量化语义，跳过
        ...
```

### 3.4 校准协同（compute_encodings 上下文）

在 `QuantizationSimModel.compute_encodings` 的 enter/exit 内，递归遍历子模块，对 `QuantizedQuantGRU` 调 hook：

```python
# QuantizationSimModel.compute_encodings 伪代码
def compute_encodings(self, forward_pass_callback, *args):
    quant_gru_modules = [m for m in self.model.modules() if isinstance(m, QuantizedQuantGRU)]
    for m in quant_gru_modules:
        m._aimet_compute_encodings_enter()
    try:
        forward_pass_callback(self.model, *args)
    finally:
        for m in quant_gru_modules:
            m._aimet_compute_encodings_exit()
    # AIMET 自身的 encoding 收集逻辑继续（针对其它 op）
    ...
```

要点：

- 双向 GRU 内部仍是单一 `QuantGRU` 实例（上下游之间共享内部 forward / reverse 状态），无需特别处理。
- 嵌套 sim（子模型 sim）：递归遍历能命中。

### 3.5 readiness 诊断

新增工具函数：

```python
def diagnose_int16_readiness(sim) -> dict:
    """返回各类问题清单：
      - missing_output_quantizer: list[(name, type)]
      - uninitialized_encoding:  list[(name, type)]
      - uncalibrated_quantgru:   list[name]      # 新增
      - missing_fixed_kernel:    list[(name, type)]
    """
```

QuantGRU 的检查项是 `gru.is_calibrated()`，未校准则报清晰错误。其它项延用现有逻辑。

## 4. 测试与验收

### 4.1 单元测试（新增 `tests/fixed_point/test_quantgru_blackbox.py`）

| 用例 | 期望 |
|---|---|
| FP32 forward | 与 `nn.GRU` 行为一致（数值近似）|
| FP32_QDQ forward 不校准 | 浮点 forward 通过，数值与 FP32 一致 |
| 校准期 forward + 退出后 use_quantization=True | 校准 hook 正确切换状态 |
| INT16_FIXED_EVAL forward | 返回 `Int16QuantizedTensor`；scale/zp 与 `get_io_quant_meta()` 一致 |
| INT16_FIXED_EVAL 未校准 | 抛 `RuntimeError("QuantGRU not calibrated")` |
| FP16_QDQ forward | 不报错，输出 fp16 fake-quant 张量 |

### 4.2 集成测试

含 QuantGRU 的 MRNN 模型：

| 场景 | 期望 |
|---|---|
| `compute_encodings` 完成、INT16_FIXED_EVAL eval | 不抛 KernelNotFoundError、不抛 missing output quantizer |
| 同模型 quant-gru-pytorch 直接 `use_quantization=True` 推理 | 与 AIMET INT16_FIXED_EVAL 数值 bit-exact 等价 |
| 反复 enter/exit `compute_encodings` | QuantGRU 状态正确恢复 |

### 4.3 跨版本兼容回归

quant-gru-pytorch 完成 rescale 改造（POT → M_int16）后：

- AIMET 侧测试**不修改**重新跑通过。
- INT16_FIXED_EVAL 下数值与 quant-gru-pytorch 自身 forward 仍一致。

> 这条是验证契约稳定性的关键。

### 4.4 双向 CI conformance（契约守护）

目的：契约 v1 是双方共同冻结的；任一方破坏协议必须**先报错再合 PR**。

AIMET 侧 CI 项（`tests/fixed_point/test_quantgru_contract.py`）：

| 检查 | 实现 |
|---|---|
| `forward(input, hx) -> (Tensor, Tensor)` 签名稳定 | `inspect.signature` 比对 |
| `forward` IO dtype/device 不变 | dummy forward 后断言 fp32 / cuda |
| `aimet_capabilities()` 关键字段存在且 major == 1 | dict key 与值断言 |
| `get_io_quant_meta()` 返回结构完整 | schema 校验，校准前/后两次 |
| `forward_quantized()` 输出 int16 + 与部署 kernel bit-exact | 与 QuantGRU 自带 `use_quantization=True` 推理逐元素 `==` |
| 公开 flag (`use_quantization` / `calibrating` / `export_mode` / `export_format`) 存在 | `hasattr` + 默认值断言 |

quant-gru-pytorch 侧 CI 项（在他们仓库的 `tests/test_aimet_contract.py`）：

| 检查 | 实现 |
|---|---|
| 同上签名 / dtype / capabilities / IO meta / forward_quantized 自洽 | 不依赖 AIMET，自测 stub |
| `forward_quantized` 与标准 `forward` 数值在 dequant 后一致 | max abs diff < 1 / 2^(bitwidth-1) |
| `aimet_configure(mode)` 五种 mode 字符串都能 round-trip 不抛 | 五个 mode 字符串顺序遍历 |

任一边 CI 红 → 阻塞 PR。**这是契约 v1 冻结后的唯一双向守护机制**。

## 5. 实施顺序

```
Phase 0 [契约冻结]     双方对齐第 1 章；QuantGRU 仓库挂版本号 adapter_version="1.0"
                      （阻塞项，所有后续依赖）

Phase 1 [stub 落地]    quant-gru-pytorch 同事先出 4 个稳定 API 的 stub：
                        - get_io_quant_meta()     1.3（先返回 dummy scale=1.0 也行）
                        - forward_quantized()     1.4（先 round-trip 实现，标 TODO=bit-exact）
                        - aimet_configure(mode)   1.x（只切 self.use_quantization 等已有 flag）
                        - aimet_capabilities()    1.8
                      AIMET 拿 stub 可单测打通；后续切真实现 AIMET 不再改

Phase 2 [AIMET 改造]   3.1 + 3.2 + 3.3 + 3.4：QuantizedQuantGRU 重写 + dispatch
                      第 8 节 flag lock 同步落地
                      4.1 单测全部通过（含 2.4 梯度协议、2.5 hidden scale 共享）

Phase 3 [readiness]    3.5 diagnose_int16_readiness 加入 QuantGRU 检查项

Phase 4 [集成]         4.2 集成测试 + 文档样例 + 与 MRNN 实际 sim 联调

Phase 5 [双向 CI]      4.4 conformance CI 在两个仓库各自接入并红绿守护

Phase 6 [跨版本回归]   quant-gru-pytorch rescale 改造 (POT → M_int16) 合入后，
                      AIMET 侧测试**不修改**重新跑过 → 验证契约 v1 稳定性
```

各 Phase 可独立回滚（git revert）。**Phase 0 是阻塞项**；Phase 1 之后两边并行。

## 6. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| `get_io_quant_meta()` 未及时落地 | INT16_FIXED_EVAL 无法 unblock | 由 quant-gru-pytorch 同事先出 stub（哪怕只返回从 `quant_params.shift_h_` 反算的 scale）|
| FP32_QDQ 与 INT16 路径混淆 | 数值不一致 | 路由表显式约束 FP32_QDQ 走浮点 forward |
| compute_encodings 嵌套 / 多 sim | hook 漏切换 | 递归遍历 + 单元测试覆盖 |
| QuantGRU forward 协议被改 | AIMET 侧崩 | 契约 1.1/1.2 写死 + 4.4 双向 CI 守护 |
| FP16_QDQ 下 QuantGRU 无原生 fp16 | 性能/精度回退 | 路由表显式降级到 fp32 forward + 外层 fp16 quantizer |
| 双向 GRU 内部 reverse 路径 | 状态切换/scale 拿错 | 单元测试覆盖 bidirectional 用例 |
| **QAT 边界双重 STE** | loss 静默异常，调参困难 | 2.4 节梯度协议强制 identity 反向 + 单测对比 backward 数值 |
| **hx/h_n scale 漂移** | 长序列误差累积、BiGRU 反向更严重 | 2.5 节 quantizer 共享同一 EncodingBase + 启动期 sanity check |
| **用户中途改 `use_quantization` flag** | 与 AIMET ExecutionMode 冲突，可能双重量化 | 第 8 节 `_aimet_lock` 拦截写入；同步 raise 引导用户走 sim 接口 |

## 7. 与配套文档的边界

- **本文档** 负责：AIMET 视角的所有接入逻辑、ExecutionMode 路由、校准协同、契约对齐项。
- [Rescale_M_int16_r_shift_实施计划.md](../../quant-gru-pytorch/docs/Rescale_M_int16_r_shift_实施计划.md) 负责：QuantGRU 内部 rescale 由 POT 切换到 (M_int16, r_shift)。

两份文档**唯一的耦合点**是契约 1.x，特别是 1.3 `get_io_quant_meta()` 与 1.4 `forward_quantized()`。除此之外两边可独立演进。

## 8. flag 锁定与 race 防护

> 此节解决「AIMET sim 管控 QuantGRU 时，用户代码或第三方 hook 同时写 `use_quantization` / `calibrating`」的 race / 双重量化问题。

### 8.1 锁机制

`QuantizedQuantGRU` 增加 `_aimet_lock`：

```python
class QuantizedQuantGRU(...):
    _LOCKED_FLAGS = ("use_quantization", "calibrating", "export_mode", "export_format")

    def __setattr__(self, name, value):
        if getattr(self, "_aimet_lock", False) and name in self._LOCKED_FLAGS:
            raise QuantGRUFlagLockedError(
                f"`{name}` is managed by AIMET sim and cannot be set directly. "
                f"Use sim.set_execution_mode() / sim.compute_encodings() instead."
            )
        super().__setattr__(name, value)

    @contextmanager
    def _aimet_unlock_ctx(self):
        """AIMET 内部短暂解锁以切 flag；with 结束自动加锁。"""
        prev = self._aimet_lock
        object.__setattr__(self, "_aimet_lock", False)
        try:
            yield
        finally:
            object.__setattr__(self, "_aimet_lock", prev)
```

锁切换时机：

| 时机 | 锁状态 | 触发位置 |
|---|---|---|
| sim 构建期，`QuantizedQuantGRU.__quant_init__` 末尾 | locked | 模块初始化 |
| `set_execution_mode()` 触发的 `aimet_configure(mode)` 调用 | 短暂 unlock → 切 flag → relock | sim API |
| `sim.compute_encodings` enter | 短暂 unlock → 切到 calibrating → relock | hook |
| `sim.compute_encodings` exit | 短暂 unlock → 切回 mode → relock | hook |
| INT16 dispatch (`_dispatch_quantgru_blackbox`) | 短暂 unlock → reconfigure → relock | adapter |
| 单元测试场景（直接构造 `quant_gru.QuantGRU`，不进 sim）| **永远 unlocked** | 实例由用户控制 |

### 8.2 错误信息约定

```python
class QuantGRUFlagLockedError(RuntimeError):
    """Raised when user tries to mutate AIMET-managed flags directly."""
```

错误文案必须**指向 sim 的正确 API**（`sim.set_execution_mode` / `sim.compute_encodings`），引导用户避免再次踩坑。

### 8.3 与 `quant_gru.QuantGRU` 直接实例的兼容

- 非 AIMET sim 场景下用户直接 `gru = QuantGRU(...)`：**根本不会有 `_aimet_lock` 属性**（只在 `QuantizedQuantGRU.__quant_init__` 内才会设置）。原仓库使用方式零侵入。
- AIMET sim 包装 (`prepare_model` → `QuantizedQuantGRU` 实例)：lock 生效。

> 这条保证 QuantGRU 团队不需要为锁机制改任何代码。锁逻辑完全在 AIMET 子类。

## 附 A. 文件改动清单

| 文件 | 动作 |
|---|---|
| `aimet_torch/v2/nn/modules/custom.py` | 改写 `QuantizedQuantGRU`（3.1）+ 第 8 节 lock 机制 |
| `aimet_torch/v2/quantization/affine/fixed_point/adapter.py` | 新增 `_dispatch_quantgru_blackbox` 走 `forward_quantized`（3.2）|
| `aimet_torch/fixed_point/sim_utils.py` | `INT16_DISPATCHABLE_MODULES` + `ensure_output_quantizers_for_int16_eval` 跳过逻辑（3.3）|
| `aimet_torch/v2/quantization/quantsim.py`（或对应 sim 入口） | `compute_encodings` enter/exit hook 调度（3.4）|
| `aimet_torch/fixed_point/diagnose.py`（新增） | `diagnose_int16_readiness`（3.5）|
| `aimet_torch/fixed_point/gradient_helpers.py`（新增） | 2.4 `_stop_grad_dequantize` / `_stop_grad_pack` |
| `aimet_torch/fixed_point/errors.py`（新增或追加） | `IncompatibleAdapterVersionError` / `QuantGRUFlagLockedError` |
| `tests/fixed_point/test_quantgru_blackbox.py`（新增） | 单元测试（4.1）|
| `tests/fixed_point/test_quantgru_contract.py`（新增） | 4.4 双向 CI conformance（AIMET 侧）|
| `tests/fixed_point/test_quantgru_qat_grad.py`（新增） | 2.4 梯度协议数值对比 |
| `tests/fixed_point/test_mrnn_int16_e2e.py`（新增或扩） | 集成测试（4.2）|

## 附 B. 关键代码片段索引（便于 AI 实现）

- 现有黑盒 wrapper：`aimet_torch/v2/nn/modules/custom.py::QuantizedQuantGRU`
- dispatch 入口：`aimet_torch/v2/nn/true_quant.py::_DispatchMixin.forward`
- INT16 dispatch：`aimet_torch/v2/quantization/affine/fixed_point/adapter.py::dispatch_int16_fixed`
- output quantizer 工具：`aimet_torch/fixed_point/sim_utils.py::ensure_output_quantizers_for_int16_eval`
- 已有测试参考：`tests/fixed_point/test_v2_int16_adapter.py`
- QuantGRU 现有内部 forward：`quant-gru-pytorch/pytorch/quant_gru.py::QuantGRU._forward_cuda`
- QuantGRU 现有 quant kernel：`quant-gru-pytorch/src/gru_forward_gpu_quant.cu`（`forward_quantized` 须直出该 kernel 输出，跳过末尾 dequantize）

## 附 C. INT16 / FP*_QDQ 后期 TODO（按优先级）

> 本节列出 plan 当前未完成的硬骨头。每条 TODO 都不能在不破坏现有路径的前提下静默落地，必须有独立 PR + 单测 + 回滚说明。优先级 P0 > P1 > P2。

### R0 wrapper FP32_QDQ / FP16_QDQ 修复 [P0]

- **状态**：`done`（2026-05-28）
- 影响：FP32_QDQ / FP16_QDQ 当前等同于 FP32，与 §0.2.1 / §2 路由表矛盾；MobileNet 等其它算子在该 mode 下行为已对齐，仅 QuantGRU 缺位。
- 行动：按 §3.1.1 修 `aimet_torch/v2/nn/modules/custom.py::QuantizedQuantGRU` 的 `__quant_init__` / `forward` / `_builtin_torch_fn_helper`，并补 §3.1.1 G 节五条用例到 `tests/fixed_point/test_quantgru_blackbox.py`。
- 已落地：
  - `forward` 在 FP32_QDQ / FP16_QDQ / FIXED_SCALE_QDQ / 校准期 走 `_DispatchMixin.forward`；INT16 由 `dispatch_quantgru_blackbox` 全权接管。
  - 重写 `_builtin_torch_fn_helper`：双输出 GRU 的 input/output_quantizers[0/1] 边界 Q/DQ；FP16_QDQ 强制 fp32 cast 满足 §1.1 入口协议。
  - **审计修复**：删除对不存在的 `ExecutionMode.FP32` 的分支；G-1/G-3 reference 改走底层 `QuantGRU.forward`。
  - 测试 G-1..G-5 已加（`tests/fixed_point/test_quantgru_blackbox.py`），本机 quant_gru + CUDA 已全绿。
- 阻塞：无。仅改 AIMET 单仓，不动契约 v1。
- 验收：§3.1.1 G 五条用例全绿；既有 INT16 测试不回归。
- 回滚：单 PR `git revert` 即可；INT16 路径与 FP32 路径无依赖。

### R0+ §2.5 hidden 端 quantizer 共享同一 EncodingBase [P0]

- **状态**：`done`（2026-05-28）
- 影响：sim builder (`_V2LazyQuantizeWrapper.realize`) 默认 propagate 给 `input_quantizers[1]` / `output_quantizers[1]` 注入两个独立实例；长序列 + BiGRU 双向会引入累积漂移误差，违反 §2.5 不变量。
- 行动：在 `QuantizedQuantGRU` 内 lazy share，不侵入 sim builder 通用路径。
- 已落地：
  - 新增 `_ensure_hidden_quantizer_shared()`：在 FP32_QDQ / FP16_QDQ / 校准期分支首次 `forward` 入口处把 `input_quantizers[1]` 替换为 `output_quantizers[1]` 的同一引用；幂等。
  - INT16 路径不触发（dispatch 不读这些 quantizer）。
  - 测试 G-6 `test_hidden_quantizer_shared_after_first_forward` + G-7 `test_hidden_share_does_not_apply_to_int16_path` 已加。
- 阻塞：无。
- 验收：G-6 / G-7 全绿；CI 真环境上 BiGRU 长序列 forward 数值漂移收敛。
- 回滚：删除 `_ensure_hidden_quantizer_shared()` 调用即可，sim builder 注入的两端独立实例自动恢复。

### R0++ FIXED_SCALE_QDQ wrapper 别名 [P0]

- **状态**：`done`（2026-05-28）
- 影响：`ExecutionMode.FIXED_SCALE_QDQ` 不在 QuantGRU contract v1 `_SUPPORTED_MODES` 内；若直接把 `mode.value` 传给 `aimet_configure` 会抛 `ValueError`。FIXED_SCALE 网格由 AIMET boundary `QuantizeDequantize` 内部按 ExecutionMode 自动切换（plan §0.2.1），QuantGRU 内部只需浮点 forward。
- 行动：在 adapter 层新增 `resolve_quantgru_mode_str()`，把 `FIXED_SCALE_QDQ` 别名为 `"fp32_qdq"`；`forward` 与 `_aimet_compute_encodings_exit` 统一调用。
- 已落地：
  - `aimet_torch/fixed_point/quantgru_adapter.py::resolve_quantgru_mode_str`
  - 测试 G-8 `test_fixed_scale_qdq_forward_via_wrapper_alias` + G-9 `test_compute_encodings_inside_fixed_scale_qdq_context_does_not_raise`
  - `tests/fixed_point/test_quantgru_adapter_helpers.py`（7 条本机静态守卫，不依赖 quant_gru）
  - `tests/fixed_point/test_int16_dispatch_cpu_smoke.py` 新增 2 条 FIXED_SCALE_QDQ MRNN 替身用例
- 阻塞：无。
- 验收：G-8 / G-9 全绿；FIXED_SCALE_QDQ vs FP32_QDQ max abs diff ≤ 0.5（POT 下 = 0）。
- 回滚：删除 `resolve_quantgru_mode_str` 别名一行即可；不影响 INT16 路径。

### R1 QAT 反向梯度严格匹配 `backward_quant` [P0]

- **状态**：`done`（2026-05-28）
- 落地：`INT16_FIXED_QAT_SIM` 改调 `_OptionalQuantGRU.forward`（`GRUFunction` + `backward_quant`）；eval 仍 `forward_quantized`；`_invoke_native_quantgru_forward` 防 wrapper 递归。
- 验收：`test_quantgru_qat_grad.py` 全绿；与原生 `QuantGRU.forward` 输入/权重梯度对齐（atol 1e-5）。
- 未做：公开 `backward_quant` Python API（非必须）；BiGRU 专项 golden（可后续补）。
- e2e：`tests/fixed_point/test_mrnn_int16_qat_smoke.py`（全图 QAT_SIM forward + backward + 多步 SGD）；`test_quantgru_qat_grad.py::test_qat_grad_bidirectional_matches_native`（BiGRU）；`examples/quick_start_int16_metric.py --qat-train-steps`。
- 回滚：QAT_SIM 分支改回 `forward_quantized`；eval 不变。

### R2 全图 INT16 MRNN backbone 分段 [P1]

- **状态**：`done`（2026-05-28：backbone 分段 e2e 本机 quant_gru + CUDA 全绿）
- 影响：原以为缺 CUDA int matmul fallback；探查后 fallback 已存在。真正阻塞项是：
  1. ``FixedPointSimTensor`` 缺 ``.shape`` / ``.permute`` 等 layout 属性（FX 图访问失败）
  2. ``model_preparer`` 把 ``b*f`` 拆成 ``QuantizedMultiply``，其输入来自 shape 整型标量，output quantizer 无法在校准期初始化
- 已落地：
  - ``aimet_torch/fixed_point/tensor.py``：补 ``shape``/``ndim``/``device``/``permute``/``view``/``contiguous`` 等 layout API
  - ``aimet_torch/fixed_point/shape_meta.py`` + ``adapter.dispatch_int16_fixed``：**shape 标量二元运算**（``b*f`` / ``t+pad`` 等）旁路 INT16 quant dispatch，不要求 output quantizer
  - ``tests/fixed_point/test_shape_meta_int16.py``（单元 + dispatch 集成）
  - ``tests/fixed_point/test_int16_tensor.py::test_sim_tensor_layout_introspection_and_permute``
  - ``tests/fixed_point/test_mrnn_int16_e2e.py`` 使用 ``view(b*f,...)`` **10/10 全绿**
  - ``tests/fixed_point/test_int16_dispatch_cpu_smoke.py``（6 条 CPU 回归）
- 行动（剩余，R3 范围）：
  1. MRNN **完整**图（含 STFT / BandConverter 前端）INT16 适配
  2.（可选）评估 ``mac_accumulator_int32_sat_enabled`` 严格模式覆盖率
- 阻塞：完整 MRNN 端到端 metric 仍受 R3 DSP 前端 kernel 缺失影响。
- 验收：backbone 分段 INT16 forward 不抛、输出 shape 正确；GRU 边界 bit-exact；``view(b*f,...)`` 与 ``view(-1,...)`` 均可用。
- 回滚：``shape_meta`` 旁路与 tensor layout 属性独立可 revert。

### R3 MRNN 前端算子 INT16 适配 [P2]

- **状态**：`done (decomposed)`（2026-05-28：全图 decomposed INT16；PowerCompress/Hypot/CLN 经 Square/Sqrt/Abs/Sign/Divide/Mean 分解）
- 影响：无 FP32 preserve 叶子；完整 MRNN 图走 INT16 dispatch（reference float 路径：Abs/Sign/Divide；CLZ：Sqrt/Square）
- 已落地：
  - PowerCompress / HypotFun / CLN trace-friendly 重写（``clamp(min=EPS)`` 代替 ``+EPS`` 常数 Add）
  - ``AbsInt16Kernel`` / ``SignInt16Kernel``（`kernels/eltwise.py`）
  - ``examples/quick_start_int16_metric.py``：无 ``modules_to_exclude``；``diagnose`` 全绿；metric Δ=0 pp
- 行动（可选）：
  1. bit-exact 整数语义（Pow0.5 / Hypot LUT）替代 reference float 路径
  2. STFT ``forward_quantized`` 黑盒（DSP 契约 §2）
  3. ``--fp-epochs 1`` 下 meaningful absolute metric
- 验收：metric vs FP32_QDQ 偏差 ≤ 0.5 pp；``diagnose_int16_readiness`` 无 blocker。
- 回滚：Abs/Sign kernel 与 quick_start 算子重写独立可 revert。

### 跟踪与回滚成本汇总

| TODO | 优先级 | 状态 | 跟踪 issue（占位） | 阻塞 | 回滚成本 |
|---|---|---|---|---|---|
| R0 | P0 | done | `aimet#TODO/QuantGRU-fpqdq-fix` | 无 | 小（仅 wrapper + 测试） |
| R0+ | P0 | done | `aimet#TODO/QuantGRU-hidden-share` | 无 | 极小（删 forward 一行调用） |
| R0++ | P0 | done | `aimet#TODO/QuantGRU-fixed-scale-alias` | 无 | 极小（删 resolve 别名） |
| R1 | P0 | done | `aimet#TODO/QuantGRU-qat-grad` | 无 | 中（涉及双仓同步） |
| R2 | P1 | done (backbone) | `aimet#TODO/INT16-mrnn-fullgraph` | R3 DSP 前端 | 中（tensor layout + view 写法独立 revert） |
| R3 | P2 | done (decomposed) | `aimet#TODO/INT16-DSP-frontends` | bit-exact LUT 可选 | 中 |

> 任一 TODO 若临时阻塞，应回到 §6 风险表追加一行新风险并标注缓解措施，确保 plan 与实际进度始终一致。

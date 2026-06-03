# MRNN 整图 QDQ 精度排查报告

> 背景：用户报告 `fp32_qdq` / `fp16_qdq` / `fixed_scale_qdq` 三档相对新 baseline (`float_native = 95.56%`) 大幅劣化（67% / 67% / 1.5%），且 QAT 也无法恢复。"不符合常理"。

## 0. TL;DR — 修两个 bug 后三档全部追平 float_native

| 模式 | 修复前 | 修复后 | Δ vs `float_native` |
| --- | --- | --- | --- |
| `float_native` | 95.56% | 95.56% | — |
| `fp32_qdq` | **67.43%** | **95.54%** | -0.02 pp |
| `fp16_qdq` | 67.74% | **95.47%** | -0.09 pp |
| `fixed_scale_qdq` | **1.50%** | **95.57%** | +0.01 pp |

QAT 在新 baseline 上不再必要（PTQ 已追平上限）；现有 QAT 流程的数值不稳定是独立技术债，由 `val_checkpoint` 保护已无回退风险。

---

## 1. 根因①：`apply_mixed_precision_bitwidth` 优先级 bug → fp32/fp16_qdq −26~28 pp

### 现象

- 默认 8bit + percentile 校准下，`fp32_qdq` 卡在 67.43%（vs float 95.56%，−28.13 pp）
- 既有 `pc1_hypot_16bit.json`（D/E 档）用通配 `power_compress_1.*` / `hypot_fun.*` 想把 frontend 数学链整段升 16bit，**结果跑出来不变**

### 定位

block-level QDQ disable ablation (`examples/int16_whole_graph_block_disable.py`)：

| disable scope | 数 | fp32_qdq | Δ |
| --- | --- | --- | --- |
| baseline | 0 | 67.43% | 0 |
| `power_compress_1` | 8 | 85.40% | +17.96 pp |
| `hypot_fun` | 9 | 79.67% | +12.24 pp |
| `pc1+hypot` | 17 | **94.18%** | **+26.74 pp** |
| `pc1+hypot+enc_seqs.0+enc_seqs.1` | 77 | 94.91% | +27.48 pp（收益饱和）|

→ **frontend `power_compress_1` + `hypot_fun` 两块 quantizer 联合作用占了 ~26 pp 损失**。再做 input-only / output-only ablation：单独 disable input 或 output 都无效，必须整段连续 dequantize 才放行信号——典型的 QDQ 链路里"任意一个截断都让链路失真"。

### 根因

打 `apply_mixed_precision_bitwidth(verbose=True)` 日志：

```
✅ [pattern: power_compress_1.*] power_compress_1.module_sign.input: 16-bit, sym
✅ [type]                       power_compress_1.module_abs_1.input: 8-bit, sym
✅ [type]                       power_compress_1.module_sqrt.input:  8-bit, sym
✅ [type]                       power_compress_1.module_mul.input:   8-bit, sym
...
```

原匹配优先级为 `精确 name > type > 通配 pattern`，原意是"避免宽泛通配（如 `*.act*`）意外覆盖子模块的 type 配置"。但当用户写显式 module-tree wildcard 想覆盖默认 type 时——`QuantizedAbs / Sqrt / Multiply / Add` 等 type 在 `layer_type_config` 里已写 8bit——**type 把 wildcard 吃掉**，只剩 `QuantizedSign / Square / Clamp` 等 type 无定义的算子走 pattern 升到 16bit，整条 frontend 数学链实际仍是 8bit。

### 修复（opt-in，向后兼容）

`aimet_torch/utils_rx.py` `apply_mixed_precision_bitwidth` 新增顶层 `pattern_priority` (bool, default `false`)：
- 默认（兼容老 config）：`精确 name > type > pattern`
- `pattern_priority: true`：`精确 name > pattern > type`

更新 `examples/config/pc1_hypot_16bit.json`：恢复简洁 wildcard 写法 + 顶层 `pattern_priority: true`，结果与精确 module name 一致。

---

## 2. 根因②：`QuantizedDivide` 在 fixed_scale_qdq 下 0/0 → NaN 传染全图 → −93 pp

### 现象

修完根因①后 `fp32_qdq=94.28%` `fp16_qdq=94.17%`，但 `fixed_scale_qdq` 仍 1.50%——差 ~93 pp，绝不可能是 M,rshift 量化精度问题。

### 定位

逐模块 hook 跟踪 NaN 出现顺序：

- 第一个 NaN 出现在 `enc_seqs.0.cln.module_div_2`（QuantizedDivide），输入无 NaN、输出 14523 NaN，之后整图沿着 `conv_t / div / sqrt` 全部传染。
- 分母（`sqrt_3` 输出）：`fp32_qdq` 25 个 0，`fixed_scale_qdq` 同样有 25 个 0（前者 2 个，本批 25 个；两模式分布接近）。
- `fp32_qdq` 下 `div_2` 输出 NaN=0，`fixed_scale_qdq` 输出 NaN=14523——同样 0/0 输入，结果不同。

进一步对比：`fp32_qdq` 下 `div_2.in[1]` 等于 0 的位置上，`in[0]` 也是 0（589 / 590 个 0），但 `div_2` 输出仍然全部非 NaN。说明 fp32_qdq 的 fake-quant dispatch 路径里某种容差（dequant 后的微小残差、cast 路径、或 dispatcher 内部 epsilon）实际把 0/0 给吃掉了，**而 fixed_scale_qdq 的 _FixedScaleQuantDequantFunc 是真正的 exact 0**，调到 `torch.div(0.0, 0.0)` 就标准 NaN。

### 根因

`QuantizedDivide._builtin_torch_fn = torch.div`，**无 0/0 保护**。CLN 内 `x / sqrt(clamp(var, min=eps))`：
- fp32_qdq：实际 div(0, 0) 也是 NaN，但被某种隐式容差掩盖
- fixed_scale_qdq：div(0, 0) → NaN → 后续 conv2d/sub/sqrt 全部传染 → 整图崩

### 修复

`aimet_torch/v2/nn/modules/custom.py`：

```python
def _safe_div(x, y, *args, **kwargs):
    out = torch.div(x, y, *args, **kwargs)
    if not isinstance(y, torch.Tensor) or not y.is_floating_point():
        return out
    return torch.where(y == 0, torch.zeros_like(out), out)

class QuantizedDivide(_DispatchMixin, QuantizationMixin, Divide):
    _builtin_torch_fn = _safe_div
```

纯 tensor op，无 python-level `if`（保留 tracing / TorchScript export 兼容性）。语义：`0/0 → 0`、`x/0 → 0`——与 fp32_qdq 的隐式行为一致，且下游 `mul/sub/conv` 再归一化时不放大异常。

副作用：fp32/fp16_qdq 也分别提升 +1.26 / +1.30 pp（说明原本就有偶发 0/0 噪声被遮住而不自知）。

---

## 3. 为何 QAT 在新 baseline 上没有再帮上忙

| | 旧 baseline 85% | 新 baseline 95.56% |
| --- | --- | --- |
| PTQ 损失主要来自 | 普通 8bit 截断 | frontend 数学链整段 8bit（type 覆盖 pattern bug） + fixed_scale 路径 NaN |
| QAT 之前 PTQ 精度 | ~80% | 67.43% |
| QAT 能补的差 | ~5 pp | ~28 pp（其中 ~26 pp 是 bug，QAT 补不了；剩 ~2 pp QAT 应能补但需稳定训练） |
| QAT 自身稳定性 | OK | 训练早期出 NaN gradient，loss 卡 3~7（已观察） |

→ 用户记忆里"QAT 能恢复 85% baseline"完全合理；新 baseline 下精度跌坑的两个主因（pattern_priority bug + Divide 0/0）**不是 QAT 能补的类型**，必须从 bug 维度修。**修完两个 bug 后 PTQ 自身就追平 float_native，QAT 不再必要**。

quick-QAT 验证（1 ep / lr=1e-5）：QAT loss 仍不稳定（3.6258，val Top-1 3.37%），但 `val_checkpoint` 自动回滚到 PTQ 最佳值，**最终三档与 skip_qat 完全一致**（95.54 / 95.47 / 95.57）——无回退风险。

---

## 4. 代码与配置改动清单

| 文件 | 改动 |
| --- | --- |
| `aimet_torch/v2/nn/modules/custom.py` | 新增 `_safe_div(x, y)`，`QuantizedDivide._builtin_torch_fn = _safe_div`（0/0 → 0） |
| `aimet_torch/utils_rx.py` | `apply_mixed_precision_bitwidth` 新增顶层 `pattern_priority` opt-in（`false` 兼容老行为；`true` 时 pattern > type） |
| `examples/config/pc1_hypot_16bit.json` | 恢复简洁 wildcard 写法 + `"pattern_priority": true` |
| `examples/int16_whole_graph_vs_float_native.py` | 新增 `--bitwidth-config` 参数 |
| `examples/int16_whole_graph_block_disable.py` | 新增：block-level disable ablation 工具（含 frontend 子算子下钻、组合 disable） |

---

## 5. 复现

```bash
cd examples
PYTHONPATH=/home/llq/workspace/aimet_rx-main:/home/llq/workspace/quant-gru-pytorch/pytorch \
  python3 int16_whole_graph_vs_float_native.py \
    --max-calib-batches 100 \
    --bitwidth-config config/pc1_hypot_16bit.json \
    --output output/int16_whole_graph_pattern_priority.json
```

预期输出：

```
fp32_qdq          95.54%  (Δ vs float_native -0.02 pp)
fp16_qdq          95.47%  (Δ vs float_native -0.09 pp)
fixed_scale_qdq   95.57%  (Δ vs float_native +0.01 pp)
```

如果要复现 67.43% / 1.50% 的旧坑：把 `pc1_hypot_16bit.json` 顶层的 `pattern_priority` 从 `true` 改为 `false`（精确 name 也删了），并暂时把 `QuantizedDivide._builtin_torch_fn` 还原为 `torch.div`。

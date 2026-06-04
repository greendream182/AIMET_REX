# AIMET RX Examples

This directory contains small entry points for common AIMET RX workflows.

## `quick_start.py`

End-to-end MRNN keyword-spotting flow covering model preparation, PTQ
calibration, power-of-2 scale alignment, QAT fine-tuning, ONNX export,
encodings export, INT16 sidecar export, and reload validation.

## `fixed_point_minimal.py`

Minimal fixed-point execution-mode demo. It builds a tiny initialized
`QuantizedLinear` and compares:

- `fp32_qdq`: standard AIMET Q/DQ reference path
- `fixed_scale_qdq`: G2 fixed-scale Q/DQ path
- `int16_fixed_eval`: G3 integer carrier and fixed-kernel path

Run:

```bash
python examples/fixed_point_minimal.py
```

## `int16_fixed_qat_sim_minimal.py`

Minimal INT16 fixed-point QAT simulation demo. It trains a tiny quantized
student against a float teacher with forward passes under
`int16_fixed_qat_sim`, then checks the result once with `int16_fixed_eval`.

This is an API and training-loop skeleton. It is not a production QAT recipe and
does not replace INT16 eval, hardware correlation, or board-level sign-off.

Run:

```bash
python examples/int16_fixed_qat_sim_minimal.py
```

## `quick_start_int16_metric.py`

MRNN 全图 **INT16_FIXED_EVAL** 验收（默认仅评 INT16；QDQ 三档请用
``int16_whole_graph_vs_float_native.py``）。可选 QAT（R1：`QuantGRU` → `backward_quant`）。

**Scale（默认，对齐 Design v2）**：校准 → CLZ fix → `convert_encodings_to_fixed_scale`
→ 分解层 **`M_int16/2^rshift`**；**QuantGRU** 黑盒内部仍用 **`2^(-shift)`**。
**不默认全图 Po2**（legacy Ada200 习惯；新项目勿用）；需要对比时用 `--apply-po2`。

```bash
cd /path/to/aimet_rx-main
python examples/quick_start_int16_metric.py \
  --data-root /home/llq/workspace/data/speech_commands \
  --bitwidth-config config/pc1_hypot_16bit.json \
  --native-trans \
  --max-calib-batches 100
```

`--qat-epochs` 结束后自动对比 QAT 前后 `int16_fixed_eval`（`--skip-qat-post-eval` 可关）。
固定步数：`--qat-train-steps`（与 `--qat-epochs` 二选一，优先 epochs）。

## `int16_whole_graph_vs_float_native.py`

整图 **QDQ 三档 + 可选 INT16 真 kernel** vs float_native。INT16 与 QDQ **分离 build**。
默认无全图 Po2；``--eval-int16`` 为 INT16 验收入口。

## Power-of-2（Po2）说明

| 用途 | 是否默认 |
|------|----------|
| 新项目 INT16 验收（whole_graph / int16_metric / single_op） | **否** — `(M,rshift)` |
| `quick_start.py` 旧端到端演示 | 是（历史行为） |
| `--apply-po2` / probe 脚本 | 可选 legacy 对比 |

## `freeze_int16_fixed.py`

CLI helper for freezing calibrated QuantSim encodings into an INT16 deployment
sidecar JSON. Use `--demo` for a smoke test, or pass `--checkpoint` from a
larger export flow.

Run:

```bash
python examples/freeze_int16_fixed.py --demo --output exports/model.int16.json
```

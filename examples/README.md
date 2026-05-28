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

## `freeze_int16_fixed.py`

CLI helper for freezing calibrated QuantSim encodings into an INT16 deployment
sidecar JSON. Use `--demo` for a smoke test, or pass `--checkpoint` from a
larger export flow.

Run:

```bash
python examples/freeze_int16_fixed.py --demo --output exports/model.int16.json
```

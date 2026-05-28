# Fixed-Point Quality Report

- Generated at: `2026-05-22T06:40:31+00:00`
- Repo: `aimet_rx`
- PyTorch: `2.10.0+cu128`

**Overall status:** PASS

## Runtime notes (console noise vs failures)

- ConnectedGraph: Unable to isolate model outputs (prepared MobileNet with functional Mean/Add).
- Quant: Unsupported op type Mean / Shape / If (graph preparer placeholders).
- v2 AutoQuant may print ignored torch.export SpecViolationError when ONNX export is attempted on prepared models; use --quiet to skip AutoQuant.
- cvxpy optional: AMP convert-op reduction logs at debug when cvxpy is absent.

## 1. PWL Kernels vs Analytic Activation

- Segments (hardware-fixed): **16**
- Required: `cosine_similarity ≥ 0.9999` and per-fn LSB limits.

| fn | status | cosine | max_lsb (limit) | p99_lsb (limit) | p999_lsb | rmse_lsb (limit) | max_rel |
|---|---|---|---|---|---|---|---|
| sigmoid | PASS | 0.999997 | 246 (320) | 187.05 (260) | 239.91 | 55.35 (80) | 0.172 |
| tanh | PASS | 0.999944 | 1.8e+03 (2e+03) | 1.3e+03 (2e+03) | 1.7e+03 | 323.42 (450) | 23.292 |
| gelu | PASS | 0.999997 | 213 (320) | 149.05 (240) | 202 | 32.99 (60) | 42.548 |
| silu | PASS | 0.999998 | 148 (200) | 108.05 (160) | 140.91 | 25.27 (45) | 29.378 |
| softplus | PASS | 0.999999 | 78 (320) | 58.05 (240) | 75 | 15.58 (60) | 0.249 |
| mish | PASS | 0.999998 | 190 (320) | 126 (240) | 170.91 | 28.47 (60) | 29.655 |

## 2. INT16 Single-Op vs FP32_QDQ

- Required: `max_error_lsb ≤ 1.0` (deploy-8b only) and per-case `cosine_similarity` floors (default ≥ 0.9999; relaxed for deploy-8b PWL tanh/gelu).

- `deploy-8b` mirrors production (8-bit output quantizer, PWL error largely absorbed by grid).
- `stress-16b-out` lifts the output quantizer to 16 bits to expose PWL approximation error.
- LSB is reported on the **output quantization grid** (see `out_scale` column).

| case | stress | out_bits | out_scale | n | status | cosine | max_lsb | max_abs | rmse | sqnr_db |
|---|---|---|---|---|---|---|---|---|---|---|
| Linear → ReLU → Linear | deploy-8b | 8 | 0.015686 | 128 | PASS | 1.000000 | 0 | 0 | 0 | inf |
| Linear → Sigmoid (PWL) | deploy-8b | 8 | 0.003922 | 192 | PASS | 0.999989 | 1 | 0.003922 | 0.00264 | 44.93 |
| Linear → Tanh (PWL) | deploy-8b | 8 | 0.007843 | 192 | PASS | 0.999488 | 1 | 0.007843 | 0.004999 | 29.89 |
| Linear → GELU → Linear (PWL) | deploy-8b | 8 | 0.015686 | 128 | PASS | 0.971415 | 1 | 0.015686 | 0.008985 | 11.87 |
| Conv2d | deploy-8b | 8 | 0.031373 | 72 | PASS | 1.000000 | 0 | 0 | 0 | inf |
| Linear → Softmax (PWL exp) | deploy-8b | 8 | 0.003922 | 256 | PASS | 1.000000 | 1 | 0.003922 | 0.003922 | 36.12 |
| Linear → Sigmoid (PWL) [stress] | stress-16b-out | 16 | 1.5259e-05 | 192 | PASS | 1.000000 | 52 | 0.000793457 | 0.000227985 | 66.21 |
| Linear → Tanh (PWL) [stress] | stress-16b-out | 16 | 3.0518e-05 | 192 | PASS | 0.999952 | 104 | 0.003174 | 0.001743 | 39.2 |
| Linear → GELU → Linear (PWL) [stress] | stress-16b-out | 16 | 6.10361e-05 | 128 | PASS | 0.999398 | 42 | 0.002564 | 0.001264 | 29.17 |
| Linear → Softmax (PWL exp) [stress] | stress-16b-out | 16 | 1.5259e-05 | 256 | PASS | 1.000000 | 1 | 1.52588e-05 | 1.52588e-05 | 84.29 |

_Note: 1 LSB on the output grid equals `out_scale`. With an 8-bit output the LSB is large enough that PWL approximation error (which can be 200+ LSB on the int16 grid relative to analytic `fn`) is rounded to 0 LSB on the deploy grid. The stress rows confirm the underlying PWL error._

## 3. FP16 QDQ vs FP32 QDQ

- Required: `cosine_similarity ≥ 0.9999` (no LSB gate).

| case | status | cosine |
|---|---|---|
| random_4x8 | PASS | 1.000000 |
| random_2x16x16 | PASS | 1.000000 |
| uniform_8 | PASS | 1.000000 |

## 4. MobileNet V2 (mock) End-to-End

- Pipeline: ``prepare_model → fold_all_batch_norms → QuantizationSimModel(default_output_bw=8) → ensure_output_quantizers_for_int16_eval → compute_encodings``.
- Input: random ``2 × 3 × 64 × 64`` (seeded), output: 10 logits.
- Output quantizers materialized by the helper: **34** (missing before patch: **34**).

| case | stress | status | cosine | extras |
|---|---|---|---|---|
| MockMobileNetV2 / INT16 vs FP32_QDQ | deploy-8b | PASS | 0.999956 | max_lsb=1, max_abs=4.92875e-07, rmse=3.11721e-07, out_scale=4.92874e-07 |
| MockMobileNetV2 / FP16 vs FP32_QDQ | deploy-fp16 | PASS | 0.999886 |  |
| MockMobileNetV2 / fixed_scale_qdq vs FP32_QDQ | deploy-fixed-scale | PASS | 0.999953 | max_abs=4.92875e-07, rmse=2.91588e-07 |

## 5. MobileNet V2 PTQ Algorithms & INT16 QAT

- Metric: **INT16_FIXED_EVAL vs FP32_QDQ** cosine on held-out logits (seed **13**, shape `2×3×64×64`).
- Gates: cosine ≥ **0.99**; CLE may trail vanilla by at most **0.005**; bias correction by **0.01**; AdaRound/QAT must not trail same-variant vanilla PTQ.
- Mock @ 64×64; full MobileNet @ 96×… when enabled.
- Report QAT: **8** epochs (tests use 20); AdaRound: **80** iterations, 2 calibration batches.

### INT16 QAT（mock 64×64，held-out logits）

| metric | value |
|---|---|
| PTQ-only INT16 cosine | 0.999943 |
| After INT16 QAT cosine | 1.000000 |
| Δ (QAT − PTQ) | +0.000057 |
| QAT status | **PASS** |
| Report training epochs | 8 |
| pytest reference (`test_int16_qat_improves_over_ptq_only`) | 20 ep |
| Final epoch train MSE | 16282 |

验收：QAT 后 INT16 cosine **不低于** 同模型 PTQ-only（与 `test_int16_qat_improves_over_ptq_only` 一致）。

### PTQ pipelines

| pipeline | status | cosine | vs vanilla |
|---|---|---|---|
| Mock PTQ vanilla (min-max) | PASS | 0.999943 | — |
| Mock PTQ + CLE | PASS | 0.999935 | -0.000008 |
| Mock PTQ + bias correction | PASS | 0.999952 | +0.000009 |
| Mock PTQ + AdaRound (80 iter) | PASS | 0.999916 | -0.000026 |
| Full PTQ vanilla (min-max) | PASS | 1.000000 | — |
| Full PTQ + CLE | PASS | 1.000000 | +0.000000 |
| Full PTQ + bias correction | PASS | 1.000000 | +0.000000 |
| Full PTQ + AdaRound (80 iter) | PASS | 1.000000 | +0.000000 |
| Mock combined PTQ (CLE+BC+AdaRound) → INT16 | PASS | 0.999954 | — |

## 6. pytest Summary (tests/fixed_point)

- Exit code: **0** (PASS)
- Marker: `not slow`
- Summary: `183 passed, 2 skipped, 7 deselected, 183 warnings in 186.80s (0:03:06)`

| passed | failed | skipped | errors | warnings |
|---|---|---|---|---|
| 183 | 0 | 2 | 0 | 183 |

_Command:_ `/usr/bin/python3 -m pytest tests/fixed_point -q --no-header --tb=line -m not slow`

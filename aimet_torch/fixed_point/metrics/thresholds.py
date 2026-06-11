# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
"""Default acceptance thresholds for fixed-point mode comparisons."""

# int16_fixed_eval vs fp32_qdq (per-tensor output grid)
INT16_VS_FP32_MAX_ERROR_LSB = 1.0
INT16_VS_FP32_MIN_COSINE_SIMILARITY = 0.9999

# INT16 Add — Path A (kernel equiv): ``ref = dequant(A) + dequant(B)`` on fixed integer inputs.
ADD_INT16_VS_DEQUANT_SUM_MAX_ERROR_LSB = INT16_VS_FP32_MAX_ERROR_LSB
ADD_INT16_VS_DEQUANT_SUM_MIN_COSINE_SIMILARITY = INT16_VS_FP32_MIN_COSINE_SIMILARITY
ADD_INT16_VS_DEQUANT_SUM_STRESS_MIN_COSINE_SIMILARITY = 0.999

# Fixed-point Add — Path B (single-op, float32): ``ref = a + b``; gates are strict:
# ``cosine > 0.9999`` and ``max_error_lsb_float < 1`` (float error < 1 * scale_out).
# See ``tests/fixed_point/kernels/test_add_ideal_float_reference.py``.
ADD_IDEAL_FLOAT_MIN_COSINE_EXCLUSIVE = 0.9999
ADD_IDEAL_FLOAT_MAX_FLOAT_LSB_EXCLUSIVE = 1.0
# Legacy aliases (inclusive assert_int16_vs_fp32_reference helpers).
ADD_INT16_VS_IDEAL_FLOAT_MAX_ERROR_LSB = INT16_VS_FP32_MAX_ERROR_LSB
ADD_INT16_VS_IDEAL_FLOAT_MIN_COSINE_SIMILARITY = INT16_VS_FP32_MIN_COSINE_SIMILARITY
ADD_INT16_VS_IDEAL_FLOAT_STRESS_MIN_COSINE_SIMILARITY = 0.999
ADD_INT16_VS_IDEAL_FLOAT_STRESS_MAX_ERROR_LSB = 2.0

# Back-compat aliases (Path A naming).
ADD_INT16_VS_FLOAT_MAX_ERROR_LSB = ADD_INT16_VS_DEQUANT_SUM_MAX_ERROR_LSB
ADD_INT16_VS_FLOAT_MIN_COSINE_SIMILARITY = ADD_INT16_VS_DEQUANT_SUM_MIN_COSINE_SIMILARITY
ADD_INT16_VS_FLOAT_STRESS_MIN_COSINE_SIMILARITY = ADD_INT16_VS_DEQUANT_SUM_STRESS_MIN_COSINE_SIMILARITY

# fp16_qdq vs fp32_qdq (no LSB — floating-point QDQ path)
FP16_VS_FP32_MIN_COSINE_SIMILARITY = 0.9999

# fixed_scale_qdq vs fp32_qdq (M,r approximation on Q/DQ boundary; Design v2 §10.1)
FIXED_SCALE_VS_FP32_MIN_COSINE_SIMILARITY = 0.9998

# Kernel / LUT vs float reference on the output encoding grid
KERNEL_VS_FLOAT_MAX_ERROR_LSB = 1.0

# Piecewise-linear nonlinear kernels (hardware-fixed segment count).
# NB: ``INT16_VS_FP32_MAX_ERROR_LSB`` applies to int16_fixed_eval vs fp32_qdq on the training grid.
# PWL approximates analytic ``fn(x)`` with only N segments; vs ideal float it needs separate, per-fn bounds.
PWL_HARDWARE_NUM_SEGMENTS = 16
PWL_EXPORT_VALIDATION_SAMPLES = 4096

# Shape-fidelity gate. PWL is well-behaved overall even when local LSB errors near the
# steep transition are large, so cosine is the primary quality signal.
PWL_VS_ANALYTIC_MIN_COSINE_SIMILARITY = 0.9999

# Per-fn quality limits vs the analytic activation, evaluated on integer-grid LSB units.
# Headroom over measured worst-case in [-8, 8] sigmoid/tanh-style ranges with default scales.
# Keys are lowered activation function names (``fn.__name__``).
PWL_VS_ANALYTIC_PER_FN_LIMITS: dict[str, dict[str, float]] = {
    "sigmoid": {"max_lsb": 320.0, "p99_lsb": 260.0, "rmse_lsb": 80.0},
    "tanh":    {"max_lsb": 2400.0, "p99_lsb": 1700.0, "rmse_lsb": 450.0},
    "gelu":    {"max_lsb": 320.0, "p99_lsb": 240.0, "rmse_lsb": 60.0},
    "silu":    {"max_lsb": 200.0, "p99_lsb": 160.0, "rmse_lsb": 45.0},
    "softplus": {"max_lsb": 320.0, "p99_lsb": 240.0, "rmse_lsb": 60.0},
    "mish":    {"max_lsb": 320.0, "p99_lsb": 240.0, "rmse_lsb": 60.0},
    "sin":     {"max_lsb": 512.0, "p99_lsb": 400.0, "rmse_lsb": 120.0},
    "cos":     {"max_lsb": 512.0, "p99_lsb": 400.0, "rmse_lsb": 120.0},
    "exp":     {"max_lsb": 2048.0, "p99_lsb": 1500.0, "rmse_lsb": 400.0},
    "log":     {"max_lsb": 2048.0, "p99_lsb": 1500.0, "rmse_lsb": 400.0},
    "abs":     {"max_lsb": 8.0, "p99_lsb": 4.0, "rmse_lsb": 2.0},
}

# Fallback when an unlisted activation is fitted. Loose enough to catch only broken fits.
PWL_VS_ANALYTIC_DEFAULT_LIMITS: dict[str, float] = {
    "max_lsb": 4096.0,
    "p99_lsb": 3000.0,
    "rmse_lsb": 1024.0,
}

# Kept for backwards compatibility with earlier test names; equals sigmoid's max_lsb limit.
PWL_VS_ANALYTIC_FLOAT_MAX_ERROR_LSB = PWL_VS_ANALYTIC_PER_FN_LIMITS["sigmoid"]["max_lsb"]

# LayerNorm — spec §4.5.4 full integer bit-parity pipeline (§4.5.2 in-line
# integer variance + RSqrt CLZ LUT + spec line 422-428 16-bit M/rshift
# affine). Two physical lsb ceilings stack:
#   1. RSqrt CLZ LUT PWL fit residual (~3-5 LSB at LUT output grid)
#      amplified by ``γ/std`` — same root cause as P7 ``custom.RSqrt``.
#   2. Spec line 414's **16-bit M + max_rshift=31** decoding of α_x =
#      S_γ · S_x · S_inv / S_y. Under i16 calibrated grids ``α_x ≈ 1e-8``,
#      requiring ``rshift ≈ 41`` for full 16-bit M mantissa — outside
#      the 31-rshift limit, so the multiplier collapses to ~5-bit and
#      the affine path lsb explodes proportionally. **This is a spec
#      design choice, not a kernel implementation bug** (the project-
#      level ``MULTIPLIER_QBITS = 16`` matches spec line 414 literally).
# Measured worst-case (16 trials × 6 configs × 2 grids = 192 trials,
# integer pipeline post FU-DSP-PARITY + FU-AFFINE-INTEGER closure):
#   noaffine path (α_x = S_x·S_inv/S_y ≈ 3e-4, 16-bit M fits cleanly):
#     i16: cos_min = 1.000000, lsb_max ≤ 9.74   → ceiling 12.0
#     i8:  cos_min ≥ 0.999630, lsb_max ≤ 2.27   → ceiling 4.0
#   affine path (α_x ≈ 1e-8 — spec 16-bit M physical ceiling):
#     i16: cos_min ≥ 0.999994, lsb_max ≤ 910    → ceiling 1100.0
#     i8:  cos_min ≥ 0.999229, lsb_max ≤ 5.33   → ceiling 8.0
# Cosine clears 0.9999 on every measured case because the residual is
# magnitude-bounded. Tightening past these ceilings requires either a
# wider multiplier ABI in hardware (out of project scope — would change
# spec line 414) or a redesigned α_x factoring (tracked as
# FU-LAYERNORM-ALPHA-X-CALIBRATION).
LAYERNORM_VS_FP32_PER_GRID_LIMITS: dict[str, dict[str, dict[str, float]]] = {
    "i16": {
        "noaffine": {"max_lsb": 12.0, "min_cosine": 0.9999},
        "affine":   {"max_lsb": 1100.0, "min_cosine": 0.9999},
    },
    "i8": {
        "noaffine": {"max_lsb": 4.0, "min_cosine": 0.9999},
        "affine":   {"max_lsb": 8.0, "min_cosine": 0.9999},
    },
}

# INT16 single-op (v2 Quantized* vs FP32_QDQ): per-case cosine floors.
# Deploy-style 8-bit outputs absorb PWL error; GELU chains need a much looser gate.
# Keys must match ``case`` names in ``collect_int16_kernel_section`` exactly.
INT16_SINGLE_OP_PER_CASE_MIN_COSINE: dict[str, float] = {
    "Linear → Tanh (PWL)": 0.9994,
    "Linear → GELU → Linear (PWL)": 0.97,
    "Linear → GELU → Linear (PWL) [stress]": 0.9993,
    "Linear → Softmax (PWL exp)": 0.999,
    "Linear → Softmax (PWL exp) [stress]": 0.9995,
}

# MobileNet V2 end-to-end cosine floors (INT16 vs FP32_QDQ logits).
MOBILENET_V2_INT16_COSINE_MIN = 0.99
MOBILENET_V2_FP16_COSINE_MIN = 0.999
MOBILENET_V2_PTQ_COSINE_MIN = 0.99

# Default AdaRound iteration counts for MobileNet e2e (report vs long regression).
MOBILENET_ADAROUND_ITERATIONS_DEFAULT = 80
MOBILENET_ADAROUND_ITERATIONS_LONG = 2000


def int16_single_op_min_cosine_similarity(case_name: str) -> float:
    """Return the cosine floor for an INT16 single-op report/test case."""
    return INT16_SINGLE_OP_PER_CASE_MIN_COSINE.get(
        case_name, INT16_VS_FP32_MIN_COSINE_SIMILARITY
    )

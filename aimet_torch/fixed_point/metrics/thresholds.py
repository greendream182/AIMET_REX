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

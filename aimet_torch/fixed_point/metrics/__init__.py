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
"""Observability helpers for fixed-point execution (see INTERFACE.md §8)."""

from aimet_torch.fixed_point.metrics.accuracy import (
    assert_fp16_vs_fp32_reference,
    assert_int16_vs_fp32_reference,
    assert_kernel_vs_float_reference,
    assert_max_error_lsb,
    assert_min_cosine_similarity,
    compute_pair_metrics,
    cosine_similarity,
    max_error_lsb_float,
    max_error_lsb_int,
    quantize_float_to_grid,
)
from aimet_torch.fixed_point.metrics.compare import (
    DEFAULT_COMPARE_MODES,
    compare_modes,
    compare_modes_against_reference,
    write_per_layer_csv,
)
from aimet_torch.fixed_point.metrics.chained import per_layer_chained_cosine
from aimet_torch.fixed_point.metrics.classification import (
    top1_accuracy,
    top1_drop,
    top_k_accuracy,
    top_k_prediction_agreement,
)
from aimet_torch.fixed_point.metrics.flags import (
    int16_eval_allow_debug_float,
    int16_eval_debug_float_allowed,
    set_int16_eval_debug_float_allowed,
)
from aimet_torch.fixed_point.metrics.isolated import per_layer_isolated_cosine
from aimet_torch.fixed_point.metrics.logits import (
    DEFAULT_VS_FP32_MODES,
    dequantize_logits,
    logits_cosine,
    mean_logits_cosine_on_loader,
    mean_logits_cosine_vs_fp32,
)
from aimet_torch.fixed_point.metrics.profiler import FixedPointProfiler
from aimet_torch.fixed_point.metrics.reporting import (
    per_layer_table_to_markdown,
    render_per_layer_table,
)
from aimet_torch.fixed_point.metrics.thresholds import (
    FIXED_SCALE_VS_FP32_MIN_COSINE_SIMILARITY,
    FP16_VS_FP32_MIN_COSINE_SIMILARITY,
    INT16_VS_FP32_MAX_ERROR_LSB,
    INT16_VS_FP32_MIN_COSINE_SIMILARITY,
    KERNEL_VS_FLOAT_MAX_ERROR_LSB,
)

__all__ = [
    "FixedPointProfiler",
    "FIXED_SCALE_VS_FP32_MIN_COSINE_SIMILARITY",
    "FP16_VS_FP32_MIN_COSINE_SIMILARITY",
    "INT16_VS_FP32_MAX_ERROR_LSB",
    "INT16_VS_FP32_MIN_COSINE_SIMILARITY",
    "KERNEL_VS_FLOAT_MAX_ERROR_LSB",
    "assert_fp16_vs_fp32_reference",
    "assert_int16_vs_fp32_reference",
    "assert_kernel_vs_float_reference",
    "assert_max_error_lsb",
    "assert_min_cosine_similarity",
    "DEFAULT_COMPARE_MODES",
    "compare_modes",
    "compare_modes_against_reference",
    "write_per_layer_csv",
    "compute_pair_metrics",
    "cosine_similarity",
    "int16_eval_allow_debug_float",
    "int16_eval_debug_float_allowed",
    "DEFAULT_VS_FP32_MODES",
    "dequantize_logits",
    "logits_cosine",
    "max_error_lsb_float",
    "max_error_lsb_int",
    "mean_logits_cosine_on_loader",
    "mean_logits_cosine_vs_fp32",
    "per_layer_chained_cosine",
    "per_layer_isolated_cosine",
    "per_layer_table_to_markdown",
    "quantize_float_to_grid",
    "render_per_layer_table",
    "set_int16_eval_debug_float_allowed",
    "top1_accuracy",
    "top1_drop",
    "top_k_accuracy",
    "top_k_prediction_agreement",
]

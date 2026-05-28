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

from aimet_torch.fixed_point.metrics.thresholds import (
    INT16_SINGLE_OP_PER_CASE_MIN_COSINE,
    INT16_VS_FP32_MIN_COSINE_SIMILARITY,
    int16_single_op_min_cosine_similarity,
)


def test_int16_single_op_per_case_thresholds_are_documented():
    assert INT16_SINGLE_OP_PER_CASE_MIN_COSINE["Linear → Tanh (PWL)"] < INT16_VS_FP32_MIN_COSINE_SIMILARITY
    assert INT16_SINGLE_OP_PER_CASE_MIN_COSINE["Linear → GELU → Linear (PWL)"] == 0.97


def test_int16_single_op_default_threshold_for_linear_ops():
    assert int16_single_op_min_cosine_similarity("Linear → ReLU → Linear") == INT16_VS_FP32_MIN_COSINE_SIMILARITY

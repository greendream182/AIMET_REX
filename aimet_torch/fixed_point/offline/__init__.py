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
"""Offline helpers for fixed-point parameter generation."""

from aimet_torch.fixed_point.offline.bias import quantize_bias_int, quantize_bias_int32
from aimet_torch.fixed_point.offline.clz_gen import (
    ClzLutGenerationError,
    clz_activation_name,
    clz_lut_to_json_dict,
    generate_clz_lut_for_export,
    resolve_abc_lut_root,
)
from aimet_torch.fixed_point.offline.lut_gen import (
    PwlLutAccuracyError,
    align_op_quant_grid_to_lut_quant_grid,
    bake_op_scale_adapter_into_pwl_lut,
    attach_pwl_sidecar_metadata,
    fold_periodic_input_to_principal_range,
    periodic_lut_fit_spec,
    principal_periodic_input_encoding,
    assert_pwl_metrics_within_limits,
    check_pwl_metrics_within_limits,
    encodings_share_quant_grid,
    generate_lut_int16,
    generate_pwl_lut,
    generate_pwl_lut_for_export,
    measure_pwl_lut_max_error_lsb,
    measure_pwl_lut_metrics,
    pwl_lut_from_json_dict,
    pwl_lut_to_json_dict,
    resolve_pwl_quality_limits,
)
from aimet_torch.fixed_point.offline.multiplier import (
    quantize_multiplier,
    record_multiplier_saturations,
)
from aimet_torch.fixed_point.offline.pipeline import (
    freeze_int16_fixed,
    freeze_int16_fixed_report_only,
    multiplier_relative_error,
)
from aimet_torch.fixed_point.offline.scale_fixed import (
    fixed_scale_float_scale,
    clear_fixed_scale_encoding_cache,
    convert_encodings_to_fixed_scale,
    fixed_scale_encoding_from_affine,
    fixed_scale_encoding_from_tensors,
    get_or_create_fixed_scale_encoding,
    quantize_scale_to_m_rshift,
)

__all__ = [
    "ClzLutGenerationError",
    "clz_activation_name",
    "attach_pwl_sidecar_metadata",
    "clz_lut_to_json_dict",
    "generate_clz_lut_for_export",
    "periodic_lut_fit_spec",
    "resolve_abc_lut_root",
    "PwlLutAccuracyError",
    "align_op_quant_grid_to_lut_quant_grid",
    "bake_op_scale_adapter_into_pwl_lut",
    "fold_periodic_input_to_principal_range",
    "principal_periodic_input_encoding",
    "encodings_share_quant_grid",
    "assert_pwl_metrics_within_limits",
    "check_pwl_metrics_within_limits",
    "generate_lut_int16",
    "generate_pwl_lut",
    "generate_pwl_lut_for_export",
    "measure_pwl_lut_max_error_lsb",
    "measure_pwl_lut_metrics",
    "pwl_lut_from_json_dict",
    "pwl_lut_to_json_dict",
    "clear_fixed_scale_encoding_cache",
    "convert_encodings_to_fixed_scale",
    "freeze_int16_fixed",
    "freeze_int16_fixed_report_only",
    "multiplier_relative_error",
    "fixed_scale_encoding_from_affine",
    "fixed_scale_float_scale",
    "fixed_scale_encoding_from_tensors",
    "get_or_create_fixed_scale_encoding",
    "quantize_bias_int",
    "quantize_bias_int32",
    "quantize_multiplier",
    "quantize_scale_to_m_rshift",
    "record_multiplier_saturations",
    "resolve_pwl_quality_limits",
]

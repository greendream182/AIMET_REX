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
"""Fixed-point and execution-mode utilities for AIMET RX."""

from aimet_torch.fixed_point.execution_mode import (
    ExecutionMode,
    get_quant_execution_mode,
    quant_execution_mode,
    set_quant_execution_mode,
)
from aimet_torch.fixed_point.encoding import FixedScaleEncoding, InputEncoding, OutputEncoding
from aimet_torch.fixed_point.encoding_export import (
    fixed_point_tensor_bundle,
    fixed_scale_encoding_from_dict,
    fixed_scale_encoding_to_dict,
    input_encoding_from_dict,
    input_encoding_to_dict,
    output_encoding_from_dict,
    output_encoding_to_dict,
)
from aimet_torch.fixed_point.fixed_scale_qdq import (
    dequantize_with_fixed_scale,
    quantize_dequantize_from_float_encoding,
    quantize_dequantize_with_fixed_scale,
    quantize_with_fixed_scale,
)
from aimet_torch.fixed_point.requantize import (
    INT16_QMAX,
    INT16_QMIN,
    INT32_QMAX,
    INT32_QMIN,
    MULTIPLIER_MAX,
    MULTIPLIER_QBITS,
    requantize_int,
    round_shift,
    saturate_int16,
    saturate_int32,
    saturate_sim_tensor,
    saturate_to_range,
    SIM_TENSOR_DTYPE,
)
from aimet_torch.fixed_point.registry import (
    FixedKernel,
    KernelNotFoundError,
    clear_fixed_kernel_registry,
    get_fixed_kernel,
    list_registered_kernels,
    register_fixed_kernel,
)
from aimet_torch.fixed_point.offline import (
    convert_encodings_to_fixed_scale,
    generate_lut_int16,
    generate_pwl_lut,
    pwl_lut_from_json_dict,
    pwl_lut_to_json_dict,
    quantize_bias_int,
    quantize_bias_int32,
    quantize_multiplier,
    quantize_scale_to_m_rshift,
    record_multiplier_saturations,
)
from aimet_torch.fixed_point.qat import FakeQuantInt16STE, fake_quantize_int16_qat
from aimet_torch.fixed_point.qat_train import (
    QatTrainScope,
    run_int16_qat_epochs,
    run_int16_qat_steps,
    select_qat_trainable_parameters,
)
from aimet_torch.fixed_point.rounding import RoundingMode
from aimet_torch.fixed_point.sim_utils import (
    ensure_output_quantizers_for_int16_eval,
    iter_missing_output_quantizers,
)
from aimet_torch.fixed_point.diagnose import diagnose_int16_readiness, is_int16_ready
from aimet_torch.fixed_point.quant_grid import (
    GRID_I8,
    GRID_I16,
    GRID_I32,
    GRID_U8,
    GRID_U16,
    GRID_U32,
    SIM_INT32_QUANT_GRIDS,
    STANDARD_QUANT_GRIDS,
    QuantGridSpec,
)
from aimet_torch.fixed_point.tensor import FixedPointSimTensor, Int16QuantizedTensor

# Phase B export (sidecar JSON); import subpackage explicitly to avoid v2 import at root.
# from aimet_torch.fixed_point.export import export_int16_sidecar_json

__all__ = [
    "ExecutionMode",
    "FakeQuantInt16STE",
    "FixedKernel",
    "FixedPointSimTensor",
    "FixedScaleEncoding",
    "GRID_I8",
    "GRID_I16",
    "GRID_I32",
    "GRID_U8",
    "GRID_U16",
    "GRID_U32",
    "QuantGridSpec",
    "SIM_INT32_QUANT_GRIDS",
    "STANDARD_QUANT_GRIDS",
    "convert_encodings_to_fixed_scale",
    "dequantize_with_fixed_scale",
    "fixed_point_tensor_bundle",
    "fixed_scale_encoding_from_dict",
    "fixed_scale_encoding_to_dict",
    "generate_lut_int16",
    "generate_pwl_lut",
    "input_encoding_from_dict",
    "input_encoding_to_dict",
    "pwl_lut_from_json_dict",
    "pwl_lut_to_json_dict",
    "InputEncoding",
    "Int16QuantizedTensor",
    "INT16_QMAX",
    "INT16_QMIN",
    "INT32_QMAX",
    "INT32_QMIN",
    "KernelNotFoundError",
    "MULTIPLIER_MAX",
    "MULTIPLIER_QBITS",
    "OutputEncoding",
    "RoundingMode",
    "clear_fixed_kernel_registry",
    "diagnose_int16_readiness",
    "ensure_output_quantizers_for_int16_eval",
    "get_fixed_kernel",
    "get_quant_execution_mode",
    "is_int16_ready",
    "iter_missing_output_quantizers",
    "list_registered_kernels",
    "quant_execution_mode",
    "quantize_bias_int",
    "quantize_bias_int32",
    "quantize_dequantize_from_float_encoding",
    "quantize_dequantize_with_fixed_scale",
    "quantize_scale_to_m_rshift",
    "quantize_with_fixed_scale",
    "fake_quantize_int16_qat",
    "quantize_multiplier",
    "QatTrainScope",
    "run_int16_qat_epochs",
    "run_int16_qat_steps",
    "select_qat_trainable_parameters",
    "output_encoding_from_dict",
    "output_encoding_to_dict",
    "record_multiplier_saturations",
    "register_fixed_kernel",
    "requantize_int",
    "round_shift",
    "saturate_int16",
    "saturate_int32",
    "saturate_sim_tensor",
    "saturate_to_range",
    "SIM_TENSOR_DTYPE",
    "set_quant_execution_mode",
]

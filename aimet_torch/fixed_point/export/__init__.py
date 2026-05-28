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
"""INT16 fixed-point sidecar export for deployment (Phase B)."""

from aimet_torch.fixed_point.export.sidecar_loader import (
    attach_int16_sidecar_to_model,
    build_runtime_extra_by_layer,
    detach_int16_sidecar_from_model,
    get_int16_sidecar_extra,
    layer_bundle_to_runtime_extra,
    maybe_attach_int16_sidecar_from_env,
    resolve_sidecar_layer_to_module_names,
)
from aimet_torch.fixed_point.export.sidecar import (
    INT16_SIDECAR_VERSION,
    attach_onnx_name_hints,
    build_int16_sidecar_document,
    build_onnx_name_hints_for_layers,
    compare_sidecar_with_model,
    default_int16_sidecar_path,
    export_int16_sidecar_json,
    layers_from_sidecar,
    load_int16_sidecar_json,
)
from aimet_torch.fixed_point.export.v2_collect import (
    collect_v2_int16_layer_record,
    collect_v2_int16_layers,
    derive_int16_clz_json,
    derive_int16_output_encoding,
    derive_int16_pwl_json,
    derive_int16_real_multiplier,
)

__all__ = [
    "attach_int16_sidecar_to_model",
    "build_runtime_extra_by_layer",
    "detach_int16_sidecar_from_model",
    "get_int16_sidecar_extra",
    "layer_bundle_to_runtime_extra",
    "maybe_attach_int16_sidecar_from_env",
    "resolve_sidecar_layer_to_module_names",
    "INT16_SIDECAR_VERSION",
    "attach_onnx_name_hints",
    "build_int16_sidecar_document",
    "build_onnx_name_hints_for_layers",
    "collect_v2_int16_layer_record",
    "collect_v2_int16_layers",
    "compare_sidecar_with_model",
    "default_int16_sidecar_path",
    "derive_int16_clz_json",
    "derive_int16_output_encoding",
    "derive_int16_pwl_json",
    "derive_int16_real_multiplier",
    "export_int16_sidecar_json",
    "layers_from_sidecar",
    "load_int16_sidecar_json",
]

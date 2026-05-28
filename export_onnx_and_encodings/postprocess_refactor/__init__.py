"""
postprocess 逻辑拆分后的模块集合。
"""

from .onnx_graph import postprocess_onnx_graph_inplace, fix_slice_end_constants
from .io_utils import load_encodings, save_encodings
from .encodings_ops import (
    merge_gru_activation_encodings,
    merge_reverse_cells_into_cells,
    rename_gru_internal_ops_keys,
    add_gru_output_from_internal_ops,
    add_gru_input_from_internal_ops,
    merge_bn_activation_encodings,
    merge_gru_bias_param_encodings,
    merge_bidirectional_gru_param_encodings,
    compact_per_channel_param_encodings,
    flatten_activation_io_index_dict,
    normalize_quant_fields_add_n,
    normalize_dtype_with_bitwidth,
    postprocess_encodings_inplace,
)
from .pipeline import postprocess_encodings_file, postprocess_all
from .renaming import rename_gru_initializers_and_change_gru_json_format
from .bn_params import add_bn_param_encodings_from_onnx

__all__ = [
    "postprocess_onnx_graph_inplace",
    "fix_slice_end_constants",
    "load_encodings",
    "save_encodings",
    "merge_gru_activation_encodings",
    "merge_reverse_cells_into_cells",
    "rename_gru_internal_ops_keys",
    "add_gru_output_from_internal_ops",
    "add_gru_input_from_internal_ops",
    "merge_bn_activation_encodings",
    "merge_gru_bias_param_encodings",
    "merge_bidirectional_gru_param_encodings",
    "compact_per_channel_param_encodings",
    "flatten_activation_io_index_dict",
    "normalize_quant_fields_add_n",
    "normalize_dtype_with_bitwidth",
    "postprocess_encodings_inplace",
    "postprocess_encodings_file",
    "rename_gru_initializers_and_change_gru_json_format",
    "postprocess_all",
    "add_bn_param_encodings_from_onnx",
]

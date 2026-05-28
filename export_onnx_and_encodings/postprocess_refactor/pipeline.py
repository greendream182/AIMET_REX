from __future__ import annotations

import logging
from typing import Tuple

from .bn_params import add_bn_param_encodings_from_onnx
from .encodings_ops import (
    add_gru_input_from_internal_ops,
    add_gru_output_from_internal_ops,
    compact_per_channel_param_encodings,
    flatten_activation_io_index_dict,
    merge_gru_activation_encodings,
    merge_reverse_cells_into_cells,
    merge_gru_bias_param_encodings,
    merge_bidirectional_gru_param_encodings,
    normalize_dtype_with_bitwidth,
    normalize_quant_fields_add_n,
    rename_gru_internal_ops_keys,
    postprocess_encodings_inplace,
)
from .io_utils import load_encodings, save_encodings
from .onnx_graph import fix_slice_end_constants, postprocess_onnx_graph_inplace
from .renaming import rename_gru_initializers_and_change_gru_json_format

logger = logging.getLogger(__name__)


def postprocess_encodings_file(
    encodings_path: str,
    output_encodings_path: str,
    onnx_path: str | None = None,
    verbose: bool = True,
) -> str:
    enc = load_encodings(encodings_path)
    enc, changed = postprocess_encodings_inplace(enc, verbose=verbose)
    if onnx_path:
        enc, bn_added = add_bn_param_encodings_from_onnx(enc, onnx_path=onnx_path, verbose=verbose)
        changed = bool(changed or bn_added)
    if changed or output_encodings_path != encodings_path:
        save_encodings(enc, output_encodings_path)
    return output_encodings_path


def postprocess_all(
    *,
    onnx_path: str,
    encodings_path: str,
    input_shape: Tuple[int, int, int],
    output_onnx_path: str | None = None,
    output_encodings_path: str | None = None,
    verbose: bool = True,
) -> Tuple[str, str]:
    if verbose:
        logger.info("=== Pipeline G1: postprocess_onnx_graph_inplace ===")
    postprocess_onnx_graph_inplace(onnx_path, input_shape)

    if verbose:
        logger.info("=== Pipeline G2: fix_slice_end_constants ===")
    fix_slice_end_constants(onnx_path, verbose=verbose)

    if verbose:
        logger.info("=== Pipeline E*: load_encodings ===")
    enc = load_encodings(encodings_path)
    changed_any = False

    if verbose:
        logger.info("=== Pipeline E1: merge_gru_activation_encodings ===")
    enc, c = merge_gru_activation_encodings(enc, verbose=verbose)
    changed_any |= c

    if verbose:
        logger.info("=== Pipeline E2: rename_gru_internal_ops_keys ===")
    enc, c = rename_gru_internal_ops_keys(enc, verbose=verbose)
    changed_any |= c

    if verbose:
        logger.info("=== Pipeline E3: add_gru_output_from_internal_ops ===")
    enc, c = add_gru_output_from_internal_ops(enc, verbose=verbose)
    changed_any |= c

    if verbose:
        logger.info("=== Pipeline E4: add_gru_input_from_internal_ops ===")
    enc, c = add_gru_input_from_internal_ops(enc, verbose=verbose)
    changed_any |= c

    if verbose:
        logger.info("=== Pipeline E4.5: merge_reverse_cells_into_cells ===")
    enc, c = merge_reverse_cells_into_cells(enc, verbose=verbose)
    changed_any |= c

    if verbose:
        logger.info("=== Pipeline E5: merge_gru_bias_param_encodings ===")
    enc, c = merge_gru_bias_param_encodings(enc, verbose=verbose)
    changed_any |= c

    if verbose:
        logger.info("=== Pipeline E6: compact_per_channel_param_encodings ===")
    enc, c = compact_per_channel_param_encodings(enc, verbose=verbose)
    changed_any |= c

    if verbose:
        logger.info("=== Pipeline E6.5: merge_bidirectional_gru_param_encodings ===")
    enc, c = merge_bidirectional_gru_param_encodings(enc, verbose=verbose)
    changed_any |= c

    if verbose:
        logger.info("=== Pipeline X1: add_bn_param_encodings_from_onnx ===")
    enc, c = add_bn_param_encodings_from_onnx(enc, onnx_path=onnx_path, verbose=verbose)
    changed_any |= c

    if verbose:
        logger.info("=== Pipeline E7: flatten_activation_io_index_dict ===")
    enc, c = flatten_activation_io_index_dict(enc, verbose=verbose)
    changed_any |= c

    if verbose:
        logger.info("=== Pipeline E8: normalize_quant_fields_add_n ===")
    enc, c = normalize_quant_fields_add_n(enc, verbose=verbose)
    changed_any |= c

    if verbose:
        logger.info("=== Pipeline E9: normalize_dtype_with_bitwidth ===")
    enc, c = normalize_dtype_with_bitwidth(enc, verbose=verbose)
    changed_any |= c

    enc_out = output_encodings_path or encodings_path
    if changed_any or enc_out != encodings_path:
        if verbose:
            logger.info("=== Pipeline E*: save_encodings -> %s ===", enc_out)
        save_encodings(enc, enc_out)
    else:
        enc_out = encodings_path

    if verbose:
        logger.info("=== Pipeline X2: rename_gru_initializers_and_change_gru_json_format ===")
    onnx_out = output_onnx_path or onnx_path
    return rename_gru_initializers_and_change_gru_json_format(
        onnx_path=onnx_path,
        encodings_path=enc_out,
        output_onnx_path=onnx_out,
        output_encodings_path=enc_out,
        verbose=verbose,
    )

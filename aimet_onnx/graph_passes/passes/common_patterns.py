# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-docstring

from aimet_common.connected_graph.operation import Op
from aimet_onnx.utils import ModelProto

from aimet_onnx.graph_passes.utils import (
    check_consecutive_ops,
    match_pow_2_pattern,
    is_constant_scalar,
)


def match_rms_norm_pattern(op: Op, model: ModelProto):
    """Common pattern for RMSNormalization which can be re-used"""
    # Match Mul(x, x) or Pow(x, 2)
    match = match_pow_2_pattern(op, model)
    if not match or len(op.output_ops) != 1:
        return []

    # Sqrt(E(Pow(x, 2)) + ε)
    match, denominator_ops = check_consecutive_ops(
        op.output_ops[0],
        ["ReduceMean", "Add", "Sqrt", "Div"],
        validate_last_op_consumers=False,
    )
    if not match:
        return []

    all_ops = [op] + denominator_ops
    div_op = all_ops[-1]

    # Div pattern 1: x * (1 / Sqrt(E(Pow(x, 2)) + ε))
    if is_constant_scalar(model, div_op.inputs[0], 1) and len(div_op.output_ops) == 1:
        mul_op = div_op.output_ops[0]
        # Mul input order can be anything.
        input_names = {input.name for input in mul_op.inputs}
        expected_inputs = {op.inputs[0].name, div_op.outputs[0].name}
        if mul_op.type != "Mul" or input_names != expected_inputs:
            return []
        all_ops.append(mul_op)

    # Div pattern 2: x / Sqrt(E(Pow(x, 2)) + ε)
    elif div_op.inputs[0] != op.inputs[0]:
        return []
    return all_ops

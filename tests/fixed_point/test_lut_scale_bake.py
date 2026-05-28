"""§3.0 mitigation 2: bake op scale adapter into PWL coefficients."""

import torch

from aimet_torch.fixed_point import InputEncoding, generate_pwl_lut
from aimet_torch.fixed_point.kernels.lut import evaluate_pwl_lut_int16
from aimet_torch.fixed_point.offline.lut_gen import (
    align_op_quant_grid_to_lut_quant_grid,
    bake_op_scale_adapter_into_pwl_lut,
)


def _enc(scale: float) -> InputEncoding:
    return InputEncoding(
        scale=torch.tensor(scale, dtype=torch.float32),
        zero_point=torch.tensor(0, dtype=torch.int32),
        qmin=-32768,
        qmax=32767,
    )


def test_bake_scale_adapter_matches_runtime_align():
    lut_enc = _enc(0.02)
    op_enc = _enc(0.04)
    out_enc = InputEncoding(
        scale=torch.tensor(0.01, dtype=torch.float32),
        zero_point=torch.tensor(0, dtype=torch.int32),
        qmin=0,
        qmax=32767,
    )
    pwl_lut = generate_pwl_lut(torch.sigmoid, lut_enc, out_enc, num_segments=8)
    baked = bake_op_scale_adapter_into_pwl_lut(pwl_lut, op_enc, lut_enc)
    assert baked["scale_adapter_baked"] is True
    assert baked["input_zero_point"].item() == 0

    q_op = torch.tensor([-500, 0, 500, 2000], dtype=torch.int16)
    q_via_align = align_op_quant_grid_to_lut_quant_grid(q_op, op_enc, lut_enc)
    y_align = evaluate_pwl_lut_int16(q_via_align, pwl_lut).to(torch.int32)
    y_baked = evaluate_pwl_lut_int16(q_op, baked).to(torch.int32)

    max_diff = (y_align - y_baked).abs().max().item()
    assert max_diff <= 2, f"bake vs align max diff {max_diff}"

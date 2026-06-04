# -*- mode: python -*-
"""Online PWL/CLZ LUT cache on quantized modules."""

from __future__ import annotations

import torch

from aimet_torch.fixed_point.export.sidecar_loader import (
    clear_int16_online_extra,
    get_int16_online_extra,
    merge_int16_online_extra,
)


def test_merge_and_read_online_extra_moves_to_device():
    mod = torch.nn.Linear(2, 2)
    lut = {"thresholds": torch.tensor([0, 1], dtype=torch.int32)}
    merge_int16_online_extra(mod, {"pwl_lut": lut, "phase_fold": "sin"})

    cpu_extra = get_int16_online_extra(mod)
    assert cpu_extra is not None
    assert cpu_extra["phase_fold"] == "sin"

    if torch.cuda.is_available():
        cuda_extra = get_int16_online_extra(mod, device=torch.device("cuda"))
        assert cuda_extra["pwl_lut"]["thresholds"].is_cuda

    clear_int16_online_extra(mod)
    assert get_int16_online_extra(mod) is None

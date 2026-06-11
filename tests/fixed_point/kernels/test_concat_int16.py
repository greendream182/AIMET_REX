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
"""INT16 Concat with mismatched input scales."""

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

import aimet_torch.fixed_point.kernels  # noqa: F401, E402
from aimet_torch._base.nn.modules import custom  # noqa: E402
from aimet_torch.fixed_point.encoding import OutputEncoding  # noqa: E402
from aimet_torch.fixed_point.export import build_int16_sidecar_document  # noqa: E402
from aimet_torch.fixed_point.registry import get_fixed_kernel  # noqa: E402
from aimet_torch.fixed_point.tensor import Int16QuantizedTensor  # noqa: E402
from aimet_torch.v2.nn.modules.custom import QuantizedAdd  # noqa: E402
from aimet_torch.v2.quantization.affine import Quantize  # noqa: E402


def test_concat_kernel_aligns_mismatched_input_scales():
    out_scale = torch.tensor(0.1, dtype=torch.float32)
    out_enc = OutputEncoding(
        scale=out_scale,
        zero_point=torch.tensor(0, dtype=torch.int32),
        qmin=-128,
        qmax=127,
        multiplier=torch.tensor(16384, dtype=torch.uint16),
        rshift=torch.tensor(14, dtype=torch.int8),
    )
    ta = Int16QuantizedTensor.from_float(
        torch.tensor([[10.0]], dtype=torch.float32),
        scale=torch.tensor(0.05, dtype=torch.float32),
        zero_point=torch.tensor(0, dtype=torch.int32),
    )
    tb = Int16QuantizedTensor.from_float(
        torch.tensor([[20.0]], dtype=torch.float32),
        scale=torch.tensor(0.2, dtype=torch.float32),
        zero_point=torch.tensor(0, dtype=torch.int32),
    )
    kernel = get_fixed_kernel(custom.Concat)
    out = kernel([ta, tb], {}, out_enc, {"axis": 1})
    assert isinstance(out, Int16QuantizedTensor)
    assert out.int_repr.shape[-1] == 2


def _init_add_dual(add: QuantizedAdd, in_ranges, out_range: float) -> None:
    add.input_quantizers[0] = Quantize((), 8, symmetric=True)
    add.input_quantizers[1] = Quantize((), 8, symmetric=True)
    add.output_quantizers[0] = Quantize((), 8, symmetric=True)
    for iq, (lo, hi) in zip(add.input_quantizers, in_ranges):
        iq.min = nn.Parameter(torch.tensor(lo))
        iq.max = nn.Parameter(torch.tensor(hi))
    add.output_quantizers[0].min = nn.Parameter(torch.tensor(-out_range))
    add.output_quantizers[0].max = nn.Parameter(torch.tensor(out_range))


def test_add_sidecar_has_input_requants():
    add = QuantizedAdd()
    _init_add_dual(add, [(-1.0, 1.0), (-2.0, 2.0)], out_range=3.0)
    model = nn.Sequential(add)
    doc = build_int16_sidecar_document(model)
    layer = doc["layers"]["0"]
    assert "input_requants" in layer
    assert len(layer["input_requants"]) == 2
    assert "multiplier" in layer["input_requants"][0]

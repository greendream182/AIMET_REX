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

import pytest

torch = pytest.importorskip("torch")

from aimet_torch.fixed_point.metrics.accuracy import (  # noqa: E402
    assert_int16_vs_fp32_reference,
    compute_pair_metrics,
    cosine_similarity,
    max_error_lsb_int,
)
from aimet_torch.fixed_point.tensor import Int16QuantizedTensor  # noqa: E402


def test_cosine_similarity_identical_vectors():
    x = torch.tensor([1.0, 2.0, 3.0])
    assert cosine_similarity(x, x) == pytest.approx(1.0)


def test_max_error_lsb_int_zero_when_same_grid():
    scale = torch.tensor(0.1, dtype=torch.float32)
    zp = torch.tensor(0, dtype=torch.int32)
    ref = torch.tensor([0.0, 0.1, 0.2])
    q = torch.tensor([0, 1, 2], dtype=torch.int16)
    err = max_error_lsb_int(ref, q, scale, zp, -32768, 32767)
    assert err == pytest.approx(0.0)


def test_assert_int16_vs_fp32_reference_passes_on_aligned_tensor():
    scale = torch.tensor(0.05, dtype=torch.float32)
    zp = torch.tensor(0, dtype=torch.int32)
    ref = torch.tensor([[0.0, 0.1, -0.05]])
    quantized = Int16QuantizedTensor.from_float(ref, scale=scale, zero_point=zp)
    assert_int16_vs_fp32_reference(quantized, ref)


def test_compute_pair_metrics_includes_lsb():
    scale = torch.tensor(0.1, dtype=torch.float32)
    zp = torch.tensor(0, dtype=torch.int32)
    ref = torch.tensor([0.0, 0.2])
    cur = Int16QuantizedTensor.from_float(ref, scale=scale, zero_point=zp)
    metrics = compute_pair_metrics(
        ref,
        cur.to_float(),
        scale=scale,
        zero_point=zp,
        candidate_int_repr=cur.int_repr,
    )
    assert metrics["max_error_lsb"] == pytest.approx(0.0)
    assert metrics["cosine_similarity"] == pytest.approx(1.0)

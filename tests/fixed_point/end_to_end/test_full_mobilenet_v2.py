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
"""Full (non-mock) MobileNet V2 end-to-end fixed-point smoke tests."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxscript")
pytest.importorskip("torchvision")

from .mobilenet_v2_helpers import (  # noqa: E402
    build_calibrated_sim,
    build_prepared_mobilenet_v2,
    int16_vs_fp32_cosine,
)

# Smaller than ImageNet 224 for CI; must remain divisible by 32.
FULL_INPUT_SIZE = 96
EVAL_BATCH = 2
MIN_COSINE = 0.99


@pytest.fixture(scope="module")
def full_mobilenet_sim():
    model, dummy = build_prepared_mobilenet_v2(
        input_size=FULL_INPUT_SIZE,
        variant="full",
    )
    return build_calibrated_sim(
        model,
        dummy,
        input_size=FULL_INPUT_SIZE,
        variant="full",
    )


@pytest.mark.slow
def test_full_mobilenet_v2_int16_vs_fp32_cosine(full_mobilenet_sim):
    """Full MobileNet V2 (96x96) INT16 dispatch meets the e2e cosine floor."""

    torch.manual_seed(17)
    x = torch.randn(EVAL_BATCH, 3, FULL_INPUT_SIZE, FULL_INPUT_SIZE)
    cos = int16_vs_fp32_cosine(full_mobilenet_sim.sim, x)
    assert cos >= MIN_COSINE, f"Full MobileNet V2 INT16 cosine={cos:.6f}"


@pytest.mark.slow
def test_full_mobilenet_v2_ptq_with_bias_correction(full_mobilenet_sim):
    """Bias correction on full MobileNet must not break the INT16 cosine floor."""

    model, dummy = build_prepared_mobilenet_v2(
        input_size=FULL_INPUT_SIZE,
        variant="full",
    )
    bc = build_calibrated_sim(
        model,
        dummy,
        input_size=FULL_INPUT_SIZE,
        variant="full",
        apply_bias_correction=True,
    )

    torch.manual_seed(19)
    x = torch.randn(EVAL_BATCH, 3, FULL_INPUT_SIZE, FULL_INPUT_SIZE)
    cos_ref = int16_vs_fp32_cosine(full_mobilenet_sim.sim, x)
    cos_bc = int16_vs_fp32_cosine(bc.sim, x)
    assert cos_ref >= MIN_COSINE
    assert cos_bc >= MIN_COSINE
    assert cos_bc + 1e-6 >= cos_ref - 0.01

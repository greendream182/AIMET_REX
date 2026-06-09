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
"""W5 SYS-FU-1.B contract regression: probe matrix frozen as pytest.

Background
----------

The W5.1 root-cause investigation (logged in ``doc/precision_validation.md``
SYS-LIMIT-1) ran an external Python probe to characterise the precision of
the INT16 ``REQUANTIZING`` kernels at varying ``(input_bw, weight_bw, N)``.
The probe established two facts that PR-1/2/3 turned into the combo-gate
contract:

* The asymmetric subset (``16+8`` and ``8+16``) is **safe** up to at
  least ``N = 4096``, with empirically measured SQNR ≥ 39 dB and
  cosine ≥ 0.99996 across the sweep.
* The full ``16+16`` combo on a MAC-reduction kernel saturates the
  INT32 ALU at ``N ≥ 1024`` and degrades catastrophically (cos ≈ 0.89,
  SQNR ≈ 4 dB at ``N = 4096``).

This file freezes the **dispatch-time contract** half of the probe as
a pytest regression: every combo the gate accepts must dispatch, every
combo the gate rejects must raise the precise gate ValueError, and the
acceptance/rejection set must not silently drift out of sync with the
SYS-FU-1.B subset definition.

Scope decision (省钱模式, B 类)
-------------------------------
Only the **dispatch contract** is frozen here, not the bit-level SQNR
sweep. Two reasons:

1. The W5.1 probe ran the kernel directly via ``LinearInt16Kernel``
   (no ``QuantizationSimModel`` adapter). Reproducing the same numbers
   through the v2 adapter pulls in ``quantize_multiplier`` /
   ``requantize_int`` rounding, fp32-QDQ reference drift, and the
   per-channel weight encoding path — the SQNR floor through that
   stack is a different (lower) number than the raw kernel probe.
   Spending a day on adapter-vs-kernel SQNR reconciliation does not
   pay back at this stage.
2. The bit-level SQNR is already documented per-combo in
   ``doc/precision_validation.md`` SYS-LIMIT-1, with explicit
   instructions to re-run the external probe whenever a kernel-side
   change affects the requantize path. That doc is the canonical
   source for SQNR numbers; this test guards the gate, not the kernel.

A failure of either test below means the SYS-FU-1.B combo subset
slipped: re-run the W5.1 probe before relaxing the assertions.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

import aimet_torch.fixed_point.kernels  # noqa: F401, E402
from aimet_torch.fixed_point import (  # noqa: E402
    ExecutionMode,
    Int16QuantizedTensor,
    quant_execution_mode,
)
from aimet_torch.v2.nn import QuantizedLinear  # noqa: E402
from aimet_torch.v2.quantization.affine import Quantize  # noqa: E402


def _build_linear_for_combo(
    n: int,
    input_bw: int,
    weight_bw: int,
    output_bw: int = 16,
    seed: int = 0,
) -> tuple[QuantizedLinear, torch.Tensor]:
    """Construct a deterministic ``QuantizedLinear(N -> N//4)`` configured
    for a (input_bw, weight_bw, output_bw) combo.

    He-style ``1 / sqrt(N)`` weight amplitude keeps the output's standard
    deviation roughly ``N``-independent, so the output quantizer range is
    fixed at ±2.0 across the entire sweep without ever clipping the
    encoding — failures therefore have to come from the dispatch path or
    the INT32 MAC accumulator, which is exactly the path SYS-LIMIT-1
    cares about.
    """

    out_features = max(4, n // 4)
    m = QuantizedLinear(n, out_features, bias=True)
    m.input_quantizers[0] = Quantize((), input_bw, symmetric=True)
    m.param_quantizers["weight"] = Quantize(
        (out_features, 1), weight_bw, symmetric=True
    )
    m.output_quantizers[0] = Quantize((), output_bw, symmetric=True)

    w_amp = 0.5 / math.sqrt(n)
    m.input_quantizers[0].min = nn.Parameter(torch.tensor(-1.5))
    m.input_quantizers[0].max = nn.Parameter(torch.tensor(1.5))
    m.param_quantizers["weight"].min = nn.Parameter(
        torch.full((out_features, 1), -w_amp)
    )
    m.param_quantizers["weight"].max = nn.Parameter(
        torch.full((out_features, 1), w_amp)
    )
    m.output_quantizers[0].min = nn.Parameter(torch.tensor(-2.0))
    m.output_quantizers[0].max = nn.Parameter(torch.tensor(2.0))

    g = torch.Generator()
    g.manual_seed(seed)
    with torch.no_grad():
        m.weight.copy_(
            torch.empty(out_features, n).uniform_(-w_amp, w_amp, generator=g)
        )
        m.bias.copy_(torch.zeros(out_features))

    g.manual_seed(seed + 1)
    x = torch.empty(8, n).uniform_(-1.5, 1.5, generator=g)
    return m, x


@pytest.mark.parametrize(
    "input_bw,weight_bw,n",
    [
        # Baseline: legacy 8+8 across the whole N-sweep
        pytest.param(8, 8, 64, id="W8A8-N64"),
        pytest.param(8, 8, 1024, id="W8A8-N1024"),
        pytest.param(8, 8, 4096, id="W8A8-N4096"),
        # SYS-FU-1.B asymmetric subset — both halves of the
        # (input=16, weight=8) and (input=8, weight=16) tile, at the
        # three reduction depths the W5.1 probe documented as safe.
        pytest.param(16, 8, 64, id="W16A8-N64"),
        pytest.param(16, 8, 1024, id="W16A8-N1024"),
        pytest.param(16, 8, 4096, id="W16A8-N4096"),
        pytest.param(8, 16, 64, id="W8A16-N64"),
        pytest.param(8, 16, 1024, id="W8A16-N1024"),
        pytest.param(8, 16, 4096, id="W8A16-N4096"),
    ],
)
def test_w5_linear_combo_dispatch_succeeds(
    input_bw: int, weight_bw: int, n: int
):
    """Every ``(input_bw, weight_bw)`` combo accepted by the SYS-FU-1.B
    gate must dispatch successfully through the v2 adapter at the three
    reduction depths the W5.1 probe documented as safe.

    A failure of this test means one of the following:

    1. ``REQUANTIZING_COMBO_BITWIDTH_BUDGET`` was tightened below 24
       (now rejecting a combo we used to support) or
       ``_REQUANTIZING_COMBO_VALIDATED_BITWIDTHS`` lost an entry —
       both regressions of the gate.
    2. The kernel started rejecting an input shape it used to handle
       (e.g. ``saturate_mac_accumulator`` developed an int24 path that
       refuses the ``W16+W8`` MAC).
    3. The output became NaN/Inf — the adapter dispatched but the
       kernel produced garbage. SYS-LIMIT-1 is supposed to be ALU
       saturation, not value-explosion.

    All three deserve an explicit follow-up commit; do not turn this
    smoke check into ``xfail`` without re-running the W5.1 probe.

    Note on what this does NOT assert: the bit-level SQNR floor lives
    in the external W5.1 probe (``doc/precision_validation.md``
    SYS-LIMIT-1) because the v2 adapter's
    ``quantize_multiplier`` + ``requantize_int`` stack drifts SQNR by
    a kernel-version-dependent amount that is not the contract this
    file is meant to lock down.
    """

    m, x = _build_linear_for_combo(
        n, input_bw=input_bw, weight_bw=weight_bw, output_bw=16
    )
    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y_int = m(x)

    assert isinstance(y_int, Int16QuantizedTensor)
    y_fp = y_int.to_float()
    assert y_fp.shape == (8, max(4, n // 4))
    assert torch.isfinite(y_fp).all(), (
        f"INT16 ({input_bw}+{weight_bw}) Linear at N={n} produced "
        f"non-finite output; the dispatch path must hand back finite "
        f"values even for combos at the edge of the budget."
    )


@pytest.mark.parametrize(
    "n",
    [
        # N=64 already crosses the int32-saturation onset for full 16+16
        # (cos drops below the elementwise floor). N=1024 / N=4096 are
        # where the W5.1 probe measured SQNR ≈ 9 / 4 dB and motivated
        # the gate in the first place.
        pytest.param(64, id="N64"),
        pytest.param(1024, id="N1024"),
        pytest.param(4096, id="N4096"),
    ],
)
def test_w5_full_16bit_reduction_remains_blocked(n: int):
    """Negative regression: full 16+16 on a Linear reduction MUST be
    blocked by the gate at every reduction depth, including ``N=64``
    where the kernel "almost works" (W5.1 probe: 31 dB SQNR).

    Pairs with ``test_w5_linear_combo_dispatch_succeeds`` — together they
    pin the W5 SYS-FU-1.B contract: gate accepts everything the kernel
    can honour, gate rejects everything it cannot.
    """

    m, x = _build_linear_for_combo(
        n=n, input_bw=16, weight_bw=16, output_bw=16
    )
    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        with pytest.raises(ValueError, match="REQUANTIZING-with-MAC-reduction"):
            m(x)

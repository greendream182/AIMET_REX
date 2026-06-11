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
"""``cLN2D`` (spec § 4.5.7) and ``SimCln2d`` (spec § 4.5.8) module tests.

These two normalization operators are listed in
``doc/04_算子详细规格/04_05_归一化类算子.md`` but were not previously
present in the project codebase (only the cfLN2D variant lived in
``_base/nn/modules/custom.py``). This file pins:

  * ``cLN2D``: forward shape ``[B, C, T, F]`` → ``[B, C, T, F]``;
    numeric correctness vs. the spec reference code (rearrange + var
    over flattened ``(C·F)`` axis); explicit op-module wiring (``Add /
    Sqrt / Divide``) so AIMET's v2 ``QuantizationMixin`` can intercept
    each step for INT16 dispatch.

  * ``SimCln2d``: forward + inverse round-trip; ``self.std`` cache
    populated correctly; explicit op-module wiring (``Abs / Add /
    Divide / Multiply``).

Both are ``sub-op composition`` style: there is no dedicated INT16
kernel for them — they reuse the already-implemented sub-op kernels
(``Add`` / ``Sqrt`` / ``Divide`` / ``Multiply`` / ``Abs``). The
DSP-single-instruction parity work is tracked as
``FU-NORM-SUBOP-VS-DSP-PARITY`` (see ``precision_validation.md``
``§ D2-4.5.7`` / ``§ D2-4.5.8``).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from aimet_torch._base.nn.modules.custom import (  # noqa: E402
    Abs,
    Add,
    Divide,
    Multiply,
    Sqrt,
    SimCln2d,
    cLN2D,
)

_RTOL = 1e-5
_ATOL = 1e-5

_SHAPE_CASES = (
    pytest.param((2, 3, 4, 5), id="B2_C3_T4_F5"),
    pytest.param((1, 8, 1, 4), id="B1_C8_T1_F4"),
    pytest.param((4, 4, 8, 4), id="B4_C4_T8_F4"),
)


@pytest.mark.parametrize("shape", _SHAPE_CASES)
def test_cln2d_forward_matches_spec_reference(shape):
    """cLN2D output equals the spec § 4.5.7 reference (rearrange-style).

    Spec reference (line 793-803 in normalization spec):

        o = rearrange(y, 'b c t f -> b (c f) t').contiguous()
        std = torch.var(o, dim=1, keepdim=True, unbiased=False).unsqueeze(-1)
        std = torch.pow(std + EPS, 0.5)
        o = y / std

    Our class avoids the einops dep by computing
    ``torch.var(y, dim=(1, 3), keepdim=True, unbiased=False)`` directly
    (mathematically equivalent — both flatten C and F into the variance
    reduce). This test pins the equivalence so any future refactor that
    touches the reduce dims breaks loudly.
    """
    eps = 1e-5
    torch.manual_seed(11)
    y = torch.randn(shape, dtype=torch.float32)

    module = cLN2D(eps=eps)
    actual = module(y)

    # Spec-reference (no einops).
    var_ref = torch.var(y, dim=(1, 3), keepdim=True, unbiased=False)
    std_ref = torch.sqrt(var_ref + eps)
    expected = y / std_ref

    assert actual.shape == y.shape, (
        f"cLN2D output shape {actual.shape} != input {y.shape}"
    )
    assert torch.allclose(actual, expected, rtol=_RTOL, atol=_ATOL), (
        f"cLN2D output diverges from spec reference (max abs "
        f"{(actual - expected).abs().max().item():.3e})."
    )


def test_cln2d_uses_explicit_op_modules():
    """cLN2D must expose ``Add`` / ``Sqrt`` / ``Divide`` as submodules.

    AIMET's v2 INT16 dispatch relies on the qmodule converter walking
    ``named_children`` to find quantizable op modules. If a refactor
    accidentally inlines ``var + eps`` as a free ``+`` (no ``Add``
    submodule), the int16 path silently drops to FP32_QDQ. This guard
    fails fast in that case.
    """
    module = cLN2D(eps=1e-5)
    children = dict(module.named_children())
    assert isinstance(children["add"], Add), "cLN2D.add must be an Add op-module"
    assert isinstance(children["sqrt"], Sqrt), "cLN2D.sqrt must be a Sqrt op-module"
    assert isinstance(children["divide"], Divide), "cLN2D.divide must be a Divide op-module"


@pytest.mark.parametrize("shape", _SHAPE_CASES)
def test_simcln2d_forward_matches_spec_reference(shape):
    """SimCln2d forward equals the spec § 4.5.8 reference.

    Spec reference (line 931-936):

        self.std = torch.mean(torch.abs(x), dim=(1, 3), keepdim=True) + self.eps
        x = x / self.std
    """
    eps = 0.009765625
    torch.manual_seed(13)
    x = torch.randn(shape, dtype=torch.float32)

    module = SimCln2d(eps=eps)
    actual = module(x)

    abs_x = x.abs()
    mean_ref = abs_x.mean(dim=(1, 3), keepdim=True)
    std_ref = mean_ref + eps
    expected = x / std_ref

    assert actual.shape == x.shape
    assert torch.allclose(actual, expected, rtol=_RTOL, atol=_ATOL)
    # ``self.std`` cache must be populated and have the broadcast shape
    # ``[B, 1, T, 1]`` ready for inverse — spec line 933 docstring.
    assert module.std is not None, "SimCln2d.std cache must be populated post-forward"
    assert module.std.shape == std_ref.shape, (
        f"SimCln2d.std cache shape {module.std.shape} != expected {std_ref.shape}"
    )


@pytest.mark.parametrize("shape", _SHAPE_CASES)
def test_simcln2d_inverse_round_trip(shape):
    """``inverse(forward(x))`` recovers x (modulo float-precision residual).

    Spec line 938-948 — ``inverse(y) = y · self.std`` exactly undoes the
    division. Round-trip residual is bounded by float32 multiply-divide
    associativity (~1e-6 on N(0, 1) inputs). Inverse without a prior
    forward must raise — ``self.std`` is undefined in that case.
    """
    torch.manual_seed(17)
    x = torch.randn(shape, dtype=torch.float32)

    module = SimCln2d()
    y = module(x)
    recovered = module.inverse(y)
    assert torch.allclose(recovered, x, rtol=1e-4, atol=1e-5), (
        f"SimCln2d round-trip diverges (max abs "
        f"{(recovered - x).abs().max().item():.3e})."
    )

    # Fresh module — calling inverse before forward must raise.
    fresh = SimCln2d()
    with pytest.raises(RuntimeError, match="forward must be called once"):
        fresh.inverse(x)


def test_simcln2d_uses_explicit_op_modules():
    """SimCln2d must expose ``Abs / Add / Divide / Multiply`` as submodules.

    Same rationale as ``test_cln2d_uses_explicit_op_modules`` — guards
    the INT16 sub-op dispatch path.
    """
    module = SimCln2d()
    children = dict(module.named_children())
    assert isinstance(children["abs"], Abs)
    assert isinstance(children["add"], Add)
    assert isinstance(children["divide"], Divide)
    assert isinstance(children["multiply"], Multiply)

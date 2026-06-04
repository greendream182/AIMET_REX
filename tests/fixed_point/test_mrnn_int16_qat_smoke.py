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
"""Full MRNN graph smoke: INT16_FIXED_QAT_SIM forward + backward (R1 e2e)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxscript")

_EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

import aimet_torch.fixed_point.kernels  # noqa: F401,E402
from aimet_torch.fixed_point import (  # noqa: E402
    ExecutionMode,
    diagnose_int16_readiness,
    ensure_output_quantizers_for_int16_eval,
    quant_execution_mode,
    run_int16_qat_steps,
)


def _has_quant_gru() -> bool:
    try:
        import quant_gru  # noqa: F401

        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not (_has_quant_gru() and torch.cuda.is_available()),
    reason="Full MRNN INT16 QAT smoke requires quant_gru + CUDA",
)


def _build_mrnn_sim(dummy: torch.Tensor):
    from aimet_torch import model_preparer
    from aimet_torch.v2 import quantsim
    from aimet_torch.utils_rx import apply_mixed_precision_bitwidth

    from quick_start import (  # noqa: E402
        BITWIDTH_CONFIG_FILE,
        CONFIG_FILE,
        DEFAULT_BW,
        MRNN,
        NUM_CLASSES,
        PERCENTILE_VALUE,
        QUANT_SCHEME,
    )

    model = MRNN(output_dim=NUM_CLASSES).to(dummy.device).eval()
    prepared = model_preparer.prepare_model(model)
    sim = quantsim.QuantizationSimModel(
        prepared,
        dummy_input=dummy,
        quant_scheme=QUANT_SCHEME,
        config_file=str(CONFIG_FILE),
        default_output_bw=DEFAULT_BW,
        default_param_bw=DEFAULT_BW,
    )
    sim.set_percentile_value(PERCENTILE_VALUE)
    apply_mixed_precision_bitwidth(
        sim.model, config_file=str(BITWIDTH_CONFIG_FILE), verbose=False,
    )
    ensure_output_quantizers_for_int16_eval(sim)
    sim.model.to(dummy.device)

    def _calib(m):
        with torch.no_grad():
            for _ in range(2):
                m(torch.randn_like(dummy))

    import aimet_torch.v2 as aimet

    with torch.no_grad(), aimet.nn.compute_encodings(sim.model):
        _calib(sim.model)

    return sim


@pytest.fixture(scope="module")
def mrnn_qat_bundle():
    device = torch.device("cuda")
    dummy = torch.randn(2, 16000, 1, device=device)
    sim = _build_mrnn_sim(dummy)
    return {"sim": sim, "dummy": dummy, "device": device}


def test_mrnn_diagnose_ready_before_qat(mrnn_qat_bundle):
    report = diagnose_int16_readiness(mrnn_qat_bundle["sim"])
    assert not any(report.values())


def test_mrnn_int16_qat_sim_forward_backward_smoke(mrnn_qat_bundle):
    """Full graph: train mode + INT16_FIXED_QAT_SIM + loss.backward (QuantGRU uses backward_quant)."""
    from aimet_torch.v2.nn.modules.custom import QuantizedQuantGRU

    sim = mrnn_qat_bundle["sim"]
    x = mrnn_qat_bundle["dummy"].detach().clone().requires_grad_(False)
    sim.model.train()

    quant_grus = [m for m in sim.model.modules() if isinstance(m, QuantizedQuantGRU)]
    assert len(quant_grus) == 4
    for gru in quant_grus:
        assert gru.is_calibrated()

    with quant_execution_mode(ExecutionMode.INT16_FIXED_QAT_SIM):
        y = sim.model(x)

    if hasattr(y, "to_float"):
        loss = y.to_float().sum()
    else:
        loss = y.sum()
    loss.backward()

    grus_with_grad = [
        gru
        for gru in quant_grus
        if gru.weight_ih_l0.grad is not None or gru.weight_hh_l0.grad is not None
    ]
    assert len(grus_with_grad) >= 1, "expected at least one QuantGRU to receive backward_quant grads"
    for gru in grus_with_grad:
        for param in (gru.weight_ih_l0, gru.weight_hh_l0):
            if param.grad is not None:
                assert torch.isfinite(param.grad).all()

    fc = sim.model.fc0
    if hasattr(fc, "weight") and getattr(fc.weight, "requires_grad", False):
        assert fc.weight.grad is not None
        assert torch.isfinite(fc.weight.grad).all()


def test_mrnn_int16_qat_sim_gru_boundary_returns_fp32_not_int_carrier(mrnn_qat_bundle):
    """INT16_FIXED_QAT_SIM: QuantGRU hook output is fp32 STE, not eval Int16 carrier.

    Eval-mode GRU carrier dtype is covered by ``test_quantgru_blackbox``; this smoke
    checks the same QAT contract on the full MRNN graph. Do not run full-graph
    ``INT16_FIXED_EVAL`` here — the minimal test sim lacks ``force_native_trans_float``
    / post-calib fixed_scale, and the STFT frontend is not INT16-dispatchable.
    """
    from aimet_torch.fixed_point.qat.carrier import clear_int16_carriers
    from aimet_torch.fixed_point.tensor import Int16QuantizedTensor
    from aimet_torch.v2.nn.modules.custom import QuantizedQuantGRU

    sim = mrnn_qat_bundle["sim"]
    x = mrnn_qat_bundle["dummy"]
    gru = next(m for m in sim.model.modules() if isinstance(m, QuantizedQuantGRU))

    clear_int16_carriers()
    sim.model.train()
    captured: dict[str, object] = {}

    def _hook(_mod, _inp, out):
        captured["out"] = out[0] if isinstance(out, tuple) else out

    handle = gru.register_forward_hook(_hook)
    try:
        with quant_execution_mode(ExecutionMode.INT16_FIXED_QAT_SIM):
            sim.model(x)
    finally:
        handle.remove()

    assert "out" in captured
    out = captured["out"]
    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.float32
    assert not isinstance(out, Int16QuantizedTensor)


def test_mrnn_int16_qat_multistep_train_smoke(mrnn_qat_bundle):
    """Several QAT_SIM steps: CE loss, optimizer step, head weights move."""
    sim = mrnn_qat_bundle["sim"]
    device = mrnn_qat_bundle["device"]
    first_weight = sim.model.fc0.weight.detach().clone()
    dummy = mrnn_qat_bundle["dummy"]
    labels = torch.zeros(dummy.shape[0], dtype=torch.long, device=device)

    losses = run_int16_qat_steps(
        sim.model,
        [(dummy, labels)],
        device,
        steps=3,
        lr=1e-3,
        scope="head",
        grad_clip_norm=1.0,
    )

    assert all(torch.isfinite(torch.tensor(losses)))
    assert sim.model.fc0.weight.detach().sub(first_weight).abs().max() > 0


def test_mrnn_int16_qat_multistep_weights_scope_smoke():
    """Full weight QAT (no quantizers); fresh sim — module fixture may be mutated by prior tests."""
    device = torch.device("cuda")
    dummy = torch.randn(2, 16000, 1, device=device)
    sim = _build_mrnn_sim(dummy)
    labels = torch.zeros(dummy.shape[0], dtype=torch.long, device=device)

    losses = run_int16_qat_steps(
        sim.model,
        [(dummy, labels)],
        device,
        steps=3,
        lr=1e-5,
        scope="weights",
        grad_clip_norm=1.0,
    )
    assert len(losses) == 3
    assert all(torch.isfinite(torch.tensor(losses)))

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
"""PTQ algorithm and QAT validation on Mock MobileNet V2."""

import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxscript")
pytest.importorskip("torchvision")

from .mobilenet_v2_helpers import (  # noqa: E402
    CALIB_BATCH,
    INPUT_SIZE,
    MOBILENET_ADAROUND_ITERATIONS_LONG,
    build_calibrated_sim,
    build_prepared_mobilenet_v2,
    int16_vs_fp32_cosine,
    make_adaround_loader,
    teacher_logits,
    train_int16_qat,
    train_int16_qat_on_loader,
)

EVAL_BATCH = 2
MIN_PTQ_COSINE = 0.99
# QAT should not regress vs the same sim's PTQ-only INT16 cosine on held-out data.
QAT_COSINE_MARGIN = 0.0


@pytest.fixture(scope="module")
def prepared_mobilenet():
    return build_prepared_mobilenet_v2()


@pytest.fixture(scope="module")
def baseline_ptq_sim(prepared_mobilenet):
    model, dummy = prepared_mobilenet
    return build_calibrated_sim(model, dummy)


def test_ptq_cle_maintains_int16_accuracy(prepared_mobilenet, baseline_ptq_sim):
    """Cross-layer equalization (CLE) should not degrade INT16 vs FP32_QDQ."""

    model, dummy = prepared_mobilenet
    cle_sim = build_calibrated_sim(model, dummy, apply_cle=True)

    torch.manual_seed(17)
    x = torch.randn(EVAL_BATCH, 3, INPUT_SIZE, INPUT_SIZE)
    cos_baseline = int16_vs_fp32_cosine(baseline_ptq_sim.sim, x)
    cos_cle = int16_vs_fp32_cosine(cle_sim.sim, x)

    assert cos_baseline >= MIN_PTQ_COSINE
    assert cos_cle >= MIN_PTQ_COSINE
    assert cos_cle + 1e-6 >= cos_baseline - 0.005, (
        f"CLE regressed INT16 cosine: baseline={cos_baseline:.6f} cle={cos_cle:.6f}"
    )


def test_ptq_adaround_improves_or_maintains_int16_accuracy(prepared_mobilenet, baseline_ptq_sim):
    """AdaRound on 8-bit weights should meet the PTQ floor and not trail vanilla PTQ."""

    model, dummy = prepared_mobilenet
    loader = make_adaround_loader(INPUT_SIZE, num_batches=4)

    with tempfile.TemporaryDirectory() as tmpdir:
        adaround_sim = build_calibrated_sim(
            model,
            dummy,
            adaround_loader=loader,
            adaround_num_batches=2,
            adaround_iterations=60,
            adaround_export_dir=Path(tmpdir),
        )

    torch.manual_seed(23)
    x = torch.randn(EVAL_BATCH, 3, INPUT_SIZE, INPUT_SIZE)
    cos_baseline = int16_vs_fp32_cosine(baseline_ptq_sim.sim, x)
    cos_adaround = int16_vs_fp32_cosine(adaround_sim.sim, x)

    assert cos_baseline >= MIN_PTQ_COSINE
    assert cos_adaround >= MIN_PTQ_COSINE
    assert cos_adaround + 1e-6 >= cos_baseline - QAT_COSINE_MARGIN, (
        f"AdaRound below vanilla PTQ: baseline={cos_baseline:.6f} adaround={cos_adaround:.6f}"
    )


def test_int16_qat_improves_over_ptq_only(prepared_mobilenet, baseline_ptq_sim):
    """~20-epoch INT16 QAT should match or beat PTQ-only INT16 vs FP32_QDQ on held-out data."""

    model, dummy = prepared_mobilenet
    teacher = model

    torch.manual_seed(31)
    x_holdout = torch.randn(EVAL_BATCH, 3, INPUT_SIZE, INPUT_SIZE)
    cos_ptq = int16_vs_fp32_cosine(baseline_ptq_sim.sim, x_holdout)

    qat_sim = build_calibrated_sim(model, dummy)
    losses = train_int16_qat(
        qat_sim.sim,
        teacher=teacher,
        input_size=INPUT_SIZE,
        epochs=20,
        lr=1e-3,
        batches_per_epoch=4,
        seed=99,
    )

    assert len(losses) == 20
    assert all(torch.isfinite(torch.tensor(v)) for v in losses), (
        f"Non-finite QAT losses: {losses}"
    )

    cos_qat = int16_vs_fp32_cosine(qat_sim.sim, x_holdout)
    assert cos_ptq >= MIN_PTQ_COSINE
    assert cos_qat + 1e-6 >= cos_ptq - QAT_COSINE_MARGIN, (
        f"QAT INT16 cosine {cos_qat:.6f} below PTQ-only {cos_ptq:.6f}"
    )


def test_int16_qat_training_step_produces_gradients(baseline_ptq_sim):
    """Smoke: one QAT step updates trainable parameters (surrogate path alive)."""

    sim = baseline_ptq_sim.sim
    sim.model.train()
    for module in sim.model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()

    from aimet_torch.fixed_point import ExecutionMode, quant_execution_mode

    x = torch.randn(CALIB_BATCH, 3, INPUT_SIZE, INPUT_SIZE, requires_grad=False)
    target = torch.randn(CALIB_BATCH, 10)

    params = [p for p in sim.model.parameters() if p.requires_grad]
    before = [p.detach().clone() for p in params[:3]]

    opt = torch.optim.SGD(params, lr=0.05)
    opt.zero_grad(set_to_none=True)
    with quant_execution_mode(ExecutionMode.INT16_FIXED_QAT_SIM):
        pred = sim.model(x)
    loss = torch.nn.functional.mse_loss(pred, target)
    loss.backward()
    opt.step()

    changed = sum(
        1 for p, b in zip(params[:3], before) if not torch.allclose(p, b, atol=0, rtol=0)
    )
    assert changed >= 1, "Expected at least one parameter to change after QAT step"


def test_int16_qat_on_loader_smoke(prepared_mobilenet):
    """Real-loader QAT helper should consume image batches and report finite loss."""

    model, dummy = prepared_mobilenet
    qat_sim = build_calibrated_sim(model, dummy)
    torch.manual_seed(37)
    images = torch.randn(2 * CALIB_BATCH, 3, INPUT_SIZE, INPUT_SIZE)
    labels = torch.zeros(2 * CALIB_BATCH, dtype=torch.long)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(images, labels),
        batch_size=CALIB_BATCH,
    )

    losses = train_int16_qat_on_loader(
        qat_sim.sim,
        teacher=model,
        train_loader=loader,
        epochs=1,
        lr=1e-4,
        max_batches=2,
    )

    assert len(losses) == 1
    assert torch.isfinite(torch.tensor(losses[0]))


def test_ptq_bias_correction_maintains_int16_accuracy():
    """Empirical bias correction should meet the PTQ floor and not trail vanilla by much."""

    model_v, dummy_v = build_prepared_mobilenet_v2()
    vanilla = build_calibrated_sim(model_v, dummy_v)
    model_bc, dummy_bc = build_prepared_mobilenet_v2()
    bc_sim = build_calibrated_sim(
        model_bc,
        dummy_bc,
        apply_bias_correction=True,
        bias_correction_samples=16,
    )

    torch.manual_seed(41)
    x = torch.randn(EVAL_BATCH, 3, INPUT_SIZE, INPUT_SIZE)
    cos_baseline = int16_vs_fp32_cosine(vanilla.sim, x)
    cos_bc = int16_vs_fp32_cosine(bc_sim.sim, x)

    assert cos_baseline >= MIN_PTQ_COSINE
    assert cos_bc >= MIN_PTQ_COSINE
    assert cos_bc + 1e-6 >= cos_baseline - 0.01, (
        f"Bias correction regressed INT16 cosine: baseline={cos_baseline:.6f} bc={cos_bc:.6f}"
    )


def test_v2_autoquant_or_combined_fallback_int16_smoke():
    """AutoQuant or combined PTQ fallback must yield INT16 cosine above the PTQ floor."""

    import tempfile
    from aimet_torch.fixed_point.e2e.autoquant import make_autoquant_loader, try_run_v2_autoquant_ptq
    from aimet_torch.fixed_point.metrics import cosine_similarity

    model, dummy = build_prepared_mobilenet_v2()
    ref_model = model
    holdout = torch.randn(EVAL_BATCH, 3, INPUT_SIZE, INPUT_SIZE)

    def eval_cb(quant_model, *_args, **_kwargs):
        quant_model.eval()
        device = next(quant_model.parameters()).device
        with torch.no_grad():
            ref = ref_model(holdout.to(device))
            out = quant_model(holdout.to(device))
            if hasattr(out, "dequantize"):
                out = out.dequantize()
            return cosine_similarity(ref, out.float())

    loader = make_autoquant_loader(INPUT_SIZE, num_samples=16, batch_size=2)
    with tempfile.TemporaryDirectory() as tmpdir:
        result = try_run_v2_autoquant_ptq(
            model,
            dummy,
            data_loader=loader,
            eval_callback=eval_cb,
            results_dir=Path(tmpdir),
            allowed_accuracy_drop=0.05,
            model_prepare_required=False,
            adaround_iterations=60,
            use_combined_fallback=True,
        )

    assert result is not None, "Expected AutoQuant or combined PTQ fallback to succeed"

    bundle = build_calibrated_sim(result.model, dummy)
    cos = int16_vs_fp32_cosine(bundle.sim, holdout)
    assert cos >= MIN_PTQ_COSINE, (
        f"PTQ source={result.source} INT16 cosine={cos:.6f}"
    )


@pytest.mark.slow
def test_ptq_adaround_long_iterations_maintains_int16_accuracy():
    """Long AdaRound (2k iter) regression: must not trail vanilla PTQ."""

    pytest.importorskip("psutil", reason="AdaRound optimizer requires psutil (see requirements.txt)")

    model_v, dummy_v = build_prepared_mobilenet_v2()
    vanilla = build_calibrated_sim(model_v, dummy_v)
    model, dummy = build_prepared_mobilenet_v2()
    loader = make_adaround_loader(INPUT_SIZE, num_batches=4)

    with tempfile.TemporaryDirectory() as tmpdir:
        adaround_sim = build_calibrated_sim(
            model,
            dummy,
            adaround_loader=loader,
            adaround_num_batches=2,
            adaround_iterations=MOBILENET_ADAROUND_ITERATIONS_LONG,
            adaround_export_dir=Path(tmpdir),
        )

    torch.manual_seed(43)
    x = torch.randn(EVAL_BATCH, 3, INPUT_SIZE, INPUT_SIZE)
    cos_baseline = int16_vs_fp32_cosine(vanilla.sim, x)
    cos_adaround = int16_vs_fp32_cosine(adaround_sim.sim, x)

    assert cos_baseline >= MIN_PTQ_COSINE
    assert cos_adaround >= MIN_PTQ_COSINE
    assert cos_adaround + 1e-6 >= cos_baseline - QAT_COSINE_MARGIN, (
        f"Long AdaRound below vanilla PTQ: baseline={cos_baseline:.6f} adaround={cos_adaround:.6f}"
    )

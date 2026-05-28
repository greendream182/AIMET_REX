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
"""MobileNet V2 PTQ / QAT pipeline (thin wrapper over the shared e2e skeleton).

The model-construction entry (:func:`build_prepared_mobilenet_v2`) and the
classification-specific evaluators (:func:`int16_vs_fp32_cosine`,
:func:`train_int16_qat`) live here. The heavy PTQ machinery (CLE / BN fold /
empirical bias correction / AdaRound / v2 sim / INT16 patching /
compute_encodings) lives in :mod:`aimet_torch.fixed_point.e2e.sim_builder`,
and synthetic input generators live in
:mod:`aimet_torch.fixed_point.e2e.inputs` — both are model-family-agnostic
and reused by attention / YOLO / audio backbones once they land.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

import torch
import torch.nn as nn

from aimet_torch.fixed_point import (
    ExecutionMode,
    Int16QuantizedTensor,
    quant_execution_mode,
)
from aimet_torch.fixed_point.e2e.inputs import (
    make_image_adaround_loader,
    make_image_bc_dataloader,
    make_image_sampler,
)
from aimet_torch.fixed_point.e2e.sim_builder import (
    apply_v2_empirical_bias_correction,
    build_calibrated_v2_sim,
)
from aimet_torch.fixed_point.metrics import compute_pair_metrics, int16_eval_allow_debug_float
from aimet_torch.fixed_point.metrics.thresholds import (
    MOBILENET_ADAROUND_ITERATIONS_DEFAULT,
    MOBILENET_ADAROUND_ITERATIONS_LONG,
)

INPUT_SIZE = 64
CALIB_BATCH = 2
CALIB_ITERS = 4
MobilenetVariant = Literal["mock", "full", "torchvision"]


@dataclass
class MobileNetV2SimBundle:
    """Prepared model + calibrated v2 QuantizationSim (MobileNet-flavored)."""

    sim: Any
    model: nn.Module
    dummy_input: torch.Tensor
    n_oq_patched: int
    input_size: int
    variant: MobilenetVariant


def _resolve_input_size(dummy_input: torch.Tensor, input_size: Optional[int]) -> int:
    if input_size is not None:
        return int(input_size)
    return int(dummy_input.shape[-1])


def apply_empirical_bias_correction(
    model: nn.Module,
    dummy_input: torch.Tensor,  # noqa: ARG001 (kept for signature stability)
    *,
    input_size: int,
    num_samples: int = 16,
    batch_size: int = 2,
    seed: int = 42,
) -> None:
    """Run AIMET empirical bias correction on a prepared, BN-folded float model."""

    loader = make_image_bc_dataloader(
        input_size,
        in_channels=3,
        num_samples=num_samples,
        batch=batch_size,
        seed=seed,
    )
    apply_v2_empirical_bias_correction(
        model, data_loader=loader, num_samples=num_samples
    )


def build_prepared_mobilenet_v2(
    *,
    n_class: int = 10,
    input_size: int = INPUT_SIZE,
    variant: MobilenetVariant = "mock",
) -> Tuple[nn.Module, torch.Tensor]:
    """Return ``(prepared_model, dummy_input)`` for mock, full, or torchvision MobileNet V2."""
    from aimet_torch.model_preparer import prepare_model

    if input_size % 32 != 0:
        raise ValueError(f"input_size must be divisible by 32, got {input_size}")

    torch.manual_seed(0)
    if variant == "torchvision":
        from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

        try:
            model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        except Exception:
            model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        if n_class != 1000:
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, n_class)
        model = model.eval()
    else:
        from aimet_torch.examples.mobilenet import MockMobileNetV2, MobileNetV2

        if variant == "full":
            model = MobileNetV2(n_class=n_class, input_size=input_size).eval()
        else:
            model = MockMobileNetV2(n_class=n_class, input_size=input_size).eval()
    model = prepare_model(model)
    dummy = torch.randn(1, 3, input_size, input_size)
    return model, dummy


def build_calibrated_sim(
    model: nn.Module,
    dummy_input: torch.Tensor,
    *,
    input_size: Optional[int] = None,
    variant: MobilenetVariant = "mock",
    default_param_bw: int = 8,
    default_output_bw: int = 8,
    int16_eval_bw: int = 8,
    int16_eval_symmetric: bool = True,
    output_quantizer_overrides: Optional[Dict[str, Tuple[int, bool]]] = None,
    apply_cle: bool = False,
    apply_bias_correction: bool = False,
    bias_correction_loader: Optional[Any] = None,
    bias_correction_samples: int = 16,
    adaround_loader: Optional[Iterable] = None,
    adaround_num_batches: int = 2,
    adaround_iterations: int = MOBILENET_ADAROUND_ITERATIONS_DEFAULT,
    adaround_export_dir: Optional[Path] = None,
    calibration_batches: Optional[Iterable[torch.Tensor]] = None,
) -> MobileNetV2SimBundle:
    """Build v2 QuantizationSimModel and run ``compute_encodings``.

    Thin MobileNet-flavored wrapper over
    :func:`aimet_torch.fixed_point.e2e.sim_builder.build_calibrated_v2_sim`:
    plugs in image-shaped calibration / BC data, fixes the AdaRound export
    prefix to ``mnv2``, and tags the bundle with ``input_size`` / ``variant``.

    Optional PTQ refinements (order):
      * cross-layer equalization (``apply_cle``)
      * batch-norm fold
      * empirical bias correction (``apply_bias_correction``)
      * AdaRound (``adaround_loader`` set)

    When ``calibration_batches`` is set, those image tensors are used for
    ``compute_encodings`` instead of the default random Gaussian calibration.
    """

    sz = _resolve_input_size(dummy_input, input_size)

    if calibration_batches is None:
        sampler = make_image_sampler(sz, in_channels=3, batch=CALIB_BATCH)
    else:
        sampler = None

    bc_loader = None
    if apply_bias_correction:
        bc_loader = bias_correction_loader or make_image_bc_dataloader(
            sz,
            in_channels=3,
            num_samples=bias_correction_samples,
            batch=2,
            seed=42,
        )

    base = build_calibrated_v2_sim(
        model,
        dummy_input,
        calibration_batches=calibration_batches,
        calibration_sampler=sampler,
        calibration_iters=CALIB_ITERS,
        default_param_bw=default_param_bw,
        default_output_bw=default_output_bw,
        int16_eval_bw=int16_eval_bw,
        int16_eval_symmetric=int16_eval_symmetric,
        output_quantizer_overrides=output_quantizer_overrides,
        apply_cle=apply_cle,
        apply_bn_fold=True,
        bn_fold_input_shape=(1, 3, sz, sz),
        bias_correction_data=bc_loader,
        bias_correction_num_samples=bias_correction_samples,
        adaround_loader=adaround_loader,
        adaround_num_batches=adaround_num_batches,
        adaround_iterations=adaround_iterations,
        adaround_export_dir=adaround_export_dir
        or Path("/tmp/aimet_adaround_mnv2"),
        adaround_filename_prefix="mnv2",
    )

    return MobileNetV2SimBundle(
        sim=base.sim,
        model=base.model,
        dummy_input=base.dummy_input,
        n_oq_patched=base.n_oq_patched,
        input_size=sz,
        variant=variant,
    )


def int16_vs_fp32_cosine(sim: Any, x: torch.Tensor) -> float:
    """Return cosine similarity between INT16_FIXED_EVAL and FP32_QDQ logits."""

    with torch.no_grad():
        with quant_execution_mode(ExecutionMode.FP32_QDQ):
            y_ref = sim.model(x)
            if hasattr(y_ref, "dequantize"):
                y_ref = y_ref.dequantize()
        with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
            y_int = sim.model(x)
    assert isinstance(y_int, Int16QuantizedTensor)
    with int16_eval_allow_debug_float():
        cand = y_int.to_float()
    metrics = compute_pair_metrics(
        y_ref,
        cand,
        scale=y_int.scale,
        zero_point=y_int.zero_point,
        qmin=y_int.qmin,
        qmax=y_int.qmax,
        candidate_int_repr=y_int.int_repr,
    )
    return float(metrics["cosine_similarity"])


def make_adaround_loader(
    input_size: int,
    num_batches: int = 4,
    *,
    n_class: int = 10,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Minimal batches for AdaRound: ``(images, labels)`` pairs."""

    return make_image_adaround_loader(
        input_size,
        in_channels=3,
        n_class=n_class,
        num_batches=num_batches,
        batch=CALIB_BATCH,
        seed=42,
    )


def teacher_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(x).detach()


def train_int16_qat(
    sim: Any,
    *,
    teacher: nn.Module,
    input_size: int,
    epochs: int = 20,
    lr: float = 1e-3,
    batches_per_epoch: int = 4,
    seed: int = 99,
) -> List[float]:
    """Fine-tune sim weights under INT16_FIXED_QAT_SIM against a float teacher.

    Returns per-epoch training loss (mean MSE per epoch).
    """
    sim.model.train()
    for module in sim.model.modules():
        if isinstance(module, nn.Dropout):
            module.eval()

    params = [p for p in sim.model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)
    epoch_losses: list[float] = []

    for epoch in range(epochs):
        torch.manual_seed(seed + epoch)
        running = 0.0
        for step in range(batches_per_epoch):
            x = torch.randn(CALIB_BATCH, 3, input_size, input_size)
            target = teacher_logits(teacher, x)
            optimizer.zero_grad(set_to_none=True)
            with quant_execution_mode(ExecutionMode.INT16_FIXED_QAT_SIM):
                pred = sim.model(x)
            loss = torch.nn.functional.mse_loss(pred, target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite QAT loss at epoch={epoch} step={step}")
            loss.backward()
            optimizer.step()
            running += float(loss.item())
        epoch_losses.append(running / batches_per_epoch)

    sim.model.eval()
    return epoch_losses


def train_int16_qat_on_loader(
    sim: Any,
    *,
    teacher: nn.Module,
    train_loader: Iterable,
    epochs: int = 1,
    lr: float = 1e-4,
    max_batches: int = 8,
) -> List[float]:
    """Short INT16 QAT pass using real image batches and teacher logits."""

    device = next(sim.model.parameters()).device
    teacher = teacher.to(device).eval()
    sim.model.train()
    for module in sim.model.modules():
        if isinstance(module, nn.Dropout):
            module.eval()

    params = [p for p in sim.model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)
    losses: list[float] = []
    for _epoch in range(int(epochs)):
        running = 0.0
        steps = 0
        for batch in train_loader:
            if steps >= int(max_batches):
                break
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            images = images.to(device)
            with torch.no_grad():
                target = teacher(images)
                if hasattr(target, "dequantize"):
                    target = target.dequantize()
                target = target.detach().float()
            optimizer.zero_grad(set_to_none=True)
            with quant_execution_mode(ExecutionMode.INT16_FIXED_QAT_SIM):
                pred = sim.model(images)
            if hasattr(pred, "dequantize"):
                pred = pred.dequantize()
            loss = torch.nn.functional.mse_loss(pred.float(), target)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite QAT loss.")
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            steps += 1
        if steps == 0:
            raise ValueError("QAT train_loader produced no batches.")
        losses.append(running / steps)

    sim.model.eval()
    return losses


__all__ = [
    "CALIB_BATCH",
    "CALIB_ITERS",
    "INPUT_SIZE",
    "MOBILENET_ADAROUND_ITERATIONS_DEFAULT",
    "MOBILENET_ADAROUND_ITERATIONS_LONG",
    "MobileNetV2SimBundle",
    "MobilenetVariant",
    "apply_empirical_bias_correction",
    "build_calibrated_sim",
    "build_prepared_mobilenet_v2",
    "int16_vs_fp32_cosine",
    "make_adaround_loader",
    "teacher_logits",
    "train_int16_qat",
    "train_int16_qat_on_loader",
]

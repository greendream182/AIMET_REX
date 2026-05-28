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
"""TEMPORARY reference scaffold for YOLO-family detection models.

v0 scaffold — **not exported** from ``aimet_torch.fixed_point.e2e`` (see
``__init__.py``); copy this file as ``<your_yolo_variant>.py`` and adapt.

Why a separate file:

* Detection backbones share the **same PTQ skeleton** as MobileNet (CNN + BN),
  so ``build_calibrated_v2_sim`` is reused with ``apply_bn_fold=True``.
* But the **output is multi-head** (typically tuples like ``(cls, box, obj)``
  or per-FPN-level lists). The single-tensor ``int16_vs_fp32_cosine`` and
  MSE-on-logits ``train_int16_qat`` from ``mobilenet_v2.py`` are **not
  applicable** — you must write your own evaluator and QAT loop.

The ``_MinimalDetector`` model below is a 3-head placeholder so this file is
self-contained-runnable. Replace with your real YOLOv5/v8/etc. backbone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from aimet_torch.fixed_point import ExecutionMode, quant_execution_mode
from aimet_torch.fixed_point.e2e.inputs import make_image_sampler
from aimet_torch.fixed_point.e2e.sim_builder import (
    CalibratedSimBundle,
    build_calibrated_v2_sim,
)


# ---------------------------------------------------------------------------
# Placeholder model — REPLACE with your real YOLO backbone.
# ---------------------------------------------------------------------------


class _MinimalDetector(nn.Module):
    """Tiny 3-head detector (cls / box / obj) on a single FPN level.

    Real YOLOs typically emit 3 FPN levels × (cls, box, obj). This scaffold
    only emits one level for simplicity; extend by repeating the head pattern.
    """

    def __init__(
        self,
        in_channels: int = 3,
        n_class: int = 20,
        feat_channels: int = 32,
    ) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
        )
        self.cls_head = nn.Conv2d(feat_channels, n_class, 1)
        self.box_head = nn.Conv2d(feat_channels, 4, 1)
        self.obj_head = nn.Conv2d(feat_channels, 1, 1)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.backbone(x)
        return self.cls_head(feat), self.box_head(feat), self.obj_head(feat)


# ---------------------------------------------------------------------------
# Public API of this scaffold.
# ---------------------------------------------------------------------------


@dataclass
class YoloSimBundle:
    """``CalibratedSimBundle`` + detection-specific tags (input_size, n_class)."""

    sim: Any
    model: nn.Module
    dummy_input: torch.Tensor
    n_oq_patched: int
    input_size: int
    n_class: int


def build_prepared_yolo(
    *,
    n_class: int = 20,
    input_size: int = 64,
) -> tuple[nn.Module, torch.Tensor]:
    """Return ``(prepared_model, dummy_input)``.

    REPLACE with your real YOLO. ``prepare_model`` is recommended for YOLOv5+
    (they use ``torch.cat`` etc. that benefit from rewriting).
    """

    from aimet_torch.model_preparer import prepare_model

    torch.manual_seed(0)
    model = _MinimalDetector(in_channels=3, n_class=n_class).eval()
    model = prepare_model(model)
    dummy = torch.randn(1, 3, input_size, input_size)
    return model, dummy


def build_calibrated_yolo_sim(
    model: nn.Module,
    dummy_input: torch.Tensor,
    *,
    n_class: int,
    input_size: Optional[int] = None,
    apply_cle: bool = True,
    calibration_batches: Optional[Iterable[torch.Tensor]] = None,
    adaround_loader: Optional[Iterable] = None,
    adaround_iterations: int = 80,
    adaround_export_dir: Optional[Path] = None,
) -> YoloSimBundle:
    """YOLO-flavored thin wrapper over ``build_calibrated_v2_sim``.

    Defaults vs ``mobilenet_v2.build_calibrated_sim``:

    * ``apply_cle=True`` — CLE typically helps on YOLO CNN backbones.
    * ``apply_bn_fold=True`` — standard for CNN.
    * ``bias_correction_data=None`` — empirical BC assumes single-output
      models; for multi-head detectors it may need a custom data path
      (TODO: investigate when first real YOLO lands).
    """

    sz = int(input_size) if input_size is not None else int(dummy_input.shape[-1])
    sampler = (
        make_image_sampler(sz, in_channels=3, batch=2)
        if calibration_batches is None
        else None
    )

    base: CalibratedSimBundle = build_calibrated_v2_sim(
        model,
        dummy_input,
        calibration_batches=calibration_batches,
        calibration_sampler=sampler,
        calibration_iters=4,
        apply_cle=apply_cle,
        apply_bn_fold=True,
        bn_fold_input_shape=(1, 3, sz, sz),
        bias_correction_data=None,
        adaround_loader=adaround_loader,
        adaround_num_batches=2,
        adaround_iterations=adaround_iterations,
        adaround_export_dir=adaround_export_dir
        or Path("/tmp/aimet_adaround_yolo"),
        adaround_filename_prefix="yolo",
    )
    return YoloSimBundle(
        sim=base.sim,
        model=base.model,
        dummy_input=base.dummy_input,
        n_oq_patched=base.n_oq_patched,
        input_size=sz,
        n_class=n_class,
    )


# ---------------------------------------------------------------------------
# Evaluator — CUSTOMER MUST ADAPT to their actual head layout.
# ---------------------------------------------------------------------------


@torch.no_grad()
def int16_vs_fp32_per_head_cosine(sim: Any, x: torch.Tensor) -> List[float]:
    """Per-head cosine between INT16_FIXED_EVAL and FP32_QDQ.

    REPLACE ``_to_floats`` below with your model's head layout once you've
    determined how INT16 outputs are packaged (``Int16QuantizedTensor`` per
    head, or a single concatenated tensor, etc.).
    """

    from aimet_torch.fixed_point import Int16QuantizedTensor
    from aimet_torch.fixed_point.metrics import int16_eval_allow_debug_float

    def _to_floats(out: Any) -> List[torch.Tensor]:
        if isinstance(out, (tuple, list)):
            return [_one(o) for o in out]
        return [_one(out)]

    def _one(o: Any) -> torch.Tensor:
        if isinstance(o, Int16QuantizedTensor):
            with int16_eval_allow_debug_float():
                return o.to_float()
        if hasattr(o, "dequantize"):
            return o.dequantize()
        return o

    with quant_execution_mode(ExecutionMode.FP32_QDQ):
        refs = _to_floats(sim.model(x))
    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        cands = _to_floats(sim.model(x))

    cosines: List[float] = []
    for ref, cand in zip(refs, cands):
        a = ref.float().flatten().unsqueeze(0)
        b = cand.float().flatten().unsqueeze(0)
        cosines.append(
            float(torch.nn.functional.cosine_similarity(a, b).item())
        )
    return cosines


# ---------------------------------------------------------------------------
# QAT — CUSTOMER MUST WRITE detection loss; the snippet below is a
# weighted per-head MSE distillation baseline (NOT a real detection loss).
# ---------------------------------------------------------------------------


def train_int16_qat_distill(
    sim: Any,
    *,
    teacher: nn.Module,
    input_size: int,
    epochs: int = 5,
    lr: float = 1e-3,
    batches_per_epoch: int = 4,
    batch_size: int = 2,
    head_weights: Optional[List[float]] = None,
    seed: int = 99,
) -> List[float]:
    """Baseline INT16 QAT: per-head weighted MSE against the float teacher.

    THIS IS NOT A REAL DETECTION LOSS. For meaningful detection QAT, replace
    with your standard YOLO loss (CIoU + classification BCE + objectness BCE
    + label assignment). The point of this snippet is only to show the
    boilerplate (which modes to use, how to step the optimizer).
    """

    sim.model.train()
    for module in sim.model.modules():
        if isinstance(module, nn.Dropout):
            module.eval()

    params = [p for p in sim.model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)
    epoch_losses: List[float] = []

    for epoch in range(epochs):
        torch.manual_seed(seed + epoch)
        running = 0.0
        for step in range(batches_per_epoch):
            x = torch.randn(batch_size, 3, input_size, input_size)
            with torch.no_grad():
                teacher_outs = teacher(x)
            if not isinstance(teacher_outs, (tuple, list)):
                teacher_outs = (teacher_outs,)

            optimizer.zero_grad(set_to_none=True)
            with quant_execution_mode(ExecutionMode.INT16_FIXED_QAT_SIM):
                preds = sim.model(x)
            if not isinstance(preds, (tuple, list)):
                preds = (preds,)
            assert len(preds) == len(teacher_outs), (
                f"Head count mismatch: teacher={len(teacher_outs)} sim={len(preds)}"
            )

            weights = head_weights or [1.0] * len(preds)
            loss = sum(
                w * torch.nn.functional.mse_loss(p, t)
                for w, p, t in zip(weights, preds, teacher_outs)
            )
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite QAT loss at epoch={epoch} step={step}"
                )
            loss.backward()
            optimizer.step()
            running += float(loss.item())
        epoch_losses.append(running / batches_per_epoch)

    sim.model.eval()
    return epoch_losses


if __name__ == "__main__":  # pragma: no cover
    model, dummy = build_prepared_yolo(n_class=20, input_size=64)
    bundle = build_calibrated_yolo_sim(model, dummy, n_class=20)
    print(f"yolo scaffold: n_oq_patched={bundle.n_oq_patched}")
    x = torch.randn(2, 3, 64, 64)
    try:
        per_head = int16_vs_fp32_per_head_cosine(bundle.sim, x)
        print(f"yolo scaffold: per-head cosine={per_head}")
    except Exception as exc:
        print(
            f"yolo scaffold: INT16 forward failed: {exc}\n"
            "  (expected if backend lacks INT16 support for some op; "
            "FP32_QDQ path still validates the PTQ skeleton.)"
        )

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
"""TEMPORARY reference scaffold for audio backbones (1D waveform / mel-spec).

v0 scaffold — **not exported** from ``aimet_torch.fixed_point.e2e`` (see
``__init__.py``); copy this file as ``<your_audio_model>.py`` and adapt.

Two input modalities are demonstrated below — pick the one matching your
model:

* **1D waveform** ``(B, C, T)`` — wav2vec / sincnet / 1D-CRNN style.
  Use ``make_audio_1d_sampler`` and ``_Minimal1DAudioModel``.
* **Mel-spectrogram** ``(B, 1, n_mels, n_frames)`` — log-mel frontend +
  2D CNN/CRNN. Use ``make_melspec_sampler`` and ``_MinimalMelspecModel``.

Output semantics vary heavily by task (classification / CTC / embedding) —
the example evaluator below assumes **single logits tensor**; CTC and
embedding need their own evaluators (TODO when first real audio model lands).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Optional, Tuple

import torch
import torch.nn as nn

from aimet_torch.fixed_point.e2e.inputs import (
    InputSampler,
    make_audio_1d_sampler,
    make_melspec_sampler,
)
from aimet_torch.fixed_point.e2e.sim_builder import (
    CalibratedSimBundle,
    build_calibrated_v2_sim,
)

AudioInputKind = Literal["audio_1d", "melspec"]


# ---------------------------------------------------------------------------
# Placeholder models — REPLACE with your real audio backbone.
# ---------------------------------------------------------------------------


class _Minimal1DAudioModel(nn.Module):
    """Tiny 1D-CNN classifier on ``(B, channels, T)`` waveforms."""

    def __init__(
        self,
        channels: int = 1,
        n_class: int = 10,
        feat: int = 16,
    ) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(channels, feat, 9, stride=4, padding=4),
            nn.BatchNorm1d(feat),
            nn.ReLU(inplace=True),
            nn.Conv1d(feat, feat * 2, 5, stride=4, padding=2),
            nn.BatchNorm1d(feat * 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(feat * 2, n_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.body(x).flatten(1)
        return self.head(x)


class _MinimalMelspecModel(nn.Module):
    """Tiny 2D-CNN classifier on ``(B, 1, n_mels, n_frames)`` mel spectrograms."""

    def __init__(self, n_class: int = 10, feat: int = 16) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, feat, 3, stride=2, padding=1),
            nn.BatchNorm2d(feat),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat, feat * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(feat * 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(feat * 2, n_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.body(x).flatten(1)
        return self.head(x)


# ---------------------------------------------------------------------------
# Public API of this scaffold.
# ---------------------------------------------------------------------------


@dataclass
class AudioSimBundle:
    """``CalibratedSimBundle`` + audio-specific tags."""

    sim: Any
    model: nn.Module
    dummy_input: torch.Tensor
    n_oq_patched: int
    input_kind: AudioInputKind
    input_shape: Tuple[int, ...]


def build_prepared_audio_1d(
    *,
    channels: int = 1,
    length: int = 2048,
    n_class: int = 10,
) -> tuple[nn.Module, torch.Tensor]:
    """Return ``(prepared_model, dummy_input)`` for the 1D waveform path."""

    from aimet_torch.model_preparer import prepare_model

    torch.manual_seed(0)
    model = _Minimal1DAudioModel(channels=channels, n_class=n_class).eval()
    model = prepare_model(model)
    dummy = torch.randn(1, channels, length)
    return model, dummy


def build_prepared_melspec(
    *,
    n_mels: int = 40,
    n_frames: int = 128,
    n_class: int = 10,
) -> tuple[nn.Module, torch.Tensor]:
    """Return ``(prepared_model, dummy_input)`` for the mel-spec path."""

    from aimet_torch.model_preparer import prepare_model

    torch.manual_seed(0)
    model = _MinimalMelspecModel(n_class=n_class).eval()
    model = prepare_model(model)
    dummy = torch.randn(1, 1, n_mels, n_frames)
    return model, dummy


def _audio_sampler_from_dummy(dummy: torch.Tensor, kind: AudioInputKind) -> InputSampler:
    if kind == "audio_1d":
        batch, channels, length = dummy.shape[0] * 2, dummy.shape[1], dummy.shape[2]
        return make_audio_1d_sampler(length, channels=channels, batch=batch)
    if kind == "melspec":
        batch = dummy.shape[0] * 2
        n_mels, n_frames = dummy.shape[2], dummy.shape[3]
        return make_melspec_sampler(n_mels, n_frames, batch=batch)
    raise ValueError(f"Unknown audio input kind: {kind!r}")


def build_calibrated_audio_sim(
    model: nn.Module,
    dummy_input: torch.Tensor,
    *,
    input_kind: AudioInputKind,
    calibration_batches: Optional[Iterable[torch.Tensor]] = None,
    apply_cle: bool = False,
    adaround_loader: Optional[Iterable] = None,
    adaround_iterations: int = 80,
    adaround_export_dir: Optional[Path] = None,
) -> AudioSimBundle:
    """Audio-flavored thin wrapper over ``build_calibrated_v2_sim``.

    Defaults vs ``mobilenet_v2.build_calibrated_sim``:

    * ``apply_cle=False`` — CLE is a convolution-pair heuristic; not always
      meaningful on 1D/audio CNNs. Enable only if you've verified it helps.
    * ``apply_bn_fold=True`` — works for ``BatchNorm1d`` / ``BatchNorm2d``.
    * Sampler is auto-derived from ``input_kind`` + ``dummy_input`` shape.
    """

    sampler = (
        _audio_sampler_from_dummy(dummy_input, input_kind)
        if calibration_batches is None
        else None
    )
    bn_fold_shape = (1, *tuple(int(d) for d in dummy_input.shape[1:]))

    base: CalibratedSimBundle = build_calibrated_v2_sim(
        model,
        dummy_input,
        calibration_batches=calibration_batches,
        calibration_sampler=sampler,
        calibration_iters=4,
        apply_cle=apply_cle,
        apply_bn_fold=True,
        bn_fold_input_shape=bn_fold_shape,
        bias_correction_data=None,
        adaround_loader=adaround_loader,
        adaround_num_batches=2,
        adaround_iterations=adaround_iterations,
        adaround_export_dir=adaround_export_dir
        or Path(f"/tmp/aimet_adaround_audio_{input_kind}"),
        adaround_filename_prefix=f"audio_{input_kind}",
    )
    return AudioSimBundle(
        sim=base.sim,
        model=base.model,
        dummy_input=base.dummy_input,
        n_oq_patched=base.n_oq_patched,
        input_kind=input_kind,
        input_shape=tuple(int(d) for d in dummy_input.shape),
    )


# ---------------------------------------------------------------------------
# Evaluator — assumes single-logits output (classification).
# REPLACE for CTC / embedding tasks.
# ---------------------------------------------------------------------------


@torch.no_grad()
def int16_vs_fp32_audio_cosine(sim: Any, x: torch.Tensor) -> float:
    """Cosine between INT16_FIXED_EVAL and FP32_QDQ on a single logits output.

    For CTC: replace with cosine on per-frame logits (flatten T into batch)
    or compare WER on a small dev set.

    For embedding: cosine on the embedding tensor (no logits needed).
    """

    from aimet_torch.fixed_point import ExecutionMode, Int16QuantizedTensor, quant_execution_mode
    from aimet_torch.fixed_point.metrics import compute_pair_metrics, int16_eval_allow_debug_float

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


# ---------------------------------------------------------------------------
# QAT — same shape gotcha as mobilenet_v2.train_int16_qat: hard-coded input
# shape. We provide a parametric variant for audio inputs; replace loss_fn
# with your CTC / contrastive / triplet loss as needed.
# ---------------------------------------------------------------------------


def train_int16_qat_audio(
    sim: Any,
    *,
    teacher: nn.Module,
    input_shape: Tuple[int, ...],
    epochs: int = 5,
    lr: float = 1e-3,
    batches_per_epoch: int = 4,
    seed: int = 99,
) -> list[float]:
    """Distillation INT16 QAT loop for audio classification.

    ``input_shape`` is the per-sample shape (without batch dim), e.g.
    ``(1, 16000)`` for 1D waveform or ``(1, 40, 128)`` for mel-spec.

    Replace MSE with your task loss (CTC: ``F.ctc_loss``; classification with
    real labels: ``F.cross_entropy``; embedding: triplet loss) for production.
    """

    from aimet_torch.fixed_point import ExecutionMode, quant_execution_mode

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
            x = torch.randn(2, *input_shape)
            with torch.no_grad():
                target = teacher(x).detach()
            optimizer.zero_grad(set_to_none=True)
            with quant_execution_mode(ExecutionMode.INT16_FIXED_QAT_SIM):
                pred = sim.model(x)
            loss = torch.nn.functional.mse_loss(pred, target)
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
    # Smoke 1: 1D waveform path.
    model_1d, dummy_1d = build_prepared_audio_1d(channels=1, length=2048, n_class=10)
    bundle_1d = build_calibrated_audio_sim(model_1d, dummy_1d, input_kind="audio_1d")
    print(f"audio_1d scaffold: n_oq_patched={bundle_1d.n_oq_patched}")
    x_1d = torch.randn(2, 1, 2048)
    try:
        cos = int16_vs_fp32_audio_cosine(bundle_1d.sim, x_1d)
        print(f"audio_1d scaffold: INT16 vs FP32 cosine={cos:.6f}")
    except Exception as exc:
        print(f"audio_1d scaffold: INT16 forward failed: {exc}")

    # Smoke 2: mel-spec path.
    model_mel, dummy_mel = build_prepared_melspec(n_mels=40, n_frames=128, n_class=10)
    bundle_mel = build_calibrated_audio_sim(model_mel, dummy_mel, input_kind="melspec")
    print(f"melspec scaffold: n_oq_patched={bundle_mel.n_oq_patched}")
    x_mel = torch.randn(2, 1, 40, 128)
    try:
        cos = int16_vs_fp32_audio_cosine(bundle_mel.sim, x_mel)
        print(f"melspec scaffold: INT16 vs FP32 cosine={cos:.6f}")
    except Exception as exc:
        print(f"melspec scaffold: INT16 forward failed: {exc}")

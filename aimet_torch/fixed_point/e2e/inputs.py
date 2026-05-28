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
"""Synthetic input generators for fixed-point e2e calibration / AdaRound / BC.

Three families covered:

* Image classification ``(B, in_channels, H, W)`` — MobileNet, ViT / Swin.
* 1D audio ``(B, channels, T)`` — waveform-style audio backbones.
* Mel spectrogram ``(B, 1, n_mels, n_frames)`` — log-mel audio frontends.

Output formats:

* ``InputSampler`` = ``Callable[[], Tensor]`` returning **one** batch per call.
* ``*_calibration_batches`` = list of pre-materialized batches (eager, reproducible).
* ``*_bc_dataloader`` = ``DataLoader`` yielding ``(input, label)`` — what
  ``aimet_torch.bias_correction.correct_bias`` expects.
* ``make_image_adaround_loader`` = list of ``(images, labels)`` tuples,
  matching the input contract of ``aimet_torch.v2.adaround.Adaround``.

All helpers accept an optional ``seed`` for local determinism without touching
the global RNG state more than necessary.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

InputSampler = Callable[[], torch.Tensor]


def _seeded_randn(*shape: int, seed: Optional[int] = None) -> torch.Tensor:
    if seed is None:
        return torch.randn(*shape)
    gen = torch.Generator().manual_seed(int(seed))
    return torch.randn(*shape, generator=gen)


# ---------------------------------------------------------------------------
# Image (B, in_channels, H, W)
# ---------------------------------------------------------------------------


def make_image_sampler(
    input_size: int,
    *,
    in_channels: int = 3,
    batch: int = 2,
    seed: Optional[int] = None,
) -> InputSampler:
    """Return ``Callable[[], Tensor]`` producing ``(batch, in_channels, sz, sz)``."""

    if seed is None:
        def _sample() -> torch.Tensor:
            return torch.randn(batch, in_channels, input_size, input_size)
        return _sample

    counter = [0]

    def _sample_seeded() -> torch.Tensor:
        gen = torch.Generator().manual_seed(int(seed) + counter[0])
        counter[0] += 1
        return torch.randn(batch, in_channels, input_size, input_size, generator=gen)

    return _sample_seeded


def make_image_calibration_batches(
    input_size: int,
    *,
    in_channels: int = 3,
    batch: int = 2,
    iters: int = 4,
    seed: Optional[int] = None,
) -> List[torch.Tensor]:
    """Materialize ``iters`` random image batches."""

    sampler = make_image_sampler(
        input_size, in_channels=in_channels, batch=batch, seed=seed
    )
    return [sampler() for _ in range(iters)]


def make_image_bc_dataloader(
    input_size: int,
    *,
    in_channels: int = 3,
    num_samples: int = 16,
    batch: int = 2,
    seed: int = 42,
) -> DataLoader:
    """DataLoader of ``(image, label)`` for ``correct_bias`` (labels are zeros)."""

    torch.manual_seed(int(seed))
    images = torch.randn(num_samples, in_channels, input_size, input_size)
    labels = torch.zeros(num_samples, dtype=torch.long)
    return DataLoader(
        TensorDataset(images, labels),
        batch_size=batch,
        shuffle=False,
    )


def make_image_adaround_loader(
    input_size: int,
    *,
    in_channels: int = 3,
    n_class: int = 10,
    num_batches: int = 4,
    batch: int = 2,
    seed: int = 42,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """List of ``(images, labels)`` for ``v2.Adaround.apply_adaround``.

    ``labels`` shape is ``(batch, n_class)`` (matches existing MobileNet
    AdaRound contract; AdaRound itself ignores labels, but the existing
    callers rely on this exact shape).
    """

    torch.manual_seed(int(seed))
    return [
        (
            torch.randn(batch, in_channels, input_size, input_size),
            torch.zeros(batch, n_class, dtype=torch.long),
        )
        for _ in range(num_batches)
    ]


# ---------------------------------------------------------------------------
# 1D audio (B, channels, T)
# ---------------------------------------------------------------------------


def make_audio_1d_sampler(
    length: int,
    *,
    channels: int = 1,
    batch: int = 2,
    seed: Optional[int] = None,
) -> InputSampler:
    """Return ``Callable[[], Tensor]`` producing ``(batch, channels, length)``."""

    if seed is None:
        def _sample() -> torch.Tensor:
            return torch.randn(batch, channels, length)
        return _sample

    counter = [0]

    def _sample_seeded() -> torch.Tensor:
        gen = torch.Generator().manual_seed(int(seed) + counter[0])
        counter[0] += 1
        return torch.randn(batch, channels, length, generator=gen)

    return _sample_seeded


def make_audio_1d_calibration_batches(
    length: int,
    *,
    channels: int = 1,
    batch: int = 2,
    iters: int = 4,
    seed: Optional[int] = None,
) -> List[torch.Tensor]:
    """Materialize ``iters`` random 1D audio batches."""

    sampler = make_audio_1d_sampler(
        length, channels=channels, batch=batch, seed=seed
    )
    return [sampler() for _ in range(iters)]


# ---------------------------------------------------------------------------
# Mel spectrogram (B, 1, n_mels, n_frames)
# ---------------------------------------------------------------------------


def make_melspec_sampler(
    n_mels: int,
    n_frames: int,
    *,
    batch: int = 2,
    seed: Optional[int] = None,
) -> InputSampler:
    """Return ``Callable[[], Tensor]`` producing ``(batch, 1, n_mels, n_frames)``."""

    if seed is None:
        def _sample() -> torch.Tensor:
            return torch.randn(batch, 1, n_mels, n_frames)
        return _sample

    counter = [0]

    def _sample_seeded() -> torch.Tensor:
        gen = torch.Generator().manual_seed(int(seed) + counter[0])
        counter[0] += 1
        return torch.randn(batch, 1, n_mels, n_frames, generator=gen)

    return _sample_seeded


def make_melspec_calibration_batches(
    n_mels: int,
    n_frames: int,
    *,
    batch: int = 2,
    iters: int = 4,
    seed: Optional[int] = None,
) -> List[torch.Tensor]:
    """Materialize ``iters`` random mel-spectrogram batches."""

    sampler = make_melspec_sampler(n_mels, n_frames, batch=batch, seed=seed)
    return [sampler() for _ in range(iters)]


__all__ = [
    "InputSampler",
    "make_audio_1d_calibration_batches",
    "make_audio_1d_sampler",
    "make_image_adaround_loader",
    "make_image_bc_dataloader",
    "make_image_calibration_batches",
    "make_image_sampler",
    "make_melspec_calibration_batches",
    "make_melspec_sampler",
]

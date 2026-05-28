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
"""TEMPORARY reference scaffold for attention / ViT / Swin / Transformer models.

v0 scaffold — **not exported** from ``aimet_torch.fixed_point.e2e`` (see
``__init__.py``); copy this file as ``<your_real_model>.py`` and adapt.

Why a separate file from ``mobilenet_v2.py``:

* attention backbones generally have **no BN** (LayerNorm instead) → must pass
  ``apply_bn_fold=False`` and ``apply_cle=False`` to the shared PTQ skeleton.
* output semantics are still "single logits tensor", so the existing
  ``int16_vs_fp32_cosine`` and ``train_int16_qat`` from ``mobilenet_v2.py``
  can be **reused unchanged** as long as the input is ``(B, 3, H, W)``.

The ``_MinimalAttentionLike`` model below is **only a placeholder** so this
file is runnable end-to-end without external deps. Replace with your real
ViT / Swin / MultiheadAttention backbone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
import torch.nn as nn

from aimet_torch.fixed_point.e2e.inputs import make_image_sampler
from aimet_torch.fixed_point.e2e.sim_builder import (
    CalibratedSimBundle,
    build_calibrated_v2_sim,
)


# ---------------------------------------------------------------------------
# Placeholder model — REPLACE with your real attention backbone.
# ---------------------------------------------------------------------------


class _MinimalAttentionLike(nn.Module):
    """Tiny Transformer-style classifier (patch embed → LN → MLP block → head).

    Does **not** use ``nn.MultiheadAttention`` so that INT16 dispatch can run
    end-to-end on this scaffold without backend gaps. Your real model very
    likely *does* use MultiheadAttention — make sure your backend supports
    softmax / matmul in INT16, or fall back to FP32_QDQ for those ops.
    """

    def __init__(
        self,
        input_size: int = 64,
        in_channels: int = 3,
        patch: int = 8,
        dim: int = 64,
        n_class: int = 10,
    ) -> None:
        super().__init__()
        if input_size % patch != 0:
            raise ValueError(f"input_size {input_size} must be divisible by patch {patch}")
        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=patch, stride=patch)
        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * 2, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        h = self.norm1(x)
        h = self.fc2(self.act(self.fc1(h)))
        x = x + h
        x = self.norm2(x).mean(dim=1)
        return self.head(x)


# ---------------------------------------------------------------------------
# Public API of this scaffold (mirror ``mobilenet_v2.py`` structure).
# ---------------------------------------------------------------------------


@dataclass
class AttentionSimBundle:
    """``CalibratedSimBundle`` + attention-specific tags (input_size)."""

    sim: Any
    model: nn.Module
    dummy_input: torch.Tensor
    n_oq_patched: int
    input_size: int


def build_prepared_attention(
    *,
    n_class: int = 10,
    input_size: int = 64,
) -> tuple[nn.Module, torch.Tensor]:
    """Return ``(prepared_model, dummy_input)``.

    REPLACE the model construction with your real ViT / Swin / etc.
    ``prepare_model`` is **optional** for attention — many ViT
    implementations are already ``nn.Module``-only and need no rewrite.
    """

    torch.manual_seed(0)
    model = _MinimalAttentionLike(
        input_size=input_size, in_channels=3, n_class=n_class
    ).eval()
    dummy = torch.randn(1, 3, input_size, input_size)
    return model, dummy


def build_calibrated_attention_sim(
    model: nn.Module,
    dummy_input: torch.Tensor,
    *,
    input_size: Optional[int] = None,
    calibration_batches: Optional[Iterable[torch.Tensor]] = None,
    adaround_loader: Optional[Iterable] = None,
    adaround_iterations: int = 80,
    adaround_export_dir: Optional[Path] = None,
) -> AttentionSimBundle:
    """Attention-flavored thin wrapper over ``build_calibrated_v2_sim``.

    Differences vs ``mobilenet_v2.build_calibrated_sim``:

    * ``apply_cle=False`` (CLE is a CNN trick).
    * ``apply_bn_fold=False`` (no BN in typical attention backbones).
    * ``bias_correction_data=None`` (rarely useful for attention; opt-in if
      your model has BN somewhere).
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
        apply_cle=False,
        apply_bn_fold=False,
        bn_fold_input_shape=(1, 3, sz, sz),
        bias_correction_data=None,
        adaround_loader=adaround_loader,
        adaround_num_batches=2,
        adaround_iterations=adaround_iterations,
        adaround_export_dir=adaround_export_dir
        or Path("/tmp/aimet_adaround_attention"),
        adaround_filename_prefix="attention",
    )
    return AttentionSimBundle(
        sim=base.sim,
        model=base.model,
        dummy_input=base.dummy_input,
        n_oq_patched=base.n_oq_patched,
        input_size=sz,
    )


# ---------------------------------------------------------------------------
# Evaluator / QAT — for single-logits attention classifiers you can REUSE
# ``mobilenet_v2.int16_vs_fp32_cosine`` and ``mobilenet_v2.train_int16_qat``
# directly. The block below shows how, no need to re-implement.
# ---------------------------------------------------------------------------


def example_int16_vs_fp32_cosine(sim: Any, x: torch.Tensor) -> float:
    """Show the recommended pattern: reuse ``mobilenet_v2.int16_vs_fp32_cosine``."""

    from aimet_torch.fixed_point.e2e.mobilenet_v2 import int16_vs_fp32_cosine

    return int16_vs_fp32_cosine(sim, x)


def example_train_int16_qat(
    sim: Any,
    *,
    teacher: nn.Module,
    input_size: int,
    epochs: int = 5,
) -> list[float]:
    """Show the recommended pattern: reuse ``mobilenet_v2.train_int16_qat``.

    Only valid when your input is ``(B, 3, input_size, input_size)`` and your
    output is a single logits tensor (which is the case for ViT classifiers).
    Otherwise see the YOLO / audio scaffolds.
    """

    from aimet_torch.fixed_point.e2e.mobilenet_v2 import train_int16_qat

    return train_int16_qat(
        sim, teacher=teacher, input_size=input_size, epochs=epochs
    )


if __name__ == "__main__":  # pragma: no cover
    # Self-contained smoke: scaffold runs end-to-end on the placeholder model.
    model, dummy = build_prepared_attention(n_class=10, input_size=64)
    bundle = build_calibrated_attention_sim(model, dummy)
    print(f"attention scaffold: n_oq_patched={bundle.n_oq_patched}")
    x = torch.randn(2, 3, 64, 64)
    try:
        cos = example_int16_vs_fp32_cosine(bundle.sim, x)
        print(f"attention scaffold: INT16 vs FP32 cosine={cos:.6f}")
    except Exception as exc:
        print(
            f"attention scaffold: INT16 forward failed: {exc}\n"
            "  (expected if backend lacks INT16 support for some op; "
            "FP32_QDQ path still validates the PTQ skeleton.)"
        )

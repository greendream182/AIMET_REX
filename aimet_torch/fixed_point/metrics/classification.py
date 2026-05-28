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
"""Top-k accuracy and agreement metrics for classification models.

Reusable across ImageNet, CIFAR, custom classifiers — any model whose
forward returns logits ``[N, C]`` (or a quantized carrier that
:func:`dequantize_logits` can unwrap) and whose loader yields
``(images, labels[, ...])`` tuples.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch.utils.data import DataLoader

from aimet_torch.fixed_point import ExecutionMode, quant_execution_mode
from aimet_torch.fixed_point.metrics.logits import (
    _resolve_model_and_device,
    dequantize_logits,
)


@torch.no_grad()
def top_k_accuracy(
    sim_or_model: Any,
    loader: DataLoader,
    mode: ExecutionMode,
    *,
    k: int = 1,
    max_batches: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> float:
    """Top-``k`` accuracy under ``mode`` over the (label-aware) ``loader``.

    For ``k > 1`` a sample counts as correct if its true label is among the
    top-``k`` predicted classes. Returns the fraction in ``[0, 1]``.
    """

    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")

    model, device = _resolve_model_and_device(sim_or_model, device)

    correct = 0
    total = 0
    with quant_execution_mode(mode):
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images = batch[0].to(device)
            labels = batch[1].to(device)
            logits = dequantize_logits(model(images))
            if k == 1:
                preds = logits.argmax(dim=-1)
                correct += int((preds == labels).sum().item())
            else:
                top_k = logits.topk(k, dim=-1).indices  # [N, k]
                hits = (top_k == labels.unsqueeze(-1)).any(dim=-1)
                correct += int(hits.sum().item())
            total += int(labels.numel())
    if total == 0:
        return 0.0
    return correct / total


def top1_accuracy(
    sim_or_model: Any,
    loader: DataLoader,
    mode: ExecutionMode,
    *,
    max_batches: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> float:
    """Top-1 accuracy — convenience alias for :func:`top_k_accuracy` with ``k=1``."""

    return top_k_accuracy(
        sim_or_model,
        loader,
        mode,
        k=1,
        max_batches=max_batches,
        device=device,
    )


def top1_drop(reference_acc: float, candidate_acc: float) -> float:
    """Absolute non-negative top-1 drop (``max(0, ref - cand)``)."""

    return max(0.0, float(reference_acc) - float(candidate_acc))


@torch.no_grad()
def top_k_prediction_agreement(
    sim_or_model: Any,
    loader: DataLoader,
    *,
    ref_mode: ExecutionMode = ExecutionMode.FP32_QDQ,
    cand_mode: ExecutionMode = ExecutionMode.INT16_FIXED_EVAL,
    k: int = 1,
    max_batches: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> float:
    """Fraction of samples whose top-``k`` predictions agree across modes.

    Unlike :func:`top_k_accuracy` this does **not** look at the labels —
    it measures *decision-boundary stability* between two execution modes,
    answering "how often does the model predict the same class regardless
    of quantization?". Useful as a label-free proxy for quantization
    quality.
    """

    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")

    model, device = _resolve_model_and_device(sim_or_model, device)

    agree = 0
    total = 0
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        if isinstance(batch, (list, tuple)):
            images = batch[0].to(device)
        else:
            images = batch.to(device)
        with quant_execution_mode(ref_mode):
            y_ref = dequantize_logits(model(images))
        with quant_execution_mode(cand_mode):
            y_cand = dequantize_logits(model(images))
        if k == 1:
            ref_top = y_ref.argmax(dim=-1)
            cand_top = y_cand.argmax(dim=-1)
            agree += int((ref_top == cand_top).sum().item())
        else:
            ref_top = y_ref.topk(k, dim=-1).indices  # [N, k]
            cand_top = y_cand.topk(k, dim=-1).indices  # [N, k]
            ref_set = ref_top.sort(dim=-1).values
            cand_set = cand_top.sort(dim=-1).values
            hits = (ref_set == cand_set).all(dim=-1)
            agree += int(hits.sum().item())
        total += int(images.shape[0])
    if total == 0:
        return 0.0
    return agree / total

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
"""Logits-level diagnostics across execution modes.

These helpers compare the **final** network output between two
:class:`ExecutionMode` choices (e.g. ``FP32_QDQ`` vs ``INT16_FIXED_EVAL``)
on the *same* inputs. The single-batch :func:`logits_cosine` is fully
model-agnostic ``(model, inputs)``; loader-aware wrappers (which average
the cosine over a few batches) accept either a ``sim`` (anything with a
``.model`` attribute) or a bare ``nn.Module``.

Designed to be reused beyond ImageNet — pass any classification /
regression model whose forward returns a single output tensor (or a
quantized carrier with a ``dequantize`` method).
"""

from __future__ import annotations

from typing import Any, Iterable, List, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from aimet_torch.fixed_point import ExecutionMode, quant_execution_mode
from aimet_torch.fixed_point.metrics.accuracy import compute_pair_metrics
from aimet_torch.fixed_point.metrics.flags import int16_eval_allow_debug_float
from aimet_torch.fixed_point.tensor import Int16QuantizedTensor

# Default sweep mirrors :data:`compare.DEFAULT_COMPARE_MODES` minus the
# reference. Update both sides if you ever extend the mode lattice.
DEFAULT_VS_FP32_MODES: Tuple[ExecutionMode, ...] = (
    ExecutionMode.INT16_FIXED_EVAL,
    ExecutionMode.FIXED_SCALE_QDQ,
    ExecutionMode.FP16_QDQ,
)


def dequantize_logits(y: Any) -> torch.Tensor:
    """Coerce a forward output to a float ``Tensor`` for logit comparisons.

    Handles three shapes commonly produced by the fixed-point stack:

    1. ``DequantizedTensor`` / any object with a ``dequantize()`` method.
    2. :class:`Int16QuantizedTensor` — converted via
       :func:`int16_eval_allow_debug_float` to satisfy the debug guard.
    3. Plain :class:`torch.Tensor` — returned as-is.
    """

    if hasattr(y, "dequantize") and not isinstance(y, torch.Tensor):
        return y.dequantize()
    if isinstance(y, Int16QuantizedTensor):
        with int16_eval_allow_debug_float():
            return y.to_float()
    return y


def _resolve_model_and_device(
    sim_or_model: Any, device: Optional[torch.device]
) -> Tuple[nn.Module, torch.device]:
    """Accept either ``sim`` (with ``.model``) or a bare ``nn.Module``."""

    model = getattr(sim_or_model, "model", sim_or_model)
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    else:
        model = model.to(device)
    return model, device


def _normalize_inputs(
    inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
) -> Tuple[torch.Tensor, ...]:
    if isinstance(inputs, torch.Tensor):
        return (inputs,)
    return tuple(inputs)


@torch.no_grad()
def logits_cosine(
    model: nn.Module,
    inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    *,
    ref_mode: ExecutionMode = ExecutionMode.FP32_QDQ,
    cand_mode: ExecutionMode = ExecutionMode.INT16_FIXED_EVAL,
) -> float:
    """Single-batch logits cosine between two execution modes.

    Pure ``(model, inputs)`` signature — no loader, no labels.
    """

    args = _normalize_inputs(inputs)
    with quant_execution_mode(ref_mode):
        y_ref = dequantize_logits(model(*args))
    with quant_execution_mode(cand_mode):
        y_cand = dequantize_logits(model(*args))
    return float(compute_pair_metrics(y_ref, y_cand)["cosine_similarity"])


@torch.no_grad()
def mean_logits_cosine_on_loader(
    sim_or_model: Any,
    loader: DataLoader,
    *,
    ref_mode: ExecutionMode = ExecutionMode.FP32_QDQ,
    cand_mode: ExecutionMode = ExecutionMode.INT16_FIXED_EVAL,
    max_batches: int = 2,
    device: Optional[torch.device] = None,
) -> float:
    """Mean batch logits cosine between two modes across the first ``max_batches`` batches.

    ``loader`` may yield tuples ``(images, labels, ...)`` or bare tensors;
    the first element is taken as the model input.
    """

    model, device = _resolve_model_and_device(sim_or_model, device)

    scores: List[float] = []
    for idx, batch in enumerate(loader):
        if idx >= max_batches:
            break
        if isinstance(batch, (list, tuple)):
            images = batch[0].to(device)
        else:
            images = batch.to(device)
        scores.append(
            logits_cosine(model, images, ref_mode=ref_mode, cand_mode=cand_mode)
        )
    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))


def mean_logits_cosine_vs_fp32(
    sim_or_model: Any,
    loader: DataLoader,
    *,
    cand_modes: Iterable[ExecutionMode] = DEFAULT_VS_FP32_MODES,
    max_batches: int = 2,
    device: Optional[torch.device] = None,
) -> Mapping[str, float]:
    """Sweep multiple candidate modes vs ``FP32_QDQ``; return ``{mode.value: cosine}``.

    Keys are :attr:`ExecutionMode.value` strings (e.g. ``"int16_fixed_eval"``)
    so downstream JSON reports stay stable across refactors.
    """

    return {
        mode.value: mean_logits_cosine_on_loader(
            sim_or_model,
            loader,
            ref_mode=ExecutionMode.FP32_QDQ,
            cand_mode=mode,
            max_batches=max_batches,
            device=device,
        )
        for mode in cand_modes
    }

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
"""Helpers for INT16_FIXED_QAT_SIM fine-tuning (MRNN and similar graphs)."""

from __future__ import annotations

import gc
import time
from typing import Callable, Iterable, Iterator, Literal, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from aimet_torch.fixed_point.execution_mode import ExecutionMode, quant_execution_mode
from aimet_torch.fixed_point.qat.carrier import clear_int16_carriers
from aimet_torch.utils_rx import set_train_mode_freeze_bn
from aimet_torch.v2.utils import enable_recompute, no_recompute

QatTrainScope = Literal["weights", "head", "all"]
QatOptimizerName = Literal["sgd", "adam"]


def _make_qat_optimizer(
    trainable: list[nn.Parameter],
    *,
    name: QatOptimizerName,
    lr: float,
) -> torch.optim.Optimizer:
    if name == "adam":
        return torch.optim.Adam(trainable, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(trainable, lr=lr)
    raise ValueError(f"optimizer must be 'sgd' or 'adam'; got {name!r}.")


def _snapshot_trainable_state(trainable: list[nn.Parameter]) -> list[torch.Tensor]:
    return [p.detach().cpu().clone() for p in trainable]


def _restore_trainable_state(
    trainable: list[nn.Parameter],
    snapshot: list[torch.Tensor],
) -> None:
    for param, saved in zip(trainable, snapshot, strict=True):
        param.data.copy_(saved.to(param.device, non_blocking=torch.cuda.is_available()))


def release_qat_grads(model: nn.Module | None = None) -> None:
    """Drop per-step QAT state only (no allocator sync)."""

    if model is not None:
        model.zero_grad(set_to_none=True)
    clear_int16_carriers()


def release_cuda_memory(model: nn.Module | None = None) -> None:
    """Drop autograd grads and return cached CUDA blocks to the allocator."""

    release_qat_grads(model)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def warm_int16_qat_lut_cache(
    model: nn.Module,
    sample_input: torch.Tensor,
    device: torch.device,
) -> float:
    """One no-grad INT16 QAT forward to bake per-module PWL/CLZ LUT caches."""

    model.train()
    t0 = time.perf_counter()
    with torch.no_grad():
        with quant_execution_mode(ExecutionMode.INT16_FIXED_QAT_SIM):
            model(sample_input.to(device, non_blocking=torch.cuda.is_available()))
    release_qat_grads(model)
    return time.perf_counter() - t0


def cache_training_batches(
    data_iter: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    *,
    max_batches: int,
    batch_size: Optional[int] = None,
    cache_on_gpu: bool = True,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Materialize a small set of QAT batches to avoid audio I/O stalls."""

    if max_batches <= 0:
        return []

    cached: list[tuple[torch.Tensor, torch.Tensor]] = []
    target = device if cache_on_gpu else torch.device("cpu")
    for batch_idx, (inputs, labels) in enumerate(data_iter):
        if batch_idx >= max_batches:
            break
        if batch_size is not None and batch_size > 0 and inputs.size(0) > batch_size:
            inputs = inputs[:batch_size]
            labels = labels[:batch_size]
        cached.append(
            (
                inputs.to(target, non_blocking=torch.cuda.is_available()),
                labels.to(target, non_blocking=torch.cuda.is_available()),
            )
        )
    return cached


def select_qat_trainable_parameters(
    model: nn.Module,
    scope: QatTrainScope = "weights",
) -> list[nn.Parameter]:
    """Pick parameters safe to update under INT16_FIXED_QAT_SIM.

    ``weights`` — all ``requires_grad`` tensors except AIMET quantizers (recommended).
    ``head`` — only ``fc0`` / ``fc1`` weights (smoke / last-layer tune).
    ``all`` — same as ``weights``; quantizer tensors are never included (they NaN scales).
    """

    if scope not in ("weights", "head", "all"):
        raise ValueError(f"scope must be 'weights', 'head', or 'all'; got {scope!r}.")
    if scope == "all":
        scope = "weights"

    params: list[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "quantizer" in name.lower():
            continue
        if scope == "head" and not (name.startswith("fc0.") or name.startswith("fc1.")):
            continue
        params.append(p)
    return params


def suggest_int16_qat_batch_size(
    loader_batch_size: int,
    *,
    min_batch: int = 4,
) -> int:
    """Heuristic QAT micro-batch from free GPU memory (MRNN-scale graphs)."""

    if loader_batch_size <= min_batch or not torch.cuda.is_available():
        return loader_batch_size
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    # MRNN full-graph INT16 QAT at bs=64 often needs ~30 GiB; ~400 MiB / sample.
    per_sample = max(int(total_bytes * 0.45 / 64), 128 * 1024 * 1024)
    suggested = max(min_batch, int(free_bytes * 0.65 / per_sample))
    return max(min_batch, min(loader_batch_size, suggested))


def _to_float_logits(output: torch.Tensor) -> torch.Tensor:
    if hasattr(output, "to_float"):
        return output.to_float(torch.float32)
    return output


def _int16_qat_optimizer_step(
    model: nn.Module,
    trainable: list[nn.Parameter],
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    grad_clip_norm: Optional[float],
    batch_size: Optional[int] = None,
    execution_mode: ExecutionMode = ExecutionMode.INT16_FIXED_QAT_SIM,
    activation_recompute: bool = False,
    log_timing: bool = False,
    check_finite: bool = True,
) -> torch.Tensor:
    if batch_size is not None and batch_size > 0 and inputs.size(0) > batch_size:
        inputs = inputs[:batch_size]
        labels = labels[:batch_size]
    inputs = inputs.to(device, non_blocking=torch.cuda.is_available())
    labels = labels.to(device, non_blocking=torch.cuda.is_available())
    optimizer.zero_grad(set_to_none=True)
    step_t0 = time.perf_counter()

    recompute_ctx = enable_recompute() if activation_recompute else no_recompute()
    with recompute_ctx:
        with quant_execution_mode(execution_mode):
            logits = _to_float_logits(model(inputs))
    loss = loss_fn(logits, labels)
    if check_finite and not bool(torch.isfinite(loss.detach()).cpu().item()):
        raise RuntimeError(f"Non-finite INT16 QAT loss: {loss.item()}")
    loss.backward()
    for param in trainable:
        grad = param.grad
        if grad is None:
            continue
        torch.nan_to_num_(grad, nan=0.0, posinf=0.0, neginf=0.0)
    if grad_clip_norm is not None and grad_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(trainable, grad_clip_norm)
    optimizer.step()
    if check_finite:
        for param in trainable:
            if not bool(torch.isfinite(param).all().detach().cpu().item()):
                raise RuntimeError(
                    "INT16 QAT produced non-finite weights after optimizer.step(); "
                    "lower lr or use scope='head'."
                )
    loss_val = loss.detach()
    if log_timing:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - step_t0
        print(f"    [INT16 QAT timing] forward+backward {elapsed:.1f}s", flush=True)
    del loss, logits
    return loss_val


def run_int16_qat_steps(
    model: nn.Module,
    data_iter: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    *,
    steps: int,
    lr: float = 1e-4,
    scope: QatTrainScope = "weights",
    grad_clip_norm: Optional[float] = 1.0,
    loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    log_every: int = 1,
    batch_size: Optional[int] = None,
    execution_mode: ExecutionMode = ExecutionMode.INT16_FIXED_QAT_SIM,
    empty_cache_every: int = 0,
    activation_recompute: bool = False,
    log_timing: bool = False,
    check_finite_every: int = 1,
) -> list[float]:
    """Run ``steps`` of SGD under INT16_FIXED_QAT_SIM; return per-step losses."""

    if steps <= 0:
        return []

    trainable = select_qat_trainable_parameters(model, scope=scope)
    if not trainable:
        raise RuntimeError(f"No trainable parameters for INT16 QAT scope={scope!r}.")

    if loss_fn is None:
        loss_fn = lambda logits, labels: F.cross_entropy(logits, labels)

    model.train()
    optimizer = torch.optim.SGD(trainable, lr=lr)
    set_train_mode_freeze_bn(model, verbose=False)
    iterator: Iterator[tuple[torch.Tensor, torch.Tensor]] = iter(data_iter)

    losses: list[torch.Tensor] = []
    for _ in range(steps):
        try:
            inputs, labels = next(iterator)
        except StopIteration:
            iterator = iter(data_iter)
            inputs, labels = next(iterator)
        try:
            loss_val = _int16_qat_optimizer_step(
                model,
                trainable,
                optimizer,
                inputs,
                labels,
                device,
                loss_fn,
                grad_clip_norm,
                batch_size=batch_size,
                execution_mode=execution_mode,
                activation_recompute=activation_recompute,
                log_timing=log_timing,
                check_finite=(
                    check_finite_every > 0
                    and (len(losses) + 1) % check_finite_every == 0
                ),
            )
        except Exception:
            release_cuda_memory(model)
            raise
        losses.append(loss_val)
        step_no = len(losses)
        if empty_cache_every > 0 and step_no % empty_cache_every == 0:
            release_cuda_memory(model)
        elif empty_cache_every == 0:
            release_qat_grads(model)
        if log_every > 0 and (step_no % log_every == 0 or step_no == steps):
            loss_float = float(loss_val.detach().cpu())
            print(
                f"  [INT16 QAT] step {step_no}/{steps} ce_loss={loss_float:.4f}",
                flush=True,
            )

    return [float(loss.detach().cpu()) for loss in losses]


def run_int16_qat_epochs(
    model: nn.Module,
    train_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    *,
    epochs: int,
    lr: float = 1e-4,
    scope: QatTrainScope = "weights",
    optimizer: QatOptimizerName = "sgd",
    lr_scheduler: bool = False,
    grad_clip_norm: Optional[float] = 1.0,
    max_batches_per_epoch: Optional[int] = None,
    loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    log_every: int = 1,
    batch_size: Optional[int] = None,
    execution_mode: ExecutionMode = ExecutionMode.INT16_FIXED_QAT_SIM,
    empty_cache_every: int = 0,
    activation_recompute: bool = False,
    log_timing: bool = False,
    check_finite_every: int = 1,
    val_loader: Optional[Iterable[tuple[torch.Tensor, torch.Tensor]]] = None,
    val_fn: Optional[Callable[[nn.Module], float]] = None,
    restore_best: bool = True,
) -> list[dict[str, Union[float, str]]]:
    """Run ``epochs`` passes over ``train_loader``; return per-epoch stats.

    Each entry includes ``epoch``, ``batches``, ``loss_mean``, ``loss_last``;
    when ``val_fn`` is set, also ``val_top1`` (and ``val_restored`` on the last row).
    """

    if epochs <= 0:
        return []

    trainable = select_qat_trainable_parameters(model, scope=scope)
    if not trainable:
        raise RuntimeError(f"No trainable parameters for INT16 QAT scope={scope!r}.")

    if loss_fn is None:
        loss_fn = lambda logits, labels: F.cross_entropy(logits, labels)

    model.train()
    optim = _make_qat_optimizer(trainable, name=optimizer, lr=lr)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
        if lr_scheduler and epochs > 0
        else None
    )
    history: list[dict[str, Union[float, str]]] = []

    best_val = -1.0
    best_state: list[torch.Tensor] | None = None
    if val_loader is not None and val_fn is not None and restore_best:
        model.eval()
        best_val = val_fn(model)
        best_state = _snapshot_trainable_state(trainable)
        print(f"  [INT16 QAT] val before QAT: {best_val * 100:.2f}%", flush=True)
        model.train()

    for epoch in range(epochs):
        set_train_mode_freeze_bn(model, verbose=False)
        batch_losses: list[torch.Tensor] = []
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            if max_batches_per_epoch is not None and batch_idx >= max_batches_per_epoch:
                break
            try:
                loss_val = _int16_qat_optimizer_step(
                    model,
                    trainable,
                    optim,
                    inputs,
                    labels,
                    device,
                    loss_fn,
                    grad_clip_norm,
                    batch_size=batch_size,
                    execution_mode=execution_mode,
                    activation_recompute=activation_recompute,
                    log_timing=log_timing,
                    check_finite=(
                        check_finite_every > 0
                        and (len(batch_losses) + 1) % check_finite_every == 0
                    ),
                )
            except Exception:
                release_cuda_memory(model)
                raise
            batch_losses.append(loss_val)
            batch_no = len(batch_losses)
            if empty_cache_every > 0 and batch_no % empty_cache_every == 0:
                release_cuda_memory(model)
            elif empty_cache_every == 0:
                release_qat_grads(model)
            if log_every > 0 and (batch_no % log_every == 0):
                loss_float = float(loss_val.detach().cpu())
                print(
                    f"  [INT16 QAT] epoch {epoch + 1}/{epochs} "
                    f"batch {batch_no} ce_loss={loss_float:.4f}",
                    flush=True,
                )
        if not batch_losses:
            raise RuntimeError("train_loader yielded no batches for INT16 QAT.")
        loss_values = [float(loss.detach().cpu()) for loss in batch_losses]
        row: dict[str, Union[float, str]] = {
            "epoch": float(epoch + 1),
            "batches": float(len(loss_values)),
            "loss_mean": sum(loss_values) / len(loss_values),
            "loss_last": loss_values[-1],
        }
        if scheduler is not None:
            row["lr"] = scheduler.get_last_lr()[0]
            scheduler.step()

        if val_loader is not None and val_fn is not None:
            model.eval()
            val_acc = val_fn(model)
            row["val_top1"] = val_acc
            print(
                f"  [INT16 QAT] epoch {epoch + 1}/{epochs} "
                f"val Top-1={val_acc * 100:.2f}%",
                flush=True,
            )
            if restore_best and val_acc >= best_val:
                best_val = val_acc
                best_state = _snapshot_trainable_state(trainable)
            model.train()

        history.append(row)

    if restore_best and best_state is not None:
        _restore_trainable_state(trainable, best_state)
        if history:
            history[-1]["val_restored"] = best_val
        print(
            f"  [INT16 QAT] restored best val checkpoint: {best_val * 100:.2f}%",
            flush=True,
        )

    return history

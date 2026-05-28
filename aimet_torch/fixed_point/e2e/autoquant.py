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
"""v2 AutoQuant integration with combined-PTQ fallback for fixed-point MobileNet."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from aimet_common.defs import QuantScheme
from aimet_torch import utils
from aimet_torch.utils import change_tensor_device_placement, in_eval_mode

AutoQuantSource = Literal["autoquant", "combined_fallback"]


@dataclass
class AutoQuantPtqResult:
    """Output of v2 AutoQuant or the manual combined PTQ fallback."""

    model: nn.Module
    eval_score: float
    encoding_path: str
    results_dir: str
    source: AutoQuantSource = "autoquant"


def _patch_autoquant_forward_pass(auto_quant: Any, data_loader: DataLoader) -> None:
    """Use only the image tensor when the loader yields ``(images, labels)``."""

    def forward_pass_callback(model, _=None):
        device = utils.get_device(model)
        with in_eval_mode(model), torch.no_grad():
            for input_data in data_loader:
                if isinstance(input_data, (tuple, list)):
                    input_data = input_data[0]
                input_data = change_tensor_device_placement(input_data, device)
                model(input_data)

    auto_quant.forward_pass_callback = forward_pass_callback


def make_autoquant_loader(
    input_size: int,
    *,
    num_samples: int = 16,
    batch_size: int = 2,
    n_class: int = 10,
    seed: int = 42,
) -> DataLoader:
    """Calibration / AdaRound loader with ``(images, labels)`` batches."""
    torch.manual_seed(seed)
    images = torch.randn(num_samples, 3, input_size, input_size)
    labels = torch.zeros(num_samples, dtype=torch.long)
    return DataLoader(
        TensorDataset(images, labels),
        batch_size=batch_size,
        shuffle=False,
    )


def run_v2_autoquant_ptq(
    model: nn.Module,
    dummy_input: torch.Tensor,
    *,
    data_loader: DataLoader,
    eval_callback: Callable[..., float],
    results_dir: Path,
    allowed_accuracy_drop: float = 0.05,
    param_bw: int = 8,
    output_bw: int = 8,
    quant_scheme: QuantScheme = QuantScheme.min_max,
    model_prepare_required: bool = False,
    adaround_num_batches: int = 2,
    adaround_iterations: int = 40,
) -> AutoQuantPtqResult:
    """Run AIMET v2 AutoQuant (BN fold + CLE + AdaRound).

    Raises when AutoQuant fails (e.g. ``torch.export`` on prepared GraphModule).
    """
    from aimet_torch.v2.adaround import AdaroundParameters
    from aimet_torch.v2.auto_quant import AutoQuant

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    auto_quant = AutoQuant(
        model=model,
        dummy_input=dummy_input,
        data_loader=data_loader,
        eval_callback=eval_callback,
        param_bw=param_bw,
        output_bw=output_bw,
        quant_scheme=quant_scheme,
        results_dir=str(results_dir),
        strict_validation=False,
        model_prepare_required=model_prepare_required,
    )
    _patch_autoquant_forward_pass(auto_quant, data_loader)
    auto_quant.set_adaround_params(
        AdaroundParameters(
            data_loader,
            num_batches=adaround_num_batches,
            default_num_iterations=adaround_iterations,
        )
    )

    optimized, score, encoding_path = auto_quant.optimize(
        allowed_accuracy_drop=allowed_accuracy_drop
    )
    return AutoQuantPtqResult(
        model=optimized,
        eval_score=float(score),
        encoding_path=str(encoding_path),
        results_dir=str(results_dir),
        source="autoquant",
    )


def run_combined_ptq_pipeline(
    model: nn.Module,
    dummy_input: torch.Tensor,
    *,
    data_loader: DataLoader,
    eval_callback: Callable[..., float],
    input_size: Optional[int] = None,
    results_dir: Optional[Path] = None,
    apply_cle: bool = True,
    apply_bias_correction: bool = True,
    adaround_num_batches: int = 2,
    adaround_iterations: int = 80,
) -> AutoQuantPtqResult:
    """Manual PTQ chain (CLE → fold → BC → AdaRound) when v2 AutoQuant is unavailable."""
    from aimet_torch.fixed_point.e2e.mobilenet_v2 import build_calibrated_sim

    if input_size is None:
        input_size = int(dummy_input.shape[-1])

    export_dir = results_dir or Path(tempfile.mkdtemp(prefix="aimet_combined_ptq_"))
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    bundle = build_calibrated_sim(
        model,
        dummy_input,
        input_size=input_size,
        apply_cle=apply_cle,
        apply_bias_correction=apply_bias_correction,
        adaround_loader=data_loader,
        adaround_num_batches=adaround_num_batches,
        adaround_iterations=adaround_iterations,
        adaround_export_dir=export_dir / "adaround",
    )
    score = float(eval_callback(bundle.model))
    return AutoQuantPtqResult(
        model=bundle.model,
        eval_score=score,
        encoding_path=str(export_dir / "adaround"),
        results_dir=str(export_dir),
        source="combined_fallback",
    )


def try_run_v2_autoquant_ptq(
    model: nn.Module,
    dummy_input: torch.Tensor,
    *,
    data_loader: DataLoader,
    eval_callback: Callable[..., float],
    results_dir: Path,
    use_combined_fallback: bool = True,
    try_unprepared_autoquant: bool = True,
    **kwargs: Any,
) -> Optional[AutoQuantPtqResult]:
    """Run v2 AutoQuant, or on failure the combined PTQ pipeline if enabled."""
    try:
        return run_v2_autoquant_ptq(
            model,
            dummy_input,
            data_loader=data_loader,
            eval_callback=eval_callback,
            results_dir=results_dir,
            **kwargs,
        )
    except Exception:  # pragma: no cover - export / GraphModule dependent
        pass

    if try_unprepared_autoquant and kwargs.get("model_prepare_required") is False:
        try:
            from aimet_torch.examples.mobilenet import MockMobileNetV2

            sz = int(dummy_input.shape[-1])
            raw = MockMobileNetV2(n_class=10, input_size=sz).eval()
            dummy_raw = torch.randn(1, 3, sz, sz)
            subdir = Path(results_dir) / "unprepared_autoquant"
            return run_v2_autoquant_ptq(
                raw,
                dummy_raw,
                data_loader=data_loader,
                eval_callback=eval_callback,
                results_dir=subdir,
                model_prepare_required=True,
                **{k: v for k, v in kwargs.items() if k != "model_prepare_required"},
            )
        except Exception:  # pragma: no cover
            pass

    if not use_combined_fallback:
        return None
    try:
        return run_combined_ptq_pipeline(
            model,
            dummy_input,
            data_loader=data_loader,
            eval_callback=eval_callback,
            input_size=int(dummy_input.shape[-1]),
            results_dir=results_dir,
            adaround_iterations=int(kwargs.get("adaround_iterations", 80)),
            adaround_num_batches=int(kwargs.get("adaround_num_batches", 2)),
        )
    except Exception:  # pragma: no cover
        return None

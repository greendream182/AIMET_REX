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
"""Model-family-agnostic v2 INT16 fixed-point sim construction.

This is the shared PTQ skeleton consumed by every model wrapper under
``aimet_torch.fixed_point.e2e`` (MobileNet today, attention / YOLO / audio
backbones once they land).

The pipeline (all steps optional and driven by keyword args):

    optional CLE  ->  optional BN fold  ->  optional empirical bias correction
    ->  optional v2 AdaRound  ->  v2 QuantizationSimModel
    ->  ensure_output_quantizers_for_int16_eval  ->  compute_encodings

Input data generation (calibration / BC / AdaRound) is deliberately *not*
baked in here — see :mod:`aimet_torch.fixed_point.e2e.inputs` for image /
1D audio / mel-spectrogram samplers. Callers compose: pick a sampler from
``inputs.py``, pass it (or a list of materialized batches) to this builder.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn

import aimet_torch.fixed_point.kernels  # noqa: F401  (registers fixed-point ops)
from aimet_torch.fixed_point import (
    ExecutionMode,
    ensure_output_quantizers_for_int16_eval,
    set_quant_execution_mode,
)
from aimet_torch.fixed_point.e2e.inputs import InputSampler


@dataclass
class CalibratedSimBundle:
    """Output of :func:`build_calibrated_v2_sim`.

    Model-family-specific wrappers (e.g. ``MobileNetV2SimBundle``) may add
    domain fields like ``input_size`` / ``variant`` on top of this.
    """

    sim: Any
    model: nn.Module
    dummy_input: torch.Tensor
    n_oq_patched: int


def apply_v2_empirical_bias_correction(
    model: nn.Module,
    *,
    data_loader: Any,
    num_samples: int = 16,
    weight_bw: int = 8,
    act_bw: int = 8,
    quant_scheme: Optional[Any] = None,
) -> None:
    """Run AIMET empirical bias correction on a prepared, (typically BN-folded) model.

    ``data_loader`` is forwarded to ``correct_bias`` unchanged — it must yield
    ``(input, label)`` tuples in whatever shape the model expects.
    """

    from aimet_common.defs import QuantScheme
    from aimet_torch._base.quantsim import QuantParams
    from aimet_torch.bias_correction import correct_bias

    if quant_scheme is None:
        quant_scheme = QuantScheme.min_max
    quant_params = QuantParams(
        weight_bw=weight_bw,
        act_bw=act_bw,
        quant_scheme=quant_scheme,
    )
    correct_bias(
        model,
        quant_params,
        num_quant_samples=num_samples,
        data_loader=data_loader,
        num_bias_correct_samples=num_samples,
        perform_only_empirical_bias_corr=True,
    )


def _default_bn_fold_shape(dummy_input: torch.Tensor) -> Tuple[int, ...]:
    """``(1, *dummy_input.shape[1:])`` — single-sample fold shape."""

    return (1, *tuple(int(d) for d in dummy_input.shape[1:]))


def build_calibrated_v2_sim(
    model: nn.Module,
    dummy_input: torch.Tensor,
    *,
    calibration_batches: Optional[Iterable[torch.Tensor]] = None,
    calibration_sampler: Optional[InputSampler] = None,
    calibration_iters: int = 4,
    default_param_bw: int = 8,
    default_output_bw: int = 8,
    int16_eval_bw: int = 8,
    int16_eval_symmetric: bool = True,
    output_quantizer_overrides: Optional[Dict[str, Tuple[int, bool]]] = None,
    quant_scheme: Optional[Any] = None,
    apply_cle: bool = False,
    apply_bn_fold: bool = True,
    bn_fold_input_shape: Optional[Tuple[int, ...]] = None,
    bias_correction_data: Optional[Any] = None,
    bias_correction_num_samples: int = 16,
    adaround_loader: Optional[Iterable] = None,
    adaround_num_batches: int = 2,
    adaround_iterations: int = 80,
    adaround_export_dir: Optional[Path] = None,
    adaround_filename_prefix: str = "sim",
) -> CalibratedSimBundle:
    """Run optional PTQ steps and return a calibrated v2 QuantizationSim.

    Calibration data is taken from ``calibration_batches`` when provided,
    otherwise from ``calibration_sampler`` (called ``calibration_iters``
    times). At least one of the two must be set.

    All PTQ refinements are optional: pass ``apply_cle=True`` for CLE,
    ``apply_bn_fold=True`` (default) for BN-fold, ``bias_correction_data`` to
    run empirical bias correction, and ``adaround_loader`` to run v2 AdaRound.
    """

    from aimet_common.defs import QuantScheme
    from aimet_torch.batch_norm_fold import fold_all_batch_norms
    from aimet_torch.cross_layer_equalization import equalize_model
    from aimet_torch.v2.quantsim import QuantizationSimModel

    if quant_scheme is None:
        quant_scheme = QuantScheme.min_max
    if calibration_batches is None and calibration_sampler is None:
        raise ValueError(
            "Must provide either calibration_batches or calibration_sampler."
        )

    set_quant_execution_mode(ExecutionMode.FP32_QDQ)

    working_model = model
    if apply_cle:
        equalize_model(working_model, dummy_input=dummy_input)

    if apply_bn_fold:
        fold_shape = bn_fold_input_shape or _default_bn_fold_shape(dummy_input)
        fold_all_batch_norms(
            working_model,
            input_shapes=fold_shape,
            dummy_input=dummy_input,
        )

    if bias_correction_data is not None:
        apply_v2_empirical_bias_correction(
            working_model,
            data_loader=bias_correction_data,
            num_samples=bias_correction_num_samples,
            quant_scheme=QuantScheme.min_max
            if quant_scheme is QuantScheme.min_max
            else quant_scheme,
        )

    if adaround_loader is not None:
        from aimet_torch.v2.adaround import Adaround, AdaroundParameters

        export_dir = adaround_export_dir or Path("/tmp/aimet_adaround_sim")
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        params = AdaroundParameters(
            data_loader=adaround_loader,
            num_batches=adaround_num_batches,
            default_num_iterations=adaround_iterations,
        )
        working_model = Adaround.apply_adaround(
            working_model,
            dummy_input=dummy_input,
            params=params,
            path=str(export_dir),
            filename_prefix=adaround_filename_prefix,
            default_param_bw=default_param_bw,
            default_quant_scheme=quant_scheme,
        )

    sim = QuantizationSimModel(
        working_model,
        dummy_input=dummy_input,
        default_output_bw=default_output_bw,
        default_param_bw=default_param_bw,
        quant_scheme=quant_scheme,
    )
    n_patched = len(
        ensure_output_quantizers_for_int16_eval(
            sim, bitwidth=int16_eval_bw, symmetric=int16_eval_symmetric
        )
    )
    if output_quantizer_overrides:
        from aimet_torch.v2.quantization.affine import Quantize

        modules = dict(sim.model.named_modules())
        for name, (bitwidth, symmetric) in output_quantizer_overrides.items():
            module = modules.get(name)
            if module is None:
                raise KeyError(f"Output quantizer override target not found: {name}")
            oq_list = getattr(module, "output_quantizers", None)
            if oq_list is None or len(oq_list) == 0:
                raise ValueError(f"Module {name} has no output_quantizers slot.")
            oq_list[0] = Quantize((), int(bitwidth), symmetric=bool(symmetric))

    def _calibrate(m, _):
        with torch.no_grad():
            if calibration_batches is not None:
                for batch in calibration_batches:
                    m(batch)
            else:
                assert calibration_sampler is not None  # narrowed above
                for _i in range(calibration_iters):
                    m(calibration_sampler())

    sim.compute_encodings(_calibrate, None)

    from aimet_torch.fixed_point.export.sidecar_loader import (
        maybe_attach_int16_sidecar_from_env,
    )

    maybe_attach_int16_sidecar_from_env(sim.model)

    return CalibratedSimBundle(
        sim=sim,
        model=working_model,
        dummy_input=dummy_input,
        n_oq_patched=n_patched,
    )


__all__ = [
    "CalibratedSimBundle",
    "apply_v2_empirical_bias_correction",
    "build_calibrated_v2_sim",
]

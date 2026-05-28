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
"""End-to-end fixed-point validation helpers (shared by tests and reports).

Layering:

* :mod:`aimet_torch.fixed_point.e2e.inputs` — synthetic input generators
  (image / 1D audio / mel-spec) used by every model family.
* :mod:`aimet_torch.fixed_point.e2e.sim_builder` — model-agnostic v2 INT16
  PTQ skeleton (CLE / BN fold / BC / AdaRound / QuantizationSim /
  ``ensure_output_quantizers_for_int16_eval`` / ``compute_encodings``).
* :mod:`aimet_torch.fixed_point.e2e.mobilenet_v2` — MobileNet-flavored thin
  wrappers and classification-specific evaluators (cosine, INT16 QAT loop).
* :mod:`aimet_torch.fixed_point.e2e.autoquant` — v2 AutoQuant + combined PTQ
  fallback (depends on the MobileNet wrappers today; will be re-pointed at
  ``sim_builder`` once non-classification models land).
"""

from aimet_torch.fixed_point.e2e.autoquant import (
    AutoQuantPtqResult,
    AutoQuantSource,
    make_autoquant_loader,
    run_combined_ptq_pipeline,
    run_v2_autoquant_ptq,
    try_run_v2_autoquant_ptq,
)
from aimet_torch.fixed_point.e2e.inputs import (
    InputSampler,
    make_audio_1d_calibration_batches,
    make_audio_1d_sampler,
    make_image_adaround_loader,
    make_image_bc_dataloader,
    make_image_calibration_batches,
    make_image_sampler,
    make_melspec_calibration_batches,
    make_melspec_sampler,
)
from aimet_torch.fixed_point.e2e.mobilenet_v2 import (
    CALIB_BATCH,
    CALIB_ITERS,
    INPUT_SIZE,
    MOBILENET_ADAROUND_ITERATIONS_DEFAULT,
    MOBILENET_ADAROUND_ITERATIONS_LONG,
    MobileNetV2SimBundle,
    MobilenetVariant,
    apply_empirical_bias_correction,
    build_calibrated_sim,
    build_prepared_mobilenet_v2,
    int16_vs_fp32_cosine,
    make_adaround_loader,
    teacher_logits,
    train_int16_qat,
)
from aimet_torch.fixed_point.e2e.sim_builder import (
    CalibratedSimBundle,
    apply_v2_empirical_bias_correction,
    build_calibrated_v2_sim,
)

__all__ = [
    "AutoQuantPtqResult",
    "AutoQuantSource",
    "CALIB_BATCH",
    "CALIB_ITERS",
    "CalibratedSimBundle",
    "INPUT_SIZE",
    "InputSampler",
    "MOBILENET_ADAROUND_ITERATIONS_DEFAULT",
    "MOBILENET_ADAROUND_ITERATIONS_LONG",
    "MobileNetV2SimBundle",
    "MobilenetVariant",
    "apply_empirical_bias_correction",
    "apply_v2_empirical_bias_correction",
    "build_calibrated_sim",
    "build_calibrated_v2_sim",
    "build_prepared_mobilenet_v2",
    "int16_vs_fp32_cosine",
    "make_adaround_loader",
    "make_audio_1d_calibration_batches",
    "make_audio_1d_sampler",
    "make_autoquant_loader",
    "make_image_adaround_loader",
    "make_image_bc_dataloader",
    "make_image_calibration_batches",
    "make_image_sampler",
    "make_melspec_calibration_batches",
    "make_melspec_sampler",
    "run_combined_ptq_pipeline",
    "run_v2_autoquant_ptq",
    "teacher_logits",
    "train_int16_qat",
    "try_run_v2_autoquant_ptq",
]

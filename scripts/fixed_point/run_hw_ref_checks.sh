#!/usr/bin/env bash
# Strict hardware/abc parity checks (LUT + requantize). Does not run full e2e.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export AIMET_RX_HW_REF=1
export AIMET_RX_REQUANTIZE_INT32_SAT=1
python3 -m pytest \
  tests/fixed_point/kernels/test_lut_abc_reference.py \
  tests/fixed_point/kernels/test_conv_linear.py::test_linear_int32_accumulator_sat_differs_under_hw_ref \
  tests/fixed_point/kernels/test_hw_ref_eltwise_pool.py \
  tests/fixed_point/kernels/test_clz_sqrt_golden.py \
  tests/fixed_point/kernels/test_lut_sin_cos.py \
  tests/fixed_point/kernels/test_clz_reciprocal_golden.py \
  tests/fixed_point/kernels/test_clz_power2_golden.py \
  tests/fixed_point/test_requantize.py \
  tests/fixed_point/offline/test_clz_gen.py::test_generate_clz_reciprocal_output_scale_sane_with_fit_domain \
  tests/fixed_point/export/test_sidecar_clz_golden.py \
  tests/fixed_point/kernels/test_clz_signed_domain.py \
  tests/fixed_point/test_clz_adapter_soft_fail.py \
  -q --tb=short "$@"

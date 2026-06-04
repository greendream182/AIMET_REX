#!/usr/bin/env bash
# 两组 ablation + baseline 行（baseline 来自 int16_eval_diagnosis.log）
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="/home/llq/workspace/aimet_rx-main:/home/llq/workspace/quant-gru-pytorch/pytorch"
DATA="/home/llq/workspace/data/speech_commands"
OUT="output/int16_ablation_compare.log"
: > "$OUT"

run_case() {
  local tag="$1"
  shift
  echo "======== $tag ========" | tee -a "$OUT"
  python3 quick_start_int16_metric.py \
    --data-root "$DATA" \
    --fp-epochs 0 \
    --max-calib-batches 100 \
    --modes fp32_qdq int16_fixed_eval \
    "$@" 2>&1 | tee -a "$OUT" | grep -E '^(fp32_qdq|int16_fixed_eval|float_native|Δ\(INT16|Bitwidth|模式):'
  echo "" | tee -a "$OUT"
}

echo "baseline 见 int16_eval_diagnosis.log: fp32_qdq=33.85% int16=20.41% float_native=85.53%" | tee -a "$OUT"

run_case "A_native_trans" \
  --native-trans \
  --bitwidth-config config/mrnn_acceptance_mixed_precision.json

run_case "B_pc1_sign_sqrt_only_16bit" \
  --bitwidth-config config/pc1_sign_sqrt_only_16bit.json

run_case "C_pc1_sign_sqrt_mul_16bit" \
  --bitwidth-config config/pc1_sign_sqrt_mul_16bit.json

echo "======== D_acceptance_fp32_qat_1ep ========" | tee -a "$OUT"
python3 quick_start_int16_metric.py \
  --data-root "$DATA" \
  --fp-epochs 0 \
  --max-calib-batches 100 \
  --bitwidth-config config/mrnn_acceptance_mixed_precision.json \
  --qat-epochs 1 \
  --qat-mode fp32_qdq \
  --modes fp32_qdq int16_fixed_eval \
  2>&1 | tee -a "$OUT" | grep -E '^(fp32_qdq|int16_fixed_eval|float_native|Δ\(|post-QAT|baseline|QAT 前|QAT 后|---)'
echo "" | tee -a "$OUT"

echo "DONE" | tee -a "$OUT"

# Fixed-point quality & CI

## Local commands

```bash
cd /path/to/aimet_rx-main
export PYTHONPATH="$(pwd)"

# Fast: mock MobileNet metrics + baseline check (no pytest, quieter logs, no v2 AutoQuant row)
bash scripts/fixed_point/run_report_fast.sh
# equivalent:
python3 scripts/fixed_point/run_baseline.py

# Full report + pytest summary + baseline
bash scripts/fixed_point/run_report.sh
python3 scripts/fixed_point/run_baseline.py --full

# Pytest fast (~30s, skip PTQ/QAT mobilenet + slow + ImageNet)
bash scripts/fixed_point/run_fixed_point_fast.sh

# Pytest full (~5min, includes PTQ/QAT; still skips ImageNet)
bash scripts/fixed_point/run_fixed_point_full.sh

# Strict LUT / CLZ / requantize (HW_REF)
bash scripts/fixed_point/run_hw_ref_checks.sh

# Toy ImageNet layout + PTQ pipeline (no download)
bash scripts/fixed_point/run_imagenet_toy_smoke.sh

# Pytest only (PR-equivalent, skip slow + ImageNet)
python3 -m pytest tests/fixed_point -m "not slow and not imagenet" -q

# Coverage gate (same as CI, requires `pip install coverage`)
python3 -m coverage run -m pytest tests/fixed_point -m "not slow and not imagenet" -q
python3 -m coverage report

# ImageNet-scale e2e (224×224 MobileNet) — light verification needs few images, no labels

# Recommended: synthetic tensors + cosine-only (no ImageNet download, no labels)
bash scripts/fixed_point/run_imagenet_validation.sh \
  --source synthetic --cosine-only --calib-max-samples 64 --cosine-batches 2

# Any folder of photos (recursive); labels ignored with --cosine-only
bash scripts/fixed_point/run_imagenet_validation.sh \
  --source unlabeled --image-dir /path/to/images --cosine-only

# Full check (needs labels): local ImageFolder or HF
export AIMET_RX_IMAGENET_VAL=/path/to/imagenet/val
bash scripts/fixed_point/run_imagenet_validation.sh --source local
# HF: https://huggingface.co/datasets/ILSVRC/imagenet-1k + huggingface-cli login
bash scripts/fixed_point/run_imagenet_validation.sh --source huggingface --hf-stream --calib-max-samples 128

python3 -m pytest tests/fixed_point/end_to_end/test_imagenet_mobilenet_v2.py -m imagenet -v -s
```

## compare_modes CLI (spec 12)

```bash
python3 scripts/fixed_point/compare_quant_modes.py \
  --model dual_linear \
  --report scripts/fixed_point/reports/quant_mode_report.json \
  --per-layer-csv scripts/fixed_point/reports/per_layer.csv \
  --saturation

# Calibrated mock MobileNet (slower; needs onnxscript)
python3 scripts/fixed_point/compare_quant_modes.py \
  --model mock_mobilenet --input-size 64 \
  --report /tmp/quant_mode_report.json
```

Or use Makefile: `make compare-modes`, `make baseline`, `make coverage`.

## CI (GitHub Actions)

Workflow: [`.github/workflows/fixed_point_ci.yml`](../../.github/workflows/fixed_point_ci.yml)

| Job | Trigger | Contents |
|-----|---------|----------|
| `pr-fast` | PR / push to main | pytest (fast) + **coverage ≥85%**, quality report, baseline check |
| `nightly-full` | cron / manual | Full `tests/fixed_point` (excl. `imagenet`), full report |

Failed runs upload `scripts/fixed_point/reports/` as artifacts.

## Baseline updates

Thresholds live in [`baseline.json`](baseline.json). Changing floors requires a PR labeled `baseline-update` and owner review (spec 14).

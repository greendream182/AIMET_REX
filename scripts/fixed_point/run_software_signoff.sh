#!/usr/bin/env bash
# Pure software pre-production signoff for AIMET RX INT16 fixed-point flows.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

RUN_IMAGENET="${AIMET_RX_SIGNOFF_IMAGENET:-auto}"
IMAGENET_ARGS="${AIMET_RX_SIGNOFF_IMAGENET_ARGS:---source auto --batch-size 8 --calib-batches 4 --calib-max-samples 512 --eval-batches 16 --cosine-batches 4}"
IMAGENET_VAL_DIR="$(
  python3 - <<'PY'
from aimet_torch.fixed_point.e2e.imagenet_eval import resolve_imagenet_val_dir

path = resolve_imagenet_val_dir()
print(path if path is not None else "")
PY
)"
IMAGENET_VAL_ZIP="$(
  python3 - <<'PY'
from aimet_torch.fixed_point.e2e.imagenet_eval import resolve_imagenet_val_zip

path = resolve_imagenet_val_zip()
print(path if path is not None else "")
PY
)"

echo "== AIMET RX INT16 software signoff =="
echo "repo: $ROOT"
echo

case "$RUN_IMAGENET" in
  1|true|True|required|REQUIRED)
    if [[ (-z "$IMAGENET_VAL_DIR" || ! -d "$IMAGENET_VAL_DIR") && (-z "$IMAGENET_VAL_ZIP" || ! -f "$IMAGENET_VAL_ZIP") ]]; then
      echo "ERROR: ImageNet signoff required but no ImageNet val directory or zip was found." >&2
      echo "       Set AIMET_RX_IMAGENET_VAL=/path/to/imagenet/val or AIMET_RX_IMAGENET_VAL_ZIP=/path/to/imagenet_val.zip." >&2
      exit 2
    fi
    ;;
esac

echo "== 1/4 full fixed_point regression =="
bash scripts/fixed_point/run_fixed_point_full.sh "$@"
echo

echo "== 2/4 strict HW_REF / abc software checks =="
bash scripts/fixed_point/run_hw_ref_checks.sh "$@"
echo

echo "== 3/4 sidecar JSON reload / fail-fast checks =="
python3 -m pytest \
  tests/fixed_point/export/test_sidecar_loader.py \
  tests/fixed_point/export/test_sidecar_export.py \
  tests/fixed_point/export/test_sidecar_clz_golden.py \
  -q --tb=short "$@"
echo

echo "== 4/4 ImageNet validation =="
case "$RUN_IMAGENET" in
  0|false|False|skip|SKIP)
    echo "SKIP: AIMET_RX_SIGNOFF_IMAGENET=$RUN_IMAGENET"
    ;;
  auto)
    if [[ -n "$IMAGENET_VAL_DIR" && -d "$IMAGENET_VAL_DIR" ]]; then
      # shellcheck disable=SC2086
      bash scripts/fixed_point/run_imagenet_validation.sh $IMAGENET_ARGS --val-dir "$IMAGENET_VAL_DIR"
    elif [[ -n "$IMAGENET_VAL_ZIP" && -f "$IMAGENET_VAL_ZIP" ]]; then
      # shellcheck disable=SC2086
      bash scripts/fixed_point/run_imagenet_validation.sh $IMAGENET_ARGS --zip-path "$IMAGENET_VAL_ZIP"
    else
      echo "SKIP: no ImageNet val directory or zip found."
      echo "      Re-run with AIMET_RX_IMAGENET_VAL=/path/to/imagenet/val or AIMET_RX_IMAGENET_VAL_ZIP=/path/to/imagenet_val.zip."
    fi
    ;;
  1|true|True|required|REQUIRED)
    if [[ -n "$IMAGENET_VAL_DIR" && -d "$IMAGENET_VAL_DIR" ]]; then
      # shellcheck disable=SC2086
      bash scripts/fixed_point/run_imagenet_validation.sh $IMAGENET_ARGS --val-dir "$IMAGENET_VAL_DIR"
    else
      # shellcheck disable=SC2086
      bash scripts/fixed_point/run_imagenet_validation.sh $IMAGENET_ARGS --zip-path "$IMAGENET_VAL_ZIP"
    fi
    ;;
  *)
    echo "ERROR: AIMET_RX_SIGNOFF_IMAGENET must be auto, required/true/1, or skip/false/0." >&2
    exit 2
    ;;
esac

echo
echo "AIMET RX INT16 software signoff complete."

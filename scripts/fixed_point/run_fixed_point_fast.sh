#!/usr/bin/env bash
# Fast fixed_point regression (~30s): unit + kernel + sidecar; excludes PTQ/QAT mobilenet.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
python3 -m pytest tests/fixed_point/ -q \
  --ignore=tests/fixed_point/end_to_end/test_mobilenet_v2_ptq_qat.py \
  -m "not slow and not imagenet" \
  "$@"

#!/usr/bin/env bash
# Fast report: skip pytest and full MobileNet (mock PTQ/QAT only).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${ROOT}"
exec python3 "${ROOT}/scripts/fixed_point/run_quality_report.py" \
  --fast \
  --check-baseline \
  "$@"

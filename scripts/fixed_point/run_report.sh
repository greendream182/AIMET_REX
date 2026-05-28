#!/usr/bin/env bash
# Full quality report: pytest (-m 'not slow') + mock/full MobileNet PTQ/QAT.
# For slow AdaRound/full MobileNet pytest too: add --pytest-include-slow to the python cmd.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${ROOT}"
exec python3 "${ROOT}/scripts/fixed_point/run_quality_report.py" \
  --check-baseline \
  "$@"

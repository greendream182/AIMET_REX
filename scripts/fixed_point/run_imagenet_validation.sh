#!/usr/bin/env bash
# ImageNet val validation for MobileNet V2 fixed-point modes.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-$ROOT}"
exec python3 "$ROOT/scripts/fixed_point/run_imagenet_validation.py" "$@"

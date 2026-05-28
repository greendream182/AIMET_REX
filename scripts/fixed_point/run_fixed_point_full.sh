#!/usr/bin/env bash
# Full fixed_point regression (~5min): includes mobilenet PTQ/QAT; excludes ImageNet unless -m imagenet passed.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
python3 -m pytest tests/fixed_point/ -q -m "not imagenet" "$@"

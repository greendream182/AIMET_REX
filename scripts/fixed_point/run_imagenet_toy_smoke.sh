#!/usr/bin/env bash
# Toy ImageFolder PTQ + INT16 cosine (no real ImageNet download).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
python3 -m pytest \
  tests/fixed_point/end_to_end/test_imagenet_mobilenet_v2.py::test_imagenet_val_layout_smoke \
  tests/fixed_point/end_to_end/test_imagenet_mobilenet_v2.py::test_imagenet_val_loader_requires_class_subdirs \
  -q --tb=short "$@"

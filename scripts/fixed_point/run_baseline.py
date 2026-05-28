#!/usr/bin/env python3
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
"""Run fast quality report and verify metrics against ``baseline.json`` (spec 14)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full report (includes slow MobileNet PTQ/QAT), not --fast.",
    )
    parser.add_argument(
        "--no-check-baseline",
        action="store_true",
        help="Generate report only; skip check_thresholds.",
    )
    args = parser.parse_args(argv)

    cmd = [
        sys.executable,
        str(root / "scripts" / "fixed_point" / "run_quality_report.py"),
    ]
    if not args.full:
        cmd.append("--fast")
    if args.no_check_baseline:
        cmd.append("--no-check-baseline")
    else:
        cmd.append("--check-baseline")

    env = dict(**__import__("os").environ)
    env.setdefault("PYTHONPATH", str(root))

    completed = subprocess.run(cmd, cwd=root, env=env, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())

# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
"""CLI smoke for ``scripts/fixed_point/compare_quant_modes.py``."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
CLI = ROOT / "scripts" / "fixed_point" / "compare_quant_modes.py"


def test_compare_quant_modes_cli_dual_linear(tmp_path):
    report = tmp_path / "report.json"
    env = {"PYTHONPATH": str(ROOT)}
    proc = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--model",
            "dual_linear",
            "--modes",
            "fp32_qdq",
            "int16_fixed_eval",
            "--report",
            str(report),
        ],
        cwd=ROOT,
        env={**dict(__import__("os").environ), **env},
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(report.read_text(encoding="utf-8"))
    assert "fp32_qdq_vs_int16_fixed_eval" in data["pairwise"]
    assert data["pairwise"]["fp32_qdq_vs_int16_fixed_eval"]["cosine_similarity"] <= 1.0


def test_compare_quant_modes_cli_writes_per_layer_csv(tmp_path):
    report = tmp_path / "report.json"
    csv_path = tmp_path / "per_layer.csv"
    env = {"PYTHONPATH": str(ROOT)}
    proc = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--model",
            "dual_linear",
            "--modes",
            "fp32_qdq",
            "int16_fixed_eval",
            "--report",
            str(report),
            "--per-layer-csv",
            str(csv_path),
        ],
        cwd=ROOT,
        env={**dict(__import__("os").environ), **env},
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert csv_path.is_file()
    assert "layer" in csv_path.read_text(encoding="utf-8")

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
"""CLI for ``compare_modes`` (spec 12 / design v2 §10).

Example::

    export PYTHONPATH=/path/to/aimet_rx
    python3 scripts/fixed_point/compare_quant_modes.py \\
        --model mock_mobilenet \\
        --report output/quant_mode_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn

from aimet_torch.fixed_point import ExecutionMode, set_quant_execution_mode
from aimet_torch.fixed_point.metrics import (
    DEFAULT_COMPARE_MODES,
    compare_modes,
    write_per_layer_csv,
)
from aimet_torch.fixed_point.tensor import Int16QuantizedTensor
from aimet_torch.fixed_point import get_quant_execution_mode


class _DualModeLinearish(nn.Module):
    """Smoke model: float under QDQ modes, INT16 carrier under int16_fixed_eval."""

    def forward(self, x: torch.Tensor):
        y = x * 1.1 + 0.01
        if get_quant_execution_mode() is ExecutionMode.INT16_FIXED_EVAL:
            return Int16QuantizedTensor.from_float(
                y,
                scale=torch.tensor(0.05, device=x.device, dtype=torch.float32),
                zero_point=torch.zeros((), dtype=torch.int32, device=x.device),
            )
        return y


def _parse_modes(names: Optional[List[str]]) -> List[ExecutionMode]:
    if not names:
        return list(DEFAULT_COMPARE_MODES)
    out: List[ExecutionMode] = []
    for name in names:
        try:
            out.append(ExecutionMode(name))
        except ValueError as exc:
            valid = ", ".join(m.value for m in ExecutionMode)
            raise SystemExit(f"Unknown mode {name!r}. Valid: {valid}") from exc
    return out


def _build_mock_mobilenet(input_size: int) -> Tuple[nn.Module, torch.Tensor]:
    import aimet_torch.fixed_point.kernels  # noqa: F401
    from aimet_torch.fixed_point.e2e.mobilenet_v2 import (
        build_calibrated_sim,
        build_prepared_mobilenet_v2,
    )

    model, dummy = build_prepared_mobilenet_v2(input_size=input_size, variant="mock")
    bundle = build_calibrated_sim(model, dummy, input_size=input_size, variant="mock")
    x = torch.randn(2, 3, input_size, input_size)
    return bundle.sim.model, x


def _build_model(name: str, input_size: int) -> Tuple[nn.Module, torch.Tensor]:
    if name == "dual_linear":
        return _DualModeLinearish(), torch.randn(2, 8)
    if name == "mock_mobilenet":
        return _build_mock_mobilenet(input_size)
    raise ValueError(f"Unknown model {name!r}")


def _print_summary(report: dict[str, Any]) -> None:
    print("Modes:", " → ".join(report.get("modes", [])))
    for pair, metrics in report.get("pairwise", {}).items():
        cos = metrics.get("cosine_similarity")
        lsb = metrics.get("max_error_lsb", metrics.get("max_error_lsb_float"))
        parts = [f"  {pair}:"]
        if cos is not None:
            parts.append(f" cosine={cos:.6f}")
        if lsb is not None:
            parts.append(f" max_lsb={lsb:.4f}")
        print("".join(parts))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=("dual_linear", "mock_mobilenet"),
        default="dual_linear",
        help="dual_linear: instant smoke; mock_mobilenet: calibrated QuantSim (slower).",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=None,
        help=f"Execution modes (default: {' '.join(m.value for m in DEFAULT_COMPARE_MODES)}).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="Output JSON path (parent dirs created).",
    )
    parser.add_argument("--input-size", type=int, default=64, help="For mock_mobilenet.")
    parser.add_argument("--seed", type=int, default=13, help="RNG seed for input tensor.")
    parser.add_argument(
        "--per-layer-csv",
        type=Path,
        default=None,
        help="Optional CSV of per-leaf-layer metrics vs reference mode.",
    )
    parser.add_argument(
        "--saturation",
        action="store_true",
        help="Include INT16 FixedPointProfiler saturation stats in JSON.",
    )
    args = parser.parse_args(argv)

    set_quant_execution_mode(ExecutionMode.FP32_QDQ)
    torch.manual_seed(args.seed)
    modes = _parse_modes(args.modes)
    model, x = _build_model(args.model, args.input_size)

    report = compare_modes(
        model,
        x,
        modes=modes,
        include_per_layer=args.per_layer_csv is not None,
        include_saturation=args.saturation,
    )
    report["model"] = args.model
    report["input_shape"] = list(x.shape)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.per_layer_csv is not None:
        write_per_layer_csv(report, args.per_layer_csv)
        n_layers = len(report.get("per_layer", {}))
        print(f"Wrote per-layer CSV ({n_layers} layers): {args.per_layer_csv}")
    _print_summary(report)
    print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

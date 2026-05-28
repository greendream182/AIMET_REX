#!/usr/bin/env python3
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
"""CLI: freeze calibrated QuantSim encodings into an INT16 fixed-point sidecar.

Typical workflow after PTQ/QAT + ``ensure_output_quantizers_for_int16_eval``::

    from aimet_torch.fixed_point import ensure_output_quantizers_for_int16_eval
    from aimet_torch.fixed_point.offline import freeze_int16_fixed

    ensure_output_quantizers_for_int16_eval(sim)
    sim.compute_encodings(...)
    report = freeze_int16_fixed(sim, "exports/model.int16.json")

This script can also run a built-in tiny linear demo when ``--demo`` is set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _build_demo_model():
    import torch
    import torch.nn as nn

    import aimet_torch.fixed_point.kernels  # noqa: F401 — register fixed kernels

    from aimet_torch.v2.nn import QuantizedLinear
    from aimet_torch.v2.quantization.affine import Quantize

    linear = QuantizedLinear(4, 3)
    linear.input_quantizers[0] = Quantize((), 8, symmetric=True)
    linear.param_quantizers["weight"] = Quantize((3, 1), 8, symmetric=True)
    linear.output_quantizers[0] = Quantize((), 8, symmetric=True)
    linear.input_quantizers[0].min = nn.Parameter(torch.tensor(-2.0))
    linear.input_quantizers[0].max = nn.Parameter(torch.tensor(2.0))
    linear.param_quantizers["weight"].min = nn.Parameter(torch.full((3, 1), -0.5))
    linear.param_quantizers["weight"].max = nn.Parameter(torch.full((3, 1), 0.5))
    linear.output_quantizers[0].min = nn.Parameter(torch.tensor(-2.0))
    linear.output_quantizers[0].max = nn.Parameter(torch.tensor(2.0))
    nn.init.constant_(linear.weight, 0.1)
    nn.init.constant_(linear.bias, 0.0)
    return nn.Sequential(linear)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write INT16 sidecar JSON (e.g. model.int16.json).",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Optional path to write freeze diagnostics JSON.",
    )
    parser.add_argument(
        "--aimet-encoding",
        default=None,
        help="Optional AIMET .encodings file for ONNX tensor name hints.",
    )
    parser.add_argument(
        "--no-binaries",
        action="store_true",
        help="Do not write bias_int32 binary files.",
    )
    parser.add_argument(
        "--error-threshold",
        type=float,
        default=0.005,
        help="Relative multiplier warning threshold (default 0.005).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run built-in tiny QuantizedLinear demo instead of loading a checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Torch checkpoint containing a quantized model or QuantSim state.",
    )
    args = parser.parse_args(argv)

    if args.demo and args.checkpoint:
        print("Use either --demo or --checkpoint, not both.", file=sys.stderr)
        return 2

    if args.demo:
        model = _build_demo_model()
    elif args.checkpoint:
        import torch

        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and "model" in payload:
            model = payload["model"]
        else:
            model = payload
    else:
        print(
            "Provide --demo for a smoke test, or --checkpoint with a saved QuantSim/model.\n"
            "For production, call freeze_int16_fixed(sim, path) from your export script.",
            file=sys.stderr,
        )
        return 2

    from aimet_torch.fixed_point.offline.pipeline import freeze_int16_fixed

    summary = freeze_int16_fixed(
        model,
        args.output,
        write_binaries=not args.no_binaries,
        error_threshold=args.error_threshold,
        aimet_encoding_path=args.aimet_encoding,
    )
    print(f"Wrote sidecar: {summary['sidecar_path']}")
    print(
        f"Layers={summary['layer_count']} warnings={summary['warning_count']} "
        f"skipped={summary['skipped_count']}"
    )

    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote report: {args.report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

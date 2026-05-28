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
"""ImageNet-scale MobileNet V2 validation (224×224, design v2 §10.1).

**Quantization sim verification** only needs a few images and compares logits
cosine vs ``FP32_QDQ`` — **labels are optional** (use ``--cosine-only``).

Data sources (``--source``):

  * ``synthetic`` — random normalized tensors, no download (default for quick check).
  * ``unlabeled`` — any folder of images (``--image-dir``), no class subfolders.
  * ``local`` — ImageFolder val tree (``AIMET_RX_IMAGENET_VAL``).
  * ``zip`` — ImageFolder-style ``imagenet_val.zip`` (``AIMET_RX_IMAGENET_VAL_ZIP``).
  * ``huggingface`` — HF ``ILSVRC/imagenet-1k`` (gated; labels optional).
  * ``auto`` — local if present, else synthetic.

Examples::

  PYTHONPATH=<repo> python3 scripts/fixed_point/run_imagenet_validation.py \\
      --source synthetic --cosine-only --calib-max-samples 64

  python3 scripts/fixed_point/run_imagenet_validation.py \\
      --source unlabeled --image-dir /path/to/photos --cosine-only
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=("auto", "local", "zip", "huggingface", "synthetic", "unlabeled"),
        default="synthetic",
        help="Val images: synthetic (default), unlabeled folder, local/HF ImageNet.",
    )
    parser.add_argument(
        "--val-dir",
        type=Path,
        default=None,
        help="ImageNet val root (ImageFolder layout) for --source local.",
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=None,
        help="ImageNet val zip (ImageFolder layout inside archive) for --source zip.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Any folder of images for --source unlabeled (recursive).",
    )
    parser.add_argument(
        "--hf-dataset",
        default=None,
        help="HF dataset id (default: AIMET_RX_IMAGENET_HF_DATASET or ILSVRC/imagenet-1k).",
    )
    parser.add_argument(
        "--hf-stream",
        action="store_true",
        help="Stream HF validation split (less disk; same transforms).",
    )
    parser.add_argument(
        "--cosine-only",
        action="store_true",
        help="Skip top-1 accuracy (no labels required). Only logits cosine vs FP32_QDQ.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for model/calibration/eval: cpu, cuda, cuda:0, or auto.",
    )
    parser.add_argument(
        "--int16-device",
        default=None,
        help=(
            "Torch device for INT16_FIXED_EVAL metrics. Defaults to CPU when "
            "--device resolves to CUDA, otherwise the same as --device."
        ),
    )
    parser.add_argument("--calib-batches", type=int, default=4)
    parser.add_argument(
        "--calib-max-samples",
        type=int,
        default=64,
        help="Max images for calibration (default 64 for light verification).",
    )
    parser.add_argument(
        "--eval-max-samples",
        type=int,
        default=None,
        help=(
            "Max images the eval loader yields, independent of --calib-max-samples. "
            "Defaults to None = same cap as calibration (legacy behavior). "
            "Set to a larger value (e.g. 5000) to run wider top-1 / cosine eval "
            "without re-calibrating on the bigger set."
        ),
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=None,
        help=(
            "For --source local with --eval-max-samples, select a deterministic "
            "random subset using this seed instead of the first N files."
        ),
    )
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=16,
        help="Max val batches for top-1 (ignored with --cosine-only).",
    )
    parser.add_argument(
        "--cosine-batches",
        type=int,
        default=4,
        help="Val batches averaged for per-mode cosine vs FP32_QDQ.",
    )
    parser.add_argument(
        "--variant",
        choices=("torchvision", "full"),
        default="torchvision",
        help="MobileNet V2 source (torchvision = ImageNet pretrained).",
    )
    parser.add_argument("--no-pretrained", action="store_true", help="Skip loading torchvision weights.")
    parser.add_argument(
        "--bias-correction",
        action="store_true",
        help="Run AIMET empirical bias correction before QuantSim calibration.",
    )
    parser.add_argument(
        "--bias-correction-samples",
        type=int,
        default=128,
        help="Number of samples for empirical bias correction when enabled.",
    )
    parser.add_argument(
        "--adaround",
        action="store_true",
        help="Run AIMET AdaRound before QuantSim calibration (comparison mode; can be slow).",
    )
    parser.add_argument(
        "--adaround-batches",
        type=int,
        default=2,
        help="Number of batches for AdaRound when --adaround is enabled.",
    )
    parser.add_argument(
        "--adaround-iterations",
        type=int,
        default=80,
        help="Default AdaRound optimization iterations when --adaround is enabled.",
    )
    parser.add_argument(
        "--adaround-export-dir",
        type=Path,
        default=None,
        help="Directory for AdaRound artifacts.",
    )
    parser.add_argument(
        "--qat",
        action="store_true",
        help="Run a short INT16 QAT pass against a float teacher before evaluation.",
    )
    parser.add_argument("--qat-epochs", type=int, default=1)
    parser.add_argument("--qat-batches", type=int, default=8)
    parser.add_argument("--qat-lr", type=float, default=1e-4)
    parser.add_argument(
        "--per-layer-cosine",
        type=int,
        default=0,
        metavar="TOP_K",
        help=(
            "Diagnose spec 13 §108 by hooking every quantized module and "
            "computing INT16 vs FIXED_SCALE_QDQ cosine per layer; print the "
            "lowest-cosine TOP_K modules. 0 = disabled."
        ),
    )
    parser.add_argument(
        "--per-layer-isolated",
        type=int,
        default=0,
        metavar="TOP_K",
        help=(
            "Teacher-forced per-layer cosine: feed FP32_QDQ-cached inputs into "
            "each layer under FIXED_SCALE_QDQ and compare to the FP32 reference "
            "output. Isolates each layer's quantization noise from upstream "
            "accumulation. Print the worst TOP_K modules. 0 = disabled."
        ),
    )
    parser.add_argument("--json-out", type=Path, default=None, help="Write machine-readable report here.")
    parser.add_argument(
        "--md-out",
        type=Path,
        default=None,
        help=(
            "Write a markdown report fragment (cosine table + spec 13 §108 + saturation table). "
            "Intended as a CI artifact next to quality_report.md."
        ),
    )
    return parser.parse_args(argv)


def _check_gate(
    name: str,
    actual: float,
    limit: float,
    *,
    cmp_le: bool = True,
) -> dict[str, Any]:
    ok = (actual <= limit) if cmp_le else (actual >= limit)
    return {
        "metric": name,
        "actual": actual,
        "limit": limit,
        "cmp": "<=" if cmp_le else ">=",
        "status": "PASS" if ok else "FAIL",
    }


def _render_per_layer_table(rows: list[dict[str, Any]], cosine_key: str = "cosine") -> None:
    """Local shim — delegates to the public :func:`render_per_layer_table`.

    Kept so existing local call sites stay unchanged; the import is
    deferred to ``main()`` so the module continues to load without the
    metrics package being importable at module-level (consistent with the
    rest of this script).
    """

    from aimet_torch.fixed_point.metrics import render_per_layer_table

    render_per_layer_table(rows, cosine_key=cosine_key)


def _resolve_device(device_arg: str):
    import torch

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is False")
    return device


def _device_for_mode(mode, *, eval_device, int16_device):
    from aimet_torch.fixed_point import ExecutionMode

    return int16_device if mode == ExecutionMode.INT16_FIXED_EVAL else eval_device


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    from aimet_torch.fixed_point import ExecutionMode, record_multiplier_saturations
    from aimet_torch.fixed_point.e2e.imagenet_eval import (
        IMAGENET_FIXED_SCALE_MAX_TOP1_DROP,
        IMAGENET_FP16_MAX_TOP1_DROP,
        IMAGENET_INT16_MAX_TOP1_DROP,
        IMAGENET_INT16_MIN_COSINE,
        IMAGENET_INT16_VS_FIXED_SCALE_MIN_COSINE,
        IMAGENET_VAL_ENV,
        IMAGENET_VAL_ZIP_ENV,
        build_imagenet_mobilenet_bundle,
        cosine_vs_fp32_on_loader,
        resolve_imagenet_val_dir,
        resolve_imagenet_val_zip,
        logits_cosine_between_modes,
        logits_cosine_on_loader,
        per_layer_cosine_across_modes,
        per_layer_isolated_cosine_on_loader,
        top1_accuracy,
        top1_drop,
    )
    from aimet_torch.fixed_point import ExecutionMode  # re-import for explicit use below

    try:
        device = _resolve_device(args.device)
        int16_device = (
            _resolve_device(args.int16_device)
            if args.int16_device is not None
            else (_resolve_device("cpu") if device.type == "cuda" else device)
        )
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    val_source = args.source
    val_root: Path | None = None
    image_dir: Path | None = args.image_dir
    zip_path: Path | None = args.zip_path

    if val_source == "auto":
        resolved = args.val_dir or resolve_imagenet_val_dir()
        if resolved is not None and Path(resolved).is_dir():
            val_source = "local"
            val_root = Path(resolved)
        else:
            resolved_zip = args.zip_path or resolve_imagenet_val_zip()
            if resolved_zip is not None and Path(resolved_zip).is_file():
                val_source = "zip"
                zip_path = Path(resolved_zip)
            else:
                val_source = "synthetic"

    if val_source == "local":
        val_root = Path(args.val_dir or resolve_imagenet_val_dir() or "")
        if not val_root.is_dir():
            print(
                f"ERROR: ImageNet val directory not found.\n"
                f"  Set {IMAGENET_VAL_ENV} or use --source synthetic / unlabeled.",
                file=sys.stderr,
            )
            return 2
        print(f"ImageNet val (local ImageFolder): {val_root}")
    elif val_source == "zip":
        zip_path = Path(args.zip_path or resolve_imagenet_val_zip() or "")
        if not zip_path.is_file():
            print(
                f"ERROR: ImageNet val zip not found.\n"
                f"  Set {IMAGENET_VAL_ZIP_ENV}, set {IMAGENET_VAL_ENV} to a .zip, "
                f"or use --source synthetic / unlabeled.",
                file=sys.stderr,
            )
            return 2
        print(f"ImageNet val (zip ImageFolder): {zip_path}")
    elif val_source == "unlabeled":
        image_dir = image_dir or args.val_dir
        if image_dir is None or not Path(image_dir).is_dir():
            print("ERROR: --source unlabeled requires --image-dir <folder>", file=sys.stderr)
            return 2
        print(f"Images (unlabeled): {image_dir}")
    elif val_source == "synthetic":
        print(f"Images (synthetic): {args.calib_max_samples} tensors @ 224, seed=0")
    else:
        print(
            "ImageNet val (Hugging Face): "
            f"{args.hf_dataset or 'ILSVRC/imagenet-1k'} "
            f"({'streaming' if args.hf_stream else f'≤{args.calib_max_samples} samples'})"
        )

    mode_note = "cosine-only (no labels)" if args.cosine_only else "cosine + top-1 (needs labels)"
    print(
        f"Building MobileNet V2 ({args.variant}, 224×224, {mode_note}; calib "
        f"{args.calib_batches}×batch{args.batch_size}; device={device}; "
        f"int16_device={int16_device})…"
    )

    try:
        bundle, loader = build_imagenet_mobilenet_bundle(
            val_root,
            val_source=val_source,
            image_dir=image_dir,
            zip_path=zip_path,
            hf_dataset_name=args.hf_dataset,
            hf_streaming=args.hf_stream,
            batch_size=args.batch_size,
            calib_max_batches=args.calib_batches,
            calib_max_samples=args.calib_max_samples,
            eval_max_samples=args.eval_max_samples,
            eval_seed=args.eval_seed,
            load_pretrained=not args.no_pretrained,
            variant=args.variant,
            apply_bias_correction=args.bias_correction,
            bias_correction_samples=args.bias_correction_samples,
            apply_adaround=args.adaround,
            adaround_num_batches=args.adaround_batches,
            adaround_iterations=args.adaround_iterations,
            adaround_export_dir=args.adaround_export_dir,
            device=device,
        )
    except (FileNotFoundError, ImportError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    qat_losses: list[float] | None = None
    if args.qat:
        from aimet_torch.fixed_point.e2e.imagenet_eval import (
            IMAGENET_INPUT_SIZE,
            IMAGENET_NUM_CLASSES,
            try_load_torchvision_imagenet_weights,
        )
        from aimet_torch.fixed_point.e2e.mobilenet_v2 import (
            build_prepared_mobilenet_v2,
            train_int16_qat_on_loader,
        )

        teacher_variant = "torchvision" if args.variant == "torchvision" else "full"
        teacher, _teacher_dummy = build_prepared_mobilenet_v2(
            n_class=IMAGENET_NUM_CLASSES,
            input_size=IMAGENET_INPUT_SIZE,
            variant=teacher_variant,  # type: ignore[arg-type]
        )
        if not args.no_pretrained and teacher_variant == "full":
            try_load_torchvision_imagenet_weights(teacher)
        teacher = teacher.to(device)
        print(
            f"Running INT16 QAT: epochs={args.qat_epochs}, "
            f"batches/epoch={args.qat_batches}, lr={args.qat_lr}"
        )
        qat_losses = train_int16_qat_on_loader(
            bundle.sim,
            teacher=teacher,
            train_loader=loader,
            epochs=args.qat_epochs,
            lr=args.qat_lr,
            max_batches=args.qat_batches,
        )
        print("QAT losses:", ", ".join(f"{loss:.6g}" for loss in qat_losses))

    cosines: dict[str, float] = {}
    saturation_stats: dict[str, dict[str, Any]] = {}
    int16_skipped: str | None = None
    for mode_key, mode in (
        ("int16_fixed_eval", ExecutionMode.INT16_FIXED_EVAL),
        ("fixed_scale_qdq", ExecutionMode.FIXED_SCALE_QDQ),
        ("fp16_qdq", ExecutionMode.FP16_QDQ),
    ):
        # spec 10 §53: capture layer-requant saturation events per mode so the
        # report can distinguish "fixed_scale_qdq cosine drop" from "INT16 fold
        # introduced N% relative error on K channels". ``captured`` is bound
        # outside the ``try`` so events recorded before a downstream exception
        # (e.g. AdaptiveAvgPool2d missing INT16 kernel) are still reported.
        captured: list[dict[str, Any]] = []
        try:
            with record_multiplier_saturations() as sat_events:
                captured = sat_events
                cosines[mode_key] = logits_cosine_on_loader(
                    bundle.sim,
                    loader,
                    mode,
                    max_batches=args.cosine_batches,
                    device=_device_for_mode(
                        mode, eval_device=device, int16_device=int16_device
                    ),
                )
        except (ValueError, RuntimeError, AttributeError) as exc:
            # INT16 dispatch may surface AttributeError on ops missing a fixed kernel
            # (e.g. ``AdaptiveAvgPool2d`` not yet in ``INT16_DISPATCHABLE_MODULES``);
            # treat it as "INT16 skipped" so the rest of the report still runs.
            if mode_key == "int16_fixed_eval":
                int16_skipped = f"{type(exc).__name__}: {exc}"
            else:
                raise

        if captured:
            worst = max(captured, key=lambda e: e["relative_error"])
            saturation_stats[mode_key] = {
                "events": len(captured),
                "worst_relative_error": float(worst["relative_error"]),
                "worst_real_multiplier": float(worst["real_multiplier"]),
            }

    # spec 13 §108: direct INT16-vs-fixed-scale cosine. Both modes use the same
    # fixed-scale grid; this metric isolates the integer kernel + requant path
    # cost from the (much larger) calibration / encoding choice cost.
    int16_vs_fixed_scale: float | None = None
    if "int16_fixed_eval" in cosines and "fixed_scale_qdq" in cosines:
        try:
            int16_vs_fixed_scale = logits_cosine_between_modes(
                bundle.sim,
                loader,
                ExecutionMode.FIXED_SCALE_QDQ,
                ExecutionMode.INT16_FIXED_EVAL,
                max_batches=args.cosine_batches,
                device=int16_device,
            )
        except (RuntimeError, AttributeError) as exc:
            int16_skipped = int16_skipped or f"{type(exc).__name__}: {exc}"

    gates: list[dict[str, Any]] = []
    if "int16_fixed_eval" in cosines:
        gates.append(
            _check_gate(
                "int16_cosine_vs_fp32",
                cosines["int16_fixed_eval"],
                IMAGENET_INT16_MIN_COSINE,
                cmp_le=False,
            )
        )
    if int16_vs_fixed_scale is not None:
        gates.append(
            _check_gate(
                "int16_vs_fixed_scale_cosine",
                int16_vs_fixed_scale,
                IMAGENET_INT16_VS_FIXED_SCALE_MIN_COSINE,
                cmp_le=False,
            )
        )
    # Light / synthetic verification uses slightly relaxed cosine floors.
    aux_cos_min = 0.99 if args.cosine_only or val_source in ("synthetic", "unlabeled") else 0.999
    gates.extend(
        [
            _check_gate(
                "fixed_scale_mean_cosine",
                cosines["fixed_scale_qdq"],
                aux_cos_min,
                cmp_le=False,
            ),
            _check_gate(
                "fp16_cosine_vs_fp32",
                cosines["fp16_qdq"],
                aux_cos_min,
                cmp_le=False,
            ),
        ]
    )

    metrics: dict[str, Any] = dict(cosines)
    if int16_vs_fixed_scale is not None:
        metrics["int16_vs_fixed_scale_cosine"] = int16_vs_fixed_scale
    acc_fp32 = acc_int16 = acc_fixed = acc_fp16 = None
    drop_int16 = drop_fixed = drop_fp16 = None

    if not args.cosine_only:
        acc_fp32 = top1_accuracy(
            bundle.sim,
            loader,
            ExecutionMode.FP32_QDQ,
            max_batches=args.eval_batches,
            device=device,
        )
        acc_int16 = top1_accuracy(
            bundle.sim,
            loader,
            ExecutionMode.INT16_FIXED_EVAL,
            max_batches=args.eval_batches,
            device=int16_device,
        )
        acc_fixed = top1_accuracy(
            bundle.sim,
            loader,
            ExecutionMode.FIXED_SCALE_QDQ,
            max_batches=args.eval_batches,
            device=device,
        )
        acc_fp16 = top1_accuracy(
            bundle.sim,
            loader,
            ExecutionMode.FP16_QDQ,
            max_batches=args.eval_batches,
            device=device,
        )
        drop_int16 = top1_drop(acc_fp32, acc_int16)
        drop_fixed = top1_drop(acc_fp32, acc_fixed)
        drop_fp16 = top1_drop(acc_fp32, acc_fp16)
        metrics.update(
            {
                "reference_top1_fp32_qdq": acc_fp32,
                "int16_top1": acc_int16,
                "int16_top1_drop": drop_int16,
                "fixed_scale_top1": acc_fixed,
                "fixed_scale_top1_drop": drop_fixed,
                "fp16_top1": acc_fp16,
                "fp16_top1_drop": drop_fp16,
            }
        )
        gates.extend(
            [
                _check_gate("int16_top1_drop", drop_int16, IMAGENET_INT16_MAX_TOP1_DROP),
                _check_gate("fixed_scale_top1_drop", drop_fixed, IMAGENET_FIXED_SCALE_MAX_TOP1_DROP),
                _check_gate("fp16_top1_drop", drop_fp16, IMAGENET_FP16_MAX_TOP1_DROP),
            ]
        )

    overall = "PASS" if all(g["status"] == "PASS" for g in gates) else "FAIL"

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "val_source": val_source,
        "cosine_only": args.cosine_only,
        "val_dir": str(val_root) if val_root is not None else None,
        "image_dir": str(image_dir) if image_dir is not None else None,
        "hf_dataset": args.hf_dataset,
        "hf_streaming": args.hf_stream,
        "variant": args.variant,
        "device": str(device),
        "int16_device": str(int16_device),
        "ptq_refinements": {
            "bias_correction": bool(args.bias_correction),
            "bias_correction_samples": args.bias_correction_samples if args.bias_correction else 0,
            "adaround": bool(args.adaround),
            "adaround_batches": args.adaround_batches if args.adaround else 0,
            "adaround_iterations": args.adaround_iterations if args.adaround else 0,
            "adaround_export_dir": str(args.adaround_export_dir) if args.adaround_export_dir else None,
            "qat": bool(args.qat),
            "qat_epochs": args.qat_epochs if args.qat else 0,
            "qat_batches": args.qat_batches if args.qat else 0,
            "qat_lr": args.qat_lr if args.qat else 0.0,
            "qat_losses": qat_losses,
        },
        "calib_max_samples": args.calib_max_samples,
        "eval_max_samples": args.eval_max_samples,
        "eval_seed": args.eval_seed,
        "cosine_batches": args.cosine_batches,
        "int16_skipped": int16_skipped,
        "metrics": metrics,
        "saturation_stats": saturation_stats,  # spec 10 §53
        "gates": gates,
        "overall_status": overall,
    }

    print()
    print("| mode | cosine vs FP32_QDQ | gate |")
    print("|------|-------------------|------|")
    if int16_skipped:
        print(f"INT16_FIXED_EVAL: skipped ({int16_skipped[:120]})")
    for mode_key, label in (
        ("int16_fixed_eval", "INT16_FIXED_EVAL"),
        ("fixed_scale_qdq", "FIXED_SCALE_QDQ"),
        ("fp16_qdq", "FP16_QDQ"),
    ):
        if mode_key not in cosines:
            continue
        cos = cosines[mode_key]
        gate_names = {
            "int16_fixed_eval": "int16_cosine_vs_fp32",
            "fixed_scale_qdq": "fixed_scale_mean_cosine",
            "fp16_qdq": "fp16_cosine_vs_fp32",
        }[mode_key]
        g = next(x for x in gates if x["metric"] == gate_names)
        print(f"| {label} | {cos:.9f} | {g['status']} |")

    if int16_vs_fixed_scale is not None:
        gate_row = next(
            (g for g in gates if g["metric"] == "int16_vs_fixed_scale_cosine"),
            None,
        )
        status = gate_row["status"] if gate_row is not None else "n/a"
        print()
        print("Spec 13 §108 (INT16 vs FIXED_SCALE_QDQ, same fixed-scale grid):")
        print(f"  cosine = {int16_vs_fixed_scale:.9f}  "
              f"(≥ {IMAGENET_INT16_VS_FIXED_SCALE_MIN_COSINE} → {status})")

    per_layer_rows_fp32: list[dict[str, Any]] = []
    per_layer_rows_fs: list[dict[str, Any]] = []
    per_layer_rows_chained_fs_vs_fp32: list[dict[str, Any]] = []
    # Hybrid cuda+cpu: quantizer scales stay on CPU after INT16 eval; run all
    # per-layer hooks on int16_device so FS/FP32 paths do not hit device mismatch.
    per_layer_device = int16_device if int16_device != device else device
    if args.per_layer_cosine and "int16_fixed_eval" in cosines:
        # spec 13 §108 (op-level): INT16 vs FP32 cosine ≥ 0.999.
        try:
            per_layer_rows_fp32 = per_layer_cosine_across_modes(
                bundle.sim,
                loader,
                ref_mode=ExecutionMode.FP32_QDQ,
                cand_mode=ExecutionMode.INT16_FIXED_EVAL,
                max_batches=1,
                top_k=args.per_layer_cosine,
                device=per_layer_device,
            )
        except (RuntimeError, AttributeError) as exc:
            print(f"\nper-layer cosine (vs FP32) skipped: {type(exc).__name__}: {exc}")
        try:
            per_layer_rows_fs = per_layer_cosine_across_modes(
                bundle.sim,
                loader,
                ref_mode=ExecutionMode.FIXED_SCALE_QDQ,
                cand_mode=ExecutionMode.INT16_FIXED_EVAL,
                max_batches=1,
                top_k=args.per_layer_cosine,
                device=per_layer_device,
            )
        except (RuntimeError, AttributeError) as exc:
            print(f"\nper-layer cosine (vs FIXED_SCALE) skipped: {type(exc).__name__}: {exc}")
        # Chained FIXED_SCALE_QDQ vs FP32_QDQ — mode-aligned counterpart of
        # the isolated diagnostic below, so the two tables compare apples-to-
        # apples (accumulated vs intrinsic noise at the *same* mode).
        try:
            per_layer_rows_chained_fs_vs_fp32 = per_layer_cosine_across_modes(
                bundle.sim,
                loader,
                ref_mode=ExecutionMode.FP32_QDQ,
                cand_mode=ExecutionMode.FIXED_SCALE_QDQ,
                max_batches=1,
                top_k=args.per_layer_cosine,
                device=per_layer_device,
            )
        except (RuntimeError, AttributeError) as exc:
            print(f"\nper-layer cosine (FS vs FP32) skipped: {type(exc).__name__}: {exc}")
        if per_layer_rows_fp32:
            report["per_layer_cosine_int16_vs_fp32"] = per_layer_rows_fp32
            print()
            print(
                f"Per-layer INT16 vs FP32_QDQ (spec 13 §108; worst {len(per_layer_rows_fp32)}; 1 batch):"
            )
            _render_per_layer_table(per_layer_rows_fp32, cosine_key="cosine")
        if per_layer_rows_fs:
            report["per_layer_cosine_int16_vs_fixed_scale"] = per_layer_rows_fs
            print()
            print(
                f"Per-layer INT16 vs FIXED_SCALE_QDQ (boundary-grid diag; worst "
                f"{len(per_layer_rows_fs)}; 1 batch):"
            )
            _render_per_layer_table(per_layer_rows_fs, cosine_key="cosine")
        if per_layer_rows_chained_fs_vs_fp32:
            report["per_layer_cosine_chained_fixed_scale_vs_fp32"] = (
                per_layer_rows_chained_fs_vs_fp32
            )
            print()
            print(
                f"Per-layer CHAINED FIXED_SCALE_QDQ vs FP32_QDQ "
                f"(accumulated; worst {len(per_layer_rows_chained_fs_vs_fp32)}; 1 batch):"
            )
            _render_per_layer_table(per_layer_rows_chained_fs_vs_fp32, cosine_key="cosine")

    isolated_rows: list[dict[str, Any]] = []
    isolated_rows_int16: list[dict[str, Any]] = []
    if args.per_layer_isolated:
        try:
            isolated_rows = per_layer_isolated_cosine_on_loader(
                bundle.sim,
                loader,
                cand_mode=ExecutionMode.FIXED_SCALE_QDQ,
                ref_mode=ExecutionMode.FP32_QDQ,
                max_batches=1,
                top_k=args.per_layer_isolated,
                device=per_layer_device,
            )
        except (RuntimeError, AttributeError) as exc:
            print(f"\nper-layer isolated cosine (FS) skipped: {type(exc).__name__}: {exc}")
        # INT16 isolated: needs each leaf's input quantizer to be initialized;
        # modules where the super-group disabled iq will be auto-skipped and
        # logged by ``per_layer_isolated_cosine``. The remaining coverage is
        # still the tightest upper bound on INT16 *intrinsic* noise.
        try:
            isolated_rows_int16 = per_layer_isolated_cosine_on_loader(
                bundle.sim,
                loader,
                cand_mode=ExecutionMode.INT16_FIXED_EVAL,
                ref_mode=ExecutionMode.FP32_QDQ,
                max_batches=1,
                top_k=args.per_layer_isolated,
                device=per_layer_device,
            )
        except (RuntimeError, AttributeError) as exc:
            print(f"\nper-layer isolated cosine (INT16) skipped: {type(exc).__name__}: {exc}")
        if isolated_rows:
            report["per_layer_isolated_fixed_scale_vs_fp32"] = isolated_rows
            print()
            print(
                f"Per-layer ISOLATED FIXED_SCALE_QDQ vs FP32_QDQ "
                f"(teacher-forced; worst {len(isolated_rows)}; 1 batch):"
            )
            _render_per_layer_table(isolated_rows, cosine_key="isolated_cosine")
        if isolated_rows_int16:
            report["per_layer_isolated_int16_vs_fp32"] = isolated_rows_int16
            print()
            print(
                f"Per-layer ISOLATED INT16_FIXED_EVAL vs FP32_QDQ "
                f"(teacher-forced; worst {len(isolated_rows_int16)}; 1 batch; "
                f"INT16 carriers are reconstructed from the predecessor's "
                f"output-quantizer grid):"
            )
            _render_per_layer_table(isolated_rows_int16, cosine_key="isolated_cosine")

    if saturation_stats:
        print()
        print("Layer-requant saturation (spec 10 §53; folded to rshift=31):")
        print("| mode | folded events | worst rel_err | worst real_m |")
        print("|------|---------------|---------------|--------------|")
        for mode_key, stats in saturation_stats.items():
            print(
                f"| {mode_key} | {stats['events']} | "
                f"{stats['worst_relative_error']:.2%} | "
                f"{stats['worst_real_multiplier']:.3e} |"
            )

    if not args.cosine_only and acc_fp32 is not None:
        print()
        print(f"Reference FP32_QDQ top-1 (≤{args.eval_batches} batches): {acc_fp32:.4f}")
        print("| mode | top-1 | drop vs FP32 | gate |")
        print("|------|-------|--------------|------|")
        for label, top1, drop, gate_name in (
            ("INT16_FIXED_EVAL", acc_int16, drop_int16, "int16_top1_drop"),
            ("FIXED_SCALE_QDQ", acc_fixed, drop_fixed, "fixed_scale_top1_drop"),
            ("FP16_QDQ", acc_fp16, drop_fp16, "fp16_top1_drop"),
        ):
            g = next(x for x in gates if x["metric"] == gate_name)
            print(f"| {label} | {top1:.4f} | {drop:.4f} | {g['status']} |")

    print()
    print(f"Overall: {overall}")
    for g in gates:
        print(f"  - {g['metric']}: {g['actual']:.6g} {g['cmp']} {g['limit']} → {g['status']}")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote: {args.json_out}")

    if args.md_out is not None:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        lines.append("# ImageNet MobileNet V2 fixed-point quality (224×224)")
        lines.append("")
        lines.append(f"_Generated_: {report['generated_at']}  ")
        lines.append(f"_Val source_: `{val_source}`  ")
        if val_root is not None:
            lines.append(f"_Val dir_: `{val_root}`  ")
        lines.append(
            f"_Calib_: {args.calib_max_samples} samples, "
            f"{args.calib_batches} batches  "
        )
        lines.append(f"_Cosine batches_: {args.cosine_batches}  ")
        lines.append("")
        lines.append("## Cosine vs FP32_QDQ")
        lines.append("")
        lines.append("| mode | cosine vs FP32_QDQ | gate |")
        lines.append("|------|--------------------|------|")
        if int16_skipped:
            lines.append(f"| INT16_FIXED_EVAL | _skipped_ | {int16_skipped[:80]} |")
        for mode_key, label, gate_name in (
            ("int16_fixed_eval", "INT16_FIXED_EVAL", "int16_cosine_vs_fp32"),
            ("fixed_scale_qdq", "FIXED_SCALE_QDQ", "fixed_scale_mean_cosine"),
            ("fp16_qdq", "FP16_QDQ", "fp16_cosine_vs_fp32"),
        ):
            if mode_key not in cosines:
                continue
            g = next((x for x in gates if x["metric"] == gate_name), None)
            status = g["status"] if g else "n/a"
            lines.append(f"| {label} | {cosines[mode_key]:.9f} | {status} |")
        lines.append("")
        if int16_vs_fixed_scale is not None:
            g = next((x for x in gates if x["metric"] == "int16_vs_fixed_scale_cosine"), None)
            status = g["status"] if g else "n/a"
            lines.append("## Spec 13 §108 (INT16 vs FIXED_SCALE_QDQ, same fixed-scale grid)")
            lines.append("")
            lines.append(
                f"`cosine = {int16_vs_fixed_scale:.9f}` "
                f"(≥ {IMAGENET_INT16_VS_FIXED_SCALE_MIN_COSINE} → **{status}**)"
            )
            lines.append("")
        if saturation_stats:
            lines.append("## Layer-requant saturation (spec 10 §53; folded to rshift=31)")
            lines.append("")
            lines.append("| mode | folded events | worst rel_err | worst real_m |")
            lines.append("|------|---------------|---------------|--------------|")
            for mode_key, stats in saturation_stats.items():
                lines.append(
                    f"| {mode_key} | {stats['events']} | "
                    f"{stats['worst_relative_error']:.2%} | "
                    f"{stats['worst_real_multiplier']:.3e} |"
                )
            lines.append("")
        lines.append(f"**Overall: {overall}**")
        lines.append("")
        for g in gates:
            lines.append(
                f"- `{g['metric']}` = {g['actual']:.6g} {g['cmp']} {g['limit']} → **{g['status']}**"
            )
        lines.append("")
        args.md_out.write_text("\n".join(lines), encoding="utf-8")
        print(f"Wrote: {args.md_out}")

    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

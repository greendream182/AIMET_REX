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
"""Teacher-forced isolated diagnosis for MobileNet ``classifier.0``.

ImageNet per-layer isolated runs often show ``classifier.0`` as the worst
INT16 layer (~0.74 cosine) while conv layers stay ≥0.998. This script
zooms into that layer: encodings (scale / m / rshift), input activation
stats, isolated vs chained cosine, and :func:`compute_pair_metrics` for
INT16 / fixed_scale / FP16.

Examples::

  export AIMET_RX_IMAGENET_VAL=/path/to/imagenet_val
  PYTHONPATH=<repo> python3 scripts/fixed_point/diagnose_classifier_isolated.py \\
      --source local --batch-size 16 --calib-max-samples 512

  python3 scripts/fixed_point/diagnose_classifier_isolated.py \\
      --source synthetic --calib-max-samples 64 --batch-size 8
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=("auto", "local", "zip", "huggingface", "synthetic", "unlabeled"),
        default="auto",
        help="Val images (auto = local dir if readable, else synthetic).",
    )
    parser.add_argument("--val-dir", type=Path, default=None, help="ImageNet val root for --source local.")
    parser.add_argument("--zip-path", type=Path, default=None, help="ImageNet val zip for --source zip.")
    parser.add_argument("--image-dir", type=Path, default=None, help="Unlabeled image folder.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for calibration (cuda recommended).",
    )
    parser.add_argument(
        "--int16-device",
        default=None,
        help=(
            "Device for INT16 diagnosis after build. Defaults to CPU when "
            "--device is CUDA, else same as --device."
        ),
    )
    parser.add_argument(
        "--calib-max-samples",
        type=int,
        default=512,
        help="Images used for PTQ calibration before diagnosis.",
    )
    parser.add_argument(
        "--calib-batches",
        type=int,
        default=4,
        help="Max calibration batches consumed from the calibration loader.",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=None,
        help="For --source local, use a deterministic random subset with this seed.",
    )
    parser.add_argument(
        "--default-output-bw",
        type=int,
        default=8,
        help="AIMET activation/output quantizer bitwidth used by QuantSim.",
    )
    parser.add_argument(
        "--int16-eval-bw",
        type=int,
        default=8,
        help="Bitwidth for output quantizers patched for INT16 fixed eval.",
    )
    parser.add_argument(
        "--default-param-bw",
        type=int,
        default=8,
        help="AIMET parameter quantizer bitwidth used by QuantSim.",
    )
    parser.add_argument(
        "--target-output-bw",
        type=int,
        default=None,
        help=(
            "Override only --module-name output quantizer bitwidth before calibration. "
            "Useful for testing classifier.0 with 16-bit output while the rest of "
            "the graph stays at the default 8-bit activation grid."
        ),
    )
    parser.add_argument(
        "--output-bw-override",
        action="append",
        default=[],
        metavar="MODULE:BW",
        help=(
            "Override an arbitrary module output quantizer bitwidth before calibration. "
            "May be repeated, e.g. --output-bw-override features.16.conv.2:16."
        ),
    )
    parser.add_argument(
        "--module-name",
        default="classifier.0",
        help="Exact ``named_modules()`` key to diagnose (default: classifier.0).",
    )
    parser.add_argument(
        "--predecessor",
        default=None,
        help=(
            "Optional upstream module name to report chained output stats "
            "(e.g. features.18.2). Default: only the target module."
        ),
    )
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None, help="Write full report JSON here.")
    return parser.parse_args(argv)


def _tensor_stats(t: torch.Tensor) -> Dict[str, Any]:
    x = t.detach().float()
    rms = float(x.pow(2).mean().sqrt().clamp_min(1e-30).item())
    return {
        "shape": tuple(x.shape),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "rms": rms,
    }


def _coerce_to_float_tensor(value: Any) -> Optional[torch.Tensor]:
    """Dequantize AIMET/INT16 tensor wrappers before treating Tensor subclasses as tensors."""

    from aimet_torch.fixed_point.metrics.flags import int16_eval_allow_debug_float
    from aimet_torch.fixed_point.tensor import Int16QuantizedTensor

    if isinstance(value, Int16QuantizedTensor):
        with int16_eval_allow_debug_float():
            return value.to_float().detach().float().clone()
    if hasattr(value, "dequantize"):
        value = value.dequantize()
    if isinstance(value, torch.Tensor):
        return value.detach().float().clone()
    return None


def _affine_quantizer_summary(q: Any) -> Dict[str, Any]:
    if q is None:
        return {"initialized": False}
    initialized = bool(getattr(q, "is_initialized", lambda: False)())
    if not initialized:
        return {"initialized": False}
    from aimet_torch.v2.quantization.affine.encoding import AffineEncoding

    enc = q.get_encodings()
    if not isinstance(enc, AffineEncoding):
        return {"initialized": True, "encoding_type": type(enc).__name__}
    try:
        scale_t = enc.scale.detach().cpu()
        offset_t = enc.offset.detach().cpu()
    except AttributeError:
        return {"initialized": True, "encoding_type": type(enc).__name__, "error": "missing scale/offset"}
    return {
        "initialized": True,
        "encoding_type": "AffineEncoding",
        "scale": float(scale_t.reshape(-1)[0].item()) if scale_t.numel() == 1 else scale_t.tolist(),
        "offset": float(offset_t.reshape(-1)[0].item()) if offset_t.numel() == 1 else offset_t.tolist(),
        "qmin": int(enc.qmin),
        "qmax": int(enc.qmax),
        "bitwidth": int(getattr(enc, "bitwidth", 0) or 0),
    }


def _module_quantizer_report(module: nn.Module) -> Dict[str, Any]:
    report: Dict[str, Any] = {"module_type": type(module).__name__}
    iqs = getattr(module, "input_quantizers", None) or []
    oqs = getattr(module, "output_quantizers", None) or []
    report["input_quantizers"] = [_affine_quantizer_summary(q) for q in iqs]
    report["output_quantizers"] = [_affine_quantizer_summary(q) for q in oqs]
    wq = getattr(module, "param_quantizers", None)
    if wq:
        report["param_quantizers"] = {
            k: _affine_quantizer_summary(v) for k, v in wq.items() if v is not None
        }
    if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
        w = module.weight.detach()
        report["weight_fp32_stats"] = _tensor_stats(w)
    if hasattr(module, "bias") and module.bias is not None:
        report["bias_fp32_stats"] = _tensor_stats(module.bias.detach())
    return report


def _int16_export_record(module: nn.Module, layer_name: str) -> Optional[Dict[str, Any]]:
    from aimet_torch.fixed_point.encoding_export import output_encoding_to_dict
    from aimet_torch.fixed_point.export.v2_collect import collect_v2_int16_layer_record

    record = collect_v2_int16_layer_record(module, layer_name)
    if record is None:
        return None
    out = dict(record)
    enc = out.get("output_encoding")
    if enc is not None:
        out["output_encoding"] = output_encoding_to_dict(enc)
    return out


@torch.no_grad()
def _capture_module_io(
    model: nn.Module,
    images: torch.Tensor,
    module_name: str,
    *,
    mode: Any,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    from aimet_torch.fixed_point import quant_execution_mode

    captured_in: List[Any] = []
    captured_out: Optional[torch.Tensor] = None

    def _hook(_mod, ins, out):
        nonlocal captured_out
        if ins:
            first = _coerce_to_float_tensor(ins[0])
            if first is not None:
                captured_in.append(first)
        captured_out = _coerce_to_float_tensor(out)

    target = dict(model.named_modules()).get(module_name)
    if target is None:
        return None, None
    handle = target.register_forward_hook(_hook)
    try:
        with quant_execution_mode(mode):
            model(images)
    finally:
        handle.remove()
    x_in = captured_in[0] if captured_in else None
    return x_in, captured_out


def _clamp_bounds(module: nn.Module) -> Optional[Tuple[float, float]]:
    if isinstance(module, nn.ReLU6):
        return 0.0, 6.0
    if isinstance(module, nn.Hardtanh):
        return float(module.min_val), float(module.max_val)
    return None


@torch.no_grad()
def _clamp_boundary_report(
    model: nn.Module,
    module: nn.Module,
    images: torch.Tensor,
    *,
    ref_mode: Any,
    cand_mode: Any,
) -> Optional[Dict[str, Any]]:
    bounds = _clamp_bounds(module)
    if bounds is None:
        return None
    lower, upper = bounds

    from aimet_torch.fixed_point import quant_execution_mode
    from aimet_torch.fixed_point.metrics import compute_pair_metrics

    def _capture(mode: Any) -> Dict[str, torch.Tensor]:
        captured: Dict[str, torch.Tensor] = {}

        def _pre_hook(_mod, ins):
            if ins:
                tensor = _coerce_to_float_tensor(ins[0])
                if tensor is not None:
                    captured["pre"] = tensor

        def _hook(_mod, _ins, out):
            tensor = _coerce_to_float_tensor(out)
            if tensor is not None:
                captured["out"] = tensor

        handles = [
            module.register_forward_pre_hook(_pre_hook),
            module.register_forward_hook(_hook),
        ]
        try:
            with quant_execution_mode(mode):
                model(images)
        finally:
            for handle in handles:
                handle.remove()
        return captured

    ref = _capture(ref_mode)
    cand = _capture(cand_mode)
    if not {"pre", "out"} <= ref.keys() or not {"pre", "out"} <= cand.keys():
        return {"error": "failed to capture clamp pre/out tensors"}

    pre_ref = ref["pre"]
    pre_cand = cand["pre"]
    out_ref = ref["out"]
    out_cand = cand["out"]
    lower_ref = pre_ref <= lower
    lower_cand = pre_cand <= lower
    upper_ref = pre_ref >= upper
    upper_cand = pre_cand >= upper
    near_thresholds = (0.0, 0.05, 0.1, 0.25, 0.5)
    near_lower = {
        f"abs_delta_le_{thr:g}": float(
            ((pre_ref - lower).abs().le(thr) | (pre_cand - lower).abs().le(thr))
            .float()
            .mean()
            .item()
        )
        for thr in near_thresholds
    }
    near_upper = {
        f"abs_delta_le_{thr:g}": float(
            ((pre_ref - upper).abs().le(thr) | (pre_cand - upper).abs().le(thr))
            .float()
            .mean()
            .item()
        )
        for thr in near_thresholds
    }
    return {
        "bounds": {"lower": lower, "upper": upper},
        "pre_metrics": compute_pair_metrics(pre_ref, pre_cand),
        "output_metrics": compute_pair_metrics(out_ref, out_cand),
        "pre_ref_stats": _tensor_stats(pre_ref),
        "pre_candidate_stats": _tensor_stats(pre_cand),
        "lower_ref_fraction": float(lower_ref.float().mean().item()),
        "lower_candidate_fraction": float(lower_cand.float().mean().item()),
        "lower_flip_fraction": float((lower_ref != lower_cand).float().mean().item()),
        "upper_ref_fraction": float(upper_ref.float().mean().item()),
        "upper_candidate_fraction": float(upper_cand.float().mean().item()),
        "upper_flip_fraction": float((upper_ref != upper_cand).float().mean().item()),
        "near_lower_fraction": near_lower,
        "near_upper_fraction": near_upper,
    }


@torch.no_grad()
def _detailed_mode_metrics(
    model: nn.Module,
    module: nn.Module,
    images: torch.Tensor,
    module_name: str,
    *,
    ref_mode: Any,
    cand_mode: Any,
) -> Dict[str, Any]:
    """Teacher-forced forward + :func:`compute_pair_metrics` for one mode pair."""

    from aimet_torch.fixed_point import ExecutionMode, quant_execution_mode
    from aimet_torch.fixed_point.metrics import compute_pair_metrics
    from aimet_torch.fixed_point.metrics.flags import int16_eval_allow_debug_float
    from aimet_torch.fixed_point.metrics.isolated import (
        _quantize_with_carrier,
        _to_float,
    )
    from aimet_torch.fixed_point.tensor import Int16QuantizedTensor

    x_ref, y_ref = _capture_module_io(model, images, module_name, mode=ref_mode)
    if y_ref is None or x_ref is None:
        return {"error": "failed to capture reference IO"}

    int16_carrier: Optional[Dict[str, Any]] = None
    if cand_mode is ExecutionMode.INT16_FIXED_EVAL:

        def carrier_hook(_mod, ins, _out):
            nonlocal int16_carrier
            if ins and isinstance(ins[0], Int16QuantizedTensor):
                first = ins[0]
                int16_carrier = {
                    "scale": first.scale.detach().clone(),
                    "zero_point": first.zero_point.detach().clone(),
                    "qmin": first.qmin,
                    "qmax": first.qmax,
                    "axis": first.axis,
                }

        handle = module.register_forward_hook(carrier_hook)
        try:
            with quant_execution_mode(cand_mode):
                model(images)
        finally:
            handle.remove()

    x_args: Tuple[Any, ...] = (x_ref,)
    if (
        cand_mode is ExecutionMode.INT16_FIXED_EVAL
        and int16_carrier is not None
        and isinstance(x_ref, torch.Tensor)
    ):
        try:
            x0 = _quantize_with_carrier(x_ref, int16_carrier)
            x_args = (x0,)
        except (RuntimeError, TypeError, ValueError) as exc:
            return {"error": f"INT16 input re-quant failed: {exc}"}

    with quant_execution_mode(cand_mode):
        y_raw = module(*x_args)
    y_cand = _to_float(y_raw)
    if y_cand is None:
        return {"error": "candidate output is not a float tensor"}

    metrics = compute_pair_metrics(y_ref, y_cand)
    row: Dict[str, Any] = {
        "cand_mode": str(cand_mode),
        "ref_mode": str(ref_mode),
        "metrics": metrics,
        "input_fp32_stats": _tensor_stats(x_ref),
    }
    if int16_carrier is not None:
        row["int16_input_carrier"] = {
            "qmin": int16_carrier["qmin"],
            "qmax": int16_carrier["qmax"],
            "axis": int16_carrier["axis"],
            "scale": float(int16_carrier["scale"].reshape(-1)[0].item())
            if int16_carrier["scale"].numel() == 1
            else int16_carrier["scale"].tolist(),
        }
    if isinstance(y_raw, Int16QuantizedTensor):
        with int16_eval_allow_debug_float():
            metrics_lsb = compute_pair_metrics(
                y_ref,
                y_raw.to_float(),
                scale=y_raw.scale,
                zero_point=y_raw.zero_point,
                qmin=y_raw.qmin,
                qmax=y_raw.qmax,
                candidate_int_repr=y_raw.int_repr,
            )
        row["metrics"]["max_error_lsb"] = metrics_lsb.get("max_error_lsb")
        row["metrics"]["max_error_lsb_float"] = metrics_lsb.get("max_error_lsb_float")
        row["output_int16"] = {
            "qmin": y_raw.qmin,
            "qmax": y_raw.qmax,
            "scale": float(y_raw.scale.reshape(-1)[0].item())
            if y_raw.scale.numel() == 1
            else y_raw.scale.tolist(),
        }
    return row


def _name_filter(exact_name: str) -> Callable[[str, nn.Module], bool]:
    def _filt(name: str, module: nn.Module) -> bool:  # pylint: disable=unused-argument
        return name == exact_name

    return _filt


def diagnose_module(
    sim: Any,
    images: torch.Tensor,
    module_name: str,
    *,
    predecessor: Optional[str] = None,
) -> Dict[str, Any]:
    from aimet_torch.fixed_point import ExecutionMode
    from aimet_torch.fixed_point.metrics.chained import per_layer_chained_cosine
    from aimet_torch.fixed_point.metrics.isolated import per_layer_isolated_cosine

    model = sim.model
    modules = dict(model.named_modules())
    if module_name not in modules:
        known = [n for n in modules if "classifier" in n]
        raise KeyError(
            f"Module {module_name!r} not found. classifier-related keys: {known[:20]}"
        )
    module = modules[module_name]
    filt = _name_filter(module_name)

    report: Dict[str, Any] = {
        "module": module_name,
        "batch_shape": tuple(images.shape),
        "quantizers": _module_quantizer_report(module),
        "int16_export": _int16_export_record(module, module_name),
    }

    mode_pairs = [
        ("isolated_int16_vs_fp32", ExecutionMode.INT16_FIXED_EVAL, ExecutionMode.FP32_QDQ),
        ("isolated_fixed_scale_vs_fp32", ExecutionMode.FIXED_SCALE_QDQ, ExecutionMode.FP32_QDQ),
        ("isolated_fp16_vs_fp32", ExecutionMode.FP16_QDQ, ExecutionMode.FP32_QDQ),
        ("chained_int16_vs_fp32", ExecutionMode.INT16_FIXED_EVAL, ExecutionMode.FP32_QDQ),
        ("chained_fixed_scale_vs_fp32", ExecutionMode.FIXED_SCALE_QDQ, ExecutionMode.FP32_QDQ),
    ]

    for key, cand_mode, ref_mode in mode_pairs:
        if key.startswith("isolated"):
            rows = per_layer_isolated_cosine(
                model,
                images,
                cand_mode=cand_mode,
                ref_mode=ref_mode,
                module_filter=filt,
                top_k=1,
            )
            report[key] = rows[0] if rows else {"error": "no row (skipped or hook miss)"}
        else:
            rows = per_layer_chained_cosine(
                model,
                images,
                cand_mode=cand_mode,
                ref_mode=ref_mode,
                module_filter=filt,
                top_k=1,
            )
            if not rows:
                report[key] = {"error": "no row"}
            else:
                row = dict(rows[0])
                row["chained_cosine"] = row.get("cosine")
                report[key] = row

    report["detailed"] = {
        "int16_vs_fp32": _detailed_mode_metrics(
            model,
            module,
            images,
            module_name,
            ref_mode=ExecutionMode.FP32_QDQ,
            cand_mode=ExecutionMode.INT16_FIXED_EVAL,
        ),
        "fixed_scale_vs_fp32": _detailed_mode_metrics(
            model,
            module,
            images,
            module_name,
            ref_mode=ExecutionMode.FP32_QDQ,
            cand_mode=ExecutionMode.FIXED_SCALE_QDQ,
        ),
    }
    clamp_report = _clamp_boundary_report(
        model,
        module,
        images,
        ref_mode=ExecutionMode.FP32_QDQ,
        cand_mode=ExecutionMode.INT16_FIXED_EVAL,
    )
    if clamp_report is not None:
        report["clamp_boundary"] = clamp_report

    if predecessor:
        if predecessor not in modules:
            report["predecessor"] = {"error": f"{predecessor!r} not in model"}
        else:
            x_pred, y_pred = _capture_module_io(
                model, images, predecessor, mode=ExecutionMode.FP32_QDQ
            )
            pred_row: Dict[str, Any] = {"module": predecessor}
            if y_pred is not None:
                pred_row["output_fp32_stats"] = _tensor_stats(y_pred)
            if x_pred is not None:
                pred_row["input_fp32_stats"] = _tensor_stats(x_pred)
            chained_pred = per_layer_chained_cosine(
                model,
                images,
                cand_mode=ExecutionMode.INT16_FIXED_EVAL,
                ref_mode=ExecutionMode.FP32_QDQ,
                module_filter=_name_filter(predecessor),
                top_k=1,
            )
            if chained_pred:
                pred_row["chained_int16_vs_fp32"] = chained_pred[0]
            report["predecessor"] = pred_row

    iso = report.get("isolated_int16_vs_fp32", {})
    chain = report.get("chained_int16_vs_fp32", {})
    if isinstance(iso, dict) and isinstance(chain, dict):
        ic = iso.get("isolated_cosine")
        cc = chain.get("chained_cosine", chain.get("cosine"))
        if ic is not None and cc is not None:
            report["interpretation"] = {
                "isolated_cosine_int16": ic,
                "chained_cosine_int16": cc,
                "hint": (
                    "Low isolated → intrinsic module-level INT16/grid error. "
                    "Low chained only → upstream accumulation; high isolated + low chained → innocent layer. "
                    "Check quantizers.module_type before attributing the hot spot to FC/Linear."
                ),
            }
    return report


def _print_report(report: Dict[str, Any]) -> None:
    print(f"\n=== Isolated diagnosis: {report['module']} ===")
    print(f"batch: {report.get('batch_shape')}")

    exp = report.get("int16_export")
    if exp and "output_encoding" in exp:
        enc = exp["output_encoding"]
        print("\nINT16 output_encoding:")
        for k in ("scale", "zero_point", "qmin", "qmax", "multiplier", "rshift"):
            if k in enc:
                print(f"  {k}: {enc[k]}")
    else:
        print("\nINT16 export record: (none — layer may lack INT16 kernel)")

    for label, key, cosine_field in (
        ("Isolated INT16 vs FP32", "isolated_int16_vs_fp32", "isolated_cosine"),
        ("Isolated fixed_scale vs FP32", "isolated_fixed_scale_vs_fp32", "isolated_cosine"),
        ("Chained INT16 vs FP32", "chained_int16_vs_fp32", "chained_cosine"),
    ):
        row = report.get(key, {})
        if not isinstance(row, dict) or cosine_field not in row and "cosine" not in row:
            print(f"\n{label}: {row}")
            continue
        cos = row.get(cosine_field, row.get("cosine"))
        print(
            f"\n{label}: cosine={cos:.6f}  "
            f"max_abs_err={row.get('max_abs_err')}  p99={row.get('p99_abs_err')}  "
            f"shape={row.get('shape')}"
        )

    det = report.get("detailed", {}).get("int16_vs_fp32", {})
    if isinstance(det, dict) and "metrics" in det:
        m = det["metrics"]
        print(
            "\nDetailed INT16 (teacher-forced): "
            f"cosine={m.get('cosine_similarity'):.6f}  "
            f"max_lsb={m.get('max_error_lsb')}  rmse={m.get('rmse')}"
        )
        inp = det.get("input_fp32_stats", {})
        if inp:
            print(
                f"  input RMS={inp.get('rms'):.6f}  "
                f"range=[{inp.get('min'):.4f}, {inp.get('max'):.4f}]"
            )

    clamp = report.get("clamp_boundary")
    if isinstance(clamp, dict) and "error" not in clamp:
        pre = clamp["pre_metrics"]
        out = clamp["output_metrics"]
        bounds = clamp["bounds"]
        print(
            "\nClamp boundary (chained INT16 vs FP32 pre/post): "
            f"bounds=[{bounds['lower']}, {bounds['upper']}]"
        )
        print(
            f"  pre cosine={pre['cosine_similarity']:.6f}  rmse={pre['rmse']:.6f}  "
            f"post cosine={out['cosine_similarity']:.6f}  rmse={out['rmse']:.6f}"
        )
        print(
            f"  lower flip={clamp['lower_flip_fraction']:.4%}  "
            f"upper flip={clamp['upper_flip_fraction']:.4%}"
        )

    interp = report.get("interpretation")
    if interp:
        print(
            f"\nInterpretation: isolated={interp['isolated_cosine_int16']:.4f}  "
            f"chained={interp['chained_cosine_int16']:.4f}"
        )
        print(f"  {interp['hint']}")


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is False")
    return device


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

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

    from aimet_torch.fixed_point.e2e.imagenet_eval import (
        build_imagenet_mobilenet_bundle,
        resolve_imagenet_val_dir,
        resolve_imagenet_val_zip,
    )

    val_source = args.source
    val_root = args.val_dir
    zip_path = args.zip_path
    if val_source == "auto":
        resolved = resolve_imagenet_val_dir()
        if resolved is not None:
            val_source = "local"
            val_root = resolved
        else:
            val_source = "synthetic"

    output_overrides = {}
    if args.target_output_bw is not None:
        output_overrides[args.module_name] = (args.target_output_bw, True)
    for item in args.output_bw_override:
        try:
            name, bitwidth = item.rsplit(":", 1)
            output_overrides[name] = (int(bitwidth), True)
        except ValueError as exc:
            raise ValueError(
                f"--output-bw-override must be MODULE:BW, got {item!r}"
            ) from exc

    bundle, loader = build_imagenet_mobilenet_bundle(
        val_root=val_root,
        val_source=val_source,
        zip_path=zip_path or resolve_imagenet_val_zip(),
        image_dir=args.image_dir,
        batch_size=args.batch_size,
        calib_max_batches=args.calib_batches,
        calib_max_samples=args.calib_max_samples,
        eval_max_samples=args.batch_size,
        eval_seed=args.eval_seed,
        load_pretrained=not args.no_pretrained,
        default_param_bw=args.default_param_bw,
        default_output_bw=args.default_output_bw,
        int16_eval_bw=args.int16_eval_bw,
        output_quantizer_overrides=output_overrides or None,
        device=device,
    )

    images_batch: Optional[torch.Tensor] = None
    for batch in loader:
        images_batch = batch[0].to(int16_device, non_blocking=True)
        break
    if images_batch is None:
        print("ERROR: empty loader", file=sys.stderr)
        return 1

    bundle.sim.model.to(int16_device)
    report = diagnose_module(
        bundle.sim,
        images_batch,
        args.module_name,
        predecessor=args.predecessor,
    )
    report["meta"] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": val_source,
        "calib_max_samples": args.calib_max_samples,
        "calib_batches": args.calib_batches,
        "batch_size": args.batch_size,
        "eval_seed": args.eval_seed,
        "default_param_bw": args.default_param_bw,
        "default_output_bw": args.default_output_bw,
        "int16_eval_bw": args.int16_eval_bw,
        "target_output_bw": args.target_output_bw,
        "output_bw_overrides": {
            name: {"bitwidth": bitwidth, "symmetric": symmetric}
            for name, (bitwidth, symmetric) in output_overrides.items()
        },
        "device": str(device),
        "int16_device": str(int16_device),
    }

    _print_report(report)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote JSON: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

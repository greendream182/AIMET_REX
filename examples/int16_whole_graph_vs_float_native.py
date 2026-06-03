#!/usr/bin/env python3
"""整图 Top-1 vs float_native（path B）：PTQ + CLZ encoding fix 三档 QDQ 对比。

基准：未包装 MRNN（``model_fp.pth``），与单算子脚本的 float_native Top-1 口径一致。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONHASHSEED", "42")

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_QG = os.path.abspath(os.path.join(_REPO, "..", "quant-gru-pytorch", "pytorch"))
if os.path.isdir(_QG) and _QG not in sys.path:
    sys.path.insert(0, _QG)

import soundfile as sf
import torch
import torchaudio

_DATA = "/home/llq/workspace/data/speech_commands"
if not os.path.isdir(_DATA):
    _DATA = "/mnt/data8t/share/datasets/speech_commands/SpeechCommands/speech_commands_v0.02"


def _soundfile_load(path, *args, **kwargs):
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    return torch.from_numpy(data.T.copy()), sr


torchaudio.load = _soundfile_load

import quick_start as qs  # noqa: E402
import int16_single_op_vs_float_native as iso  # noqa: E402
from aimet_torch.fixed_point import ExecutionMode, quant_execution_mode  # noqa: E402


def _eval_modes(model, loader, device) -> dict[str, float | str]:
    out: dict[str, float | str] = {}
    for mode in (
        ExecutionMode.FP32_QDQ,
        ExecutionMode.FP16_QDQ,
        ExecutionMode.FIXED_SCALE_QDQ,
    ):
        try:
            with quant_execution_mode(mode):
                out[mode.value] = qs.evaluate(model, loader, device)
        except Exception as exc:  # noqa: BLE001
            out[mode.value] = f"FAILED: {exc}"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="MRNN whole-graph vs float_native")
    parser.add_argument("--data-root", default=_DATA)
    parser.add_argument("--max-calib-batches", type=int, default=100)
    parser.add_argument(
        "--skip-qat",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="跳过 QAT（默认）；用 --no-skip-qat 启用 QAT",
    )
    parser.add_argument("--qat-epochs", type=int, default=1)
    parser.add_argument(
        "--qat-max-batches",
        type=int,
        default=None,
        help="每 epoch 最多训练 batch 数（默认全量 train）",
    )
    parser.add_argument(
        "--clz-encoding-fix",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="§2.3 sign/reciprocal/power_2 encoding（默认开）",
    )
    parser.add_argument(
        "--clz-post-qat",
        action="store_true",
        help="reciprocal/power_2 在 QAT 之后再应用（避免 QAT 与手工 encoding 冲突）",
    )
    parser.add_argument(
        "--apply-po2",
        action="store_true",
        help="校准后 apply_power_of_2_workflow（默认关；Ada200 主线用 M,rshift 即可）",
    )
    parser.add_argument(
        "--no-clz-sign-bypass",
        action="store_true",
        help="CLZ fix 时不 bypass sign input Q（默认 bypass）",
    )
    parser.add_argument("--qat-lr", type=float, default=None)
    parser.add_argument(
        "--no-qat-restore-best",
        action="store_true",
        help="QAT 不恢复 val 最优 checkpoint（观察 QAT 是否真在学习）",
    )
    parser.add_argument(
        "--bitwidth-config",
        type=Path,
        default=None,
        help="mixed-precision JSON（默认 quick_start_full_quant.json，8bit）；"
             "推荐 config/pc1_hypot_16bit.json 修复 frontend 主损失",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "int16_whole_graph_vs_float_native.json",
    )
    args = parser.parse_args()

    qs.DATA_ROOT = args.data_root
    qs.FP_EPOCHS = 0
    qs.MAX_CALIB_BATCHES = args.max_calib_batches
    qs.set_seed(qs.SEED)
    qs.setup_audio_backend()
    device = qs.DEVICE

    print("=" * 72)
    print("MRNN 整图 vs float_native（path B）")
    print(f"  apply_po2={args.apply_po2}  clz_fix={args.clz_encoding_fix}  "
          f"clz_post_qat={args.clz_post_qat}  skip_qat={args.skip_qat}")
    print("=" * 72)

    loaders = qs.build_dataloaders(qs.DATA_ROOT)
    sd = torch.load(qs.FP_MODEL_PATH, map_location=device, weights_only=False)
    state = sd.get("model", sd) if isinstance(sd, dict) else sd

    fp = qs.MRNN(output_dim=qs.NUM_CLASSES).to(device)
    fp.load_state_dict(state, strict=False)
    fp.eval()

    t0 = time.time()
    float_native_top1 = qs.evaluate(fp, loaders["test"], device)
    print(f"float_native Top-1: {float_native_top1 * 100:.2f}%  ({time.time() - t0:.1f}s)")

    t0 = time.time()
    sim, _ = iso.build_sim(
        fp,
        loaders,
        device,
        skip_qat=args.skip_qat,
        qat_epochs=args.qat_epochs,
        max_calib=args.max_calib_batches,
        bitwidth_config=args.bitwidth_config,
        clz_encoding_fix=args.clz_encoding_fix,
        clz_post_qat=args.clz_post_qat,
        apply_po2=args.apply_po2,
        clz_sign_bypass=not args.no_clz_sign_bypass,
        qat_val_loader=loaders["val"],
        qat_max_batches=args.qat_max_batches,
        qat_lr=args.qat_lr,
        qat_restore_best=not args.no_qat_restore_best,
    )
    print(f"sim 构建+校准{' (+QAT)' if not args.skip_qat else ''}: {time.time() - t0:.1f}s")

    t0 = time.time()
    mode_acc = _eval_modes(sim.model, loaders["test"], device)
    print(f"三档 QDQ 评估: {time.time() - t0:.1f}s")
    for k, v in mode_acc.items():
        if isinstance(v, float):
            delta_pp = (v - float_native_top1) * 100
            print(f"  {k:22s} {v * 100:7.2f}%  (Δ vs float_native {delta_pp:+.2f} pp)")
        else:
            print(f"  {k:22s} {v}")

    report = {
        "float_native_top1": float_native_top1,
        "bitwidth_config": str(args.bitwidth_config) if args.bitwidth_config else None,
        "apply_po2": args.apply_po2,
        "clz_sign_bypass": not args.no_clz_sign_bypass if args.clz_encoding_fix else None,
        "qat_lr": args.qat_lr,
        "qat_restore_best": not args.no_qat_restore_best,
        "clz_encoding_fix": args.clz_encoding_fix,
        "clz_post_qat": args.clz_post_qat,
        "clz_encoding_fix_stats": getattr(sim, "_clz_encoding_fix_stats", None),
        "qat_stats": getattr(sim, "_qat_stats", None),
        "skip_qat": args.skip_qat,
        "qat_max_batches": args.qat_max_batches,
        "max_calib_batches": args.max_calib_batches,
        "mode_top1": mode_acc,
        "delta_pp_vs_float_native": {
            k: (v - float_native_top1) * 100
            for k, v in mode_acc.items()
            if isinstance(v, float)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n报告: {args.output}")


if __name__ == "__main__":
    main()

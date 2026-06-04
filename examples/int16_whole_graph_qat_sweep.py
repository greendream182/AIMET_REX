#!/usr/bin/env python3
"""整图 QAT 超参 / sign-bypass 扫参（无 Po2，CLZ + clz_post_qat 默认）。"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
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


def _soundfile_load(path, *args, **kwargs):
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    return torch.from_numpy(data.T.copy()), sr


torchaudio.load = _soundfile_load

import quick_start as qs  # noqa: E402
import int16_single_op_vs_float_native as iso  # noqa: E402
import int16_whole_graph_vs_float_native as wg  # noqa: E402

_DATA = "/home/llq/workspace/data/speech_commands"
if not os.path.isdir(_DATA):
    _DATA = "/mnt/data8t/share/datasets/speech_commands/SpeechCommands/speech_commands_v0.02"


@dataclass
class SweepCase:
    name: str
    clz_sign_bypass: bool = True
    qat_lr: float = 1e-4
    qat_epochs: int = 1
    qat_max_batches: int | None = None
    qat_restore_best: bool = True


DEFAULT_CASES: tuple[SweepCase, ...] = (
    SweepCase("baseline_sign_bypass_lr1e4_e1", clz_sign_bypass=True),
    SweepCase("no_sign_bypass_lr1e4_e1", clz_sign_bypass=False),
    SweepCase("no_sign_bypass_lr1e5_e1", clz_sign_bypass=False, qat_lr=1e-5),
    SweepCase("no_sign_bypass_lr5e5_e1", clz_sign_bypass=False, qat_lr=5e-5),
    SweepCase("no_sign_bypass_lr1e4_e3", clz_sign_bypass=False, qat_epochs=3),
    SweepCase("no_sign_bypass_lr1e4_e1_no_ckpt", clz_sign_bypass=False, qat_restore_best=False),
    SweepCase("no_sign_bypass_lr1e4_e1_b200", clz_sign_bypass=False, qat_max_batches=200),
    SweepCase("sign_bypass_lr1e5_e3", clz_sign_bypass=True, qat_lr=1e-5, qat_epochs=3),
)


def run_case(
    fp,
    loaders,
    device,
    case: SweepCase,
    *,
    max_calib: int,
) -> dict:
    t0 = time.time()
    sim, _ = iso.build_sim(
        fp,
        loaders,
        device,
        skip_qat=False,
        qat_epochs=case.qat_epochs,
        max_calib=max_calib,
        clz_encoding_fix=True,
        clz_post_qat=True,
        apply_po2=False,
        clz_sign_bypass=case.clz_sign_bypass,
        qat_val_loader=loaders["val"],
        qat_max_batches=case.qat_max_batches,
        qat_lr=case.qat_lr,
        qat_restore_best=case.qat_restore_best,
    )
    build_s = time.time() - t0

    t0 = time.time()
    mode_acc = wg._eval_modes(sim.model, loaders["test"], device)
    eval_s = time.time() - t0

    row = {
        "name": case.name,
        **asdict(case),
        "build_s": round(build_s, 1),
        "eval_s": round(eval_s, 1),
        "mode_top1": mode_acc,
        "qat_stats": getattr(sim, "_qat_stats", None),
    }
    fp32 = mode_acc.get("fp32_qdq")
    if isinstance(fp32, float):
        row["fp32_qdq_pct"] = round(fp32 * 100, 2)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="MRNN whole-graph QAT sweep")
    parser.add_argument("--data-root", default=_DATA)
    parser.add_argument("--max-calib-batches", type=int, default=100)
    parser.add_argument(
        "--cases",
        default="",
        help="逗号分隔 case name；空=跑 DEFAULT_CASES 全套",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "int16_whole_graph_qat_sweep.json",
    )
    args = parser.parse_args()

    qs.DATA_ROOT = args.data_root
    qs.FP_EPOCHS = 0
    qs.MAX_CALIB_BATCHES = args.max_calib_batches
    qs.set_seed(qs.SEED)
    qs.setup_audio_backend()
    device = qs.DEVICE

    loaders = qs.build_dataloaders(qs.DATA_ROOT)
    sd = torch.load(qs.FP_MODEL_PATH, map_location=device, weights_only=False)
    state = sd.get("model", sd) if isinstance(sd, dict) else sd
    fp = qs.MRNN(output_dim=qs.NUM_CLASSES).to(device)
    fp.load_state_dict(state, strict=False)
    fp.eval()
    float_native = qs.evaluate(fp, loaders["test"], device)

    if args.cases.strip():
        names = {n.strip() for n in args.cases.split(",") if n.strip()}
        cases = [c for c in DEFAULT_CASES if c.name in names]
        missing = names - {c.name for c in cases}
        if missing:
            raise SystemExit(f"unknown case names: {sorted(missing)}")
    else:
        cases = list(DEFAULT_CASES)

    print("=" * 72)
    print(f"QAT sweep  float_native={float_native * 100:.2f}%  cases={len(cases)}")
    print("  apply_po2=False  clz_post_qat=True")
    print("=" * 72)

    results: list[dict] = []
    for idx, case in enumerate(cases, 1):
        print(f"\n[{idx}/{len(cases)}] {case.name}  {asdict(case)}")
        row = run_case(fp, loaders, device, case, max_calib=args.max_calib_batches)
        results.append(row)
        fp32 = row.get("fp32_qdq_pct", row["mode_top1"].get("fp32_qdq"))
        qat = row.get("qat_stats") or {}
        loss_key = f"train_loss_epoch_{case.qat_epochs}"
        print(
            f"  fp32_qdq={fp32}%  val_before={qat.get('val_before_qat', 0) * 100:.2f}%  "
            f"val_end={qat.get(f'val_epoch_{case.qat_epochs}', 0) * 100:.2f}%  "
            f"loss={qat.get(loss_key, float('nan')):.4f}  "
            f"restore={qat.get('val_restored')}"
        )

    report = {
        "float_native_top1": float_native,
        "max_calib_batches": args.max_calib_batches,
        "apply_po2": False,
        "clz_encoding_fix": True,
        "clz_post_qat": True,
        "results": results,
        "ranking_fp32_qdq": sorted(
            [
                {"name": r["name"], "fp32_qdq": r["mode_top1"].get("fp32_qdq")}
                for r in results
                if isinstance(r["mode_top1"].get("fp32_qdq"), float)
            ],
            key=lambda x: x["fp32_qdq"],
            reverse=True,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n报告: {args.output}")
    print("Top fp32_qdq:")
    for item in report["ranking_fp32_qdq"][:5]:
        print(f"  {item['name']:40s} {item['fp32_qdq'] * 100:.2f}%")


if __name__ == "__main__":
    main()

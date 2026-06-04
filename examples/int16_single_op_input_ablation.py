#!/usr/bin/env python3
"""单算子输入来源对比：真实 calib vs 随机 vs 多 batch 统计。

复用 ``int16_single_op_vs_float_native`` 的 teacher-forced 流程；默认只评
``fp32_qdq``（最快），可选 ``--all-modes``。
"""
from __future__ import annotations

import argparse
import json
import math
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
from aimet_torch.fixed_point import ExecutionMode  # noqa: E402

import int16_single_op_vs_float_native as iso  # noqa: E402


def _summarize_rows(rows: list[dict], min_cosine: float) -> dict:
    with_cos = [r for r in rows if "cosine" in r and not _is_bad_cos(r["cosine"])]
    degenerate = [r for r in rows if "cosine" in r and _is_bad_cos(r["cosine"])]
    return {
        "pass_strict": sum(1 for r in with_cos if r["cosine"] >= min_cosine),
        "pass_0_999": sum(1 for r in with_cos if r["cosine"] >= 0.999),
        "pass_0_99": sum(1 for r in with_cos if r["cosine"] >= 0.99),
        "n_with_cosine": len(with_cos),
        "n_degenerate": len(degenerate),
        "n_skip": sum(1 for r in rows if r.get("status") == "SKIP"),
        "n_error": sum(1 for r in rows if r.get("status") == "ERROR"),
        "mean_cosine": (
            sum(r["cosine"] for r in with_cos) / len(with_cos) if with_cos else None
        ),
        "min_cosine": min((r["cosine"] for r in with_cos), default=None),
        "worst3": sorted(with_cos, key=lambda r: r["cosine"])[:3],
    }


def _is_bad_cos(c: float) -> bool:
    return c is None or (isinstance(c, float) and (math.isnan(c) or math.isinf(c)))


def _make_input(
    source: str,
    *,
    loaders,
    device: torch.device,
    batch_size: int,
    batch_idx: int,
    seed: int,
    shape: tuple[int, ...] | None = None,
) -> tuple[torch.Tensor, dict]:
    meta = {"source": source, "batch_size": batch_size}
    if source.startswith("real_calib"):
        idx = batch_idx if source == "real_calib" else int(source.split("_")[-1])
        calib_iter = iter(loaders["calib"])
        x = None
        for i, (bx, _) in enumerate(calib_iter):
            if i == idx:
                x = bx[:batch_size].to(device)
                meta["batch_idx"] = idx
                break
        if x is None:
            raise ValueError(f"calib batch_idx {idx} out of range")
        meta["input_stats"] = _tensor_stats(x)
        return x, meta

    if shape is None:
        bx, _ = next(iter(loaders["calib"]))
        shape = tuple(bx[:batch_size].shape)

    gen = torch.Generator()
    gen.manual_seed(seed)
    if source == "random_gaussian":
        x = torch.randn(shape, generator=gen, dtype=torch.float32).to(device)
    elif source == "random_uniform":
        x = (torch.rand(shape, generator=gen) * 2 - 1).to(torch.float32).to(device)
    elif source == "random_small":
        x = (torch.rand(shape, generator=gen) * 0.1 - 0.05).to(torch.float32).to(device)
    else:
        raise ValueError(f"unknown source {source!r}")
    meta["shape"] = list(shape)
    meta["seed"] = seed
    meta["input_stats"] = _tensor_stats(x)
    return x, meta


def _tensor_stats(x: torch.Tensor) -> dict:
    xf = x.detach().float()
    return {
        "min": float(xf.min().item()),
        "max": float(xf.max().item()),
        "mean": float(xf.mean().item()),
        "std": float(xf.std().item()),
    }


def _multi_batch_mean_cosine(
    sim_model,
    prepared_float,
    loaders,
    device,
    *,
    n_batches: int,
    batch_size: int,
    mode: ExecutionMode,
    min_cosine: float,
) -> dict:
    """对前 n 个 calib batch 逐 batch 评 fp32，再对每 module 取 cosine 均值。"""
    per_module: dict[str, list[float]] = {}
    batch_summaries = []
    for bi in range(n_batches):
        x, meta = _make_input(
            "real_calib", loaders=loaders, device=device,
            batch_size=batch_size, batch_idx=bi, seed=0,
        )
        rows = iso.isolated_vs_float_native(
            sim_model, prepared_float, x, cand_mode=mode, min_cosine=min_cosine,
        )
        batch_summaries.append(_summarize_rows(rows, min_cosine))
        for r in rows:
            if "cosine" not in r or _is_bad_cos(r["cosine"]):
                continue
            per_module.setdefault(r["module"], []).append(r["cosine"])

    agg_rows = []
    for mod, cosines in sorted(per_module.items()):
        agg_rows.append({
            "module": mod,
            "mean_cosine": sum(cosines) / len(cosines),
            "min_cosine": min(cosines),
            "max_cosine": max(cosines),
            "n_batches": len(cosines),
        })
    agg_rows.sort(key=lambda r: r["mean_cosine"])
    return {
        "n_batches": n_batches,
        "per_batch_summary": batch_summaries,
        "worst5_mean": agg_rows[:5],
        "best5_mean": agg_rows[-5:],
        "mean_of_means": (
            sum(r["mean_cosine"] for r in agg_rows) / len(agg_rows) if agg_rows else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=_DATA)
    parser.add_argument("--skip-qat", action="store_true", default=True)
    parser.add_argument("--max-calib-batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--min-cosine", type=float, default=iso.MIN_COSINE_DEFAULT)
    parser.add_argument("--all-modes", action="store_true")
    parser.add_argument("--multi-batches", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent
        / "output"
        / "int16_single_op_input_ablation.json",
    )
    args = parser.parse_args()

    qs.DATA_ROOT = args.data_root
    qs.FP_EPOCHS = 0
    qs.MAX_CALIB_BATCHES = args.max_calib_batches
    qs.set_seed(qs.SEED)
    qs.setup_audio_backend()
    device = qs.DEVICE

    print("=" * 72)
    print("单算子输入来源 ablation（teacher-forced vs prepared float）")
    print("=" * 72)

    loaders = qs.build_dataloaders(qs.DATA_ROOT)
    sd = torch.load(qs.FP_MODEL_PATH, map_location=device, weights_only=False)
    state = sd.get("model", sd) if isinstance(sd, dict) else sd
    fp = qs.MRNN(output_dim=qs.NUM_CLASSES).to(device)
    fp.load_state_dict(state, strict=False)

    t0 = time.time()
    sim, prepared_float = iso.build_sim(
        fp, loaders, device,
        skip_qat=args.skip_qat,
        qat_epochs=1,
        max_calib=args.max_calib_batches,
    )
    print(f"sim ready in {time.time() - t0:.1f}s (skip_qat={args.skip_qat})")

    modes = list(iso.CAND_MODES) if args.all_modes else [ExecutionMode.FP32_QDQ]
    sources = [
        "real_calib",
        "real_calib_5",
        "random_gaussian",
        "random_uniform",
        "random_small",
    ]

    bx, _ = next(iter(loaders["calib"]))
    shape = tuple(bx[: args.batch_size].shape)

    report: dict = {
        "note": (
            "每层输入=该来源整网 forward 时 hook 到的激活；"
            "real_calib 来自 SpeechCommands calib loader，非随机。"
        ),
        "tensor_shape": list(shape),
        "modes": [m.value for m in modes],
        "sources": {},
        "multi_batch_aggregate": {},
    }

    for src in sources:
        src_key = src if src != "real_calib_5" else "real_calib_batch5"
        x, meta = _make_input(
            src,
            loaders=loaders,
            device=device,
            batch_size=args.batch_size,
            batch_idx=0 if src == "real_calib" else 5,
            seed=qs.SEED + 17,
            shape=shape,
        )
        report["sources"][src_key] = {"input_meta": meta, "modes": {}}
        print(f"\n### input={src_key} stats={meta.get('input_stats')}")
        for mode in modes:
            rows = iso.isolated_vs_float_native(
                sim.model, prepared_float, x, cand_mode=mode, min_cosine=args.min_cosine,
            )
            sm = _summarize_rows(rows, args.min_cosine)
            report["sources"][src_key]["modes"][mode.value] = sm
            print(
                f"  {mode.value}: pass@{args.min_cosine}={sm['pass_strict']}/"
                f"{sm['n_with_cosine']} mean={sm['mean_cosine']:.4f} "
                f"min={sm['min_cosine']:.4f} degenerate={sm['n_degenerate']}"
            )
            for w in sm["worst3"]:
                print(f"    worst {w['module']}: {w['cosine']:.4f}")

    if args.multi_batches > 0:
        print(f"\n### multi-batch mean cosine ({args.multi_batches} calib batches)")
        agg = _multi_batch_mean_cosine(
            sim.model,
            prepared_float,
            loaders,
            device,
            n_batches=args.multi_batches,
            batch_size=args.batch_size,
            mode=ExecutionMode.FP32_QDQ,
            min_cosine=args.min_cosine,
        )
        report["multi_batch_aggregate"] = agg
        print(f"  mean_of_means={agg['mean_of_means']:.4f}")
        for r in agg["worst5_mean"]:
            print(
                f"  worst {r['module']}: mean={r['mean_cosine']:.4f} "
                f"min={r['min_cosine']:.4f} over {r['n_batches']} batches"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n报告: {args.output}")


if __name__ == "__main__":
    main()

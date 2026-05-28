#!/usr/bin/env python3
# -*- mode: python -*-
"""MRNN SpeechCommands：FP32_QDQ vs INT16_FIXED_EVAL metric 对比。

在 ``quick_start.py`` 的 sim 构建/校准流程基础上，增加 INT16 评估。
用法（在 ``examples/`` 目录下）::

    export PYTHONPATH=/path/to/quant-gru-pytorch/pytorch:$PYTHONPATH
    python quick_start_int16_metric.py \\
        --data-root /home/llq/workspace/data/speech_commands \\
        --max-calib-batches 20 \\
        --max-eval-batches 100

完整训练请加 ``--fp-epochs 1``（耗时显著增加）。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

# quant_gru sibling checkout（与 tests/fixed_point/conftest 一致）
_QG = Path(__file__).resolve().parents[1].parent / "quant-gru-pytorch" / "pytorch"
if _QG.is_dir() and str(_QG) not in sys.path:
    sys.path.insert(0, str(_QG))

import aimet_torch.v2 as aimet  # noqa: E402
import aimet_torch.fixed_point.kernels  # noqa: F401, E402
from aimet_torch import model_preparer  # noqa: E402
from aimet_torch.fixed_point import (  # noqa: E402
    ExecutionMode,
    diagnose_int16_readiness,
    ensure_output_quantizers_for_int16_eval,
    quant_execution_mode,
)
from aimet_torch.utils_rx import apply_mixed_precision_bitwidth, apply_power_of_2_workflow  # noqa: E402
from aimet_torch.v2 import quantsim  # noqa: E402

from aimet_torch.quantizable_batchnorm import QuantizedQuantizableBatchNorm2d  # noqa: F401, E402
from quick_start import (  # noqa: E402
    BATCH_SIZE,
    BITWIDTH_CONFIG_FILE,
    CONFIG_FILE,
    DEFAULT_BW,
    DEVICE,
    FP_LR,
    MRNN,
    NUM_CLASSES,
    PERCENTILE_VALUE,
    QUANT_SCHEME,
    build_dataloaders,
    set_seed,
    setup_audio_backend,
    train_floating_point,
)


def evaluate_limited(model, loader, device, max_batches: int | None = None) -> float:
    """Top-1 精度；``max_batches`` 限制 batch 数以加速 smoke。"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            inputs, labels = inputs.to(device), labels.to(device)
            out = model(inputs)
            if hasattr(out, "to_float"):
                out = out.to_float()
            preds = out.max(1).indices
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
    if total == 0:
        raise RuntimeError("evaluate_limited: no samples evaluated")
    return correct / total


def _calib_fn(sim_model, loader, device, max_batches: int):
    def _run(m):
        with torch.no_grad():
            for idx, (x, _) in enumerate(loader):
                if idx >= max_batches:
                    break
                m(x.to(device))

    return _run


def build_sim(model: torch.nn.Module, dummy_input: torch.Tensor):
    prepared = model_preparer.prepare_model(model)
    sim = quantsim.QuantizationSimModel(
        prepared,
        dummy_input=dummy_input,
        quant_scheme=QUANT_SCHEME,
        config_file=str(CONFIG_FILE),
        default_output_bw=DEFAULT_BW,
        default_param_bw=DEFAULT_BW,
    )
    sim.set_percentile_value(PERCENTILE_VALUE)
    apply_mixed_precision_bitwidth(
        sim.model, config_file=str(BITWIDTH_CONFIG_FILE), verbose=False,
    )
    patched = ensure_output_quantizers_for_int16_eval(sim)
    print(f"ensure_output_quantizers_for_int16_eval: patched {len(patched)} slots")
    return sim


def _patch_torchaudio_with_soundfile() -> None:
    """torchaudio 2.10+ 默认走 torchcodec；SpeechCommands 用 soundfile 读 wav 即可。"""
    import soundfile as sf
    import torch
    import torchaudio

    def _load(path, frame_offset=0, num_frames=-1, normalize=True, channels_first=True, **kwargs):
        del frame_offset, num_frames, normalize, kwargs
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        tensor = torch.from_numpy(data.T.copy())
        if not channels_first:
            tensor = tensor.T
        return tensor, sr

    torchaudio.load = _load  # type: ignore[method-assign]


def main() -> None:
    parser = argparse.ArgumentParser(description="MRNN INT16 metric comparison")
    parser.add_argument(
        "--data-root",
        default=os.environ.get(
            "SPEECH_COMMANDS_ROOT",
            "/home/llq/workspace/data/speech_commands",
        ),
        help="SpeechCommands v0.02 根目录",
    )
    parser.add_argument("--fp-epochs", type=int, default=0, help="浮点预训练 epoch 数（0=跳过）")
    parser.add_argument("--max-calib-batches", type=int, default=20)
    parser.add_argument("--max-eval-batches", type=int, default=None, help="限制 test batch 数；None=全量")
    parser.add_argument("--skip-po2", action="store_true", help="跳过 Power-of-2 scale 对齐")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["fp32_qdq", "int16_fixed_eval"],
        choices=[m.value for m in ExecutionMode],
        help="要评估的 ExecutionMode 列表",
    )
    args = parser.parse_args()

    set_seed()
    _patch_torchaudio_with_soundfile()
    setup_audio_backend()

    data_root = Path(args.data_root)
    if not data_root.is_dir():
        sys.exit(f"Data root not found: {data_root}")

    print("=" * 70)
    print("MRNN SpeechCommands — INT16 metric 对比")
    print("=" * 70)
    print(f"设备:       {DEVICE}")
    print(f"数据根目录: {data_root}")
    print(f"校准 batch: {args.max_calib_batches}")
    print(f"评估 batch: {args.max_eval_batches or '全量 test'}")
    print(f"模式:       {args.modes}")
    print("说明:       全图 decomposed INT16（STFT/BN/PowerCompress/Hypot/CLN）")

    loaders = build_dataloaders(str(data_root))
    model = MRNN(output_dim=NUM_CLASSES).to(DEVICE)

    if args.fp_epochs > 0:
        t0 = time.time()
        fp_acc = train_floating_point(
            model,
            loaders["train"],
            loaders["test"],
            DEVICE,
            epochs=args.fp_epochs,
            lr=FP_LR,
        )
        print(f"浮点训练 ({args.fp_epochs} epoch) 精度: {fp_acc * 100:.2f}%  耗时 {time.time() - t0:.1f}s")
    else:
        print("跳过浮点训练（随机初始化；绝对精度偏低，但 mode 间 Δ 仍有参考价值）")

    sample_input, _ = next(iter(loaders["train"]))
    dummy_input = sample_input.to(DEVICE)

    sim = build_sim(model, dummy_input)
    sim.model.to(DEVICE).eval()

    calib = _calib_fn(sim.model, loaders["calib"], DEVICE, args.max_calib_batches)
    t0 = time.time()
    with torch.no_grad(), aimet.nn.compute_encodings(sim.model):
        calib(sim.model)
    print(f"compute_encodings 完成，耗时 {time.time() - t0:.1f}s")

    if not args.skip_po2:
        apply_power_of_2_workflow(
            sim.model,
            method="round",
            tolerance=0.02,
            align_bias_scale=True,
            verbose=False,
        )
        print("apply_power_of_2_workflow 完成")

    report = diagnose_int16_readiness(sim)
    for key, items in report.items():
        if items:
            print(f"diagnose_int16_readiness[{key}]: {len(items)} 项，示例 {items[:3]}")
    if not any(report.values()):
        print("diagnose_int16_readiness: 全部通过 ✅")

    results: dict[str, float | str] = {}
    for mode_str in args.modes:
        mode = ExecutionMode(mode_str)
        label = mode.value
        print(f"\n--- 评估 {label} ---")
        try:
            with quant_execution_mode(mode):
                acc = evaluate_limited(
                    sim.model,
                    loaders["test"],
                    DEVICE,
                    max_batches=args.max_eval_batches,
                )
            results[label] = acc
            print(f"{label}: {acc * 100:.2f}%")
        except Exception as exc:
            results[label] = f"FAILED: {exc}"
            print(f"{label}: FAILED — {exc}")

    print("\n" + "=" * 70)
    print("Metric 对比汇总")
    print("=" * 70)
    for label, acc in results.items():
        if isinstance(acc, float):
            print(f"  {label:22s} {acc * 100:7.2f}%")
        else:
            print(f"  {label:22s} {acc}")

    fp_ref = results.get("fp32_qdq")
    int16 = results.get("int16_fixed_eval")
    if isinstance(fp_ref, float) and isinstance(int16, float):
        delta_pp = (int16 - fp_ref) * 100
        print(f"\n  Δ(INT16 − FP32_QDQ) = {delta_pp:+.2f} pp")
        if abs(delta_pp) <= 0.5:
            print("  ✅ |Δ| ≤ 0.5 pp（R3 验收参考阈值）")
        else:
            print("  ⚠️  |Δ| > 0.5 pp（可能含未收敛权重或 reference kernel 误差）")
    print("=" * 70)


if __name__ == "__main__":
    main()

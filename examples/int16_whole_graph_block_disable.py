#!/usr/bin/env python3
"""整图 fp32_qdq block-level QDQ disable ablation。

目的：定位 prepared_float(95.56%) → fp32_qdq(67%) 的主要损失模块。

方法：以 CLZ + no Po2 + skip QAT 的 sim 为基线（fp32_qdq≈67%）；
对每个候选 block，临时**移除其内部所有 input/output/param quantizer**（不重新校准，
直接置 None），评 test fp32_qdq；评完恢复，下一组继续。

「disable=精度回升越多」⇒「该 block 是当前主要量化损失来源」。
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


def _soundfile_load(path, *args, **kwargs):
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    return torch.from_numpy(data.T.copy()), sr


torchaudio.load = _soundfile_load

import quick_start as qs  # noqa: E402
import int16_single_op_vs_float_native as iso  # noqa: E402
from aimet_torch.fixed_point import ExecutionMode, quant_execution_mode  # noqa: E402


# Block 候选：(scope_name, predicate_on_module_name)
BLOCKS: list[tuple[str, str]] = [
    ("frontend.trans", "trans."),
    ("frontend.power_compress_1", "power_compress_1."),
    ("frontend.pre_bn", "pre_bn."),
    ("frontend.hypot_fun", "hypot_fun."),
    ("frontend.fft2band", "fft2band."),
    ("frontend.power_compress_2", "power_compress_2."),
    ("frontend.clamps_top", "module_clamp"),
    ("conv_in", "conv_in"),
    ("freq_downs.0", "freq_downs.0"),
    ("enc_seqs.0", "enc_seqs.0"),
    ("freq_downs.1", "freq_downs.1"),
    ("enc_seqs.1", "enc_seqs.1"),
    ("freq_downs.2", "freq_downs.2"),
    ("neck_seqs.0", "neck_seqs.0"),
    ("neck_seqs.1", "neck_seqs.1"),
    ("head.fc0", "fc0"),
    ("head.module_mean_4", "module_mean_4"),
]

# Frontend 内部细分：power_compress_1 / hypot_fun 是当前主损失来源
FRONTEND_SUBBLOCKS: list[tuple[str, str]] = [
    ("pc1.module_sign", "power_compress_1.module_sign"),
    ("pc1.module_abs_1", "power_compress_1.module_abs_1"),
    ("pc1.module_sqrt", "power_compress_1.module_sqrt"),
    ("pc1.module_mul", "power_compress_1.module_mul"),
    ("hypot.module_square", "hypot_fun.module_square"),
    ("hypot.module_square_1", "hypot_fun.module_square_1"),
    ("hypot.module_add_1", "hypot_fun.module_add_1"),
    ("hypot.module_clamp_1", "hypot_fun.module_clamp_1"),
    ("hypot.module_sqrt_1", "hypot_fun.module_sqrt_1"),
]


# 子类别：把每个 RNN2D 内部更细分
SUBBLOCKS: list[tuple[str, str]] = [
    ("enc_seqs.0.rnn2d_bn", "enc_seqs.0.rnn2d_bn"),
    ("enc_seqs.0.cln", "enc_seqs.0.cln"),
    ("enc_seqs.0.seq_t", "enc_seqs.0.seq_t"),
    ("enc_seqs.0.conv_t", "enc_seqs.0.conv_t"),
    ("enc_seqs.0.module_mul_3", "enc_seqs.0.module_mul_3"),
    ("enc_seqs.1.rnn2d_bn", "enc_seqs.1.rnn2d_bn"),
    ("enc_seqs.1.cln", "enc_seqs.1.cln"),
    ("enc_seqs.1.seq_t", "enc_seqs.1.seq_t"),
    ("enc_seqs.1.conv_t", "enc_seqs.1.conv_t"),
    ("enc_seqs.1.module_mul_4", "enc_seqs.1.module_mul_4"),
    ("neck_seqs.0.rnn2d_bn", "neck_seqs.0.rnn2d_bn"),
    ("neck_seqs.0.cln", "neck_seqs.0.cln"),
    ("neck_seqs.0.seq_t", "neck_seqs.0.seq_t"),
    ("neck_seqs.0.conv_t", "neck_seqs.0.conv_t"),
    ("neck_seqs.0.module_mul_5", "neck_seqs.0.module_mul_5"),
    ("neck_seqs.1.rnn2d_bn", "neck_seqs.1.rnn2d_bn"),
    ("neck_seqs.1.cln", "neck_seqs.1.cln"),
    ("neck_seqs.1.seq_t", "neck_seqs.1.seq_t"),
    ("neck_seqs.1.conv_t", "neck_seqs.1.conv_t"),
    ("neck_seqs.1.module_mul_6", "neck_seqs.1.module_mul_6"),
]


def _name_matches(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".") or name.startswith(prefix)


def disable_block(sim_model, prefix: str):
    """暂存并禁用 prefix 命中的所有 input/output/param quantizer。返回 restore 回调。"""
    saved: list[tuple] = []  # (mod, kind, idx_or_key, original)

    for name, mod in sim_model.named_modules():
        if not _name_matches(name, prefix):
            continue
        iqs = getattr(mod, "input_quantizers", None)
        if iqs is not None:
            for i, q in enumerate(iqs):
                if q is not None:
                    saved.append((mod, "in", i, q))
                    iqs[i] = None
        oqs = getattr(mod, "output_quantizers", None)
        if oqs is not None:
            for i, q in enumerate(oqs):
                if q is not None:
                    saved.append((mod, "out", i, q))
                    oqs[i] = None
        pqs = getattr(mod, "param_quantizers", None)
        if pqs is not None and hasattr(pqs, "items"):
            for k, q in list(pqs.items()):
                if q is not None:
                    saved.append((mod, "param", k, q))
                    pqs[k] = None

    def restore():
        for mod, kind, key, q in saved:
            if kind == "in":
                mod.input_quantizers[key] = q
            elif kind == "out":
                mod.output_quantizers[key] = q
            else:
                mod.param_quantizers[key] = q

    return len(saved), restore


def main() -> None:
    parser = argparse.ArgumentParser(description="Whole-graph fp32_qdq block disable ablation")
    parser.add_argument("--data-root", default="/home/llq/workspace/data/speech_commands")
    parser.add_argument("--max-calib-batches", type=int, default=100)
    parser.add_argument("--max-eval-batches", type=int, default=None,
                        help="None=全 test；调小可加速 ablation")
    parser.add_argument("--include-subblocks", action="store_true",
                        help="附加 enc/neck RNN2D 内部子模块")
    parser.add_argument("--include-frontend-subblocks", action="store_true",
                        help="附加 power_compress_1 / hypot_fun 内部子算子")
    parser.add_argument("--apply-po2", action="store_true")
    parser.add_argument("--no-clz", action="store_true")
    parser.add_argument(
        "--mode",
        default="fp32_qdq",
        choices=["fp32_qdq", "fp16_qdq", "fixed_scale_qdq"],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "int16_whole_graph_block_disable.json",
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
    print(f"float_native: {float_native * 100:.2f}%")

    t0 = time.time()
    sim, _ = iso.build_sim(
        fp,
        loaders,
        device,
        skip_qat=True,
        qat_epochs=1,
        max_calib=args.max_calib_batches,
        clz_encoding_fix=not args.no_clz,
        clz_post_qat=False,
        apply_po2=args.apply_po2,
    )
    print(f"sim build+calib: {time.time() - t0:.1f}s")

    mode = ExecutionMode(args.mode)

    def eval_with_subset() -> float:
        loader = loaders["test"]
        if args.max_eval_batches is None:
            with quant_execution_mode(mode):
                return qs.evaluate(sim.model, loader, device)
        # 子集评估
        sim.model.eval()
        correct = total = 0
        with torch.no_grad(), quant_execution_mode(mode):
            for i, (x, y) in enumerate(loader):
                if i >= args.max_eval_batches:
                    break
                x, y = x.to(device), y.to(device)
                out = sim.model(x)
                if hasattr(out, "to_float"):
                    out = out.to_float()
                pred = out.argmax(1)
                correct += pred.eq(y).sum().item()
                total += y.numel()
        return correct / max(total, 1)

    t0 = time.time()
    baseline = eval_with_subset()
    print(f"baseline {mode.value}: {baseline * 100:.2f}%  ({time.time() - t0:.1f}s)")

    blocks = list(BLOCKS)
    if args.include_frontend_subblocks:
        blocks += FRONTEND_SUBBLOCKS
    if args.include_subblocks:
        blocks += SUBBLOCKS

    results: list[dict] = []
    print(f"\n{'block':32s} {'count':>6s} {'acc':>8s} {'Δ vs base':>12s}  time")
    print("-" * 72)
    for scope, prefix in blocks:
        n, restore = disable_block(sim.model, prefix)
        if n == 0:
            print(f"{scope:32s} {0:6d}     skip (no quantizer matched)")
            continue
        t0 = time.time()
        try:
            acc = eval_with_subset()
        finally:
            restore()
        dt = time.time() - t0
        delta_pp = (acc - baseline) * 100
        print(
            f"{scope:32s} {n:6d} {acc * 100:7.2f}% {delta_pp:+11.2f}pp  {dt:5.1f}s"
        )
        results.append(
            {
                "block": scope,
                "prefix": prefix,
                "disabled_quantizers": n,
                "acc": acc,
                "delta_pp_vs_baseline": delta_pp,
            }
        )

    ranking = sorted(results, key=lambda r: r["delta_pp_vs_baseline"], reverse=True)
    report = {
        "float_native_top1": float_native,
        "mode": mode.value,
        "apply_po2": args.apply_po2,
        "clz": not args.no_clz,
        "max_calib_batches": args.max_calib_batches,
        "max_eval_batches": args.max_eval_batches,
        "baseline": baseline,
        "results": results,
        "ranking_by_delta": ranking,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n报告: {args.output}")
    print("Top contributors (越靠前 disable 后回升越多 → 当前主损失模块):")
    for r in ranking[:5]:
        print(f"  {r['block']:32s} +{r['delta_pp_vs_baseline']:+.2f}pp  ({r['acc'] * 100:.2f}%)")


if __name__ == "__main__":
    main()

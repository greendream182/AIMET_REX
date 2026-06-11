#!/usr/bin/env python3
"""SYS-FU-2: full-graph segment bottleneck probe (post W17).

Reports teacher-forced **isolated** vs **chained** SQNR (dB) for backbone modules,
plus worst INT16 isolated layers graph-wide.

Usage::

    bash scripts/run-in-container.sh python examples/sys_fu2_segment_probe.py
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_REPO / "examples"), str(_REPO)]
_QG = _REPO.parent / "quant-gru-pytorch" / "pytorch"
if _QG.is_dir():
    sys.path.insert(0, str(_QG))

import aimet_torch.v2 as aimet  # noqa: E402
from aimet_torch.fixed_point import ExecutionMode, convert_encodings_to_fixed_scale  # noqa: E402
from aimet_torch.fixed_point.metrics.isolated import per_layer_isolated_cosine  # noqa: E402
from common.mrnn_clz_encoding import (  # noqa: E402
    apply_mrnn_clz_encoding_fixes,
    apply_mrnn_clz_encoding_fixes_post_calib,
    collect_power2_float_out_fmax,
)
from quick_start import DEVICE, FP_MODEL_PATH, MRNN, set_seed, setup_audio_backend  # noqa: E402
from quick_start_int16_metric import (  # noqa: E402
    ACCEPTANCE_BITWIDTH_CONFIG,
    SYSQ1_W6_PERCENTILE_VALUE,
    _calib_fn,
    _patch_torchaudio_with_soundfile,
    build_sim,
)
from sys_fu2_teacher_forced import chained_sqnr, teacher_forced_sqnr  # noqa: E402

_WATCH = (
    "freq_downs.0.conv2d",
    "freq_downs.1.conv2d",
    "freq_downs.2.conv2d",
    "enc_seqs.0.conv_t",
    "enc_seqs.0.seq_t",
    "enc_seqs.1.conv_t",
    "enc_seqs.1.seq_t",
    "neck_seqs.0.conv_t",
    "neck_seqs.0.seq_t",
    "neck_seqs.1.conv_t",
    "neck_seqs.1.seq_t",
    "fc0",
    "module_mean_4",
)

_BLOCK_OUT = (
    "power_compress_1.module_mul",
    "pre_bn.module_add",
    "module_clamp",
    "hypot_fun.module_sqrt_1",
    "module_clamp_2",
    "fft2band.module_matmul",
    "module_clamp_3",
    "power_compress_2.module_mul_2",
    "module_clamp_4",
    "conv_in",
    "enc_seqs.0",
    "enc_seqs.1",
    "neck_seqs.1",
    "fc0",
)

_PC2 = (
    "power_compress_2.module_sign_1",
    "power_compress_2.module_abs_2",
    "power_compress_2.module_sqrt_2",
    "power_compress_2.module_mul_2",
)


def _force_prefix_activation_bitwidth(model: torch.nn.Module, prefix: str, bitwidth: int) -> int:
    touched = 0
    for name, module in model.named_modules():
        if not name.startswith(prefix):
            continue
        for attr in ("input_quantizers", "output_quantizers"):
            qcontainer = getattr(module, attr, None)
            if qcontainer is None:
                continue
            keys = qcontainer.keys() if hasattr(qcontainer, "keys") else range(len(qcontainer))
            for key in keys:
                q = qcontainer[key]
                if q is None:
                    continue
                q.bitwidth = bitwidth
                if getattr(q, "symmetric", True):
                    q.qmin = -(2 ** (bitwidth - 1))
                    q.qmax = 2 ** (bitwidth - 1) - 1
                else:
                    q.qmin = 0
                    q.qmax = 2**bitwidth - 1
                touched += 1
    return touched


def _q_summary(q) -> str:
    if q is None:
        return "None"
    try:
        if not q.is_initialized():
            return f"bw={getattr(q, 'bitwidth', '?')} uninit"
        mn = float(q.get_min().reshape(-1)[0].item())
        mx = float(q.get_max().reshape(-1)[0].item())
        scale = getattr(q, "scale", None)
        s = float(scale.reshape(-1)[0].item()) if scale is not None else float("nan")
        return (
            f"bw={getattr(q, 'bitwidth', '?')} q=[{getattr(q, 'qmin', '?')},"
            f"{getattr(q, 'qmax', '?')}] scale={s:.3e} range=[{mn:.3e},{mx:.3e}]"
        )
    except (AttributeError, RuntimeError, ValueError):
        return f"bw={getattr(q, 'bitwidth', '?')} <summary failed>"


def _print_pc2_quantizers(model: torch.nn.Module) -> None:
    print(f"\n--- power_compress_2 quantizers ---")
    mods = dict(model.named_modules())
    for suffix in _PC2:
        mod = mods.get(suffix)
        if mod is None:
            print(f"  {suffix}: <missing>")
            continue
        iqs = getattr(mod, "input_quantizers", None)
        oqs = getattr(mod, "output_quantizers", None)
        in0 = _q_summary(iqs[0]) if iqs and len(iqs) > 0 else "n/a"
        in1 = _q_summary(iqs[1]) if iqs and len(iqs) > 1 else ""
        out0 = _q_summary(oqs[0]) if oqs and len(oqs) > 0 else "n/a"
        extra = f" in1={in1}" if in1 else ""
        print(f"  {suffix}: in0={in0}{extra} out={out0}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SYS-FU-2 segment bottleneck probe")
    parser.add_argument("--max-calib-batches", type=int, default=16)
    parser.add_argument("--top-k-worst", type=int, default=15)
    parser.add_argument("--sign-input-scale", type=float, default=2e-9)
    parser.add_argument("--force-pc2-16", action="store_true")
    args = parser.parse_args()

    set_seed(0)
    setup_audio_backend()
    _patch_torchaudio_with_soundfile()
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    model = MRNN().to(device)
    model.load_state_dict(
        torch.load(FP_MODEL_PATH, map_location=device, weights_only=True)
    )
    import quick_start as qs

    qs.BATCH_SIZE = 32
    loaders = qs.build_dataloaders(
        os.environ.get("SPEECH_COMMANDS_ROOT", "/home/llq/workspace/data/speech_commands")
    )
    sample = next(iter(loaders["calib"]))[0][:1].to(device)

    sim, pf = build_sim(
        copy.deepcopy(model),
        sample,
        bitwidth_config=ACCEPTANCE_BITWIDTH_CONFIG,
        quant_scheme="percentile",
        percentile_value=SYSQ1_W6_PERCENTILE_VALUE,
        native_trans=True,
    )
    sim.model.to(device).eval()
    apply_mrnn_clz_encoding_fixes(sim.model, sign_input_bypass=False, verbose=False)
    if args.force_pc2_16:
        touched = _force_prefix_activation_bitwidth(sim.model, "power_compress_2.", 16)
        print(f"[DIAG] forced power_compress_2.* activation quantizers to 16-bit: {touched} slots")
    calib = _calib_fn(sim.model, loaders["calib"], device, args.max_calib_batches)
    with torch.no_grad(), aimet.nn.compute_encodings(sim.model):
        calib(sim.model)
    if pf is not None:
        apply_mrnn_clz_encoding_fixes_post_calib(
            sim.model,
            sign_input_scale=args.sign_input_scale,
            sign_output_unit=True,
            power2_float_out_fmax=collect_power2_float_out_fmax(
                pf.to(device), loaders["calib"], device, args.max_calib_batches,
            ),
            verbose=False,
        )
    convert_encodings_to_fixed_scale(sim)
    _print_pc2_quantizers(sim.model)

    mode = ExecutionMode.INT16_FIXED_EVAL
    iso = teacher_forced_sqnr(sim.model, sample, _WATCH, cand_mode=mode)
    chn = chained_sqnr(sim.model, sample, _WATCH, cand_mode=mode)
    blk = chained_sqnr(sim.model, sample, _BLOCK_OUT, cand_mode=mode)
    pc2_iso = teacher_forced_sqnr(sim.model, sample, _PC2, cand_mode=mode)
    pc2_chn = chained_sqnr(sim.model, sample, _PC2, cand_mode=mode)

    print(f"\n=== SYS-FU-2 segment probe (cand={mode.value}, calib={args.max_calib_batches}) ===\n")
    print(f"{'module':32s} {'iso_dB':>7s} {'chn_dB':>7s} {'Δiso-chn':>9s}")
    for suffix in _WATCH:
        i = iso.get(suffix, float("nan"))
        c = chn.get(suffix, float("nan"))
        delta = i - c if i == i and c == c else float("nan")
        print(f"{suffix:32s} {i:7.2f} {c:7.2f} {delta:9.2f}")

    print(f"\n--- block outputs (chained SQNR dB) ---")
    for suffix in _BLOCK_OUT:
        c = blk.get(suffix, float("nan"))
        print(f"  {suffix:32s} {c:7.2f}")

    print(f"\n--- power_compress_2 internals ---")
    print(f"  {'module':36s} {'iso_dB':>7s} {'chn_dB':>7s}")
    for suffix in _PC2:
        print(
            f"  {suffix:36s} {pc2_iso.get(suffix, float('nan')):7.2f} "
            f"{pc2_chn.get(suffix, float('nan')):7.2f}"
        )

    rows = per_layer_isolated_cosine(
        sim.model, sample, cand_mode=mode, top_k=args.top_k_worst,
    )
    print(f"\n--- worst {args.top_k_worst} isolated INT16 layers (teacher-forced) ---")
    print(f"{'module':48s} {'cos':>8s} {'sqnr_dB':>8s}")
    for row in rows:
        print(
            f"{row['module']:48s} {row['isolated_cosine']:8.4f} "
            f"{row.get('sqnr_db', float('nan')):8.2f}"
        )


if __name__ == "__main__":
    main()

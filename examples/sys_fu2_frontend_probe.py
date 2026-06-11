#!/usr/bin/env python3
"""SYS-FU-2 FU-2-6b: frontend power_compress_1 / pre_bn INT16 root-cause probe.

Compares ``sign_input_bypass`` on/off, teacher-forced isolated SQNR, and staged
cosine at frontend boundaries.

Usage::

    bash scripts/run-in-container.sh python examples/sys_fu2_frontend_probe.py
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
from common.mrnn_clz_encoding import (  # noqa: E402
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
    evaluate_limited,
    report_per_node_cosine,
)
from sys_fu2_teacher_forced import chained_sqnr, sqnr_db, teacher_forced_sqnr  # noqa: E402

_PC1 = (
    "power_compress_1.module_sign",
    "power_compress_1.module_abs_1",
    "power_compress_1.module_sqrt",
    "power_compress_1.module_mul",
)
_PREBN = (
    "pre_bn.module_sub",
    "pre_bn.module_div",
    "pre_bn.module_mul_1",
    "pre_bn.module_add",
)
_STAGES = (
    "power_compress_1.module_mul",
    "pre_bn.module_add",
    "hypot_fun.module_sqrt_1",
    "fft2band.module_matmul",
)


def _prepare(
    *,
    sign_bypass: bool,
    sign_input_scale: float,
    calib_batches: int,
    device: torch.device,
):
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
        sign_input_bypass=sign_bypass,
    )
    sim.model.to(device).eval()
    calib = _calib_fn(sim.model, loaders["calib"], device, calib_batches)
    with torch.no_grad(), aimet.nn.compute_encodings(sim.model):
        calib(sim.model)
    if pf is not None:
        apply_mrnn_clz_encoding_fixes_post_calib(
            sim.model,
            sign_input_scale=None if sign_bypass else sign_input_scale,
            sign_output_unit=not sign_bypass,
            power2_float_out_fmax=collect_power2_float_out_fmax(
                pf.to(device), loaders["calib"], device, calib_batches,
            ),
            verbose=False,
        )
    convert_encodings_to_fixed_scale(sim)
    return sim, loaders, sample, device


def main() -> None:
    parser = argparse.ArgumentParser(description="SYS-FU-2 frontend probe")
    parser.add_argument("--max-calib-batches", type=int, default=16)
    parser.add_argument("--max-eval-batches", type=int, default=80)
    parser.add_argument("--sign-input-scale", type=float, default=2e-9)
    args = parser.parse_args()

    set_seed(0)
    setup_audio_backend()
    _patch_torchaudio_with_soundfile()
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    mode = ExecutionMode.INT16_FIXED_EVAL

    print(f"\n=== SYS-FU-2 frontend probe (calib={args.max_calib_batches}) ===\n")

    for sign_bypass in (False, True):
        label = "sign_bypass=ON" if sign_bypass else "sign_bypass=OFF"
        sim, loaders, sample, device = _prepare(
            sign_bypass=sign_bypass,
            sign_input_scale=args.sign_input_scale,
            calib_batches=args.max_calib_batches,
            device=device,
        )
        iso_pc1 = teacher_forced_sqnr(sim.model, sample, _PC1, cand_mode=mode)
        chn_st = chained_sqnr(sim.model, sample, _STAGES, cand_mode=mode)
        rows = report_per_node_cosine(sim.model, "power_compress_1", sample, device)
        from aimet_torch.fixed_point import quant_execution_mode

        with quant_execution_mode(mode):
            acc = evaluate_limited(
                sim.model, loaders["test"], device, max_batches=args.max_eval_batches,
            )
        acc_msg = f"{acc * 100:.2f}%"
        print(f"--- {label}  Top-1({args.max_eval_batches} batches)={acc_msg} ---")
        print(f"  {'node':36s} {'iso_dB':>7s} {'chn_dB':>7s}")
        for n in _PC1:
            print(f"  {n:36s} {iso_pc1.get(n, float('nan')):7.2f} {chn_st.get(n, float('nan')):7.2f}")
        print(f"  {'pre_bn iso':36s}", end="")
        for n in _PREBN:
            v = teacher_forced_sqnr(sim.model, sample, (n,), cand_mode=mode).get(n, float("nan"))
            print(f" {n.split('.')[-1]}={v:.1f}", end="")
        print()
        if rows:
            worst = min(rows, key=lambda r: r[3])
            print(
                f"  pc1 worst local: {worst[0]} cos_cum={worst[2]:.4f} "
                f"cos_local={worst[3]:.4f}"
            )
        del sim

    # STFT → pc1 input snapshot (float path, one batch)
    sim, _, sample, device = _prepare(
        sign_bypass=False,
        sign_input_scale=args.sign_input_scale,
        calib_batches=args.max_calib_batches,
        device=device,
    )
    stft_out: dict[str, torch.Tensor] = {}

    def _hook(_m, _i, o):
        stft_out["trans"] = o.detach().float().clone()

    for name, mod in sim.model.named_modules():
        if name == "trans":
            h = mod.register_forward_hook(_hook)
            break
    else:
        h = None
    with torch.no_grad():
        sim.model(sample)
    if h is not None:
        h.remove()
    if "trans" in stft_out:
        t = stft_out["trans"]
        sl = t[:, :, :, 1:, :]
        print(f"\n--- STFT slice → power_compress_1 input (fp32) ---")
        print(f"  shape={tuple(sl.shape)}  abs_p99={float(sl.abs().quantile(0.99)):.4e}")
        print(f"  near_zero_frac(|x|<1e-4)={float((sl.abs() < 1e-4).float().mean()):.3f}")
        nz = sl.abs() < 1e-4
        print(f"  sign flip risk: mixed sign near zero = {float((nz.sum() > 0).float().mean()):.3f} channels")


if __name__ == "__main__":
    main()

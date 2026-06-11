#!/usr/bin/env python3
"""SYS-FU-2: sweep ``module_sign`` input scale for integer sign (spec §4.3.6).

Hardware path requires ``sign(q_x - Z_x)`` with input Q enabled (no bypass).
This script measures ``sign_agreement(Q_in)`` vs float on captured STFT→sign
activations, sweeps symmetric i16 scale, optionally applies the best encoding
and runs a short INT16_FIXED_EVAL Top-1 check.

Usage::

    bash scripts/run-in-container.sh python examples/sys_fu2_sign_encoding_sweep.py
    bash scripts/run-in-container.sh python examples/sys_fu2_sign_encoding_sweep.py \\
        --apply-best --max-eval-batches 80
"""

from __future__ import annotations

import argparse
import copy
import math
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
from aimet_torch.fixed_point import (  # noqa: E402
    ExecutionMode,
    convert_encodings_to_fixed_scale,
    quant_execution_mode,
)
from aimet_torch.fixed_point.metrics.accuracy import quantize_float_to_grid  # noqa: E402
from aimet_torch.v2.quantization.affine import AffineQuantizerBase  # noqa: E402
from common.mrnn_clz_encoding import (  # noqa: E402
    apply_mrnn_clz_encoding_fixes_post_calib,
    collect_power2_float_out_fmax,
    fix_sign_output_encodings,
)
from quick_start import DEVICE, FP_MODEL_PATH, MRNN, set_seed, setup_audio_backend  # noqa: E402
from quick_start_int16_metric import (  # noqa: E402
    ACCEPTANCE_BITWIDTH_CONFIG,
    SYSQ1_W6_PERCENTILE_VALUE,
    _calib_fn,
    _patch_torchaudio_with_soundfile,
    build_sim,
    evaluate_limited,
)
from sys_fu2_teacher_forced import teacher_forced_sqnr  # noqa: E402

_SIGN_NODES = (
    "power_compress_1.module_sign",
    "power_compress_2.module_sign_1",
)
_I16_QMIN, _I16_QMAX = -32768, 32767
_AGREE_GATE = 0.99


def _collect_sign_float_inputs(
    model: torch.nn.Module,
    loaders,
    device: torch.device,
    *,
    max_batches: int,
) -> dict[str, list[torch.Tensor]]:
    """Hook ``module_sign`` inputs (fp32 from STFT slice path)."""

    bufs: dict[str, list[torch.Tensor]] = {n: [] for n in _SIGN_NODES}
    handles = []

    def _make_hook(name: str):
        def _h(_mod, inp, _out):
            if not inp or not isinstance(inp[0], torch.Tensor):
                return
            bufs[name].append(inp[0].detach().float().cpu())

        return _h

    for name, mod in model.named_modules():
        if name in bufs:
            handles.append(mod.register_forward_hook(_make_hook(name)))

    with torch.no_grad():
        for bi, (x, _) in enumerate(loaders["calib"]):
            if bi >= max_batches:
                break
            model(x.to(device))

    for h in handles:
        h.remove()
    return bufs


def _sign_metrics(
    x: torch.Tensor,
    scale: float,
) -> dict[str, float]:
    dev = x.device
    scale_t = torch.tensor(scale, device=dev, dtype=torch.float32)
    zp = torch.tensor(0, device=dev, dtype=torch.float32)
    q = quantize_float_to_grid(x, scale_t, zp, _I16_QMIN, _I16_QMAX)
    signs_q = (q > 0).to(torch.int32) - (q < 0).to(torch.int32)
    signs_f = torch.sign(x.to(torch.float32))
    agree = float((signs_f == signs_q.to(torch.float32)).float().mean().item())
    sat = float((q.abs() >= _I16_QMAX).float().mean().item())
    zero_q = float((q == 0).float().mean().item())
    near = float((x.abs() < scale * 0.5).float().mean().item())
    return {
        "sign_agreement": agree,
        "saturation_rate": sat,
        "q_zero_frac": zero_q,
        "near_lsb_frac": near,
        "scale": scale,
    }


def _subsample(x: torch.Tensor, max_n: int = 1_048_576) -> torch.Tensor:
    flat = x.reshape(-1)
    if flat.numel() <= max_n:
        return flat
    step = max(1, (flat.numel() + max_n - 1) // max_n)
    return flat[::step][:max_n]


def _sweep_scales(x_cat: torch.Tensor) -> list[dict[str, float]]:
    abs_x = _subsample(x_cat.abs())
    ref_max = float(abs_x.quantile(0.995).item())
    ref_max = max(ref_max, 1e-8)
    base_scale = ref_max / float(_I16_QMAX)
    multipliers = [2 ** e for e in range(-16, 5)]  # 1/65536 .. 16×
    rows: list[dict[str, float]] = []
    for mult in multipliers:
        scale = base_scale * mult
        if scale <= 0.0 or not math.isfinite(scale):
            continue
        row = _sign_metrics(x_cat, scale)
        row["mult_vs_p995"] = mult
        row["amax"] = scale * float(_I16_QMAX)
        rows.append(row)
    return rows


def _current_sign_input_scale(model: torch.nn.Module, name: str) -> float | None:
    mod = dict(model.named_modules()).get(name)
    if mod is None:
        return None
    iqs = getattr(mod, "input_quantizers", None)
    if not iqs or iqs[0] is None:
        return None
    iq = iqs[0]
    if not isinstance(iq, AffineQuantizerBase) or not iq.is_initialized():
        return None
    enc = iq.get_encodings()
    return float(enc.scale.detach().reshape(-1)[0].item())


def _apply_symmetric_scale(model: torch.nn.Module, name: str, scale: float) -> None:
    mod = dict(model.named_modules()).get(name)
    if mod is None:
        raise KeyError(name)
    iq = mod.input_quantizers[0]
    if not isinstance(iq, AffineQuantizerBase):
        raise TypeError(f"{name}.input_quantizers[0] is not AffineQuantizerBase")
    amax = scale * float(_I16_QMAX)
    device = iq.scale.device if hasattr(iq, "scale") else torch.device("cpu")
    mn = torch.tensor(-amax, device=device, dtype=torch.float32)
    mx = torch.tensor(amax, device=device, dtype=torch.float32)
    iq.set_range(mn, mx)


def _actual_sign_output_agreement(
    model: torch.nn.Module,
    loaders,
    device: torch.device,
    name: str,
) -> float | None:
    mod = dict(model.named_modules()).get(name)
    if mod is None:
        return None
    seen: dict[str, torch.Tensor] = {}

    def _hook(_mod, inp, out):
        if inp and isinstance(inp[0], torch.Tensor):
            seen["x"] = inp[0].detach().float().cpu()
        if hasattr(out, "to_float"):
            seen["y"] = out.to_float(torch.float32).detach().float().cpu()
        elif hasattr(out, "dequantize"):
            seen["y"] = out.dequantize().detach().float().cpu()
        elif isinstance(out, torch.Tensor):
            seen["y"] = out.detach().float().cpu()

    h = mod.register_forward_hook(_hook)
    x, _ = next(iter(loaders["calib"]))
    with torch.no_grad(), quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        model(x[:1].to(device))
    h.remove()
    if "x" not in seen or "y" not in seen:
        return None
    return float((torch.sign(seen["x"]) == torch.sign(seen["y"])).float().mean().item())


def _prepare(
    *,
    sign_bypass: bool,
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
            power2_float_out_fmax=collect_power2_float_out_fmax(
                pf.to(device), loaders["calib"], device, calib_batches,
            ),
            verbose=False,
        )
    convert_encodings_to_fixed_scale(sim)
    return sim, loaders, device


def main() -> None:
    parser = argparse.ArgumentParser(description="SYS-FU-2 sign input encoding sweep")
    parser.add_argument("--max-calib-batches", type=int, default=16)
    parser.add_argument("--max-eval-batches", type=int, default=80)
    parser.add_argument(
        "--apply-best",
        action="store_true",
        help="将 sweep 最优 scale 写入 sign input Q 并跑短 E2E",
    )
    args = parser.parse_args()

    set_seed(0)
    setup_audio_backend()
    _patch_torchaudio_with_soundfile()
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    print(f"\n=== SYS-FU-2 sign encoding sweep (calib={args.max_calib_batches}) ===\n")

    sim, loaders, device = _prepare(
        sign_bypass=False,
        calib_batches=args.max_calib_batches,
        device=device,
    )
    captured = _collect_sign_float_inputs(
        sim.model, loaders, device, max_batches=args.max_calib_batches,
    )

    best_per_node: dict[str, dict[str, float]] = {}
    for name in _SIGN_NODES:
        chunks = captured.get(name, [])
        if not chunks:
            print(f"--- {name}: no captures ---")
            continue
        x_cat = torch.cat([t.reshape(-1) for t in chunks])
        cur_scale = _current_sign_input_scale(sim.model, name)
        rows = _sweep_scales(x_cat)
        feasible = [
            r for r in rows
            if r["sign_agreement"] >= _AGREE_GATE and r["saturation_rate"] <= 0.01
        ]
        if feasible:
            best = min(feasible, key=lambda r: r["scale"])
            best_tag = "feasible(agree≥0.99,sat≤1%)"
        else:
            best = max(rows, key=lambda r: r["sign_agreement"])
            best_tag = "agree-only (no sat-feasible point)"
        best_per_node[name] = best

        print(f"--- {name}  (N={x_cat.numel()}, PTQ scale={cur_scale}) ---")
        if cur_scale is not None:
            cur = _sign_metrics(x_cat.to(device), cur_scale)
            print(
                f"  PTQ: agree={cur['sign_agreement']:.4f} sat={cur['saturation_rate']:.4f} "
                f"q_zero={cur['q_zero_frac']:.3f}"
            )
        hi_agree = [r for r in rows if r["sign_agreement"] >= _AGREE_GATE]
        if hi_agree:
            lo_sat = min(hi_agree, key=lambda r: r["saturation_rate"])
            print(
                f"  Pareto(agree≥{_AGREE_GATE}): min_sat={lo_sat['saturation_rate']*100:.2f}% "
                f"@ scale={lo_sat['scale']:.3e} agree={lo_sat['sign_agreement']:.4f}"
            )
        else:
            print(f"  Pareto: 无 scale 达 agree≥{_AGREE_GATE}")
        print(f"  {'mult':>8s} {'scale':>10s} {'agree':>7s} {'sat%':>6s} {'q0%':>6s}")
        for r in sorted(rows, key=lambda z: (-z["sign_agreement"], z["saturation_rate"]))[:10]:
            mark = "*" if r is best else " "
            gate = "✓" if r["sign_agreement"] >= _AGREE_GATE else " "
            print(
                f"{mark}{gate} {r['mult_vs_p995']:8.4f} {r['scale']:10.6e} "
                f"{r['sign_agreement']:7.4f} {r['saturation_rate']*100:6.2f} "
                f"{r['q_zero_frac']*100:6.2f}"
            )
        print(
            f"  pick[{best_tag}]: scale={best['scale']:.6e} "
            f"agree={best['sign_agreement']:.4f} sat={best['saturation_rate']:.4f}"
        )

    if args.apply_best and best_per_node:
        for name, best in best_per_node.items():
            if best["sign_agreement"] < _AGREE_GATE:
                print(
                    f"\n  skip apply-best for {name}: no feasible point "
                    f"(agree≥{_AGREE_GATE})"
                )
                continue
            _apply_symmetric_scale(sim.model, name, best["scale"])
        sign_out = fix_sign_output_encodings(sim.model)
        print(f"  sign output unit-grid touched={sign_out['touched']}")
        convert_encodings_to_fixed_scale(sim)
        for name in best_per_node:
            actual = _actual_sign_output_agreement(sim.model, loaders, device, name)
            print(f"  actual INT16 output sign agreement: {name} = {actual}")
        sample = next(iter(loaders["calib"]))[0][:1].to(device)
        mode = ExecutionMode.INT16_FIXED_EVAL
        iso = teacher_forced_sqnr(
            sim.model, sample, ("power_compress_1.module_sign",), cand_mode=mode,
        )
        with quant_execution_mode(mode):
            acc = evaluate_limited(
                sim.model, loaders["test"], device, max_batches=args.max_eval_batches,
            )
        print(f"\n--- apply-best INT16 (sign_bypass=OFF) ---")
        print(f"  module_sign iso_dB={iso.get('power_compress_1.module_sign', float('nan')):.2f}")
        print(f"  Top-1({args.max_eval_batches} batches)={acc * 100:.2f}%")


if __name__ == "__main__":
    main()

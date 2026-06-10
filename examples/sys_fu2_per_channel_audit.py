#!/usr/bin/env python3
"""SYS-FU-2 FU-2-1: per-channel vs per-tensor activation audit on freq_downs Conv2d.

Teacher-forced protocol (same float input at each ``freq_downs.*.conv2d``):

- **per_tensor**: requantize with the module's calibrated ``input_quantizers[0]``
  grid (current production path).
- **per_channel**: oracle symmetric 8-bit grid with one scale per NCHW channel
  (axis=1), scale from calib-batch abs-percentile (default 99.99).

If per_channel SQNR >> per_tensor on the same fp32 input, W7's multi-mode
hypothesis is confirmed and per-tensor grid is the bottleneck (not Conv kernel).

Usage::

    python examples/sys_fu2_per_channel_audit.py --max-calib-batches 16
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_EXAMPLES = _REPO_ROOT / "examples"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

_QG = _REPO_ROOT.parent / "quant-gru-pytorch" / "pytorch"
if _QG.is_dir() and str(_QG) not in sys.path:
    sys.path.insert(0, str(_QG))

import aimet_torch.v2 as aimet  # noqa: E402
import aimet_torch.fixed_point.kernels  # noqa: F401, E402
from aimet_torch.fixed_point import (  # noqa: E402
    ExecutionMode,
    convert_encodings_to_fixed_scale,
    quant_execution_mode,
)
from aimet_torch.fixed_point.metrics.isolated import (  # noqa: E402
    _cosine,
    _to_float,
)
from aimet_torch.fixed_point.requantize import saturate_sim_tensor  # noqa: E402
from aimet_torch.fixed_point.tensor import FixedPointSimTensor  # noqa: E402
from common.mrnn_clz_encoding import (  # noqa: E402
    apply_mrnn_clz_encoding_fixes,
    apply_mrnn_clz_encoding_fixes_post_calib,
    collect_power2_float_out_fmax,
)
from quick_start import (  # noqa: E402
    DEVICE,
    FP_MODEL_PATH,
    MRNN,
    set_seed,
    setup_audio_backend,
)
from quick_start_int16_metric import (  # noqa: E402
    SYSQ1_W6_PERCENTILE_VALUE,
    _calib_fn,
    _patch_torchaudio_with_soundfile,
    build_sim,
)

_TARGETS = (
    "freq_downs.0.conv2d",
    "freq_downs.1.conv2d",
    "freq_downs.2.conv2d",
)


def _find_module(model: torch.nn.Module, suffix: str) -> tuple[str, torch.nn.Module] | None:
    for qualname, mod in model.named_modules():
        if qualname == suffix or qualname.endswith("." + suffix):
            return qualname, mod
    return None


def _sqnr_db(ref: torch.Tensor, cand: torch.Tensor) -> float:
    diff = ref.reshape(-1).float() - cand.reshape(-1).float()
    sig = ref.reshape(-1).float().pow(2).mean().item()
    noise = diff.pow(2).mean().item()
    if noise <= 0 or sig <= 0:
        return float("inf")
    return float(10.0 * math.log10(sig / noise))


def _per_channel_median_sqnr(
    ref: torch.Tensor,
    cand: torch.Tensor,
    *,
    min_rms: float = 1e-6,
) -> float:
    """Median per-channel SQNR over channels with sufficient reference energy."""
    if ref.ndim != 4:
        return _sqnr_db(ref, cand)
    sqnrs: list[float] = []
    for ch in range(ref.shape[1]):
        r = ref[:, ch]
        if float(r.pow(2).mean().sqrt().item()) < min_rms:
            continue
        sq = _sqnr_db(r, cand[:, ch])
        if math.isfinite(sq):
            sqnrs.append(sq)
    if not sqnrs:
        return float("nan")
    sqnrs.sort()
    return sqnrs[len(sqnrs) // 2]


@torch.no_grad()
def _collect_int16_input_carriers(
    model: torch.nn.Module,
    sample: torch.Tensor,
    targets: tuple[str, ...],
) -> dict[str, dict]:
    carriers: dict[str, dict] = {}

    def _hook(name: str):
        def _fn(_mod, inputs):
            if not inputs:
                return
            first = inputs[0]
            if isinstance(first, FixedPointSimTensor):
                carriers[name] = {
                    "scale": first.scale.detach().clone(),
                    "zero_point": first.zero_point.detach().clone(),
                    "qmin": first.qmin,
                    "qmax": first.qmax,
                    "axis": first.axis,
                }
        return _fn

    handles = []
    for suffix in targets:
        hit = _find_module(model, suffix)
        if hit is None:
            continue
        qualname, mod = hit
        handles.append(mod.register_forward_pre_hook(_hook(qualname)))

    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        model(sample)
    for h in handles:
        h.remove()
    return carriers


def _sim_from_carrier(x: torch.Tensor, carrier: dict) -> FixedPointSimTensor:
    scale = carrier["scale"].to(device=x.device, dtype=torch.float32)
    zp = carrier["zero_point"].to(device=x.device, dtype=torch.float32)
    q = torch.round(x / scale + zp)
    int_repr = saturate_sim_tensor(q, carrier["qmin"], carrier["qmax"])
    return FixedPointSimTensor(
        int_repr=int_repr,
        scale=scale,
        zero_point=zp.to(torch.int32),
        qmin=carrier["qmin"],
        qmax=carrier["qmax"],
        axis=carrier.get("axis"),
    )


def _sim_per_channel_nchw(
    x: torch.Tensor,
    *,
    bitwidth: int = 8,
    symmetric: bool = True,
    percentile: float = 99.99,
) -> FixedPointSimTensor:
    if x.ndim != 4:
        raise ValueError(f"expected NCHW, got {x.shape}")
    qmax = (1 << (bitwidth - 1)) - 1 if symmetric else (1 << bitwidth) - 1
    qmin = -(1 << (bitwidth - 1)) if symmetric else 0
    c = x.shape[1]
    scales = torch.empty(c, device=x.device, dtype=torch.float32)
    for ch in range(c):
        flat = x[:, ch].reshape(-1).abs()
        amax = float(torch.quantile(flat, percentile / 100.0).item())
        scales[ch] = max(amax, 1e-8) / qmax
    scale_b = scales.view(1, c, 1, 1)
    q = torch.round(x / scale_b)
    int_repr = saturate_sim_tensor(q, qmin, qmax)
    zp = torch.zeros(1, c, 1, 1, device=x.device, dtype=torch.int32)
    return FixedPointSimTensor(
        int_repr=int_repr,
        scale=scale_b,
        zero_point=zp,
        qmin=qmin,
        qmax=qmax,
        axis=1,
    )


@torch.no_grad()
def _collect_conv_inputs(
    model: torch.nn.Module,
    sample: torch.Tensor,
    targets: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    found: dict[str, torch.Tensor] = {}

    def _hook(name: str):
        def _fn(_mod, inputs):
            if inputs and isinstance(inputs[0], torch.Tensor):
                found[name] = inputs[0].detach().clone()
        return _fn

    handles = []
    for suffix in targets:
        hit = _find_module(model, suffix)
        if hit is None:
            continue
        qualname, mod = hit
        handles.append(mod.register_forward_pre_hook(_hook(qualname)))

    with quant_execution_mode(ExecutionMode.FP32_QDQ):
        model(sample)
    for h in handles:
        h.remove()
    return found


@torch.no_grad()
def _dequant_sim(x_sim: FixedPointSimTensor) -> torch.Tensor:
    with torch.no_grad():
        from aimet_torch.fixed_point.metrics.flags import int16_eval_allow_debug_float

        with int16_eval_allow_debug_float():
            return x_sim.to_float()


@torch.no_grad()
def _audit_one(
    module: torch.nn.Module,
    x_float: torch.Tensor,
    carrier: dict,
    *,
    percentile: float,
) -> dict[str, float]:
    with quant_execution_mode(ExecutionMode.FP32_QDQ):
        y_ref = _to_float(module(x_float))
    assert y_ref is not None

    x_pt = _sim_from_carrier(x_float, carrier)
    x_pt_dq = _dequant_sim(x_pt)
    in_pt_cos = _cosine(x_pt_dq, x_float)
    in_pt_sqnr = _sqnr_db(x_float, x_pt_dq)

    x_pc = _sim_per_channel_nchw(x_float, percentile=percentile)
    x_pc_dq = _dequant_sim(x_pc)
    in_pc_cos = _cosine(x_pc_dq, x_float)
    in_pc_sqnr = _sqnr_db(x_float, x_pc_dq)

    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y_pt = _to_float(module(x_pt))
    assert y_pt is not None

    # Upper bound: fp32 conv on per-channel-dequantized input (kernel not PC-input yet).
    with quant_execution_mode(ExecutionMode.FP32_QDQ):
        y_pc_fp32 = _to_float(module(x_pc_dq))
    assert y_pc_fp32 is not None

    return {
        "in_pt_cos": in_pt_cos,
        "in_pt_sqnr": in_pt_sqnr,
        "in_pt_ch_med_sqnr": _per_channel_median_sqnr(x_float, x_pt_dq),
        "in_pc_cos": in_pc_cos,
        "in_pc_sqnr": in_pc_sqnr,
        "in_pc_ch_med_sqnr": _per_channel_median_sqnr(x_float, x_pc_dq),
        "in_delta_sqnr_db": in_pc_sqnr - in_pt_sqnr,
        "in_delta_ch_med_db": _per_channel_median_sqnr(x_float, x_pc_dq)
        - _per_channel_median_sqnr(x_float, x_pt_dq),
        "out_pt_cos": _cosine(y_pt, y_ref),
        "out_pt_sqnr": _sqnr_db(y_ref, y_pt),
        "out_pc_fp32_cos": _cosine(y_pc_fp32, y_ref),
        "out_pc_fp32_sqnr": _sqnr_db(y_ref, y_pc_fp32),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SYS-FU-2 FU-2-1 per-channel activation audit")
    parser.add_argument(
        "--data-root",
        default=os.environ.get("SPEECH_COMMANDS_ROOT", "/home/llq/workspace/data/speech_commands"),
    )
    parser.add_argument("--fp-ckpt", type=str, default=str(FP_MODEL_PATH))
    parser.add_argument("--max-calib-batches", type=int, default=16)
    parser.add_argument(
        "--bitwidth-config",
        type=str,
        default=str(_EXAMPLES / "config" / "quick_start_full_quant.json"),
    )
    parser.add_argument("--percentile-value", type=float, default=SYSQ1_W6_PERCENTILE_VALUE)
    parser.add_argument(
        "--per-channel-percentile",
        type=float,
        default=99.99,
        help="abs-percentile for per-channel oracle scale (per channel, NCHW axis=1)",
    )
    args = parser.parse_args()

    set_seed(0)
    setup_audio_backend()
    _patch_torchaudio_with_soundfile()

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    model = MRNN().to(device)
    ckpt = Path(args.fp_ckpt)
    if not ckpt.is_file():
        sys.exit(f"fp ckpt not found: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    import quick_start as qs

    prev_bs = qs.BATCH_SIZE
    qs.BATCH_SIZE = 32
    try:
        loaders = qs.build_dataloaders(str(args.data_root))
    finally:
        qs.BATCH_SIZE = prev_bs

    sample_input = next(iter(loaders["calib"]))[0][:1].to(device)

    sim, prepared_float = build_sim(
        model,
        sample_input,
        bitwidth_config=args.bitwidth_config,
        quant_scheme="percentile",
        percentile_value=args.percentile_value,
        native_trans=True,
    )
    sim.model.to(device).eval()
    apply_mrnn_clz_encoding_fixes(sim.model, sign_input_bypass=False, verbose=False)

    calib = _calib_fn(sim.model, loaders["calib"], device, args.max_calib_batches)
    with torch.no_grad(), aimet.nn.compute_encodings(sim.model):
        calib(sim.model)

    if prepared_float is not None:
        pf = prepared_float.to(device).eval()
        power2_fmax = collect_power2_float_out_fmax(
            pf, loaders["calib"], device, args.max_calib_batches,
        )
        apply_mrnn_clz_encoding_fixes_post_calib(
            sim.model, power2_float_out_fmax=power2_fmax, verbose=False,
        )

    convert_encodings_to_fixed_scale(sim)

    # Single-batch protocol (align with per_layer_isolated_cosine teacher-forced).
    xs = _collect_conv_inputs(sim.model, sample_input, _TARGETS)
    carriers = _collect_int16_input_carriers(sim.model, sample_input, _TARGETS)

    print(
        f"\n=== SYS-FU-2 FU-2-1 per-channel activation audit "
        f"(config={args.bitwidth_config}, calib_batches={args.max_calib_batches}, "
        f"PTQ percentile={args.percentile_value}, PC scale p={args.per_channel_percentile}) ===\n"
    )
    header = (
        f"{'module':28s} {'inPT_ch':>7s} {'inPC_ch':>7s} {'Δch_dB':>7s} "
        f"{'outPT_cos':>8s} {'outPT_dB':>7s} {'outPCfp_dB':>9s}"
    )
    print(header)
    for suffix in _TARGETS:
        hit = _find_module(sim.model, suffix)
        if hit is None or hit[0] not in xs or hit[0] not in carriers:
            print(f"{suffix:28s}  (not found / no inputs / no carrier)")
            continue
        qualname, mod = hit
        stats = _audit_one(
            mod,
            xs[qualname],
            carriers[qualname],
            percentile=args.per_channel_percentile,
        )
        print(
            f"{qualname:28s} "
            f"{stats['in_pt_ch_med_sqnr']:7.2f} "
            f"{stats['in_pc_ch_med_sqnr']:7.2f} "
            f"{stats['in_delta_ch_med_db']:+7.2f} "
            f"{stats['out_pt_cos']:8.4f} {stats['out_pt_sqnr']:7.2f} "
            f"{stats['out_pc_fp32_sqnr']:9.2f}"
        )
    print(
        "\ninPT_ch/inPC_ch = **median per-channel** input recon SQNR (PT = upstream "
        "INT16 carrier grid; PC = oracle per-channel). Global in_cos≈1 hides W7 "
        "low-magnitude channel collapse. outPT = INT16 conv; outPCfp = fp32 conv "
        "upper bound if PC input were available. INT16 Conv kernel lacks per-channel "
        "input scale MAC contract today."
    )


if __name__ == "__main__":
    main()

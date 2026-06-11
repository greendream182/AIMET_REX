#!/usr/bin/env python3
"""SYS-FU-2 FU-2-4e: ``real_m = S_x·S_w/S_y`` vs ``(M, rshift)`` encoding audit.

Teacher-forced on ``freq_downs.*.conv2d``; replays captured int32 MAC ``acc`` through:

| path | meaning |
|---|---|
| mac_fp | ``acc * S_x * S_w`` float dequant (no output requant) |
| or_rq | ideal float round/clamp to output grid (52 dB ceiling) |
| hw_rq | dispatch ``OutputEncoding`` ``requantize_int(M, rshift)`` |
| ideal_rm | float round using **exact** ``real_m`` (no M quant error) |
| hw_rm | ``quantize_multiplier(real_m)`` then ``requantize_int`` |

Also reports ``eff_m = M/2**r`` vs ``real_m`` relative error and clip rates.

Usage::

    bash scripts/run-in-container.sh python examples/sys_fu2_4e_output_encoding_audit.py
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from torch import nn

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
from aimet_torch.fixed_point.channel_align import (  # noqa: E402
    align_per_channel_activation_for_conv_input,
)
from aimet_torch.fixed_point.export.v2_collect import (  # noqa: E402
    derive_int16_output_encoding,
    derive_int16_real_multiplier,
)
from aimet_torch.fixed_point.kernels import conv_linear as conv_linear_mod  # noqa: E402
from aimet_torch.fixed_point.metrics.flags import int16_eval_allow_debug_float  # noqa: E402
from aimet_torch.fixed_point.metrics.isolated import _to_float  # noqa: E402
from aimet_torch.fixed_point.offline.multiplier import quantize_multiplier  # noqa: E402
from aimet_torch.fixed_point.requantize import (  # noqa: E402
    requantize_int,
    saturate_int32,
    saturate_mac_accumulator,
    saturate_sim_tensor,
)
from aimet_torch.fixed_point.rounding import RoundingMode  # noqa: E402
from aimet_torch.fixed_point.requantize import round_shift  # noqa: E402
from aimet_torch.fixed_point.tensor import align_stat_rank  # noqa: E402
from aimet_torch.v2.nn.true_quant import QuantizationMixin  # noqa: E402
from aimet_torch.v2.quantization.affine.encoding import AffineEncoding  # noqa: E402
from common.mrnn_clz_encoding import (  # noqa: E402
    apply_mrnn_clz_encoding_fixes,
    apply_mrnn_clz_encoding_fixes_post_calib,
    collect_power2_float_out_fmax,
)
from quick_start import DEVICE, FP_MODEL_PATH, MRNN, set_seed, setup_audio_backend  # noqa: E402
from quick_start_int16_metric import (  # noqa: E402
    SYSQ1_W6_PERCENTILE_VALUE,
    _calib_fn,
    _patch_torchaudio_with_soundfile,
    build_sim,
)
from sys_fu2_per_channel_audit import (  # noqa: E402
    _TARGETS,
    _collect_conv_inputs,
    _find_module,
    _sim_per_channel_nchw,
    _sqnr_db,
)

_CAPTURE: dict[str, Any] = {}


@contextmanager
def _capture_conv_acc():
    orig = conv_linear_mod._requantize_output

    def _hook(acc, output_encoding):
        _CAPTURE["acc"] = saturate_mac_accumulator(acc).detach().clone()
        _CAPTURE["out_enc"] = output_encoding
        return orig(acc, output_encoding)

    conv_linear_mod._requantize_output = _hook
    try:
        yield
    finally:
        conv_linear_mod._requantize_output = orig


def _broadcast_real_m(
    real_m: torch.Tensor,
    acc: torch.Tensor,
) -> torch.Tensor:
    """Broadcast per-channel ``real_m`` to ``acc`` layout ``(N,C,H,W)``."""

    rm = real_m.to(device=acc.device, dtype=torch.float32)
    if rm.ndim == 0 or rm.numel() == 1:
        return rm.reshape(())
    if rm.ndim == 1 and rm.shape[0] == acc.shape[1]:
        return rm.reshape(1, -1, 1, 1)
    return align_stat_rank(rm, acc)


def _compute_real_m(
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    y_scale: torch.Tensor,
    *,
    out_channels: int,
) -> torch.Tensor:
    dev = x_scale.device
    xs = x_scale.to(dev, dtype=torch.float32)
    ws = w_scale.to(dev, dtype=torch.float32)
    ys = y_scale.to(dev, dtype=torch.float32)
    # Per-tensor / per-channel input: fold to scalar S_x for MAC (align makes max).
    if xs.numel() == 1:
        sx = xs.reshape(())
    else:
        sx = xs.reshape(-1).max()
    if ws.numel() == out_channels:
        sw = ws.reshape(out_channels, 1, 1, 1)
    elif ws.numel() == 1:
        sw = ws.reshape(())
    else:
        sw = ws.reshape(-1)[0]
    if ys.numel() == out_channels:
        sy = ys.reshape(out_channels, 1, 1, 1)
    elif ys.numel() == 1:
        sy = ys.reshape(())
    else:
        sy = ys.reshape(-1)[0]
    rm = (sx * sw) / sy
    return rm.reshape(-1)[:out_channels]


def _mac_dequant_float(
    acc: torch.Tensor,
    *,
    real_m: torch.Tensor,
    y_scale: torch.Tensor,
) -> torch.Tensor:
    """Float MAC dequant: ``acc * S_x * S_w = acc * real_m * S_y``."""

    rm = _broadcast_real_m(real_m, acc)
    sy = align_stat_rank(y_scale.to(device=acc.device, dtype=torch.float32), acc)
    if rm.ndim == 0:
        return acc.to(torch.float32) * rm * sy
    return acc.to(torch.float32) * rm * sy


def _oracle_output_quant(y_mac: torch.Tensor, out_enc) -> torch.Tensor:
    sy = align_stat_rank(
        out_enc.scale.to(device=y_mac.device, dtype=torch.float32), y_mac
    )
    zp = align_stat_rank(
        out_enc.zero_point.to(device=y_mac.device, dtype=torch.int32), y_mac
    )
    q = torch.round(y_mac / sy + zp.to(torch.float32))
    q = torch.clamp(q, out_enc.qmin, out_enc.qmax)
    return (q - zp.to(torch.float32)) * sy


def _ideal_real_m_quant(
    acc: torch.Tensor,
    real_m: torch.Tensor,
    out_enc,
) -> torch.Tensor:
    rm = _broadcast_real_m(real_m, acc)
    zp = align_stat_rank(
        out_enc.zero_point.to(device=acc.device, dtype=torch.int32), acc
    )
    sy = align_stat_rank(
        out_enc.scale.to(device=acc.device, dtype=torch.float32), acc
    )
    q = torch.round(acc.to(torch.float32) * rm + zp.to(torch.float32))
    q = torch.clamp(q, out_enc.qmin, out_enc.qmax)
    return (q - zp.to(torch.float32)) * sy


def _hw_requant_variant(
    acc: torch.Tensor,
    out_enc,
    *,
    prod_sat: bool,
) -> torch.Tensor:
    mult = out_enc.multiplier.to(device=acc.device)
    rsh = out_enc.rshift.to(device=acc.device)
    zp = out_enc.zero_point.to(device=acc.device, dtype=torch.int32)
    prod = acc.to(torch.int64) * mult.to(torch.int64)
    if prod_sat:
        prod = saturate_int32(prod).to(torch.int64)
    rounded = round_shift(prod, rsh, RoundingMode.HALF_TO_EVEN)
    shifted = rounded + zp.to(torch.int64)
    int_repr = saturate_sim_tensor(shifted, out_enc.qmin, out_enc.qmax)
    sy = align_stat_rank(
        out_enc.scale.to(device=acc.device, dtype=torch.float32), int_repr
    )
    zpb = align_stat_rank(zp, int_repr)
    return (int_repr.to(torch.float32) - zpb.to(torch.float32)) * sy


def _prod_sat_frac(acc: torch.Tensor, out_enc) -> float:
    mult = out_enc.multiplier.to(device=acc.device)
    prod = acc.to(torch.int64) * mult.to(torch.int64)
    sat = saturate_int32(prod).to(torch.int64)
    n = prod.numel()
    if n == 0:
        return float("nan")
    return float((prod != sat).sum().item()) / n


def _effective_m(multiplier: torch.Tensor, rshift: torch.Tensor) -> torch.Tensor:
    return multiplier.to(torch.float64) / (2.0 ** rshift.to(torch.int64))


def _rel_err_stats(real_m: torch.Tensor, eff_m: torch.Tensor) -> dict[str, float]:
    rm = real_m.detach().cpu().to(torch.float64).reshape(-1)
    em = eff_m.detach().cpu().to(torch.float64).reshape(-1)
    n = min(rm.numel(), em.numel())
    if n == 0:
        return {"m_rel_med": float("nan"), "m_rel_max": float("nan")}
    rm = rm[:n]
    em = em[:n]
    denom = rm.abs().clamp_min(1e-30)
    rel = ((em - rm).abs() / denom).tolist()
    rel.sort()
    return {
        "m_rel_med": float(rel[len(rel) // 2]),
        "m_rel_max": float(rel[-1]),
    }


def _clip_frac(int_repr: torch.Tensor, qmin: int, qmax: int) -> float:
    flat = int_repr.reshape(-1)
    n = flat.numel()
    if n == 0:
        return float("nan")
    lo = int((flat <= qmin).sum().item())
    hi = int((flat >= qmax).sum().item())
    return (lo + hi) / n


@torch.no_grad()
def _audit_one(
    qmodule: nn.Module,
    x_float: torch.Tensor,
    *,
    percentile: float,
) -> dict[str, float]:
    base_cls = QuantizationMixin.qcls_to_cls.get(type(qmodule), nn.Conv2d)
    dev = x_float.device

    with quant_execution_mode(ExecutionMode.FP32_QDQ):
        y_ref = _to_float(qmodule(x_float))
    assert y_ref is not None

    x_pc = _sim_per_channel_nchw(x_float, percentile=percentile)
    x_pc = align_per_channel_activation_for_conv_input(x_pc)

    _CAPTURE.clear()
    with int16_eval_allow_debug_float(), quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        with _capture_conv_acc():
            y_e2e = _to_float(qmodule(x_pc))
    assert y_e2e is not None
    acc = _CAPTURE.get("acc")
    out_enc = _CAPTURE.get("out_enc")
    if acc is None or out_enc is None:
        raise RuntimeError("failed to capture MAC acc / OutputEncoding")

    wq = qmodule.param_quantizers["weight"]
    w_enc = wq.get_encodings()
    oq = qmodule.output_quantizers[0]
    y_enc = oq.get_encodings()
    iq = qmodule.input_quantizers[0]
    x_prod_enc = iq.get_encodings()

    x_scale_mac = x_pc.scale.to(dev, dtype=torch.float32)
    w_scale = w_enc.scale.to(dev, dtype=torch.float32)
    y_scale = y_enc.scale.to(dev, dtype=torch.float32)
    out_ch = int(acc.shape[1])

    real_m_mac = _compute_real_m(
        x_scale_mac, w_scale, y_scale, out_channels=out_ch,
    )
    real_m_prod = None
    if isinstance(x_prod_enc, AffineEncoding):
        real_m_prod = _compute_real_m(
            x_prod_enc.scale.to(dev, dtype=torch.float32),
            w_scale,
            y_scale,
            out_channels=out_ch,
        )

    real_m_derived = derive_int16_real_multiplier(
        qmodule, base_cls=base_cls, device=dev,
    )

    eff_m = _effective_m(out_enc.multiplier, out_enc.rshift)
    rm_flat = real_m_mac.reshape(-1)
    em_flat = eff_m.to(dev, dtype=torch.float32).reshape(-1)
    n = min(rm_flat.numel(), em_flat.numel())
    m_stats = _rel_err_stats(rm_flat[:n], em_flat[:n])

    y_mac = _mac_dequant_float(acc, real_m=real_m_mac, y_scale=y_scale)

    with int16_eval_allow_debug_float(), quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y_hw = _hw_requant_variant(acc, out_enc, prod_sat=True)
        y_hw_ns = _hw_requant_variant(acc, out_enc, prod_sat=False)
        y_or = _oracle_output_quant(y_mac, out_enc)
        y_ideal_rm = _ideal_real_m_quant(acc, real_m_mac, out_enc)

        mult_rm, rsh_rm = quantize_multiplier(real_m_mac.detach())
        if mult_rm.numel() == out_ch:
            mult_rm = mult_rm.reshape(1, out_ch, 1, 1)
            rsh_rm = rsh_rm.reshape(1, out_ch, 1, 1)
        from aimet_torch.fixed_point.encoding import OutputEncoding

        enc_rm = OutputEncoding(
            scale=out_enc.scale,
            zero_point=out_enc.zero_point,
            qmin=out_enc.qmin,
            qmax=out_enc.qmax,
            multiplier=mult_rm.to(dev),
            rshift=rsh_rm.to(dev),
            axis=out_enc.axis,
            bias_bits=out_enc.bias_bits,
        )
        y_hw_rm = _hw_requant_variant(acc, enc_rm, prod_sat=True)
        y_hw_rm_ns = _hw_requant_variant(acc, enc_rm, prod_sat=False)

    out_derived = derive_int16_output_encoding(qmodule, base_cls=base_cls, device=dev)

    with int16_eval_allow_debug_float(), quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y_hw_derived = (
            _hw_requant_variant(acc, out_derived, prod_sat=True)
            if out_derived is not None
            else y_hw
        )

    int_hw = requantize_int(
        acc,
        out_enc.multiplier.to(dev),
        out_enc.rshift.to(dev),
        out_enc.zero_point.to(dev, dtype=torch.int32),
        out_enc.qmin,
        out_enc.qmax,
    )

    return {
        "mac_fp": _sqnr_db(y_ref, y_mac),
        "or_rq": _sqnr_db(y_ref, y_or),
        "hw_rq": _sqnr_db(y_ref, y_hw),
        "hw_ns": _sqnr_db(y_ref, y_hw_ns),
        "ideal_rm": _sqnr_db(y_ref, y_ideal_rm),
        "hw_rm": _sqnr_db(y_ref, y_hw_rm),
        "hw_der": _sqnr_db(y_ref, y_hw_derived),
        "e2e": _sqnr_db(y_ref, y_e2e),
        "or_vs_hw": _sqnr_db(y_or, y_hw),
        "ideal_vs_hw": _sqnr_db(y_ideal_rm, y_hw),
        "clip_hw%": 100.0 * _clip_frac(int_hw, out_enc.qmin, out_enc.qmax),
        "psat%": 100.0 * _prod_sat_frac(acc, out_enc),
        **m_stats,
        "real_m_med": float(real_m_mac.reshape(-1).median().item()),
        "eff_m_med": float(eff_m.reshape(-1).median().item()),
        "prod_x_eq_mac": (
            float(
                torch.max(
                    (real_m_prod.reshape(-1)[:n] - rm_flat[:n]).abs()
                    / rm_flat[:n].abs().clamp_min(1e-30)
                ).item()
            )
            if real_m_prod is not None and n > 0
            else float("nan")
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SYS-FU-2 FU-2-4e output encoding audit")
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
    parser.add_argument("--per-channel-percentile", type=float, default=99.99)
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
        copy.deepcopy(model),
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

    xs = _collect_conv_inputs(sim.model, sample_input, _TARGETS)
    print(
        f"\n=== SYS-FU-2 FU-2-4e real_m / M-rshift audit "
        f"(calib_batches={args.max_calib_batches}) ===\n"
    )
    hdr = (
        f"{'module':28s} {'mac':>6s} {'or_rq':>6s} {'ideal':>6s} "
        f"{'hw_rq':>6s} {'hw_ns':>6s} {'e2e':>6s} {'psat%':>6s} {'m_rel%':>7s}"
    )
    print(hdr)
    for suffix in _TARGETS:
        hit = _find_module(sim.model, suffix)
        if hit is None or hit[0] not in xs:
            print(f"{suffix:28s}  (skip)")
            continue
        qualname, mod = hit
        stats = _audit_one(
            mod, xs[qualname], percentile=args.per_channel_percentile,
        )
        print(
            f"{qualname:28s} "
            f"{stats['mac_fp']:6.2f} {stats['or_rq']:6.2f} {stats['ideal_rm']:6.2f} "
            f"{stats['hw_rq']:6.2f} {stats['hw_ns']:6.2f} {stats['e2e']:6.2f} "
            f"{stats['psat%']:6.2f} {100 * stats['m_rel_med']:7.3f}"
        )
        print(
            f"  {'':28s} or_vs_hw={stats['or_vs_hw']:.2f}dB "
            f"real_m={stats['real_m_med']:.4e} eff_m={stats['eff_m_med']:.4e}"
        )
    print(
        "\nSQNR vs fp32 ref. ideal/or_rq = float output quant on mac domain. "
        "hw_rq = integer (acc*M)>>r with INT32 prod sat (ADR-015). "
        "hw_ns = same without prod sat. psat% = acc*M int32 overflow rate."
    )


if __name__ == "__main__":
    main()

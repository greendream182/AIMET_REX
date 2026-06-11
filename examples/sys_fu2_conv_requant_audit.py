#!/usr/bin/env python3
"""SYS-FU-2 FU-2-4b: decompose freq_downs Conv2d INT16 SQNR loss.

Teacher-forced on ``freq_downs.*.conv2d`` (same batch as FU-2-1):

| stage | meaning |
|---|---|
| ref | fp32 conv on float input |
| inpc_fp | fp32 conv on oracle per-channel **dequant** input (upper bound) |
| int16_e2e | INT16 dispatch end-to-end |
| mac_fp | int32 MAC + bias → float ``acc * S_x S_w`` (no output requant) |
| mac_oracle_rq | mac_fp → ideal float output quant |
| mac_hw_rq | mac_fp → ``requantize_int(M, rshift)`` (hardware path) |

Usage::

    bash scripts/run-in-container.sh python examples/sys_fu2_conv_requant_audit.py
"""

from __future__ import annotations

import argparse
import copy
import math
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
from aimet_torch.fixed_point.kernels import conv_linear as conv_linear_mod  # noqa: E402
from aimet_torch.fixed_point.metrics.flags import int16_eval_allow_debug_float  # noqa: E402
from aimet_torch.fixed_point.metrics.isolated import _to_float  # noqa: E402
from aimet_torch.fixed_point.requantize import (  # noqa: E402
    requantize_int,
    saturate_mac_accumulator,
)
from aimet_torch.fixed_point.tensor import FixedPointSimTensor, align_stat_rank  # noqa: E402
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
    _dequant_sim,
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


def _mac_dequant_float(
    acc: torch.Tensor,
    *,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
) -> torch.Tensor:
    sx = x_scale.to(device=acc.device, dtype=torch.float32).reshape(-1)
    sw = w_scale.to(device=acc.device, dtype=torch.float32).reshape(-1)
    if sx.numel() == 1:
        acc_scale = sx.item() * sw
    else:
        acc_scale = sx[0].item() * sw
    return acc.to(torch.float32) * acc_scale.view(1, -1, 1, 1)


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


def _hw_output_quant(acc: torch.Tensor, out_enc) -> torch.Tensor:
    int_repr = requantize_int(
        acc,
        out_enc.multiplier.to(device=acc.device),
        out_enc.rshift.to(device=acc.device),
        out_enc.zero_point.to(device=acc.device, dtype=torch.int32),
        out_enc.qmin,
        out_enc.qmax,
    )
    sy = align_stat_rank(
        out_enc.scale.to(device=acc.device, dtype=torch.float32), int_repr
    )
    zp = align_stat_rank(
        out_enc.zero_point.to(device=acc.device, dtype=torch.int32), int_repr
    )
    return (int_repr.to(torch.float32) - zp.to(torch.float32)) * sy


@torch.no_grad()
def _audit_one(
    module: nn.Module,
    x_float: torch.Tensor,
    *,
    percentile: float,
) -> dict[str, float]:
    with quant_execution_mode(ExecutionMode.FP32_QDQ):
        y_ref = _to_float(module(x_float))
    assert y_ref is not None

    x_pc = _sim_per_channel_nchw(x_float, percentile=percentile)
    x_pc = align_per_channel_activation_for_conv_input(x_pc)
    x_pc_dq = _dequant_sim(x_pc)

    with quant_execution_mode(ExecutionMode.FP32_QDQ):
        y_inpc_fp = _to_float(module(x_pc_dq))
    assert y_inpc_fp is not None

    _CAPTURE.clear()
    with int16_eval_allow_debug_float(), quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        with _capture_conv_acc():
            y_int16 = _to_float(module(x_pc))
    assert y_int16 is not None
    acc = _CAPTURE.get("acc")
    out_enc = _CAPTURE.get("out_enc")
    if acc is None or out_enc is None:
        raise RuntimeError("failed to capture Conv MAC accumulator")

    wq = module.param_quantizers["weight"]
    w_enc = wq.get_encodings()
    y_mac = _mac_dequant_float(acc, x_scale=x_pc.scale, w_scale=w_enc.scale)
    y_oracle = _oracle_output_quant(y_mac, out_enc)
    with int16_eval_allow_debug_float(), quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        y_hw = _hw_output_quant(acc, out_enc)

    return {
        "inpc_fp": _sqnr_db(y_ref, y_inpc_fp),
        "int16_e2e": _sqnr_db(y_ref, y_int16),
        "mac_fp": _sqnr_db(y_ref, y_mac),
        "mac_oracle_rq": _sqnr_db(y_ref, y_oracle),
        "mac_hw_rq": _sqnr_db(y_ref, y_hw),
        "hw_vs_e2e": _sqnr_db(y_int16, y_hw),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SYS-FU-2 FU-2-4b conv requant decomposition")
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
        f"\n=== SYS-FU-2 FU-2-4b Conv MAC/requant decomposition "
        f"(calib_batches={args.max_calib_batches}) ===\n"
    )
    header = (
        f"{'module':28s} {'inpc_fp':>7s} {'int16':>7s} {'mac_fp':>7s} "
        f"{'or_rq':>7s} {'hw_rq':>7s} {'hw=e2e':>7s}"
    )
    print(header)
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
            f"{stats['inpc_fp']:7.2f} {stats['int16_e2e']:7.2f} "
            f"{stats['mac_fp']:7.2f} {stats['mac_oracle_rq']:7.2f} "
            f"{stats['mac_hw_rq']:7.2f} {stats['hw_vs_e2e']:7.2f}"
        )
    print(
        "\nAll columns = SQNR (dB) vs fp32 ref. hw/or_rq replay must stay inside "
        "INT16_FIXED_EVAL (see FU-2-4c). hw=e2e checks kernel parity."
    )


if __name__ == "__main__":
    main()

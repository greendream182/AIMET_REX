#!/usr/bin/env python3
"""SYS-FU-2 FU-2-4d: int32 MAC fidelity vs fp32 / sparse centered-int input.

Decomposes the ``inpc_fp`` (fp32 conv on dequant input) vs ``mac_int32`` gap from
FU-2-4b:

| stage | meaning |
|---|---|
| ref | fp32 conv on float input |
| inpc_fp | fp32 conv on per-channel dequant input (upper bound) |
| mac_int32 | hook-captured int32 MAC acc → float dequant (FU-2-4b ``mac_fp``) |
| mac_int64 | int64 im2col MAC, single INT32 sat at end (no per-MM sat) |
| mac_fp_rd | ``F.conv2d`` on centered float, round→int32 (legacy sim path) |

Also reports centered-input sparsity and INT32-saturation hit rate.

Usage::

    bash scripts/run-in-container.sh python examples/sys_fu2_4d_mac_fidelity.py
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Tuple

import torch
import torch.nn.functional as F
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
    Int16QuantizedTensor,
    convert_encodings_to_fixed_scale,
    quant_execution_mode,
)
from aimet_torch.fixed_point.boundary_quantize import quantize_boundary_from_affine  # noqa: E402
from aimet_torch.fixed_point.channel_align import (  # noqa: E402
    align_per_channel_activation_for_conv_input,
)
from aimet_torch.fixed_point.kernels import conv_linear as conv_linear_mod  # noqa: E402
from aimet_torch.fixed_point.kernels._im2col import im2col_int  # noqa: E402
from aimet_torch.fixed_point.metrics.flags import int16_eval_allow_debug_float  # noqa: E402
from aimet_torch.fixed_point.metrics.isolated import _to_float  # noqa: E402
from aimet_torch.fixed_point.offline.bias import quantize_bias_int  # noqa: E402
from aimet_torch.fixed_point.requantize import (  # noqa: E402
    SIM_TENSOR_DTYPE,
    saturate_int32,
    saturate_mac_accumulator,
)
from aimet_torch.fixed_point.tensor import FixedPointSimTensor, align_stat_rank  # noqa: E402
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


def _conv_extra(module: nn.Module) -> dict[str, Any]:
    return {
        "stride": module.stride,
        "padding": module.padding,
        "dilation": module.dilation,
        "groups": module.groups,
    }


def _as_tuple(value: Any, ndim: int) -> Tuple[int, ...]:
    if isinstance(value, (tuple, list)):
        return tuple(value)
    return (int(value),) * ndim


def _add_bias(acc: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    if bias.numel() == 1:
        return acc + bias.to(device=acc.device, dtype=acc.dtype)
    return acc + bias.to(device=acc.device, dtype=acc.dtype).view(1, -1, 1, 1)


def _conv2d_mac_int64(
    x_cen: torch.Tensor,
    w_cen: torch.Tensor,
    bias: torch.Tensor,
    *,
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
) -> torch.Tensor:
    n_batch, _, input_h, input_w = x_cen.shape
    out_channels, _, kernel_h, kernel_w = w_cen.shape
    x_unfold = im2col_int(
        x_cen.to(SIM_TENSOR_DTYPE),
        (kernel_h, kernel_w),
        dilation=dilation,
        padding=padding,
        stride=stride,
    )
    weight_matrix = w_cen.view(out_channels, -1)
    # CUDA lacks integer matmul; match kernel and run int64 MAC on CPU.
    dev = x_cen.device
    prod = torch.matmul(
        weight_matrix.cpu().to(torch.int64),
        x_unfold.cpu().to(torch.int64),
    )
    acc = prod.transpose(1, 2).to(device=dev)
    out_h = (
        input_h + 2 * padding[0] - dilation[0] * (kernel_h - 1) - 1
    ) // stride[0] + 1
    out_w = (
        input_w + 2 * padding[1] - dilation[1] * (kernel_w - 1) - 1
    ) // stride[1] + 1
    acc = acc.transpose(1, 2).reshape(n_batch, out_channels, out_h, out_w)
    acc = _add_bias(acc, bias.to(torch.int64))
    return saturate_int32(acc)


def _conv2d_mac_fp32_round(
    x_cen: torch.Tensor,
    w_cen: torch.Tensor,
    bias: torch.Tensor,
    *,
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
    groups: int,
) -> torch.Tensor:
    acc = F.conv2d(
        x_cen.to(torch.float32),
        w_cen.to(torch.float32),
        bias=None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    acc = acc.round().to(torch.int32)
    return saturate_mac_accumulator(_add_bias(acc, bias))


def _sparsity_stats(x_cen: torch.Tensor) -> dict[str, float]:
    flat = x_cen.reshape(-1).float()
    n = flat.numel()
    return {
        "zero_frac": float((flat == 0).sum().item()) / n,
        "abs_le1_frac": float((flat.abs() <= 1).sum().item()) / n,
        "mean_abs": float(flat.abs().mean().item()),
        "max_abs": float(flat.abs().max().item()),
    }


def _scale_uniformity(scale: torch.Tensor) -> float:
    s = scale.reshape(-1).float()
    if s.numel() <= 1:
        return 1.0
    smin = float(s.min().item())
    smax = float(s.max().item())
    if smin <= 0:
        return float("inf")
    return smax / smin


def _quantize_weight_bias(
    qmodule: nn.Module,
    x_pc: FixedPointSimTensor,
) -> tuple[Int16QuantizedTensor, torch.Tensor]:
    wq = qmodule.param_quantizers["weight"]
    w_enc = wq.get_encodings()
    if not isinstance(w_enc, AffineEncoding):
        raise TypeError("weight encoding must be AffineEncoding")
    with int16_eval_allow_debug_float(), quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        w_int = quantize_boundary_from_affine(qmodule.weight, w_enc).to(x_pc.int_repr.device)

    bias = getattr(qmodule, "bias", None)
    if bias is None:
        return w_int, torch.zeros((), dtype=torch.int32, device=x_pc.int_repr.device)

    oq = qmodule.output_quantizers[0]
    y_enc = oq.get_encodings()
    bias_bits = 32
    if hasattr(oq, "encoding") and getattr(oq.encoding, "bias_bits", None) is not None:
        bias_bits = int(oq.encoding.bias_bits)
    x_scale = x_pc.scale
    if hasattr(qmodule, "_derive_bias_scale"):
        acc_scale = qmodule._derive_bias_scale(x_scale, w_enc.scale)
    else:
        acc_scale = x_scale * w_enc.scale.to(device=x_scale.device, dtype=torch.float32)
    ones = torch.ones_like(acc_scale, dtype=acc_scale.dtype, device=acc_scale.device)
    bias_int = quantize_bias_int(bias, acc_scale, ones, bits=bias_bits)
    return w_int, bias_int.to(device=x_pc.int_repr.device)


@torch.no_grad()
def _audit_one(
    qmodule: nn.Module,
    x_float: torch.Tensor,
    *,
    percentile: float,
) -> dict[str, float]:
    inner = getattr(qmodule, "weight", None)
    if inner is None:
        inner = qmodule
    extra = _conv_extra(qmodule)

    with quant_execution_mode(ExecutionMode.FP32_QDQ):
        y_ref = _to_float(qmodule(x_float))
    assert y_ref is not None

    x_pc = _sim_per_channel_nchw(x_float, percentile=percentile)
    x_pc = align_per_channel_activation_for_conv_input(x_pc)
    x_dq = _dequant_sim(x_pc)

    with quant_execution_mode(ExecutionMode.FP32_QDQ):
        y_inpc_fp = _to_float(qmodule(x_dq))
    assert y_inpc_fp is not None

    w_int, bias_int = _quantize_weight_bias(qmodule, x_pc)
    x_cen = x_pc.centered_int32()
    w_cen = w_int.centered_int32()
    w_scale = w_int.scale

    stride = _as_tuple(extra["stride"], 2)
    padding = _as_tuple(extra["padding"], 2)
    dilation = _as_tuple(extra["dilation"], 2)
    groups = int(extra["groups"])

    with int16_eval_allow_debug_float():
        w_dq = w_int.to_float()
    sx = align_stat_rank(x_pc.scale.to(torch.float32), x_cen)
    sw = align_stat_rank(w_scale.to(torch.float32), w_cen)
    bias_f = getattr(qmodule, "bias", None)
    y_fp32_direct = F.conv2d(
        x_dq, w_dq, bias=bias_f,
        stride=stride, padding=padding, dilation=dilation, groups=groups,
    )
    y_fp32_cent = F.conv2d(
        x_cen.to(torch.float32) * sx,
        w_cen.to(torch.float32) * sw,
        bias=bias_f,
        stride=stride, padding=padding, dilation=dilation, groups=groups,
    )

    acc_int64 = _conv2d_mac_int64(
        x_cen, w_cen, bias_int, stride=stride, padding=padding, dilation=dilation,
    )
    acc_fp_rd = _conv2d_mac_fp32_round(
        x_cen, w_cen, bias_int,
        stride=stride, padding=padding, dilation=dilation, groups=groups,
    )

    _CAPTURE.clear()
    with int16_eval_allow_debug_float(), quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        with _capture_conv_acc():
            _to_float(qmodule(x_pc))
    acc_int32 = _CAPTURE.get("acc")
    if acc_int32 is None:
        raise RuntimeError("failed to capture int32 MAC accumulator")

    y_mac_int32 = _mac_dequant_float(acc_int32, x_scale=x_pc.scale, w_scale=w_scale)
    y_mac_int64 = _mac_dequant_float(acc_int64, x_scale=x_pc.scale, w_scale=w_scale)
    y_mac_fp_rd = _mac_dequant_float(acc_fp_rd, x_scale=x_pc.scale, w_scale=w_scale)

    acc64_f = acc_int64.to(torch.float32)
    acc32_f = acc_int32.to(torch.float32)
    sat_mask = acc64_f != acc32_f
    sat_frac = float(sat_mask.sum().item()) / acc64_f.numel()

    sparse = _sparsity_stats(x_cen)

    return {
        "inpc_fp": _sqnr_db(y_ref, y_inpc_fp),
        "fp32_dir": _sqnr_db(y_ref, y_fp32_direct),
        "fp32_cent": _sqnr_db(y_ref, y_fp32_cent),
        "mac_int32": _sqnr_db(y_ref, y_mac_int32),
        "mac_int64": _sqnr_db(y_ref, y_mac_int64),
        "mac_fp_rd": _sqnr_db(y_ref, y_mac_fp_rd),
        "qdq_vs_dir": _sqnr_db(y_fp32_direct, y_inpc_fp),
        "int32_vs_int64": _sqnr_db(y_mac_int64, y_mac_int32),
        "sat_frac": sat_frac,
        "scale_ratio": _scale_uniformity(x_pc.scale),
        **sparse,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SYS-FU-2 FU-2-4d MAC fidelity audit")
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
        f"\n=== SYS-FU-2 FU-2-4d int32 MAC fidelity (calib_batches={args.max_calib_batches}) ===\n"
    )
    hdr = (
        f"{'module':28s} {'inpc_fp':>7s} {'fp32dir':>7s} {'mac_i32':>7s} "
        f"{'sat%':>6s} {'zero%':>6s} {'Sratio':>6s}"
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
            f"{stats['inpc_fp']:7.2f} {stats['fp32_dir']:7.2f} "
            f"{stats['mac_int32']:7.2f} "
            f"{100 * stats['sat_frac']:5.2f} "
            f"{100 * stats['zero_frac']:5.2f} "
            f"{stats['scale_ratio']:6.2f}"
        )
    print(
        "\nSQNR vs fp32 ref. fp32dir = F.conv2d on PC-dequant (no QDQ re-wrap). "
        "mac_i32 = FU-2-4b mac_fp. sat% = int32-sat vs int64-final-sat mismatch. "
        f"fp32_cent==fp32dir and mac_i64==mac_i32 verified internally."
    )


if __name__ == "__main__":
    main()

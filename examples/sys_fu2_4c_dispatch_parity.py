#!/usr/bin/env python3
"""FU-2-4c: dispatch return vs hook acc-requant parity.

Verifies:
1. ``to_float(module(x))`` == manual dequant of kernel ``FixedPointSimTensor``.
2. ``requantize_int`` replay on captured acc must stay inside ``INT16_FIXED_EVAL``
   (outside that mode ``requantize_int32_prod_sat`` defaults differ → false +48 dB).
"""

from __future__ import annotations

import copy
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_REPO / "examples"), str(_REPO)]
_QG = _REPO.parent / "quant-gru-pytorch" / "pytorch"
if _QG.is_dir():
    sys.path.insert(0, str(_QG))

import aimet_torch.v2 as aimet
import aimet_torch.fixed_point.kernels  # noqa: F401
from aimet_torch.fixed_point import ExecutionMode, convert_encodings_to_fixed_scale, quant_execution_mode
from aimet_torch.fixed_point.channel_align import align_per_channel_activation_for_conv_input
from aimet_torch.fixed_point.kernels import conv_linear as conv_linear_mod
from aimet_torch.fixed_point.metrics.flags import int16_eval_allow_debug_float
from aimet_torch.fixed_point.metrics.isolated import _to_float
from aimet_torch.fixed_point.requantize import requantize_int, saturate_mac_accumulator
from aimet_torch.fixed_point.tensor import FixedPointSimTensor, align_stat_rank
from common.mrnn_clz_encoding import (
    apply_mrnn_clz_encoding_fixes,
    apply_mrnn_clz_encoding_fixes_post_calib,
    collect_power2_float_out_fmax,
)
from quick_start import MRNN, FP_MODEL_PATH, set_seed, setup_audio_backend
from quick_start_int16_metric import build_sim, _calib_fn, _patch_torchaudio_with_soundfile
from sys_fu2_per_channel_audit import (
    _collect_conv_inputs,
    _find_module,
    _sim_per_channel_nchw,
    _sqnr_db,
)

_CAP: dict[str, Any] = {"n": 0}


@contextmanager
def _capture():
    orig = conv_linear_mod._requantize_output

    _CAP.clear()
    _CAP["n"] = 0

    def _hook(acc, enc):
        n = _CAP.get("n", 0)
        _CAP["n"] = n + 1
        acc_c = acc.detach().clone()
        torch.cuda.synchronize()
        same = torch.equal(acc, acc_c)
        out_live = orig(acc, enc)
        out_clone = orig(acc_c, enc)
        _CAP[f"acc_{n}"] = acc_c
        _CAP[f"enc_{n}"] = enc
        _CAP[f"out_live_{n}"] = out_live
        _CAP[f"out_clone_{n}"] = out_clone
        _CAP[f"acc_eq_clone_{n}"] = same
        _CAP[f"live_eq_clone_{n}"] = torch.equal(out_live.int_repr, out_clone.int_repr)
        _CAP["kernel_out"] = out_live
        _CAP["out_enc"] = enc
        return out_live

    conv_linear_mod._requantize_output = _hook
    try:
        yield
    finally:
        conv_linear_mod._requantize_output = orig


def _dequant_carrier(t: FixedPointSimTensor) -> torch.Tensor:
    sy = align_stat_rank(t.scale.to(torch.float32), t.int_repr)
    zp = align_stat_rank(t.zero_point.to(torch.int32), t.int_repr)
    return (t.int_repr.to(torch.float32) - zp.to(torch.float32)) * sy


def _hw_from_acc(acc: torch.Tensor, enc, *, saturate: bool) -> torch.Tensor:
    a = saturate_mac_accumulator(acc) if saturate else acc
    with int16_eval_allow_debug_float(), quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        q = requantize_int(
            a,
            enc.multiplier.to(acc.device),
            enc.rshift.to(acc.device),
            enc.zero_point.to(acc.device, dtype=torch.int32),
            enc.qmin,
            enc.qmax,
        )
    sy = align_stat_rank(enc.scale.to(acc.device, dtype=torch.float32), q)
    zp = align_stat_rank(enc.zero_point.to(acc.device, dtype=torch.int32), q)
    return (q.to(torch.float32) - zp.to(torch.float32)) * sy


def main() -> None:
    set_seed(0)
    setup_audio_backend()
    _patch_torchaudio_with_soundfile()
    device = torch.device("cuda")
    model = MRNN().to(device)
    model.load_state_dict(torch.load(FP_MODEL_PATH, map_location=device, weights_only=True))
    import quick_start as qs

    qs.BATCH_SIZE = 32
    loaders = qs.build_dataloaders(os.environ.get("SPEECH_COMMANDS_ROOT", "/home/llq/workspace/data/speech_commands"))
    sample = next(iter(loaders["calib"]))[0][:1].to(device)
    sim, pf = build_sim(copy.deepcopy(model), sample, native_trans=True)
    sim.model.to(device).eval()
    apply_mrnn_clz_encoding_fixes(sim.model, sign_input_bypass=False, verbose=False)
    calib = _calib_fn(sim.model, loaders["calib"], device, 16)
    with torch.no_grad(), aimet.nn.compute_encodings(sim.model):
        calib(sim.model)
    if pf is not None:
        apply_mrnn_clz_encoding_fixes_post_calib(
            sim.model,
            power2_float_out_fmax=collect_power2_float_out_fmax(pf.to(device), loaders["calib"], device, 16),
            verbose=False,
        )
    convert_encodings_to_fixed_scale(sim)

    suffix = "freq_downs.0.conv2d"
    qual, mod = _find_module(sim.model, suffix)
    xs = _collect_conv_inputs(sim.model, sample, (suffix,))
    xf = xs[qual]
    x_pc = align_per_channel_activation_for_conv_input(_sim_per_channel_nchw(xf))

    with quant_execution_mode(ExecutionMode.FP32_QDQ):
        y_ref = _to_float(mod(xf))

    _CAP.clear()
    _CAP["n"] = 0
    with int16_eval_allow_debug_float(), quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        with _capture():
            mod_out = mod(x_pc)

    print("mod_out type:", type(mod_out).__name__)
    kout: FixedPointSimTensor = _CAP["kernel_out"]
    enc = _CAP["out_enc"]
    print("requantize calls:", _CAP["n"])
    for i in range(_CAP["n"]):
        print(
            f"  call {i}: acc_eq_clone={_CAP.get(f'acc_eq_clone_{i}')} "
            f"live_eq_clone={_CAP.get(f'live_eq_clone_{i}')} "
            f"max|live-clone|={(_CAP[f'out_live_{i}'].int_repr - _CAP[f'out_clone_{i}'].int_repr).abs().max().item()}"
        )
    acc = _CAP.get("acc_0")

    with int16_eval_allow_debug_float():
        y_to_float = _to_float(mod_out)
    y_manual = _dequant_carrier(kout)
    y_hw_sat = _hw_from_acc(acc, enc, saturate=True)
    y_hw_nosat = _hw_from_acc(acc, enc, saturate=False)

    print("scale shape", tuple(kout.scale.shape), "axis", kout.axis)
    print("out_enc scale shape", tuple(enc.scale.shape))
    print("rms ref", float(y_ref.pow(2).mean().sqrt()))
    for name, y in [
        ("to_float(mod_out)", y_to_float),
        ("manual dequant(kernel_out)", y_manual),
        ("hw(acc,sat=1)", y_hw_sat),
        ("hw(acc,sat=0)", y_hw_nosat),
    ]:
        print(
            f"{name:28s} rms={float(y.pow(2).mean().sqrt()):.4f} "
            f"sqnr_vs_ref={_sqnr_db(y_ref, y):.2f} "
            f"max|diff-kernel|={(y - y_manual).abs().max():.4e}"
        )
    from aimet_torch.fixed_point.kernels.conv_linear import _requantize_output

    with int16_eval_allow_debug_float(), quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        replay_in = _requantize_output(acc, enc)
    replay_out = _requantize_output(acc, enc)

    print("replay inside INT16 == kout?", torch.equal(kout.int_repr, replay_in.int_repr))
    print("replay outside INT16 == kout?", torch.equal(kout.int_repr, replay_out.int_repr))
    print(
        "sqnr replay-outside (false positive if used in audit):",
        f"{_sqnr_db(y_ref, _dequant_carrier(replay_out)):.2f} dB",
    )


if __name__ == "__main__":
    main()

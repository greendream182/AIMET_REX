#!/usr/bin/env python3
"""Probe sign/mul int_repr vs fixed_scale grid on one calib batch (R3 frontend)."""
from __future__ import annotations

import contextlib
import os
import sys

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

import aimet_torch.fixed_point.kernels  # noqa: F401
import quick_start as qs
from aimet_torch.fixed_point import (
    ExecutionMode,
    convert_encodings_to_fixed_scale,
    ensure_output_quantizers_for_int16_eval,
    quant_execution_mode,
)
from aimet_torch.fixed_point.metrics import int16_eval_allow_debug_float
from aimet_torch.fixed_point.metrics.accuracy import quantize_float_to_grid

_DATA = "/home/llq/workspace/data/speech_commands"
qs.DATA_ROOT = _DATA
qs.FP_EPOCHS = 0
qs.QAT_EPOCHS = 1
qs.MAX_CALIB_BATCHES = 20


def build_sim(device):
    loaders = qs.build_dataloaders(_DATA)
    ckpt = torch.load(qs.FP_MODEL_PATH, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    fp = qs.MRNN(output_dim=qs.NUM_CLASSES).to(device)
    fp.load_state_dict(state, strict=False)
    prepared = qs.model_preparer.prepare_model(
        fp,
        stateless_modules_to_preserve=[
            qs.PowerCompress,
            qs.HypotFun,
            qs.CLN,
            qs.QuantizableBatchNorm2d,
        ],
    )
    dummy = next(iter(loaders["train"]))[0].to(device)
    sim = qs.quantsim.QuantizationSimModel(
        prepared,
        dummy_input=dummy,
        quant_scheme=qs.QUANT_SCHEME,
        config_file=str(qs.CONFIG_FILE),
        default_output_bw=qs.DEFAULT_BW,
        default_param_bw=qs.DEFAULT_BW,
    )
    sim.set_percentile_value(qs.PERCENTILE_VALUE)
    ensure_output_quantizers_for_int16_eval(sim)
    qs.apply_mixed_precision_bitwidth(
        sim.model, config_file=str(qs.BITWIDTH_CONFIG_FILE), verbose=False
    )
    sim.model.to(device).eval()
    with torch.no_grad(), qs.aimet.nn.compute_encodings(sim.model):
        for i, (x, _) in enumerate(loaders["calib"]):
            if i >= qs.MAX_CALIB_BATCHES:
                break
            sim.model(x.to(device))
    qs.apply_power_of_2_workflow(
        sim.model, method="round", tolerance=0.02, align_bias_scale=True, verbose=False
    )
    qs.freeze_quantizer_parameters(sim.model, verbose=False, freeze_bn_affine=True)
    qs.qat_finetune(sim, loaders["train"], device)
    convert_encodings_to_fixed_scale(sim)
    return sim, loaders


def _grid_from_float(y_f: torch.Tensor, enc) -> torch.Tensor:
    return quantize_float_to_grid(
        y_f,
        enc.scale,
        enc.zero_point,
        enc.qmin,
        enc.qmax,
    ).to(torch.int32)


def main() -> None:
    device = qs.DEVICE
    sim, loaders = build_sim(device)
    x = next(iter(loaders["test"]))[0][:2].to(device)

    hooks = {}
    captures: dict[str, dict] = {}

    def make_hook(name):
        def _hook(_mod, _inp, out):
            out_type = type(out).__name__
            if hasattr(out, "int_repr"):
                y_f = out.to_float() if hasattr(out, "to_float") else out.dequantize()
                y_i = out.int_repr.to(torch.int32)
                y_c = out.centered_int32()
            elif isinstance(out, torch.Tensor):
                y_f = out.detach()
                y_i = None
                y_c = None
            else:
                return
            captures[name] = {
                "type": out_type,
                "float": y_f.detach().cpu(),
                "int_repr": y_i.cpu() if y_i is not None else None,
                "centered": y_c.cpu() if y_c is not None else None,
            }

        return _hook

    for name in (
        "power_compress_1.module_sign",
        "power_compress_1.module_abs_1",
        "power_compress_1.module_sqrt",
        "power_compress_1.module_mul",
    ):
        mod = dict(sim.model.named_modules())[name]
        hooks[name] = mod.register_forward_hook(make_hook(name))

    with torch.no_grad(), int16_eval_allow_debug_float(), quant_execution_mode(
        ExecutionMode.INT16_FIXED_EVAL
    ):
        sim.model(x)
    cap_i16 = {k: v.copy() for k, v in captures.items()}

    captures.clear()
    with torch.no_grad(), quant_execution_mode(ExecutionMode.FIXED_SCALE_QDQ):
        sim.model(x)
    cap_fix = {k: v.copy() for k, v in captures.items()}

    for h in hooks.values():
        h.remove()

    print("=== int_repr / centered vs fixed_scale grid (batch=2) ===")
    for name in cap_i16:
        a = cap_i16[name]
        b = cap_fix[name]
        y_f_i16 = a["float"]
        y_f_fix = b["float"]
        cos = torch.nn.functional.cosine_similarity(
            y_f_i16.reshape(1, -1), y_f_fix.reshape(1, -1)
        ).item()
        if a["int_repr"] is not None and b["int_repr"] is not None:
            match = (a["int_repr"] == b["int_repr"]).float().mean().item()
            c_match = (a["centered"] == b["centered"]).float().mean().item()
            max_i = (a["int_repr"] - b["int_repr"]).abs().max().item()
            print(
                f"{name:35s} [{a['type']}] cos={cos:.6f} int_repr_match={match*100:.2f}% "
                f"centered_match={c_match*100:.2f}% max|Δint|={max_i}"
            )
        else:
            print(f"{name:35s} [{a['type']}] cos={cos:.6f} (no int_repr)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""排查 sign / sqrt / square / div 单算子 vs float 偏差：逐步对比 Q 前后与符号一致率。"""
from __future__ import annotations

import copy
import math
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
import torch.nn as nn
import torchaudio

_DATA = "/home/llq/workspace/data/speech_commands"


def _soundfile_load(path, *args, **kwargs):
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    return torch.from_numpy(data.T.copy()), sr


torchaudio.load = _soundfile_load

import quick_start as qs
from aimet_torch.fixed_point import ExecutionMode, convert_encodings_to_fixed_scale, quant_execution_mode

import int16_single_op_vs_float_native as iso


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.reshape(-1).float()
    bf = b.reshape(-1).float()
    d = (af.norm() * bf.norm()).clamp_min(1e-30)
    v = float(torch.dot(af, bf).item() / float(d))
    return v if math.isfinite(v) else float("nan")


def _sign_agreement(a: torch.Tensor, b: torch.Tensor) -> float:
    sa = torch.sign(a.reshape(-1))
    sb = torch.sign(b.reshape(-1))
    return float((sa == sb).float().mean().item())


def _capture_io(model: nn.Module, x, names: set[str]):
    ins, outs = {}, {}

    def hook(n):
        def _h(_m, inp, out):
            if n not in names:
                return
            ins[n] = inp[0].detach().float().clone() if inp and isinstance(inp[0], torch.Tensor) else None
            if isinstance(out, torch.Tensor):
                outs[n] = out.detach().float().clone()
            elif hasattr(out, "dequantize"):
                outs[n] = out.dequantize().detach().float().clone()
        return _h

    hooks = [m.register_forward_hook(hook(n)) for n, m in model.named_modules() if n in names]
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    return ins, outs


def _probe_module(
    name: str,
    float_mod: nn.Module,
    quant_mod: nn.Module,
    x: torch.Tensor,
) -> dict:
    x = x.detach()
    with torch.no_grad():
        y_ref = float_mod(x)
        if not isinstance(y_ref, torch.Tensor):
            y_ref = y_ref[0] if isinstance(y_ref, (tuple, list)) else y_ref
        y_ref = y_ref.float()

        with quant_execution_mode(ExecutionMode.FP32_QDQ):
            y_q = quant_mod(x)
        if hasattr(y_q, "dequantize"):
            y_q = y_q.dequantize()
        y_q = y_q.float()

        # 纯 float sign/sqrt 作用于「Q 输入后」的值 — 分离边界误差 vs 算子误差
        x_q = x
        iq = getattr(quant_mod, "input_quantizers", [None])[0]
        if iq is not None and iq.is_initialized():
            x_q = iq(x).dequantize() if hasattr(iq(x), "dequantize") else iq(x)
            x_q = x_q.float()

        y_ref_on_qin = None
        if "sign" in name:
            y_ref_on_qin = torch.sign(x_q)
        elif "square" in name:
            y_ref_on_qin = torch.square(x_q)
        elif "sqrt" in name and "square" not in name:
            y_ref_on_qin = torch.sqrt(x_q.clamp(min=qs.EPS))
        elif "div" in name:
            y_ref_on_qin = x_q  # div 需两输入，下面单独处理

    diff = y_q - y_ref
    out = {
        "name": name,
        "x_min": float(x.min()),
        "x_max": float(x.max()),
        "x_near_zero_frac": float((x.abs() < 1e-6).float().mean()),
        "y_ref_min": float(y_ref.min()),
        "y_ref_max": float(y_ref.max()),
        "y_q_min": float(y_q.min()),
        "y_q_max": float(y_q.max()),
        "cos_vs_float": _cos(y_ref, y_q),
        "max_abs": float(diff.abs().max()),
        "rmse": float(diff.pow(2).mean().sqrt()),
        "ref_norm": float(y_ref.norm()),
        "q_norm": float(y_q.norm()),
    }
    if "sign" in name:
        out["sign_agreement_vs_float"] = _sign_agreement(y_ref, y_q)
        out["sign_agreement_qin_vs_float"] = _sign_agreement(torch.sign(x), torch.sign(x_q))
        if y_ref_on_qin is not None:
            out["cos_sign_on_qinput"] = _cos(y_ref_on_qin, y_q)
            out["sign_agreement_on_qinput"] = _sign_agreement(y_ref_on_qin, y_q)
    if y_ref_on_qin is not None and "div" not in name:
        out["cos_float_op_on_qinput"] = _cos(y_ref_on_qin, y_q)
    return out


def main():
    qs.DATA_ROOT = _DATA
    qs.FP_EPOCHS = 0
    qs.MAX_CALIB_BATCHES = 20
    qs.set_seed(qs.SEED)
    qs.setup_audio_backend()
    device = qs.DEVICE

    loaders = qs.build_dataloaders(_DATA)
    ckpt = torch.load(qs.FP_MODEL_PATH, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    fp = qs.MRNN(output_dim=qs.NUM_CLASSES).to(device)
    fp.load_state_dict(state, strict=False)

    prepared = qs.model_preparer.prepare_model(
        copy.deepcopy(fp),
        stateless_modules_to_preserve=[
            qs.PowerCompress, qs.HypotFun, qs.CLN, qs.QuantizableBatchNorm2d,
        ],
    )
    prepared_float = copy.deepcopy(prepared).to(device).eval()
    sim, _ = iso.build_sim(fp, loaders, device, skip_qat=True, qat_epochs=1, max_calib=20)

    x, _ = next(iter(loaders["calib"]))
    x = x[:2].to(device)

    targets = [
        "power_compress_1.module_sign",
        "power_compress_1.module_sqrt",
        "power_compress_1.module_mul",
        "enc_seqs.0.cln.module_square_2",
        "hypot_fun.module_sqrt_1",
        "enc_seqs.0.rnn2d_bn.module_div_1",
        "pre_bn.module_div",
        "freq_downs.0.conv2d",
    ]

    float_ins, float_outs = iso._capture_prepared_float_io(
        prepared_float, (x,), set(targets),
    )

    print("=" * 78)
    print("单算子排查：teacher-forced 同一 x，float 模块 vs Quantized fp32_qdq")
    print("=" * 78)

    for name in targets:
        if name not in float_ins:
            print(f"\n[{name}] SKIP — no float IO")
            continue
        x_in = float_ins[name][0]
        qmod = dict(sim.model.named_modules()).get(name)
        fmod = dict(prepared_float.named_modules()).get(name)
        if qmod is None or fmod is None:
            print(f"\n[{name}] SKIP — module missing")
            continue

        if "div" in name and len(float_ins[name]) >= 2:
            a, b = float_ins[name][0], float_ins[name][1]
            with torch.no_grad():
                y_ref = fmod(a, b) if b is not None else fmod(a)
                with quant_execution_mode(ExecutionMode.FP32_QDQ):
                    y_q = qmod(a, b) if b is not None else qmod(a)
                if hasattr(y_q, "dequantize"):
                    y_q = y_q.dequantize()
                y_ref, y_q = y_ref.float(), y_q.float()
            print(f"\n[{name}]")
            print(f"  cos={_cos(y_ref, y_q):.6f} max_abs={(y_q-y_ref).abs().max():.4g} "
                  f"ref_norm={y_ref.norm():.4g} q_norm={y_q.norm():.4g}")
            print(f"  denom b: min={b.min():.4g} max={b.max():.4g} near_zero={(b.abs()<1e-8).float().mean():.4g}")
            print(f"  ref allclose0={bool((y_ref.abs()<1e-6).all())} q allclose0={bool((y_q.abs()<1e-6).all())}")
            continue

        if "mul" in name and len(float_ins.get(name, ())) >= 2:
            a, b = float_ins[name][0], float_ins[name][1]
            with torch.no_grad():
                y_ref = fmod(a, b).float()
                with quant_execution_mode(ExecutionMode.FP32_QDQ):
                    y_q = qmod(a, b)
                if hasattr(y_q, "dequantize"):
                    y_q = y_q.dequantize()
                y_q = y_q.float()
            print(f"\n[{name}]")
            print(f"  cos={_cos(y_ref, y_q):.6f} max_abs={(y_q-y_ref).abs().max():.4g}")
            continue

        if "mul" in name:
            print(f"\n[{name}] SKIP — need 2 inputs")
            continue

        r = _probe_module(name, fmod, qmod, x_in)
        print(f"\n[{name}]")
        for k, v in r.items():
            if k != "name":
                print(f"  {k}: {v}")

    # div 上游：square + sub 链
    print("\n--- CLN/BN div 上游（enc_seqs.0.rnn2d_bn）---")
    chain = [
        "enc_seqs.0.rnn2d_bn.module_sub_1",
        "enc_seqs.0.rnn2d_bn.module_square_1",
        "enc_seqs.0.rnn2d_bn.module_add",
        "enc_seqs.0.rnn2d_bn.module_sqrt",
        "enc_seqs.0.rnn2d_bn.module_div_1",
    ]
    fi, fo = iso._capture_prepared_float_io(prepared_float, (x,), set(chain))
    for n in chain:
        if n in fo:
            t = fo[n]
            print(f"  {n}: min={t.min():.4g} max={t.max():.4g} std={t.std():.4g} norm={t.norm():.4g}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""INT16 大块二分：判断 int16_fixed_eval 崩盘来自无参 functional 还是 conv/GRU/边界 carrier。

在 QAT 后、同一 sim 上切换 surrogate 标记，只评 test 子集（默认 100 batch）。
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys

os.environ.setdefault("PYTHONHASHSEED", "42")

import soundfile as sf
import torch
import torchaudio

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_QG = os.path.abspath(os.path.join(_REPO, "..", "quant-gru-pytorch", "pytorch"))
if os.path.isdir(_QG) and _QG not in sys.path:
    sys.path.insert(0, _QG)


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
from aimet_torch.fixed_point.tensor import Int16QuantizedTensor
from aimet_torch.v2.nn.fake_quant._legacy_impl import FakeQuantizationMixin
from aimet_torch.v2.nn.true_quant import QuantizationMixin
from aimet_torch.v2.quantization.affine.encoding import AffineEncoding
from aimet_torch.v2.quantization.affine.fixed_point import adapter
from aimet_torch.v2.quantization.base import QuantizerBase
from aimet_torch.v2.quantization.tensor import QuantizedTensorBase

_orig_dispatch = adapter.dispatch_int16_fixed


def _is_paramless_quantized(module: torch.nn.Module) -> bool:
    if not type(module).__name__.startswith("Quantized"):
        return False
    pq = getattr(module, "param_quantizers", None)
    if pq is not None and any(v is not None for v in pq.values()):
        return False
    return True


def _has_initialized_output_quantizer(module: torch.nn.Module) -> bool:
    oqs = getattr(module, "output_quantizers", None)
    if not oqs:
        return False
    oq = oqs[0] if len(oqs) else None
    return isinstance(oq, QuantizerBase) and oq.is_initialized()


def _float_tensor(x):
    if isinstance(x, Int16QuantizedTensor):
        return x.to_float(torch.float32)
    if isinstance(x, QuantizedTensorBase):
        return x.dequantize().to(torch.float32)
    if isinstance(x, torch.Tensor) and x.is_floating_point():
        return x.to(torch.float32)
    return x


def _surrogate_extra(qmodule, base_cls, args, kwargs):
    from aimet_torch._base.nn.modules import custom

    extra: dict = {}
    if isinstance(qmodule, torch.nn.Conv2d):
        extra.update(
            stride=qmodule.stride,
            padding=qmodule.padding,
            dilation=qmodule.dilation,
            groups=qmodule.groups,
        )
    elif isinstance(qmodule, torch.nn.Conv1d):
        extra.update(
            stride=qmodule.stride,
            padding=qmodule.padding,
            dilation=qmodule.dilation,
            groups=qmodule.groups,
        )
    elif isinstance(qmodule, torch.nn.Flatten):
        extra.update(start_dim=qmodule.start_dim, end_dim=qmodule.end_dim)
    elif base_cls is custom.Reshape and len(args) > 1:
        shape = args[1]
        if isinstance(shape, torch.Tensor):
            shape = tuple(int(v) for v in shape.detach().cpu().reshape(-1).tolist())
        extra["shape"] = tuple(shape)
    elif base_cls in (custom.Clamp, custom.Clip):
        extra["min"] = kwargs.get("min", args[1] if len(args) > 1 else None)
        extra["max"] = kwargs.get("max", args[2] if len(args) > 2 else None)
    elif base_cls is custom.Mean:
        extra["dim"] = kwargs.get("dim", args[1] if len(args) > 1 else None)
        extra["keepdim"] = kwargs.get("keepdim", args[2] if len(args) > 2 else False)
    elif base_cls is custom.AdaptiveAvgPool2d:
        extra["output_size"] = kwargs.get("output_size", args[1] if len(args) > 1 else (1, 1))
    elif isinstance(qmodule, (torch.nn.MaxPool2d, torch.nn.AvgPool2d)):
        extra.update(
            kernel_size=qmodule.kernel_size,
            stride=qmodule.stride,
            padding=qmodule.padding,
            dilation=getattr(qmodule, "dilation", 1),
            ceil_mode=qmodule.ceil_mode,
        )
    return extra


def _patched_dispatch(qmodule, *args, **kwargs):
    if not getattr(qmodule, "_debug_eval_surrogate", False):
        return _orig_dispatch(qmodule, *args, **kwargs)

    base_cls = QuantizationMixin.qcls_to_cls.get(type(qmodule))
    if base_cls is None:
        base_cls = FakeQuantizationMixin.qcls_to_cls.get(type(qmodule))
    oqs = getattr(qmodule, "output_quantizers", None)
    oq = oqs[0] if oqs else None
    if base_cls is None or not isinstance(oq, QuantizerBase) or not oq.is_initialized():
        return _orig_dispatch(qmodule, *args, **kwargs)
    y_enc = oq.get_encodings()
    if not isinstance(y_enc, AffineEncoding):
        return _orig_dispatch(qmodule, *args, **kwargs)

    n_in = len(getattr(qmodule, "input_quantizers", []))
    sur_in = [_float_tensor(a) for a in args[:n_in]]
    if any(not (isinstance(t, torch.Tensor) and t.is_floating_point()) for t in sur_in):
        return _orig_dispatch(qmodule, *args, **kwargs)

    out = adapter._qat_surrogate_float(
        qmodule, base_cls, sur_in, {}, _surrogate_extra(qmodule, base_cls, args, kwargs)
    )
    if out is None:
        return _orig_dispatch(qmodule, *args, **kwargs)
    return Int16QuantizedTensor.from_affine_encoding(out, y_enc)


adapter.dispatch_int16_fixed = _patched_dispatch

_FRONTEND_PREFIXES = (
    "trans.",
    "power_compress_1.",
    "pre_bn.",
    "hypot_fun.",
    "fft2band.",
    "power_compress_2.",
    "module_clamp",
)

_BN_DECOMP_SUFFIXES = (".module_sub", ".module_div", ".module_mul_1", ".module_add")


def _base_cls(module: torch.nn.Module):
    bc = QuantizationMixin.qcls_to_cls.get(type(module))
    if bc is None:
        bc = FakeQuantizationMixin.qcls_to_cls.get(type(module))
    return bc


def _op_category(name: str, module: torch.nn.Module) -> str | None:
    """Classify param-less Quantized* ops for third-level bisection."""
    from aimet_torch._base.nn.modules import custom

    bc = _base_cls(module)
    if bc is None:
        return None
    if any(s in name for s in _BN_DECOMP_SUFFIXES) and any(
        p in name for p in ("pre_bn.", "cln.", "rnn2d_bn.")
    ):
        return "bn_decomp"
    if bc in (custom.Sqrt, custom.RSqrt, custom.Square, custom.Reciprocal):
        return "pwl"
    if bc in (custom.Sin, custom.Cos, custom.Log, custom.Exponential):
        return "lut"
    if bc is custom.MatMul:
        return "matmul"
    if bc in (custom.Add, custom.Subtract, custom.Multiply, custom.Divide):
        return "eltwise"
    if bc is custom.Abs:
        return "abs"
    if bc is custom.ElementwiseUnarySign:
        return "sign"
    if bc in (custom.Clamp, custom.Clip):
        return "clamp"
    if bc in (custom.Mean,):
        return "reduce"
    if bc in (nn.AvgPool2d, custom.AdaptiveAvgPool2d, nn.MaxPool2d):
        return "pool"
    if bc in (custom.Pad, custom.Reshape, custom.FloorDivide, nn.Flatten, custom.Concat):
        return "shape"
    return "other"


def _is_stateless_int16(module: torch.nn.Module) -> bool:
    return _is_paramless_quantized(module) and _has_initialized_output_quantizer(module)


def _scope_match(name: str, module: torch.nn.Module, scope: str) -> bool:
    if scope == "none":
        return False
    if scope == "all_stateless":
        return _is_stateless_int16(module)
    if scope == "all_quantized":
        return _has_initialized_output_quantizer(module)
    if scope == "quantgru":
        return type(module).__name__ == "QuantizedQuantGRU"
    if scope == "frontend_stateless":
        if not _is_stateless_int16(module):
            return False
        return any(name.startswith(p) or name == p.rstrip(".") for p in _FRONTEND_PREFIXES)
    if scope == "backend_weighted":
        if not _has_initialized_output_quantizer(module):
            return False
        if _is_paramless_quantized(module):
            return False
        if type(module).__name__ == "QuantizedQuantGRU":
            return False
        return True
    if scope == "conv_only":
        return type(module).__name__.startswith("QuantizedConv")

    # Third level: only category X uses surrogate (rest stay INT16).
    if scope.startswith("only_"):
        cat = scope.removeprefix("only_")
        if cat == "unary":
            return _is_stateless_int16(module) and _op_category(name, module) in ("sign", "abs")
        return _is_stateless_int16(module) and _op_category(name, module) == cat

    # All stateless surrogate EXCEPT category X (X stays INT16 — if acc drops, X is suspect).
    if scope.startswith("except_"):
        cat = scope.removeprefix("except_")
        if not _is_stateless_int16(module):
            return False
        if cat == "unary":
            return _op_category(name, module) not in ("sign", "abs")
        return _op_category(name, module) != cat

    if scope.startswith("int16_only:"):
        cats = frozenset(scope.split(":", 1)[1].split("+"))
        if not _is_stateless_int16(module):
            return False
        # True → surrogate; keep listed categories on real INT16 kernels.
        return _op_category(name, module) not in cats

    raise ValueError(f"unknown scope: {scope}")


def build_qat_sim(device: torch.device, data_root: str, max_calib: int):
    loaders = qs.build_dataloaders(data_root)
    ckpt = torch.load(qs.FP_MODEL_PATH, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    fp = qs.MRNN(output_dim=qs.NUM_CLASSES).to(device)
    fp.load_state_dict(state, strict=False)
    fp_acc = qs.evaluate(fp, loaders["test"], device)

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
    n_patch = ensure_output_quantizers_for_int16_eval(sim)
    qs.apply_mixed_precision_bitwidth(
        sim.model, config_file=str(qs.BITWIDTH_CONFIG_FILE), verbose=False
    )
    sim.model.to(device).eval()
    with torch.no_grad(), qs.aimet.nn.compute_encodings(sim.model):
        for i, (x, _) in enumerate(loaders["calib"]):
            if i >= max_calib:
                break
            sim.model(x.to(device))
    qs.apply_power_of_2_workflow(
        sim.model, method="round", tolerance=0.02, align_bias_scale=True, verbose=False
    )
    qs.freeze_quantizer_parameters(sim.model, verbose=False, freeze_bn_affine=True)
    qs.qat_finetune(sim, loaders["train"], device)
    convert_encodings_to_fixed_scale(sim)
    return sim, loaders, fp_acc, n_patch


def eval_top1(model, loader, device, mode: ExecutionMode, max_batches: int | None) -> float:
    ctx = (
        int16_eval_allow_debug_float()
        if mode is ExecutionMode.INT16_FIXED_EVAL
        else contextlib.nullcontext()
    )
    correct = total = 0
    model.eval()
    with torch.no_grad(), ctx, quant_execution_mode(mode):
        for i, (x, y) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            out = model(x.to(device))
            if hasattr(out, "to_float"):
                out = out.to_float()
            pred = out.max(1).indices.cpu()
            correct += pred.eq(y).sum().item()
            total += y.numel()
    return correct / total if total else 0.0


def apply_scope(model: torch.nn.Module, scope: str) -> list[str]:
    marked: list[str] = []
    for name, m in model.named_modules():
        if hasattr(m, "_debug_eval_surrogate"):
            delattr(m, "_debug_eval_surrogate")
        if _scope_match(name, m, scope):
            setattr(m, "_debug_eval_surrogate", True)
            marked.append(name)
    return marked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/home/llq/workspace/data/speech_commands")
    parser.add_argument("--max-calib-batches", type=int, default=100)
    parser.add_argument("--max-eval-batches", type=int, default=100)
    parser.add_argument("--qat-epochs", type=int, default=1)
    parser.add_argument(
        "--scopes",
        default="",
        help="逗号分隔 scope 列表；空=跑默认全套；支持 int16_only:pwl+bn_decomp",
    )
    args = parser.parse_args()

    qs.DATA_ROOT = args.data_root
    qs.FP_EPOCHS = 0
    qs.QAT_EPOCHS = args.qat_epochs
    qs.MAX_CALIB_BATCHES = args.max_calib_batches
    qs.set_seed(qs.SEED)
    qs.setup_audio_backend()
    device = qs.DEVICE

    print("=" * 72, flush=True)
    print("INT16 大块二分（surrogate 在 int16_fixed_eval 内替换 kernel）", flush=True)
    print("=" * 72, flush=True)

    sim, loaders, fp_acc, n_ensure = build_qat_sim(device, args.data_root, args.max_calib_batches)
    print(f"float_native={fp_acc * 100:.2f}% ensure_patched={n_ensure}", flush=True)

    mb = args.max_eval_batches
    fp32 = eval_top1(sim.model, loaders["test"], device, ExecutionMode.FP32_QDQ, mb)
    fixed = eval_top1(sim.model, loaders["test"], device, ExecutionMode.FIXED_SCALE_QDQ, mb)
    print(f"QAT+eval{mb} fp32_qdq={fp32 * 100:.2f}% fixed_scale_qdq={fixed * 100:.2f}%", flush=True)

    default_scopes = [
        "none",
        "all_stateless",
        "only_sign",
        "only_abs",
        "only_unary",
        "except_sign",
        "except_abs",
        "except_unary",
        "backend_weighted",
        "quantgru",
        "all_quantized",
    ]
    scopes = (
        [s.strip() for s in args.scopes.split(",") if s.strip()]
        if args.scopes
        else default_scopes
    )
    print(f"\n{'scope':22s} {'marked':>6s} {'int16':>8s} {'Δ vs fixed':>12s}", flush=True)
    print("-" * 54, flush=True)
    for scope in scopes:
        names = apply_scope(sim.model, scope)
        int16 = eval_top1(sim.model, loaders["test"], device, ExecutionMode.INT16_FIXED_EVAL, mb)
        delta = (int16 - fixed) * 100
        print(
            f"{scope:22s} {len(names):6d} {int16 * 100:7.2f}% {delta:+11.2f}pp",
            flush=True,
        )


if __name__ == "__main__":
    main()

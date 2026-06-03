#!/usr/bin/env python3
"""MRNN 单算子 teacher-forced 精度：四模式 vs prepared 浮点参考（float_native 图节点）。

参考输出取自 ``prepare_model`` 后的浮点副本（同分解图、无 QDQ），与 sim 上
Quantized* 模块名一一对应；整图 Top-1 baseline 仍用未包装 MRNN 报告。
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

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
if not os.path.isdir(_DATA):
    _DATA = "/mnt/data8t/share/datasets/speech_commands/SpeechCommands/speech_commands_v0.02"


def _soundfile_load(path, *args, **kwargs):
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    return torch.from_numpy(data.T.copy()), sr


torchaudio.load = _soundfile_load

import quick_start as qs  # noqa: E402
from common.mrnn_clz_encoding import (  # noqa: E402
    DIV_NAME_RE,
    SIGN_NAME_RE,
    SQUARE_NAME_RE,
    apply_mrnn_clz_encoding_fixes,
    apply_mrnn_clz_encoding_fixes_post_calib,
    collect_power2_float_out_fmax,
    disable_input_quantizers_by_pattern,
    disable_output_quantizers_by_pattern,
)
from aimet_torch.fixed_point import (  # noqa: E402
    ExecutionMode,
    convert_encodings_to_fixed_scale,
    quant_execution_mode,
)
from aimet_torch.fixed_point.metrics.accuracy import cosine_similarity  # noqa: E402
from aimet_torch.fixed_point.metrics.isolated import (  # noqa: E402
    _default_filter,
    _normalize_inputs,
    _quantize_with_carrier,
    _to_float,
)
from aimet_torch.fixed_point.tensor import Int16QuantizedTensor  # noqa: E402


CAND_MODES = (
    ExecutionMode.FP32_QDQ,
    ExecutionMode.FP16_QDQ,
    ExecutionMode.FIXED_SCALE_QDQ,
    ExecutionMode.INT16_FIXED_EVAL,
)

MIN_COSINE_DEFAULT = 0.9999
REF_NORM_DEGEN = 1e-6

# 与 doc/MRNN_SingleOp_LUT_Validation.md §4 对齐
OP_THRESHOLDS = {
    "sign": {"metric": "sign_agreement_output", "min": 0.99, "min_input_sign_agreement": 0.99},
    "sqrt": {"metric": "cos_on_qinput", "min": 0.999},
    "square": {"metric": "cos_on_qinput", "min": 0.99, "max_saturation_rate": 0.01},
    "div": {"metric": "cos_on_qinput", "min": 0.99, "max_zero_denom_rate": 0.0},
    "reciprocal": {"metric": "cos_on_qinput", "min": 0.99, "max_zero_denom_rate": 0.0},
    "conv": {"metric": "cos_on_qinput", "min": 0.999},
    "default": {"metric": "cos_on_qinput", "min": 0.999},
}


def classify_op(name: str) -> str:
    n = name.lower()
    if "sign" in n:
        return "sign"
    if "square" in n:
        return "square"
    if "div" in n or "floordiv" in n:
        return "reciprocal"
    if "sqrt" in n or "rsqrt" in n:
        return "sqrt"
    if "conv" in n or "linear" in n or "gru" in n:
        return "conv"
    return "default"


def _qdq_tensor(q, t: torch.Tensor) -> torch.Tensor:
    if q is None or not getattr(q, "is_initialized", lambda: False)():
        return t
    y = q(t)
    return y.dequantize() if hasattr(y, "dequantize") else y


def _apply_float_op(name: str, mod: nn.Module, args: tuple) -> torch.Tensor | None:
    """float 参考算子在 Q 后输入上的输出（口径 B）。"""
    q_args = list(args)
    if len(args) >= 1 and isinstance(args[0], torch.Tensor):
        iq = getattr(mod, "input_quantizers", [None])
        if iq and iq[0] is not None:
            q_args[0] = _qdq_tensor(iq[0], args[0]).float()
    if len(args) >= 2 and isinstance(args[1], torch.Tensor):
        iqs = getattr(mod, "input_quantizers", [])
        if len(iqs) > 1 and iqs[1] is not None:
            q_args[1] = _qdq_tensor(iqs[1], args[1]).float()
    q_args = tuple(q_args)
    cat = classify_op(name)
    with torch.no_grad():
        if cat == "sign" and len(q_args) >= 1:
            return torch.sign(q_args[0])
        if cat == "square" and len(q_args) >= 1:
            return torch.square(q_args[0])
        if cat == "sqrt" and len(q_args) >= 1:
            return torch.sqrt(q_args[0].clamp(min=qs.EPS))
        if cat == "div" and len(q_args) >= 2:
            b = q_args[1].clamp(min=qs.EPS)
            return (q_args[0] / b).float()
        out = mod(*q_args)
        if isinstance(out, torch.Tensor):
            return out.float()
        if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
            return out[0].float()
    return None


def _extra_metrics(name: str, module: nn.Module, x_args: tuple, y_ref: torch.Tensor, y_cand: torch.Tensor) -> dict:
    """口径 B 与 LUT 文档相关的附加指标。"""
    out: dict = {}
    ref_norm = float(y_ref.norm().item())
    out["ref_norm"] = ref_norm
    out["op_category"] = classify_op(name)

    y_ref_q = _apply_float_op(name, module, x_args)
    if y_ref_q is not None and y_ref_q.shape == y_cand.shape:
        out["cos_on_qinput"] = cosine_similarity(y_ref_q, y_cand)
    out["cos_total"] = cosine_similarity(y_ref, y_cand)

    cat = out["op_category"]
    if cat == "sign" and len(x_args) >= 1 and isinstance(x_args[0], torch.Tensor):
        x = x_args[0]
        iq = module.input_quantizers[0] if getattr(module, "input_quantizers", None) else None
        x_q = _qdq_tensor(iq, x).float() if iq is not None else x.float()
        out["sign_agreement"] = float((torch.sign(x) == torch.sign(x_q)).float().mean())
        if y_ref_q is not None:
            out["sign_agreement_output"] = float(
                (torch.sign(y_ref_q) == torch.sign(y_cand)).float().mean()
            )

    if cat == "square" and y_ref.numel() > 0:
        oq = module.output_quantizers[0] if getattr(module, "output_quantizers", None) else None
        if oq is not None and hasattr(oq, "max"):
            try:
                ymax = float(oq.max.detach().reshape(-1)[0].item())
                out["saturation_rate"] = float((y_ref.abs() > ymax + 1e-6).float().mean())
            except (RuntimeError, AttributeError):
                pass

    if cat == "div" and len(x_args) >= 2 and isinstance(x_args[1], torch.Tensor):
        b = x_args[1]
        iqs = getattr(module, "input_quantizers", [])
        b_q = _qdq_tensor(iqs[1], b).float() if len(iqs) > 1 and iqs[1] is not None else b.float()
        out["zero_denom_rate"] = float((b_q.abs() < 1e-8).float().mean())

    return out


def _gate_status(name: str, metrics: dict) -> str:
    if metrics.get("ref_norm", 1.0) < REF_NORM_DEGEN:
        return "DEGEN"
    th = OP_THRESHOLDS.get(classify_op(name), OP_THRESHOLDS["default"])
    if th.get("max_zero_denom_rate") is not None:
        zdr = metrics.get("zero_denom_rate", 0.0)
        if zdr > th["max_zero_denom_rate"]:
            return "FAIL"
    key = th["metric"]
    val = metrics.get(key)
    if val is None or (isinstance(val, float) and val != val):
        if classify_op(name) == "div":
            return "FAIL"
        return "DEGEN"
    ok = val >= th["min"]
    if "min_input_sign_agreement" in th:
        inp = metrics.get("sign_agreement")
        if inp is not None and inp < th["min_input_sign_agreement"]:
            ok = False
    if "max_saturation_rate" in th:
        ok = ok and metrics.get("saturation_rate", 0.0) <= th["max_saturation_rate"]
    return "PASS" if ok else "FAIL"


def _capture_prepared_float_io(
    prepared_float: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    target_names: set[str],
) -> tuple[dict[str, tuple], dict[str, torch.Tensor]]:
    cached_in: dict[str, tuple] = {}
    cached_out: dict[str, torch.Tensor] = {}

    def make_hook(name: str):
        def _hook(_mod, ins, out):
            if name not in target_names:
                return
            captured = []
            for arg in ins:
                if isinstance(arg, torch.Tensor):
                    captured.append(arg.detach().float().clone())
                else:
                    captured.append(arg)
            cached_in[name] = tuple(captured)
            if isinstance(out, torch.Tensor):
                cached_out[name] = out.detach().float().clone()
            elif isinstance(out, (tuple, list)):
                tensors = [o for o in out if isinstance(o, torch.Tensor)]
                if len(tensors) == 1:
                    cached_out[name] = tensors[0].detach().float().clone()

        return _hook

    hooks = []
    for name, mod in prepared_float.named_modules():
        if name in target_names:
            hooks.append(mod.register_forward_hook(make_hook(name)))

    prepared_float.eval()
    with torch.no_grad():
        prepared_float(*inputs)
    for h in hooks:
        h.remove()
    return cached_in, cached_out


def _capture_int16_carriers(
    sim_model: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    target_names: set[str],
) -> dict[str, dict]:
    carriers: dict[str, dict] = {}

    def make_hook(name: str):
        def _hook(_mod, ins, _out):
            if not ins:
                return
            first = ins[0]
            if isinstance(first, Int16QuantizedTensor):
                carriers[name] = {
                    "scale": first.scale.detach().clone(),
                    "zero_point": first.zero_point.detach().clone(),
                    "qmin": first.qmin,
                    "qmax": first.qmax,
                    "axis": first.axis,
                }

        return _hook

    hooks = [
        m.register_forward_hook(make_hook(n))
        for n, m in sim_model.named_modules()
        if n in target_names
    ]
    try:
        with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
            with torch.no_grad():
                sim_model(*inputs)
    except (RuntimeError, AttributeError, TypeError):
        pass
    finally:
        for h in hooks:
            h.remove()
    return carriers


def isolated_vs_float_native(
    sim_model: nn.Module,
    prepared_float: nn.Module,
    inputs: torch.Tensor | tuple[torch.Tensor, ...],
    *,
    cand_mode: ExecutionMode,
    min_cosine: float = MIN_COSINE_DEFAULT,
) -> list[dict]:
    in_args = _normalize_inputs(inputs)
    targets = [(n, m) for n, m in sim_model.named_modules() if _default_filter(n, m)]
    names = {n for n, _ in targets}
    float_in, float_out = _capture_prepared_float_io(prepared_float, in_args, names)

    int16_carriers: dict[str, dict] = {}
    if cand_mode is ExecutionMode.INT16_FIXED_EVAL:
        int16_carriers = _capture_int16_carriers(sim_model, in_args, names)

    rows: list[dict] = []
    for name, module in targets:
        if name not in float_in or name not in float_out:
            rows.append({
                "module": name,
                "mode": cand_mode.value,
                "status": "SKIP",
                "reason": "no float_native IO (name mismatch or non-tensor out)",
            })
            continue

        x_args = float_in[name]
        y_ref = float_out[name]
        x_run = x_args
        if (
            cand_mode is ExecutionMode.INT16_FIXED_EVAL
            and name in int16_carriers
            and x_args
            and isinstance(x_args[0], torch.Tensor)
        ):
            try:
                x0 = _quantize_with_carrier(x_args[0], int16_carriers[name])
                x_run = (x0,) + x_args[1:]
            except (RuntimeError, TypeError, ValueError) as exc:
                rows.append({
                    "module": name,
                    "mode": cand_mode.value,
                    "status": "SKIP",
                    "reason": f"int16 requant: {exc}",
                })
                continue

        try:
            with quant_execution_mode(cand_mode):
                y_raw = module(*x_run)
        except (RuntimeError, AttributeError, TypeError) as exc:
            rows.append({
                "module": name,
                "mode": cand_mode.value,
                "status": "ERROR",
                "reason": str(exc),
            })
            continue

        y_cand = _to_float(y_raw)
        if y_cand is None or y_cand.shape != y_ref.shape:
            rows.append({
                "module": name,
                "mode": cand_mode.value,
                "status": "SKIP",
                "reason": "shape/type mismatch",
            })
            continue

        cos = cosine_similarity(y_ref, y_cand)
        diff = (y_cand - y_ref).reshape(-1)
        extra = _extra_metrics(name, module, x_args, y_ref, y_cand)
        gate = _gate_status(name, extra)
        row = {
            "module": name,
            "mode": cand_mode.value,
            "cosine": cos,
            "cos_total": extra.get("cos_total", cos),
            "cos_on_qinput": extra.get("cos_on_qinput"),
            "max_abs_err": float(diff.abs().max().item()),
            "rmse": float(diff.pow(2).mean().sqrt().item()),
            "shape": list(y_ref.shape),
            "op_category": extra.get("op_category", "default"),
            "ref_norm": extra.get("ref_norm"),
            "status": gate,
            "legacy_status_cos09999": "PASS" if cos + 1e-12 >= min_cosine else "FAIL",
            "min_cosine_legacy": min_cosine,
        }
        for k in ("sign_agreement", "sign_agreement_output", "saturation_rate", "zero_denom_rate"):
            if k in extra:
                row[k] = extra[k]
        rows.append(row)
    return rows


def build_sim(
    fp: nn.Module,
    loaders,
    device: torch.device,
    *,
    skip_qat: bool,
    qat_epochs: int,
    max_calib: int,
    bitwidth_config: Path | str | None = None,
    sign_input_bypass: bool = False,
    div_denom_input_bypass: bool = False,
    square_output_bypass: bool = False,
    clz_encoding_fix: bool = False,
    clz_post_qat: bool = False,
    apply_po2: bool = True,
    clz_sign_bypass: bool = True,
    qat_val_loader=None,
    qat_max_batches: int | None = None,
    qat_lr: float | None = None,
    qat_restore_best: bool = True,
) -> tuple[object, nn.Module]:
    prepared = qs.model_preparer.prepare_model(
        copy.deepcopy(fp),
        stateless_modules_to_preserve=[
            qs.PowerCompress, qs.HypotFun, qs.CLN, qs.QuantizableBatchNorm2d,
        ],
    )
    prepared_float = copy.deepcopy(prepared).to(device).eval()

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
    bw_cfg = bitwidth_config or qs.BITWIDTH_CONFIG_FILE
    qs.apply_mixed_precision_bitwidth(
        sim.model, config_file=str(bw_cfg), verbose=False,
    )
    if clz_encoding_fix:
        apply_mrnn_clz_encoding_fixes(
            sim.model, sign_input_bypass=clz_sign_bypass, verbose=False,
        )
    elif sign_input_bypass:
        n = disable_input_quantizers_by_pattern(sim.model, SIGN_NAME_RE, verbose=False)
        if n == 0:
            print("WARNING: --sign-input-bypass 未匹配任何 input quantizer")
    if div_denom_input_bypass and not clz_encoding_fix:
        n = disable_input_quantizers_by_pattern(
            sim.model, DIV_NAME_RE, input_indices=(1,), verbose=False,
        )
        if n == 0:
            print("WARNING: --div-denom-input-bypass 未匹配任何 denominator input quantizer")
    if square_output_bypass and not clz_encoding_fix:
        n = disable_output_quantizers_by_pattern(
            sim.model, SQUARE_NAME_RE, verbose=False,
        )
        if n == 0:
            print("WARNING: --square-output-bypass 未匹配任何 output quantizer")

    sim.model.to(device).eval()
    with torch.no_grad(), qs.aimet.nn.compute_encodings(sim.model):
        for idx, (x, _) in enumerate(loaders["calib"]):
            if idx >= max_calib:
                break
            sim.model(x.to(device))

    power2_float_fmax = None
    if clz_encoding_fix:
        power2_float_fmax = collect_power2_float_out_fmax(
            prepared_float, loaders["calib"], device, max_calib,
        )

    if apply_po2:
        qs.apply_power_of_2_workflow(
            sim.model, method="round", tolerance=0.02, align_bias_scale=True, verbose=False,
        )

    encoding_fix_stats = None
    if clz_encoding_fix and not clz_post_qat:
        encoding_fix_stats = apply_mrnn_clz_encoding_fixes_post_calib(
            sim.model,
            power2_float_out_fmax=power2_float_fmax,
            verbose=False,
        )

    qat_stats = None
    if not skip_qat:
        qs.freeze_quantizer_parameters(sim.model, verbose=False, freeze_bn_affine=True)
        val_loader = qat_val_loader if qat_val_loader is not None else loaders.get("val")
        qat_stats = qs.qat_finetune(
            sim,
            loaders["train"],
            device,
            epochs=qat_epochs,
            lr=qat_lr if qat_lr is not None else qs.QAT_LR,
            val_loader=val_loader,
            max_batches_per_epoch=qat_max_batches,
            restore_best=qat_restore_best,
        )

    if clz_encoding_fix and clz_post_qat:
        encoding_fix_stats = apply_mrnn_clz_encoding_fixes_post_calib(
            sim.model,
            power2_float_out_fmax=power2_float_fmax,
            verbose=False,
        )

    convert_encodings_to_fixed_scale(sim)
    if qat_stats is not None:
        sim._qat_stats = qat_stats  # noqa: SLF001
    if encoding_fix_stats is not None:
        sim._clz_encoding_fix_stats = encoding_fix_stats  # noqa: SLF001
    sim._clz_sign_bypass = clz_sign_bypass if clz_encoding_fix else None  # noqa: SLF001
    return sim, prepared_float


def main() -> None:
    parser = argparse.ArgumentParser(description="MRNN single-op vs float_native (prepared graph)")
    parser.add_argument("--data-root", default=_DATA)
    parser.add_argument("--max-calib-batches", type=int, default=100)
    parser.add_argument("--skip-qat", action="store_true")
    parser.add_argument("--qat-epochs", type=int, default=1)
    parser.add_argument("--min-cosine", type=float, default=MIN_COSINE_DEFAULT)
    parser.add_argument(
        "--bitwidth-config",
        type=Path,
        default=None,
        help="mixed-precision JSON（默认 quick_start_full_quant.json）",
    )
    parser.add_argument(
        "--sign-input-bypass",
        action="store_true",
        help="禁用 *module_sign* 的 input quantizer，保留 output Q",
    )
    parser.add_argument(
        "--square-output-bypass",
        action="store_true",
        help="禁用 *module_square_* 的 output quantizer（避免 x^2 动态范围溢出 16bit）",
    )
    parser.add_argument(
        "--div-denom-input-bypass",
        action="store_true",
        help="[诊断] 禁用 *module_div_* 分母 input Q；生产路径用 --clz-encoding-fix",
    )
    parser.add_argument(
        "--clz-encoding-fix",
        action="store_true",
        help="§2.3 encoding 修复：sign input bypass + reciprocal 分母 + power_2 out_max",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "int16_single_op_vs_float_native.json",
    )
    args = parser.parse_args()

    qs.DATA_ROOT = args.data_root
    qs.FP_EPOCHS = 0
    qs.MAX_CALIB_BATCHES = args.max_calib_batches
    qs.set_seed(qs.SEED)
    qs.setup_audio_backend()
    device = qs.DEVICE

    print("=" * 72)
    print("MRNN 单算子 vs float_native（prepared 浮点节点参考）")
    print("=" * 72)

    loaders = qs.build_dataloaders(qs.DATA_ROOT)
    sd = torch.load(qs.FP_MODEL_PATH, map_location=device, weights_only=False)
    state = sd.get("model", sd) if isinstance(sd, dict) else sd

    fp = qs.MRNN(output_dim=qs.NUM_CLASSES).to(device)
    fp.load_state_dict(state, strict=False)
    fp_acc = qs.evaluate(fp, loaders["test"], device)
    print(f"整图 float_native (MRNN): {fp_acc * 100:.2f}%")

    t0 = time.time()
    sim, prepared_float = build_sim(
        fp,
        loaders,
        device,
        skip_qat=args.skip_qat,
        qat_epochs=args.qat_epochs,
        max_calib=args.max_calib_batches,
        bitwidth_config=args.bitwidth_config,
        sign_input_bypass=args.sign_input_bypass,
        div_denom_input_bypass=args.div_denom_input_bypass,
        square_output_bypass=args.square_output_bypass,
        clz_encoding_fix=args.clz_encoding_fix,
    )
    print(f"sim 构建+校准{' (+QAT)' if not args.skip_qat else ''}: {time.time() - t0:.1f}s")

    x, _ = next(iter(loaders["calib"]))
    x = x[:2].to(device)

    all_rows: list[dict] = []
    summary: dict[str, dict] = {}
    for mode in CAND_MODES:
        rows = isolated_vs_float_native(
            sim.model,
            prepared_float,
            x,
            cand_mode=mode,
            min_cosine=args.min_cosine,
        )
        all_rows.extend(rows)
        ok = [r for r in rows if r.get("status") == "PASS"]
        fail = [r for r in rows if r.get("status") == "FAIL"]
        skip = [r for r in rows if r.get("status") == "SKIP"]
        err = [r for r in rows if r.get("status") == "ERROR"]
        degen = [r for r in rows if r.get("status") == "DEGEN"]
        worst = sorted(
            [r for r in rows if r.get("cos_on_qinput") is not None],
            key=lambda r: r["cos_on_qinput"],
        )[:5]
        if not worst:
            worst = sorted(
                [r for r in rows if "cosine" in r],
                key=lambda r: r["cosine"],
            )[:5]
        summary[mode.value] = {
            "pass": len(ok),
            "fail": len(fail),
            "skip": len(skip),
            "error": len(err),
            "degen": len(degen),
            "worst5": worst,
        }
        print(
            f"\n--- {mode.value} --- "
            f"pass={len(ok)} fail={len(fail)} degen={len(degen)} "
            f"skip={len(skip)} error={len(err)}"
        )
        for r in worst:
            cq = r.get("cos_on_qinput")
            extra = ""
            if cq is not None:
                extra = f" cos_qin={cq:.6f}"
            if r.get("sign_agreement") is not None:
                extra += f" sign={r['sign_agreement']:.4f}"
            print(
                f"  worst {r['module']} [{r.get('op_category')}]: "
                f"cos_total={r.get('cos_total', r.get('cosine', 0)):.6f}{extra} "
                f"status={r.get('status')}"
            )

    bw_used = args.bitwidth_config or qs.BITWIDTH_CONFIG_FILE
    report = {
        "float_native_top1": fp_acc,
        "validation_doc": "doc/MRNN_SingleOp_LUT_Validation.md",
        "gate_metric": "cos_on_qinput / sign_agreement / category thresholds",
        "min_cosine_legacy": args.min_cosine,
        "bitwidth_config": str(bw_used),
        "sign_input_bypass": args.sign_input_bypass,
        "div_denom_input_bypass": args.div_denom_input_bypass,
        "square_output_bypass": args.square_output_bypass,
        "clz_encoding_fix": args.clz_encoding_fix,
        "clz_encoding_fix_stats": getattr(sim, "_clz_encoding_fix_stats", None),
        "calib_batch_shape": list(x.shape),
        "skip_qat": args.skip_qat,
        "summary": summary,
        "rows": all_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n报告已写入: {args.output}")


if __name__ == "__main__":
    main()

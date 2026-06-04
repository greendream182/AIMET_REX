#!/usr/bin/env python3
"""整图 Top-1 vs float_native（path B）：QDQ 三档 + 可选 INT16 真 kernel 路径。

- **QDQ 路径**（默认）：``iso.build_sim`` + CLZ encoding fix，评 fp32/fp16/fixed_scale QDQ。
- **INT16 路径**（``--eval-int16``）：独立 ``build_int16_sim``（ensure + native_trans），
  不与 QDQ 共用同一 sim——``ensure_output_quantizers`` 会破坏同 sim 上的 QDQ forward。
- **qat_sim 单独测**（``--eval-int16-qat-sim-only``）：跳过 QDQ 与 ``int16_fixed_eval``，仅评
  ``int16_fixed_qat_sim``（省显存）。

Scale：默认 **无全图 Po2**（``convert_encodings_to_fixed_scale``）；``--apply-po2`` 仅 legacy。

基准：未包装 MRNN（``model_fp.pth``），与单算子脚本的 float_native Top-1 口径一致。
"""
from __future__ import annotations

import argparse
import contextlib
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
import torchaudio

_DATA = "/home/llq/workspace/data/speech_commands"
if not os.path.isdir(_DATA):
    _DATA = "/mnt/data8t/share/datasets/speech_commands/SpeechCommands/speech_commands_v0.02"


def _soundfile_load(path, *args, **kwargs):
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    return torch.from_numpy(data.T.copy()), sr


torchaudio.load = _soundfile_load

import quick_start as qs  # noqa: E402
import int16_single_op_vs_float_native as iso  # noqa: E402
import aimet_torch.fixed_point.kernels  # noqa: F401,E402
from aimet_torch.fixed_point import (  # noqa: E402
    ExecutionMode,
    convert_encodings_to_fixed_scale,
    ensure_output_quantizers_for_int16_eval,
    quant_execution_mode,
    run_int16_qat_steps,
)
from aimet_torch.fixed_point.diagnose import diagnose_int16_readiness  # noqa: E402
from aimet_torch.fixed_point.metrics import int16_eval_allow_debug_float  # noqa: E402
from aimet_torch.fixed_point.qat_train import release_cuda_memory, warm_int16_qat_lut_cache  # noqa: E402
from aimet_torch.utils_rx import (  # noqa: E402
    apply_mixed_precision_bitwidth,
    freeze_quantizer_parameters,
    set_train_mode_freeze_bn,
)
from common.mrnn_clz_encoding import (  # noqa: E402
    apply_mrnn_clz_encoding_fixes,
    apply_mrnn_clz_encoding_fixes_post_calib,
    collect_power2_float_out_fmax,
)
from common.torch_stft import STFT  # noqa: E402
from quantized_stft import force_native_trans_float  # noqa: E402


def build_int16_sim(
    fp: torch.nn.Module,
    loaders,
    device: torch.device,
    *,
    max_calib: int,
    bitwidth_config: Path | str | None,
    clz_encoding_fix: bool = True,
) -> object:
    """INT16 验收专用 sim：ensure → bitwidth → CLZ fix → calib → post-calib → fixed_scale。"""
    prepared = qs.model_preparer.prepare_model(
        copy.deepcopy(fp),
        stateless_modules_to_preserve=[
            qs.PowerCompress, qs.HypotFun, qs.CLN, qs.QuantizableBatchNorm2d,
        ],
        module_classes_to_exclude=[STFT],
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
    patched = ensure_output_quantizers_for_int16_eval(sim)
    bw_cfg = bitwidth_config or Path(__file__).resolve().parent / "config" / "pc1_hypot_16bit.json"
    apply_mixed_precision_bitwidth(sim.model, config_file=str(bw_cfg), verbose=False)
    clz_stats = None
    if clz_encoding_fix:
        # Sign 已在 INT16 kernel/adapter 侧对 float 取 sign；此处不再 bypass input Q。
        apply_mrnn_clz_encoding_fixes(sim.model, sign_input_bypass=False, verbose=False)
    n_trans = force_native_trans_float(sim.model)
    print(f"  INT16 build: ensure patched {len(patched)} slots; native_trans cleared {n_trans} trans Q")
    sim.model.to(device).eval()
    calib_loader = qs.fresh_calib_loader(loaders["calib"])
    with torch.no_grad(), qs.aimet.nn.compute_encodings(sim.model):
        for idx, (x, _) in enumerate(calib_loader):
            if idx >= max_calib:
                break
            sim.model(x.to(device))
    if clz_encoding_fix:
        power2_fmax = collect_power2_float_out_fmax(
            prepared_float, loaders["calib"], device, max_calib,
        )
        clz_stats = apply_mrnn_clz_encoding_fixes_post_calib(
            sim.model, power2_float_out_fmax=power2_fmax, verbose=False,
        )
        sim._clz_encoding_fix_stats = clz_stats  # noqa: SLF001
    n_fixed = convert_encodings_to_fixed_scale(sim)
    print(f"  convert_encodings_to_fixed_scale: {n_fixed} affine quantizers cached (M,r)")
    return sim


def _eval_modes(model, loader, device) -> dict[str, float | str]:
    out: dict[str, float | str] = {}
    for mode in (
        ExecutionMode.FP32_QDQ,
        ExecutionMode.FP16_QDQ,
        ExecutionMode.FIXED_SCALE_QDQ,
    ):
        try:
            with quant_execution_mode(mode):
                out[mode.value] = qs.evaluate(model, loader, device)
        except Exception as exc:  # noqa: BLE001
            out[mode.value] = f"FAILED: {exc}"
    return out


def _eval_int16_modes(
    model,
    loader,
    device,
    *,
    max_batches: int | None = None,
    include_eval: bool = True,
    include_qat_sim: bool = False,
    qat_sim_micro_batch: int | None = 16,
) -> dict[str, float | str]:
    out: dict[str, float | str] = {}
    modes: list[tuple[ExecutionMode, bool]] = []
    if include_eval:
        modes.append((ExecutionMode.INT16_FIXED_EVAL, False))
    if include_qat_sim:
        modes.append((ExecutionMode.INT16_FIXED_QAT_SIM, True))
    if not modes:
        return out

    for mode, train_mode in modes:
        release_cuda_memory(model)
        ctx = (
            int16_eval_allow_debug_float()
            if mode is ExecutionMode.INT16_FIXED_EVAL
            else contextlib.nullcontext()
        )
        try:
            if train_mode:
                set_train_mode_freeze_bn(model)
            else:
                model.eval()
            if max_batches is None:
                with ctx, quant_execution_mode(mode):
                    if train_mode:
                        # qs.evaluate() 会 model.eval()，破坏 INT16_FIXED_QAT_SIM 的 GRU train 要求
                        correct = total = 0
                        mb = qat_sim_micro_batch
                        with torch.no_grad():
                            for batch_idx, (inputs, labels) in enumerate(loader):
                                if max_batches is not None and batch_idx >= max_batches:
                                    break
                                labels = labels.to(device)
                                if mb is None or inputs.size(0) <= mb:
                                    chunks = [(inputs.to(device), labels)]
                                else:
                                    chunks = [
                                        (
                                            inputs[i : i + mb].to(device),
                                            labels[i : i + mb],
                                        )
                                        for i in range(0, inputs.size(0), mb)
                                    ]
                                for chunk_x, chunk_y in chunks:
                                    logits = model(chunk_x)
                                    if hasattr(logits, "to_float"):
                                        logits = logits.to_float()
                                    preds = logits.max(1).indices
                                    total += chunk_y.size(0)
                                    correct += preds.eq(chunk_y).sum().item()
                                if train_mode:
                                    release_cuda_memory(model)
                        acc = correct / max(total, 1)
                    else:
                        acc = qs.evaluate(model, loader, device)
            else:
                correct = total = 0
                with torch.no_grad(), ctx, quant_execution_mode(mode):
                    for i, (x, y) in enumerate(loader):
                        if i >= max_batches:
                            break
                        pred = model(x.to(device))
                        if hasattr(pred, "to_float"):
                            pred = pred.to_float()
                        correct += pred.argmax(1).cpu().eq(y).sum().item()
                        total += y.numel()
                acc = correct / max(total, 1)
            out[mode.value] = acc
        except Exception as exc:  # noqa: BLE001
            out[mode.value] = f"FAILED: {exc}"
    return out


def _decide_verify_int16_qat_sim_pass(
    results: dict[str, object],
    *,
    min_top1: float,
    max_delta_pp: float | None,
    float_native_top1: float | None,
) -> tuple[bool, list[str], list[str], float | None]:
    """阈值决策（纯函数；与训练/评估解耦，便于单测）。

    要求 ``results`` 已包含：
      - ``readiness``/``backward``/``qat_steps``: 字符串，"OK" 才通过
      - ``int16_fixed_eval_top1``: float Top-1 或失败描述

    返回 ``(passed, pass_reasons, fail_reasons, delta_pp)``。
    """
    pass_reasons: list[str] = []
    fail_reasons: list[str] = []
    for key in ("readiness", "backward", "qat_steps"):
        if results.get(key) == "OK":
            pass_reasons.append(key)
        else:
            fail_reasons.append(f"{key}={results.get(key)!r}")

    eval_acc = results.get("int16_fixed_eval_top1")
    delta_pp: float | None = None
    if not isinstance(eval_acc, float):
        fail_reasons.append(f"int16_fixed_eval_top1={eval_acc!r}")
        return False, pass_reasons, fail_reasons, None

    if isinstance(float_native_top1, float):
        delta_pp = (eval_acc - float_native_top1) * 100

    if eval_acc >= min_top1:
        pass_reasons.append(
            f"top1={eval_acc * 100:.2f}%>=min_top1={min_top1 * 100:.2f}%"
        )
    else:
        fail_reasons.append(
            f"top1={eval_acc * 100:.2f}%<min_top1={min_top1 * 100:.2f}%"
        )

    if max_delta_pp is not None and delta_pp is not None:
        if delta_pp >= -abs(max_delta_pp):
            pass_reasons.append(
                f"delta_pp={delta_pp:+.2f}>=-{abs(max_delta_pp):.2f}"
            )
        else:
            fail_reasons.append(
                f"delta_pp={delta_pp:+.2f}<-{abs(max_delta_pp):.2f}"
            )

    return (not fail_reasons), pass_reasons, fail_reasons, delta_pp


def verify_int16_qat_sim(
    sim_model,
    loaders,
    device: torch.device,
    *,
    qat_steps: int = 20,
    qat_scope: str = "head",
    qat_lr: float = 1e-4,
    qat_batch_size: int = 16,
    eval_max_batches: int | None = 50,
    min_top1: float = 0.90,
    max_delta_pp: float | None = None,
    float_native_top1: float | None = None,
) -> dict[str, object]:
    """验收 int16_fixed_qat_sim 训练路径。

    验收项（全部满足才 PASS）：
      1. ``readiness`` OK
      2. ``backward`` OK（至少 1 个 QuantGRU 收到 backward_quant 梯度）
      3. ``qat_steps`` loss finite
      4. QAT 后 ``int16_fixed_eval`` Top-1 ``>= min_top1``
      5. 若提供 ``float_native_top1`` 与 ``max_delta_pp``，QAT 后 Top-1 相对 baseline
         下降 ``<= max_delta_pp``

    Top-1 不看 PTQ-only ``int16_fixed_qat_sim`` 自身（surrogate 训练态语义，
    不能作为部署精度）。
    """
    import math

    model = sim_model.model
    results: dict[str, object] = {
        "verify_min_top1": min_top1,
        "verify_max_delta_pp": max_delta_pp,
        "float_native_top1": float_native_top1,
    }

    report = diagnose_int16_readiness(sim_model)
    blockers = {k: v for k, v in report.items() if v}
    results["readiness"] = "OK" if not blockers else f"FAILED: {blockers}"
    if blockers:
        print(f"  readiness: FAILED — {blockers}")
        return results
    print("  readiness: OK")

    sample_x = next(iter(loaders["train"]))[0][:4].to(device)
    try:
        warm_s = warm_int16_qat_lut_cache(model, sample_x, device)
        results["lut_warm_sec"] = warm_s
        print(f"  LUT warm: OK ({warm_s:.1f}s)")
    except Exception as exc:  # noqa: BLE001
        results["lut_warm"] = f"FAILED: {exc}"
        print(f"  LUT warm: FAILED — {exc}")
        return results

    model.train()
    try:
        with quant_execution_mode(ExecutionMode.INT16_FIXED_QAT_SIM):
            y_qat = model(sample_x)
        loss_tensor = y_qat.to_float() if hasattr(y_qat, "to_float") else y_qat
        loss_tensor.sum().backward()
        n_gru_grad = sum(
            1
            for m in model.modules()
            if m.__class__.__name__ == "QuantizedQuantGRU"
            and getattr(m, "weight_ih_l0", None) is not None
            and m.weight_ih_l0.grad is not None
        )
        ok = n_gru_grad >= 1
        results["backward"] = "OK" if ok else f"FAILED: no QuantGRU grad (n={n_gru_grad})"
        print(f"  backward: {results['backward']}" + (f" (QuantGRU grad modules={n_gru_grad})" if ok else ""))
    except Exception as exc:  # noqa: BLE001
        results["backward"] = f"FAILED: {exc}"
        print(f"  backward: FAILED — {exc}")
        return results
    finally:
        release_cuda_memory(model)
        model.eval()

    freeze_quantizer_parameters(model, verbose=False, freeze_bn_affine=True)
    try:
        losses = run_int16_qat_steps(
            model,
            loaders["train"],
            device,
            steps=qat_steps,
            lr=qat_lr,
            scope=qat_scope,  # type: ignore[arg-type]
            batch_size=qat_batch_size,
            empty_cache_every=1,
        )
        finite = all(math.isfinite(x) for x in losses)
        results["qat_steps"] = "OK" if finite else "FAILED: non-finite loss"
        results["qat_loss_last"] = losses[-1] if losses else None
        print(
            f"  QAT steps ({qat_steps}, scope={qat_scope}): "
            f"{results['qat_steps']}"
            + (f", last_loss={losses[-1]:.4f}" if losses else "")
        )
    except Exception as exc:  # noqa: BLE001
        results["qat_steps"] = f"FAILED: {exc}"
        print(f"  QAT steps: FAILED — {exc}")
        return results
    finally:
        release_cuda_memory(model)
        model.eval()

    acc_map = _eval_int16_modes(
        model,
        loaders["test"],
        device,
        max_batches=eval_max_batches,
        include_eval=True,
        include_qat_sim=False,
    )
    eval_acc = acc_map.get(ExecutionMode.INT16_FIXED_EVAL.value)
    results["int16_fixed_eval_top1"] = eval_acc
    if isinstance(eval_acc, float):
        print(f"  int16_fixed_eval (max_batches={eval_max_batches}): {eval_acc * 100:.2f}%")
    else:
        print(f"  int16_fixed_eval: {eval_acc}")

    passed, pass_reasons, fail_reasons, delta_pp = _decide_verify_int16_qat_sim_pass(
        results,
        min_top1=min_top1,
        max_delta_pp=max_delta_pp,
        float_native_top1=float_native_top1,
    )
    if delta_pp is not None:
        results["int16_fixed_eval_delta_pp_vs_float_native"] = delta_pp
        print(
            f"  Δ vs float_native: {delta_pp:+.2f} pp "
            f"(baseline {float_native_top1 * 100:.2f}%)"
        )
    results["verify_passed"] = passed
    results["verify_pass_reasons"] = pass_reasons
    results["verify_fail_reasons"] = fail_reasons
    print(f"\n  verify_int16_qat_sim: {'PASS ✅' if passed else 'FAIL ❌'}")
    if fail_reasons:
        for reason in fail_reasons:
            print(f"    - {reason}")
    print(
        "  说明: PTQ 权重下 int16_fixed_qat_sim Top-1 无验收意义；"
        "交付指标看 QAT 后的 int16_fixed_eval。"
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="MRNN whole-graph vs float_native")
    parser.add_argument("--data-root", default=_DATA)
    parser.add_argument("--max-calib-batches", type=int, default=100)
    parser.add_argument(
        "--skip-qat",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="跳过 QAT（默认）；用 --no-skip-qat 启用 QAT",
    )
    parser.add_argument("--qat-epochs", type=int, default=1)
    parser.add_argument(
        "--qat-max-batches",
        type=int,
        default=None,
        help="每 epoch 最多训练 batch 数（默认全量 train）",
    )
    parser.add_argument(
        "--clz-encoding-fix",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="§2.3 sign/reciprocal/power_2 encoding（默认开）",
    )
    parser.add_argument(
        "--clz-post-qat",
        action="store_true",
        help="reciprocal/power_2 在 QAT 之后再应用（避免 QAT 与手工 encoding 冲突）",
    )
    parser.add_argument(
        "--apply-po2",
        action="store_true",
        help="校准后 apply_power_of_2_workflow（默认关；Ada200 主线用 M,rshift 即可）",
    )
    parser.add_argument(
        "--no-clz-sign-bypass",
        action="store_true",
        help="CLZ fix 时不 bypass sign input Q（默认 bypass）",
    )
    parser.add_argument("--qat-lr", type=float, default=None)
    parser.add_argument(
        "--no-qat-restore-best",
        action="store_true",
        help="QAT 不恢复 val 最优 checkpoint（观察 QAT 是否真在学习）",
    )
    parser.add_argument(
        "--bitwidth-config",
        type=Path,
        default=None,
        help="mixed-precision JSON（默认 quick_start_full_quant.json，8bit）；"
             "推荐 config/pc1_hypot_16bit.json 修复 frontend 主损失",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "int16_whole_graph_vs_float_native.json",
    )
    parser.add_argument(
        "--eval-int16",
        action="store_true",
        help="额外构建 INT16 sim 并评 int16_fixed_eval（及可选 qat_sim）；与 QDQ 分离 build",
    )
    parser.add_argument(
        "--eval-int16-qat-sim",
        action="store_true",
        help="在 --eval-int16 时一并评 int16_fixed_qat_sim（显存占用更高）",
    )
    parser.add_argument(
        "--eval-int16-qat-sim-only",
        action="store_true",
        help="仅 build INT16 sim 并评 int16_fixed_qat_sim（跳过 QDQ 与 int16_fixed_eval，省显存）",
    )
    parser.add_argument(
        "--eval-max-batches",
        type=int,
        default=None,
        help="INT16 评估最多 test batch 数（默认全量 test）；smoke 可设 50",
    )
    parser.add_argument(
        "--no-int16-clz-fix",
        action="store_true",
        help="INT16 build 不应用 reciprocal/power_2 encoding fix（默认开启）",
    )
    parser.add_argument(
        "--qat-sim-micro-batch",
        type=int,
        default=1,
        help="int16_fixed_qat_sim 评估 micro-batch（默认 1；train 模式在 32GB GPU 上 batch=64 易 OOM）",
    )
    parser.add_argument(
        "--verify-int16-qat-sim",
        action="store_true",
        help="验收 int16_fixed_qat_sim（readiness/backward/QAT steps/int16_fixed_eval；跳过无意义 PTQ-only qat_sim Top-1）",
    )
    parser.add_argument(
        "--verify-qat-steps",
        type=int,
        default=20,
        help="--verify-int16-qat-sim 时 QAT 训练步数（默认 20）",
    )
    parser.add_argument(
        "--verify-qat-scope",
        default="head",
        choices=("head", "weights", "all"),
        help="--verify-int16-qat-sim 时 QAT 可训练 scope（默认 head）",
    )
    parser.add_argument(
        "--verify-min-top1",
        type=float,
        default=0.90,
        help="--verify-int16-qat-sim 时 QAT 后 int16_fixed_eval 的 Top-1 下限（默认 0.90，smoke 用）",
    )
    parser.add_argument(
        "--verify-max-delta-pp",
        type=float,
        default=None,
        help="--verify-int16-qat-sim 时 QAT 后 int16_fixed_eval 相对 float_native 允许的最大下降（pp，正数）；"
             "默认不约束。需要 float_native baseline。",
    )
    args = parser.parse_args()
    if args.eval_int16_qat_sim_only or args.verify_int16_qat_sim:
        args.eval_int16 = True

    qs.DATA_ROOT = args.data_root
    qs.FP_EPOCHS = 0
    qs.MAX_CALIB_BATCHES = args.max_calib_batches
    qs.set_seed(qs.SEED)
    qs.setup_audio_backend()
    device = qs.DEVICE

    verify_only = args.verify_int16_qat_sim
    qat_sim_only = args.eval_int16_qat_sim_only and not verify_only
    skip_qdq_path = qat_sim_only or verify_only
    print("=" * 72)
    print("MRNN 整图 vs float_native（path B）")
    print(f"  apply_po2={args.apply_po2}  clz_fix={args.clz_encoding_fix}  "
          f"clz_post_qat={args.clz_post_qat}  skip_qat={args.skip_qat}  "
          f"eval_int16={args.eval_int16}  qat_sim_only={qat_sim_only}  "
          f"verify_qat_sim={verify_only}")
    print("=" * 72)

    loaders = qs.build_dataloaders(qs.DATA_ROOT)
    sd = torch.load(qs.FP_MODEL_PATH, map_location=device, weights_only=False)
    state = sd.get("model", sd) if isinstance(sd, dict) else sd

    fp = qs.MRNN(output_dim=qs.NUM_CLASSES).to(device)
    fp.load_state_dict(state, strict=False)
    fp.eval()

    float_native_top1: float | None = None
    if not qat_sim_only:
        t0 = time.time()
        float_native_top1 = qs.evaluate(fp, loaders["test"], device)
        print(f"float_native Top-1: {float_native_top1 * 100:.2f}%  ({time.time() - t0:.1f}s)")
    else:
        print("qat_sim_only：跳过 float_native 全量评估（仅 INT16 qat_sim 路径）")

    sim = None
    mode_acc: dict[str, float | str] = {}
    if not skip_qdq_path:
        t0 = time.time()
        sim, _ = iso.build_sim(
            fp,
            loaders,
            device,
            skip_qat=args.skip_qat,
            qat_epochs=args.qat_epochs,
            max_calib=args.max_calib_batches,
            bitwidth_config=args.bitwidth_config,
            clz_encoding_fix=args.clz_encoding_fix,
            clz_post_qat=args.clz_post_qat,
            apply_po2=args.apply_po2,
            clz_sign_bypass=not args.no_clz_sign_bypass,
            qat_val_loader=loaders["val"],
            qat_max_batches=args.qat_max_batches,
            qat_lr=args.qat_lr,
            qat_restore_best=not args.no_qat_restore_best,
        )
        print(f"sim 构建+校准{' (+QAT)' if not args.skip_qat else ''}: {time.time() - t0:.1f}s")

        t0 = time.time()
        mode_acc = _eval_modes(sim.model, loaders["test"], device)
        print(f"三档 QDQ 评估: {time.time() - t0:.1f}s")
        for k, v in mode_acc.items():
            if isinstance(v, float):
                delta_pp = (v - float_native_top1) * 100
                print(f"  {k:22s} {v * 100:7.2f}%  (Δ vs float_native {delta_pp:+.2f} pp)")
            else:
                print(f"  {k:22s} {v}")
    else:
        print("\n--- 跳过 QDQ sim 构建与三档评估（qat_sim_only / verify_int16_qat_sim）---")

    int16_mode_acc: dict[str, float | str] | None = None
    sim_int16 = None
    verify_report: dict[str, object] | None = None
    if args.eval_int16:
        label = (
            "INT16 qat_sim 专用（独立 sim；跳过 int16_fixed_eval）"
            if qat_sim_only
            else "INT16 真 kernel 路径（独立 sim；勿与 QDQ 混用 ensure）"
        )
        print(f"\n--- {label} ---")
        t0 = time.time()
        sim_int16 = build_int16_sim(
            fp,
            loaders,
            device,
            max_calib=args.max_calib_batches,
            bitwidth_config=args.bitwidth_config,
            clz_encoding_fix=not args.no_int16_clz_fix,
        )
        print(f"INT16 sim 构建+校准: {time.time() - t0:.1f}s")
        fp.cpu()
        del fp
        release_cuda_memory(sim_int16.model)

        if args.verify_int16_qat_sim:
            print("\n--- verify int16_fixed_qat_sim（可用性验收）---")
            if args.verify_max_delta_pp is not None and float_native_top1 is None:
                print(
                    "  WARNING: --verify-max-delta-pp 需要 float_native baseline，"
                    "当前未跑（qat_sim_only 模式），delta 约束将被跳过"
                )
            t0 = time.time()
            verify_report = verify_int16_qat_sim(
                sim_int16,
                loaders,
                device,
                qat_steps=args.verify_qat_steps,
                qat_scope=args.verify_qat_scope,
                eval_max_batches=args.eval_max_batches or 50,
                min_top1=args.verify_min_top1,
                max_delta_pp=args.verify_max_delta_pp,
                float_native_top1=float_native_top1,
            )
            print(f"verify 总耗时: {time.time() - t0:.1f}s")
            int16_mode_acc = {
                ExecutionMode.INT16_FIXED_EVAL.value: verify_report.get("int16_fixed_eval_top1"),
            }
        else:
            if qat_sim_only or args.eval_int16_qat_sim:
                print(
                    "  WARNING: int16_fixed_qat_sim 是训练 surrogate 路径，"
                    "其 PTQ-only Top-1 不作为验收指标；交付精度看 int16_fixed_eval。"
                )
            t0 = time.time()
            int16_mode_acc = _eval_int16_modes(
                sim_int16.model,
                loaders["test"],
                device,
                max_batches=args.eval_max_batches,
                include_eval=not qat_sim_only,
                include_qat_sim=qat_sim_only or args.eval_int16_qat_sim,
                qat_sim_micro_batch=args.qat_sim_micro_batch,
            )
            print(f"INT16 模式评估: {time.time() - t0:.1f}s")
            for k, v in int16_mode_acc.items():
                tag = (
                    "  [非验收: train surrogate]"
                    if k == ExecutionMode.INT16_FIXED_QAT_SIM.value
                    else ""
                )
                if isinstance(v, float) and float_native_top1 is not None:
                    delta_pp = (v - float_native_top1) * 100
                    print(
                        f"  {k:22s} {v * 100:7.2f}%  (Δ vs float_native {delta_pp:+.2f} pp){tag}"
                    )
                elif isinstance(v, float):
                    print(f"  {k:22s} {v * 100:7.2f}%{tag}")
                else:
                    print(f"  {k:22s} {v}{tag}")

    report = {
        "float_native_top1": float_native_top1,
        "bitwidth_config": str(args.bitwidth_config) if args.bitwidth_config else None,
        "apply_po2": args.apply_po2,
        "clz_sign_bypass": not args.no_clz_sign_bypass if args.clz_encoding_fix else None,
        "qat_lr": args.qat_lr,
        "qat_restore_best": not args.no_qat_restore_best,
        "clz_encoding_fix": args.clz_encoding_fix,
        "clz_post_qat": args.clz_post_qat,
        "clz_encoding_fix_stats": getattr(sim, "_clz_encoding_fix_stats", None) if sim else None,
        "qat_stats": getattr(sim, "_qat_stats", None) if sim else None,
        "skip_qat": args.skip_qat,
        "qat_max_batches": args.qat_max_batches,
        "max_calib_batches": args.max_calib_batches,
        "mode_top1": mode_acc if mode_acc else None,
        "int16_clz_encoding_fix": None if not args.eval_int16 else not args.no_int16_clz_fix,
        "int16_clz_encoding_fix_stats": (
            getattr(sim_int16, "_clz_encoding_fix_stats", None) if sim_int16 else None
        ),
        "int16_mode_top1": int16_mode_acc,
        "eval_int16": args.eval_int16,
        "eval_int16_qat_sim": args.eval_int16_qat_sim or qat_sim_only,
        "eval_int16_qat_sim_only": qat_sim_only,
        "verify_int16_qat_sim": verify_only,
        "verify_int16_qat_sim_report": verify_report if verify_only else None,
        "qat_sim_micro_batch": args.qat_sim_micro_batch,
        "eval_max_batches": args.eval_max_batches,
        "delta_pp_vs_float_native": (
            {
                k: (v - float_native_top1) * 100
                for k, v in mode_acc.items()
                if isinstance(v, float)
            }
            if mode_acc
            else None
        ),
        "int16_delta_pp_vs_float_native": (
            {
                k: (v - float_native_top1) * 100
                for k, v in int16_mode_acc.items()
                if isinstance(v, float)
            }
            if int16_mode_acc and float_native_top1 is not None
            else None
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n报告: {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Path A：quick_start QAT + INT16 评估（目标接近 float baseline）。

与 ``quick_start_int16_metric`` 的区别：
- 使用 ``quick_start.qat_finetune``（已验证 fp32 QAT ~88%）
- 不在 MRNN 分解图上调用 ``ensure_output_quantizers``（会迫使 sign 走 INT16 kernel 但前端无 carrier，直接 RuntimeError）
- 报告 ``fixed_scale_qdq``（定点网格代理，目标接近 baseline）与 ``int16_fixed_eval``（当前分解图仅 ~4%%）
"""
from __future__ import annotations

import contextlib
import os
import sys

os.environ.setdefault("PYTHONHASHSEED", "42")
_DATA = "/home/llq/workspace/data/speech_commands"
if not os.path.isdir(_DATA):
    _DATA = "/mnt/data8t/share/datasets/speech_commands/SpeechCommands/speech_commands_v0.02"

import soundfile as sf
import torch
import torchaudio


def _soundfile_load(path, *args, **kwargs):
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    return torch.from_numpy(data.T.copy()), sr


torchaudio.load = _soundfile_load

import quick_start as qs  # noqa: E402
from aimet_torch.fixed_point import (  # noqa: E402
    ExecutionMode,
    convert_encodings_to_fixed_scale,
    quant_execution_mode,
)
from aimet_torch.fixed_point.metrics import (  # noqa: E402
    int16_eval_allow_debug_float,
    mean_logits_cosine_vs_fp32,
)


def eval_mode(model, loader, device, mode, *, max_batches: int | None = None):
    ctx = (
        int16_eval_allow_debug_float()
        if mode in (ExecutionMode.INT16_FIXED_EVAL, ExecutionMode.INT16_FIXED_QAT_SIM)
        else contextlib.nullcontext()
    )
    model.eval()
    correct, total = 0, 0
    with torch.no_grad(), ctx, quant_execution_mode(mode):
        for idx, (inputs, labels) in enumerate(loader):
            if max_batches is not None and idx >= max_batches:
                break
            inputs, labels = inputs.to(device), labels.to(device)
            out = model(inputs)
            if hasattr(out, "to_float"):
                out = out.to_float()
            preds = out.max(1).indices
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
    return correct / total if total else 0.0


def main() -> None:
    qs.DATA_ROOT = _DATA
    qs.FP_EPOCHS = 0
    qs.QAT_EPOCHS = 1
    qs.MAX_CALIB_BATCHES = 100

    qs.set_seed(qs.SEED)
    qs.setup_audio_backend()
    device = qs.DEVICE
    print("=" * 70)
    print("INT16 baseline probe（full_quant + ensure OQ + quick_start QAT）")
    print("=" * 70)

    loaders = qs.build_dataloaders(qs.DATA_ROOT)
    ckpt = torch.load(qs.FP_MODEL_PATH, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt

    fp = qs.MRNN(output_dim=qs.NUM_CLASSES).to(device)
    fp.load_state_dict(state, strict=False)
    fp_acc = qs.evaluate(fp, loaders["test"], device)
    print(f"float_native (no sim): {fp_acc * 100:.2f}%")

    prepared = qs.model_preparer.prepare_model(
        fp,
        stateless_modules_to_preserve=[
            qs.PowerCompress, qs.HypotFun, qs.CLN, qs.QuantizableBatchNorm2d,
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
    qs.apply_mixed_precision_bitwidth(
        sim.model, config_file=str(qs.BITWIDTH_CONFIG_FILE), verbose=False,
    )

    sim.model.to(device).eval()
    with torch.no_grad(), qs.aimet.nn.compute_encodings(sim.model):
        for idx, (x, _) in enumerate(loaders["calib"]):
            if idx >= qs.MAX_CALIB_BATCHES:
                break
            sim.model(x.to(device))

    ptq = eval_mode(sim.model, loaders["test"], device, ExecutionMode.FP32_QDQ)
    print(f"PTQ fp32_qdq: {ptq * 100:.2f}%")

    qs.apply_power_of_2_workflow(
        sim.model, method="round", tolerance=0.02, align_bias_scale=True, verbose=False,
    )
    po2 = eval_mode(sim.model, loaders["test"], device, ExecutionMode.FP32_QDQ)
    print(f"Po2 fp32_qdq: {po2 * 100:.2f}%")

    qs.freeze_quantizer_parameters(sim.model, verbose=False, freeze_bn_affine=True)
    qs.qat_finetune(sim, loaders["train"], device)

    n_fixed = convert_encodings_to_fixed_scale(sim)
    print(f"convert_encodings_to_fixed_scale: {n_fixed}")
    qat_fp32 = eval_mode(sim.model, loaders["test"], device, ExecutionMode.FP32_QDQ)
    qat_fix = eval_mode(sim.model, loaders["test"], device, ExecutionMode.FIXED_SCALE_QDQ)
    print("\n--- QAT 后（定点验收看 fixed_scale_qdq）---")
    print(f"fp32_qdq:          {qat_fp32 * 100:.2f}%")
    print(f"fixed_scale_qdq:   {qat_fix * 100:.2f}%")
    print(
        f"Δ vs float_native: fp32 {(qat_fp32 - fp_acc) * 100:+.2f} pp  "
        f"fixed_scale {(qat_fix - fp_acc) * 100:+.2f} pp"
    )
    try:
        qat_int16 = eval_mode(sim.model, loaders["test"], device, ExecutionMode.INT16_FIXED_EVAL)
        cos_int = mean_logits_cosine_vs_fp32(
            sim.model, loaders["test"], device,
            cand_mode=ExecutionMode.INT16_FIXED_EVAL, max_batches=20,
        )
        print(f"int16_fixed_eval:  {qat_int16 * 100:.2f}%  (logits cosine vs fp32: {cos_int:.6f})")
    except RuntimeError as exc:
        print(f"int16_fixed_eval:  FAILED — {exc}")


if __name__ == "__main__":
    main()

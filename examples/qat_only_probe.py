#!/usr/bin/env python3
"""从已有 FP checkpoint 快速跑 PTQ→Po2→QAT，用于 bisect aimet_torch 差异。"""
import os
import sys

# 本地数据与 soundfile 后端（无 torchcodec）
os.environ.setdefault("PYTHONHASHSEED", "42")
_DATA = "/home/llq/workspace/data/speech_commands"
if not os.path.isdir(_DATA):
    _DATA = "/mnt/data8t/share/datasets/speech_commands/SpeechCommands/speech_commands_v0.02"

import soundfile as sf
import numpy as np
import torch
import torchaudio


def _soundfile_load(path, *args, **kwargs):
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    return torch.from_numpy(data.T.copy()), sr


torchaudio.load = _soundfile_load

# 必须在 import quick_start 前改 DATA_ROOT
import quick_start as qs  # noqa: E402

qs.DATA_ROOT = _DATA
qs.FP_EPOCHS = 0
qs.QAT_EPOCHS = 1
qs.MAX_CALIB_BATCHES = 100


def main():
    qs.set_seed(qs.SEED)
    qs.setup_audio_backend()
    print("DATA_ROOT:", qs.DATA_ROOT)
    print("FP checkpoint:", qs.FP_MODEL_PATH, "exists:", qs.FP_MODEL_PATH.is_file())

    loaders = qs.build_dataloaders(qs.DATA_ROOT)
    model = qs.MRNN(output_dim=qs.NUM_CLASSES).to(qs.DEVICE)

    if qs.FP_MODEL_PATH.is_file():
        ckpt = torch.load(qs.FP_MODEL_PATH, map_location=qs.DEVICE, weights_only=False)
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state, strict=False)
        fp_accuracy = qs.evaluate(model, loaders["test"], qs.DEVICE)
        print(f"加载 FP 权重后精度: {fp_accuracy * 100:.2f}%")
    else:
        raise FileNotFoundError(f"需要先有 FP 权重: {qs.FP_MODEL_PATH}")

    prepared_model = qs.model_preparer.prepare_model(
        model,
        stateless_modules_to_preserve=[
            qs.PowerCompress, qs.HypotFun, qs.CLN, qs.QuantizableBatchNorm2d,
        ],
    )
    sample_input, _ = next(iter(loaders["train"]))
    dummy_input = sample_input.to(qs.DEVICE)

    sim = qs.quantsim.QuantizationSimModel(
        prepared_model,
        dummy_input=dummy_input,
        quant_scheme=qs.QUANT_SCHEME,
        config_file=str(qs.CONFIG_FILE),
        default_output_bw=qs.DEFAULT_BW,
        default_param_bw=qs.DEFAULT_BW,
    )
    sim.set_percentile_value(qs.PERCENTILE_VALUE)
    qs.apply_mixed_precision_bitwidth(
        sim.model, config_file=str(qs.BITWIDTH_CONFIG_FILE), verbose=False,
    )

    sim.model.to(qs.DEVICE).eval()
    with torch.no_grad(), qs.aimet.nn.compute_encodings(sim.model):
        for idx, (x, _) in enumerate(loaders["calib"]):
            if idx >= qs.MAX_CALIB_BATCHES:
                break
            sim.model(x.to(qs.DEVICE))

    ptq = qs.evaluate(sim.model, loaders["test"], qs.DEVICE)
    qs.apply_power_of_2_workflow(
        sim.model, method="round", tolerance=0.02, align_bias_scale=True, verbose=False,
    )
    po2 = qs.evaluate(sim.model, loaders["test"], qs.DEVICE)

    qs.freeze_quantizer_parameters(sim.model, verbose=False, freeze_bn_affine=True)
    qs.qat_finetune(sim, loaders["train"], qs.DEVICE)
    qat = qs.evaluate(sim.model, loaders["test"], qs.DEVICE)

    tag = os.environ.get("BISECT_TAG", "aimet_rex")
    print(f"\n[{tag}] FP(load)={fp_accuracy*100:.2f}% PTQ={ptq*100:.2f}% Po2={po2*100:.2f}% QAT={qat*100:.2f}%")


if __name__ == "__main__":
    main()

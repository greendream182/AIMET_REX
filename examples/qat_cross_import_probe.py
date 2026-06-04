#!/usr/bin/env python3
"""交叉导入 quick_start / aimet_torch，定位 QAT 差异在脚本还是库。"""
import importlib.util
import os
import sys

_DATA = "/home/llq/workspace/data/speech_commands"
_MAIN = "/home/llq/workspace/aimet_rx-main"
_OLD = "/home/llq/workspace/aimet_rx_old"
_QUANT_GRU = "/home/llq/workspace/quant-gru-pytorch/pytorch"

import soundfile as sf
import torch
import torchaudio


def _soundfile_load(path, *args, **kwargs):
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    return torch.from_numpy(data.T.copy()), sr


torchaudio.load = _soundfile_load


def load_quick_start(which: str):
    examples = f"{_OLD}/examples" if which == "old" else f"{_MAIN}/examples"
    path = f"{examples}/quick_start.py"
    for p in (examples, _QUANT_GRU):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location(f"quick_start_{which}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_probe(qs, tag: str):
    qs.DATA_ROOT = _DATA
    qs.FP_EPOCHS = 0
    qs.QAT_EPOCHS = 1
    qs.MAX_CALIB_BATCHES = 100
    qs.set_seed(qs.SEED)
    qs.setup_audio_backend()

    loaders = qs.build_dataloaders(qs.DATA_ROOT)
    model = qs.MRNN(output_dim=qs.NUM_CLASSES).to(qs.DEVICE)
    ckpt = torch.load(qs.FP_MODEL_PATH, map_location=qs.DEVICE, weights_only=False)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=False)

    prepared = qs.model_preparer.prepare_model(
        model,
        stateless_modules_to_preserve=[
            qs.PowerCompress, qs.HypotFun, qs.CLN, qs.QuantizableBatchNorm2d,
        ],
    )
    dummy_input = next(iter(loaders["train"]))[0].to(qs.DEVICE)
    sim = qs.quantsim.QuantizationSimModel(
        prepared,
        dummy_input=dummy_input,
        quant_scheme=qs.QUANT_SCHEME,
        config_file=str(qs.CONFIG_FILE),
        default_output_bw=qs.DEFAULT_BW,
        default_param_bw=qs.DEFAULT_BW,
    )
    sim.set_percentile_value(qs.PERCENTILE_VALUE)
    qs.apply_mixed_precision_bitwidth(sim.model, config_file=str(qs.BITWIDTH_CONFIG_FILE), verbose=False)

    sim.model.to(qs.DEVICE).eval()
    with torch.no_grad(), qs.aimet.nn.compute_encodings(sim.model):
        for idx, (x, _) in enumerate(loaders["calib"]):
            if idx >= qs.MAX_CALIB_BATCHES:
                break
            sim.model(x.to(qs.DEVICE))

    ptq = qs.evaluate(sim.model, loaders["test"], qs.DEVICE)
    qs.apply_power_of_2_workflow(sim.model, method="round", tolerance=0.02, align_bias_scale=True, verbose=False)
    po2 = qs.evaluate(sim.model, loaders["test"], qs.DEVICE)
    qs.freeze_quantizer_parameters(sim.model, verbose=False, freeze_bn_affine=True)
    qs.qat_finetune(sim, loaders["train"], qs.DEVICE)
    qat = qs.evaluate(sim.model, loaders["test"], qs.DEVICE)
    print(f"\n[{tag}] PTQ={ptq*100:.2f}% Po2={po2*100:.2f}% QAT={qat*100:.2f}%")


def main():
    combo = os.environ.get("CROSS_COMBO", "main_qs_main_aimet")
    # 清空已加载的 aimet_torch，便于切换 PYTHONPATH
    for k in list(sys.modules):
        if k == "aimet_torch" or k.startswith("aimet_torch."):
            del sys.modules[k]

    if combo == "main_qs_main_aimet":
        sys.path[:0] = [_MAIN, _QUANT_GRU]
        qs = load_quick_start("main")
        qs.FP_MODEL_PATH = __import__("pathlib").Path(_MAIN) / "examples" / "model_fp.pth"
    elif combo == "old_qs_old_aimet":
        sys.path[:0] = [_OLD, _QUANT_GRU]
        qs = load_quick_start("old")
        qs.FP_MODEL_PATH = __import__("pathlib").Path(_MAIN) / "examples" / "model_fp.pth"
    elif combo == "old_qs_main_aimet":
        sys.path[:0] = [_MAIN, _OLD + "/examples", _QUANT_GRU]
        qs = load_quick_start("old")
        qs.FP_MODEL_PATH = __import__("pathlib").Path(_MAIN) / "examples" / "model_fp.pth"
    elif combo == "main_qs_old_aimet":
        sys.path[:0] = [_OLD, _MAIN + "/examples", _QUANT_GRU]
        qs = load_quick_start("main")
        qs.FP_MODEL_PATH = __import__("pathlib").Path(_MAIN) / "examples" / "model_fp.pth"
    else:
        raise SystemExit(f"unknown CROSS_COMBO={combo}")

    run_probe(qs, combo)


if __name__ == "__main__":
    main()

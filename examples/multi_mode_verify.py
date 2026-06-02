#!/usr/bin/env python3
"""PTQ→Po2→QAT 后在各 ExecutionMode 下评估（验收 conv1d 修复 + 多模式）。"""
import contextlib
import os

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
import quick_start_int16_metric as qs_int16  # noqa: E402
from aimet_torch.fixed_point import (
    ExecutionMode,
    convert_encodings_to_fixed_scale,
    quant_execution_mode,
)
from aimet_torch.fixed_point.metrics import int16_eval_allow_debug_float

qs.DATA_ROOT = _DATA
qs.FP_EPOCHS = 0
qs.QAT_EPOCHS = 1
qs.MAX_CALIB_BATCHES = 100


def eval_mode(model, loader, device, mode):
    ctx = (
        int16_eval_allow_debug_float()
        if mode in (ExecutionMode.INT16_FIXED_EVAL, ExecutionMode.INT16_FIXED_QAT_SIM)
        else contextlib.nullcontext()
    )
    with ctx, quant_execution_mode(mode):
        return qs.evaluate(model, loader, device)


def build_sim_qs(fp_model, dummy, bitwidth_json):
    """quick_start 路径：full_quant，无 INT16 ensure。"""
    prepared = qs.model_preparer.prepare_model(
        fp_model,
        stateless_modules_to_preserve=[
            qs.PowerCompress, qs.HypotFun, qs.CLN, qs.QuantizableBatchNorm2d,
        ],
    )
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
        sim.model, config_file=str(bitwidth_json), verbose=False,
    )
    return sim


def run_calib_ptq_po2_qat(sim, loaders, device, *, apply_po2: bool = True):
    sim.model.to(device).eval()
    with torch.no_grad(), qs.aimet.nn.compute_encodings(sim.model):
        for idx, (x, _) in enumerate(loaders["calib"]):
            if idx >= qs.MAX_CALIB_BATCHES:
                break
            sim.model(x.to(device))

    ptq = eval_mode(sim.model, loaders["test"], device, ExecutionMode.FP32_QDQ)
    if apply_po2:
        qs.apply_power_of_2_workflow(
            sim.model, method="round", tolerance=0.02, align_bias_scale=True, verbose=False,
        )
        po2 = eval_mode(sim.model, loaders["test"], device, ExecutionMode.FP32_QDQ)
    else:
        po2 = ptq
        print("  (跳过全图 Po2，与 quick_start_int16_metric 默认一致)")
    qs.freeze_quantizer_parameters(sim.model, verbose=False, freeze_bn_affine=True)
    qs.qat_finetune(sim, loaders["train"], device)
    qat = eval_mode(sim.model, loaders["test"], device, ExecutionMode.FP32_QDQ)
    return ptq, po2, qat


def eval_modes_after_qat(sim, loaders, device, dummy, modes, *, convert_fixed: bool = True):
    results = {}
    if convert_fixed:
        n = convert_encodings_to_fixed_scale(sim)
        print(f"  convert_encodings_to_fixed_scale: {n} quantizers")

    for mode in modes:
        try:
            acc = eval_mode(sim.model, loaders["test"], device, mode)
            results[mode.value] = acc
            print(f"  {mode.value:28s} {acc * 100:7.2f}%")
        except Exception as exc:  # noqa: BLE001
            results[mode.value] = f"FAILED: {exc}"
            print(f"  {mode.value:28s} FAILED — {exc}")

    if ExecutionMode.INT16_FIXED_QAT_SIM in modes:
        print("  --- INT16_FIXED_QAT_SIM backward 冒烟 ---")
        sim.model.train()
        x = dummy.to(device)
        try:
            with int16_eval_allow_debug_float(), quant_execution_mode(
                ExecutionMode.INT16_FIXED_QAT_SIM,
            ):
                y = sim.model(x)
            loss = y.to_float() if hasattr(y, "to_float") else y
            loss.sum().backward()
            print("  backward: OK")
            results["int16_backward"] = "OK"
        except Exception as exc:  # noqa: BLE001
            print(f"  backward: FAILED — {exc}")
            results["int16_backward"] = f"FAILED: {exc}"
        finally:
            sim.model.eval()
    return results


def main():
    qs.set_seed(qs.SEED)
    qs.setup_audio_backend()
    print("=" * 70)
    print("多模式验收（model_preparer 无 conv1d 特殊处理）")
    print("=" * 70)

    loaders = qs.build_dataloaders(qs.DATA_ROOT)
    ckpt = torch.load(qs.FP_MODEL_PATH, map_location=qs.DEVICE, weights_only=False)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt

    fp_probe = qs.MRNN(output_dim=qs.NUM_CLASSES).to(qs.DEVICE)
    fp_probe.load_state_dict(state, strict=False)
    fp_load = qs.evaluate(fp_probe, loaders["test"], qs.DEVICE)
    print(f"FP 权重加载后: {fp_load * 100:.2f}%")

    dummy = next(iter(loaders["train"]))[0].to(qs.DEVICE)

    print("\n--- 路径 A: quick_start_full_quant（fp32/fp16/fixed_scale + QAT）---")
    fp_a = qs.MRNN(output_dim=qs.NUM_CLASSES).to(qs.DEVICE)
    fp_a.load_state_dict(state, strict=False)
    sim_a = build_sim_qs(fp_a, dummy, qs.BITWIDTH_CONFIG_FILE)
    ptq, po2, qat = run_calib_ptq_po2_qat(sim_a, loaders, qs.DEVICE)
    print(f"  PTQ={ptq*100:.2f}% Po2={po2*100:.2f}% QAT={qat*100:.2f}%")
    print("  QAT 后模式评估:")
    res_a = eval_modes_after_qat(
        sim_a,
        loaders,
        qs.DEVICE,
        dummy,
        (
            ExecutionMode.FP32_QDQ,
            ExecutionMode.FP16_QDQ,
            ExecutionMode.FIXED_SCALE_QDQ,
            ExecutionMode.INT16_FIXED_EVAL,
        ),
    )

    print("\n--- 路径 B: mrnn_acceptance（INT16；build_sim 同 int16_metric）---")
    int16_bw = qs._HERE / "config" / "mrnn_acceptance_mixed_precision.json"
    res_b: dict = {}
    ptq_b = po2_b = qat_b = None
    if not int16_bw.is_file():
        print(f"  跳过：未找到 {int16_bw}")
    else:
        fp_b = qs.MRNN(output_dim=qs.NUM_CLASSES).to(qs.DEVICE)
        fp_b.load_state_dict(state, strict=False)
        sim_b = qs_int16.build_sim(fp_b, dummy, bitwidth_config=int16_bw)
        # INT16 验收：calib 后转 fixed_scale，默认不跑全图 Po2；可选 fp32 QAT 再评 INT16
        sim_b.model.to(qs.DEVICE).eval()
        with torch.no_grad(), qs.aimet.nn.compute_encodings(sim_b.model):
            for idx, (x, _) in enumerate(loaders["calib"]):
                if idx >= qs.MAX_CALIB_BATCHES:
                    break
                sim_b.model(x.to(qs.DEVICE))
        ptq_b = eval_mode(sim_b.model, loaders["test"], qs.DEVICE, ExecutionMode.FP32_QDQ)
        convert_encodings_to_fixed_scale(sim_b)
        print(f"  PTQ(fp32)={ptq_b*100:.2f}%（无 Po2）")
        print("  INT16 模式评估（post-calib，无 fp32 QAT）:")
        res_b = eval_modes_after_qat(
            sim_b,
            loaders,
            qs.DEVICE,
            dummy,
            (ExecutionMode.INT16_FIXED_EVAL, ExecutionMode.INT16_FIXED_QAT_SIM),
            convert_fixed=False,
        )

    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)
    print(f"  FP(load)              {fp_load * 100:.2f}%")
    print(f"  [A] PTQ/Po2/QAT       {ptq*100:.2f}% / {po2*100:.2f}% / {qat*100:.2f}%")
    for k in ("fp32_qdq", "fp16_qdq", "fixed_scale_qdq", "int16_fixed_eval"):
        v = res_a.get(k)
        if isinstance(v, float):
            print(f"  [A] {k:22s} {v * 100:.2f}%")
        elif v:
            print(f"  [A] {k:22s} {v}")
    if ptq_b is not None:
        print(f"  [B] PTQ(fp32)         {ptq_b*100:.2f}%")
        for k in ("int16_fixed_eval", "int16_fixed_qat_sim", "int16_backward"):
            v = res_b.get(k)
            if v:
                print(f"  [B] {k:22s} {v if isinstance(v, str) else f'{float(v)*100:.2f}%'}")

    qat_ok = isinstance(qat, float) and qat > 0.5
    fp16_ok = isinstance(res_a.get("fp16_qdq"), float)
    int16_ok = isinstance(res_b.get("int16_fixed_eval"), float)
    print(f"\n路径 A QAT 修复:     {'✅' if qat_ok else '❌'}")
    print(f"路径 A fp16_qdq:     {'✅' if fp16_ok else '❌'}")
    print(f"路径 B int16_fixed_eval: {'✅' if int16_ok else '❌'}")


if __name__ == "__main__":
    main()

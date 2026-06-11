#!/usr/bin/env python3
"""SYS-FU-2 FU-2-3: ConvT isolated vs chained SQNR (baseline vs upstream_only)."""

from __future__ import annotations

import argparse
import copy
import math
import os
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_EXAMPLES = _REPO_ROOT / "examples"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))
_QG = _REPO_ROOT.parent / "quant-gru-pytorch" / "pytorch"
if _QG.is_dir() and str(_QG) not in sys.path:
    sys.path.insert(0, str(_QG))

import aimet_torch.v2 as aimet  # noqa: E402
from aimet_torch.fixed_point import convert_encodings_to_fixed_scale  # noqa: E402
from common.mrnn_clz_encoding import (  # noqa: E402
    apply_mrnn_clz_encoding_fixes,
    apply_mrnn_clz_encoding_fixes_post_calib,
    collect_power2_float_out_fmax,
)
from quick_start import DEVICE, FP_MODEL_PATH, MRNN, set_seed, setup_audio_backend  # noqa: E402
from quick_start_int16_metric import (  # noqa: E402
    SYSQ1_W6_PERCENTILE_VALUE,
    _calib_fn,
    _patch_torchaudio_with_soundfile,
    build_sim,
)
from sys_fu2_teacher_forced import chained_sqnr, teacher_forced_sqnr  # noqa: E402

_CONVT_SUFFIXES = (
    "enc_seqs.0.conv_t",
    "enc_seqs.1.conv_t",
    "neck_seqs.0.conv_t",
    "neck_seqs.1.conv_t",
)


def _activation_only_conv_down(sim_model: torch.nn.Module) -> None:
    """Keep activation quant on conv_in + freq_downs only; preserve param quantizers."""

    from quick_start_int16_metric import _activation_scope_match

    for name, module in sim_model.named_modules():
        if not type(module).__name__.startswith("Quantized"):
            continue
        keep = _activation_scope_match(name, "conv_down")
        for attr in ("input_quantizers", "output_quantizers"):
            qcontainer = getattr(module, attr, None)
            if qcontainer is None:
                continue
            keys = (
                list(qcontainer.keys())
                if isinstance(qcontainer, torch.nn.ModuleDict)
                else list(range(len(qcontainer)))
            )
            for key in keys:
                if qcontainer[key] is None:
                    continue
                if not keep:
                    qcontainer[key] = None


def _prepare_sim(
    model: torch.nn.Module,
    sample: torch.Tensor,
    loaders,
    device: torch.device,
    *,
    bitwidth_config: Path,
    max_calib_batches: int,
    percentile_value: float,
    upstream_only: bool,
):
    sim, prepared_float = build_sim(
        model,
        sample,
        bitwidth_config=str(bitwidth_config),
        quant_scheme="percentile",
        percentile_value=percentile_value,
        native_trans=True,
    )
    if upstream_only:
        _activation_only_conv_down(sim.model)
    sim.model.to(device).eval()
    apply_mrnn_clz_encoding_fixes(sim.model, sign_input_bypass=False, verbose=False)
    calib = _calib_fn(sim.model, loaders["calib"], device, max_calib_batches)
    with torch.no_grad(), aimet.nn.compute_encodings(sim.model):
        calib(sim.model)
    if prepared_float is not None:
        pf = prepared_float.to(device).eval()
        apply_mrnn_clz_encoding_fixes_post_calib(
            sim.model,
            power2_float_out_fmax=collect_power2_float_out_fmax(
                pf, loaders["calib"], device, max_calib_batches,
            ),
            verbose=False,
        )
    convert_encodings_to_fixed_scale(sim)
    return sim


def main() -> None:
    parser = argparse.ArgumentParser(description="SYS-FU-2 FU-2-3 ConvT upstream isolation")
    parser.add_argument(
        "--data-root",
        default=os.environ.get("SPEECH_COMMANDS_ROOT", "/home/llq/workspace/data/speech_commands"),
    )
    parser.add_argument("--fp-ckpt", type=str, default=str(FP_MODEL_PATH))
    parser.add_argument("--max-calib-batches", type=int, default=16)
    parser.add_argument("--percentile-value", type=float, default=SYSQ1_W6_PERCENTILE_VALUE)
    args = parser.parse_args()

    set_seed(0)
    setup_audio_backend()
    _patch_torchaudio_with_soundfile()

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    model = MRNN().to(device)
    ckpt = Path(args.fp_ckpt)
    if not ckpt.is_file():
        sys.exit(f"fp ckpt not found: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    import quick_start as qs

    prev_bs = qs.BATCH_SIZE
    qs.BATCH_SIZE = 32
    try:
        loaders = qs.build_dataloaders(str(args.data_root))
    finally:
        qs.BATCH_SIZE = prev_bs

    sample_input = next(iter(loaders["calib"]))[0][:1].to(device)
    bitwidth_config = _REPO_ROOT / "examples" / "config" / "quick_start_full_quant.json"

    print(
        f"\n=== SYS-FU-2 FU-2-3 ConvT isolated vs chained "
        f"(calib_batches={args.max_calib_batches}) ===\n"
    )

    modes = (("baseline", False), ("upstream_only", True))
    iso: dict[str, dict[str, float]] = {}
    chn: dict[str, dict[str, float]] = {}
    for label, upstream_only in modes:
        sim = _prepare_sim(
            copy.deepcopy(model),
            sample_input,
            loaders,
            device,
            bitwidth_config=bitwidth_config,
            max_calib_batches=args.max_calib_batches,
            percentile_value=args.percentile_value,
            upstream_only=upstream_only,
        )
        iso[label] = teacher_forced_sqnr(sim.model, sample_input, _CONVT_SUFFIXES)
        chn[label] = chained_sqnr(sim.model, sample_input, _CONVT_SUFFIXES)

    header = (
        f"{'module':28s} {'iso_base':>8s} {'chn_base':>8s} "
        f"{'iso_up':>8s} {'chn_up':>8s} {'Δchn':>7s}"
    )
    print(header)
    for suffix in _CONVT_SUFFIXES:
        ib = iso["baseline"].get(suffix, float("nan"))
        cb = chn["baseline"].get(suffix, float("nan"))
        iu = iso["upstream_only"].get(suffix, float("nan"))
        cu = chn["upstream_only"].get(suffix, float("nan"))
        delta = cu - cb if math.isfinite(cu) and math.isfinite(cb) else float("nan")
        print(
            f"{suffix:28s} {ib:8.2f} {cb:8.2f} {iu:8.2f} {cu:8.2f} {delta:+7.2f}"
        )
    print(
        "\niso_* = teacher-forced; chn_* = full-graph chained. "
        "upstream_only = activation quant only on conv_in + freq_downs.*."
    )


if __name__ == "__main__":
    main()

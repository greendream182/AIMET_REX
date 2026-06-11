#!/usr/bin/env python3
"""SYS-FU-2 FU-2-2: frontend activation-quant ablation (teacher-forced)."""

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
from sys_fu2_teacher_forced import teacher_forced_sqnr  # noqa: E402

_ABLATION_SCOPES: dict[str, tuple[str, ...]] = {
    "conv_in_only": ("conv_in",),
    "conv_in_fd0": ("conv_in", "freq_downs.0"),
    "conv_down": ("conv_in", "freq_downs."),
}

_WATCH_SUFFIXES = (
    "freq_downs.0.conv2d",
    "freq_downs.1.conv2d",
    "freq_downs.2.conv2d",
    "enc_seqs.0.conv_t",
)


def _scope_match(module_name: str, prefixes: tuple[str, ...]) -> bool:
    for prefix in prefixes:
        if prefix.endswith("."):
            if module_name.startswith(prefix):
                return True
        elif module_name == prefix or module_name.startswith(prefix + "."):
            return True
    return False


def keep_activation_quantizers_for_prefixes(
    sim_model: torch.nn.Module,
    prefixes: tuple[str, ...],
) -> tuple[int, int, int]:
    """Disable activation quantizers outside ``prefixes``; keep all param quantizers."""

    act_disabled = 0
    act_kept = 0
    for name, module in sim_model.named_modules():
        if not type(module).__name__.startswith("Quantized"):
            continue
        keep = _scope_match(name, prefixes)
        for attr in ("input_quantizers", "output_quantizers"):
            qcontainer = getattr(module, attr, None)
            if qcontainer is None:
                continue
            keys = list(qcontainer.keys()) if isinstance(qcontainer, torch.nn.ModuleDict) else list(range(len(qcontainer)))
            for key in keys:
                if qcontainer[key] is None:
                    continue
                if keep:
                    act_kept += 1
                else:
                    qcontainer[key] = None
                    act_disabled += 1
    return 0, act_disabled, act_kept


def main() -> None:
    parser = argparse.ArgumentParser(description="SYS-FU-2 FU-2-2 frontend ablation")
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
        f"\n=== SYS-FU-2 FU-2-2 frontend ablation (teacher-forced) "
        f"(calib_batches={args.max_calib_batches}) ===\n"
    )

    results: dict[str, dict[str, float]] = {}
    for scope_key, prefixes in _ABLATION_SCOPES.items():
        sim, prepared_float = build_sim(
            copy.deepcopy(model),
            sample_input,
            bitwidth_config=str(bitwidth_config),
            quant_scheme="percentile",
            percentile_value=args.percentile_value,
            native_trans=True,
        )
        _p, _a, a_keep = keep_activation_quantizers_for_prefixes(sim.model, prefixes)
        print(f"[{scope_key}] kept {a_keep} activation slots")
        sim.model.to(device).eval()
        apply_mrnn_clz_encoding_fixes(sim.model, sign_input_bypass=False, verbose=False)
        calib = _calib_fn(sim.model, loaders["calib"], device, args.max_calib_batches)
        with torch.no_grad(), aimet.nn.compute_encodings(sim.model):
            calib(sim.model)
        if prepared_float is not None:
            pf = prepared_float.to(device).eval()
            apply_mrnn_clz_encoding_fixes_post_calib(
                sim.model,
                power2_float_out_fmax=collect_power2_float_out_fmax(
                    pf, loaders["calib"], device, args.max_calib_batches,
                ),
                verbose=False,
            )
        convert_encodings_to_fixed_scale(sim)
        results[scope_key] = teacher_forced_sqnr(sim.model, sample_input, _WATCH_SUFFIXES)

    header = f"{'scope':16s} " + " ".join(f"{s.split('.')[-1]:>12s}" for s in _WATCH_SUFFIXES)
    print(header)
    for scope_key in _ABLATION_SCOPES:
        cols = []
        for suffix in _WATCH_SUFFIXES:
            sq = results[scope_key].get(suffix, float("nan"))
            cols.append(f"{sq:12.2f}" if math.isfinite(sq) else f"{'n/a':>12s}")
        print(f"{scope_key:16s} " + " ".join(cols))
    print("\nColumns = teacher-forced isolated INT16 SQNR (dB) vs fp32 module output.")


if __name__ == "__main__":
    main()

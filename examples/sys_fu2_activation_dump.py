#!/usr/bin/env python3
"""SYS-FU-2 path-1: dump MRNN backbone activation distributions (fp32 forward).

Hooks Part A / Part B suspect nodes from SYS-OPEN-Q-1 and records per-channel
abs-percentile tables to diagnose long-tail / multi-mode / per-tensor grid
mismatch.

Usage (from repo root, inside quant-gru container)::

    python examples/sys_fu2_activation_dump.py \\
        --max-calib-batches 16 \\
        --fp-ckpt examples/model_fp.pth

Output: stdout table + optional ``--json-out /tmp/sysfu2_act.json``.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_QG = _REPO_ROOT.parent / "quant-gru-pytorch" / "pytorch"
if _QG.is_dir() and str(_QG) not in sys.path:
    sys.path.insert(0, str(_QG))

from aimet_torch import model_preparer  # noqa: E402
from aimet_torch.quantizable_batchnorm import QuantizableBatchNorm2d  # noqa: E402
from common.torch_stft import STFT  # noqa: E402
from quick_start import (  # noqa: E402
    BATCH_SIZE,
    CLN,
    DEVICE,
    FP_MODEL_PATH,
    HypotFun,
    MRNN,
    PowerCompress,
    build_dataloaders,
    set_seed,
    setup_audio_backend,
)

# Part A (calib-sensitive) vs Part B (calib-insensitive) from W6 sweep.
_HOOK_TARGETS = (
    "conv_in",
    "freq_downs.0.conv2d",
    "freq_downs.1.conv2d",
    "freq_downs.2.conv2d",
    "enc_seqs.0.conv_t",
    "enc_seqs.1.conv_t",
    "neck_seqs.0.conv_t",
    "neck_seqs.1.conv_t",
    "enc_seqs.0.cln",
    "enc_seqs.1.cln",
)

_PART_A = frozenset({"freq_downs.2.conv2d", "neck_seqs.1.conv_t"})
_PART_B = frozenset({
    "freq_downs.0.conv2d",
    "freq_downs.1.conv2d",
    "neck_seqs.0.conv_t",
    "enc_seqs.0.conv_t",
    "enc_seqs.1.conv_t",
})

_MAX_SAMPLES_PER_CHANNEL = 65536
_PERCENTILES = (50.0, 99.0, 99.9, 99.99, 100.0)


def _find_modules(model: torch.nn.Module, suffix: str) -> list[tuple[str, torch.nn.Module]]:
    """Return all (qualname, module) whose qualname equals or ends with suffix."""
    hits: list[tuple[str, torch.nn.Module]] = []
    for qualname, mod in model.named_modules():
        if qualname == suffix or qualname.endswith("." + suffix):
            hits.append((qualname, mod))
    return hits


def _register_hooks(
    model: torch.nn.Module,
    targets: tuple[str, ...],
    collector: _ActivationCollector,
) -> list:
    handles = []
    for suffix in targets:
        hits = _find_modules(model, suffix)
        if not hits:
            print(f"WARNING: module not found: {suffix}")
            continue
        if len(hits) > 1:
            print(f"WARNING: multiple matches for {suffix}: {[h[0] for h in hits]}")
        qualname, mod = hits[0]
        if suffix.endswith(".conv2d") or suffix.endswith(".conv_t") or suffix == "conv_in":
            handles.append(mod.register_forward_pre_hook(collector.hook_pre(qualname)))
        else:
            handles.append(mod.register_forward_hook(collector.hook_post(qualname)))
    return handles


def _subsample_row(row: torch.Tensor) -> torch.Tensor:
    n = row.numel()
    if n <= _MAX_SAMPLES_PER_CHANNEL // 16:
        return row.reshape(-1)
    k = _MAX_SAMPLES_PER_CHANNEL // 16
    idx = torch.randperm(n, device=row.device)[:k]
    return row.reshape(-1)[idx]


def _flat_channels(x: torch.Tensor) -> torch.Tensor:
    """Return (C, N) abs activations."""
    x = x.detach().float().abs()
    if x.ndim == 4:
        return x.permute(1, 0, 2, 3).reshape(x.shape[1], -1)
    if x.ndim == 3:
        return x.permute(1, 0, 2).reshape(x.shape[1], -1)
    if x.ndim == 2:
        return x.T
    return x.reshape(1, -1)


def _percentiles_from_chunks(chunks: list[torch.Tensor]) -> dict[str, float]:
    """Merge per-batch activations via per-channel subsample pool."""
    if not chunks:
        return {}

    per_ch_parts: dict[int, list[torch.Tensor]] = defaultdict(list)
    gmax = 0.0
    gsum = 0.0
    gcount = 0
    n_channels = None

    for chunk in chunks:
        flat = _flat_channels(chunk)
        if n_channels is None:
            n_channels = flat.shape[0]
        gmax = max(gmax, float(flat.max().item()))
        gsum += float(flat.sum().item())
        gcount += flat.numel()
        for c in range(flat.shape[0]):
            per_ch_parts[c].append(_subsample_row(flat[c]))

    ch_p: dict[str, list[float]] = {f"p{p:g}": [] for p in _PERCENTILES}
    ratios: list[float] = []
    low_frac = 0

    for c in range(n_channels or 0):
        merged = torch.cat(per_ch_parts[c])
        if merged.numel() > _MAX_SAMPLES_PER_CHANNEL:
            idx = torch.randperm(merged.numel())[:_MAX_SAMPLES_PER_CHANNEL]
            merged = merged[idx]
        p50 = float(torch.quantile(merged, 0.50).item())
        p9999 = float(torch.quantile(merged, 0.9999).item())
        for p in _PERCENTILES:
            ch_p[f"p{p:g}"].append(float(torch.quantile(merged, p / 100.0).item()))
        ratios.append(p9999 / max(p50, 1e-8))
        if p50 < 0.01 * max(p9999, 1e-8):
            low_frac += 1

    out: dict[str, float] = {
        "channels": float(n_channels or 0),
        "global_abs_max": gmax,
        "global_abs_mean": gsum / max(gcount, 1),
        "batches": len(chunks),
        "tensor_shape": list(chunks[-1].shape),
    }
    for p in _PERCENTILES:
        key = f"p{p:g}"
        vals = torch.tensor(ch_p[key])
        out[f"{key}_med"] = float(vals.median().item())
        out[f"{key}_max"] = float(vals.max().item())
    ratio_t = torch.tensor(ratios)
    out["dr_p9999_over_p50_med"] = float(ratio_t.median().item())
    out["dr_p9999_over_p50_max"] = float(ratio_t.max().item())
    out["low_p50_frac"] = low_frac / max(n_channels or 1, 1)
    return out


class _ActivationCollector:
    def __init__(self) -> None:
        self._bufs: dict[str, list[torch.Tensor]] = defaultdict(list)

    def hook_pre(self, name: str):
        def _fn(_mod, inputs):
            if inputs and isinstance(inputs[0], torch.Tensor):
                self._bufs[name].append(inputs[0].detach().cpu())
        return _fn

    def hook_post(self, name: str):
        def _fn(_mod, _inputs, output):
            if isinstance(output, torch.Tensor):
                self._bufs[name].append(output.detach().cpu())
            elif isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
                self._bufs[name].append(output[0].detach().cpu())
        return _fn

    def summarize(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for name, chunks in self._bufs.items():
            stats = _percentiles_from_chunks(chunks)
            if stats:
                out[name] = stats
        return out


def _prepare_mrnn(model: torch.nn.Module, dummy: torch.Tensor) -> torch.nn.Module:
    return model_preparer.prepare_model(
        copy.deepcopy(model),
        stateless_modules_to_preserve=[PowerCompress, HypotFun, CLN, QuantizableBatchNorm2d],
    )


def _patch_torchaudio_with_soundfile() -> None:
    import soundfile as sf
    import torchaudio

    def _load(path, frame_offset=0, num_frames=-1, normalize=True, channels_first=True, **kwargs):
        del frame_offset, num_frames, normalize, kwargs
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        tensor = torch.from_numpy(data.T.copy())
        if not channels_first:
            tensor = tensor.T
        return tensor, sr

    torchaudio.load = _load  # type: ignore[method-assign]


def main() -> None:
    parser = argparse.ArgumentParser(description="SYS-FU-2 MRNN activation dump")
    parser.add_argument(
        "--data-root",
        default=os.environ.get("SPEECH_COMMANDS_ROOT", "/home/llq/workspace/data/speech_commands"),
    )
    parser.add_argument("--fp-ckpt", type=str, default=str(FP_MODEL_PATH))
    parser.add_argument("--max-calib-batches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--json-out", type=str, default="")
    parser.add_argument("--device", type=str, default=DEVICE)
    args = parser.parse_args()

    set_seed(0)
    setup_audio_backend()
    _patch_torchaudio_with_soundfile()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = MRNN().to(device)
    ckpt = Path(args.fp_ckpt)
    if not ckpt.is_file():
        sys.exit(f"fp ckpt not found: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    dummy = torch.randn(1, 16000, device=device)
    prepared = _prepare_mrnn(model, dummy).to(device)

    collector = _ActivationCollector()
    handles = _register_hooks(prepared, _HOOK_TARGETS, collector)

    import quick_start as qs

    prev_bs = qs.BATCH_SIZE
    qs.BATCH_SIZE = args.batch_size
    try:
        loaders = build_dataloaders(str(args.data_root))
        calib_loader = loaders["calib"]
    finally:
        qs.BATCH_SIZE = prev_bs

    with torch.no_grad():
        for bi, (inputs, _labels) in enumerate(calib_loader):
            if bi >= args.max_calib_batches:
                break
            prepared(inputs.to(device))

    for h in handles:
        h.remove()

    summary = collector.summarize()
    if not summary:
        sys.exit("No activations collected — check hook targets / dataloader.")

    print(
        f"\n=== SYS-FU-2 activation dump "
        f"(fp32 prepared MRNN, {args.max_calib_batches} calib batches, device={device}) ===\n"
    )
    header = (
        f"{'module':28s} {'grp':5s} {'C':>4s} {'|x|max':>8s} "
        f"{'p50_med':>8s} {'p99_med':>8s} {'p99.99_med':>10s} "
        f"{'DR_med':>7s} {'low%':>6s}"
    )
    # Map collected qualnames back to W6 group labels via suffix match.
    def _w6_group(qualname: str) -> str:
        for suffix in _PART_A:
            if qualname == suffix or qualname.endswith("." + suffix):
                return "A"
        for suffix in _PART_B:
            if qualname == suffix or qualname.endswith("." + suffix):
                return "B"
        return "-"

    # Stable print order: follow _HOOK_TARGETS suffix order.
    ordered: list[tuple[str, dict]] = []
    for suffix in _HOOK_TARGETS:
        for qualname, stats in summary.items():
            if qualname == suffix or qualname.endswith("." + suffix):
                ordered.append((qualname, stats))
                break

    print(header)
    for qualname, s in ordered:
        grp = _w6_group(qualname)
        print(
            f"{qualname:28s} {grp:5s} {int(s['channels']):4d} "
            f"{s['global_abs_max']:8.4f} "
            f"{s.get('p50_med', 0):8.4f} "
            f"{s.get('p99_med', 0):8.4f} "
            f"{s.get('p99.99_med', 0):10.4f} "
            f"{s.get('dr_p9999_over_p50_med', 0):7.1f} "
            f"{100 * s.get('low_p50_frac', 0):5.1f}%"
        )

    print(
        "\nLegend: grp A/B = SYS-OPEN-Q-1 W6 part A/B backbone nodes; "
        "DR_med = median(p99.99/p50) per channel; "
        "low% = fraction of channels with p50 < 1% of p99.99 "
        "(multi-mode / low-magnitude dominance proxy)."
    )

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()

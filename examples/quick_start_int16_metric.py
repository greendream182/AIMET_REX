#!/usr/bin/env python3
# -*- mode: python -*-
"""MRNN SpeechCommands：INT16_FIXED_EVAL 整图验收（默认）。

本脚本 sim 含 ``ensure_output_quantizers_for_int16_eval``，**不可**在同 sim 上评
fp32/fp16/fixed_scale QDQ（会得到 0% 等无效结果）。QDQ 三档请用
``int16_whole_graph_vs_float_native.py``。

**STFT 边界（默认）**：``trans``（STFT）走**板端独立硬化模块**，不参与
AIMET INT16_FIXED_EVAL 定点化；默认 ``--native-trans`` 保留 fp32 黑盒 leaf
（``module_classes_to_exclude=[STFT]`` + 关闭 trans quantizer）。仅软件
decomposed STFT 对照实验需显式 ``--no-native-trans``。

用法（在 ``examples/`` 目录下）::

    cd /path/to/aimet_rx-main/examples
    python quick_start_int16_metric.py \\
        --data-root /home/llq/workspace/data/speech_commands \\
        --bitwidth-config config/mrnn_acceptance_mixed_precision.json

默认：**不**全图 Po2；校准 → 可选 ``--apply-m-po2``（M/2^n 写回 + re-calib）→ CLZ encoding fix → ``convert_encodings_to_fixed_scale``。
可选 ``--apply-po2`` 为 legacy 1/2^n 全图圆整。
INT16 QAT 用 ``--qat-epochs``。
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import os
import sys
import time
from pathlib import Path

import torch

# aimet_rx-main 根目录（含 aimet_torch / aimet_common）；与 tests 里 pytest 从仓库根跑等价
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# quant_gru sibling checkout（与 tests/fixed_point/conftest 一致）
_QG = _REPO_ROOT.parent / "quant-gru-pytorch" / "pytorch"
if _QG.is_dir() and str(_QG) not in sys.path:
    sys.path.insert(0, str(_QG))

import aimet_torch.v2 as aimet  # noqa: E402
import aimet_torch.fixed_point.kernels  # noqa: F401, E402
from aimet_torch import model_preparer  # noqa: E402
from aimet_torch.fixed_point import (  # noqa: E402
    ExecutionMode,
    QatTrainScope,
    convert_encodings_to_fixed_scale,
    diagnose_int16_readiness,
    ensure_output_quantizers_for_int16_eval,
    quant_execution_mode,
    run_int16_qat_epochs,
    run_int16_qat_steps,
    select_qat_trainable_parameters,
)
from aimet_torch.fixed_point.qat_train import (  # noqa: E402
    cache_training_batches,
    release_cuda_memory,
    suggest_int16_qat_batch_size,
    warm_int16_qat_lut_cache,
)
from aimet_torch.m_po2_quantization import apply_m_po2_recalib_workflow  # noqa: E402
from aimet_torch.fixed_point.metrics import int16_eval_allow_debug_float  # noqa: E402
from aimet_torch.utils_rx import apply_mixed_precision_bitwidth, apply_power_of_2_workflow  # noqa: E402
from aimet_torch.v2 import quantsim  # noqa: E402

from aimet_torch.quantizable_batchnorm import QuantizedQuantizableBatchNorm2d  # noqa: F401, E402
from common.torch_stft import STFT  # noqa: E402
import quantized_stft  # noqa: F401, E402  — register QuantizedSTFT
from quantized_stft import force_native_trans_float  # noqa: E402
from common.mrnn_clz_encoding import (  # noqa: E402
    apply_mrnn_clz_encoding_fixes,
    apply_mrnn_clz_encoding_fixes_post_calib,
    collect_power2_float_out_fmax,
)
from quick_start import (  # noqa: E402
    BATCH_SIZE,
    BITWIDTH_CONFIG_FILE,
    CLN,
    CONFIG_FILE,
    DEFAULT_BW,
    DEVICE,
    FP_LR,
    FP_MODEL_PATH,
    HypotFun,
    MRNN,
    NUM_CLASSES,
    PERCENTILE_VALUE,
    PowerCompress,
    QUANT_SCHEME,
    build_dataloaders,
    fresh_calib_loader,
    set_seed,
    setup_audio_backend,
    train_floating_point,
)
from aimet_torch.quantizable_batchnorm import QuantizableBatchNorm2d  # noqa: E402
from aimet_torch.utils_rx import freeze_quantizer_parameters  # noqa: E402

_HERE = Path(__file__).resolve().parent
ACCEPTANCE_BITWIDTH_CONFIG = _HERE / "config" / "mrnn_acceptance_mixed_precision.json"
DEFAULT_MAX_CALIB_BATCHES = 100

# SYS-OPEN-Q-1 W6 (2026-06-09 commit 78fffef): default calib percentile
# overrides the upstream educational default (99.99 in `quick_start.py`).
#
# W6 sweep on `quick_start_full_quant.json` 8bit baseline showed:
# - 99.5 unblocks SYS-OPEN-Q-1 part A: fc0 SQNR 5.34 -> 13.02 dB
#   (+7.7 dB), cos 0.873 -> 0.979; freq_downs.2.conv2d cos 0.776 ->
#   0.918; neck_seqs.1.conv_t cos 0.844 -> 0.911.
# - 99.0 partially regresses some layers (e.g. enc_seqs.0.conv_t cos
#   0.915 -> 0.882). 99.5 is the local optimum across the swept set.
# - tf / tf_enhanced behave like 99.99 baseline (non-outlier paths
#   dominate). part B (freq_downs.0/1, neck_seqs.0, enc_seqs.0/1)
#   is unaffected by calib and tracked separately.
#
# CLI `--percentile-value` still accepts any float; this just changes
# the default users get when they omit the flag, so SYS-OPEN-Q-1 part A
# is mitigated by default in the INT16 metric script.
SYSQ1_W6_PERCENTILE_VALUE = 99.5
_QDQ_MODE_VALUES = frozenset({
    ExecutionMode.FP32_QDQ.value,
    ExecutionMode.FP16_QDQ.value,
    ExecutionMode.FIXED_SCALE_QDQ.value,
})


def _sanitize_modes_for_int16_sim(modes: list[str]) -> list[str]:
    """ensure 后的 sim 不能评 QDQ；显式传入时跳过并提示正确脚本。"""
    skipped = [m for m in modes if m in _QDQ_MODE_VALUES]
    if skipped:
        print(
            "\nWARNING: 本脚本 sim 经 ensure_output_quantizers_for_int16_eval 构建，"
            f"以下 QDQ 模式无效，已跳过: {skipped}\n"
            "         QDQ 三档请用: int16_whole_graph_vs_float_native.py\n"
        )
    kept = [m for m in modes if m not in _QDQ_MODE_VALUES]
    if not kept:
        kept = [ExecutionMode.INT16_FIXED_EVAL.value]
        print("WARNING: 无有效评估模式，回退为 int16_fixed_eval\n")
    return kept


def evaluate_limited(
    model,
    loader,
    device,
    max_batches: int | None = None,
    *,
    micro_batch_size: int | None = None,
    label: str | None = None,
    log_every: int = 10,
    out_logits: list[torch.Tensor] | None = None,
) -> float:
    """Top-1 精度；``max_batches`` 限制 batch 数以加速 smoke。

    传入 ``out_logits=[]`` 时，会把每个 chunk 的 logits（detach + fp32 + cpu）
    追加进列表，调用方可在所有模式跑完后统一计算 cosine / max_abs / mean_abs，
    支持「3 模式 × PTQ × GPU」接入侧的相似度审计。
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            labels = labels.to(device)
            if micro_batch_size is None or inputs.size(0) <= micro_batch_size:
                chunks = [(inputs.to(device), labels)]
            else:
                chunks = [
                    (
                        inputs[i : i + micro_batch_size].to(device),
                        labels[i : i + micro_batch_size],
                    )
                    for i in range(0, inputs.size(0), micro_batch_size)
                ]
            for chunk_x, chunk_y in chunks:
                out = model(chunk_x)
                if hasattr(out, "to_float"):
                    out = out.to_float()
                if out_logits is not None:
                    out_logits.append(out.detach().float().cpu())
                preds = out.max(1).indices
                total += chunk_y.size(0)
                correct += preds.eq(chunk_y).sum().item()
            batch_no = batch_idx + 1
            if log_every > 0 and batch_no % log_every == 0:
                prefix = f"{label}: " if label else ""
                limit = max_batches if max_batches is not None else "?"
                print(
                    f"  [{prefix}eval] batch {batch_no}/{limit}, samples={total}",
                    flush=True,
                )
    if total == 0:
        raise RuntimeError("evaluate_limited: no samples evaluated")
    return correct / total


def _encoding_recalib_post_qat(
    sim,
    loaders,
    device: torch.device,
    *,
    max_calib_batches: int,
    sign_input_bypass: bool,
    sign_input_scale: float | None,
    power2_fmax: dict | None,
) -> int:
    """Re-calibrate encodings on updated QAT weights; refresh CLZ fix + (M,r)."""

    import aimet_torch.v2 as aimet
    from aimet_torch.m_po2_quantization import clear_sim_fixed_scale_caches

    calib_loader = fresh_calib_loader(loaders["calib"])
    clear_sim_fixed_scale_caches(sim.model)
    sim.model.eval()
    t0 = time.time()
    with torch.no_grad(), aimet.nn.compute_encodings(sim.model):
        for idx, (x, _) in enumerate(calib_loader):
            if idx >= max_calib_batches:
                break
            sim.model(x.to(device))
    if power2_fmax is not None:
        apply_mrnn_clz_encoding_fixes_post_calib(
            sim.model,
            sign_input_scale=None if sign_input_bypass else sign_input_scale,
            sign_output_unit=not sign_input_bypass,
            power2_float_out_fmax=power2_fmax,
            verbose=False,
        )
    n_fixed = convert_encodings_to_fixed_scale(sim)
    print(
        f"QAT 后 encoding re-calib 完成（calib={max_calib_batches} batch, "
        f"convert={n_fixed}），耗时 {time.time() - t0:.1f}s"
    )
    return n_fixed


def _calib_fn(sim_model, loader, device, max_batches: int):
    calib_loader = fresh_calib_loader(loader)

    def _run(m):
        with torch.no_grad():
            for idx, (x, _) in enumerate(calib_loader):
                if idx >= max_batches:
                    break
                m(x.to(device))

    return _run


def _eval_modes(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    modes: list[str],
    *,
    max_batches: int | None,
    micro_batch_size: int | None = None,
    log_every: int = 10,
    out_logits_per_mode: dict[str, torch.Tensor] | None = None,
) -> dict[str, float | str]:
    """Evaluate ``modes`` on ``loader``; return label -> accuracy or error string.

    ``out_logits_per_mode`` 若不为 ``None``，每个成功模式会写入拼接后的 logits
    （CPU fp32），用于多模式相似度审计（cosine / max_abs / mean_abs）。
    """

    results: dict[str, float | str] = {}
    for mode_str in modes:
        mode = ExecutionMode(mode_str)
        label = mode.value
        release_cuda_memory(model)
        print(f"\n--- 评估 {label} ---", flush=True)
        t0 = time.time()
        debug_ctx = (
            int16_eval_allow_debug_float()
            if mode is ExecutionMode.INT16_FIXED_EVAL
            else contextlib.nullcontext()
        )
        try:
            collected: list[torch.Tensor] | None = (
                [] if out_logits_per_mode is not None else None
            )
            with debug_ctx, quant_execution_mode(mode):
                results[label] = evaluate_limited(
                    model,
                    loader,
                    device,
                    max_batches=max_batches,
                    micro_batch_size=micro_batch_size,
                    label=label,
                    log_every=log_every,
                    out_logits=collected,
                )
            if out_logits_per_mode is not None and collected:
                out_logits_per_mode[label] = torch.cat(collected, dim=0)
            print(f"{label}: 完成，耗时 {time.time() - t0:.1f}s", flush=True)
        except Exception as exc:
            results[label] = f"FAILED: {exc}"
            print(f"{label}: FAILED — {exc}", flush=True)
    return results


def _print_metric_table(
    title: str,
    results: dict[str, float | str],
    *,
    baseline_label: str = "fp32_qdq",
    pp_threshold: float = 3.0,
) -> None:
    """打印 mode→Top1 表，并按 baseline 给出 ΔTop1 / <pp_threshold pp 阈值标记。

    考核口径（设计 §10.1 / 用户验收）：与 fp32_qdq 误差 < 3 pp 视为通过。
    """
    print(f"\n{title}")
    fp_ref = results.get(baseline_label)
    has_baseline = isinstance(fp_ref, float)
    # baseline 行永远排首位，便于读者快速锚定参照系。
    ordered_labels = (
        [baseline_label] + [k for k in results.keys() if k != baseline_label]
        if baseline_label in results
        else list(results.keys())
    )
    for label in ordered_labels:
        acc = results[label]
        if isinstance(acc, float):
            line = f"  {label:22s} {acc * 100:7.2f}%"
            if has_baseline:
                if label == baseline_label:
                    line += "   (baseline)"
                else:
                    delta_pp = (acc - fp_ref) * 100
                    marker = "✅" if abs(delta_pp) < pp_threshold else "⚠️"
                    line += f"   Δ(vs {baseline_label}) = {delta_pp:+.2f} pp {marker}"
            print(line)
        else:
            print(f"  {label:22s} {acc}")


def _print_similarity_table(
    title: str,
    logits_per_mode: dict[str, torch.Tensor],
    *,
    baseline_label: str = "fp32_qdq",
    cosine_threshold: float = 0.9995,
) -> None:
    """打印 ``baseline_label`` 与其他模式的 logits 相似度（cosine / max_abs / rel_max / mean_abs）。

    考核口径（设计 §10.1 参考量级）：fp16_qdq cosine ≈ 0.9999；fixed_scale_qdq cosine ≈ 0.99985+。
    这里取保守 ``cosine ≥ 0.9995`` 作为接入侧合规阈值，证明「3 模式数值差异极小」。
    """

    print(f"\n{title}")
    fp_ref = logits_per_mode.get(baseline_label)
    if fp_ref is None:
        print(f"  (no baseline logits for '{baseline_label}'; skip similarity table)")
        return

    header = f"  {'mode':22s}  {'cosine':>9s}  {'max_abs':>10s}  {'rel_max':>10s}  {'mean_abs':>10s}"
    print(header)
    fp_ref_flat = fp_ref.flatten()
    fp_ref_norm = max(float(fp_ref.abs().max().item()), 1e-6)

    for label, logits in logits_per_mode.items():
        if label == baseline_label:
            continue
        if logits.shape != fp_ref.shape:
            print(f"  {label:22s}  shape mismatch {tuple(logits.shape)} vs {tuple(fp_ref.shape)}")
            continue
        diff = (logits - fp_ref).abs()
        max_abs = float(diff.max().item())
        mean_abs = float(diff.mean().item())
        rel_max = max_abs / fp_ref_norm
        cos = torch.nn.functional.cosine_similarity(
            fp_ref_flat.unsqueeze(0), logits.flatten().unsqueeze(0)
        ).item()
        marker = "✅" if cos >= cosine_threshold else "⚠️"
        print(
            f"  {label:22s}  {cos:9.6f}  {max_abs:10.3e}  {rel_max:10.3e}  {mean_abs:10.3e}  {marker}"
        )


def _print_qat_delta(
    pre: dict[str, float | str],
    post: dict[str, float | str],
    *,
    pp_threshold: float = 3.0,
) -> None:
    """打印 QAT 前后每模式 ΔTop1，并对训练后每个非 fp32 模式与 fp32 的差距做 <3 pp 阈值标记。"""

    print("\n--- QAT 前后 metric 对比 ---")
    for label in pre:
        a, b = pre.get(label), post.get(label)
        if isinstance(a, float) and isinstance(b, float):
            print(f"  {label:22s} {a * 100:.2f}% → {b * 100:.2f}%  "
                  f"({(b - a) * 100:+.2f} pp)")

    fp_post = post.get(ExecutionMode.FP32_QDQ.value)
    if isinstance(fp_post, float):
        for label, acc in post.items():
            if label == ExecutionMode.FP32_QDQ.value or not isinstance(acc, float):
                continue
            delta_pp = (acc - fp_post) * 100
            marker = "✅" if abs(delta_pp) < pp_threshold else "⚠️"
            print(f"  Δ({label} − fp32_qdq) 训练后: {delta_pp:+.2f} pp {marker}")


_DECOMPOSED_STATELESS_PARENT_PREFIXES: tuple[str, ...] = (
    "power_compress_1",
    "power_compress_2",
    "hypot_fun",
    "pre_bn",
    # CLN 实例命名按需扩展（MRNN 当前未直接顶层挂 CLN，预留前缀模式）
    "cln",
)


@contextlib.contextmanager
def temporarily_disable_all_quantizers(sim_model: torch.nn.Module):
    """暂时将 sim.model 的全部 quantizer slot 置 None，退出 with 时原样恢复。

    用于 per-node cosine 诊断：在同一份 graph、同一份 sample input 上对比
    「quantized forward」与「float forward」每个子节点输出。
    """
    saved: list[tuple] = []
    for _name, module in sim_model.named_modules():
        if not type(module).__name__.startswith("Quantized"):
            continue
        for attr in ("input_quantizers", "output_quantizers", "param_quantizers"):
            qcontainer = getattr(module, attr, None)
            if qcontainer is None:
                continue
            if isinstance(qcontainer, torch.nn.ModuleDict):
                keys = list(qcontainer.keys())
            else:
                keys = list(range(len(qcontainer)))
            for key in keys:
                val = qcontainer[key]
                if val is None:
                    continue
                saved.append((qcontainer, key, val))
                qcontainer[key] = None
    try:
        yield
    finally:
        for container, key, val in saved:
            container[key] = val


def _to_cpu_float(t):
    """to_float() + detach + float + cpu，兼容 QuantizedTensor / Tensor。"""
    if hasattr(t, "to_float"):
        t = t.to_float()
    return t.detach().float().cpu()


def _cosine(a_cpu: torch.Tensor, b_cpu: torch.Tensor) -> tuple[float, float, float]:
    a = a_cpu.flatten().to(torch.float64)
    b = b_cpu.flatten().to(torch.float64)
    if a.numel() != b.numel() or a.numel() == 0:
        return float("nan"), float("nan"), float("nan")
    cos = float(
        torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    )
    max_abs = float((a - b).abs().max().item())
    peak = float(b.abs().max().item())
    rel = max_abs / max(peak, 1e-12)
    return cos, max_abs, rel


def report_per_node_cosine(
    sim_model: torch.nn.Module,
    scope_prefix: str,
    sample_input: torch.Tensor,
    device: str | torch.device,
) -> list[tuple]:
    """对 ``scope_prefix.*`` 下每个 Quantized* 子模块输出 cosine 诊断。

    返回 ``[(name, cls, cos_cum, cos_local, max_abs_local, dyn_range), ...]``。

    - ``cos_cum``：「全图 quantizer 全开 vs 全关」该节点输出 cosine，反映 **整图累积损伤**
      （包含上游 QDQ 误差累积）。
    - ``cos_local``：用 float 路径的该节点输入重放一次 quantized forward，得到 ``op_q(in_f)``，
      与 ``op_f(in_f)`` 对比；反映 **该步 QDQ 单独的损伤**。
    """
    targets: list[tuple[str, torch.nn.Module]] = []
    for name, mod in sim_model.named_modules():
        if not name.startswith(scope_prefix + "."):
            continue
        if not type(mod).__name__.startswith("Quantized"):
            continue
        targets.append((name, mod))
    if not targets:
        return []

    sim_model.eval()
    x = sample_input.to(device)

    def _make_out_hook(nm, store):
        def hook(_m, _inp, output):
            store[nm] = _to_cpu_float(output)
        return hook

    def _make_io_hook(nm, in_store, kw_store, out_store):
        def hook(_m, inputs, kwargs, output):
            in_store[nm] = tuple(_to_cpu_float(t) if torch.is_tensor(t) else t for t in inputs)
            kw_store[nm] = {
                k: (_to_cpu_float(v) if torch.is_tensor(v) else v) for k, v in kwargs.items()
            }
            out_store[nm] = _to_cpu_float(output)
        return hook

    q_out: dict[str, torch.Tensor] = {}
    handles = [m.register_forward_hook(_make_out_hook(n, q_out)) for n, m in targets]
    with torch.no_grad():
        sim_model(x)
    for h in handles:
        h.remove()

    f_in: dict[str, tuple] = {}
    f_kw: dict[str, dict] = {}
    f_out: dict[str, torch.Tensor] = {}
    handles = [
        m.register_forward_hook(_make_io_hook(n, f_in, f_kw, f_out), with_kwargs=True)
        for n, m in targets
    ]
    with torch.no_grad(), temporarily_disable_all_quantizers(sim_model):
        sim_model(x)
    for h in handles:
        h.remove()

    local_out: dict[str, torch.Tensor | str] = {}
    with torch.no_grad():
        for name, mod in targets:
            args = f_in.get(name)
            if args is None:
                continue
            try:
                dev_args = tuple(
                    a.to(device) if torch.is_tensor(a) else a for a in args
                )
                dev_kwargs = {
                    k: (v.to(device) if torch.is_tensor(v) else v)
                    for k, v in f_kw.get(name, {}).items()
                }
                out = mod(*dev_args, **dev_kwargs)
                local_out[name] = _to_cpu_float(out)
            except Exception as exc:  # noqa: BLE001
                local_out[name] = f"ERR: {type(exc).__name__}: {exc}"

    rows: list[tuple] = []
    for name, mod in targets:
        cls = type(mod).__name__
        qt = q_out.get(name)
        ft = f_out.get(name)
        lt = local_out.get(name)
        cos_cum, _, _ = _cosine(qt, ft) if (qt is not None and ft is not None) else (float("nan"),) * 3
        if isinstance(lt, torch.Tensor) and ft is not None:
            cos_local, max_abs_local, rel_local = _cosine(lt, ft)
        else:
            cos_local, max_abs_local, rel_local = float("nan"), float("nan"), float("nan")
        dyn = float(ft.max().item() - ft.min().item()) if ft is not None and ft.numel() else float("nan")
        rows.append((name, cls, cos_cum, cos_local, max_abs_local, rel_local, dyn))
    rows.sort(key=lambda r: (r[3] if r[3] == r[3] else 1.0))  # cos_local asc, NaN last
    return rows


def _print_per_node_cosine(scope: str, rows: list[tuple]) -> None:
    if not rows:
        print(f"[per-node-cosine] scope={scope}: 未找到 Quantized* 子节点")
        return
    print(
        f"\n[per-node-cosine] scope={scope}  "
        f"(cos_local 升序；cos_cum=整图累积，cos_local=单步 QDQ 真实损伤)"
    )
    header = (
        f"  {'name':50s} {'class':28s} "
        f"{'cos_cum':>9s} {'cos_local':>10s} {'max_abs_l':>10s} {'rel_l':>9s} {'float_range':>11s}"
    )
    print(header)
    for name, cls, cos_cum, cos_loc, mabs, rel, dyn in rows:
        def _fmt(v, w, p=6):
            return f"{'n/a':>{w}s}" if v != v else f"{v:>{w}.{p}f}"
        flag = ""
        if cos_loc == cos_loc:
            flag = "" if cos_loc >= 0.999 else (" ⚠️" if cos_loc >= 0.95 else " ❌")
        mabs_s = "n/a" if mabs != mabs else f"{mabs:.3e}"
        rel_s = "n/a" if rel != rel else f"{rel:.3e}"
        dyn_s = "n/a" if dyn != dyn else f"{dyn:.3e}"
        print(
            f"  {name:50s} {cls:28s} "
            f"{_fmt(cos_cum, 9)} {_fmt(cos_loc, 10)} "
            f"{mabs_s:>10s} {rel_s:>9s} {dyn_s:>11s}{flag}"
        )


def _clear_quantizer_container(qcontainer) -> int:
    """把 ModuleList / ModuleDict 中的 quantizer 槽位全部置 None，返回清除数量。"""
    n = 0
    if qcontainer is None:
        return 0
    if isinstance(qcontainer, torch.nn.ModuleDict):
        keys = list(qcontainer.keys())
    else:
        keys = list(range(len(qcontainer)))
    for key in keys:
        if qcontainer[key] is not None:
            qcontainer[key] = None
            n += 1
    return n


def disable_all_quantizers(sim_model: torch.nn.Module) -> tuple[int, int]:
    """诊断用：把 sim.model 所有 Quantized* 节点的全部 input/output/param quantizer 置 None。

    用于回答「fp32_qdq 损失是否完全来自 quantizer」这个问题。如果此函数生效后 cosine 仍 < 0.99，
    则说明 prepare_model 的图改写已改变数值（更严重的问题）。
    """
    n_mod, n_slot = 0, 0
    for _name, module in sim_model.named_modules():
        if not type(module).__name__.startswith("Quantized"):
            continue
        cleared_here = 0
        for attr in ("input_quantizers", "output_quantizers", "param_quantizers"):
            cleared_here += _clear_quantizer_container(getattr(module, attr, None))
        if cleared_here:
            n_slot += cleared_here
            n_mod += 1
    return n_mod, n_slot


def disable_quantizers_by_attrs(
    sim_model: torch.nn.Module,
    attrs: tuple[str, ...],
) -> tuple[int, int]:
    """诊断用：按 quantizer 容器类型批量置 None。

    ``attrs=("input_quantizers", "output_quantizers")`` 可隔离「只有权重量化」；
    ``attrs=("param_quantizers",)`` 可隔离「只有激活量化」。
    """
    n_mod, n_slot = 0, 0
    for _name, module in sim_model.named_modules():
        if not type(module).__name__.startswith("Quantized"):
            continue
        cleared_here = 0
        for attr in attrs:
            cleared_here += _clear_quantizer_container(getattr(module, attr, None))
        if cleared_here:
            n_slot += cleared_here
            n_mod += 1
    return n_mod, n_slot


_FRONTEND_CLAMP_NAMES = frozenset({
    "module_clamp",
    "module_floordiv_2",
    "module_clamp_2",
    "module_clamp_3",
    "module_clamp_4",
})

_ACTIVATION_ONLY_SCOPE_CHOICES = (
    "all",
    "frontend",
    "trans",
    "pre_bn",
    "power_compress_1",
    "hypot_fun",
    "fft2band",
    "power_compress_2",
    "frontend_clamps",
    "conv_down",
    "quant_gru",
    "rnn2d_no_gru",
    "fc_head",
)


def _activation_scope_match(name: str, scope: str) -> bool:
    if scope == "all":
        return True
    if scope == "frontend":
        return (
            name.startswith(("trans.", "pre_bn.", "power_compress_", "hypot_fun.", "fft2band."))
            or name in _FRONTEND_CLAMP_NAMES
        )
    if scope == "trans":
        return name.startswith("trans.")
    if scope == "pre_bn":
        return name.startswith("pre_bn.")
    if scope == "power_compress_1":
        return name.startswith("power_compress_1.")
    if scope == "power_compress_2":
        return name.startswith("power_compress_2.")
    if scope == "hypot_fun":
        return name.startswith("hypot_fun.")
    if scope == "fft2band":
        return name.startswith("fft2band.")
    if scope == "frontend_clamps":
        return name in _FRONTEND_CLAMP_NAMES
    if scope == "conv_down":
        return name.startswith(("conv_in", "freq_downs."))
    if scope == "quant_gru":
        return ".seq_t" in name
    if scope == "rnn2d_no_gru":
        return name.startswith(("enc_seqs.", "neck_seqs.")) and ".seq_t" not in name
    if scope == "fc_head":
        return name.startswith("fc0") or name == "module_mean_4"
    raise ValueError(f"Unknown activation quantizer scope: {scope}")


def keep_only_activation_quantizers_in_scope(
    sim_model: torch.nn.Module,
    scope: str,
) -> tuple[int, int, int]:
    """诊断用：关闭全部 param quantizer，并只保留指定 scope 的 activation quantizer。

    返回 ``(param_slots_disabled, activation_slots_disabled, activation_slots_kept)``。
    """
    param_disabled = 0
    act_disabled = 0
    act_kept = 0
    for name, module in sim_model.named_modules():
        if not type(module).__name__.startswith("Quantized"):
            continue
        param_disabled += _clear_quantizer_container(getattr(module, "param_quantizers", None))
        keep = _activation_scope_match(name, scope)
        for attr in ("input_quantizers", "output_quantizers"):
            qcontainer = getattr(module, attr, None)
            if qcontainer is None:
                continue
            if isinstance(qcontainer, torch.nn.ModuleDict):
                keys = list(qcontainer.keys())
            else:
                keys = list(range(len(qcontainer)))
            for key in keys:
                if qcontainer[key] is None:
                    continue
                if keep:
                    act_kept += 1
                else:
                    qcontainer[key] = None
                    act_disabled += 1
    return param_disabled, act_disabled, act_kept


def disable_decomposed_stateless_quantizers(sim_model: torch.nn.Module) -> tuple[int, int]:
    """Disable input/output quantizer 在「无参数的 functional Quantized* 节点」上。

    背景：``model_preparer.prepare_model`` 即使传 ``stateless_modules_to_preserve`` 也只会
    保留 module 命名空间，**仍会**把 forward 里 sqrt / sign / abs / div / clamp / sub / mul /
    square / mean / floordiv / matmul / reshape / pad 等 functional 拆成可量化的 child
    Module（v2 quantsim 包成 ``QuantizedAdd / QuantizedSubtract / QuantizedMultiply /
    QuantizedDivide / QuantizedSqrt / QuantizedClamp / QuantizedSquare / QuantizedMean /
    QuantizedAbs / QuantizedFloorDivide / QuantizedElementwiseUnarySign / QuantizedReshape /
    QuantizedPad / QuantizedMatMul / ...``）。MRNN 经过这种拆解后，每步 QDQ 舍入误差被
    非线性运算放大、跨层累积，整图 fp32_qdq cosine 会从 ≥0.999 跌到 ~0.7。

    判别规则：``len(param_quantizers) >= 1`` 视为「带参数的真量化 op」（Conv/Linear/GRU/
    BatchNorm/...），**保留**所有 quantizer；否则视为「纯 functional 拆解节点」，把
    input / output quantizer 全部置 ``None`` 让它走 float 数值路径。

    返回：``(modules_touched, slots_disabled)``——被处理 module 数 + 被置 None 的 slot 总数。
    """
    modules_touched = 0
    slots_disabled = 0
    for _name, module in sim_model.named_modules():
        cls_name = type(module).__name__
        if not cls_name.startswith("Quantized"):
            continue
        param_q = getattr(module, "param_quantizers", None)
        if param_q is not None and any(v is not None for v in param_q.values()):
            continue
        cleared_here = 0
        for attr in ("input_quantizers", "output_quantizers"):
            cleared_here += _clear_quantizer_container(getattr(module, attr, None))
        if cleared_here:
            slots_disabled += cleared_here
            modules_touched += 1
    return modules_touched, slots_disabled


def build_sim(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    *,
    bitwidth_config: Path | str = ACCEPTANCE_BITWIDTH_CONFIG,
    quant_scheme: str = QUANT_SCHEME,
    percentile_value: float = PERCENTILE_VALUE,
    native_trans: bool = True,
    disable_decomposed_functional: bool = False,
    disable_all: bool = False,
    disable_activation: bool = False,
    disable_param: bool = False,
    activation_only_scope: str | None = None,
    clz_encoding_fix: bool = True,
    sign_input_bypass: bool = False,
):
    # 验收路径：全图 quantizer 保持开启；前端 STFT/PowerCompress/Hypot 通过
    # mrnn_acceptance_mixed_precision.json 提 activation 至 16bit（设计 §D.2）。
    # 仅当显式传入 disable_* / activation_only_scope 时才进入诊断分支。
    prepare_kwargs: dict = {
        "stateless_modules_to_preserve": [PowerCompress, HypotFun, CLN, QuantizableBatchNorm2d],
    }
    if native_trans:
        prepare_kwargs["module_classes_to_exclude"] = [STFT]
    prepared = model_preparer.prepare_model(copy.deepcopy(model), **prepare_kwargs)
    prepared_float = copy.deepcopy(prepared)
    sim = quantsim.QuantizationSimModel(
        prepared,
        dummy_input=dummy_input,
        quant_scheme=quant_scheme,
        config_file=str(CONFIG_FILE),
        default_output_bw=DEFAULT_BW,
        default_param_bw=DEFAULT_BW,
    )
    sim.set_percentile_value(percentile_value)
    # 先补齐缺失 quantizer slot，再应用混合精度；顺序反了会把前端 16bit 覆盖回 8bit。
    patched = ensure_output_quantizers_for_int16_eval(sim)
    print(f"ensure_output_quantizers_for_int16_eval: patched {len(patched)} slots")
    apply_mixed_precision_bitwidth(
        sim.model, config_file=str(bitwidth_config), verbose=False,
    )
    if activation_only_scope is not None:
        p_dis, a_dis, a_keep = keep_only_activation_quantizers_in_scope(
            sim.model, activation_only_scope
        )
        print(
            f"[DIAG] activation_only_scope={activation_only_scope}: "
            f"disabled {p_dis} param slots, disabled {a_dis} activation slots, "
            f"kept {a_keep} activation slots"
        )
    elif disable_all:
        n_mod, n_slot = disable_all_quantizers(sim.model)
        print(
            f"[DIAG] disable_all_quantizers: {n_slot} quantizer slots on {n_mod} Quantized* "
            f"modules set to None — sim.model 退化为 prepare 后的纯 float forward"
        )
    elif disable_activation or disable_param:
        if disable_activation:
            n_mod, n_slot = disable_quantizers_by_attrs(
                sim.model, ("input_quantizers", "output_quantizers")
            )
            print(
                f"[DIAG] disable_activation_quantizers: {n_slot} input/output slots "
                f"on {n_mod} Quantized* modules set to None"
            )
        if disable_param:
            n_mod, n_slot = disable_quantizers_by_attrs(sim.model, ("param_quantizers",))
            print(
                f"[DIAG] disable_param_quantizers: {n_slot} param slots "
                f"on {n_mod} Quantized* modules set to None"
            )
    elif disable_decomposed_functional:
        n_mod, n_slot = disable_decomposed_stateless_quantizers(sim.model)
        print(
            f"[DIAG] disable_decomposed_functional: {n_slot} input/output quantizer slots "
            f"on {n_mod} param-less Quantized* modules set to None"
        )
    if native_trans:
        n_cleared = force_native_trans_float(sim.model)
        print(
            f"[DIAG] native_trans: STFT leaf fp32 黑盒，已关闭 trans 上 {n_cleared} 个 quantizer slot"
        )
    if clz_encoding_fix:
        apply_mrnn_clz_encoding_fixes(
            sim.model, sign_input_bypass=sign_input_bypass, verbose=False,
        )
    return sim, (prepared_float if clz_encoding_fix else None)


def _default_int16_batch_size() -> int:
    """Conservative default for full-graph MRNN on common 32GB GPUs."""

    if torch.cuda.is_available():
        _, total = torch.cuda.mem_get_info()
        if total <= 24 * 1024**3:
            return 16
        if total <= 48 * 1024**3:
            return 32
    return BATCH_SIZE


def _build_loaders(data_root: Path, batch_size: int):
    import quick_start as qs

    prev = qs.BATCH_SIZE
    qs.BATCH_SIZE = batch_size
    try:
        return build_dataloaders(str(data_root))
    finally:
        qs.BATCH_SIZE = prev


def _patch_torchaudio_with_soundfile() -> None:
    """torchaudio 2.10+ 默认走 torchcodec；SpeechCommands 用 soundfile 读 wav 即可。"""
    import soundfile as sf
    import torch
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
    parser = argparse.ArgumentParser(description="MRNN INT16 metric comparison")
    parser.add_argument(
        "--data-root",
        default=os.environ.get(
            "SPEECH_COMMANDS_ROOT",
            "/home/llq/workspace/data/speech_commands",
        ),
        help="SpeechCommands v0.02 根目录",
    )
    parser.add_argument("--fp-epochs", type=int, default=0, help="浮点预训练 epoch 数（0=跳过）")
    parser.add_argument(
        "--max-calib-batches",
        type=int,
        default=DEFAULT_MAX_CALIB_BATCHES,
        help=f"PTQ 校准 batch 数（默认 {DEFAULT_MAX_CALIB_BATCHES}，与 quick_start 对齐）",
    )
    parser.add_argument("--max-eval-batches", type=int, default=None, help="限制 test batch 数；None=全量")
    parser.add_argument(
        "--bitwidth-config",
        type=str,
        default=str(ACCEPTANCE_BITWIDTH_CONFIG),
        help="混合精度 JSON；默认 mrnn_acceptance_mixed_precision.json（前端 activation 16bit）",
    )
    parser.add_argument(
        "--quant-scheme",
        choices=("min_max", "tf", "percentile", "tf_enhanced"),
        default=QUANT_SCHEME,
        help="AIMET PTQ 校准 scheme（QuantGRU 内部方法自动推断）",
    )
    parser.add_argument(
        "--percentile-value",
        type=float,
        default=SYSQ1_W6_PERCENTILE_VALUE,
        help=(
            "percentile scheme 分位点（仅 quant_scheme=percentile 时生效）；"
            f"默认 {SYSQ1_W6_PERCENTILE_VALUE}（SYS-OPEN-Q-1 W6 推荐值，"
            f"vs upstream 通用 99.99）"
        ),
    )
    parser.add_argument(
        "--apply-po2",
        action="store_true",
        help="legacy：全图 float scale 圆整为 1/2^n（M=1）；与 --apply-m-po2 互斥",
    )
    parser.add_argument(
        "--apply-m-po2",
        action="store_true",
        help="硬件 M/2^n grid：snap scale 为 M/2^r（M 可>1）→ re-calib → re-snap；CLZ fix 在其后",
    )
    parser.add_argument(
        "--clz-encoding-fix",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="§2.3 reciprocal/power_2 CLZ encoding fix（默认开）",
    )
    parser.add_argument(
        "--sign-input-bypass",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="module_sign input Q bypass；默认关闭以匹配硬件整数 sign",
    )
    parser.add_argument(
        "--sign-input-scale",
        type=float,
        default=2e-9,
        help="硬件整数 sign 路径的 input Q fine scale；仅 --no-sign-input-bypass 时应用",
    )
    parser.add_argument(
        "--skip-float-native",
        action="store_true",
        help="跳过原生浮点 baseline 评估（默认开启：在 sim 之前 deepcopy 一份 MRNN，作为接入侧合规性的真实基准）",
    )
    parser.add_argument(
        "--fp-ckpt",
        type=str,
        default=str(FP_MODEL_PATH),
        help="浮点 MRNN ckpt 路径；若文件存在则启动时自动 load_state_dict（避免重训）。默认指向 quick_start.FP_MODEL_PATH",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="忽略已有 --fp-ckpt 强制重新训练浮点模型（默认：ckpt 存在则跳过浮点训练）",
    )
    parser.add_argument(
        "--disable-decomposed-functional",
        action="store_true",
        help="[诊断] disable 无 param 的 Quantized* functional 子节点 quantizer；验收路径勿开",
    )
    parser.add_argument(
        "--keep-decomposed-quantizers",
        action="store_true",
        help="[已废弃别名] 等价于不开 --disable-decomposed-functional；保留兼容",
    )
    parser.add_argument(
        "--disable-all-quantizers",
        action="store_true",
        help=(
            "诊断用：把 sim.model 所有 Quantized* 节点的 input/output/param quantizer 全部置 None。"
            "若评估 cosine 仍 < 0.99 → prepare_model 图改写本身已改变数值；若 cosine ≈ 1.0 → "
            "全部损失来自 quantizer 总和。**只用于诊断**，会让 sim 退化为 prepare 后的纯 float forward。"
        ),
    )
    parser.add_argument(
        "--disable-activation-quantizers",
        action="store_true",
        help=(
            "诊断用：把所有 Quantized* 节点的 input/output quantizer 置 None，仅保留 param quantizer，"
            "用于隔离「权重量化单独造成多少损失」。"
        ),
    )
    parser.add_argument(
        "--disable-param-quantizers",
        action="store_true",
        help=(
            "诊断用：把所有 Quantized* 节点的 param quantizer 置 None，仅保留 input/output quantizer，"
            "用于隔离「激活量化单独造成多少损失」。"
        ),
    )
    parser.add_argument(
        "--native-trans",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "STFT(trans) 作为板端独立硬化模块保留 fp32 黑盒（默认开："
            "module_classes_to_exclude=[STFT]，关闭 trans quantizer，不参与 "
            "INT16_FIXED_EVAL 定点化）。仅 decomposed STFT 软件对照实验时用 "
            "--no-native-trans。"
        ),
    )
    parser.add_argument(
        "--activation-only-scope",
        choices=_ACTIVATION_ONLY_SCOPE_CHOICES,
        default=None,
        help=(
            "诊断用：关闭所有 param quantizer，并只保留指定模块组的 input/output quantizer。"
            "frontend 子 scope：trans / pre_bn / power_compress_1 / hypot_fun / fft2band / "
            "power_compress_2 / frontend_clamps。与 --native-trans 联用时 trans scope 无 quantizer。"
        ),
    )
    parser.add_argument(
        "--per-node-cosine",
        nargs="+",
        default=None,
        choices=(
            "power_compress_1",
            "power_compress_2",
            "hypot_fun",
            "pre_bn",
            "fft2band",
            "trans",
        ),
        help=(
            "诊断用：calib 后在同一 batch 上对指定 scope 子节点逐个对比"
            "「quantized vs float」cosine，定位链路中最敏感的那一步。"
        ),
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=[ExecutionMode.INT16_FIXED_EVAL.value],
        choices=[m.value for m in ExecutionMode],
        help=(
            "要评估的 ExecutionMode 列表（默认仅 int16_fixed_eval）。"
            "QDQ 三档会被自动跳过——请用 int16_whole_graph_vs_float_native.py"
        ),
    )
    parser.add_argument(
        "--qat-epochs",
        type=int,
        default=0,
        help=">0 时在 INT16_FIXED_QAT_SIM 下按 train loader 训练 N 个 epoch",
    )
    parser.add_argument(
        "--qat-batches-per-epoch",
        type=int,
        default=None,
        help="每个 QAT epoch 最多训练多少个 batch；None=用完整 train",
    )
    parser.add_argument(
        "--qat-train-steps",
        type=int,
        default=0,
        help=">0 时跑固定步数 SGD 冒烟（与 --qat-epochs 互斥时优先 epochs）",
    )
    parser.add_argument(
        "--qat-lr",
        type=float,
        default=1e-4,
        help="INT16 QAT 学习率",
    )
    parser.add_argument(
        "--qat-optimizer",
        choices=("sgd", "adam"),
        default="adam",
        help="QAT 优化器（epoch 模式；steps 模式仍用 SGD）",
    )
    parser.add_argument(
        "--qat-lr-scheduler",
        action="store_true",
        help="epoch QAT 启用 CosineAnnealingLR",
    )
    parser.add_argument(
        "--qat-val-batches",
        type=int,
        default=50,
        help="QAT val checkpoint batch 数；0=跳过 val；-1=全量 val",
    )
    parser.add_argument(
        "--qat-val-mode",
        choices=(
            ExecutionMode.FP32_QDQ.value,
            ExecutionMode.INT16_FIXED_EVAL.value,
        ),
        default=ExecutionMode.INT16_FIXED_EVAL.value,
        help="QAT val 选模指标（本 sim 上 fp32_qdq val 不可靠，默认 int16_fixed_eval）",
    )
    parser.add_argument(
        "--qat-post-encoding-recalib",
        action="store_true",
        help="QAT 后对 calib 重跑 compute_encodings + CLZ post_calib + convert (M,r)",
    )
    parser.add_argument(
        "--no-qat-restore-best",
        action="store_true",
        help="QAT 不恢复 val 最优 checkpoint",
    )
    parser.add_argument(
        "--qat-mode",
        choices=(
            ExecutionMode.FP32_QDQ.value,
            ExecutionMode.FIXED_SCALE_QDQ.value,
            ExecutionMode.INT16_FIXED_QAT_SIM.value,
        ),
        default=ExecutionMode.FP32_QDQ.value,
        help=(
            "QAT forward mode；默认 fp32_qdq 走 GPU 友好快路径，"
            "需要严格 INT16 约束时显式设 int16_fixed_qat_sim"
        ),
    )
    parser.add_argument(
        "--skip-qat-post-eval",
        action="store_true",
        help="QAT 后不自动跑 fp32_qdq / int16_fixed_eval 前后对比",
    )
    parser.add_argument(
        "--skip-qat-pre-eval",
        action="store_true",
        help="跳过 QAT 前 baseline，只在 QAT 后评估（更快开始训练）",
    )
    parser.add_argument(
        "--qat-train-scope",
        choices=("weights", "head", "all"),
        default="weights",
        help="QAT 可训练参数：weights=全网权重(不含quantizer)；head=仅fc；all=同weights",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=f"DataLoader batch size（默认按 GPU 容量选 16/32/64，当前 heuristic={_default_int16_batch_size()}）",
    )
    parser.add_argument(
        "--eval-micro-batch-size",
        type=int,
        default=None,
        help="评估时按 micro-batch 切分 forward，降低 INT16 峰值显存",
    )
    parser.add_argument(
        "--eval-log-every",
        type=int,
        default=10,
        help="评估阶段每 N 个 batch 打印一次进度（0=关闭）",
    )
    parser.add_argument(
        "--eval-cache-batches",
        type=int,
        default=None,
        help="缓存前 N 个 test batch 用于评估；默认在 --max-eval-batches 设置时缓存相同数量，None 全量时关闭",
    )
    parser.add_argument(
        "--eval-cache-on-cpu",
        action="store_true",
        help="评估 batch 只缓存到 CPU；默认缓存到 GPU，减少 DataLoader/I/O 等待",
    )
    parser.add_argument(
        "--qat-batch-size",
        type=int,
        default=None,
        help="QAT 每步实际 batch（<= train loader batch）；默认按空闲显存自动推断",
    )
    parser.add_argument(
        "--activation-recompute",
        action="store_true",
        help="显式开启 QuantizedLinear 激活重算（MRNN QAT 默认关闭以避免 checkpoint 重算状态不一致）",
    )
    parser.add_argument(
        "--qat-empty-cache-every",
        type=int,
        default=0,
        help="每 N 步调用 empty_cache（0=不调用；频繁 empty_cache 会拖慢并导致 GPU 利用率显示为 0）",
    )
    parser.add_argument(
        "--qat-log-timing",
        action="store_true",
        help="打印每步 QAT forward+backward 耗时",
    )
    parser.add_argument(
        "--qat-log-every",
        type=int,
        default=10,
        help="每 N 步打印一次 QAT loss（0=不打印；打印会同步 GPU）",
    )
    parser.add_argument(
        "--qat-cache-batches",
        type=int,
        default=8,
        help="预先缓存 N 个 QAT batch，减少音频 I/O/DataLoader 等待（0=关闭）",
    )
    parser.add_argument(
        "--qat-cache-on-cpu",
        action="store_true",
        help="QAT batch 只缓存到 CPU 内存；默认直接缓存到 GPU",
    )
    parser.add_argument(
        "--qat-check-finite-every",
        type=int,
        default=0,
        help="每 N 步同步检查 loss/权重 finite（0=关闭；检查会同步 GPU）",
    )
    parser.add_argument(
        "--qat-grad-clip",
        type=float,
        default=1.0,
        help="QAT 梯度裁剪 max_norm；0 表示不裁剪",
    )
    args = parser.parse_args()

    args.batch_size = args.batch_size if args.batch_size is not None else _default_int16_batch_size()
    if args.qat_batch_size is None and (args.qat_epochs > 0 or args.qat_train_steps > 0):
        args.qat_batch_size = suggest_int16_qat_batch_size(args.batch_size)
    if args.qat_batch_size is not None and args.qat_batch_size > args.batch_size:
        sys.exit(
            f"--qat-batch-size ({args.qat_batch_size}) 不能大于 --batch-size ({args.batch_size})"
        )

    args.modes = _sanitize_modes_for_int16_sim(list(args.modes))

    set_seed()
    _patch_torchaudio_with_soundfile()
    setup_audio_backend()

    data_root = Path(args.data_root)
    if not data_root.is_dir():
        sys.exit(f"Data root not found: {data_root}")

    print("=" * 70)
    print("MRNN SpeechCommands — INT16 metric 对比")
    print("=" * 70)
    print(f"设备:       {DEVICE}")
    print(f"数据根目录: {data_root}")
    print(f"Batch size: {args.batch_size}")
    if args.apply_po2 and args.apply_m_po2:
        parser.error("--apply-po2 与 --apply-m-po2 互斥")

    qat_execution_mode = ExecutionMode(args.qat_mode)

    if args.qat_epochs > 0 or args.qat_train_steps > 0:
        print(f"QAT batch:  {args.qat_batch_size or args.batch_size}")
        print(f"QAT mode:   {qat_execution_mode.value}")
        print(
            "QAT 显存优化: "
            + ("Linear 重算 开" if args.activation_recompute else "Linear 重算 关")
        )
        print(
            f"QAT 同步控制: log_every={args.qat_log_every}, "
            f"check_finite_every={args.qat_check_finite_every}, "
            f"cache_batches={args.qat_cache_batches}"
        )
    print(f"校准 batch: {args.max_calib_batches}")
    print(f"Quant scheme: {args.quant_scheme}  percentile={args.percentile_value}")
    print(f"Bitwidth cfg: {args.bitwidth_config}")
    print(f"评估 batch: {args.max_eval_batches or '全量 test'}")
    print(f"模式:       {args.modes}")
    scale_note = "默认 (M_int16,rshift)；QuantGRU 内部 2^(-shift)"
    if args.apply_m_po2:
        scale_note = "M_Po2 snap + re-calib（M/2^n，M 可>1）"
    elif args.apply_po2:
        scale_note = "legacy 全图 apply_power_of_2_workflow（1/2^n）"
    print(f"Scale 策略: {scale_note}")
    stft_note = "STFT 原生 leaf（硬件路径）" if args.native_trans else "STFT 拆分为 QuantizedConv1d"
    print(f"说明:       {stft_note}；其余 BN/PowerCompress/Hypot/CLN/QuantGRU 仍 decomposed")
    if args.native_trans and args.activation_only_scope == "trans":
        print("警告:       --native-trans 下 trans 无 Quantized* 子节点，--activation-only-scope trans 将保留 0 个 slot")

    loaders = _build_loaders(data_root, args.batch_size)
    eval_loader = loaders["test"]
    eval_max_batches = args.max_eval_batches
    eval_cache_batches = (
        args.max_eval_batches if args.eval_cache_batches is None else args.eval_cache_batches
    )
    if eval_cache_batches is not None and eval_cache_batches > 0:
        cache_t0 = time.time()
        cached_eval = cache_training_batches(
            loaders["test"],
            DEVICE,
            max_batches=eval_cache_batches,
            cache_on_gpu=not args.eval_cache_on_cpu,
        )
        if cached_eval:
            eval_loader = cached_eval
            eval_max_batches = (
                min(args.max_eval_batches, len(cached_eval))
                if args.max_eval_batches is not None
                else len(cached_eval)
            )
            cache_place = "CPU" if args.eval_cache_on_cpu else "GPU"
            print(
                f"Eval batch cache: {len(cached_eval)} batches on {cache_place}, "
                f"耗时 {time.time() - cache_t0:.1f}s",
                flush=True,
            )
    model = MRNN(output_dim=NUM_CLASSES).to(DEVICE)

    fp_ckpt_path = Path(args.fp_ckpt)
    ckpt_loaded = False
    if fp_ckpt_path.is_file() and not args.force_retrain:
        sd = torch.load(fp_ckpt_path, map_location=DEVICE)
        model.load_state_dict(sd, strict=False)
        print(f"已加载浮点 ckpt: {fp_ckpt_path}（继续训练或评估均以此为起点；如需从零重训请加 --force-retrain）")
        ckpt_loaded = True

    if args.fp_epochs > 0:
        prefix = "继续训练" if ckpt_loaded else "从零训练"
        t0 = time.time()
        fp_acc = train_floating_point(
            model,
            loaders["train"],
            loaders["test"],
            DEVICE,
            epochs=args.fp_epochs,
            lr=FP_LR,
            save_path=fp_ckpt_path,
            val_loader=loaders["val"],
        )
        print(f"浮点{prefix} ({args.fp_epochs} epoch) 精度: {fp_acc * 100:.2f}%  耗时 {time.time() - t0:.1f}s")
    elif ckpt_loaded:
        print("跳过浮点训练（使用已加载 ckpt）")
    else:
        print("跳过浮点训练（随机初始化；绝对精度偏低，但 mode 间 Δ 仍有参考价值）")

    sample_input, _ = next(iter(loaders["train"]))
    dummy_input = sample_input.to(DEVICE)

    # 在 build_sim / prepare_model 之前 deepcopy 保存「原生浮点」MRNN，作为
    # 接入侧合规性审计的真实 baseline——它没有任何 QDQ / quantizer 包装，
    # 所有 3 模式（fp32_qdq / fp16_qdq / fixed_scale_qdq）相对它的差就是
    # 「scale 表示 + dtype 转换」引入的总误差。
    if not args.skip_float_native:
        float_native_model = copy.deepcopy(model).to(DEVICE).eval()
    else:
        float_native_model = None

    sim, prepared_float = build_sim(
        model,
        dummy_input,
        bitwidth_config=Path(args.bitwidth_config),
        quant_scheme=args.quant_scheme,
        percentile_value=args.percentile_value,
        native_trans=args.native_trans,
        disable_decomposed_functional=args.disable_decomposed_functional,
        disable_all=args.disable_all_quantizers,
        disable_activation=args.disable_activation_quantizers,
        disable_param=args.disable_param_quantizers,
        activation_only_scope=args.activation_only_scope,
        clz_encoding_fix=args.clz_encoding_fix,
        sign_input_bypass=args.sign_input_bypass,
    )
    if args.native_trans:
        trans_cls = type(getattr(sim.model, "trans", None)).__name__
        n_trans_q = sum(
            1
            for name, m in sim.model.named_modules()
            if name.startswith("trans.") and type(m).__name__.startswith("Quantized")
        )
        print(
            f"[DIAG] native_trans: sim.model.trans={trans_cls}, "
            f"trans 子树 Quantized* 数 = {n_trans_q}（期望 0）"
        )
    sim.model.to(DEVICE).eval()

    calib = _calib_fn(sim.model, loaders["calib"], DEVICE, args.max_calib_batches)
    t0 = time.time()
    with torch.no_grad(), aimet.nn.compute_encodings(sim.model):
        calib(sim.model)
    print(f"compute_encodings 完成，耗时 {time.time() - t0:.1f}s")

    power2_fmax = None
    if args.clz_encoding_fix and prepared_float is not None:
        pf = prepared_float.to(DEVICE).eval()
        power2_fmax = collect_power2_float_out_fmax(
            pf, loaders["calib"], DEVICE, args.max_calib_batches,
        )

    if args.apply_m_po2:
        t_mpo2 = time.time()
        calib_loader = fresh_calib_loader(loaders["calib"])
        mpo2_stats = apply_m_po2_recalib_workflow(
            sim.model,
            calib_loader,
            DEVICE,
            max_calib_batches=args.max_calib_batches,
            verbose=False,
        )
        print(
            f"M_Po2 + re-calib 完成（pre {mpo2_stats['pre_snap']['modified']}/"
            f"{mpo2_stats['pre_snap']['total']} → recalib {mpo2_stats['recalib_batches']} batch → "
            f"post {mpo2_stats['post_snap']['modified']}/{mpo2_stats['post_snap']['total']}），"
            f"耗时 {time.time() - t_mpo2:.1f}s"
        )
    elif args.apply_po2:
        apply_power_of_2_workflow(
            sim.model,
            method="round",
            tolerance=0.02,
            align_bias_scale=True,
            verbose=False,
        )
        print("apply_power_of_2_workflow 完成（全图 float scale → 2^n）")
    else:
        print("跳过 Po2/M_Po2：QuantGRU 用 internal shift；分解层边界用 (M_int16, rshift)")

    if args.clz_encoding_fix and prepared_float is not None and power2_fmax is not None:
        t_clz = time.time()
        clz_stats = apply_mrnn_clz_encoding_fixes_post_calib(
            sim.model,
            sign_input_scale=None if args.sign_input_bypass else args.sign_input_scale,
            sign_output_unit=not args.sign_input_bypass,
            power2_float_out_fmax=power2_fmax,
            verbose=False,
        )
        sim._clz_encoding_fix_stats = clz_stats  # noqa: SLF001
        rec = clz_stats["reciprocal_denom"]["touched"]
        sq = clz_stats["power2_output"]["touched"]
        sign = clz_stats.get("sign_input", {}).get("touched", 0)
        sign_out = clz_stats.get("sign_output", {}).get("touched", 0)
        clamp_pos = clz_stats.get("positive_clamp_output", {}).get("touched", 0)
        pc2_pos = clz_stats.get("pc2_positive", {}).get("touched", 0)
        print(
            f"CLZ encoding fix 完成（sign_input={sign}, sign_output={sign_out}, "
            f"clamp_pos={clamp_pos}, pc2_positive={pc2_pos}, "
            f"reciprocal={rec}, power_2={sq}），耗时 {time.time() - t_clz:.1f}s"
        )

    n_fixed = convert_encodings_to_fixed_scale(sim)
    print(f"convert_encodings_to_fixed_scale: {n_fixed} 个 affine quantizer 已缓存 (M,r)")

    report = diagnose_int16_readiness(sim)
    for key, items in report.items():
        if items:
            print(f"diagnose_int16_readiness[{key}]: {len(items)} 项，示例 {items[:3]}")
    if not any(report.values()):
        print("diagnose_int16_readiness: 全部通过 ✅")

    if args.per_node_cosine:
        print("\n=== Per-node cosine 诊断（同一 batch、同一图，quantizer 全关 vs 全开） ===")
        for scope in args.per_node_cosine:
            rows = report_per_node_cosine(sim.model, scope, sample_input, DEVICE)
            _print_per_node_cosine(scope, rows)

    if (
        ExecutionMode.INT16_FIXED_QAT_SIM.value in args.modes
        or qat_execution_mode is ExecutionMode.INT16_FIXED_QAT_SIM
    ):
        print(
            "\n--- INT16 QAT LUT 预热（CPU 在线拟合 PWL/CLZ，首次较慢；"
            "结果缓存到各 module，避免每步重复）---"
        )
        warm_s = warm_int16_qat_lut_cache(sim.model, sample_input, DEVICE)
        print(f"LUT 预热完成，耗时 {warm_s:.1f}s")

    if (
        ExecutionMode.INT16_FIXED_QAT_SIM.value in args.modes
        or qat_execution_mode is ExecutionMode.INT16_FIXED_QAT_SIM
    ):
        print("\n--- INT16_FIXED_QAT_SIM backward 冒烟 ---")
        sim.model.train()
        x_qat = sample_input.to(DEVICE)
        try:
            with quant_execution_mode(ExecutionMode.INT16_FIXED_QAT_SIM):
                y_qat = sim.model(x_qat)
            loss_tensor = y_qat.to_float() if hasattr(y_qat, "to_float") else y_qat
            loss_tensor.sum().backward()
            n_gru_grad = sum(
                1
                for m in sim.model.modules()
                if m.__class__.__name__ == "QuantizedQuantGRU"
                and getattr(m, "weight_ih_l0", None) is not None
                and m.weight_ih_l0.grad is not None
            )
            print(f"QAT backward OK（QuantGRU 有梯度模块数: {n_gru_grad}）")
        except Exception as exc:
            print(f"INT16_FIXED_QAT_SIM backward 失败: {exc}")
        finally:
            release_cuda_memory(sim.model)
        sim.model.eval()

    run_qat = args.qat_epochs > 0 or args.qat_train_steps > 0
    qat_compare_modes = [ExecutionMode.INT16_FIXED_EVAL.value]
    metrics_pre_qat: dict[str, float | str] | None = None
    metrics_post_qat: dict[str, float | str] | None = None
    qat_failed = False

    if run_qat and not args.skip_qat_pre_eval and not args.skip_qat_post_eval:
        print("\n--- QAT 前 baseline（test）---")
        sim.model.eval()
        metrics_pre_qat = _eval_modes(
            sim.model,
            eval_loader,
            DEVICE,
            qat_compare_modes,
            max_batches=eval_max_batches,
            micro_batch_size=args.eval_micro_batch_size,
            log_every=args.eval_log_every,
        )
        _print_metric_table("  baseline", metrics_pre_qat)

    if args.qat_epochs > 0:
        scope: QatTrainScope = args.qat_train_scope  # type: ignore[assignment]
        trainable = select_qat_trainable_parameters(sim.model, scope=scope)
        print(
            f"\n--- {qat_execution_mode.value} 训练 ({args.qat_epochs} epoch(s), "
            f"scope={scope}, opt={args.qat_optimizer}, lr={args.qat_lr}, "
            f"params={len(trainable)}) ---"
        )
        if not trainable:
            print("无可训练参数，跳过 QAT")
        else:
            freeze_quantizer_parameters(sim.model, verbose=False, freeze_bn_affine=True)
            clip = args.qat_grad_clip if args.qat_grad_clip > 0 else None
            val_fn = None
            val_loader = None
            if args.qat_val_batches != 0 and "val" in loaders:
                val_max = None if args.qat_val_batches < 0 else args.qat_val_batches
                val_loader = loaders["val"]
                val_mode = ExecutionMode(args.qat_val_mode)

                def _qat_val_fn(m: torch.nn.Module) -> float:
                    release_cuda_memory(m)
                    m.eval()
                    if val_mode is ExecutionMode.INT16_FIXED_EVAL:
                        with quant_execution_mode(val_mode):
                            with int16_eval_allow_debug_float():
                                return evaluate_limited(
                                    m,
                                    val_loader,
                                    DEVICE,
                                    max_batches=val_max,
                                    micro_batch_size=args.eval_micro_batch_size,
                                    label="qat_val",
                                    log_every=0,
                                )
                    with quant_execution_mode(val_mode):
                        return evaluate_limited(
                            m,
                            val_loader,
                            DEVICE,
                            max_batches=val_max,
                            micro_batch_size=args.eval_micro_batch_size,
                            label="qat_val",
                            log_every=0,
                        )

                val_fn = _qat_val_fn
                val_batches_label = "全量" if val_max is None else str(val_max)
                print(
                    f"  QAT val checkpoint: {val_mode.value}, batches={val_batches_label}, "
                    f"restore_best={not args.no_qat_restore_best}",
                    flush=True,
                )
            try:
                history = run_int16_qat_epochs(
                    sim.model,
                    loaders["train"],
                    DEVICE,
                    epochs=args.qat_epochs,
                    lr=args.qat_lr,
                    scope=scope,
                    optimizer=args.qat_optimizer,  # type: ignore[arg-type]
                    lr_scheduler=args.qat_lr_scheduler,
                    grad_clip_norm=clip,
                    max_batches_per_epoch=args.qat_batches_per_epoch,
                    batch_size=args.qat_batch_size,
                    execution_mode=qat_execution_mode,
                    activation_recompute=args.activation_recompute,
                    empty_cache_every=args.qat_empty_cache_every,
                    log_timing=args.qat_log_timing,
                    log_every=args.qat_log_every,
                    check_finite_every=args.qat_check_finite_every,
                    val_loader=val_loader,
                    val_fn=val_fn,
                    restore_best=not args.no_qat_restore_best,
                )
                for row in history:
                    msg = (
                        f"  epoch {int(row['epoch'])}: "
                        f"batches={int(row['batches'])} "
                        f"loss_mean={row['loss_mean']:.4f} "
                        f"loss_last={row['loss_last']:.4f}"
                    )
                    if "val_top1" in row:
                        msg += f" val={float(row['val_top1']) * 100:.2f}%"
                    print(msg)
                if history and "val_restored" in history[-1]:
                    print(
                        f"  val restored: {float(history[-1]['val_restored']) * 100:.2f}%"
                    )
            except Exception as exc:
                qat_failed = True
                print(f"INT16 QAT epoch 训练失败: {exc}")
        if not qat_failed and args.qat_post_encoding_recalib:
            _encoding_recalib_post_qat(
                sim,
                loaders,
                DEVICE,
                max_calib_batches=args.max_calib_batches,
                sign_input_bypass=args.sign_input_bypass,
                sign_input_scale=args.sign_input_scale,
                power2_fmax=power2_fmax,
            )
        release_cuda_memory(sim.model)
        sim.model.eval()

    elif args.qat_train_steps > 0:
        scope = args.qat_train_scope  # type: ignore[assignment]
        trainable = select_qat_trainable_parameters(sim.model, scope=scope)
        print(
            f"\n--- {qat_execution_mode.value} QAT ({args.qat_train_steps} steps, "
            f"scope={scope}, lr={args.qat_lr}, params={len(trainable)}) ---",
            flush=True,
        )
        if qat_execution_mode is ExecutionMode.INT16_FIXED_QAT_SIM:
            print(
                "  说明: INT16 QAT 大量算子在 CPU Python dispatch；QuantGRU/surrogate 才会占用 GPU。"
                " GPU 利用率间歇为 0 通常正常。",
                flush=True,
            )
        else:
            print(
                "  说明: 当前使用 QAT 快路径；训练阶段走 QDQ/float CUDA，INT16 语义留给评估验收。",
                flush=True,
            )
        if not trainable:
            print("无可训练参数，跳过")
        else:
            freeze_quantizer_parameters(sim.model, verbose=False, freeze_bn_affine=True)
            clip = args.qat_grad_clip if args.qat_grad_clip > 0 else None
            qat_data_iter = loaders["train"]
            if args.qat_cache_batches > 0:
                cache_count = min(args.qat_cache_batches, args.qat_train_steps)
                cache_t0 = time.time()
                cached_batches = cache_training_batches(
                    loaders["train"],
                    DEVICE,
                    max_batches=cache_count,
                    batch_size=args.qat_batch_size,
                    cache_on_gpu=not args.qat_cache_on_cpu,
                )
                if cached_batches:
                    qat_data_iter = cached_batches
                    cache_place = "CPU" if args.qat_cache_on_cpu else "GPU"
                    print(
                        f"  QAT batch cache: {len(cached_batches)} batches on {cache_place}, "
                        f"耗时 {time.time() - cache_t0:.1f}s",
                        flush=True,
                    )
            try:
                losses = run_int16_qat_steps(
                    sim.model,
                    qat_data_iter,
                    DEVICE,
                    steps=args.qat_train_steps,
                    lr=args.qat_lr,
                    scope=scope,
                    grad_clip_norm=clip,
                    batch_size=args.qat_batch_size,
                    execution_mode=qat_execution_mode,
                    activation_recompute=args.activation_recompute,
                    empty_cache_every=args.qat_empty_cache_every,
                    log_timing=args.qat_log_timing,
                    log_every=args.qat_log_every,
                    check_finite_every=args.qat_check_finite_every,
                )
                for idx, loss_val in enumerate(losses, start=1):
                    print(f"  step {idx}/{args.qat_train_steps} ce_loss={loss_val:.4f}")
                print(f"QAT 完成，末步 loss={losses[-1]:.4f}")
            except Exception as exc:
                qat_failed = True
                print(f"INT16_FIXED_QAT_SIM QAT 失败: {exc}")
                if torch.cuda.is_available():
                    free_mb = torch.cuda.mem_get_info()[0] / (1024 ** 2)
                    print(
                        f"  提示: 当前 GPU 空闲约 {free_mb:.0f} MiB；"
                        "可试 --batch-size 32 --qat-batch-size 16 "
                        "或 export PYTORCH_ALLOC_CONF=expandable_segments:True"
                    )
        if not qat_failed and args.qat_post_encoding_recalib:
            _encoding_recalib_post_qat(
                sim,
                loaders,
                DEVICE,
                max_calib_batches=args.max_calib_batches,
                sign_input_bypass=args.sign_input_bypass,
                sign_input_scale=args.sign_input_scale,
                power2_fmax=power2_fmax,
            )
        release_cuda_memory(sim.model)
        sim.model.eval()

    if qat_failed:
        print("\nQAT 未完成，已清理 CUDA 状态并跳过后续 metric；请降低 QAT batch 或启用 allocator 防碎片后重跑。")
        return

    if run_qat and not args.skip_qat_post_eval:
        print("\n--- QAT 后 metric（test）---")
        metrics_post_qat = _eval_modes(
            sim.model,
            eval_loader,
            DEVICE,
            qat_compare_modes,
            max_batches=eval_max_batches,
            micro_batch_size=args.eval_micro_batch_size,
            log_every=args.eval_log_every,
        )
        _print_metric_table("  post-QAT", metrics_post_qat)
        if metrics_pre_qat is not None:
            _print_qat_delta(metrics_pre_qat, metrics_post_qat)
        if args.per_node_cosine:
            print("\n=== Per-node cosine 诊断（QAT 后，同一 batch） ===")
            for scope in args.per_node_cosine:
                rows = report_per_node_cosine(sim.model, scope, sample_input, DEVICE)
                _print_per_node_cosine(f"{scope}/post_qat", rows)

    eval_modes = list(dict.fromkeys(args.modes))
    if run_qat and not args.skip_qat_post_eval:
        for m in qat_compare_modes:
            if m not in eval_modes:
                eval_modes.append(m)

    results: dict[str, float | str] = {}
    logits_per_mode: dict[str, torch.Tensor] = {}
    for mode_str in eval_modes:
        label = ExecutionMode(mode_str).value
        if (
            metrics_post_qat is not None
            and label in metrics_post_qat
            and label in (ExecutionMode.INT16_FIXED_EVAL.value, ExecutionMode.INT16_FIXED_QAT_SIM.value)
        ):
            results[label] = metrics_post_qat[label]
            print(f"\n--- 评估 {label}（复用 QAT 后结果）---")
            acc = metrics_post_qat[label]
            if isinstance(acc, float):
                print(f"{label}: {acc * 100:.2f}%")
            else:
                print(f"{label}: {acc}")
            continue
        print(f"\n--- 评估 {label} ---")
        release_cuda_memory(sim.model)
        mode = ExecutionMode(mode_str)
        debug_ctx = (
            int16_eval_allow_debug_float()
            if mode is ExecutionMode.INT16_FIXED_EVAL
            else contextlib.nullcontext()
        )
        try:
            collected: list[torch.Tensor] = []
            with debug_ctx, quant_execution_mode(mode):
                acc = evaluate_limited(
                    sim.model,
                    eval_loader,
                    DEVICE,
                    max_batches=eval_max_batches,
                    micro_batch_size=args.eval_micro_batch_size,
                    label=label,
                    log_every=args.eval_log_every,
                    out_logits=collected,
                )
            results[label] = acc
            if collected:
                logits_per_mode[label] = torch.cat(collected, dim=0)
            print(f"{label}: {acc * 100:.2f}%")
        except Exception as exc:
            results[label] = f"FAILED: {exc}"
            print(f"{label}: FAILED — {exc}")

    # 原生浮点 baseline：未做 sim/QDQ 包装的纯 MRNN forward，作为接入侧合规性
    # 的真实基准。它给出「量化引入的总误差」上界，而不是 fp32_qdq 内部一致性。
    if float_native_model is not None:
        print("\n--- 评估 float_native（原生浮点；无 sim/QDQ）---")
        try:
            float_collected: list[torch.Tensor] = []
            t0 = time.time()
            float_acc = evaluate_limited(
                float_native_model,
                eval_loader,
                DEVICE,
                max_batches=eval_max_batches,
                micro_batch_size=args.eval_micro_batch_size,
                label="float_native",
                log_every=args.eval_log_every,
                out_logits=float_collected,
            )
            results["float_native"] = float_acc
            if float_collected:
                logits_per_mode["float_native"] = torch.cat(float_collected, dim=0)
            print(f"float_native: {float_acc * 100:.2f}%  耗时 {time.time() - t0:.1f}s")
        except Exception as exc:
            results["float_native"] = f"FAILED: {exc}"
            print(f"float_native: FAILED — {exc}")

    if "float_native" in results:
        baseline = "float_native"
    elif ExecutionMode.FP32_QDQ.value in results and isinstance(
        results.get(ExecutionMode.FP32_QDQ.value), float
    ):
        baseline = ExecutionMode.FP32_QDQ.value
    else:
        baseline = next(
            (k for k, v in results.items() if isinstance(v, float)),
            ExecutionMode.INT16_FIXED_EVAL.value,
        )

    print("\n" + "=" * 70)
    print(f"Metric 对比汇总（验收：每模式 vs {baseline} 误差 < 3 pp）")
    print("=" * 70)
    _print_metric_table("", results, baseline_label=baseline)

    if logits_per_mode and baseline in logits_per_mode:
        # 接入侧合规性：3 模式 × PTQ × GPU 的数值一致性证据。
        # baseline=float_native 时给出「量化总误差」（含 QDQ 舍入 + scale 离散化 + dtype 转换）；
        # baseline=fp32_qdq 时仅给出 fp16/fixed_scale 相对 fp32_qdq 的额外接入误差。
        # 设计 §10.1 参考量级：3 模式 vs fp32_qdq cosine ≈ 0.9999；
        # 3 模式 vs raw float 量级取决于 W/A bitwidth + 校准方法（典型 0.999~0.9999）。
        threshold = 0.999 if baseline == "float_native" else 0.9995
        _print_similarity_table(
            f"相似度对比（vs {baseline}；接入侧合规阈值 cosine ≥ {threshold}）",
            logits_per_mode,
            baseline_label=baseline,
            cosine_threshold=threshold,
        )

    int16 = results.get(ExecutionMode.INT16_FIXED_EVAL.value)
    float_ref = results.get("float_native")
    if isinstance(float_ref, float) and isinstance(int16, float):
        delta_pp = (int16 - float_ref) * 100
        print(f"\n  Δ(INT16_FIXED_EVAL − float_native) = {delta_pp:+.2f} pp")
        if abs(delta_pp) <= 3.0:
            print("  ✅ |Δ| ≤ 3 pp（验收参考阈值）")
        else:
            print("  ⚠️  |Δ| > 3 pp")
    print("=" * 70)


if __name__ == "__main__":
    main()

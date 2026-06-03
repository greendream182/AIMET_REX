"""MRNN 分解图与 §2.3 CLZ 规格化类的 encoding 修复（非 bypass 主路径）。

- sign: bypass input Q（§3.1，非 LUT head）
- module_div 分母 b=std: reciprocal 正域 encoding（unsigned + min≥eps）
- module_square: power_2 输出 range 重标定 out_min/out_max
"""
from __future__ import annotations

import re
from typing import Any, Iterator

import torch
import torch.nn as nn

SIGN_NAME_RE = re.compile(r"(^|\.)module_sign(?:_\d+)?$")
DIV_NAME_RE = re.compile(r"(^|\.)module_div_\d+$")
SQUARE_NAME_RE = re.compile(r"(^|\.)module_square(?:_\d+)?$")

SQUARE_OP_TYPES = frozenset({"QuantizedSquare", "QuantizedPow"})
DIV_OP_TYPES = frozenset({"QuantizedDivide"})


def _iter_named_quant_ops(
    model: nn.Module,
    name_re: re.Pattern[str],
    op_types: frozenset[str],
) -> Iterator[tuple[str, nn.Module]]:
    for name, mod in model.named_modules():
        if not name_re.search(name):
            continue
        if type(mod).__name__ not in op_types:
            continue
        yield name, mod


def disable_input_quantizers_by_pattern(
    model: nn.Module,
    name_re: re.Pattern[str] = SIGN_NAME_RE,
    *,
    input_indices: tuple[int, ...] | None = None,
    verbose: bool = False,
) -> int:
    count = 0
    for name, mod in model.named_modules():
        if not name_re.search(name):
            continue
        iqs = getattr(mod, "input_quantizers", None)
        if not iqs:
            continue
        idxs = range(len(iqs)) if input_indices is None else input_indices
        for idx in idxs:
            if idx >= len(iqs) or iqs[idx] is None:
                continue
            if verbose:
                print(f"  input Q bypass: {name}.input[{idx}]")
            iqs[idx] = None
            count += 1
    return count


def disable_output_quantizers_by_pattern(
    model: nn.Module,
    name_re: re.Pattern[str] = SQUARE_NAME_RE,
    *,
    verbose: bool = False,
) -> int:
    count = 0
    for name, mod in model.named_modules():
        if not name_re.search(name):
            continue
        oqs = getattr(mod, "output_quantizers", None)
        if not oqs:
            continue
        for idx in range(len(oqs)):
            if oqs[idx] is None:
                continue
            if verbose:
                print(f"  output Q bypass: {name}.output[{idx}]")
            oqs[idx] = None
            count += 1
    return count


def _q_initialized(q) -> bool:
    return q is not None and getattr(q, "is_initialized", lambda: False)()


def _float_max(q) -> float | None:
    if not _q_initialized(q):
        return None
    try:
        return float(q.get_max().reshape(-1)[0].item())
    except (RuntimeError, AttributeError, ValueError):
        return None


def _float_min(q) -> float | None:
    if not _q_initialized(q):
        return None
    try:
        return float(q.get_min().reshape(-1)[0].item())
    except (RuntimeError, AttributeError, ValueError):
        return None


def _q_device_dtype(q):
    if hasattr(q, "scale") and q.scale is not None:
        return q.scale.device, q.scale.dtype
    if hasattr(q, "min") and q.min is not None:
        return q.min.device, q.min.dtype
    return torch.device("cpu"), torch.float32


def collect_power2_float_out_fmax(
    float_model: nn.Module,
    loader,
    device: torch.device,
    max_batches: int,
) -> dict[str, float]:
    """校准数据上观测 float 图 power_2 输出 fmax（用于 out_max / CLZ head）。"""
    out: dict[str, float] = {}
    handles: list = []

    def _make_hook(name: str):
        def _hook(_mod, _inp, output):
            if not isinstance(output, torch.Tensor):
                return
            v = float(output.abs().max().item())
            out[name] = max(out.get(name, 0.0), v)

        return _hook

    skip_types = frozenset({"CLN", "ModuleList", "Sequential", "MRNN", "RNN2D"})
    for name, mod in float_model.named_modules():
        if not SQUARE_NAME_RE.search(name):
            continue
        if type(mod).__name__ in skip_types:
            continue
        handles.append(mod.register_forward_hook(_make_hook(name)))

    was_training = float_model.training
    float_model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if idx >= max_batches:
                break
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            float_model(x.to(device))
    float_model.train(was_training)

    for h in handles:
        h.remove()
    return out


def fix_reciprocal_denom_encodings(
    model: nn.Module,
    *,
    eps: float = 1e-8,
    verbose: bool = False,
) -> dict[str, Any]:
    """分母 b=std 按 reciprocal 正域重标定：unsigned + dequant_min ≥ eps。"""
    stats = {"touched": 0, "skipped": []}
    for name, mod in _iter_named_quant_ops(model, DIV_NAME_RE, DIV_OP_TYPES):
        iqs = getattr(mod, "input_quantizers", None)
        if not iqs or len(iqs) < 2 or iqs[1] is None:
            stats["skipped"].append((name, "no denom quantizer"))
            continue
        q = iqs[1]
        if not _q_initialized(q):
            stats["skipped"].append((name, "denom not initialized"))
            continue

        bw = int(getattr(q, "bitwidth", 8))
        cur_max = _float_max(q)
        if cur_max is None or cur_max <= eps:
            stats["skipped"].append((name, "bad cur_max"))
            continue

        q.symmetric = False
        q.qmin = 0
        q.qmax = 2**bw - 1
        dev, dt = _q_device_dtype(q)
        q.set_range(
            torch.tensor(eps, device=dev, dtype=dt),
            torch.tensor(cur_max, device=dev, dtype=dt),
        )
        stats["touched"] += 1
        if verbose:
            print(f"  reciprocal denom: {name}.input[1] unsigned [{eps}, {cur_max}]")
    return stats


def fix_power2_output_encodings(
    model: nn.Module,
    *,
    margin: float = 1.01,
    min_bitwidth: int = 16,
    float_out_fmax: dict[str, float] | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """square 输出按 power_2 动态范围重标定 out_max（优先 16-bit unsigned）。"""
    stats: dict[str, Any] = {"touched": 0, "skipped": [], "out_max": {}}
    fmax_map = float_out_fmax or {}
    for name, mod in _iter_named_quant_ops(model, SQUARE_NAME_RE, SQUARE_OP_TYPES):
        oqs = getattr(mod, "output_quantizers", None)
        if not oqs or oqs[0] is None:
            stats["skipped"].append((name, "no output quantizer"))
            continue
        oq = oqs[0]
        if not _q_initialized(oq):
            stats["skipped"].append((name, "output not initialized"))
            continue

        peak = 0.0
        iqs = getattr(mod, "input_quantizers", None)
        if iqs and iqs[0] is not None and _q_initialized(iqs[0]):
            in_min = _float_min(iqs[0])
            in_max = _float_max(iqs[0])
            if in_min is not None and in_max is not None:
                peak = max(abs(in_min), abs(in_max))

        cur_out_max = _float_max(oq) or 0.0
        fmax_obs = fmax_map.get(name, 0.0)
        if peak > 0:
            out_max = max(cur_out_max, peak * peak * margin, fmax_obs * margin)
        else:
            out_max = max(cur_out_max * margin, fmax_obs * margin, 1e-8)

        bw = max(int(getattr(oq, "bitwidth", 8)), min_bitwidth)
        oq.bitwidth = bw
        oq.symmetric = False
        oq.qmin = 0
        oq.qmax = 2**bw - 1
        dev, dt = _q_device_dtype(oq)
        oq.set_range(
            torch.tensor(0.0, device=dev, dtype=dt),
            torch.tensor(out_max, device=dev, dtype=dt),
        )
        stats["touched"] += 1
        stats["out_max"][name] = out_max
        if verbose:
            print(
                f"  power_2 output: {name} bw={bw} out_max={out_max:.4f} "
                f"peak_in={peak:.4f} float_fmax={fmax_obs:.4f}"
            )
    return stats


def apply_mrnn_clz_encoding_fixes(
    model: nn.Module,
    *,
    sign_input_bypass: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """apply_mixed_precision 之后、compute_encodings 之前：sign input bypass。"""
    out: dict[str, Any] = {}
    if sign_input_bypass:
        out["sign_input_bypass"] = disable_input_quantizers_by_pattern(
            model, SIGN_NAME_RE, verbose=verbose,
        )
    return out


def apply_mrnn_clz_encoding_fixes_post_calib(
    model: nn.Module,
    *,
    reciprocal_denom: bool = True,
    power2_output: bool = True,
    power2_float_out_fmax: dict[str, float] | None = None,
    eps: float = 1e-8,
    verbose: bool = False,
) -> dict[str, Any]:
    """compute_encodings + Po2 之后：reciprocal 分母 + power_2 输出。"""
    out: dict[str, Any] = {}
    if reciprocal_denom:
        out["reciprocal_denom"] = fix_reciprocal_denom_encodings(
            model, eps=eps, verbose=verbose,
        )
    if power2_output:
        out["power2_output"] = fix_power2_output_encodings(
            model,
            float_out_fmax=power2_float_out_fmax,
            verbose=verbose,
        )
        if power2_float_out_fmax:
            out["power2_float_out_fmax"] = power2_float_out_fmax
    return out

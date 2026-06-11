"""MRNN 分解图与 §2.3 CLZ 规格化类的 encoding 修复（非 bypass 主路径）。

- sign: 可选 bypass input Q；硬件整数 sign 路径用专用 fine-scale input Q（仅 pc1）
- power_compress_2: 幅度链正域 unsigned grid [0,4]/[0,2]
- module_div 分母 b=std: reciprocal 正域 encoding（unsigned + min≥eps）
- module_square: power_2 输出 range 重标定 out_min/out_max
"""
from __future__ import annotations

import re
from typing import Any, Iterator

import torch
import torch.nn as nn

SIGN_NAME_RE = re.compile(r"(^|\.)module_sign(?:_\d+)?$")
PC1_SIGN_ONLY_RE = re.compile(r"power_compress_1\.module_sign$")
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


def _fresh_fmax_scan_loader(loader) -> "torch.utils.data.DataLoader":
    """独立于 PTQ calib 迭代状态的 fmax 扫描 loader（shuffle=False，全量）。"""
    from torch.utils.data import DataLoader

    if not isinstance(loader, DataLoader):
        return loader
    return DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=loader.num_workers,
        pin_memory=getattr(loader, "pin_memory", False),
        collate_fn=loader.collate_fn,
        drop_last=False,
        worker_init_fn=getattr(loader, "worker_init_fn", None),
    )


def collect_power2_float_out_fmax(
    float_model: nn.Module,
    loader,
    device: torch.device,
    max_batches: int | None = None,
    *,
    full_scan: bool = True,
) -> dict[str, float]:
    """校准数据上观测 float 图 power_2 输出 fmax（用于 out_max / CLZ head）。

    ``full_scan=True``（默认）：用独立 DataLoader 对 ``loader.dataset`` 做
    shuffle=False 全量扫描，避免与 ``loaders['calib']`` 共用迭代器时 epoch 偏移
    导致 out_max 低估（整图 QDQ+INT16 同进程会差 ~4 pp）。
    ``full_scan=False``：沿用传入 loader，最多 ``max_batches`` 个 batch。
    """
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

    scan_loader = _fresh_fmax_scan_loader(loader) if full_scan else loader

    was_training = float_model.training
    float_model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(scan_loader):
            if not full_scan and max_batches is not None and idx >= max_batches:
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


def fix_sign_input_encodings(
    model: nn.Module,
    *,
    scale: float = 2e-9,
    min_bitwidth: int = 16,
    verbose: bool = False,
) -> dict[str, Any]:
    """重标定 ``module_sign`` 输入为符号专用 fine scale。

    该输入只供 spec §4.3.6 ``sign(q_x - Z_x)`` 比较器使用，幅值饱和不影响
    输出符号；目标是避免 STFT 近零非零值被 input Q round 到 0。
    """

    stats: dict[str, Any] = {"touched": 0, "skipped": [], "scale": scale}
    if scale <= 0:
        raise ValueError(f"sign input scale must be positive; got {scale}")

    for name, mod in model.named_modules():
        if not SIGN_NAME_RE.search(name):
            continue
        # pc2 输入为 [0,4] 幅度，不需要 STFT 近零 fine scale（仅 pc1 需要）
        if not PC1_SIGN_ONLY_RE.search(name):
            continue
        iqs = getattr(mod, "input_quantizers", None)
        if not iqs or iqs[0] is None:
            stats["skipped"].append((name, "no input quantizer"))
            continue
        q = iqs[0]
        if not _q_initialized(q):
            stats["skipped"].append((name, "input not initialized"))
            continue

        bw = max(int(getattr(q, "bitwidth", min_bitwidth)), min_bitwidth)
        q.bitwidth = bw
        q.symmetric = True
        q.qmin = -(2 ** (bw - 1))
        q.qmax = 2 ** (bw - 1) - 1
        amax = scale * float(q.qmax)
        dev, dt = _q_device_dtype(q)
        q.set_range(
            torch.tensor(-amax, device=dev, dtype=dt),
            torch.tensor(amax, device=dev, dtype=dt),
        )
        stats["touched"] += 1
        if verbose:
            print(
                f"  sign input: {name}.input[0] bw={bw} scale={scale:.3e} "
                f"range=[{-amax:.3e}, {amax:.3e}]"
            )
    return stats


def fix_sign_output_encodings(
    model: nn.Module,
    *,
    bitwidth: int = 8,
    verbose: bool = False,
) -> dict[str, Any]:
    """将 ``module_sign`` 输出重标定为 unit grid。

    spec §4.3.6 的硬件输出码值就是 ``{-1, 0, 1}``。因此输出 scale 必须为
    1.0，后续算子解读该 tensor 时才得到实值 ``{-1, 0, 1}``。
    """

    stats: dict[str, Any] = {"touched": 0, "skipped": [], "scale": 1.0}
    if bitwidth < 2:
        raise ValueError(f"sign output bitwidth must be >= 2; got {bitwidth}")

    qmin = -(2 ** (bitwidth - 1))
    qmax = 2 ** (bitwidth - 1) - 1
    for name, mod in model.named_modules():
        if not SIGN_NAME_RE.search(name):
            continue
        oqs = getattr(mod, "output_quantizers", None)
        if not oqs or oqs[0] is None:
            stats["skipped"].append((name, "no output quantizer"))
            continue
        q = oqs[0]
        if not _q_initialized(q):
            stats["skipped"].append((name, "output not initialized"))
            continue

        q.bitwidth = bitwidth
        q.symmetric = True
        q.qmin = qmin
        q.qmax = qmax
        dev, dt = _q_device_dtype(q)
        q.set_range(
            torch.tensor(float(qmin), device=dev, dtype=dt),
            torch.tensor(float(qmax), device=dev, dtype=dt),
        )
        stats["touched"] += 1
        if verbose:
            print(f"  sign output: {name}.output[0] bw={bitwidth} scale=1.0")
    return stats


def _set_unsigned_range(q, lo: float, hi: float) -> None:
    bw = int(getattr(q, "bitwidth", 16))
    q.bitwidth = bw
    q.symmetric = False
    q.qmin = 0
    q.qmax = 2**bw - 1
    dev, dt = _q_device_dtype(q)
    q.set_range(
        torch.tensor(lo, device=dev, dtype=dt),
        torch.tensor(hi, device=dev, dtype=dt),
    )


def fix_positive_clamp_output_encodings(
    model: nn.Module,
    *,
    verbose: bool = False,
) -> dict[str, Any]:
    """正域 clamp 输出改为 unsigned，避免 pc2 入口仍是对称 grid。"""

    stats: dict[str, Any] = {"touched": 0, "skipped": []}
    targets = {
        "module_clamp_2": (0.0, 4.0),
        "module_clamp_3": (0.0, 4.0),
        "module_clamp_4": (0.0, 2.0),
    }
    mods = dict(model.named_modules())
    for name, (lo, hi) in targets.items():
        mod = mods.get(name)
        if mod is None:
            stats["skipped"].append((name, "missing module"))
            continue
        oqs = getattr(mod, "output_quantizers", None)
        if not oqs or oqs[0] is None or not _q_initialized(oqs[0]):
            stats["skipped"].append((name, "output not initialized"))
            continue
        _set_unsigned_range(oqs[0], lo, hi)
        stats["touched"] += 1
        if verbose:
            print(f"  clamp output unsigned: {name} [0, {hi}]")
    return stats


def fix_power_compress_2_positive_encodings(
    model: nn.Module,
    *,
    mag_max: float = 4.0,
    sqrt_max: float = 2.0,
    verbose: bool = False,
) -> dict[str, Any]:
    """pc2 输入为 ``module_clamp_3`` 后的非负幅度 ``[0, mag_max]``。

    16-bit 全链需正域 unsigned grid；对称 ``[-mag_max, mag_max]`` 会让
    ``module_abs_2`` isolated/chained SQNR 崩到 ~0 dB。
    """

    stats: dict[str, Any] = {"touched": 0, "skipped": []}
    targets: dict[str, dict[str, tuple[float, float] | None]] = {
        "power_compress_2.module_sign_1": {"in0": (0.0, mag_max)},
        "power_compress_2.module_abs_2": {"in0": (0.0, mag_max), "out0": (0.0, mag_max)},
        "power_compress_2.module_sqrt_2": {
            "in0": (0.0, mag_max),
            "out0": (0.0, sqrt_max),
        },
        "power_compress_2.module_mul_2": {"out0": (0.0, sqrt_max)},
    }
    mods = dict(model.named_modules())
    for name, slots in targets.items():
        mod = mods.get(name)
        if mod is None:
            stats["skipped"].append((name, "missing module"))
            continue
        touched_here = 0
        if "in0" in slots and slots["in0"] is not None:
            iqs = getattr(mod, "input_quantizers", None)
            if not iqs or iqs[0] is None or not _q_initialized(iqs[0]):
                stats["skipped"].append((name, "input0 not initialized"))
            else:
                lo, hi = slots["in0"]
                _set_unsigned_range(iqs[0], lo, hi)
                touched_here += 1
        if "out0" in slots and slots["out0"] is not None:
            oqs = getattr(mod, "output_quantizers", None)
            if not oqs or oqs[0] is None or not _q_initialized(oqs[0]):
                stats["skipped"].append((name, "output0 not initialized"))
            else:
                lo, hi = slots["out0"]
                _set_unsigned_range(oqs[0], lo, hi)
                touched_here += 1
        if touched_here:
            stats["touched"] += touched_here
            if verbose:
                print(f"  pc2 positive domain: {name} slots={touched_here}")
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
    positive_clamp_output: bool = True,
    pc2_positive: bool = True,
    sign_input_scale: float | None = None,
    sign_output_unit: bool = False,
    power2_float_out_fmax: dict[str, float] | None = None,
    eps: float = 1e-8,
    verbose: bool = False,
) -> dict[str, Any]:
    """compute_encodings 之后（可选 legacy Po2 之后）：reciprocal 分母 + power_2 输出。"""
    out: dict[str, Any] = {}
    if sign_input_scale is not None:
        out["sign_input"] = fix_sign_input_encodings(
            model, scale=sign_input_scale, verbose=verbose,
        )
    if sign_output_unit:
        out["sign_output"] = fix_sign_output_encodings(model, verbose=verbose)
    if positive_clamp_output:
        out["positive_clamp_output"] = fix_positive_clamp_output_encodings(
            model, verbose=verbose,
        )
    if pc2_positive:
        out["pc2_positive"] = fix_power_compress_2_positive_encodings(
            model, verbose=verbose,
        )
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

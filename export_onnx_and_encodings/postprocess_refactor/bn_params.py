from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Tuple

import numpy as np
import onnx
from onnx import numpy_helper

logger = logging.getLogger(__name__)


def _find_no_clip_power_of_2_scale(scale0: float) -> Tuple[float, int]:
    """
    返回 **不小于** scale0 的最小 2 的幂次 scale（即 2^(-floor(log2(1/scale0)))）。

    与 round 版本的区别：
    - round 版：scale 四舍五入，可能 < scale0 → 实际值被截断（clipping）
    - floor 版：scale 只向"大方向"对齐 2 次幂，保证 scale >= scale0 → 无截断

    对 running_mean / running_var 这类固定常量，零截断比极限精度更重要。
    """
    if scale0 <= 0 or not math.isfinite(scale0):
        raise ValueError(f"scale0 must be finite positive, got {scale0}")
    n = -math.log2(scale0)           # n = log2(1/scale0)，即 n 越大 scale 越小
    n_floor = math.floor(n)           # floor → scale = 2^(-n_floor) >= scale0
    return float(2.0 ** (-n_floor)), int(n_floor)


def _build_per_tensor_no_clip_encoding(
    arr: np.ndarray,
    *,
    bitwidth: int,
    dtype: str = "int",
) -> Dict[str, Any]:
    """
    对 arr **所有元素**取全局 max_abs，计算 no-clip 对称 Po2 编码（per-tensor）。

    running_mean / running_var 为何必须用 per-tensor（而非 per-channel）：
    ─────────────────────────────────────────────────────────────────────
    在定点 BN 计算中：
        x_centered = x - running_mean[c]
    若 x 是 per-tensor（全局 scale S_x）而 running_mean 是 per-channel
    （每通道 scale S_mean[c] 各不相同），则无法直接做整数减法：
        INT(x_centered) = INT(x) - INT(mean[c])
        仅当 S_x == S_mean 时成立，否则必须反量化到 float 再相减，
        使 running_mean 的量化失去意义。
    因此 running_mean 必须用 per-tensor，与输入保持相同的 scale 粒度。

    running_var 走的是 sqrt 分支（不与 x 做减法），理论上可以 per-channel，
    但为简单一致性，同样使用 per-tensor。
    """
    bw = int(bitwidth)
    qmax_s = (2 ** (bw - 1)) - 1   # 127 for int8
    qmin_s = -(2 ** (bw - 1))       # -128

    arr_f = np.asarray(arr).astype(np.float64).ravel()
    max_abs = float(np.max(np.abs(arr_f))) if arr_f.size else 0.0
    max_abs = max(max_abs, 1e-12)

    scale0 = max_abs / float(qmax_s)
    scale, n = _find_no_clip_power_of_2_scale(scale0)

    real_min = float(qmin_s) * scale
    real_max = float(qmax_s) * scale

    return {
        "bitwidth": bw,
        "dtype": dtype,
        "is_symmetric": "True",
        "min": real_min,
        "max": real_max,
        "offset": 0,
        "scale": scale,
    }


def add_bn_param_encodings_from_onnx(
    enc: Dict[str, Any],
    *,
    onnx_path: str,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], bool]:
    """
    对 ONNX 图中每个 BatchNormalization 节点，若 encodings 中缺少
    running_mean / running_var 的量化参数，则从 ONNX initializer 的实际数值
    计算 **per-tensor、no-clip** 的对称 Po2 编码并补齐。

    若 encodings 中已有对应 key（由 AIMET 校准写入），则不覆盖。
    """
    if "param_encodings" not in enc or not isinstance(enc["param_encodings"], dict):
        enc["param_encodings"] = {}
    pe: Dict[str, Any] = enc["param_encodings"]

    try:
        model = onnx.load(onnx_path)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"无法加载 ONNX 用于补齐 BN param_encodings: {exc}") from exc

    init_map = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
    changed = False

    for node in model.graph.node:
        if node.op_type != "BatchNormalization":
            continue
        bn_name = node.name or (node.output[0] if node.output else "")
        if not bn_name:
            continue
        if len(node.input) < 5:
            continue

        # 从已有的 weight / bias 编码中获取 bitwidth
        bw = None
        for k in (f"{bn_name}.weight", f"{bn_name}.bias"):
            v = pe.get(k)
            if isinstance(v, list) and v and isinstance(v[0], dict) and "bitwidth" in v[0]:
                bw = int(v[0]["bitwidth"])
                break
            if isinstance(v, dict) and "bitwidth" in v:
                bw = int(v["bitwidth"])
                break
        if bw is None:
            bw = int(enc.get("quantizer_args", {}).get("param_bitwidth", 8))

        targets = {
            "running_mean": node.input[3],
            "running_var":  node.input[4],
        }
        for suffix, init_name in targets.items():
            key = f"{bn_name}.{suffix}"
            if key in pe:
                # 已由 AIMET 校准写入，不覆盖
                if verbose:
                    logger.debug("⏭️  跳过已有 BN 编码: %s", key)
                continue
            if init_name not in init_map:
                if verbose:
                    logger.warning("⚠️  ONNX initializer '%s' 不存在，跳过 %s", init_name, key)
                continue

            arr = init_map[init_name]
            enc_one = _build_per_tensor_no_clip_encoding(arr, bitwidth=bw, dtype="int")
            pe[key] = [enc_one]
            changed = True
            if verbose:
                logger.info(
                    "✅ 补齐 BN param_encodings (per-tensor, no-clip): %s  "
                    "scale=%.6g  range=[%.4g, %.4g]",
                    key, enc_one["scale"], enc_one["min"], enc_one["max"],
                )

    enc["param_encodings"] = pe
    return enc, changed

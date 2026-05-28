from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def merge_gru_bias_param_encodings(enc: Dict[str, Any], verbose: bool = True) -> Tuple[Dict[str, Any], bool]:
    if "param_encodings" not in enc or not isinstance(enc["param_encodings"], dict):
        return enc, False

    pe: Dict[str, Any] = enc["param_encodings"]
    changed = False

    prefixes: set[str] = set()
    for k in pe.keys():
        if k.endswith(".weight_ih.bias"):
            prefixes.add(k[: -len(".weight_ih.bias")])
        elif k.endswith(".weight_hh.bias"):
            prefixes.add(k[: -len(".weight_hh.bias")])

    for prefix in sorted(prefixes):
        ih_key = prefix + ".weight_ih.bias"
        hh_key = prefix + ".weight_hh.bias"
        out_key = prefix + ".bias"

        ih_val = pe.get(ih_key)
        hh_val = pe.get(hh_key)
        if ih_val is None and hh_val is None:
            continue

        merged = _as_list(ih_val) + _as_list(hh_val)
        if not merged:
            continue

        if out_key in pe and pe[out_key] != merged:
            if verbose:
                logger.warning("GRU bias encodings 冲突：%s 已存在，将覆盖为合并后的结果", out_key)
        pe[out_key] = merged

        if ih_key in pe:
            del pe[ih_key]
            changed = True
        if hh_key in pe:
            del pe[hh_key]
            changed = True
        changed = True

    enc["param_encodings"] = pe
    return enc, changed


def compact_per_channel_param_encodings(enc: Dict[str, Any], verbose: bool = True) -> Tuple[Dict[str, Any], bool]:
    if "param_encodings" not in enc or not isinstance(enc["param_encodings"], dict):
        return enc, False

    pe: Dict[str, Any] = enc["param_encodings"]
    changed = False
    always_list_fields = {"min", "max", "offset", "scale"}

    for k, v in list(pe.items()):
        if not (isinstance(v, list) and v and all(isinstance(e, dict) for e in v)):
            continue

        entries: List[Dict[str, Any]] = v  # type: ignore[assignment]
        keys = set(entries[0].keys())
        for e in entries[1:]:
            keys &= set(e.keys())
        if not keys:
            continue

        common: Dict[str, Any] = {}
        per_ch: Dict[str, List[Any]] = {}
        for kk in sorted(keys):
            vals = [e.get(kk) for e in entries]
            if kk in always_list_fields:
                per_ch[kk] = vals
                continue
            all_same = all(val == vals[0] for val in vals)
            if all_same:
                common[kk] = vals[0]
            else:
                per_ch[kk] = vals

        all_keys = set().union(*(set(e.keys()) for e in entries))
        extra_keys = all_keys - keys
        for kk in sorted(extra_keys):
            per_ch[kk] = [e.get(kk) for e in entries]

        compacted: Dict[str, Any] = {}
        compacted.update(common)
        compacted.update(per_ch)

        pe[k] = compacted
        changed = True

    if changed:
        if verbose:
            logger.info("✅ 已将 param_encodings 中的 per-channel list[dict] 压缩为 dict 格式")
        enc["param_encodings"] = pe
    return enc, changed


def merge_gru_activation_encodings(enc: Dict[str, Any], verbose: bool = True) -> Tuple[Dict[str, Any], bool]:
    if "activation_encodings" not in enc or not isinstance(enc["activation_encodings"], dict):
        return enc, False

    act = enc["activation_encodings"]
    grouped: Dict[str, Dict[str, Any]] = {}
    for key, val in list(act.items()):
        parts = key.split(".")
        # 同时处理 cells（单向/双向前向）和 reverse_cells（双向反向）
        if len(parts) < 4 or parts[-3] not in ("cells", "reverse_cells"):
            continue
        gru_prefix = ".".join(parts[:-1])
        op_name = parts[-1]
        grouped.setdefault(gru_prefix, {})[op_name] = val

    if not grouped:
        return enc, False

    changed = False
    for gru_prefix, ops in grouped.items():
        for op in ops:
            old_key = f"{gru_prefix}.{op}"
            if old_key in act:
                del act[old_key]
                changed = True
        if gru_prefix not in act:
            act[gru_prefix] = {"internal_ops": ops}
        else:
            entry = act[gru_prefix]
            if not isinstance(entry, dict):
                entry = {}
            internal_ops = entry.get("internal_ops", {})
            internal_ops.update(ops)
            entry["internal_ops"] = internal_ops
            act[gru_prefix] = entry
        changed = True

    return enc, changed


def rename_gru_internal_ops_keys(enc: Dict[str, Any], verbose: bool = True) -> Tuple[Dict[str, Any], bool]:
    if "activation_encodings" not in enc or not isinstance(enc["activation_encodings"], dict):
        return enc, False

    act: Dict[str, Any] = enc["activation_encodings"]
    changed = False
    rename_map = {
        "weight_ih": "weight_ih_linear",
        "weight_hh": "weight_hh_linear",
        "add_r_gate": "reset_gate_input",
        "add_u_gate": "update_gate_input",
        "mul_reset": "mul_reset_hidden",
        "add_n_gate": "new_gate_input",
        "sub_one_minus": "sub_one_minus_update",
        "add_final": "add_final_hidden",
    }

    for gru_name, entry in list(act.items()):
        if not isinstance(entry, dict):
            continue

        local_changed = False
        # 同时处理 internal_ops（前向）和 internal_ops_reverse（反向双向 GRU）
        for ops_field in ("internal_ops", "internal_ops_reverse"):
            internal_ops = entry.get(ops_field)
            if not isinstance(internal_ops, dict):
                continue
            if (
                "weight_ih" not in internal_ops
                and "weight_hh" not in internal_ops
                and "weight_ih_linear" not in internal_ops
                and "weight_hh_linear" not in internal_ops
            ):
                continue

            new_internal = dict(internal_ops)
            for old_k, new_k in rename_map.items():
                if old_k not in new_internal:
                    continue
                if new_k not in new_internal:
                    new_internal[new_k] = new_internal[old_k]
                del new_internal[old_k]
                local_changed = True

            if local_changed:
                entry[ops_field] = new_internal

        if local_changed:
            act[gru_name] = entry
            changed = True
            if verbose:
                logger.info("✅ 已重命名 GRU(%s) internal_ops key 以对齐后端字段名", gru_name)

    if changed:
        enc["activation_encodings"] = act
    return enc, changed


def add_gru_output_from_internal_ops(enc: Dict[str, Any], verbose: bool = True) -> Tuple[Dict[str, Any], bool]:
    if "activation_encodings" not in enc or not isinstance(enc["activation_encodings"], dict):
        return enc, False

    act: Dict[str, Any] = enc["activation_encodings"]
    changed = False

    for gru_name, entry in list(act.items()):
        if not isinstance(entry, dict):
            continue
        internal_ops = entry.get("internal_ops")
        if not isinstance(internal_ops, dict):
            continue
        if "output" in entry:
            continue

        add_final = internal_ops.get("add_final_hidden") or internal_ops.get("add_final")
        if not isinstance(add_final, dict):
            continue
        out_enc = add_final.get("output")
        if out_enc is None:
            continue

        entry["output"] = out_enc
        act[gru_name] = entry
        changed = True
        if verbose:
            logger.info("✅ 已为 GRU(%s) 补齐 output（来源 internal_ops.add_final_hidden.output）", gru_name)

    if changed:
        enc["activation_encodings"] = act
    return enc, changed


def add_gru_input_from_internal_ops(enc: Dict[str, Any], verbose: bool = True) -> Tuple[Dict[str, Any], bool]:
    if "activation_encodings" not in enc or not isinstance(enc["activation_encodings"], dict):
        return enc, False

    act: Dict[str, Any] = enc["activation_encodings"]
    changed = False
    candidates = ("weight_ih_linear", "weight_ih", "weight_ih_l0", "ih", "linear_ih")

    for gru_name, entry in list(act.items()):
        if not isinstance(entry, dict):
            continue
        internal_ops = entry.get("internal_ops")
        if not isinstance(internal_ops, dict):
            continue
        if "input" in entry:
            continue

        picked = None
        used_key = None
        for ck in candidates:
            op_entry = internal_ops.get(ck)
            if isinstance(op_entry, dict) and "input" in op_entry:
                picked = op_entry.get("input")
                if picked is not None:
                    used_key = ck
                    break

        if picked is None:
            continue

        entry["input"] = picked
        act[gru_name] = entry
        changed = True
        if verbose:
            logger.info("✅ 已为 GRU(%s) 补齐 input（来源 internal_ops.%s.input）", gru_name, used_key or "weight_ih_linear")

    if changed:
        enc["activation_encodings"] = act
    return enc, changed


def merge_bn_activation_encodings(enc: Dict[str, Any], verbose: bool = True) -> Tuple[Dict[str, Any], bool]:
    if "activation_encodings" not in enc or not isinstance(enc["activation_encodings"], dict):
        return enc, False

    act = enc["activation_encodings"]
    grouped: Dict[str, Dict[str, Any]] = {}
    for key, val in list(act.items()):
        parts = key.split(".")
        if len(parts) < 2:
            continue
        if "pre_bn" not in parts and "rnn2d_bn" not in parts:
            continue
        prefix = ".".join(parts[:-1])
        op_name = parts[-1]
        grouped.setdefault(prefix, {})[op_name] = val

    if not grouped:
        return enc, False

    changed = False
    for bn_prefix, ops in grouped.items():
        for op in ops:
            old_key = f"{bn_prefix}.{op}"
            if old_key in act:
                del act[old_key]
                changed = True
        if bn_prefix not in act:
            act[bn_prefix] = {"internal_ops": ops}
        else:
            entry = act[bn_prefix]
            if not isinstance(entry, dict):
                entry = {}
            internal_ops = entry.get("internal_ops", {})
            internal_ops.update(ops)
            entry["internal_ops"] = internal_ops
            act[bn_prefix] = entry
        changed = True

    enc["activation_encodings"] = act
    return enc, changed


def flatten_activation_io_index_dict(enc: Dict[str, Any], verbose: bool = True) -> Tuple[Dict[str, Any], bool]:
    if "activation_encodings" not in enc or not isinstance(enc["activation_encodings"], dict):
        return enc, False

    act: Dict[str, Any] = enc["activation_encodings"]
    changed = False

    def _is_index_dict(x: Any) -> bool:
        if not isinstance(x, dict) or not x:
            return False
        return all(isinstance(k, str) and k.isdigit() for k in x.keys())

    def _sorted_index_items(d: Dict[str, Any]) -> List[Any]:
        return [d[k] for k in sorted(d.keys(), key=lambda s: int(s))]

    def _to_list(v: Any) -> Any:
        if _is_index_dict(v):
            return _sorted_index_items(v)
        if isinstance(v, dict):
            return [v]
        if isinstance(v, list):
            return v
        return v

    def _normalize_entry_io(entry: Dict[str, Any]) -> bool:
        local_changed = False
        for io_key in ("input", "output"):
            if io_key not in entry:
                continue
            before = entry[io_key]
            after = _to_list(before)
            if after is not before:
                entry[io_key] = after
                local_changed = True

        for ops_field in ("internal_ops", "internal_ops_reverse"):
            internal = entry.get(ops_field)
            if isinstance(internal, dict):
                for _, op_entry in internal.items():
                    if isinstance(op_entry, dict):
                        if _normalize_entry_io(op_entry):
                            local_changed = True
        return local_changed

    for op_name, entry in list(act.items()):
        if not isinstance(entry, dict):
            continue
        if _normalize_entry_io(entry):
            changed = True
        act[op_name] = entry

    if changed:
        enc["activation_encodings"] = act
        if verbose:
            logger.info("✅ 已规范化 activation_encodings 的 input/output 为 list[dict]（并递归处理 internal_ops，删除 '0'/'1' ...）")
    return enc, changed


def normalize_quant_fields_add_n(enc: Dict[str, Any], verbose: bool = True) -> Tuple[Dict[str, Any], bool]:
    changed = False

    def _to_n(scale_val: Any) -> Any:
        if isinstance(scale_val, list):
            return [_to_n(x) for x in scale_val]
        try:
            s = float(scale_val)
        except Exception:  # noqa: BLE001
            return None
        if s <= 0:
            return None
        n_f = -math.log2(s)
        n_i = int(round(n_f))
        if abs(n_f - n_i) > 1e-6 and verbose:
            logger.warning("scale=%s 得到的 n 非整数：%s（将 round 为 %d）", scale_val, n_f, n_i)
        return n_i

    def _norm_dict(d: Dict[str, Any]) -> bool:
        local_changed = False
        if "scale" in d and "n" not in d:
            n_val = _to_n(d.get("scale"))
            if n_val is not None:
                d["n"] = n_val
                local_changed = True

        rename_map = {
            "offset": "zero_point",
            "min": "real_min",
            "max": "real_max",
        }
        for old, new in rename_map.items():
            if old in d and new not in d:
                d[new] = d.pop(old)
                local_changed = True
            elif old in d and new in d:
                d.pop(old, None)
                local_changed = True
        return local_changed

    def _is_index_dict(x: Any) -> bool:
        return isinstance(x, dict) and bool(x) and all(isinstance(k, str) and k.isdigit() for k in x.keys())

    def _visit(obj: Any) -> None:
        nonlocal changed
        if obj is None:
            return
        if isinstance(obj, list):
            for it in obj:
                _visit(it)
            return
        if isinstance(obj, dict):
            if _is_index_dict(obj):
                for k in sorted(obj.keys(), key=lambda s: int(s)):
                    _visit(obj[k])
                return
            if any(k in obj for k in ("scale", "offset", "min", "max", "zero_point", "real_min", "real_max")):
                if _norm_dict(obj):
                    changed = True
            for v in obj.values():
                _visit(v)
            return

    _visit(enc.get("activation_encodings"))
    _visit(enc.get("param_encodings"))

    if changed and verbose:
        logger.info(
            "✅ 已新增 n=-log2(scale)，并重命名字段 offset/min/max -> zero_point/real_min/real_max"
        )
    return enc, changed


def normalize_dtype_with_bitwidth(enc: Dict[str, Any], verbose: bool = True) -> Tuple[Dict[str, Any], bool]:
    changed = False

    def _norm_one(d: Dict[str, Any]) -> bool:
        if "dtype" not in d or "bitwidth" not in d:
            return False
        dt = d.get("dtype")
        bw = d.get("bitwidth")
        if not isinstance(dt, str):
            return False
        try:
            bw_i = int(bw)
        except Exception:  # noqa: BLE001
            return False
        dt_up = dt.upper()
        base = dt_up.rstrip("0123456789")
        dt_new = f"{base}{bw_i}"
        if dt_new != dt:
            d["dtype"] = dt_new
            return True
        return False

    def _is_index_dict(x: Any) -> bool:
        return isinstance(x, dict) and bool(x) and all(isinstance(k, str) and k.isdigit() for k in x.keys())

    def _visit(obj: Any) -> None:
        nonlocal changed
        if obj is None:
            return
        if isinstance(obj, list):
            for it in obj:
                _visit(it)
            return
        if isinstance(obj, dict):
            if _is_index_dict(obj):
                for k in sorted(obj.keys(), key=lambda s: int(s)):
                    _visit(obj[k])
                return
            if _norm_one(obj):
                changed = True
            for v in obj.values():
                _visit(v)
            return

    _visit(enc.get("activation_encodings"))
    _visit(enc.get("param_encodings"))

    if changed and verbose:
        logger.info("✅ 已规范化 dtype：转大写并拼接 bitwidth（如 INT8/INT16/INT32）")
    return enc, changed


def merge_reverse_cells_into_cells(
    enc: Dict[str, Any], verbose: bool = True
) -> Tuple[Dict[str, Any], bool]:
    """
    将双向 GRU 反向方向的激活量化参数合并到前向条目中。

    经 E1 合并后，encodings 中存在两个顶层条目：
      ...seq_f.cells.0          {"internal_ops": {前向 op encodings}, "output": ..., "input": ...}
      ...seq_f.reverse_cells.0  {"internal_ops": {反向 op encodings}, "output": ..., "input": ...}

    本函数将反向条目的 internal_ops 作为 internal_ops_reverse 移入前向条目，并删除反向顶层条目：
      ...seq_f.cells.0  {
          "internal_ops":         {前向 op encodings},
          "internal_ops_reverse": {反向 op encodings},
          "output":               ...,  # 前向输出激活量化
          "output_reverse":       ...,  # 反向输出激活量化（若存在）
          "input":                ...,
      }
    """
    if "activation_encodings" not in enc or not isinstance(enc["activation_encodings"], dict):
        return enc, False

    act: Dict[str, Any] = enc["activation_encodings"]
    changed = False

    for rev_key in list(act.keys()):
        if ".reverse_cells." not in rev_key:
            continue
        rev_entry = act[rev_key]
        if not isinstance(rev_entry, dict):
            continue

        # 派生对应的 cells.0 键：把 ".reverse_cells." 替换为 ".cells."
        cells_key = rev_key.replace(".reverse_cells.", ".cells.")
        if cells_key not in act or not isinstance(act[cells_key], dict):
            continue

        cells_entry = act[cells_key]

        # 移入 internal_ops_reverse
        if "internal_ops" in rev_entry:
            cells_entry["internal_ops_reverse"] = rev_entry["internal_ops"]
            changed = True

        # 移入 output_reverse（若存在且不同于前向 output）
        if "output" in rev_entry and rev_entry["output"] != cells_entry.get("output"):
            cells_entry["output_reverse"] = rev_entry["output"]
            changed = True

        act[cells_key] = cells_entry
        del act[rev_key]
        changed = True

        if verbose:
            logger.info("✅ 双向GRU：%s 的反向激活量化参数已合并至 %s.internal_ops_reverse",
                        rev_key, cells_key)

    if changed:
        enc["activation_encodings"] = act
    return enc, changed


def merge_bidirectional_gru_param_encodings(
    enc: Dict[str, Any], verbose: bool = True
) -> Tuple[Dict[str, Any], bool]:
    """
    双向 GRU 的权重/bias per-channel 量化参数合并。

    ONNX 双向 GRU 的 W/R/B 张量均为前向+反向合并后的单张量：
      W: [2, 3H, I]  R: [2, 3H, H]  B: [2, 6H]
    而 AIMET 为前向（cells.0）和反向（reverse_cells.0）分别生成了独立的
    per-channel 量化参数（各 3H 通道）。

    本函数将二者合并为二维结构 [2, 3H 通道]：
      scale:      [[fwd_ch0, ..., fwd_chN], [bwd_ch0, ..., bwd_chN]]
      zero_point: [[...], [...]]
      ... 等 per-channel 字段均变为二维列表

    最终 cells.0.weight_ih.weight 的 scale 等字段形状为 [2, 3H]，
    与 ONNX W 张量的 [2, 3H, I] 对齐。
    删除孤立的 reverse_cells.0.* 条目。
    """
    if "param_encodings" not in enc or not isinstance(enc["param_encodings"], dict):
        return enc, False

    pe: Dict[str, Any] = enc["param_encodings"]
    _per_ch_list_keys = {"scale", "n", "zero_point", "real_min", "real_max"}

    def _merge_per_channel_2d(fwd: Any, bwd: Any) -> Any:
        """将前向/反向 per-channel 量化参数合并为二维 [2, N] 结构。"""
        if isinstance(fwd, list) and isinstance(bwd, list):
            # 旧格式：list of dicts（compact 前）
            return [fwd, bwd]
        if isinstance(fwd, dict) and isinstance(bwd, dict):
            merged: Dict[str, Any] = {}
            for k in fwd:
                fv = fwd[k]
                bv = bwd.get(k, fv)
                if k in _per_ch_list_keys and isinstance(fv, list) and isinstance(bv, list):
                    # 变为二维列表：[[前向通道], [反向通道]]
                    merged[k] = [fv, bv]
                else:
                    merged[k] = fv  # bitwidth / dtype / is_symmetric 取前向值
            return merged
        return fwd  # 无法合并，保留前向

    weight_suffixes = (".weight_ih.weight", ".weight_hh.weight", ".bias")
    fwd_prefix_pat = ".cells.0"
    bwd_prefix_pat = ".reverse_cells.0"

    changed = False
    fwd_keys_seen: set = set()
    for k in list(pe.keys()):
        for suffix in weight_suffixes:
            if not k.endswith(suffix):
                continue
            base = k[: -len(suffix)]
            if not base.endswith(fwd_prefix_pat):
                continue
            parent = base[: -len(fwd_prefix_pat)]
            bwd_key = parent + bwd_prefix_pat + suffix
            if bwd_key not in pe or k in fwd_keys_seen:
                continue

            fwd_keys_seen.add(k)
            fwd_enc = pe[k]
            bwd_enc = pe[bwd_key]
            merged = _merge_per_channel_2d(fwd_enc, bwd_enc)
            pe[k] = merged
            del pe[bwd_key]
            changed = True

            fwd_ch = len(fwd_enc) if isinstance(fwd_enc, list) else len(fwd_enc.get("scale", []))
            if verbose:
                logger.info(
                    "✅ 双向GRU权重合并: %s [2, %d] (前向 %d + 反向 %d)",
                    k, fwd_ch, fwd_ch, fwd_ch,
                )

    if changed:
        enc["param_encodings"] = pe
    return enc, changed


def postprocess_encodings_inplace(enc: Dict[str, Any], verbose: bool = True) -> Tuple[Dict[str, Any], bool]:
    changed_any = False

    enc, c1 = merge_gru_activation_encodings(enc, verbose=verbose)
    changed_any |= c1

    enc, c1_2 = rename_gru_internal_ops_keys(enc, verbose=verbose)
    changed_any |= c1_2

    enc, c1_5 = add_gru_output_from_internal_ops(enc, verbose=verbose)
    changed_any |= c1_5

    enc, c1_6 = add_gru_input_from_internal_ops(enc, verbose=verbose)
    changed_any |= c1_6

    enc, c3 = merge_gru_bias_param_encodings(enc, verbose=verbose)
    changed_any |= c3

    enc, c4 = compact_per_channel_param_encodings(enc, verbose=verbose)
    changed_any |= c4

    enc, c5 = flatten_activation_io_index_dict(enc, verbose=verbose)
    changed_any |= c5

    enc, c6 = normalize_quant_fields_add_n(enc, verbose=verbose)
    changed_any |= c6

    enc, c7 = normalize_dtype_with_bitwidth(enc, verbose=verbose)
    changed_any |= c7

    return enc, changed_any

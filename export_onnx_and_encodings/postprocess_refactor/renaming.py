from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import onnx

from .io_utils import load_encodings, save_encodings

logger = logging.getLogger(__name__)


def _rename_value_references(model: onnx.ModelProto, old_name: str, new_name: str) -> None:
    if old_name == new_name:
        return

    for node in model.graph.node:
        node.input[:] = [new_name if name == old_name else name for name in node.input]
        node.output[:] = [new_name if name == old_name else name for name in node.output]

    for value in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        if value.name == old_name:
            value.name = new_name


def _strip_hash_suffix(name: str) -> str:
    if "#" in name:
        return name.split("#", 1)[0]
    return name


def _match_gru_name_by_json(base_name: str, json_keys: list[str]) -> str | None:
    """
    根据 activation_encodings 中的键名，为 GRU ONNX 节点推导最终公开命名。

    优先级：
      1. base_name 精确匹配（已经是正确名字）。
      2. base_name 是某个 json_keys 条目的前缀（json key 更具体）。
      3. 当 base_name 以 ".gru" 结尾时（ExportOptimizedQuantizableGRU 的内部
         self.gru = nn.GRU(...) 产生的实现细节后缀）：
         a. 先检查父模块路径（strip ".gru"）是否已在 json_keys（新格式，
            QuantGRU 路径，activation_encodings key 就是 module_path）。
         b. 再检查 parent.cells.0 / parent.cells.N（旧 OptimizableGRU 格式）。
         c. 无论如何，用父模块路径作为最终名（去掉 ".gru" 实现细节后缀），
            避免与 AIMET 给同名模块下其他非 GRU 节点（如 Transpose）重名。
    """
    if base_name in json_keys:
        return base_name
    prefix = base_name + "."
    candidates = [k for k in json_keys if k.startswith(prefix)]
    if candidates:
        candidates.sort(key=len)
        return candidates[0]

    # 单向 GRU：ExportOptimizedQuantizableGRU 用 self.gru = nn.GRU(...)，
    # AIMET 会同时将 Transpose 和 GRU op 都命名为 "...seq_t.gru"（depth=0），
    # 导致重名。解决方案：把 GRU op 收敛到父模块路径（去掉 ".gru" 后缀），
    # 与 activation_encodings 中的 module-path 风格键名一致。
    if base_name.endswith(".gru"):
        parent = base_name[: -len(".gru")]

        # 优先：父模块路径已作为激活量化键（QuantGRU / 后处理后的新格式）
        if parent in json_keys:
            return parent

        # 其次：旧格式，activation_encodings 键名为 parent.cells.0
        cells_key = parent + ".cells.0"
        if cells_key in json_keys:
            return cells_key
        cells_prefix = parent + ".cells."
        cells_candidates = [k for k in json_keys if k.startswith(cells_prefix)]
        if cells_candidates:
            cells_candidates.sort(key=len)
            return cells_candidates[0]

        # 兜底：无论 encodings 中是否有对应键，都去掉 ".gru" 后缀，
        # 消除与同模块其他非 GRU ONNX 节点（Transpose 等）的重名冲突。
        return parent

    return None


def _check_gru_name_collisions(model: onnx.ModelProto, act_keys: List[str]) -> None:
    """
    扫描所有 GRU op 节点，提前检测规范化后的节点名是否会产生冲突。
    如果有两个不同的 GRU 节点规范化到同一个名字，抛出带上下文的错误。
    """
    from collections import defaultdict
    canonical_to_raw: Dict[str, List[str]] = defaultdict(list)

    for node in model.graph.node:
        if node.op_type != "GRU":
            continue
        base_name = _strip_hash_suffix(node.name)
        matched = _match_gru_name_by_json(base_name, act_keys)
        canonical = matched if matched is not None else base_name
        canonical_to_raw[canonical].append(node.name)

    collisions = {c: raws for c, raws in canonical_to_raw.items() if len(raws) > 1}
    if collisions:
        details = "; ".join(
            f"'{canonical}' <- {raws}" for canonical, raws in collisions.items()
        )
        raise RuntimeError(
            f"GRU 节点命名冲突：{len(collisions)} 组 GRU 节点规范化后同名。"
            f"冲突详情: {details}。请检查模型结构或 _match_gru_name_by_json 逻辑。"
        )


def _build_gru_public_name_map(model: onnx.ModelProto, act_keys: List[str]) -> Dict[int, str]:
    """为每个 GRU 节点计算最终公开名，并确保不同 GRU 之间不会同名。"""
    from collections import defaultdict

    idx_to_name: Dict[int, str] = {}
    canonical_to_raw: Dict[str, List[str]] = defaultdict(list)

    for idx, node in enumerate(model.graph.node):
        if node.op_type != "GRU":
            continue
        base_name = _strip_hash_suffix(node.name)
        matched = _match_gru_name_by_json(base_name, act_keys)
        canonical = matched if matched is not None else base_name
        idx_to_name[idx] = canonical
        canonical_to_raw[canonical].append(node.name)

    collisions = {c: raws for c, raws in canonical_to_raw.items() if len(raws) > 1}
    if collisions:
        details = "; ".join(
            f"'{canonical}' <- {raws}" for canonical, raws in collisions.items()
        )
        raise RuntimeError(
            f"GRU 节点命名冲突：{len(collisions)} 组 GRU 节点规范化后同名。"
            f"冲突详情: {details}。请检查模型结构或 _match_gru_name_by_json 逻辑。"
        )
    return idx_to_name


def _make_unique_name(preferred: str, used_names: set[str], reserved_names: set[str]) -> str:
    if preferred not in used_names and preferred not in reserved_names:
        return preferred

    suffix = 1
    while True:
        candidate = f"{preferred}_{suffix}"
        if candidate not in used_names and candidate not in reserved_names:
            return candidate
        suffix += 1


def _rename_non_gru_nodes_conflicting_with_gru_names(
    model: onnx.ModelProto,
    gru_public_names: set[str],
    verbose: bool = True,
) -> bool:
    """
    若非 GRU 节点已占用目标 GRU 名字，则优先重命名这些非 GRU 节点。
    这样可以保留 GRU 节点与 activation_encodings key 的对应关系。
    """
    renamed = False
    used_names = {node.name for node in model.graph.node if node.name}

    for node in model.graph.node:
        if node.op_type == "GRU" or not node.name:
            continue
        if node.name not in gru_public_names:
            continue

        preferred = f"{node.name}.{node.op_type.lower()}"
        used_names.discard(node.name)
        new_name = _make_unique_name(preferred, used_names, gru_public_names)
        if verbose:
            logger.info(
                "重命名非 GRU 冲突节点: %s (%s) -> %s",
                node.name,
                node.op_type,
                new_name,
            )
        node.name = new_name
        used_names.add(new_name)
        renamed = True

    return renamed


def _check_named_node_collisions(model: onnx.ModelProto) -> None:
    """检查最终 ONNX 中是否仍有非空节点名重复。"""
    from collections import Counter

    counts = Counter(node.name for node in model.graph.node if node.name)
    collisions = {name: count for name, count in counts.items() if count > 1}
    if collisions:
        details = "; ".join(f"'{name}' x{count}" for name, count in collisions.items())
        raise RuntimeError(f"后处理后仍存在重复节点名: {details}")


def _rename_initializer_and_refs(model: onnx.ModelProto, old_name: str, new_name: str) -> bool:
    if old_name == new_name:
        return False
    renamed = False
    for init in model.graph.initializer:
        if init.name == old_name:
            init.name = new_name
            renamed = True
            break
    for inp in model.graph.input:
        if inp.name == old_name:
            inp.name = new_name
            renamed = True
    if renamed:
        _rename_value_references(model, old_name, new_name)
    return renamed


def _rename_param_encoding(enc: Dict[str, Any], old: str, new: str) -> bool:
    if "param_encodings" not in enc or not isinstance(enc["param_encodings"], dict):
        return False
    pe = enc["param_encodings"]
    if old not in pe or old == new:
        return False
    pe[new] = pe.pop(old)
    return True


def rename_gru_initializers_and_change_gru_json_format(
    onnx_path: str,
    encodings_path: str,
    output_onnx_path: str | None = None,
    output_encodings_path: str | None = None,
    verbose: bool = True,
) -> Tuple[str, str]:
    enc_out = output_encodings_path or encodings_path
    enc = load_encodings(encodings_path)
    act_keys = list(enc.get("activation_encodings", {}).keys()) if isinstance(enc.get("activation_encodings"), dict) else []

    onnx_out = output_onnx_path or onnx_path.replace(".onnx", "_renamed.onnx")
    try:
        model = onnx.load(onnx_path)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"无法加载 ONNX: {exc}") from exc

    # 提前计算 GRU 公开名；若不同 GRU 规范化后同名，失败快速
    gru_name_map = _build_gru_public_name_map(model, act_keys)
    non_gru_renamed = _rename_non_gru_nodes_conflicting_with_gru_names(
        model,
        set(gru_name_map.values()),
        verbose=verbose,
    )

    gru_renamed = False
    param_renamed = False
    for idx, node in enumerate(model.graph.node):
        if node.op_type == "GRU":
            new_name = gru_name_map[idx]
            if new_name != node.name:
                if verbose:
                    logger.info("重命名 GRU 节点: %s -> %s", node.name, new_name)
                node.name = new_name
                gru_renamed = True

            inputs = list(node.input)
            if len(inputs) >= 3:
                weight_ih_old = inputs[1]
                weight_hh_old = inputs[2]
                weight_ih_new = f"{node.name}.weight_ih.weight"
                weight_hh_new = f"{node.name}.weight_hh.weight"
                if _rename_initializer_and_refs(model, weight_ih_old, weight_ih_new):
                    if verbose:
                        logger.info("重命名 GRU weight_ih: %s -> %s", weight_ih_old, weight_ih_new)
                    _rename_param_encoding(enc, weight_ih_old, weight_ih_new)
                    inputs[1] = weight_ih_new
                    param_renamed = True
                if _rename_initializer_and_refs(model, weight_hh_old, weight_hh_new):
                    if verbose:
                        logger.info("重命名 GRU weight_hh: %s -> %s", weight_hh_old, weight_hh_new)
                    _rename_param_encoding(enc, weight_hh_old, weight_hh_new)
                    inputs[2] = weight_hh_new
                    param_renamed = True
            if len(inputs) >= 4:
                bias_old = inputs[3]
                bias_new = f"{node.name}.bias"
                if _rename_initializer_and_refs(model, bias_old, bias_new):
                    if verbose:
                        logger.info("重命名 GRU bias: %s -> %s", bias_old, bias_new)
                    _rename_param_encoding(enc, bias_old, bias_new)
                    inputs[3] = bias_new
                    param_renamed = True
            node.input[:] = inputs

        if node.op_type == "Conv" and node.name:
            inputs = list(node.input)
            if len(inputs) >= 2:
                w_old = inputs[1]
                w_new = f"{node.name}.weight"
                if _rename_initializer_and_refs(model, w_old, w_new):
                    if verbose:
                        logger.info("重命名 Conv weight: %s -> %s", w_old, w_new)
                    _rename_param_encoding(enc, w_old, w_new)
                    inputs[1] = w_new
                    param_renamed = True
            if len(inputs) >= 3:
                b_old = inputs[2]
                b_new = f"{node.name}.bias"
                if _rename_initializer_and_refs(model, b_old, b_new):
                    if verbose:
                        logger.info("重命名 Conv bias: %s -> %s", b_old, b_new)
                    _rename_param_encoding(enc, b_old, b_new)
                    inputs[2] = b_new
                    param_renamed = True
            node.input[:] = inputs

    if non_gru_renamed or gru_renamed or param_renamed or onnx_out != onnx_path:
        _check_named_node_collisions(model)
        onnx.save(model, onnx_out)
    else:
        onnx_out = onnx_path

    if param_renamed or enc_out != encodings_path:
        save_encodings(enc, enc_out)
    else:
        enc_out = encodings_path

    if verbose:
        logger.info("ONNX 输出: %s", onnx_out)
        logger.info("Encodings 输出: %s", enc_out)

    return onnx_out, enc_out

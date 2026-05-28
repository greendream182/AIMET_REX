from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
from onnx import numpy_helper, shape_inference
from onnxsim import simplify

logger = logging.getLogger(__name__)


def _get_static_shape_tuple(name: str, graph) -> Optional[Tuple[int, ...]]:
    """读取 value_info/input/output 中的静态 shape（若有动态 dim 则返回 None）。"""

    def _scan(vis):
        for vi in vis:
            if vi.name != name:
                continue
            if not vi.type.HasField("tensor_type"):
                return None
            tt = vi.type.tensor_type
            if not tt.HasField("shape"):
                return None
            dims = []
            for d in tt.shape.dim:
                if d.HasField("dim_value"):
                    dims.append(int(d.dim_value))
                else:
                    return None
            return tuple(dims)
        return None

    return _scan(graph.value_info) or _scan(graph.output) or _scan(graph.input)


def _fold_static_reshape_shapes(model: onnx.ModelProto) -> onnx.ModelProto:
    """把 Reshape 的第二输入（shape）子图尽可能折叠为静态 initializer。"""
    g = model.graph
    produced_by = {o: n for n in g.node for o in n.output}
    init_map = {init.name: numpy_helper.to_array(init) for init in g.initializer}
    memo: Dict[str, Any] = {}

    def _const_from_node(node):
        for a in node.attribute:
            if a.name == "value":
                return numpy_helper.to_array(a.t)
        return None

    def _tensor_shape(name: str) -> Optional[Tuple[int, ...]]:
        if name in init_map:
            return tuple(init_map[name].shape)
        return _get_static_shape_tuple(name, g)

    def _eval(name):
        if isinstance(name, int):
            return name
        if name in memo:
            return memo[name]
        if name in init_map:
            arr = init_map[name]
            memo[name] = int(arr) if arr.ndim == 0 else [int(x) for x in arr.flatten().tolist()]
            return memo[name]
        node = produced_by.get(name)
        if node is None:
            memo[name] = None
            return None
        op = node.op_type
        if op == "Shape":
            shp = _tensor_shape(node.input[0])
            memo[name] = list(shp) if shp is not None else None
        elif op == "Gather":
            data = _eval(node.input[0])
            idx = _eval(node.input[1])
            memo[name] = int(data[idx]) if isinstance(data, list) and isinstance(idx, int) else None
        elif op == "Unsqueeze":
            memo[name] = _eval(node.input[0])
        elif op == "Concat":
            out: List[int] = []
            for inp in node.input:
                v = _eval(inp)
                if isinstance(v, int):
                    out.append(v)
                elif isinstance(v, list):
                    out.extend(v)
                else:
                    memo[name] = None
                    return None
            memo[name] = out
        elif op == "Mul":
            a = _eval(node.input[0])
            b = _eval(node.input[1])
            memo[name] = int(a * b) if isinstance(a, int) and isinstance(b, int) else None
        elif op == "Constant":
            arr = _const_from_node(node)
            memo[name] = None if arr is None else (
                int(arr) if arr.ndim == 0 else [int(x) for x in arr.flatten().tolist()]
            )
        else:
            memo[name] = None
        return memo[name]

    existing_init = {init.name for init in g.initializer}
    to_add = []
    for node in g.node:
        if node.op_type != "Reshape" or len(node.input) < 2 or not node.output:
            continue
        shape_in = node.input[1]
        if shape_in in existing_init:
            continue
        shape_val = _eval(shape_in)
        if not isinstance(shape_val, list) or not shape_val:
            continue
        const_name = f"{node.output[0]}_shape_const"
        if const_name not in existing_init:
            to_add.append(numpy_helper.from_array(np.asarray(shape_val, dtype=np.int64), name=const_name))
            existing_init.add(const_name)
        node.input[1] = const_name
    if to_add:
        g.initializer.extend(to_add)
    return model


def _fold_constant_if_nodes(model: onnx.ModelProto) -> onnx.ModelProto:
    """折叠 If(cond) 且 cond 可被静态求值为 True/False 的分支。"""
    g = model.graph
    produced_by = {o: n for n in g.node for o in n.output}
    init_map = {init.name: numpy_helper.to_array(init) for init in g.initializer}
    memo: Dict[str, Any] = {}

    def _const_from_node(node):
        for a in node.attribute:
            if a.name == "value":
                return numpy_helper.to_array(a.t)
        return None

    def _tensor_shape(name: str) -> Optional[Tuple[int, ...]]:
        if name in init_map:
            return tuple(init_map[name].shape)
        return _get_static_shape_tuple(name, g)

    def _eval(name):
        if isinstance(name, bool):
            return bool(name)
        if isinstance(name, int):
            return int(name)
        if name in memo:
            return memo[name]
        if name in init_map:
            arr = init_map[name]
            memo[name] = int(arr) if arr.ndim == 0 else [int(x) for x in arr.flatten().tolist()]
            return memo[name]
        node = produced_by.get(name)
        if node is None:
            memo[name] = None
            return None
        op = node.op_type
        if op == "Constant":
            arr = _const_from_node(node)
            memo[name] = None if arr is None else (
                int(arr) if arr.ndim == 0 else [int(x) for x in arr.flatten().tolist()]
            )
        elif op == "Shape":
            shp = _tensor_shape(node.input[0])
            memo[name] = list(shp) if shp is not None else None
        elif op == "Gather":
            data = _eval(node.input[0])
            idx = _eval(node.input[1])
            memo[name] = int(data[idx]) if isinstance(data, list) and isinstance(idx, int) else None
        elif op == "Equal":
            a = _eval(node.input[0])
            b = _eval(node.input[1])
            memo[name] = bool(a == b) if isinstance(a, int) and isinstance(b, int) else None
        elif op == "Cast":
            v = _eval(node.input[0])
            memo[name] = v if isinstance(v, (bool, int, list)) else None
        elif op == "Identity":
            memo[name] = _eval(node.input[0])
        else:
            memo[name] = None
        return memo[name]

    def _get_attr_graph(node, name: str):
        for a in node.attribute:
            if a.name == name:
                return a.g
        return None

    new_nodes = []
    removed = set()
    for i, node in enumerate(list(g.node)):
        if node.op_type != "If" or not node.input:
            continue
        cond = _eval(node.input[0])
        if not isinstance(cond, bool):
            continue
        chosen = _get_attr_graph(node, "then_branch" if cond else "else_branch")
        if chosen is None:
            continue
        prefix = (node.name or f"If_{i}") + ("/then" if cond else "/else")
        rename: Dict[str, str] = {}
        for bn in chosen.node:
            for o in bn.output:
                rename[o] = f"{prefix}/{o}"
        for o in chosen.output:
            rename[o.name] = f"{prefix}/{o.name}"
        import copy as _copy

        for bn in chosen.node:
            nn_ = _copy.deepcopy(bn)
            nn_.output[:] = [rename.get(o, o) for o in bn.output]
            nn_.input[:] = [rename.get(i_, i_) for i_ in bn.input]
            new_nodes.append(nn_)
        for j, out_name in enumerate(node.output):
            if j >= len(chosen.output):
                break
            br_old = chosen.output[j].name
            br_new = rename.get(br_old, br_old)
            new_nodes.append(
                onnx.helper.make_node("Identity", [br_new], [out_name], name=f"{prefix}/Identity_{j}")
            )
        removed.add(id(node))

    if not removed:
        return model

    kept = [n for n in g.node if id(n) not in removed]
    g.ClearField("node")
    g.node.extend(kept + new_nodes)
    return model


def postprocess_onnx_graph_inplace(onnx_path: str, input_shape: Tuple[int, int, int]) -> None:
    """
    对导出的 ONNX 执行强制图清理（inplace 写回原文件）。
    """

    def _make_unique_gru_initial_h_names_inplace(model: onnx.ModelProto) -> onnx.ModelProto:
        """
        让每个 GRU 的 initial_h 输入拥有独立的名字：<gru_node_name>.initial_h。
        """
        g = model.graph

        existing_names = set()
        for vi in list(g.value_info) + list(g.input) + list(g.output):
            existing_names.add(vi.name)
        for init in g.initializer:
            existing_names.add(init.name)
        for n in g.node:
            for x in n.input:
                if x:
                    existing_names.add(x)
            for y in n.output:
                if y:
                    existing_names.add(y)

        produced_by: Dict[str, int] = {}
        for i, n in enumerate(g.node):
            for o in n.output:
                if o:
                    produced_by[o] = i

        consumers: Dict[str, int] = {}
        for n in g.node:
            for x in n.input:
                if not x:
                    continue
                consumers[x] = consumers.get(x, 0) + 1

        init_map = {init.name: init for init in g.initializer}
        gru_need: Dict[int, Tuple[str, str, str]] = {}

        for idx, node in enumerate(g.node):
            if node.op_type != "GRU":
                continue
            if len(node.input) < 6:
                continue
            h_in = node.input[5]
            if not h_in:
                continue

            base = f"{node.name or f'GRU_{idx}'}.initial_h"
            new_name = base
            k = 0
            while new_name in existing_names:
                k += 1
                new_name = f"{base}_{k}"

            only_this_consumer = consumers.get(h_in, 0) == 1
            if only_this_consumer:
                if h_in in init_map or h_in in produced_by:
                    gru_need[idx] = (h_in, new_name, "rename_source")
                else:
                    gru_need[idx] = (h_in, new_name, "skip")
                continue

            if h_in in init_map:
                gru_need[idx] = (h_in, new_name, "clone_init")
            elif h_in in produced_by:
                gru_need[idx] = (h_in, new_name, "clone_node")
            else:
                gru_need[idx] = (h_in, new_name, "skip")

        if not gru_need:
            return model

        import copy as _copy

        old_nodes = list(g.node)
        new_nodes: List[onnx.NodeProto] = []

        def _safe_unique(name: str) -> str:
            if name not in existing_names:
                existing_names.add(name)
                return name
            k = 0
            cand = name
            while cand in existing_names:
                k += 1
                cand = f"{name}_{k}"
            existing_names.add(cand)
            return cand

        for _, (old_h, new_h, action) in list(gru_need.items()):
            if action != "rename_source":
                continue
            if old_h in init_map:
                init_map[old_h].name = new_h
                for n in old_nodes:
                    n.input[:] = [new_h if x == old_h else x for x in n.input]
                for vi in list(g.value_info) + list(g.input) + list(g.output):
                    if vi.name == old_h:
                        vi.name = new_h
                existing_names.add(new_h)
                continue

            prod_i = produced_by.get(old_h)
            if prod_i is not None:
                prod = old_nodes[prod_i]
                prod.output[:] = [new_h if o == old_h else o for o in prod.output]
                for n in old_nodes:
                    n.input[:] = [new_h if x == old_h else x for x in n.input]
                for vi in list(g.value_info) + list(g.input) + list(g.output):
                    if vi.name == old_h:
                        vi.name = new_h
                existing_names.add(new_h)
                continue

        for i, node in enumerate(old_nodes):
            need = gru_need.get(i)
            if node.op_type == "GRU" and need is not None:
                old_h, new_h, action = need
                cur_h = node.input[5] if len(node.input) >= 6 else old_h

                if action == "clone_init":
                    init = init_map.get(cur_h)
                    if init is not None:
                        new_init = _copy.deepcopy(init)
                        new_init.name = _safe_unique(new_h)
                        g.initializer.extend([new_init])
                        node.input[5] = new_init.name
                elif action == "clone_node":
                    prod_i = produced_by.get(cur_h)
                    if prod_i is not None:
                        prod = old_nodes[prod_i]
                        prod_clone = _copy.deepcopy(prod)
                        prod_clone.name = _safe_unique(f"{new_h}_producer")
                        out_new = []
                        for j, o in enumerate(prod_clone.output):
                            if o == cur_h:
                                out_new.append(_safe_unique(new_h))
                            else:
                                out_new.append(_safe_unique(f"{new_h}_aux{j}"))
                        prod_clone.output[:] = out_new
                        node.input[5] = out_new[0]
                        new_nodes.append(prod_clone)

            new_nodes.append(node)

        g.ClearField("node")
        g.node.extend(new_nodes)
        return model

    m = onnx.load(onnx_path)

    def _deduplicate_initializers(model):
        """去重 ONNX 图中的 initializer，重命名重复的 initializer 并更新引用。"""
        g = model.graph
        seen_names = {}
        rename_map = {}
        unique_inits = []

        for init in g.initializer:
            if init.name in seen_names:
                count = seen_names[init.name][1] + 1
                seen_names[init.name] = (seen_names[init.name][0], count)
                new_name = f"{init.name}_dup_{count}"
                old_name = init.name
                init.name = new_name
                rename_map[old_name] = new_name
                print(f"⚠️ 重复的 initializer：{old_name} -> {new_name}")
            else:
                seen_names[init.name] = (init, 0)
            unique_inits.append(init)

        if rename_map:
            for node in g.node:
                new_inputs = []
                for inp in node.input:
                    new_inputs.append(rename_map.get(inp, inp))
                node.ClearField("input")
                node.input.extend(new_inputs)

            g.ClearField("initializer")
            g.initializer.extend(unique_inits)
        return model

    m = _deduplicate_initializers(m)

    m, ok = simplify(m, overwrite_input_shapes={"input": list(input_shape)})
    if not ok:
        raise RuntimeError("onnx-simplifier 校验失败")
    m = shape_inference.infer_shapes(m, data_prop=True)
    m = _fold_static_reshape_shapes(m)
    m = _fold_constant_if_nodes(m)

    m, ok = simplify(m, overwrite_input_shapes={"input": list(input_shape)})
    if not ok:
        raise RuntimeError("onnx-simplifier（二次）校验失败")
    m = shape_inference.infer_shapes(m, data_prop=True)

    m = _fold_constant_if_nodes(m)
    m, ok = simplify(m, overwrite_input_shapes={"input": list(input_shape)})
    if not ok:
        raise RuntimeError("onnx-simplifier（三次）校验失败")
    m = shape_inference.infer_shapes(m, data_prop=True)

    m = _make_unique_gru_initial_h_names_inplace(m)
    onnx.save(m, onnx_path)


def _get_shape_from_value_info(value_info) -> list:
    """从 ONNX ValueInfoProto 中提取 shape 信息（包含动态维度）。"""
    dims = []
    tensor_type = value_info.type.tensor_type
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        else:
            dims.append(None)
    return dims


def fix_slice_end_constants(onnx_path: str, verbose: bool = True) -> bool:
    """
    遍历 ONNX 中的 Slice 节点，将 `INT64_MAX` 形式的 ends 改写为真实维度。
    """
    try:
        model = onnx.load(onnx_path)
    except Exception as exc:  # noqa: BLE001
        if verbose:
            logger.warning("无法加载ONNX进行Slice修正: %s", exc)
        return False

    try:
        inferred = shape_inference.infer_shapes(model)
    except Exception as exc:  # noqa: BLE001
        inferred = None
        if verbose:
            logger.warning("Slice修正时形状推断失败: %s", exc)

    value_info_map: Dict[str, onnx.ValueInfoProto] = {}
    if inferred is not None:
        for vi in list(inferred.graph.value_info) + list(inferred.graph.output) + list(inferred.graph.input):
            value_info_map[vi.name] = vi

    init_map = {init.name: init for init in model.graph.initializer}
    int64_max = np.iinfo(np.int64).max
    changed = False

    for node in model.graph.node:
        if node.op_type != "Slice" or len(node.input) < 3:
            continue

        data_name = node.input[0]
        starts_name = node.input[1]
        ends_name = node.input[2]
        axes_name = node.input[3] if len(node.input) >= 4 else None
        steps_name = node.input[4] if len(node.input) >= 5 else None

        if ends_name not in init_map:
            continue

        ends_arr = numpy_helper.to_array(init_map[ends_name]).astype(np.int64)
        if axes_name and axes_name in init_map:
            axes_arr = numpy_helper.to_array(init_map[axes_name]).astype(np.int64)
        else:
            axes_arr = np.arange(len(ends_arr), dtype=np.int64)

        starts_arr = numpy_helper.to_array(init_map[starts_name]).astype(np.int64) if starts_name in init_map else None

        if steps_name and steps_name in init_map:
            steps_arr = numpy_helper.to_array(init_map[steps_name]).astype(np.int64)
        else:
            steps_arr = None

        vi = value_info_map.get(data_name)
        if vi is None:
            continue
        data_shape = _get_shape_from_value_info(vi)

        updated = False
        for idx, end_val in enumerate(ends_arr):
            if end_val != int64_max or idx >= len(axes_arr):
                continue

            axis = int(axes_arr[idx])
            if axis < 0 or axis >= len(data_shape):
                continue

            dim = data_shape[axis]
            if dim is None:
                continue

            start_val = 0
            if starts_arr is not None and idx < len(starts_arr):
                start_val = int(starts_arr[idx])
                if start_val < 0:
                    start_val = dim + start_val

            step_val = 1
            if steps_arr is not None and idx < len(steps_arr):
                step_val = int(steps_arr[idx])
                if step_val <= 0:
                    continue

            new_end = dim if step_val > 0 else -1
            if new_end <= start_val:
                new_end = start_val

            ends_arr[idx] = new_end
            updated = True

        if updated:
            new_tensor = numpy_helper.from_array(ends_arr.astype(np.int64), ends_name)
            init_index = next(i for i, init in enumerate(model.graph.initializer) if init.name == ends_name)
            model.graph.initializer[init_index].CopyFrom(new_tensor)
            init_map[ends_name] = model.graph.initializer[init_index]
            changed = True
            if verbose:
                logger.info("✅ 调整Slice %s 的ends: %s", node.name, ends_arr.tolist())

    if changed:
        onnx.save(model, onnx_path)

    return changed

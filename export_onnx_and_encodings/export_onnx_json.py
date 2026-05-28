"""
从 AIMET QuantizationSimModel 导出 **干净** ONNX + encodings（sidecar）。

核心特性：
- 使用 AIMET 的导出链路（节点名/张量名尽量与 encodings 对齐）
- 将 `OptimizedQuantizableGRU` 替换为单节点导出（导出为标准 ONNX `GRU` 节点）
- ONNX 图不包含 Q/DQ/fakequant（encodings 单独文件）
- 导出后强制执行：onnx-simplifier + 静态 shape inference + 折叠 Reshape shape 子图 + 折叠常量 If
"""

from __future__ import annotations

from typing import Optional, Tuple
import os
import shutil
from collections import Counter

import onnx
import torch
import torch.nn as nn
from .custom_gru_onnx import (
    ensure_custom_gru_op_registered,
    replace_optimized_gru_modules,
)
from .custom_bn_onnx import replace_quantizable_batchnorm_modules
from .postprocess_refactor import postprocess_all

# 静态输入（你要求固定）
STATIC_DUMMY_INPUT_SHAPE: Tuple[int, int, int] = (1, 16000, 1)


# 尝试导入 QuantGRU（如果已通过 pip install 安装，可以直接导入）,因为不一定使用了Qua
try:
    from quant_gru import QuantGRU
except ImportError:
    QuantGRU = None  # 如果未安装，设为 None
    

from aimet_torch.optimized_quantizable_gru import OptimizedQuantizableGRU



# ---------- Internal helpers ----------
def _is_truthy_env(name: str) -> bool:
    v = os.getenv(name, "").strip()
    return v not in ("", "0", "false", "False", "FALSE", "no", "No", "NO")


def _replace_quantgru_with_optimized(module: nn.Module, parent_name: str = "") -> list:
    """
    递归地将 QuantGRU 替换为 OptimizedQuantizableGRU

    QuantGRU 是 CUDA 实现，导出 ONNX 需要临时替换为 CPU 兼容版本。

    Returns:
        replacements: [(full_name, original_module, replacement), ...]
    """
    if QuantGRU is None or OptimizedQuantizableGRU is None:
        return []

    replacements = []
    for name, child in list(module.named_children()):
        full_name = f"{parent_name}.{name}" if parent_name else name
        if isinstance(child, QuantGRU):
            replacement = OptimizedQuantizableGRU(
                input_size=child.input_size,
                hidden_size=child.hidden_size,
                batch_first=child.batch_first,
                num_layers=child.num_layers,
                bidirectional=child.bidirectional,
            )

            # 从 QuantGRU state_dict 迁移权重到 OptimizedQuantizableGRU
            try:
                child_state = child.state_dict()

                weight_ih = child_state.get("weight_ih_l0")
                if weight_ih is None:
                    weight_ih = child_state.get("_weight_ih_l0")

                weight_hh = child_state.get("weight_hh_l0")
                if weight_hh is None:
                    weight_hh = child_state.get("_weight_hh_l0")

                bias_ih = child_state.get("bias_ih_l0")
                if bias_ih is None:
                    bias_ih = child_state.get("_bias_ih_l0")

                bias_hh = child_state.get("bias_hh_l0")
                if bias_hh is None:
                    bias_hh = child_state.get("_bias_hh_l0")

                weight_ih_reverse = child_state.get("weight_ih_l0_reverse")
                if weight_ih_reverse is None:
                    weight_ih_reverse = child_state.get("_weight_ih_l0_reverse")

                weight_hh_reverse = child_state.get("weight_hh_l0_reverse")
                if weight_hh_reverse is None:
                    weight_hh_reverse = child_state.get("_weight_hh_l0_reverse")

                bias_ih_reverse = child_state.get("bias_ih_l0_reverse")
                if bias_ih_reverse is None:
                    bias_ih_reverse = child_state.get("_bias_ih_l0_reverse")

                bias_hh_reverse = child_state.get("bias_hh_l0_reverse")
                if bias_hh_reverse is None:
                    bias_hh_reverse = child_state.get("_bias_hh_l0_reverse")

                if weight_ih is not None:
                    replacement.cells[0].weight_ih.weight.data = weight_ih.cpu().clone()
                if weight_hh is not None:
                    replacement.cells[0].weight_hh.weight.data = weight_hh.cpu().clone()
                if bias_ih is not None:
                    replacement.cells[0].weight_ih.bias.data = bias_ih.cpu().clone()
                if bias_hh is not None:
                    replacement.cells[0].weight_hh.bias.data = bias_hh.cpu().clone()

                if child.bidirectional and replacement.reverse_cells is not None:
                    if weight_ih_reverse is not None:
                        replacement.reverse_cells[0].weight_ih.weight.data = weight_ih_reverse.cpu().clone()
                    if weight_hh_reverse is not None:
                        replacement.reverse_cells[0].weight_hh.weight.data = weight_hh_reverse.cpu().clone()
                    if bias_ih_reverse is not None:
                        replacement.reverse_cells[0].weight_ih.bias.data = bias_ih_reverse.cpu().clone()
                    if bias_hh_reverse is not None:
                        replacement.reverse_cells[0].weight_hh.bias.data = bias_hh_reverse.cpu().clone()

                print(f"    ✅ {full_name}: 权重已复制")
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(
                    f"无法复制 QuantGRU 模块 {full_name} 的权重，已终止导出。"
                ) from e

            setattr(module, name, replacement)
            replacements.append((full_name, child, replacement))
            print(f"  ✅ {full_name}: QuantGRU -> OptimizedQuantizableGRU")
        else:
            replacements.extend(_replace_quantgru_with_optimized(child, full_name))
    return replacements


def _restore_quantgru_modules(root_module: nn.Module, quant_gru_replacements: list) -> None:
    """将 _replace_quantgru_with_optimized 的替换恢复回原 QuantGRU 模块。"""
    if not quant_gru_replacements:
        return

    print("\n♻️ 恢复 QuantGRU 模块...")
    for full_name, original_module, _ in quant_gru_replacements:
        parts = full_name.split(".")
        parent = root_module
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], original_module)
        print(f"  ✅ {full_name}: OptimizedQuantizableGRU -> QuantGRU")


def _pick_encodings_input_path(export_dir: str, filename_prefix: str) -> str:
    """
    选择 AIMET 导出的 encodings 输入文件路径：
    1) 优先 *_torch.encodings*（更接近 module-name 的层级格式，利于后处理）
    2) 其次是 {prefix}.encodings / {prefix}.json
    3) 再不行就挑 export_dir 下最新修改的 encodings/json
    """
    torch_cand = [
        os.path.join(export_dir, filename_prefix + "_torch.encodings"),
        os.path.join(export_dir, filename_prefix + "_torch.encodings.json"),
        os.path.join(export_dir, filename_prefix + "_torch.json"),
    ]
    enc_in_path = next((p for p in torch_cand if os.path.exists(p)), "")
    if enc_in_path:
        return enc_in_path

    cand = [
        os.path.join(export_dir, filename_prefix + ".encodings"),
        os.path.join(export_dir, filename_prefix + ".encodings.json"),
        os.path.join(export_dir, filename_prefix + ".json"),
    ]
    enc_in_path = next((p for p in cand if os.path.exists(p)), "")
    if enc_in_path:
        return enc_in_path

    files = [os.path.join(export_dir, f) for f in os.listdir(export_dir)]
    files = [
        f
        for f in files
        if f.endswith(".encodings") or f.endswith(".encodings.json") or f.endswith(".json")
    ]
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0] if files else ""


def _maybe_keep_raw_exports(*, export_dir: str, filename_prefix: str, onnx_path: str, enc_in_path: str) -> None:
    """
    Debug：保留 raw 导出物（用于判断后处理/constant folding 是否把某些节点折叠掉）。
    用法：AIMET_EXPORT_KEEP_RAW=1 python your_script.py
    """
    if not _is_truthy_env("AIMET_EXPORT_KEEP_RAW"):
        return

    raw_onnx = os.path.join(export_dir, filename_prefix + "_raw.onnx")
    try:
        shutil.copyfile(onnx_path, raw_onnx)
        print(f"🧪 已保留 raw ONNX: {raw_onnx}")
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ 保留 raw ONNX 失败：{e}")

    try:
        base, ext = os.path.splitext(enc_in_path)
        # 兼容 .encodings / .json / .encodings.json 之类
        ext2 = ""
        if base.endswith(".encodings") and ext == ".json":
            ext2 = ".encodings.json"
        raw_enc = os.path.join(export_dir, filename_prefix + "_raw" + (ext2 or ext or ".encodings"))
        shutil.copyfile(enc_in_path, raw_enc)
        print(f"🧪 已保留 raw encodings: {raw_enc}")
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ 保留 raw encodings 失败：{e}")


def _postprocess_with_retry(
    *,
    onnx_path: str,
    enc_in_path: str,
    enc_out_path: str,
    input_shape: Tuple[int, int, int],
) -> None:
    def _postprocess_and_verify(tag: str) -> None:
        postprocess_all(
            onnx_path=onnx_path,
            encodings_path=enc_in_path,
            input_shape=input_shape,
            output_onnx_path=onnx_path,
            output_encodings_path=enc_out_path,
            verbose=True,
        )
        mm = onnx.load(onnx_path)
        ops_cnt = Counter([(n.domain, n.op_type) for n in mm.graph.node])
        print(
            f"✅ postprocess[{tag}] ops: "
            f"GRU={ops_cnt.get(('', 'GRU'), 0)}, "
            f"BatchNormalization={ops_cnt.get(('', 'BatchNormalization'), 0)}, "
            f"If={ops_cnt.get(('', 'If'), 0)}, "
            f"Shape={ops_cnt.get(('', 'Shape'), 0)}, "
            f"Gather={ops_cnt.get(('', 'Gather'), 0)}"
        )

    try:
        _postprocess_and_verify("first")
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ 第一次后处理失败：{e}，尝试再次后处理…")
        _postprocess_and_verify("second")


# ---------- Public API ----------
def export_onnx_json(
    sim,
    export_dir: str,
    filename_prefix: str,
    dummy_input_shape: Tuple[int, int, int] = STATIC_DUMMY_INPUT_SHAPE,
    opset: int = 18,
    export_int16_sidecar: bool = True,
) -> Tuple[str, str, Optional[str]]:
    """
    从 AIMET sim 导出 ONNX + encodings，并强制后处理 ONNX（去除辅助节点/If 并补齐静态 shape）。

    - ONNX 保持“干净”：不导出 Q/DQ/fakequant（固定为 sidecar encodings）
    - torch.onnx.export 强制 legacy：dynamo=False
    - 可选写出 ``{filename_prefix}.int16.json``（INT16 fixed-point sidecar，与 AIMET encodings 对齐）
    """
    os.makedirs(export_dir, exist_ok=True)
    # if tuple(dummy_input_shape) != STATIC_DUMMY_INPUT_SHAPE:
    #     raise ValueError(f"导出要求静态 dummy_input_shape={STATIC_DUMMY_INPUT_SHAPE}，当前为 {dummy_input_shape}")

    dummy_input = torch.zeros(*dummy_input_shape, device="cpu")
    ensure_custom_gru_op_registered(opset=opset)

    if not hasattr(sim, "get_original_model"):
        raise RuntimeError("sim 没有 get_original_model()，无法走 sim.export 链路")

    # 临时替换 QuantGRU 为 OptimizedQuantizableGRU（导出完成后会恢复）
    quant_gru_replacements = []
    if QuantGRU is not None:
        has_quantgru = any(isinstance(m, QuantGRU) for m in sim.model.modules())
        if has_quantgru:
            print("⚠️ 检测到 QuantGRU 模块，临时替换为 OptimizedQuantizableGRU 以便在 CPU 上导出 ONNX...")
            quant_gru_replacements = _replace_quantgru_with_optimized(sim.model)
            if quant_gru_replacements:
                print(f"✅ 共替换 {len(quant_gru_replacements)} 个 QuantGRU 模块")

    try:
        # -------------------------------------------------------------------------------------------------------------
        # 将 OptimizedQuantizableGRU 替换为 ExportOptimizedQuantizableGRU
        # -------------------------------------------------------------------------------------------------------------
        original_model = sim.get_original_model(sim.model, qdq_weights=False)
        original_model = original_model.cpu().eval()
        original_model = replace_optimized_gru_modules(original_model)
        original_model = replace_quantizable_batchnorm_modules(original_model)

        onnx_export_args = {
            "opset_version": opset,
            "input_names": ["input"],
            "output_names": ["output"],
            "do_constant_folding": True,
            # 关键：强制 legacy exporter，避免 dynamo/export 路径导致自定义 symbolic 不命中
            "dynamo": False,
            # 关键：告诉 exporter 这些自定义 domain 的 opset version
            "custom_opsets": {"custom_gru": 1, "custom_bn": 1},
        }

        onnx_path = os.path.join(export_dir, filename_prefix + ".onnx")
        sim.__class__.export_onnx_model_and_encodings(  # type: ignore[attr-defined]
            export_dir,
            filename_prefix,
            original_model,
            sim.model,
            dummy_input,
            onnx_export_args,
            propagate_encodings=False,
            module_marker_map=getattr(sim, "_module_marker_map", None),
            is_conditional=getattr(sim, "_is_conditional", False),
            excluded_layer_names=getattr(sim, "_excluded_layer_names", None),
            quantizer_args=getattr(sim, "quant_args", None),
            export_model=True,
            filename_prefix_encodings=filename_prefix,
        )

        # -------------------------------------------------------------------------------------------------------------
        # 后处理 ONNX（去除辅助节点/If 并补齐静态 shape）
        # -------------------------------------------------------------------------------------------------------------
        enc_in_path = _pick_encodings_input_path(export_dir, filename_prefix)
        if not enc_in_path:
            raise RuntimeError("未找到 AIMET 导出的 encodings 文件（.encodings/.json）")

        # 最终对外输出路径固定为 {prefix}.encodings（避免调用侧需要跟随 *_torch.encodings）
        enc_out_path = os.path.join(export_dir, filename_prefix + ".encodings")

        _maybe_keep_raw_exports(
            export_dir=export_dir,
            filename_prefix=filename_prefix,
            onnx_path=onnx_path,
            enc_in_path=enc_in_path,
        )
        _postprocess_with_retry(
            onnx_path=onnx_path,
            enc_in_path=enc_in_path,
            enc_out_path=enc_out_path,
            input_shape=dummy_input_shape,
        )

        # -------------------------------------------------------------------------------------------------------------
        # 从 ONNX 模型提取 GRU 节点名，并回填到原 QuantGRU 模块
        # -------------------------------------------------------------------------------------------------------------
        if quant_gru_replacements and QuantGRU is not None:
            try:
                onnx_model = onnx.load(onnx_path)
                gru_node_names = {}
                for node in onnx_model.graph.node:
                    if node.op_type == "GRU":
                        node_name = node.name
                        gru_node_names[node_name] = node_name

                print("\n📝 提取 AIMET 分配的 GRU 节点名称...")
                for full_name, original_module, _ in quant_gru_replacements:
                    onnx_name = gru_node_names.get(full_name, full_name)
                    original_module.aimet_onnx_name = onnx_name
                    print(f"  ✅ {full_name}: ONNX 名称 = {onnx_name}")
            except Exception as e:  # noqa: BLE001
                print(f"  ⚠️ 警告：无法提取 ONNX 节点名称: {e}")

        # -------------------------------------------------------------------------------------------------------------
        # 将原 QuantGRU 的量化参数补回到 AIMET encodings
        # -------------------------------------------------------------------------------------------------------------
        if quant_gru_replacements and QuantGRU is not None:
            try:
                import json

                with open(enc_out_path, "r", encoding="utf-8") as f:
                    aimet_encodings = json.load(f)

                print("\n📤 导出 QuantGRU 量化参数到 AIMET 格式...")
                for full_name, original_module, _ in quant_gru_replacements:
                    if isinstance(original_module, QuantGRU) and original_module.is_calibrated():
                        module_name = getattr(original_module, "aimet_onnx_name", None) or full_name
                        original_module.export_quant_params_to_aimet_format(
                            aimet_encodings,
                            module_name=module_name,
                            verbose=True,
                        )

                with open(enc_out_path, "w", encoding="utf-8") as f:
                    json.dump(aimet_encodings, f, indent=2, ensure_ascii=False)
                print(f"  ✅ 已保存更新后的 encodings 到 {enc_out_path}")
            except Exception as e:  # noqa: BLE001
                print(f"  ⚠️ 警告：导出 QuantGRU 量化参数失败: {e}")

        int16_path: Optional[str] = None
        if export_int16_sidecar:
            try:
                from aimet_torch.fixed_point.export import (
                    default_int16_sidecar_path,
                    export_int16_sidecar_json,
                )

                int16_path = default_int16_sidecar_path(export_dir, filename_prefix)
                export_int16_sidecar_json(
                    sim.model,
                    int16_path,
                    aimet_encoding_path=enc_out_path,
                )
                print(f"INT16 sidecar: {int16_path}")
            except Exception as e:  # noqa: BLE001
                print(f"⚠️ INT16 sidecar 导出失败（已跳过）: {e}")
                int16_path = None

        return onnx_path, enc_out_path, int16_path
    finally:
        _restore_quantgru_modules(sim.model, quant_gru_replacements)



"""
分阶段量化工具函数
支持按算子类型进行分阶段量化，包括保存和加载量化参数
"""
import torch
import json
import os
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path


def save_quantizer_encodings(
    sim_model,
    save_path: str,
    layer_types: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    保存指定层类型的量化参数到 JSON 文件
    
    Args:
        sim_model: AIMET QuantizationSimModel.model
        save_path: 保存路径
        layer_types: 要保存的层类型列表，例如 ["QuantizedConv2d", "QuantizedGRU"]
                    如果为 None，则保存所有量化层
        verbose: 是否打印详细信息
    
    Returns:
        保存的量化参数字典
    """
    encodings_dict = {}
    saved_count = 0
    
    for module_name, module in sim_model.named_modules():
        module_type = type(module).__name__
        
        # 检查是否匹配指定的层类型
        if layer_types is not None:
            if not any(layer_type in module_type for layer_type in layer_types):
                continue
        
        # 保存 param_quantizers (权重量化器)
        if hasattr(module, 'param_quantizers'):
            for param_name, quantizer in module.param_quantizers.items():
                if quantizer is not None and quantizer.is_initialized():
                    encoding = quantizer.get_encodings()
                    if encoding is not None:
                        key = f"{module_name}.param_quantizers.{param_name}"
                        encodings_dict[key] = {
                            'module_name': module_name,
                            'module_type': module_type,
                            'quantizer_type': 'param',
                            'param_name': param_name,
                            'scale': _tensor_to_list(encoding.scale),
                            'offset': _tensor_to_list(encoding.offset),
                            'qmin': int(encoding.qmin),
                            'qmax': int(encoding.qmax),
                            'symmetric': bool(encoding.symmetry),
                            'min': _tensor_to_list(encoding.min),
                            'max': _tensor_to_list(encoding.max),
                            'shape': list(encoding.scale.shape) if isinstance(encoding.scale, torch.Tensor) else [],
                        }
                        saved_count += 1
        
        # 保存 output_quantizers (输出量化器)
        if hasattr(module, 'output_quantizers'):
            for idx, quantizer in enumerate(module.output_quantizers):
                if quantizer is not None and quantizer.is_initialized():
                    encoding = quantizer.get_encodings()
                    if encoding is not None:
                        key = f"{module_name}.output_quantizers.{idx}"
                        encodings_dict[key] = {
                            'module_name': module_name,
                            'module_type': module_type,
                            'quantizer_type': 'output',
                            'quantizer_idx': idx,
                            'scale': _tensor_to_list(encoding.scale),
                            'offset': _tensor_to_list(encoding.offset),
                            'qmin': int(encoding.qmin),
                            'qmax': int(encoding.qmax),
                            'symmetric': bool(encoding.symmetry),
                            'min': _tensor_to_list(encoding.min),
                            'max': _tensor_to_list(encoding.max),
                            'shape': list(encoding.scale.shape) if isinstance(encoding.scale, torch.Tensor) else [],
                        }
                        saved_count += 1
    
    # 保存到文件
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(encodings_dict, f, indent=2, ensure_ascii=False)
    
    if verbose:
        layer_type_str = ', '.join(layer_types) if layer_types else 'all'
        print(f"\n✅ 保存量化参数:")
        print(f"   文件路径: {save_path}")
        print(f"   层类型: {layer_type_str}")
        print(f"   保存数量: {saved_count} 个量化器")
    
    return encodings_dict


def load_quantizer_encodings(
    sim_model,
    load_path: str,
    layer_types: Optional[List[str]] = None,
    exclude_layer_types: Optional[List[str]] = None,
    skip_if_not_found: bool = True,
    verbose: bool = True,
    allow_overwrite: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    从 JSON 文件加载量化参数并应用到模型
    
    Args:
        sim_model: AIMET QuantizationSimModel.model
        load_path: 加载路径
        layer_types: 要加载的层类型列表，例如 ["QuantizedConv2d"]
                    如果为 None，则加载所有匹配的量化器
        exclude_layer_types: 要排除的层类型列表，例如 ["QuantizedGRU"]
                           优先级高于 layer_types
        skip_if_not_found: 如果量化器不存在，是否跳过
        verbose: 是否打印详细信息
        allow_overwrite: 加载后是否允许后续 compute_encodings 覆盖；
                        设为 False 可锁定已加载量化器
    
    Returns:
        加载统计信息
    
    Examples:
        # 1. 加载所有保存的量化参数
        load_quantizer_encodings(sim.model, "stage1.json")
        
        # 2. 只加载 Conv2d
        load_quantizer_encodings(sim.model, "stage1.json", layer_types=["QuantizedConv2d"])
        
        # 3. 加载除了 GRU 之外的所有参数
        load_quantizer_encodings(sim.model, "stage1.json", exclude_layer_types=["QuantizedGRU"])
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"编码文件不存在: {load_path}")
    
    with open(load_path, 'r', encoding='utf-8') as f:
        encodings_dict = json.load(f)
    
    loaded_count = 0
    skipped_count = 0
    loaded_types = set()
    
    # 尝试导入 QuantGRU
    try:
        from quant_gru import QuantGRU
    except ImportError:
        try:
            import sys
            from pathlib import Path
            quant_gru_path = Path(__file__).parent.parent.parent.parent / "quant-gru-pytorch" / "pytorch"
            if quant_gru_path.exists():
                sys.path.insert(0, str(quant_gru_path))
            from quant_gru import QuantGRU
        except ImportError:
            QuantGRU = None
    
    # 先处理 QuantGRU 模块（如果 encodings_dict 包含 activation_encodings）
    if QuantGRU is not None and "activation_encodings" in encodings_dict:
        if verbose:
            print("\n📥 尝试从 AIMET encodings 加载 QuantGRU 量化参数...")
        
        for module_name, module in sim_model.named_modules():
            if isinstance(module, QuantGRU):
                target_name = getattr(module, 'aimet_onnx_name', None)
                if not target_name:
                    # 尝试从 encodings 文件中查找对应的键
                    # 通常 encodings 中的键就是模块路径名称
                    if module_name in encodings_dict.get("activation_encodings", {}):
                        target_name = module_name
                        # 设置 aimet_onnx_name 以便后续使用
                        module.aimet_onnx_name = target_name
                    else:
                        # 如果找不到，使用模块路径名称作为后备
                        target_name = module_name
                        if verbose:
                            print(f"  ⚠️ 警告：{module_name} 的 aimet_onnx_name 未设置，使用模块路径名称作为后备")
                
                loaded_ok = module.load_quant_params_from_aimet_format(
                    encodings_dict,
                    module_name=target_name,
                    verbose=verbose
                )
                if loaded_ok and allow_overwrite is not None:
                    module.set_quant_params_locked(not allow_overwrite)

                if loaded_ok:
                    loaded_count += 1
                    loaded_types.add("QuantGRU")
                    if verbose:
                        print(f"  ✅ {module_name}: 成功加载 QuantGRU 量化参数")
    
    # 检查 encodings 文件格式：AIMET 标准格式（activation_encodings + param_encodings）
    if "activation_encodings" in encodings_dict or "param_encodings" in encodings_dict:
        if verbose:
            print("\n📝 检测到 AIMET 标准格式 encodings")
            print("   QuantGRU 参数已加载完成，正在加载 CONV 等 AIMET 量化器...")
        module_map = {name: m for name, m in sim_model.named_modules()}
        device = _get_model_device(sim_model)
        act = encodings_dict.get("activation_encodings", {})
        param_enc = encodings_dict.get("param_encodings", {})
        gru_module_names = _collect_gru_module_names_from_activation_encodings(act)
        # 1) 非 GRU 的 activation_encodings：优先用 is_GRU 标记，无则用名字启发式
        for mod_name, enc in act.items():
            if _is_gru_activation_by_encoding(enc):
                continue
            if isinstance(enc, dict) and enc.get("is_GRU") is None and _is_gru_activation_key(mod_name):
                continue
            module = module_map.get(mod_name)
            if module is None:
                if verbose:
                    print(f"  ⚠️ 模块不存在，跳过: {mod_name}")
                skipped_count += 1
                continue
            if not isinstance(enc, dict):
                continue
            # input: 对应 input_quantizers[0], [1], ...
            inp_enc = enc.get("input")
            if isinstance(inp_enc, list) and len(inp_enc) > 0 and hasattr(module, 'input_quantizers'):
                for idx, item in enumerate(inp_enc):
                    if idx >= len(module.input_quantizers) or module.input_quantizers[idx] is None:
                        break
                    q = module.input_quantizers[idx]
                    rmin = item.get("real_min")
                    rmax = item.get("real_max")
                    if rmin is not None and rmax is not None and _apply_aimet_encoding_to_quantizer(
                        q, rmin, rmax, device, item, verbose
                    ):
                        _set_quantizer_allow_overwrite(q, allow_overwrite)
                        loaded_count += 1
                        loaded_types.add(type(module).__name__)
            elif isinstance(inp_enc, dict):
                rmin, rmax = inp_enc.get("real_min"), inp_enc.get("real_max")
                if hasattr(module, 'input_quantizers') and len(module.input_quantizers) > 0 and module.input_quantizers[0] is not None and rmin is not None and rmax is not None:
                    if _apply_aimet_encoding_to_quantizer(
                        module.input_quantizers[0], rmin, rmax, device, inp_enc, verbose
                    ):
                        _set_quantizer_allow_overwrite(module.input_quantizers[0], allow_overwrite)
                        loaded_count += 1
                        loaded_types.add(type(module).__name__)
            # output: 对应 output_quantizers[0], [1], ...
            out_enc = enc.get("output")
            if isinstance(out_enc, list) and len(out_enc) > 0 and hasattr(module, 'output_quantizers'):
                for idx, item in enumerate(out_enc):
                    if idx >= len(module.output_quantizers) or module.output_quantizers[idx] is None:
                        break
                    q = module.output_quantizers[idx]
                    rmin = item.get("real_min")
                    rmax = item.get("real_max")
                    if rmin is not None and rmax is not None and _apply_aimet_encoding_to_quantizer(
                        q, rmin, rmax, device, item, verbose
                    ):
                        _set_quantizer_allow_overwrite(q, allow_overwrite)
                        loaded_count += 1
                        loaded_types.add(type(module).__name__)
            elif isinstance(out_enc, dict):
                rmin, rmax = out_enc.get("real_min"), out_enc.get("real_max")
                if hasattr(module, 'output_quantizers') and len(module.output_quantizers) > 0 and module.output_quantizers[0] is not None and rmin is not None and rmax is not None:
                    if _apply_aimet_encoding_to_quantizer(
                        module.output_quantizers[0], rmin, rmax, device, out_enc, verbose
                    ):
                        _set_quantizer_allow_overwrite(module.output_quantizers[0], allow_overwrite)
                        loaded_count += 1
                        loaded_types.add(type(module).__name__)
        # 2) 非 GRU 的 param_encodings：通用解析 key 为 "module_name.param_name"（与 param_quantizers 匹配）
        for key, enc in param_enc.items():
            if not isinstance(enc, dict):
                continue
            # 标准做法：param key 形如 "module_name.param_name"，GRU 模块由 is_GRU 已收集
            if any(key.startswith(g + ".") for g in gru_module_names):
                continue
            rmin = enc.get("real_min")
            rmax = enc.get("real_max")
            if rmin is None or rmax is None:
                continue
            parsed = _parse_param_encodings_key(key, module_map)
            if parsed is None:
                skipped_count += 1
                continue
            mod_name, param_name = parsed
            module = module_map[mod_name]
            if param_name not in module.param_quantizers:
                skipped_count += 1
                continue
            q = module.param_quantizers[param_name]
            if q is None:
                skipped_count += 1
                continue
            if _apply_aimet_encoding_to_quantizer(
                q, rmin, rmax, device, enc, verbose
            ):
                _set_quantizer_allow_overwrite(q, allow_overwrite)
                loaded_count += 1
                loaded_types.add(type(module).__name__)
        if verbose:
            print(f"\n✅ AIMET 格式加载完成: 共加载 {loaded_count} 个量化器")
        return {
            'loaded': loaded_count,
            'skipped': skipped_count,
            'loaded_types': list(loaded_types)
        }
    
    # 创建模块映射
    module_map = {name: module for name, module in sim_model.named_modules()}
    
    for key, encoding_info in encodings_dict.items():
        module_name = encoding_info['module_name']
        module_type = encoding_info.get('module_type', '')
        
        # 优先检查排除列表
        if exclude_layer_types is not None:
            if any(layer_type in module_type for layer_type in exclude_layer_types):
                skipped_count += 1
                continue
        
        # 检查层类型是否匹配
        if layer_types is not None:
            if not any(layer_type in module_type for layer_type in layer_types):
                skipped_count += 1
                continue
        
        # 检查模块是否存在
        if module_name not in module_map:
            if skip_if_not_found:
                skipped_count += 1
                continue
            else:
                raise KeyError(f"模块不存在: {module_name}")
        
        module = module_map[module_name]
        
        # 加载 param_quantizers
        if encoding_info['quantizer_type'] == 'param':
            param_name = encoding_info['param_name']
            if hasattr(module, 'param_quantizers') and param_name in module.param_quantizers:
                quantizer = module.param_quantizers[param_name]
                if quantizer is not None:
                    _apply_encoding_to_quantizer(quantizer, encoding_info)
                    _set_quantizer_allow_overwrite(quantizer, allow_overwrite)
                    loaded_count += 1
                    loaded_types.add(module_type)
                    if verbose:
                        print(f"  ✅ {module_name}.param_quantizers.{param_name}")
                else:
                    if not skip_if_not_found:
                        raise ValueError(f"量化器为 None: {key}")
                    skipped_count += 1
            else:
                if not skip_if_not_found:
                    raise ValueError(f"参数量化器不存在: {key}")
                skipped_count += 1
        
        # 加载 output_quantizers
        elif encoding_info['quantizer_type'] == 'output':
            quantizer_idx = encoding_info['quantizer_idx']
            if hasattr(module, 'output_quantizers') and quantizer_idx < len(module.output_quantizers):
                quantizer = module.output_quantizers[quantizer_idx]
                if quantizer is not None:
                    _apply_encoding_to_quantizer(quantizer, encoding_info)
                    _set_quantizer_allow_overwrite(quantizer, allow_overwrite)
                    loaded_count += 1
                    loaded_types.add(module_type)
                    if verbose:
                        print(f"  ✅ {module_name}.output_quantizers[{quantizer_idx}]")
                else:
                    if not skip_if_not_found:
                        raise ValueError(f"量化器为 None: {key}")
                    skipped_count += 1
            else:
                if not skip_if_not_found:
                    raise ValueError(f"输出量化器不存在: {key}")
                skipped_count += 1
    
    if verbose:
        print(f"\n✅ 加载量化参数:")
        print(f"   文件路径: {load_path}")
        if layer_types:
            print(f"   指定层类型: {', '.join(layer_types)}")
        if exclude_layer_types:
            print(f"   排除层类型: {', '.join(exclude_layer_types)}")
        if not layer_types and not exclude_layer_types:
            print(f"   加载模式: 全部加载")
        print(f"   加载数量: {loaded_count} 个量化器")
        print(f"   涉及层类型: {', '.join(sorted(loaded_types))}")
        if skipped_count > 0:
            print(f"   跳过数量: {skipped_count} 个")
    
    return {
        'loaded': loaded_count,
        'skipped': skipped_count,
        'loaded_types': list(loaded_types)
    }


def disable_quantization_by_layer_types(
    sim_model,
    layer_types: List[str],
    verbose: bool = True
) -> Dict[str, int]:
    """
    禁用指定层类型的所有量化器
    
    Args:
        sim_model: AIMET QuantizationSimModel.model
        layer_types: 要禁用的层类型列表，例如 ["QuantizedGRU", "QuantizedLinear"]
        verbose: 是否打印详细信息
    
    Returns:
        禁用统计信息
    """
    disabled_modules = []
    disabled_param_count = 0
    disabled_output_count = 0
    
    for module_name, module in sim_model.named_modules():
        module_type = type(module).__name__
        
        # 检查是否匹配指定的层类型
        if not any(layer_type in module_type for layer_type in layer_types):
            continue
        
        # 禁用 param_quantizers
        if hasattr(module, 'param_quantizers'):
            for param_name, quantizer in module.param_quantizers.items():
                if quantizer is not None:
                    module.param_quantizers[param_name] = None
                    disabled_param_count += 1
        
        # 禁用 output_quantizers
        if hasattr(module, 'output_quantizers'):
            for idx in range(len(module.output_quantizers)):
                if module.output_quantizers[idx] is not None:
                    module.output_quantizers[idx] = None
                    disabled_output_count += 1
        
        # 禁用 input_quantizers
        if hasattr(module, 'input_quantizers'):
            for idx in range(len(module.input_quantizers)):
                if module.input_quantizers[idx] is not None:
                    module.input_quantizers[idx] = None
        
        disabled_modules.append(module_name)
    
    if verbose:
        layer_type_str = ', '.join(layer_types)
        print(f"\n🚫 禁用量化:")
        print(f"   层类型: {layer_type_str}")
        print(f"   涉及模块: {len(disabled_modules)} 个")
        print(f"   参数量化器: {disabled_param_count} 个")
        print(f"   输出量化器: {disabled_output_count} 个")
    
    return {
        'disabled_modules': len(disabled_modules),
        'disabled_param_quantizers': disabled_param_count,
        'disabled_output_quantizers': disabled_output_count
    }


def _tensor_to_list(value):
    """将张量或标量转换为列表"""
    if isinstance(value, torch.Tensor):
        return value.cpu().tolist()
    elif isinstance(value, (int, float)):
        return value
    else:
        return value


def _is_gru_activation_by_encoding(enc: dict) -> bool:
    """根据 encodings 条目的 is_GRU 标记判断是否为 GRU（标准做法）。"""
    if isinstance(enc, dict) and "is_GRU" in enc:
        return enc.get("is_GRU") is True
    return False


def _is_gru_activation_key(key: str) -> bool:
    """启发式：key 含 seq_t/seq_f 时视为 GRU（仅当 enc 中无 is_GRU 时作后备）。"""
    return "seq_t" in key or "seq_f" in key


def _collect_gru_module_names_from_activation_encodings(act: dict) -> set:
    """从 activation_encodings 中收集 is_GRU 为 True 的模块名（用于 param 跳过）。"""
    out = set()
    for mod_name, enc in act.items():
        if isinstance(enc, dict) and enc.get("is_GRU") is True:
            out.add(mod_name)
    return out


def _parse_param_encodings_key(key: str, module_map: dict):
    """
    通用解析 param_encodings 的 key，得到 (module_name, param_name) 或 None。
    约定：key 形如 "module_name.param_name"，其中 param_name 必须在 module.param_quantizers 中存在。
    按最长匹配优先，避免 "a.b.c" 被拆成 "a.b" + "c" 而 "a.b.c" 才是模块名。
    """
    if not key or "." not in key:
        return None
    # 按模块名长度从长到短尝试，以便匹配 "generator.enc_seqs.0.conv_f" 而非 "generator"
    candidates = sorted(module_map.keys(), key=len, reverse=True)
    for mod_name in candidates:
        prefix = mod_name + "."
        if key.startswith(prefix):
            param_name = key[len(prefix):]
            module = module_map.get(mod_name)
            if module is not None and hasattr(module, 'param_quantizers') and param_name in module.param_quantizers:
                return (mod_name, param_name)
    return None


def _encoding_compatible_with_quantizer(enc_entry: dict, quantizer, verbose: bool) -> tuple:
    """
    校验 encodings 条目与量化器是否兼容（bitwidth、symmetric）。
    enc_entry 可为单条 dict（含 bitwidth, is_symmetric）或 list 的首元素。
    Returns: (ok: bool, msg: str)
    """
    if not isinstance(enc_entry, dict):
        return True, ""
    entry = enc_entry.get("input", enc_entry) if isinstance(enc_entry.get("input"), list) else enc_entry
    if isinstance(entry, list) and len(entry) > 0:
        entry = entry[0]
    if not isinstance(entry, dict):
        return True, ""
    file_bw = entry.get("bitwidth")
    file_sym = entry.get("is_symmetric")
    if file_sym is not None and isinstance(file_sym, str):
        file_sym = file_sym.lower() == "true"
    if file_bw is None and file_sym is None:
        return True, ""
    try:
        q_bw = None
        if hasattr(quantizer, 'qmin') and hasattr(quantizer, 'qmax'):
            num_steps = int(quantizer.qmax) - int(quantizer.qmin) + 1
            q_bw = int(round((num_steps - 1).bit_length())) if num_steps > 1 else 0
        q_sym = getattr(quantizer, 'symmetric', getattr(quantizer, '_symmetric', None))
        if file_bw is not None and q_bw is not None and file_bw != q_bw:
            return False, f"bitwidth 不一致: file={file_bw}, quantizer≈{q_bw}"
        if file_sym is not None and q_sym is not None and bool(file_sym) != bool(q_sym):
            return False, f"symmetric 不一致: file={file_sym}, quantizer={q_sym}"
    except Exception as e:
        if verbose:
            print(f"  [校验兼容性时忽略]: {e}")
    return True, ""


def _get_model_device(sim_model):
    """从模型获取 device，优先 parameters，其次 buffers，再次从任意量化器取。"""
    try:
        return next(sim_model.parameters()).device
    except StopIteration:
        pass
    try:
        return next(sim_model.buffers()).device
    except StopIteration:
        pass
    for _name, mod in sim_model.named_modules():
        if hasattr(mod, 'param_quantizers'):
            for q in mod.param_quantizers.values() if hasattr(mod.param_quantizers, 'values') else []:
                if q is not None and hasattr(q, 'parameters'):
                    for p in q.parameters():
                        return p.device
    return None


def _align_tensor_to_shape(value_t: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    """
    将输入张量对齐到目标形状，兼容常见标量/单元素场景：
    - [] -> [1]（expand）
    - [1] -> []（reshape/squeeze 为标量）
    """
    if value_t.shape == target_shape:
        return value_t

    # 目标为标量：单元素张量可安全压为标量
    if len(target_shape) == 0 and value_t.numel() == 1:
        return value_t.reshape(target_shape).contiguous()

    # 输入为标量，目标为向量/多维：可广播扩展
    if value_t.numel() == 1:
        return value_t.expand(target_shape).contiguous()

    # 元素个数一致：尝试重排形状
    target_numel = 1
    for dim in target_shape:
        target_numel *= int(dim)
    if value_t.numel() == target_numel:
        return value_t.reshape(target_shape).contiguous()

    return value_t


def _apply_aimet_encoding_to_quantizer(
    quantizer,
    real_min,
    real_max,
    device=None,
    enc_entry=None,
    verbose=False
):
    """
    将 AIMET 导出格式的 real_min/real_max 应用到量化器（调用 set_range）。
    real_min, real_max 可为标量或列表，会转为 tensor。
    若 enc_entry 提供且与量化器不兼容（bitwidth/symmetric），则跳过并返回 False。
    """
    if not hasattr(quantizer, 'set_range'):
        return False
    if real_min is None or real_max is None:
        return False
    if enc_entry is not None:
        ok, msg = _encoding_compatible_with_quantizer(enc_entry, quantizer, verbose)
        if not ok:
            if verbose:
                print(f"  ⚠️ 跳过（编码不兼容）: {msg}")
            return False
    min_t = torch.tensor(real_min, dtype=torch.float32)
    max_t = torch.tensor(real_max, dtype=torch.float32)
    if device is not None:
        min_t = min_t.to(device)
        max_t = max_t.to(device)
    try:
        # 对齐输入形状到 quantizer 参数形状，覆盖常见 [] -> [1] 场景
        if hasattr(quantizer, 'min') and isinstance(quantizer.min, torch.Tensor):
            q_min_shape = quantizer.min.shape
            min_t = _align_tensor_to_shape(min_t, q_min_shape)

        if hasattr(quantizer, 'max') and isinstance(quantizer.max, torch.Tensor):
            q_max_shape = quantizer.max.shape
            max_t = _align_tensor_to_shape(max_t, q_max_shape)

        quantizer.set_range(min_t, max_t)
        return True
    except Exception as e:
        if verbose:
            print(f"  ⚠️ set_range 失败: {e}")
        return False


def _apply_encoding_to_quantizer(quantizer, encoding_info: Dict):
    """将编码信息应用到量化器"""
    # 转换为张量
    scale_data = encoding_info['scale']
    offset_data = encoding_info['offset']
    min_data = encoding_info['min']
    max_data = encoding_info['max']
    
    # 处理标量或列表
    scale = torch.tensor(scale_data, dtype=torch.float32) if not isinstance(scale_data, torch.Tensor) else scale_data
    offset = torch.tensor(offset_data, dtype=torch.float32) if not isinstance(offset_data, torch.Tensor) else offset_data
    min_val = torch.tensor(min_data, dtype=torch.float32) if not isinstance(min_data, torch.Tensor) else min_data
    max_val = torch.tensor(max_data, dtype=torch.float32) if not isinstance(max_data, torch.Tensor) else max_data
    
    # 如果 shape 不为空，reshape
    if encoding_info.get('shape'):
        shape = tuple(encoding_info['shape'])
        if scale.numel() == 1 and shape:
            scale = scale.expand(shape)
        elif scale.shape != shape:
            scale = scale.view(shape)
        if offset.numel() == 1 and shape:
            offset = offset.expand(shape)
        elif offset.shape != shape:
            offset = offset.view(shape)
        if min_val.numel() == 1 and shape:
            min_val = min_val.expand(shape)
        elif min_val.shape != shape:
            min_val = min_val.view(shape)
        if max_val.numel() == 1 and shape:
            max_val = max_val.expand(shape)
        elif max_val.shape != shape:
            max_val = max_val.view(shape)
    
    # 确保在正确的设备上
    device = next((p.device for p in quantizer.parameters()), torch.device('cpu'))
    scale = scale.to(device)
    offset = offset.to(device)
    min_val = min_val.to(device)
    max_val = max_val.to(device)
    
    # 使用 set_range 方法设置编码
    try:
        quantizer.set_range(min_val, max_val)
    except Exception as e:
        print(f"    ⚠️  set_range 失败: {e}")
        # 如果 set_range 失败，尝试其他方法
        pass


def _set_quantizer_allow_overwrite(quantizer, allow_overwrite: Optional[bool]) -> None:
    """设置量化器后续是否允许被 compute_encodings 覆盖。"""
    if quantizer is None or allow_overwrite is None:
        return
    if hasattr(quantizer, 'allow_overwrite'):
        quantizer.allow_overwrite(allow_overwrite)
    elif hasattr(quantizer, '_allow_overwrite'):
        quantizer._allow_overwrite = allow_overwrite


def _has_active_quantizers(module) -> bool:
    """检查模块是否还有启用的 input/output quantizer。"""
    if hasattr(module, 'input_quantizers') and any(q is not None for q in module.input_quantizers):
        return True
    if hasattr(module, 'output_quantizers') and any(q is not None for q in module.output_quantizers):
        return True
    return False


def _is_initialized_quantizer(quantizer) -> bool:
    """判断量化器是否已初始化且能提供 encodings。"""
    if quantizer is None or not hasattr(quantizer, 'is_initialized'):
        return False
    if not quantizer.is_initialized():
        return False
    encoding = quantizer.get_encodings() if hasattr(quantizer, 'get_encodings') else None
    return encoding is not None


def _leaf_module_name(module_name: str) -> str:
    """返回模块路径最后一级名称。"""
    return module_name.split(".")[-1] if module_name else module_name


_NON_TRANSPARENT_UNQUANTIZED_MODULE_TYPES = {
    "GRU",
    "GRUCell",
    "LSTM",
    "LSTMCell",
    "RNN",
    "RNNCell",
    "OptimizedQuantizableGRU",
    "QuantGRU",
}


def _is_transparent_call_module(module_name: str, module, transparent_prefixes: Tuple[str, ...]) -> bool:
    """
    判断 call_module 节点是否视为透明节点。

    透明节点允许图遍历继续穿透，典型用于：
    - module_clamp_* 这类禁用量化的中间节点
    - 所有 input/output quantizer 都被禁用的模块
    """
    leaf_name = _leaf_module_name(module_name)
    if any(leaf_name.startswith(prefix) for prefix in transparent_prefixes):
        return True
    module_type_name = type(module).__name__
    if module_type_name in {"Concat", "QuantizedConcat", "FakeQuantizedConcat"}:
        # Concat 是真实的多输入汇聚边界，不能仅因量化器关闭就被当作透明节点穿透。
        return False
    if module_type_name in _NON_TRANSPARENT_UNQUANTIZED_MODULE_TYPES:
        # 循环层即使当前未挂量化器，也会显著改变张量分布，不能按透明节点穿透。
        return False
    return not _has_active_quantizers(module)


def _dedup_nodes(nodes: list) -> list:
    """按出现顺序去重 FX 节点。"""
    seen = set()
    result = []
    for node in nodes:
        node_id = id(node)
        if node_id in seen:
            continue
        seen.add(node_id)
        result.append(node)
    return result


def _dedup_indices(indices: list[int]) -> list[int]:
    """按顺序去重索引列表。"""
    seen = set()
    result = []
    for idx in indices:
        if idx in seen:
            continue
        seen.add(idx)
        result.append(idx)
    return result


def _iter_fx_nodes(arg_obj):
    """递归展开参数中的 FX 节点。"""
    if isinstance(arg_obj, (list, tuple)):
        for item in arg_obj:
            yield from _iter_fx_nodes(item)
    elif hasattr(arg_obj, "op"):
        yield arg_obj


def _callable_target_name(target) -> str:
    """返回 FX target 的可比对名称。"""
    return getattr(target, "__name__", str(target))


def _is_metadata_only_node(
    node,
    module_map: dict,
    transparent_prefixes: Tuple[str, ...],
    visited: Optional[Set[int]] = None,
) -> bool:
    """
    判断节点是否只参与 shape/index 等元信息计算，而不承载真实张量数据。

    典型例子：
    - getattr(x, "shape")
    - getitem(shape, i)
    - view/reshape 的尺寸参数派生链
    - 仅对 shape 标量做运算得到的 module_mul_*
    """
    if visited is None:
        visited = set()

    if isinstance(node, (list, tuple)):
        fx_nodes = list(_iter_fx_nodes(node))
        return bool(fx_nodes) and all(
            _is_metadata_only_node(item, module_map, transparent_prefixes, visited)
            for item in fx_nodes
        )

    if not hasattr(node, "op"):
        return False

    node_id = id(node)
    if node_id in visited:
        return False
    visited.add(node_id)

    if node.op in ("placeholder", "get_attr"):
        return False

    if node.op == "call_function":
        target_name = _callable_target_name(node.target)
        if (
            target_name == "getattr"
            and len(node.args) >= 2
            and node.args[1] in ("shape", "size")
        ):
            return True
        if target_name == "getitem" and len(node.args) >= 1:
            return _is_metadata_only_node(
                node.args[0], module_map, transparent_prefixes, visited
            )

    if node.op == "call_method" and str(node.target) in {
        "size",
        "dim",
        "ndim",
        "numel",
        "stride",
    }:
        return True

    if node.op == "call_module":
        module = module_map.get(node.target)
        if module is None:
            return False
        if _is_transparent_call_module(node.target, module, transparent_prefixes):
            fx_args = list(_iter_fx_nodes(node.args))
            return bool(fx_args) and all(
                _is_metadata_only_node(arg, module_map, transparent_prefixes, visited)
                for arg in fx_args
            )

    fx_args = list(_iter_fx_nodes(getattr(node, "args", ())))
    return bool(fx_args) and all(
        _is_metadata_only_node(arg, module_map, transparent_prefixes, visited)
        for arg in fx_args
    )


def _iter_data_users(node, module_map: dict, transparent_prefixes: Tuple[str, ...]):
    """仅返回真实数据流上的下游 users，跳过 shape/index 元信息分支。"""
    for user in getattr(node, "users", ()):
        if _is_metadata_only_node(user, module_map, transparent_prefixes):
            continue
        yield user


def _find_prev_quantized_from_prev_args(prev_args, module_map: dict, transparent_prefixes: Tuple[str, ...], stats: Dict[str, int], visited: Set[int]) -> list:
    """
    在一组上游参数中优先返回“最近命中”的真实量化源。

    对 view/permute/reshape/getitem 等 FX 透明节点，后续参数往往只是 shape/index 等元信息。
    继续遍历这些元信息参数会把更早祖先误收进来，导致“最近边界”被污染。
    因此这里按参数顺序查找，一旦命中真实上游源就立即返回。
    """
    for prev_arg in prev_args:
        found = _find_prev_quantized_call_modules(
            prev_arg,
            module_map,
            transparent_prefixes,
            stats,
            visited,
        )
        if found:
            return _dedup_nodes(found)
    return []


def _find_next_quantized_from_next_users(next_users, module_map: dict, transparent_prefixes: Tuple[str, ...], stats: Dict[str, int], visited: Set[int]) -> list:
    """
    在一组下游 users 中，按“每条路径只取最近一个真实量化边界”的规则查找目标。

    这样可以避免：
    conv2d_out -> ... -> module_mul_17 -> ... -> module_mul_18
    这种链路里同时返回最近边界 module_mul_17 和更远边界 module_mul_18。
    """
    result = []
    for user in next_users:
        if not hasattr(user, 'op'):
            continue

        node_id = id(user)
        if node_id in visited:
            continue
        visited.add(node_id)

        if user.op == 'call_module':
            module = module_map.get(user.target)
            if module is None:
                continue
            if _is_transparent_call_module(user.target, module, transparent_prefixes):
                stats['passthrough_disabled'] += 1
                found = _find_next_quantized_from_next_users(
                    _iter_data_users(user, module_map, transparent_prefixes),
                    module_map,
                    transparent_prefixes,
                    stats,
                    visited,
                )
                if found:
                    result.extend(found)
            else:
                result.append(user)
        elif user.op in ('call_function', 'call_method'):
            found = _find_next_quantized_from_next_users(
                _iter_data_users(user, module_map, transparent_prefixes),
                module_map,
                transparent_prefixes,
                stats,
                visited,
            )
            if found:
                result.extend(found)

    return _dedup_nodes(result)


def _find_next_quantized_call_modules(node, module_map: dict, transparent_prefixes: Tuple[str, ...], stats: Dict[str, int], visited: Optional[Set[int]] = None) -> list:
    """
    从当前节点开始向下游查找真正的量化目标模块。

    规则：
    - call_function / call_method 节点视为透明
    - 禁用量化的 call_module 视为透明，可继续穿透
    - 对每条路径，遇到第一个启用量化的 call_module 就停止，不再继续搜更远下游
    """
    if visited is None:
        visited = set()

    if not hasattr(node, 'users'):
        return []

    return _find_next_quantized_from_next_users(
        _iter_data_users(node, module_map, transparent_prefixes),
        module_map,
        transparent_prefixes,
        stats,
        visited,
    )


def _find_prev_quantized_call_modules(arg_obj, module_map: dict, transparent_prefixes: Tuple[str, ...], stats: Dict[str, int], visited: Optional[Set[int]] = None) -> list:
    """
    从当前输入参数对象向上游回溯，查找真正的量化源模块。

    - call_function / call_method 节点视为透明
    - 禁用量化的 call_module 视为透明，可继续穿透
    - 遇到启用量化的 call_module 则返回为源节点
    """
    if visited is None:
        visited = set()

    result = []

    if isinstance(arg_obj, (list, tuple)):
        return _find_prev_quantized_from_prev_args(
            arg_obj,
            module_map,
            transparent_prefixes,
            stats,
            visited,
        )

    if not hasattr(arg_obj, 'op'):
        return result

    node_id = id(arg_obj)
    if node_id in visited:
        return result
    visited.add(node_id)

    if arg_obj.op == 'call_module':
        module = module_map.get(arg_obj.target)
        if module is None:
            return result
        if _is_transparent_call_module(arg_obj.target, module, transparent_prefixes):
            stats['passthrough_disabled'] += 1
            return _find_prev_quantized_from_prev_args(
                arg_obj.args,
                module_map,
                transparent_prefixes,
                stats,
                visited,
            )
        return [arg_obj]

    if arg_obj.op in ('call_function', 'call_method'):
        return _find_prev_quantized_from_prev_args(
            arg_obj.args,
            module_map,
            transparent_prefixes,
            stats,
            visited,
        )

    return result


def _sync_quantizer_range(src_q, dst_q) -> None:
    """将源量化器的 min/max 同步到目标量化器。"""
    src_encoding = src_q.get_encodings()
    device = next(dst_q.parameters()).device
    dst_q.set_range(src_encoding.min.to(device), src_encoding.max.to(device))


def _format_encoding_range(encoding) -> str:
    """格式化 encodings 的 min/max 以便日志打印。"""
    min_v = encoding.min.item() if encoding.min.numel() == 1 else encoding.min.tolist()
    max_v = encoding.max.item() if encoding.max.numel() == 1 else encoding.max.tolist()
    if isinstance(min_v, float) and isinstance(max_v, float):
        return f"[{min_v:.4f}, {max_v:.4f}]"
    return f"[{min_v}, {max_v}]"


def _get_non_none_input_quantizer_indices(module) -> list[int]:
    """返回模块中所有非 None 的 input quantizer 索引。"""
    if not hasattr(module, 'input_quantizers'):
        return []
    return [idx for idx, q in enumerate(module.input_quantizers) if q is not None]


def _resolve_downstream_input_indices(
    dst_node,
    dst_module,
    source_node,
    module_map: dict,
    transparent_prefixes: Tuple[str, ...],
) -> list[int]:
    """
    解析下游模块中真正应该同步的 input quantizer 索引。

    规则：
    1. 优先使用 FX 参数依赖能直接命中的 idx
    2. 仅保留非 None 的 input quantizer 槽位
    3. 若直接命中为空，但模块只有一个非 None input quantizer，
       则回退到该唯一槽位（适用于 Snake2d 一类多输入但只量化主输入的模块）
    4. 若直接命中存在，但还包含 None 槽位，且模块只有一个非 None 槽位，
       则把该槽位作为后备候选追加进去
    """
    if not hasattr(dst_module, 'input_quantizers') or len(dst_module.input_quantizers) == 0:
        return []

    input_quantizers = list(dst_module.input_quantizers)
    non_none_indices = _get_non_none_input_quantizer_indices(dst_module)
    dependent_arg_indices = [
        idx for idx in range(min(len(dst_node.args), len(input_quantizers)))
        if _arg_depends_on_node(
            dst_node.args[idx],
            source_node,
            module_map,
            transparent_prefixes,
        )
    ]

    resolved = [idx for idx in dependent_arg_indices if idx in non_none_indices]

    if len(non_none_indices) == 1:
        sole_idx = non_none_indices[0]
        if dependent_arg_indices and sole_idx not in resolved:
            resolved.append(sole_idx)
        elif not resolved:
            resolved = [sole_idx]

    return _dedup_indices(resolved)


def _sync_and_verify_quantizer_range(src_q, dst_q) -> bool:
    """同步量化器范围，并验证目标量化器已真正初始化。"""
    _sync_quantizer_range(src_q, dst_q)
    return _is_initialized_quantizer(dst_q)


def sync_quantizer_pair_by_name(
    sim_model,
    src_module_name: str,
    dst_module_name: str,
    src_output_idx: int = 0,
    dst_input_idx: int = 0,
    overwrite_initialized: bool = True,
    dst_quantizer_allow_overwrite: Optional[bool] = None,
    verbose: bool = True,
) -> bool:
    """
    按模块名精确同步一对量化器：
    `src_module.output_quantizers[src_output_idx] -> dst_module.input_quantizers[dst_input_idx]`

    适用于 FX 图遍历难以稳定命中的特例链路。
    """
    module_map = dict(sim_model.named_modules())
    src_module = module_map.get(src_module_name)
    dst_module = module_map.get(dst_module_name)

    if src_module is None or dst_module is None:
        if verbose:
            print(f"  ⚠️  精确同步失败: 模块不存在 {src_module_name} -> {dst_module_name}")
        return False

    if not hasattr(src_module, 'output_quantizers') or src_output_idx >= len(src_module.output_quantizers):
        if verbose:
            print(f"  ⚠️  精确同步失败: 源 output_quantizer 不存在 {src_module_name}[{src_output_idx}]")
        return False

    if not hasattr(dst_module, 'input_quantizers') or dst_input_idx >= len(dst_module.input_quantizers):
        if verbose:
            print(f"  ⚠️  精确同步失败: 目标 input_quantizer 不存在 {dst_module_name}[{dst_input_idx}]")
        return False

    src_q = src_module.output_quantizers[src_output_idx]
    dst_q = dst_module.input_quantizers[dst_input_idx]

    if not _is_initialized_quantizer(src_q):
        if verbose:
            print(f"  ⚠️  精确同步失败: 源量化器未初始化 {src_module_name}.output_q[{src_output_idx}]")
        return False

    if dst_q is None:
        if verbose:
            print(f"  ⚠️  精确同步失败: 目标量化器为 None {dst_module_name}.input_q[{dst_input_idx}]")
        return False

    if dst_q.is_initialized() and not overwrite_initialized:
        if verbose:
            print(f"  ⏭  精确同步跳过: 目标已初始化 {dst_module_name}.input_q[{dst_input_idx}]")
        return False

    try:
        if not _sync_and_verify_quantizer_range(src_q, dst_q):
            raise RuntimeError("目标量化器在 set_range 后仍未初始化")
        _set_quantizer_allow_overwrite(dst_q, dst_quantizer_allow_overwrite)
        if verbose:
            print(
                f"  🎯 {src_module_name}.output_q[{src_output_idx}] => "
                f"{dst_module_name}.input_q[{dst_input_idx}]  "
                f"{_format_encoding_range(src_q.get_encodings())}"
            )
        return True
    except Exception as e:
        if verbose:
            print(f"  ⚠️  精确同步失败 {src_module_name} -> {dst_module_name}: {e}")
        return False


def _arg_depends_on_node(arg_obj, source_node, module_map: dict, transparent_prefixes: Tuple[str, ...], visited: Optional[Set[int]] = None) -> bool:
    """
    判断某个下游输入参数是否来自指定 source_node。

    允许穿透：
    - call_function / call_method
    - 透明 call_module（如 module_clamp_* 或量化禁用模块）
    """
    if visited is None:
        visited = set()

    if _is_metadata_only_node(arg_obj, module_map, transparent_prefixes):
        return False

    if isinstance(arg_obj, (list, tuple)):
        return any(
            _arg_depends_on_node(item, source_node, module_map, transparent_prefixes, visited)
            for item in arg_obj
        )

    if not hasattr(arg_obj, 'op'):
        return False

    node_id = id(arg_obj)
    if node_id in visited:
        return False
    visited.add(node_id)

    if arg_obj is source_node:
        return True

    if arg_obj.op == 'call_module':
        module = module_map.get(arg_obj.target)
        if module is None:
            return False
        if not _is_transparent_call_module(arg_obj.target, module, transparent_prefixes):
            return False
        return any(
            _arg_depends_on_node(prev_arg, source_node, module_map, transparent_prefixes, visited)
            for prev_arg in arg_obj.args
        )

    if arg_obj.op in ('call_function', 'call_method'):
        return any(
            _arg_depends_on_node(prev_arg, source_node, module_map, transparent_prefixes, visited)
            for prev_arg in arg_obj.args
        )

    return False


def _has_blocking_call_module_between(
    arg_obj,
    source_node,
    module_map: dict,
    transparent_prefixes: Tuple[str, ...],
    visited: Optional[Set[int]] = None,
) -> bool:
    """
    判断从 arg_obj 回溯到 source_node 的路径上，是否还存在 source_node 之前的
    非透明 call_module 阻断边界。

    用于避免：
    pre_act -> seq_t(GRU) -> reshape -> conv_t
    这类链路把 pre_act 错当作 conv_t 的“相邻上游量化边界”。
    """
    if visited is None:
        visited = set()

    if _is_metadata_only_node(arg_obj, module_map, transparent_prefixes):
        return False

    if isinstance(arg_obj, (list, tuple)):
        return any(
            _has_blocking_call_module_between(item, source_node, module_map, transparent_prefixes, visited)
            for item in arg_obj
        )

    if not hasattr(arg_obj, 'op'):
        return False

    node_id = id(arg_obj)
    if node_id in visited:
        return False
    visited.add(node_id)

    if arg_obj.op == 'call_module':
        module = module_map.get(arg_obj.target)
        if module is None:
            return False
        if arg_obj.target == source_node.target:
            return False
        if not _is_transparent_call_module(arg_obj.target, module, transparent_prefixes):
            return True

    return any(
        _has_blocking_call_module_between(prev_arg, source_node, module_map, transparent_prefixes, visited)
        for prev_arg in getattr(arg_obj, 'args', ())
    )


def sync_adjacent_quantizers(
    sim_model,
    verbose: bool = True,
    sync_input_from_upstream: bool = True,
    sync_output_to_downstream: bool = True,
    overwrite_initialized: bool = False,
    synced_quantizer_allow_overwrite: Optional[bool] = None,
    transparent_module_name_prefixes: Tuple[str, ...] = ("module_clamp_",),
) -> Dict[str, int]:
    """
    在 FX Graph 上同步相邻量化器边界，使已初始化模块与上下游保持一致。

    新版策略：
    1. 不再只处理 Conv，任何已初始化的量化模块都可作为锚点
    2. 支持双向同步
       - module.input_q  <- 上游 output_q
       - module.output_q -> 下游 input_q
    3. 支持穿透透明节点
       - call_function / call_method
       - 禁用量化的 call_module
       - 名称匹配 transparent_module_name_prefixes 的中间节点（如 module_clamp_*）
    4. 对上游多来源默认跳过，避免不明确的反向覆盖

    Args:
        sim_model: AIMET QuantizationSimModel.model（FX GraphModule）
        verbose: 是否打印详细日志
        sync_input_from_upstream: 是否同步锚点 input_q 到上游 output_q
        sync_output_to_downstream: 是否同步锚点 output_q 到下游 input_q
        overwrite_initialized: 目标量化器已初始化时是否覆盖
        synced_quantizer_allow_overwrite: 对被同步的目标量化器设置 allow_overwrite；
                                         设为 False 可在后续 calibration 中锁定
        transparent_module_name_prefixes: 允许穿透的透明模块名前缀

    Returns:
        统计信息字典
    """
    if not hasattr(sim_model, 'graph'):
        if verbose:
            print("  ⚠️  sim_model 没有 .graph 属性，跳过 sync_adjacent_quantizers")
        return {
            'synced': 0,
            'upstream_synced': 0,
            'downstream_synced': 0,
            'skipped_disabled': 0,
            'skipped_initialized': 0,
            'skipped_multi_source': 0,
            'skipped_missing_quantizer': 0,
            'passthrough_disabled': 0,
        }

    module_map = dict(sim_model.named_modules())
    stats = {
        'synced': 0,
        'upstream_synced': 0,
        'downstream_synced': 0,
        'skipped_disabled': 0,          # 兼容旧返回字段，等价于 passthrough_disabled
        'skipped_initialized': 0,
        'skipped_multi_source': 0,
        'skipped_missing_quantizer': 0,
        'skipped_blocked_path': 0,
        'passthrough_disabled': 0,
    }

    for node in sim_model.graph.nodes:
        if node.op != 'call_module':
            continue

        anchor_module = module_map.get(node.target)
        if anchor_module is None:
            continue

        has_initialized_input = (
            hasattr(anchor_module, 'input_quantizers') and
            len(anchor_module.input_quantizers) > 0 and
            any(_is_initialized_quantizer(q) for q in anchor_module.input_quantizers)
        )
        has_initialized_output = (
            hasattr(anchor_module, 'output_quantizers') and
            len(anchor_module.output_quantizers) > 0 and
            any(_is_initialized_quantizer(q) for q in anchor_module.output_quantizers)
        )

        if not has_initialized_input and not has_initialized_output:
            continue

        # 1) 反向同步：anchor.input_q <- upstream.output_q
        if sync_input_from_upstream and has_initialized_input:
            input_quantizers = list(getattr(anchor_module, 'input_quantizers', []))
            max_input_args = min(len(node.args), len(input_quantizers))

            for idx in range(max_input_args):
                anchor_input_q = input_quantizers[idx]
                if not _is_initialized_quantizer(anchor_input_q):
                    continue

                upstream_sources = _find_prev_quantized_call_modules(
                    node.args[idx],
                    module_map,
                    transparent_module_name_prefixes,
                    stats,
                )

                if len(upstream_sources) == 0:
                    stats['skipped_missing_quantizer'] += 1
                    continue
                if len(upstream_sources) > 1:
                    stats['skipped_multi_source'] += 1
                    if verbose:
                        names = ", ".join(src.target for src in upstream_sources)
                        print(f"  ⚠️  {node.target}.input[{idx}] 存在多个上游来源，跳过反向同步: {names}")
                    continue

                src_node = upstream_sources[0]
                if verbose and _leaf_module_name(node.target) in {"conv_t", "conv_f"}:
                    debug_blocked = _has_blocking_call_module_between(
                        node.args[idx],
                        src_node,
                        module_map,
                        transparent_module_name_prefixes,
                    )
                    print(
                        f"  [debug-upstream] {node.target}.input[{idx}] "
                        f"sources={[src.target for src in upstream_sources]} "
                        f"chosen={src_node.target} blocked={debug_blocked}"
                    )
                if _has_blocking_call_module_between(
                    node.args[idx],
                    src_node,
                    module_map,
                    transparent_module_name_prefixes,
                ):
                    stats['skipped_blocked_path'] += 1
                    if verbose:
                        print(
                            f"  ⏭  {src_node.target} -> {node.target}.input[{idx}] "
                            f"之间存在非透明中间模块，跳过反向同步"
                        )
                    continue

                src_module = module_map.get(src_node.target)
                if src_module is None or not hasattr(src_module, 'output_quantizers') or len(src_module.output_quantizers) == 0:
                    stats['skipped_missing_quantizer'] += 1
                    continue

                src_output_q = src_module.output_quantizers[0]
                if src_output_q is None:
                    stats['skipped_missing_quantizer'] += 1
                    continue
                if src_output_q.is_initialized() and not overwrite_initialized:
                    stats['skipped_initialized'] += 1
                    continue

                try:
                    _sync_quantizer_range(anchor_input_q, src_output_q)
                    _set_quantizer_allow_overwrite(src_output_q, synced_quantizer_allow_overwrite)
                    stats['upstream_synced'] += 1
                    stats['synced'] += 1
                    if verbose:
                        print(
                            f"  🔁 {src_node.target}.output_q <- {node.target}.input_q[{idx}]  "
                            f"{_format_encoding_range(anchor_input_q.get_encodings())}"
                        )
                except Exception as e:
                    if verbose:
                        print(f"  ⚠️  {src_node.target} <- {node.target}.input[{idx}]: 反向同步失败 - {e}")

        # 2) 正向同步：anchor.output_q -> downstream.input_q
        if sync_output_to_downstream and has_initialized_output:
            output_quantizers = list(getattr(anchor_module, 'output_quantizers', []))

            for out_idx, anchor_output_q in enumerate(output_quantizers):
                if not _is_initialized_quantizer(anchor_output_q):
                    continue

                downstream_targets = _find_next_quantized_call_modules(
                    node,
                    module_map,
                    transparent_module_name_prefixes,
                    stats,
                )

                for dst_node in downstream_targets:
                    dst_module = module_map.get(dst_node.target)
                    if dst_module is None or not hasattr(dst_module, 'input_quantizers') or len(dst_module.input_quantizers) == 0:
                        stats['skipped_missing_quantizer'] += 1
                        continue

                    input_quantizers = list(dst_module.input_quantizers)
                    candidate_input_indices = _resolve_downstream_input_indices(
                        dst_node,
                        dst_module,
                        node,
                        module_map,
                        transparent_module_name_prefixes,
                    )
                    if not candidate_input_indices:
                        stats['skipped_missing_quantizer'] += 1
                        continue

                    synced_this_dst = False
                    for dst_idx in candidate_input_indices:
                        dst_input_q = input_quantizers[dst_idx]
                        if dst_input_q is None:
                            stats['skipped_missing_quantizer'] += 1
                            continue
                        if dst_input_q.is_initialized() and not overwrite_initialized:
                            stats['skipped_initialized'] += 1
                            continue

                        try:
                            if not _sync_and_verify_quantizer_range(anchor_output_q, dst_input_q):
                                raise RuntimeError("目标量化器在 set_range 后仍未初始化")
                            _set_quantizer_allow_overwrite(dst_input_q, synced_quantizer_allow_overwrite)
                            stats['downstream_synced'] += 1
                            stats['synced'] += 1
                            synced_this_dst = True
                            if verbose:
                                print(
                                    f"  🔗 {node.target}.output_q[{out_idx}] -> {dst_node.target}.input_q[{dst_idx}]  "
                                    f"{_format_encoding_range(anchor_output_q.get_encodings())}"
                                )
                            break
                        except Exception as e:
                            if verbose:
                                print(f"  ⚠️  {node.target} -> {dst_node.target}.input[{dst_idx}]: 正向同步失败 - {e}")

                    if not synced_this_dst:
                        stats['skipped_missing_quantizer'] += 1

    stats['skipped_disabled'] = stats['passthrough_disabled']

    if verbose:
        print("\n✅ sync_adjacent_quantizers 完成:")
        print(f"   总传播: {stats['synced']}")
        print(f"   上游同步: {stats['upstream_synced']}")
        print(f"   下游同步: {stats['downstream_synced']}")
        print(f"   穿透 disabled/transparent 节点: {stats['passthrough_disabled']}")
        print(f"   跳过已初始化目标: {stats['skipped_initialized']}")
        print(f"   跳过多上游来源: {stats['skipped_multi_source']}")
        print(f"   跳过缺失量化器: {stats['skipped_missing_quantizer']}")
        print(f"   跳过存在中间阻断模块的路径: {stats['skipped_blocked_path']}")

    return stats


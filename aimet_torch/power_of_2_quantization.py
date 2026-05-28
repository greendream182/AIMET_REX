"""
Power-of-2 量化工具
用于将量化器的scale修改为2的幂次方，以便硬件实现
"""

import torch
import math
from typing import Dict, List, Tuple
from aimet_torch.v2.nn import QuantizationMixin


def print_quantizer_info(model, title="量化器信息"):
    """
    打印模型中所有量化器的完整信息
    
    Args:
        model: 量化模型
        title: 打印标题
    """
    print("\n" + "="*80)
    print(f"{title}")
    print("="*80)
    
    quantizer_info = []
    
    for name, module in model.named_modules():
        # 检查输入量化器（QuantizationMixin 模块）
        if isinstance(module, QuantizationMixin) and hasattr(module, 'input_quantizers'):
            for idx, quantizer in enumerate(module.input_quantizers):
                if quantizer is not None and quantizer.is_initialized():
                    info = get_single_quantizer_info(quantizer)
                    info['module_name'] = name
                    info['quantizer_type'] = f'input_quantizer[{idx}]'
                    quantizer_info.append(info)
        
        # 检查输出量化器（QuantizationMixin 模块）
        if isinstance(module, QuantizationMixin) and hasattr(module, 'output_quantizers'):
            for idx, quantizer in enumerate(module.output_quantizers):
                if quantizer is not None and quantizer.is_initialized():
                    info = get_single_quantizer_info(quantizer)
                    info['module_name'] = name
                    info['quantizer_type'] = f'output_quantizer[{idx}]'
                    quantizer_info.append(info)
        
        # 检查参数量化器（所有模块，包括非 QuantizationMixin）
        if hasattr(module, 'param_quantizers'):
            for param_name, quantizer in module.param_quantizers.items():
                if quantizer is not None and quantizer.is_initialized():
                    info = get_single_quantizer_info(quantizer)
                    info['module_name'] = name
                    info['quantizer_type'] = f'param_quantizer[{param_name}]'
                    quantizer_info.append(info)
    
    # 打印信息
    for i, info in enumerate(quantizer_info, 1):
        print(f"\n{i}. {info['module_name']} - {info['quantizer_type']}")
        print(f"   Qmin: {info['qmin']}, Qmax: {info['qmax']}")
        print(f"   Min: {info['min']:.6f}, Max: {info['max']:.6f}")
        
        # 打印 scale 信息
        if 'scale_tensor' in info:
            # Per-channel 量化
            scale_tensor = info['scale_tensor']
            num_channels = scale_tensor.numel()
            print(f"   Scale (per-channel): {num_channels} 个通道")
            
            if num_channels <= 10:
                # 通道数少，全部打印
                print(f"     {'Channel':>10s} {'Scale':>20s} {'Reciprocal':>15s}")
                for ch_idx in range(num_channels):
                    scale_val = scale_tensor.flatten()[ch_idx].item()
                    recip = 1/scale_val if scale_val != 0 else float('inf')
                    print(f"     {ch_idx:>10d} {scale_val:>20.10f} {recip:>15.2f}")
            else:
                # 通道数多，打印统计信息和前后几个
                print(f"   Scale 范围: [{scale_tensor.min().item():.10f}, {scale_tensor.max().item():.10f}]")
                print(f"   Scale 平均: {scale_tensor.mean().item():.10f}")
                print(f"\n   前5个通道:")
                print(f"     {'Channel':>10s} {'Scale':>20s} {'Reciprocal':>15s}")
                for ch_idx in range(min(5, num_channels)):
                    scale_val = scale_tensor.flatten()[ch_idx].item()
                    recip = 1/scale_val if scale_val != 0 else float('inf')
                    print(f"     {ch_idx:>10d} {scale_val:>20.10f} {recip:>15.2f}")
                
                if num_channels > 10:
                    print(f"     ... (省略中间 {num_channels-10} 个通道) ...")
                    print(f"\n   后5个通道:")
                    for ch_idx in range(num_channels - 5, num_channels):
                        scale_val = scale_tensor.flatten()[ch_idx].item()
                        recip = 1/scale_val if scale_val != 0 else float('inf')
                        print(f"     {ch_idx:>10d} {scale_val:>20.10f} {recip:>15.2f}")
        else:
            # 标量 scale
            print(f"   Scale: {info['scale']:.8f}")
            print(f"   Scale_reciprocal: {1/info['scale']:.2f}")
        
        print(f"   Zero Point (Offset): {info['zero_point']:.2f}")
        print(f"   Symmetric: {info['symmetric']}")
        print(f"   Shape: {info['shape']}")
    
    print(f"\n总共 {len(quantizer_info)} 个量化器")
    print("="*80 + "\n")
    
    return quantizer_info


def get_single_quantizer_info(quantizer) -> Dict:
    """
    获取单个量化器的信息
    
    Args:
        quantizer: 量化器对象
        
    Returns:
        包含量化器信息的字典
    """
    info = {}
    info['qmin'] = quantizer.qmin
    info['qmax'] = quantizer.qmax
    info['symmetric'] = quantizer.symmetric
    info['shape'] = quantizer.shape
    
    # 获取min和max
    min_val = quantizer.get_min()
    max_val = quantizer.get_max()
    
    # 转换为标量（如果是张量的话）
    if isinstance(min_val, torch.Tensor):
        info['min'] = min_val.min().item()  # 取最小值作为代表
        info['max'] = max_val.max().item()  # 取最大值作为代表
        info['min_tensor'] = min_val
        info['max_tensor'] = max_val
    else:
        info['min'] = min_val
        info['max'] = max_val
    
    # 获取scale和offset
    scale = quantizer.get_scale()
    offset = quantizer.get_offset()
    
    if isinstance(scale, torch.Tensor):
        info['scale'] = scale.mean().item()  # 取平均值作为代表
        info['scale_tensor'] = scale
    else:
        info['scale'] = scale
    
    if offset is not None:
        if isinstance(offset, torch.Tensor):
            info['zero_point'] = offset.mean().item()
            info['zero_point_tensor'] = offset
        else:
            info['zero_point'] = offset
    else:
        info['zero_point'] = 0.0
    
    return info


def is_power_of_2(value: float, relative_tolerance: float = 0.02) -> bool:
    """
    判断一个值是否接近 2 的幂次方（考虑相对误差）
    
    由于量化校准时的浮点数误差和统计波动，real_range 可能不会精确等于 2^n。
    例如：real_range = 2.007884，非常接近 2.0 (误差 0.39%)，应该被视为"是 2^n"。
    
    Args:
        value: 要检查的值
        relative_tolerance: 相对容差（默认 2%）
            - 如果 |value - 2^n| / 2^n < relative_tolerance，则认为是 2^n
            - 例如：2% 容差允许 [1.96, 2.04] 范围内的值被视为 2.0
        
    Returns:
        是否接近 2 的幂次方
        
    Examples:
        >>> is_power_of_2(2.0)          # True (精确匹配)
        >>> is_power_of_2(2.007884)     # True (误差 0.39% < 2%)
        >>> is_power_of_2(2.05)         # False (误差 2.5% > 2%)
        >>> is_power_of_2(1.98)         # True (误差 1% < 2%)
    """
    if value <= 0:
        return False
    
    # 计算 log2(value)
    log2_val = math.log2(value)
    
    # 找到最接近的整数 n
    n_nearest = round(log2_val)
    
    # 计算最接近的 2^n
    nearest_power_of_2 = 2 ** n_nearest
    
    # 计算相对误差
    relative_error = abs(value - nearest_power_of_2) / nearest_power_of_2
    
    # 检查相对误差是否在容差范围内
    return relative_error < relative_tolerance


def find_closest_power_of_2_scale(scale: float, method: str = "round", 
                                   qmin: float = None, qmax: float = None,
                                   rmin: float = None, rmax: float = None,
                                   tolerance: float = 0.02) -> Tuple[float, int]:
    """
    找到最接近给定scale的2的幂次方scale (1/2^n)
    
    Args:
        scale: 原始scale值
        method: 选择策略
            - "round": 四舍五入到最近的 2^n（默认）
            - "cover_range": 智能策略
                - 如果 real_max - real_min 接近 2^n（相对误差 < tolerance）：使用四舍五入
                - 否则：向上取整以覆盖原范围
        qmin: 量化最小值（method="cover_range" 时需要）
        qmax: 量化最大值（method="cover_range" 时需要）
        rmin: 浮点最小值（method="cover_range" 时需要）
        rmax: 浮点最大值（method="cover_range" 时需要）
        tolerance: 判断是否为 2^n 的相对容差（默认 2%）
            - 例如：real_range=2.007884 与 2.0 的相对误差 0.39% < 2%，被视为 2^n
        
    Returns:
        (new_scale, n): 新的scale值和对应的n值，其中new_scale = 1/2^n
    """
    if scale <= 0:
        raise ValueError(f"Scale必须为正数，得到: {scale}")
    
    # 计算log2(1/scale) = -log2(scale)
    # 如果scale = 1/2^n, 那么log2(1/scale) = n
    n = -math.log2(scale)
    
    if method == "round":
        # 四舍五入到最近的整数（最小化 |new_scale - old_scale|）
        n_rounded = round(n)
    elif method == "cover_range":
        # 智能策略：判断 real_range 是否接近 2^n
        if qmin is None or qmax is None or rmin is None or rmax is None:
            raise ValueError("cover_range 方法需要提供 qmin, qmax, rmin, rmax 参数")
        
        # 计算 real_range
        real_range = abs(rmax - rmin)
        
        # 判断 real_range 是否接近 2^n（使用相对容差）
        if is_power_of_2(real_range, relative_tolerance=tolerance):
            # real_range 接近 2^n，使用四舍五入
            n_rounded = round(n)
        else:
            # real_range 不接近 2^n，向上取整以覆盖原范围
            # 向上取整 = floor(n)，这样 scale 会更大，覆盖范围更大
            n_rounded = math.floor(n)
    else:
        raise ValueError(f"不支持的 method: {method}，必须是 'round' 或 'cover_range'")
    
    # 计算新的scale = 1/2^n
    new_scale = 1.0 / (2 ** n_rounded)
    
    return new_scale, n_rounded


def apply_power_of_2_quantization(model, method: str = "round", tolerance: float = 0.02, verbose=True):
    """
    将模型中所有量化器的 scale 修改为 2 的幂次方
    
    新逻辑（四舍五入策略，最小化量化误差）：
    1. 四舍五入 scale 到最近的 2^n
    2. 保持 Qmin, Qmax 不变（量化域固定）
    3. 反向调整 rmin, rmax:
       - 对称量化（权重）：围绕中心点调整 rmin 和 rmax
       - 非对称量化（激活）：保持 rmin 不变，调整 rmax
       
    效果：自动平衡"量化空间浪费"和"极端值裁剪"，选择误差最小的方案
    
    Args:
        model: 量化模型
        method: Power-of-2 选择策略
            - "round": 全部四舍五入（默认）
            - "cover_range": 智能策略，优先覆盖原范围
        tolerance: 判断 real_range 是否接近 2^n 的相对容差（默认 2%）
            - 仅在 method="cover_range" 时生效
            - 例如：real_range=2.007884 与 2.0 的相对误差 0.39% < 2%，被视为 2^n
        verbose: 是否打印详细信息
        
    Returns:
        修改统计信息
    """
    print("\n" + "="*80)
    print(f"应用 Power-of-2 量化 (method={method})")
    print("="*80)
    
    if method == "cover_range":
        print(f"策略: 如果 real_range 接近 2^n (容差 {tolerance*100:.1f}%)，使用四舍五入；否则向上覆盖原范围")
    elif method == "round":
        print("策略: 全部使用四舍五入到最近的 2^n")
    
    total_quantizers = 0
    modified_quantizers = 0
    
    for name, module in model.named_modules():
        # 处理输入量化器（只在 QuantizationMixin 模块中）
        if isinstance(module, QuantizationMixin) and hasattr(module, 'input_quantizers'):
            for idx, quantizer in enumerate(module.input_quantizers):
                if quantizer is not None and quantizer.is_initialized():
                    total_quantizers += 1
                    if modify_quantizer_to_power_of_2(quantizer, f"{name}.input_quantizer[{idx}]", method, tolerance, verbose):
                        modified_quantizers += 1
        
        # 处理输出量化器（只在 QuantizationMixin 模块中）
        if isinstance(module, QuantizationMixin) and hasattr(module, 'output_quantizers'):
            for idx, quantizer in enumerate(module.output_quantizers):
                if quantizer is not None and quantizer.is_initialized():
                    total_quantizers += 1
                    if modify_quantizer_to_power_of_2(quantizer, f"{name}.output_quantizer[{idx}]", method, tolerance, verbose):
                        modified_quantizers += 1
        
        # 处理参数量化器（所有模块，包括非 QuantizationMixin）
        if hasattr(module, 'param_quantizers'):
            for param_name, quantizer in module.param_quantizers.items():
                if quantizer is not None and quantizer.is_initialized():
                    total_quantizers += 1
                    if modify_quantizer_to_power_of_2(quantizer, f"{name}.param_quantizer[{param_name}]", method, tolerance, verbose):
                        modified_quantizers += 1
    
    print(f"\n总共处理 {total_quantizers} 个量化器")
    print(f"成功修改 {modified_quantizers} 个量化器为 Power-of-2 scale")
    print("="*80 + "\n")
    
    return {
        'total': total_quantizers,
        'modified': modified_quantizers
    }


def modify_quantizer_to_power_of_2(quantizer, quantizer_name: str, method: str = "round", 
                                   tolerance: float = 0.02, verbose: bool = True) -> bool:
    """
    修改单个量化器的 scale 为 2 的幂次方
    
    新逻辑（四舍五入策略）：
    1. 计算原始 scale = (rmax - rmin) / (qmax - qmin)
    2. 四舍五入 scale 到最近的 2^n（最小化量化误差）
    3. 保持 Qmin, Qmax 不变（量化域固定）
    4. 反向调整 rmin, rmax:
       - 对称量化（权重）：围绕中心点调整 rmin 和 rmax
       - 非对称量化（激活）：保持 rmin 不变，调整 rmax
       
    效果：
    - 如果 scale 向上调整 → rmax 变大（浪费少量量化空间）
    - 如果 scale 向下调整 → rmax 变小（裁剪少量极端值）
    - 自动选择误差最小的方案
    
    Args:
        quantizer: 量化器对象
        quantizer_name: 量化器名称（用于打印）
        method: Power-of-2 选择策略 ("round" 或 "cover_range")
        tolerance: 判断 real_range 是否接近 2^n 的相对容差（默认 2%）
        verbose: 是否打印详细信息
        
    Returns:
        是否成功修改
    """
    try:
        # 获取原始参数（在任何修改之前）
        old_min = quantizer.get_min()
        old_max = quantizer.get_max()
        old_scale = quantizer.get_scale()
        old_offset = quantizer.get_offset()
        
        qmin = quantizer.qmin
        qmax = quantizer.qmax
        symmetric = quantizer.symmetric
        
        # 确保量化器使用scale-offset参数化
        # 这样我们可以直接修改scale参数
        quantizer._reparametrize_to_scale_offset()
        
        # 如果是张量，需要对每个元素分别处理
        if isinstance(old_scale, torch.Tensor):
            new_scale = torch.zeros_like(old_scale)
            
            # 展平处理
            old_scale_flat = old_scale.flatten()
            new_scale_flat = new_scale.flatten()
            old_min_flat = old_min.flatten() if isinstance(old_min, torch.Tensor) else None
            old_max_flat = old_max.flatten() if isinstance(old_max, torch.Tensor) else None
            
            for i in range(old_scale_flat.numel()):
                scale_val = old_scale_flat[i].item()
                
                # 准备 cover_range 方法需要的参数
                if method == "cover_range" and old_min_flat is not None and old_max_flat is not None:
                    rmin_val = old_min_flat[i].item()
                    rmax_val = old_max_flat[i].item()
                    new_scale_val, n = find_closest_power_of_2_scale(
                        scale_val, method=method, 
                        qmin=qmin, qmax=qmax, 
                        rmin=rmin_val, rmax=rmax_val,
                        tolerance=tolerance
                    )
                else:
                    # 找到最接近的2的幂次方scale（四舍五入）
                    new_scale_val, n = find_closest_power_of_2_scale(scale_val, method="round")
                
                new_scale_flat[i] = new_scale_val
            
            # 重塑回原始形状
            new_scale = new_scale_flat.view(old_scale.shape)
            
            if verbose:
                # 判断 scale 是增大还是减小（取代表值）
                old_scale_repr = old_scale.mean().item()
                new_scale_repr = new_scale.mean().item()
                scale_direction = "↑" if new_scale_repr > old_scale_repr else "↓"
                scale_change_ratio = (new_scale_repr / old_scale_repr - 1) * 100
                
                print(f"\n修改 {quantizer_name}:")
                print(f"  原始 scale 范围: [{old_scale.min().item():.8f}, {old_scale.max().item():.8f}]")
                print(f"  原始 min 范围: [{old_min.min().item():.6f}, {old_min.max().item():.6f}]")
                print(f"  原始 max 范围: [{old_max.min().item():.6f}, {old_max.max().item():.6f}]")
                if old_offset is not None:
                    print(f"  原始 offset 范围: [{old_offset.min().item():.2f}, {old_offset.max().item():.2f}]")
                print(f"  新的 scale 范围: [{new_scale.min().item():.8f}, {new_scale.max().item():.8f}]")
                print(f"  Scale 调整方向: {scale_direction} ({scale_change_ratio:+.1f}%)")
        else:
            # 标量情况
            old_scale_val = old_scale.item() if hasattr(old_scale, 'item') else old_scale
            old_min_val = old_min.item() if hasattr(old_min, 'item') else old_min
            old_max_val = old_max.item() if hasattr(old_max, 'item') else old_max
            
            # 找到最接近的2的幂次方scale
            if method == "cover_range":
                new_scale_val, n = find_closest_power_of_2_scale(
                    old_scale_val, method=method,
                    qmin=qmin, qmax=qmax,
                    rmin=old_min_val, rmax=old_max_val,
                    tolerance=tolerance
                )
            else:
                new_scale_val, n = find_closest_power_of_2_scale(old_scale_val, method="round")
            
            new_scale = torch.tensor(new_scale_val, dtype=old_scale.dtype)
            
            if verbose:
                # 判断 scale 是增大还是减小
                old_scale_val_orig = old_scale.item()
                scale_direction = "↑" if new_scale_val > old_scale_val_orig else "↓"
                scale_change_ratio = (new_scale_val / old_scale_val_orig - 1) * 100
                
                print(f"\n修改 {quantizer_name}:")
                print(f"  原始 scale: {old_scale_val_orig:.8f}")
                print(f"  原始范围: [{old_min.item():.6f}, {old_max.item():.6f}]")
                if old_offset is not None:
                    print(f"  原始 offset: {old_offset.item():.2f}")
                print(f"  新的 scale: {new_scale_val:.8f} (1/2^{n})")
                print(f"  Scale 调整方向: {scale_direction} ({scale_change_ratio:+.1f}%)")
        
        # ！！！新策略：保持 Qmin, Qmax 不变，修改 rmin, rmax！！！
        # 1. 修改 scale 为 power-of-2
        # 2. 保持 Qmin, Qmax 不变（量化域固定）
        # 3. 对称量化（权重）：offset 保持不变（通常为0），重新计算 rmin, rmax
        # 4. 非对称量化（激活）：重新计算 offset, rmin, rmax
        
        if isinstance(new_scale, torch.Tensor):
            # Per-channel情况
            if symmetric or old_offset is None:
                # 对称量化：保持 qmin, qmax, offset 不变，重新计算 rmin, rmax
                new_qmin_val = qmin  # 保持不变
                new_qmax_val = qmax  # 保持不变
                new_offset = old_offset  # 保持不变（通常为0）
                
                # 重新计算 rmin, rmax
                # rmin = scale * (qmin + offset)
                # rmax = scale * (qmax + offset)
                if new_offset is not None:
                    new_min = new_scale * (new_qmin_val + new_offset)
                    new_max = new_scale * (new_qmax_val + new_offset)
                else:
                    new_min = new_scale * new_qmin_val
                    new_max = new_scale * new_qmax_val
            else:
                # 非对称量化：保持 qmin, qmax 不变，保持 rmin 不变，调整 rmax
                new_qmin_val = qmin  # 保持不变
                new_qmax_val = qmax  # 保持不变
                
                # 保持 rmin 不变，重新计算 offset
                # rmin = scale * (qmin + offset)
                # offset = rmin / scale - qmin
                new_offset = old_min / new_scale - qmin
                
                # 重新计算 rmax（会变大，因为 scale 变大了）
                # rmax = scale * (qmax + offset)
                new_max = new_scale * (qmax + new_offset)
                new_min = old_min  # 保持不变
        else:
            # Scalar情况
            old_min_val = old_min.item() if hasattr(old_min, 'item') else old_min
            old_max_val = old_max.item() if hasattr(old_max, 'item') else old_max
            new_scale_val = new_scale.item() if hasattr(new_scale, 'item') else new_scale
            
            if symmetric or old_offset is None:
                # 对称量化：保持 qmin, qmax, offset 不变，重新计算 rmin, rmax
                new_qmin_val = qmin  # 保持不变
                new_qmax_val = qmax  # 保持不变
                
                if old_offset is not None:
                    old_offset_val = old_offset.item() if hasattr(old_offset, 'item') else old_offset
                    new_offset = old_offset  # 保持不变
                    # 重新计算 rmin, rmax
                    new_min_val = new_scale_val * (new_qmin_val + old_offset_val)
                    new_max_val = new_scale_val * (new_qmax_val + old_offset_val)
                else:
                    new_offset = None
                    new_min_val = new_scale_val * new_qmin_val
                    new_max_val = new_scale_val * new_qmax_val
                
                new_min = torch.tensor(new_min_val, dtype=old_scale.dtype)
                new_max = torch.tensor(new_max_val, dtype=old_scale.dtype)
            else:
                # 非对称量化：保持 qmin, qmax 不变，保持 rmin 不变，调整 rmax
                new_qmin_val = qmin  # 保持不变
                new_qmax_val = qmax  # 保持不变
                
                # 保持 rmin 不变，重新计算 offset
                # rmin = scale * (qmin + offset)
                # offset = rmin / scale - qmin
                new_offset_val = old_min_val / new_scale_val - qmin
                new_offset = torch.tensor(new_offset_val, dtype=old_scale.dtype)
                
                # 重新计算 rmax（会变大，因为 scale 变大了）
                # rmax = scale * (qmax + offset)
                new_max_val = new_scale_val * (qmax + new_offset_val)
                new_min = old_min  # 保持不变
                new_max = torch.tensor(new_max_val, dtype=old_scale.dtype)
        
        # 修改 scale, qmin, qmax, offset, min, max
        with torch.no_grad():
            quantizer.scale.copy_(new_scale)
            quantizer.qmin = new_qmin_val
            quantizer.qmax = new_qmax_val
            if new_offset is not None and quantizer.offset is not None:
                quantizer.offset.copy_(new_offset)
            # 修改 min, max（重要！）- 使用 AIMET v2 API
            # AIMET v2 使用 set_range() 方法，而不是直接访问 min/max 属性
            try:
                # 尝试使用 AIMET v2 的 set_range() 方法
                quantizer.set_range(new_min, new_max)
            except (AttributeError, TypeError):
                # 兼容旧版本 AIMET v1（直接访问属性）
                try:
                    quantizer.min.copy_(new_min)
                    quantizer.max.copy_(new_max)
                except Exception as e:
                    if verbose:
                        print(f"  ⚠️  无法修改 min/max: {str(e)}")
        
        if verbose:
            # 验证修改后的参数
            verify_scale = quantizer.get_scale()
            verify_min = quantizer.get_min()
            verify_max = quantizer.get_max()
            verify_offset = quantizer.get_offset()
            verify_qmin = quantizer.qmin
            verify_qmax = quantizer.qmax
            
            print(f"  新的 Qmin: {verify_qmin}, Qmax: {verify_qmax} (保持不变 ✓)")
            
            if isinstance(verify_scale, torch.Tensor):
                all_po2 = all(verify_power_of_2_scale(s.item())[0] for s in verify_scale.flatten())
                print(f"  验证: {'✓ 所有scale都是2的幂次方' if all_po2 else '✗ 存在非2的幂次方scale'}")
                print(f"  新的 Scale 范围: [{verify_scale.min().item():.8f}, {verify_scale.max().item():.8f}]")
                print(f"  新的 rmin 范围: [{verify_min.min().item():.6f}, {verify_min.max().item():.6f}]")
                print(f"  新的 rmax 范围: [{verify_max.min().item():.6f}, {verify_max.max().item():.6f}]")
                if verify_offset is not None:
                    print(f"  新的 Offset 范围: [{verify_offset.min().item():.2f}, {verify_offset.max().item():.2f}]")
                
                # 显示变化情况和影响
                if symmetric or old_offset is None:
                    print(f"  → 对称量化：rmin 和 rmax 都已调整")
                else:
                    # 计算 rmax 变化（取代表值）
                    old_rmax = old_max.max().item()
                    new_rmax = verify_max.max().item()
                    rmax_change = ((new_rmax - old_rmax) / old_rmax) * 100 if old_rmax != 0 else 0
                    if new_rmax > old_rmax:
                        print(f"  → 非对称量化：rmin 保持不变，rmax 调大 ({rmax_change:+.1f}%，浪费部分量化空间）")
                    elif new_rmax < old_rmax:
                        print(f"  → 非对称量化：rmin 保持不变，rmax 调小 ({rmax_change:.1f}%，裁剪极端值）")
                    else:
                        print(f"  → 非对称量化：rmin 和 rmax 保持不变（完美匹配！）")
            else:
                is_po2, n = verify_power_of_2_scale(verify_scale.item())
                print(f"  验证: {'✓ scale是2的幂次方' if is_po2 else '✗ scale不是2的幂次方'}")
                if is_po2:
                    print(f"  新 Scale = 1/2^{n} = {verify_scale.item():.8f}")
                print(f"  新的 rmin: {verify_min.item():.6f}, rmax: {verify_max.item():.6f}")
                if verify_offset is not None:
                    print(f"  新的 Offset: {verify_offset.item():.2f}")
                
                # 显示变化情况和影响
                if symmetric or old_offset is None:
                    print(f"  → 对称量化：rmin 和 rmax 都已调整")
                else:
                    # 计算 rmax 变化
                    old_rmax = old_max.item() if hasattr(old_max, 'item') else old_max
                    new_rmax = verify_max.item()
                    rmax_change = ((new_rmax - old_rmax) / old_rmax) * 100 if old_rmax != 0 else 0
                    if new_rmax > old_rmax:
                        print(f"  → 非对称量化：rmin 保持不变，rmax 调大 ({rmax_change:+.1f}%，浪费部分量化空间）")
                    elif new_rmax < old_rmax:
                        print(f"  → 非对称量化：rmin 保持不变，rmax 调小 ({rmax_change:.1f}%，裁剪极端值）")
                    else:
                        print(f"  → 非对称量化：rmin 和 rmax 保持不变（完美匹配！）")
        
        return True
        
    except Exception as e:
        if verbose:
            print(f"\n警告: 修改 {quantizer_name} 失败: {str(e)}")
            import traceback
            traceback.print_exc()
        return False


def verify_power_of_2_scale(scale: float, tolerance: float = 1e-9) -> Tuple[bool, int]:
    """
    验证给定的scale是否为2的幂次方
    
    Args:
        scale: 要验证的scale值
        tolerance: 容差
        
    Returns:
        (is_power_of_2, n): 是否为2的幂次方，以及对应的n值
    """
    if scale <= 0:
        return False, -1
    
    # 计算log2(1/scale)
    n = -math.log2(scale)
    n_rounded = round(n)
    
    # 检查是否接近整数
    if abs(n - n_rounded) < tolerance:
        expected_scale = 1.0 / (2 ** n_rounded)
        if abs(scale - expected_scale) / expected_scale < tolerance:
            return True, n_rounded
    
    return False, -1


def verify_model_power_of_2(model):
    """
    验证模型中所有量化器的scale是否都是2的幂次方
    
    Args:
        model: 量化模型
    """
    print("\n" + "="*80)
    print("验证 Power-of-2 量化")
    print("="*80)
    
    total_quantizers = 0
    power_of_2_quantizers = 0
    
    for name, module in model.named_modules():
        quantizers_to_check = []
        
        # 检查输入量化器（只在 QuantizationMixin 模块中）
        if isinstance(module, QuantizationMixin) and hasattr(module, 'input_quantizers'):
            for idx, q in enumerate(module.input_quantizers):
                if q is not None and q.is_initialized():
                    quantizers_to_check.append((q, f"{name}.input_quantizer[{idx}]"))
        
        # 检查输出量化器（只在 QuantizationMixin 模块中）
        if isinstance(module, QuantizationMixin) and hasattr(module, 'output_quantizers'):
            for idx, q in enumerate(module.output_quantizers):
                if q is not None and q.is_initialized():
                    quantizers_to_check.append((q, f"{name}.output_quantizer[{idx}]"))
        
        # 检查参数量化器（所有模块，包括非 QuantizationMixin）
        if hasattr(module, 'param_quantizers'):
            for param_name, q in module.param_quantizers.items():
                if q is not None and q.is_initialized():
                    quantizers_to_check.append((q, f"{name}.param_quantizer[{param_name}]"))
        
        # 处理收集到的量化器
        if quantizers_to_check:
            
            for quantizer, qname in quantizers_to_check:
                total_quantizers += 1
                scale = quantizer.get_scale()
                
                if isinstance(scale, torch.Tensor):
                    # 检查张量中的所有元素
                    all_power_of_2 = True
                    for s in scale.flatten():
                        is_po2, n = verify_power_of_2_scale(s.item())
                        if not is_po2:
                            all_power_of_2 = False
                            break
                    
                    if all_power_of_2:
                        power_of_2_quantizers += 1
                        print(f"✓ {qname}: 所有scale都是2的幂次方")
                    else:
                        print(f"✗ {qname}: 存在非2的幂次方scale")
                else:
                    is_po2, n = verify_power_of_2_scale(scale.item())
                    if is_po2:
                        power_of_2_quantizers += 1
                        print(f"✓ {qname}: scale = 1/2^{n}")
                    else:
                        print(f"✗ {qname}: scale = {scale.item():.8f} (非2的幂次方)")
    
    print(f"\n验证结果: {power_of_2_quantizers}/{total_quantizers} 个量化器使用 Power-of-2 scale")
    print("="*80 + "\n")
    
    return power_of_2_quantizers == total_quantizers


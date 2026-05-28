"""
对比power-of-2转换前后的量化器参数
"""
import sys
sys.path.append(r'/home/sdong/Program/aimet/TrainingExtensions/torch/src/python')
sys.path.append(r'/home/sdong/Program/aimet/TrainingExtensions/common/src/python')

import torch

def collect_quantizer_info(model):
    """
    收集模型中所有量化器的信息
    
    注意：现在也会收集非 QuantizationMixin 模块的 param_quantizers
    （例如通过 AIMET 补丁自动创建的参数量化器）
    """
    info_list = []
    
    for name, module in model.named_modules():
        # Output quantizers（只在 QuantizationMixin 模块中）
        if hasattr(module, 'output_quantizers'):
            output_quantizers = module.output_quantizers
            
            if isinstance(output_quantizers, dict):
                items = output_quantizers.items()
            elif hasattr(output_quantizers, '__iter__'):
                items = enumerate(output_quantizers)
            else:
                continue
            
            for qname, quantizer in items:
                if quantizer is not None and hasattr(quantizer, 'is_initialized') and quantizer.is_initialized():
                    info = extract_quantizer_params(quantizer)
                    info['name'] = f"{name}.output_quantizer[{qname}]"
                    info_list.append(info)
        
        # Input quantizers（只在 QuantizationMixin 模块中）
        if hasattr(module, 'input_quantizers'):
            input_quantizers = module.input_quantizers
            
            if isinstance(input_quantizers, dict):
                items = input_quantizers.items()
            elif hasattr(input_quantizers, '__iter__'):
                items = enumerate(input_quantizers)
            else:
                continue
            
            for qname, quantizer in items:
                if quantizer is not None and hasattr(quantizer, 'is_initialized') and quantizer.is_initialized():
                    info = extract_quantizer_params(quantizer)
                    info['name'] = f"{name}.input_quantizer[{qname}]"
                    info_list.append(info)
        
        # Param quantizers（所有模块，包括非 QuantizationMixin）
        # 这样可以收集通过补丁自动创建的参数量化器
        if hasattr(module, 'param_quantizers'):
            param_quantizers = module.param_quantizers
            
            # 处理 dict 或 ModuleDict
            if hasattr(param_quantizers, 'items'):
                for param_name, quantizer in param_quantizers.items():
                    if quantizer is not None and hasattr(quantizer, 'is_initialized') and quantizer.is_initialized():
                        info = extract_quantizer_params(quantizer)
                        info['name'] = f"{name}.param_quantizer[{param_name}]"
                        info_list.append(info)
    
    return info_list


def extract_quantizer_params(quantizer):
    """提取单个量化器的参数"""
    scale = quantizer.get_scale()
    min_val = quantizer.get_min()
    max_val = quantizer.get_max()
    offset = quantizer.get_offset()
    qmin = quantizer.qmin
    qmax = quantizer.qmax
    
    # ✅ 正确判断对称性：直接使用 quantizer.symmetric 属性
    # 不能用 qmin == -qmax 判断，因为 8-bit signed 的范围是 [-128, 127]
    # 即使是对称量化，qmin (-128) 也不等于 -qmax (-127)
    # 这是二进制表示的固有限制，不影响对称性
    symmetric = quantizer.symmetric
    
    # 转换为标量或列表
    def to_scalar_or_list(tensor):
        if tensor is None:
            return None
        if isinstance(tensor, torch.Tensor):
            if tensor.numel() == 1:
                return tensor.item()
            else:
                return tensor.tolist()
        return tensor
    
    return {
        'qmin': qmin,
        'qmax': qmax,
        'scale': to_scalar_or_list(scale),
        'offset': to_scalar_or_list(offset),
        'min': to_scalar_or_list(min_val),
        'max': to_scalar_or_list(max_val),
        'symmetric': symmetric
    }


def to_scalar(value):
    """将值转换为标量，递归展开嵌套列表/元组"""
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return to_scalar(value[0])  # 递归展开单元素列表
        # 多元素列表无法转为标量，返回原值
        return value
    
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value
    
    return value


def safe_reciprocal(value):
    """计算倒数，支持标量、列表和张量（包括嵌套列表）"""

    def _calc(v):
        # 递归处理嵌套列表/元组
        if isinstance(v, (list, tuple)):
            if len(v) == 1:
                return _calc(v[0])  # 展开单元素列表
            return [_calc(item) for item in v]
        
        # 转换为浮点数并计算倒数
        try:
            v_float = float(v)
            return float('inf') if v_float == 0 else 1.0 / v_float
        except (TypeError, ValueError) as e:
            print(f"Warning: Cannot convert {v} (type={type(v)}) to float: {e}")
            return None

    if value is None:
        return None

    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _calc(value.item())
        return [_calc(v) for v in value.flatten().tolist()]

    if isinstance(value, (list, tuple)):
        return _calc(value)

    return _calc(value)


def compare_quantizers(before_list, after_list):
    """对比两个量化器信息列表"""
    print("\n" + "="*100)
    print("Power-of-2 转换前后对比")
    print("="*100)
    
    if len(before_list) != len(after_list):
        print(f"警告: 前后量化器数量不一致！before: {len(before_list)}, after: {len(after_list)}")
        return
    
    issues = []
    
    for i, (before, after) in enumerate(zip(before_list, after_list)):
        if before['name'] != after['name']:
            print(f"警告: 量化器顺序不一致！{before['name']} vs {after['name']}")
            continue
        
        name = before['name']
        
        # 检查对称性
        if before['symmetric'] != after['symmetric']:
            issues.append(f"{name}: 对称性改变了！before={before['symmetric']}, after={after['symmetric']}")
        
        # 检查浮点min/max是否基本保持（允许小幅扩展）
        if isinstance(before['min'], (int, float)) and isinstance(after['min'], (int, float)):
            if after['min'] > before['min'] + 0.01:  # min应该向下扩展或保持
                issues.append(f"{name}: min向上移动了！{before['min']:.6f} -> {after['min']:.6f}")
            if after['max'] < before['max'] - 0.01:  # max应该向上扩展或保持
                issues.append(f"{name}: max向下移动了！{before['max']:.6f} -> {after['max']:.6f}")
        
        # 检查对称量化的qmin=-qmax约束
        if after['symmetric'] and after['qmin'] != -after['qmax']:
            issues.append(f"{name}: 对称量化不满足qmin=-qmax！qmin={after['qmin']}, qmax={after['qmax']}")
        
        # 检查非对称量化的qmin是否保持为0
        if not after['symmetric'] and after['qmin'] != 0:
            issues.append(f"{name}: 非对称量化qmin不为0！qmin={after['qmin']}")
        
        # 打印前几个详细对比（示例）
        if i < 5:
            print(f"\n[{i+1}] {name}:")
            print(f"  {'':20s} {'修改前':>25s} {'修改后':>25s}")
            print(f"  {'Qmin':20s} {before['qmin']:>25d} {after['qmin']:>25d}")
            print(f"  {'Qmax':20s} {before['qmax']:>25d} {after['qmax']:>25d}")
            
            # 处理 scale：如果是标量就直接打印，如果是列表就打印详细信息
            if isinstance(before['scale'], (int, float)):
                print(f"  {'Scale':20s} {before['scale']:>25.8f} {after['scale']:>25.8f}")
                print(f"  {'Scale_reciprocal':20s} {1/before['scale']:>25.2f} {1/after['scale']:>25.2f}")
            elif isinstance(before['scale'], list):
                # Per-channel 量化：打印所有 scale
                num_channels = len(before['scale'])
                print(f"  {'Scale (per-channel)':20s} {num_channels} 个通道")
                
                # 如果通道数不多，全部打印；如果很多，打印前几个和统计信息
                if num_channels <= 10:
                    print(f"  {'':20s} {'修改前':>25s} {'修改后':>25s} {'Reciprocal(after)':>20s}")
                    for ch_idx, (b_scale, a_scale) in enumerate(zip(before['scale'], after['scale'])):
                        # 转换为标量（处理可能的嵌套列表）
                        b_scale_scalar = to_scalar(b_scale)
                        a_scale_scalar = to_scalar(a_scale)
                        recip = safe_reciprocal(a_scale_scalar)
                        
                        # 🔧 修复：如果仍然是列表，继续展开或转换为字符串显示
                        if isinstance(b_scale_scalar, (list, tuple)):
                            b_scale_scalar = b_scale_scalar[0] if len(b_scale_scalar) == 1 else str(b_scale_scalar)
                        if isinstance(a_scale_scalar, (list, tuple)):
                            a_scale_scalar = a_scale_scalar[0] if len(a_scale_scalar) == 1 else str(a_scale_scalar)
                        if isinstance(recip, (list, tuple)):
                            recip = recip[0] if len(recip) == 1 else str(recip)
                        
                        # 安全地打印（处理可能的字符串情况）
                        if isinstance(b_scale_scalar, (int, float)) and isinstance(a_scale_scalar, (int, float)) and isinstance(recip, (int, float)):
                            print(f"    Channel {ch_idx:3d}:     {b_scale_scalar:>25.10f} {a_scale_scalar:>25.10f} {recip:>20.2f}")
                        else:
                            print(f"    Channel {ch_idx:3d}:     {str(b_scale_scalar):>25s} {str(a_scale_scalar):>25s} {str(recip):>20s}")
                else:
                    # 打印前5个和后5个
                    print(f"  {'':20s} {'修改前':>25s} {'修改后':>25s} {'Reciprocal(after)':>20s}")
                    for ch_idx in range(min(5, num_channels)):
                        b_scale = to_scalar(before['scale'][ch_idx])
                        a_scale = to_scalar(after['scale'][ch_idx])
                        recip = safe_reciprocal(a_scale)
                        
                        # 🔧 修复：处理可能仍然是列表的情况
                        if isinstance(b_scale, (list, tuple)):
                            b_scale = b_scale[0] if len(b_scale) == 1 else str(b_scale)
                        if isinstance(a_scale, (list, tuple)):
                            a_scale = a_scale[0] if len(a_scale) == 1 else str(a_scale)
                        if isinstance(recip, (list, tuple)):
                            recip = recip[0] if len(recip) == 1 else str(recip)
                        
                        if isinstance(b_scale, (int, float)) and isinstance(a_scale, (int, float)) and isinstance(recip, (int, float)):
                            print(f"    Channel {ch_idx:3d}:     {b_scale:>25.10f} {a_scale:>25.10f} {recip:>20.2f}")
                        else:
                            print(f"    Channel {ch_idx:3d}:     {str(b_scale):>25s} {str(a_scale):>25s} {str(recip):>20s}")
                    
                    if num_channels > 10:
                        print(f"    {'... (省略中间通道) ...':^75s}")
                        
                        for ch_idx in range(num_channels - 5, num_channels):
                            b_scale = to_scalar(before['scale'][ch_idx])
                            a_scale = to_scalar(after['scale'][ch_idx])
                            recip = safe_reciprocal(a_scale)
                            
                            # 🔧 修复：处理可能仍然是列表的情况
                            if isinstance(b_scale, (list, tuple)):
                                b_scale = b_scale[0] if len(b_scale) == 1 else str(b_scale)
                            if isinstance(a_scale, (list, tuple)):
                                a_scale = a_scale[0] if len(a_scale) == 1 else str(a_scale)
                            if isinstance(recip, (list, tuple)):
                                recip = recip[0] if len(recip) == 1 else str(recip)
                            
                            if isinstance(b_scale, (int, float)) and isinstance(a_scale, (int, float)) and isinstance(recip, (int, float)):
                                print(f"    Channel {ch_idx:3d}:     {b_scale:>25.10f} {a_scale:>25.10f} {recip:>20.2f}")
                            else:
                                print(f"    Channel {ch_idx:3d}:     {str(b_scale):>25s} {str(a_scale):>25s} {str(recip):>20s}")
                    
                    # 打印统计信息
                    import math
                    after_scales = after['scale']
                    all_power_of_2 = True
                    power_distribution = {}
                    
                    for a_scale in after_scales:
                        # 转换为标量（处理可能的嵌套列表）
                        a_scale_scalar = to_scalar(a_scale)
                        # 检查是否是 2 的幂次方
                        if isinstance(a_scale_scalar, (int, float)) and a_scale_scalar > 0:
                            n = -math.log2(a_scale_scalar)
                            n_rounded = round(n)
                            if abs(n - n_rounded) < 1e-9:
                                power_distribution[n_rounded] = power_distribution.get(n_rounded, 0) + 1
                            else:
                                all_power_of_2 = False
                        else:
                            all_power_of_2 = False
                    
                    print(f"\n  {'统计信息':20s}")
                    print(f"    {'所有scale都是2的幂次方':30s}: {all_power_of_2}")
                    if power_distribution:
                        print(f"    {'分布统计 (1/2^n)':30s}:")
                        for n in sorted(power_distribution.keys()):
                            count = power_distribution[n]
                            print(f"      n={n:2d} (scale={1/(2**n):.10f}): {count} 个通道")
            
            if isinstance(before['offset'], (int, float)):
                print(f"  {'Offset':20s} {before['offset']:>25.2f} {after['offset']:>25.2f}")
            if isinstance(before['min'], (int, float)):
                print(f"  {'Min (float)':20s} {before['min']:>25.6f} {after['min']:>25.6f}")
            if isinstance(before['max'], (int, float)):
                print(f"  {'Max (float)':20s} {before['max']:>25.6f} {after['max']:>25.6f}")
            print(f"  {'Symmetric':20s} {str(before['symmetric']):>25s} {str(after['symmetric']):>25s}")
    
    # 汇总统计
    print("\n" + "="*100)
    print("统计汇总")
    print("="*100)
    
    qmax_increased = 0
    qmax_decreased = 0
    qmax_unchanged = 0
    
    for before, after in zip(before_list, after_list):
        if after['qmax'] > before['qmax']:
            qmax_increased += 1
        elif after['qmax'] < before['qmax']:
            qmax_decreased += 1
        else:
            qmax_unchanged += 1
    
    total = len(before_list)
    print(f"总量化器数量: {total}")
    print(f"Qmax增大: {qmax_increased} ({qmax_increased/total*100:.1f}%)")
    print(f"Qmax减小: {qmax_decreased} ({qmax_decreased/total*100:.1f}%)")
    print(f"Qmax不变: {qmax_unchanged} ({qmax_unchanged/total*100:.1f}%)")
    
    return issues


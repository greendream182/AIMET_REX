"""
AIMET 量化工具函数库
专注于量化相关的工具函数：Percentile 校准、Power-of-2 量化、量化器分析等
"""
from typing import Dict, Any
import json
from pathlib import Path


# ============================================================================
# Percentile 校准相关函数
# ============================================================================

def setup_percentile_calibration(sim_model, percentile: float = 99.0, num_bins: int = 2048, verbose: bool = True, 
                                only_params: bool = False) -> int:
    """
    将所有量化器的校准方法从 MinMax 改为 Percentile
    
    Args:
        sim_model: AIMET QuantizationSimModel.model
        percentile: 百分位数 (例如 99.0 表示排除最小1%和最大1%的极端值)
        num_bins: 直方图的 bin 数量
        verbose: 是否打印详细信息
        only_params: 如果为 True，只对参数量化器使用 Percentile，激活量化器保持 MinMax
    
    Returns:
        转换的量化器数量
    """
    from aimet_torch.v2.quantization.encoding_analyzer import PercentileEncodingAnalyzer
    
    if verbose:
        print(f"\n设置校准方法为 Percentile ({percentile}% 分位数)...")
        print(f"  这将排除最小 {(100-percentile)/2:.1f}% 和最大 {(100-percentile)/2:.1f}% 的极端数据")
        if only_params:
            print(f"  ⚠️  只对参数量化器使用 Percentile，激活量化器保持 MinMax")
    
    converted_count = 0
    
    # 转换 output_quantizers（如果 only_params=False）
    # 注意：不转换 input_quantizers，因为它们可能接收特殊值（如常量、离散值）
    # 使用 Percentile 直方图可能导致 NaN 或索引越界
    if not only_params:
        for name, module in sim_model.named_modules():
            if hasattr(module, 'output_quantizers'):
                for idx, quantizer in enumerate(module.output_quantizers):
                    if quantizer is not None and hasattr(quantizer, 'encoding_analyzer'):
                        original_shape = quantizer.encoding_analyzer.observer.shape
                        quantizer.encoding_analyzer = PercentileEncodingAnalyzer(
                            shape=original_shape,
                            num_bins=num_bins,
                            percentile=percentile
                        )
                        converted_count += 1
    
    # 转换 param_quantizers（总是转换）
    for name, module in sim_model.named_modules():
        # 转换 param_quantizers
        if hasattr(module, 'param_quantizers'):
            for param_name, quantizer in module.param_quantizers.items():
                if quantizer is not None and hasattr(quantizer, 'encoding_analyzer'):
                    original_shape = quantizer.encoding_analyzer.observer.shape
                    quantizer.encoding_analyzer = PercentileEncodingAnalyzer(
                        shape=original_shape,
                        num_bins=num_bins,
                        percentile=percentile
                    )
                    converted_count += 1
    
    if verbose:
        print(f"✅ 已将 {converted_count} 个量化器的校准方法改为 Percentile")
    
    return converted_count


def setup_selective_percentile_calibration(
    sim_model, 
    percentile: float = 99.0, 
    num_bins: int = 2048, 
    exclude_modules: list = None,
    verbose: bool = True
) -> Dict[str, int]:
    """
    选择性地设置 Percentile 校准，可以排除特定模块
    
    适用场景：
    - GRU/LSTM 等循环结构对 Percentile 敏感，需要使用 MinMax
    - 其他层（Conv2d, Linear）使用 Percentile 效果更好
    
    Args:
        sim_model: AIMET QuantizationSimModel.model
        percentile: 百分位数
        num_bins: 直方图 bin 数量
        exclude_modules: 要排除的模块名称关键字列表 (例如 ['seq_t', 'gru', 'lstm'])
        verbose: 是否打印详细信息
    
    Returns:
        统计字典 {'converted': int, 'excluded': int}
    
    示例:
        # 对 GRU 相关模块使用 MinMax，其他使用 Percentile
        stats = setup_selective_percentile_calibration(
            sim.model,
            exclude_modules=['seq_t', 'gru'],  # 排除 GRU 相关模块
            verbose=True
        )
    """
    from aimet_torch.v2.quantization.encoding_analyzer import PercentileEncodingAnalyzer
    
    if exclude_modules is None:
        exclude_modules = []
    
    if verbose:
        print(f"\n设置选择性 Percentile 校准 ({percentile}% 分位数)...")
        print(f"  排除模块关键字: {exclude_modules if exclude_modules else '无 (全部使用 Percentile)'}")
    
    converted_count = 0
    excluded_quantizer_count = 0  # 统计被排除的量化器数量
    excluded_modules_list = []  # 记录被排除的顶层模块
    
    def should_exclude(module_name: str) -> bool:
        """检查模块名是否应该被排除"""
        for keyword in exclude_modules:
            if keyword.lower() in module_name.lower():
                return True
        return False
    
    # 遍历所有模块
    for name, module in sim_model.named_modules():
        # 检查是否应该排除
        if should_exclude(name):
            # 只记录顶层被排除的模块（避免输出太多子模块）
            if name and '.' not in name[name.find('seq_t'):].replace('seq_t', '', 1):
                # 这是一个 seq_t 顶层模块
                if name not in excluded_modules_list:
                    excluded_modules_list.append(name)
            
            # 统计被排除的量化器数量
            if hasattr(module, 'input_quantizers'):
                for quantizer in module.input_quantizers:
                    if quantizer is not None and hasattr(quantizer, 'encoding_analyzer'):
                        excluded_quantizer_count += 1
            if hasattr(module, 'output_quantizers'):
                for quantizer in module.output_quantizers:
                    if quantizer is not None and hasattr(quantizer, 'encoding_analyzer'):
                        excluded_quantizer_count += 1
            if hasattr(module, 'param_quantizers'):
                for quantizer in module.param_quantizers.values():
                    if quantizer is not None and hasattr(quantizer, 'encoding_analyzer'):
                        excluded_quantizer_count += 1
            continue
        
        # 转换 output_quantizers
        # 注意：不转换 input_quantizers，避免对特殊值（常量、离散值）使用 Percentile 导致错误
        if hasattr(module, 'output_quantizers'):
            for idx, quantizer in enumerate(module.output_quantizers):
                if quantizer is not None and hasattr(quantizer, 'encoding_analyzer'):
                    original_shape = quantizer.encoding_analyzer.observer.shape
                    quantizer.encoding_analyzer = PercentileEncodingAnalyzer(
                        shape=original_shape,
                        num_bins=num_bins,
                        percentile=percentile
                    )
                    converted_count += 1
        
        # 转换 param_quantizers
        if hasattr(module, 'param_quantizers'):
            for param_name, quantizer in module.param_quantizers.items():
                if quantizer is not None and hasattr(quantizer, 'encoding_analyzer'):
                    original_shape = quantizer.encoding_analyzer.observer.shape
                    quantizer.encoding_analyzer = PercentileEncodingAnalyzer(
                        shape=original_shape,
                        num_bins=num_bins,
                        percentile=percentile
                    )
                    converted_count += 1
    
    if verbose:
        print(f"\n✅ 已将 {converted_count} 个量化器改为 Percentile")
        print(f"⏭️  排除了 {excluded_quantizer_count} 个量化器（保持原始 Analyzer）")
        if excluded_modules_list:
            print(f"   排除的顶层模块: {', '.join(excluded_modules_list)}")
    
    return {
        'converted': converted_count,
        'excluded': excluded_quantizer_count,
        'excluded_modules': excluded_modules_list
    }


def verify_percentile_calibration(sim_model, verbose: bool = True) -> Dict[str, int]:
    """
    验证 Percentile 校准设置是否成功
    
    Args:
        sim_model: AIMET QuantizationSimModel.model
        verbose: 是否打印详细信息
    
    Returns:
        统计字典 {'minmax': count, 'percentile': count, 'other': count, 'other_types': dict}
    """
    minmax_count = 0
    percentile_count = 0
    other_count = 0
    other_types = {}  # 记录其他类型的 analyzer
    
    for name, module in sim_model.named_modules():
        # 检查 input_quantizers
        if hasattr(module, 'input_quantizers'):
            for quantizer in module.input_quantizers:
                if quantizer is not None and hasattr(quantizer, 'encoding_analyzer'):
                    analyzer_type = type(quantizer.encoding_analyzer).__name__
                    if 'MinMax' in analyzer_type:
                        minmax_count += 1
                    elif 'Percentile' in analyzer_type:
                        percentile_count += 1
                    else:
                        other_count += 1
                        # 记录具体类型
                        if analyzer_type not in other_types:
                            other_types[analyzer_type] = 0
                        other_types[analyzer_type] += 1
        
        # 检查 output_quantizers
        if hasattr(module, 'output_quantizers'):
            for quantizer in module.output_quantizers:
                if quantizer is not None and hasattr(quantizer, 'encoding_analyzer'):
                    analyzer_type = type(quantizer.encoding_analyzer).__name__
                    if 'MinMax' in analyzer_type:
                        minmax_count += 1
                    elif 'Percentile' in analyzer_type:
                        percentile_count += 1
                    else:
                        other_count += 1
                        # 记录具体类型
                        if analyzer_type not in other_types:
                            other_types[analyzer_type] = 0
                        other_types[analyzer_type] += 1
        
        # 检查 param_quantizers
        if hasattr(module, 'param_quantizers'):
            for quantizer in module.param_quantizers.values():
                if quantizer is not None and hasattr(quantizer, 'encoding_analyzer'):
                    analyzer_type = type(quantizer.encoding_analyzer).__name__
                    if 'MinMax' in analyzer_type:
                        minmax_count += 1
                    elif 'Percentile' in analyzer_type:
                        percentile_count += 1
                    else:
                        other_count += 1
                        # 记录具体类型
                        if analyzer_type not in other_types:
                            other_types[analyzer_type] = 0
                        other_types[analyzer_type] += 1
    
    if verbose:
        print("\n验证校准方法:")
        print(f"  MinMaxEncodingAnalyzer:     {minmax_count} 个")
        print(f"  PercentileEncodingAnalyzer: {percentile_count} 个 ✅")
        print(f"  其他:                       {other_count} 个")
        
        # 显示其他类型的详细信息
        if other_types:
            print(f"\n  其他类型详情:")
            for analyzer_type, count in other_types.items():
                print(f"    - {analyzer_type}: {count} 个")
        
        if percentile_count > 0:
            print(f"\n✅ 替换成功！将使用 Percentile 校准方法")
        else:
            print(f"\n⚠️ 警告：未找到 PercentileEncodingAnalyzer，仍使用默认 MinMax 方法")
    
    return {
        'minmax': minmax_count,
        'percentile': percentile_count,
        'other': other_count,
        'other_types': other_types
    }


# ============================================================================
# Power-of-2 量化相关函数
# ============================================================================

def apply_power_of_2_workflow(sim_model, method: str = "round", tolerance: float = 0.02, 
                              align_bias_scale: bool = False, verbose: bool = True) -> Dict[str, Any]:
    """
    完整的 Power-of-2 量化工作流
    
    包括：
    1. 打印校准后的量化器信息
    2. 收集修改前的量化器信息
    3. 应用 power-of-2 量化
    4. 验证修改结果
    5. 对比修改前后的差异
    6. (可选) Bias Scale 对齐到 Sx * Sw
    
    Args:
        sim_model: AIMET QuantizationSimModel.model
        method: Power-of-2 选择策略
            - "round": 全部四舍五入（默认，现有方法）
            - "cover_range": 智能策略
                - 如果 real_max - real_min 接近 2^n（相对误差 < tolerance）：使用四舍五入
                - 否则：增大 scale 使得覆盖原范围
        tolerance: 判断 real_range 是否接近 2^n 的相对容差（默认 2%）
            - 仅在 method="cover_range" 时生效
            - 例如：real_range=2.007884 与 2.0 的相对误差 0.39% < 2%，被视为 2^n
        align_bias_scale: 是否将卷积层的 bias scale 对齐到 Sx * Sw（默认 False）
            - True: 执行对齐，使得硬件可以直接计算 (Qx*Qw + Qb)
            - False: 跳过对齐
        verbose: 是否打印详细信息
    
    Returns:
        结果字典，包含统计信息和对比结果
    """
    from aimet_torch.power_of_2_quantization import (
        print_quantizer_info,
        apply_power_of_2_quantization,
        verify_model_power_of_2
    )
    from aimet_torch.compare_before_after_po2 import collect_quantizer_info, compare_quantizers
    
    results = {}
    
    # 1. 打印校准后的量化器信息
    if verbose:
        print("\n" + "="*80)
        print("步骤 1: 校准后的量化器信息")
        print("="*80)
    
    original_info = print_quantizer_info(sim_model, "校准后的原始量化参数")
    results['original_info'] = original_info
    
    # 2. 收集修改前的量化器信息
    if verbose:
        print("\n收集修改前的量化器参数...")
    before_po2_info = collect_quantizer_info(sim_model)
    if verbose:
        print(f"✅ 收集了 {len(before_po2_info)} 个量化器的信息")
    results['before_count'] = len(before_po2_info)
    results['before_info'] = before_po2_info
    
    # 3. 应用 Power-of-2 量化
    if verbose:
        print("\n" + "="*80)
        print("步骤 2: 将 scale 修改为 2 的幂次方 (1/2^n)")
        print("="*80)
    
    stats = apply_power_of_2_quantization(sim_model, method=method, tolerance=tolerance, verbose=verbose)
    results['stats'] = stats
    
    # 4. 收集修改后的量化器信息
    if verbose:
        print("\n收集修改后的量化器参数...")
    after_po2_info = collect_quantizer_info(sim_model)
    if verbose:
        print(f"✅ 收集了 {len(after_po2_info)} 个量化器的信息")
    results['after_count'] = len(after_po2_info)
    results['after_info'] = after_po2_info
    
    # 5. 验证修改结果
    if verbose:
        print("\n" + "="*80)
        print("步骤 3: 验证修改后的量化参数")
        print("="*80)
    
    modified_info = print_quantizer_info(sim_model, "修改后的量化参数")
    results['modified_info'] = modified_info
    
    # 6. 验证所有 scale 是否都是 2 的幂次方
    all_power_of_2 = verify_model_power_of_2(sim_model)
    results['all_power_of_2'] = all_power_of_2
    
    if verbose:
        if all_power_of_2:
            print("\n✅ 所有量化器的 scale 已成功修改为 2 的幂次方!")
        else:
            print("\n⚠️ 警告: 部分量化器的 scale 未能修改为 2 的幂次方")
    
    # 7. 对比修改前后的参数
    if verbose:
        print("\n" + "="*80)
        print("步骤 4: 对比修改前后的参数")
        print("="*80)
    
    issues = compare_quantizers(before_po2_info, after_po2_info)
    results['issues'] = issues
    
    # 8. (可选) Bias Scale 对齐到 Sx * Sw
    if align_bias_scale:
        if verbose:
            print("\n" + "="*80)
            print("步骤 5: Bias Scale 对齐（Sb = Sx * Sw）")
            print("="*80)
            print("📝 将卷积层的 bias scale 对齐到输入和权重的 scale 乘积")
            print("📝 目标：使得硬件可以直接计算 Sy*Qy = Sx*Sw*(Qx*Qw + Qb)\n")
        
        bias_alignment_stats = _align_conv_bias_scale(sim_model, verbose=verbose)
        results['bias_alignment'] = bias_alignment_stats
    
    return results


# ============================================================================
# 量化器参数冻结相关函数
# ============================================================================

def freeze_quantizer_parameters(sim_model, verbose: bool = True, freeze_bn_affine: bool = True) -> Dict[str, Any]:
    """
    冻结量化器参数（scale、offset 等），仅训练模型权重
    
    在 QAT 微调时，通常需要冻结量化器的参数（如 scale、offset），
    只训练模型的权重参数（weights、bias）
    
    Args:
        sim_model: AIMET QuantizationSimModel.model
        verbose: 是否打印详细信息
        freeze_bn_affine: 是否冻结 BatchNorm 的 affine 参数（weight、bias）
    
    Returns:
        结果字典，包含：
        - trainable_params: 可训练参数名称列表
        - frozen_params: 冻结参数名称列表
        - frozen_quantizer_params: 冻结的量化器参数列表
        - frozen_bn_params: 冻结的 BN affine 参数列表
        - trainable_count: 可训练参数数量
        - frozen_count: 冻结参数数量
    """
    trainable_params = []
    frozen_params = []
    frozen_quantizer_params = []
    frozen_bn_params = []
    
    # 常见 BN 模块名称模式
    bn_name_patterns = ['_bn', 'bn_', 'pre_bn', 'rnn2d_bn', 'batchnorm', 'batch_norm']
    
    for name, param in sim_model.named_parameters():
        if param.requires_grad:
            should_freeze = False
            freeze_reason = None
            
            # 检查是否是量化器参数
            if 'quantizer' in name and any(kw in name for kw in ['min', 'max', 'scale', 'offset', 'encoding']):
                should_freeze = True
                freeze_reason = 'quantizer'
            
            # 检查是否是 BatchNorm 的 affine 参数（weight 或 bias）
            if freeze_bn_affine and not should_freeze:
                # 检查参数名称是否包含 BN 模块名称模式
                name_lower = name.lower()
                is_bn_param = any(pattern in name_lower for pattern in bn_name_patterns)
                # 检查是否是 weight 或 bias 参数
                is_affine_param = name.endswith('.weight') or name.endswith('.bias')
                
                if is_bn_param and is_affine_param:
                    should_freeze = True
                    freeze_reason = 'bn_affine'
            
            if should_freeze:
                param.requires_grad = False
                frozen_params.append(name)
                if freeze_reason == 'quantizer':
                    frozen_quantizer_params.append(name)
                elif freeze_reason == 'bn_affine':
                    frozen_bn_params.append(name)
            else:
                trainable_params.append(name)
    
    if verbose:
        print(f"冻结量化器参数:         {len(frozen_quantizer_params)} 个")
        if freeze_bn_affine:
            print(f"冻结 BN affine 参数:    {len(frozen_bn_params)} 个")
            if frozen_bn_params:
                print("  冻结的 BN 参数示例:")
                for param_name in frozen_bn_params[:5]:  # 只显示前5个
                    print(f"    - {param_name}")
        print(f"总冻结参数:             {len(frozen_params)} 个")
        print(f"可训练参数:             {len(trainable_params)} 个")
    
    return {
        'trainable_params': trainable_params,
        'frozen_params': frozen_params,
        'frozen_quantizer_params': frozen_quantizer_params,
        'frozen_bn_params': frozen_bn_params,
        'trainable_count': len(trainable_params),
        'frozen_count': len(frozen_params)
    }


def set_train_mode_freeze_bn(model, verbose: bool = False) -> int:
    """
    设置模型为训练模式，但保持 BatchNorm 层为评估模式
    
    在 QAT 微调时，通常需要保持 BatchNorm 层的统计信息不变，
    避免量化后的统计信息与浮点模型差异过大
    
    支持的 BatchNorm 类型：
    - nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d (标准 PyTorch BN)
    - QuantizableBatchNorm1d, QuantizableBatchNorm2d (自定义可量化 BN)
    - prepare_model 展开后的 BN（通过模块名称识别，如 pre_bn, rnn2d_bn）
    
    Args:
        model: PyTorch 模型或 AIMET QuantizationSimModel.model
        verbose: 是否打印详细信息
    
    Returns:
        冻结的 BatchNorm 层数量
    """
    import torch.nn as nn
    
    # 动态导入自定义 BatchNorm（避免循环导入）
    try:
        from aimet_torch.quantizable_batchnorm import QuantizableBatchNorm1d, QuantizableBatchNorm2d
        custom_bn_types = (QuantizableBatchNorm1d, QuantizableBatchNorm2d)
    except ImportError:
        custom_bn_types = ()
    
    # 设置整个模型为训练模式
    model.train()
    
    # 策略 1: 通过类型识别标准 BN（适用于未展开的 BN）
    bn_count = 0
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d) + custom_bn_types
    frozen_modules = set()  # 记录已冻结的模块，避免重复计数
    
    for module in model.modules():
        if isinstance(module, bn_types) and module not in frozen_modules:
            module.eval()
            frozen_modules.add(module)
            bn_count += 1
    
    # 策略 2: 通过名称识别展开后的 BN（适用于 prepare_model 后）
    # 常见 BN 模块名称模式（以 _bn 或 bn_ 结尾/开头的模块）
    bn_name_patterns = ['_bn', 'bn_', 'pre_bn', 'rnn2d_bn', 'batchnorm']
    
    for name, module in model.named_modules():
        # 检查模块名称是否匹配 BN 模式
        name_lower = name.lower()
        is_bn_by_name = any(pattern in name_lower for pattern in bn_name_patterns)
        
        if is_bn_by_name and module not in frozen_modules:
            # 将该模块及其所有子模块设置为 eval 模式
            module.eval()
            frozen_modules.add(module)
            bn_count += 1
            
            # 如果有子模块，也冻结它们
            for child in module.modules():
                if child != module and child not in frozen_modules:
                    child.eval()
                    frozen_modules.add(child)
    
    if verbose:
        print(f"✅ 模型设置为训练模式，{bn_count} 个 BatchNorm 层保持评估模式")
    
    return bn_count


# ============================================================================
# 混合精度位宽配置相关函数
# ============================================================================

def load_mixed_precision_config(config_file: str) -> Dict[str, Any]:
    """
    加载混合精度位宽配置文件
    
    Args:
        config_file: JSON 配置文件路径
    
    Returns:
        配置字典
    """
    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_file}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    return config


# def apply_mixed_precision_bitwidth(sim_model, config_file: str, verbose: bool = True) -> Dict[str, int]:
    """
    根据 JSON 配置文件应用混合精度位宽设置
    
    支持的配置项：
    - weight_bitwidth: 权重位宽
    - bias_bitwidth: 偏置位宽
    - input_bitwidth: 输入位宽
    - output_bitwidth: 输出位宽
    - weight_symmetric: 权重对称量化 (True/False)
    - output_symmetric: 输出对称量化 (True/False)
    - input_symmetric: 输入对称量化 (True/False)
    - per_channel_quantization: 逐通道量化 (True/False)
    - disable_quantization: 禁用量化 (True/False)
    
    支持模式匹配（在 layer_name_config 中）：
    - 使用 "*" 通配符匹配层名称
    - 例如：
      "*.seq_t.cells.0.weight_ih": {...}  匹配所有GRU的weight_ih层
      "enc_seqs.*.seq_t.*": {...}         匹配所有encoder的GRU内部层
    
    配置优先级：
    1. layer_name_config - 精确匹配（优先级最高）
    2. layer_name_config - 模式匹配
    3. layer_type_config（按类型匹配）
    4. default_bitwidth（默认值）
    
    Args:
        sim_model: AIMET QuantizationSimModel.model
        config_file: JSON 配置文件路径
        verbose: 是否打印详细信息
    
    Returns:
        统计字典，包含各类型层的设置数量
    """
    import fnmatch
    
    # 加载配置
    config = load_mixed_precision_config(config_file)
    
    default_bitwidth = config.get('default_bitwidth', {})
    default_config = config.get('default_config', {})  # 读取 default_config 字段
    layer_type_config = config.get('layer_type_config', {})
    layer_name_config = config.get('layer_name_config', {})
    
    # 过滤掉注释字段
    layer_name_config = {k: v for k, v in layer_name_config.items() 
                        if not k.startswith('_') and k not in ['examples', 'comment']}
    
    # 检查默认配置是否要求禁用量化
    default_disable_quantization = default_config.get('disable_quantization', False)
    
    if verbose:
        print("\n" + "="*70)
        print("🔧 应用混合精度位宽配置")
        print("="*70)
        print(f"📄 配置文件: {config_file}")
        print(f"📝 默认位宽: weight={default_bitwidth.get('weight', 8)}-bit, output={default_bitwidth.get('output', 8)}-bit")
        if default_disable_quantization:
            print(f"📝 默认策略: 禁用所有未匹配层的量化")
        print(f"📝 类型配置数量: {len(layer_type_config)} 个层类型")
        print(f"📝 名称配置数量: {len(layer_name_config)} 个具体层")
        print("="*70)
    
    # 统计信息
    stats = {
        'conv_count': 0,
        'linear_count': 0,
        'activation_count': 0,
        'add_count': 0,
        'multiply_count': 0,
        'subtract_count': 0,
        'disabled_count': 0,
        'name_matched_count': 0,
        'type_matched_count': 0,
        'default_count': 0
    }
    
    # 遍历所有模块
    for name, module in sim_model.named_modules():
        module_type = type(module).__name__
        
        # 确定使用哪个配置（优先级：精确名称 > 模式匹配 > 类型 > 默认）
        layer_config = None
        match_type = None
        matched_pattern = None
        
        # 1. 尝试按名称精确匹配
        if name in layer_name_config:
            layer_config = layer_name_config[name]
            match_type = 'name'
            stats['name_matched_count'] += 1

        # 2. 尝试按类型匹配（优先于通配符模式）
        if layer_config is None and module_type in layer_type_config:
            layer_config = layer_type_config[module_type]
            match_type = 'type'
            stats['type_matched_count'] += 1

        # 3. 尝试按模式匹配（支持通配符 *，仅在无类型配置时生效）
        if layer_config is None:
            for pattern, config in layer_name_config.items():
                if '*' in pattern and fnmatch.fnmatch(name, pattern):
                    layer_config = config
                    match_type = 'pattern'
                    matched_pattern = pattern
                    stats['name_matched_count'] += 1
                    break
        
        # 4. 对于未匹配的模块，检查是否需要禁用量化
        if layer_config is None:
            # 检查模块是否有量化器（更可靠的判断方式）
            has_quantizers = (
                (hasattr(module, 'param_quantizers') and module.param_quantizers) or
                (hasattr(module, 'output_quantizers') and len(module.output_quantizers) > 0 and any(q is not None for q in module.output_quantizers)) or
                (hasattr(module, 'input_quantizers') and len(module.input_quantizers) > 0 and any(q is not None for q in module.input_quantizers))
            )
            
            # 如果默认配置要求禁用量化，且模块有量化器，则禁用量化
            if default_disable_quantization and has_quantizers:
                _disable_quantization(module, name, verbose=False)
                stats['disabled_count'] += 1
                continue
            # 否则，对于已量化的层，使用默认位宽配置
            elif has_quantizers:
                layer_config = default_bitwidth
                match_type = 'default'
                stats['default_count'] += 1
        
        # 应用配置
        if layer_config:
            # 检查是否禁用量化
            if layer_config.get('disable_quantization', False):
                _disable_quantization(module, name, verbose)
                stats['disabled_count'] += 1
                continue
            
            # 设置权重量化参数
            if hasattr(module, 'param_quantizers') and 'weight' in module.param_quantizers:
                weight_quantizer = module.param_quantizers['weight']
                if weight_quantizer is not None:
                    changes = []
                    
                    # 位宽
                    weight_bw = layer_config.get('weight_bitwidth')
                    if weight_bw:
                        weight_quantizer.bitwidth = weight_bw
                        # 🔧 FIX: 更新 qmin 和 qmax 以匹配新的位宽
                        if hasattr(weight_quantizer, 'symmetric') and weight_quantizer.symmetric:
                            # 对称量化
                            weight_quantizer.qmin = -(2 ** (weight_bw - 1))
                            weight_quantizer.qmax = 2 ** (weight_bw - 1) - 1
                        else:
                            # 非对称量化
                            weight_quantizer.qmin = 0
                            weight_quantizer.qmax = 2 ** weight_bw - 1
                        changes.append(f"{weight_bw}-bit")
                    
                    # 对称性
                    weight_sym = layer_config.get('weight_symmetric')
                    if weight_sym is not None:
                        weight_quantizer.symmetric = weight_sym
                        # 🔧 如果改变了对称性，需要重新计算 qmin/qmax
                        # 使用配置的位宽或当前的位宽
                        bw = weight_bw if weight_bw else weight_quantizer.bitwidth
                        if weight_sym:
                            weight_quantizer.qmin = -(2 ** (bw - 1))
                            weight_quantizer.qmax = 2 ** (bw - 1) - 1
                        else:
                            weight_quantizer.qmin = 0
                            weight_quantizer.qmax = 2 ** bw - 1
                        changes.append(f"{'sym' if weight_sym else 'asym'}")
                    
                    if changes:
                        if verbose:
                            if match_type == 'pattern':
                                print(f"  ✅ [pattern: {matched_pattern}] {name}.weight: {', '.join(changes)}")
                            else:
                                print(f"  ✅ [{match_type}] {name}.weight: {', '.join(changes)}")
                    
                    if 'Conv' in module_type:
                        stats['conv_count'] += 1
                    elif 'Linear' in module_type:
                        stats['linear_count'] += 1
                    elif 'Add' in module_type:
                        stats['add_count'] += 1
                    elif 'Multiply' in module_type:
                        stats['multiply_count'] += 1
                    elif 'Subtract' in module_type:
                        stats['subtract_count'] += 1
                else:
                    # 调试：量化器是 None
                    if match_type == 'pattern' and 'Linear' in module_type and verbose:
                        print(f"  ⚠️  [pattern: {matched_pattern}] {name}.weight: 量化器是 None，无法设置")
            else:
                # 调试：没有 param_quantizers
                if match_type == 'pattern' and 'Linear' in module_type and verbose:
                    print(f"  ⚠️  [pattern: {matched_pattern}] {name}: 没有 param_quantizers['weight']")
            
            # 设置 bias 量化参数（新增）
            if hasattr(module, 'param_quantizers') and 'bias' in module.param_quantizers:
                bias_quantizer = module.param_quantizers['bias']
                if bias_quantizer is not None:
                    changes = []
                    
                    # 位宽
                    bias_bw = layer_config.get('bias_bitwidth')
                    if bias_bw:
                        bias_quantizer.bitwidth = bias_bw
                        # 🔧 FIX: 更新 qmin 和 qmax 以匹配新的位宽
                        if hasattr(bias_quantizer, 'symmetric') and bias_quantizer.symmetric:
                            # 对称量化
                            bias_quantizer.qmin = -(2 ** (bias_bw - 1))
                            bias_quantizer.qmax = 2 ** (bias_bw - 1) - 1
                        else:
                            # 非对称量化
                            bias_quantizer.qmin = 0
                            bias_quantizer.qmax = 2 ** bias_bw - 1
                        changes.append(f"{bias_bw}-bit")
                    
                    # bias 通常使用对称量化
                    bias_sym = layer_config.get('bias_symmetric')
                    if bias_sym is not None:
                        bias_quantizer.symmetric = bias_sym
                        # 🔧 如果改变了对称性，需要重新计算 qmin/qmax
                        # 使用配置的位宽或当前的位宽
                        bw = bias_bw if bias_bw else bias_quantizer.bitwidth
                        if bias_sym:
                            bias_quantizer.qmin = -(2 ** (bw - 1))
                            bias_quantizer.qmax = 2 ** (bw - 1) - 1
                        else:
                            bias_quantizer.qmin = 0
                            bias_quantizer.qmax = 2 ** bw - 1
                        changes.append(f"{'sym' if bias_sym else 'asym'}")
                    
                    if changes and verbose:
                        if match_type == 'pattern':
                            print(f"  ✅ [pattern: {matched_pattern}] {name}.bias: {', '.join(changes)}")
                        else:
                            print(f"  ✅ [{match_type}] {name}.bias: {', '.join(changes)}")
            
            # 设置输入量化参数
            if hasattr(module, 'input_quantizers') and len(module.input_quantizers) > 0:
                for idx, input_quantizer in enumerate(module.input_quantizers):
                    if input_quantizer is not None:
                        changes = []
                        
                        # 位宽
                        input_bw = layer_config.get('input_bitwidth')
                        if input_bw:
                            input_quantizer.bitwidth = input_bw
                            # 🔧 FIX: 更新 qmin 和 qmax 以匹配新的位宽
                            if hasattr(input_quantizer, 'symmetric') and input_quantizer.symmetric:
                                # 对称量化
                                input_quantizer.qmin = -(2 ** (input_bw - 1))
                                input_quantizer.qmax = 2 ** (input_bw - 1) - 1
                            else:
                                # 非对称量化
                                input_quantizer.qmin = 0
                                input_quantizer.qmax = 2 ** input_bw - 1
                            changes.append(f"{input_bw}-bit")
                        
                        # 对称性
                        input_sym = layer_config.get('input_symmetric')
                        if input_sym is not None:
                            input_quantizer.symmetric = input_sym
                            # 🔧 如果改变了对称性，需要重新计算 qmin/qmax
                            # 使用配置的位宽或当前的位宽
                            bw = input_bw if input_bw else input_quantizer.bitwidth
                            if input_sym:
                                input_quantizer.qmin = -(2 ** (bw - 1))
                                input_quantizer.qmax = 2 ** (bw - 1) - 1
                            else:
                                input_quantizer.qmin = 0
                                input_quantizer.qmax = 2 ** bw - 1
                            changes.append(f"{'sym' if input_sym else 'asym'}")
                        
                        if verbose and changes:
                            # 对于多个输入量化器，显示索引
                            if len(module.input_quantizers) > 1:
                                print(f"  ✅ [{match_type}] {name}.input[{idx}]: {', '.join(changes)}")
                            else:
                                print(f"  ✅ [{match_type}] {name}.input: {', '.join(changes)}")
            
            # 设置输出量化参数
            if hasattr(module, 'output_quantizers') and len(module.output_quantizers) > 0:
                output_quantizer = module.output_quantizers[0]
                output_bw = layer_config.get('output_bitwidth')
                output_sym = layer_config.get('output_symmetric')
                if output_quantizer is None and output_bw is not None:
                    try:
                        from aimet_torch.v2.quantization.affine import QuantizeDequantize
                        import torch.nn as _nn
                        sym = output_sym if output_sym is not None else True
                        new_q = QuantizeDequantize(shape=(), bitwidth=output_bw, symmetric=sym)
                        slots = list(module.output_quantizers)
                        slots[0] = new_q
                        module.output_quantizers = _nn.ModuleList(slots)
                        output_quantizer = new_q
                        if verbose:
                            label = f"pattern: {matched_pattern}" if match_type == 'pattern' else match_type
                            print(f"  ➕ [{label}] {name}.output[0]: 创建新 quantizer {output_bw}-bit {'sym' if sym else 'asym'}")
                    except Exception as _e:
                        if verbose:
                            print(f"  ⚠️  {name}.output[0]: 创建 quantizer 失败 - {_e}")
                if output_quantizer is not None:
                    changes = []
                    
                    # 位宽
                    if output_bw:
                        output_quantizer.bitwidth = output_bw
                        # 🔧 FIX: 更新 qmin 和 qmax 以匹配新的位宽
                        if hasattr(output_quantizer, 'symmetric') and output_quantizer.symmetric:
                            # 对称量化
                            output_quantizer.qmin = -(2 ** (output_bw - 1))
                            output_quantizer.qmax = 2 ** (output_bw - 1) - 1
                        else:
                            # 非对称量化
                            output_quantizer.qmin = 0
                            output_quantizer.qmax = 2 ** output_bw - 1
                        changes.append(f"{output_bw}-bit")
                    
                    # 对称性
                    if output_sym is not None:
                        output_quantizer.symmetric = output_sym
                        # 🔧 如果改变了对称性，需要重新计算 qmin/qmax
                        # 使用配置的位宽或当前的位宽
                        bw = output_bw if output_bw else output_quantizer.bitwidth
                        if output_sym:
                            output_quantizer.qmin = -(2 ** (bw - 1))
                            output_quantizer.qmax = 2 ** (bw - 1) - 1
                        else:
                            output_quantizer.qmin = 0
                            output_quantizer.qmax = 2 ** bw - 1
                        changes.append(f"{'sym' if output_sym else 'asym'}")
                    
                    if verbose and changes:
                        # 只在没有设置权重时打印，避免重复
                        if not (hasattr(module, 'param_quantizers') and 'weight' in module.param_quantizers):
                            print(f"  ✅ [{match_type}] {name}.output: {', '.join(changes)}")
                    
                    # 统计不同类型的层（只统计有输出量化器的层）
                    if 'Tanh' in module_type or 'Sigmoid' in module_type or 'ReLU' in module_type:
                        stats['activation_count'] += 1
                    elif 'Add' in module_type:
                        stats['add_count'] += 1
                    elif 'Multiply' in module_type:
                        stats['multiply_count'] += 1
                    elif 'Subtract' in module_type:
                        stats['subtract_count'] += 1
    
    # 打印统计
    if verbose:
        print("="*70)
        print(f"📊 位宽设置统计:")
        print(f"  • Conv 层:         {stats['conv_count']} 个")
        print(f"  • Linear 层:       {stats['linear_count']} 个")
        print(f"  • Add 层:          {stats['add_count']} 个")
        print(f"  • Multiply 层:     {stats['multiply_count']} 个")
        print(f"  • Subtract 层:     {stats['subtract_count']} 个")
        print(f"  • 激活函数层:      {stats['activation_count']} 个")
        print(f"  • 禁用量化层:      {stats['disabled_count']} 个")
        print(f"  • 按名称匹配:      {stats['name_matched_count']} 个")
        print(f"  • 按类型匹配:      {stats['type_matched_count']} 个")
        print(f"  • 使用默认值:      {stats['default_count']} 个")
        print("="*70)
    
    return stats

def apply_mixed_precision_bitwidth(sim_model, config_file: str, verbose: bool = True) -> Dict[str, int]:
    """
    根据 JSON 配置文件应用混合精度位宽设置
    
    支持的配置项：
    - weight_bitwidth: 权重位宽
    - bias_bitwidth: 偏置位宽
    - input_bitwidth: 输入位宽
    - output_bitwidth: 输出位宽
    - param_bitwidth: 通用参数位宽，作用于所有非 weight/bias 的 param_quantizers 键
                      （如 Snake2d 的 alpha、LayerNorm 的 gamma/beta 等）
    - weight_symmetric: 权重对称量化 (True/False)
    - output_symmetric: 输出对称量化 (True/False)
    - input_symmetric: 输入对称量化 (True/False)
    - param_symmetric: 通用参数对称量化 (True/False)，与 param_bitwidth 配合使用
    - per_channel_quantization: 逐通道量化 (True/False)
    - disable_quantization: 禁用量化 (True/False)
    
    支持模式匹配（在 layer_name_config 中）：
    - 使用 "*" 通配符匹配层名称
    - 例如：
      "*.seq_t.cells.0.weight_ih": {...}  匹配所有GRU的weight_ih层
      "enc_seqs.*.seq_t.*": {...}         匹配所有encoder的GRU内部层
    
    配置优先级：
    1. layer_name_config - 精确匹配（优先级最高）
    2. layer_name_config - 模式匹配
    3. layer_type_config（按类型匹配）
    4. default_bitwidth（默认值）
    
    Args:
        sim_model: AIMET QuantizationSimModel.model
        config_file: JSON 配置文件路径
        verbose: 是否打印详细信息
    
    Returns:
        统计字典，包含各类型层的设置数量
    """
    import fnmatch
    
    # 加载配置
    config = load_mixed_precision_config(config_file)
    
    default_bitwidth = config.get('default_bitwidth', {})
    default_config = config.get('default_config', {})  # 读取 default_config 字段
    layer_type_config = config.get('layer_type_config', {})
    layer_name_config = config.get('layer_name_config', {})
    
    # 过滤掉注释字段
    layer_name_config = {k: v for k, v in layer_name_config.items() 
                        if not k.startswith('_') and k not in ['examples', 'comment']}
    
    # 检查默认配置是否要求禁用量化
    default_disable_quantization = default_config.get('disable_quantization', False)
    
    if verbose:
        print("\n" + "="*70)
        print("🔧 应用混合精度位宽配置")
        print("="*70)
        print(f"📄 配置文件: {config_file}")
        print(f"📝 默认位宽: weight={default_bitwidth.get('weight', 8)}-bit, output={default_bitwidth.get('output', 8)}-bit")
        if default_disable_quantization:
            print(f"📝 默认策略: 禁用所有未匹配层的量化")
        print(f"📝 类型配置数量: {len(layer_type_config)} 个层类型")
        print(f"📝 名称配置数量: {len(layer_name_config)} 个具体层")
        print("="*70)
    
    # 统计信息
    stats = {
        'conv_count': 0,
        'linear_count': 0,
        'activation_count': 0,
        'add_count': 0,
        'multiply_count': 0,
        'subtract_count': 0,
        'disabled_count': 0,
        'name_matched_count': 0,
        'type_matched_count': 0,
        'default_count': 0
    }

    # QuantGRU 模块由内部的 use_quantization 属性管理量化开关，
    # 不走 AIMET 的通用 input/output/param_quantizers 流程；提前导入用于跳过判断。
    try:
        from quant_gru import QuantGRU as _QuantGRUType
    except ImportError:
        _QuantGRUType = None

    # 遍历所有模块
    for name, module in sim_model.named_modules():
        module_type = type(module).__name__

        # 跳过 QuantGRU：它的 param_quantizers 含 weight_ih_l0/weight_hh_l0 等占位键
        # （值均为 None），会让通用启发式误判为"有量化器"并错误地计入 disabled_count。
        # QuantGRU 的位宽配置由后面单独的 load_bitwidth_config 处理。
        if _QuantGRUType is not None and isinstance(module, _QuantGRUType):
            continue

        # 确定使用哪个配置
        # 优先级：精确名称 > 类型 > 模式匹配 > 默认
        #   类型 > 模式：避免宽泛通配符（如 *.act*）意外覆盖子模块的类型配置；
        #   模式匹配保留在类型之后，专门用于无类型配置的父 Module（如 Snake2d 包装层的 alpha）。
        layer_config = None
        match_type = None
        matched_pattern = None

        # 1. 尝试按名称精确匹配
        if name in layer_name_config:
            layer_config = layer_name_config[name]
            match_type = 'name'
            stats['name_matched_count'] += 1

        # 2. 尝试按类型匹配（优先于通配符模式，避免宽泛模式意外覆盖）
        if layer_config is None and module_type in layer_type_config:
            layer_config = layer_type_config[module_type]
            match_type = 'type'
            stats['type_matched_count'] += 1

        # 3. 尝试按模式匹配（支持通配符 *，仅在无类型配置时生效）
        if layer_config is None:
            for pattern, config in layer_name_config.items():
                if '*' in pattern and fnmatch.fnmatch(name, pattern):
                    layer_config = config
                    match_type = 'pattern'
                    matched_pattern = pattern
                    stats['name_matched_count'] += 1
                    break

        # 4. 对于未匹配的模块，检查是否需要禁用量化
        if layer_config is None:
            has_quantizers = (
                (hasattr(module, 'param_quantizers') and module.param_quantizers) or
                (hasattr(module, 'output_quantizers') and len(module.output_quantizers) > 0 and any(q is not None for q in module.output_quantizers)) or
                (hasattr(module, 'input_quantizers') and len(module.input_quantizers) > 0 and any(q is not None for q in module.input_quantizers))
            )
            if default_disable_quantization and has_quantizers:
                _disable_quantization(module, name, verbose=False)
                stats['disabled_count'] += 1
                continue
            elif has_quantizers:
                layer_config = default_bitwidth
                match_type = 'default'
                stats['default_count'] += 1

        # 应用配置
        if layer_config:
            if layer_config.get('disable_quantization', False):
                _disable_quantization(module, name, verbose)
                stats['disabled_count'] += 1
                continue

            # 设置权重量化参数
            if hasattr(module, 'param_quantizers') and 'weight' in module.param_quantizers:
                weight_quantizer = module.param_quantizers['weight']
                if weight_quantizer is not None:
                    changes = []
                    weight_bw = layer_config.get('weight_bitwidth')
                    if weight_bw:
                        weight_quantizer.bitwidth = weight_bw
                        changes.append(f"{weight_bw}-bit")
                    weight_sym = layer_config.get('weight_symmetric')
                    if weight_sym is not None:
                        weight_quantizer.symmetric = weight_sym
                        changes.append(f"{'sym' if weight_sym else 'asym'}")
                    if changes and verbose:
                        label = f"pattern: {matched_pattern}" if match_type == 'pattern' else match_type
                        print(f"  ✅ [{label}] {name}.weight: {', '.join(changes)}")
                    if 'Conv' in module_type:
                        stats['conv_count'] += 1
                    elif 'Linear' in module_type:
                        stats['linear_count'] += 1
                    elif 'Add' in module_type:
                        stats['add_count'] += 1
                    elif 'Multiply' in module_type:
                        stats['multiply_count'] += 1
                    elif 'Subtract' in module_type:
                        stats['subtract_count'] += 1
            
            # 设置 bias 量化参数（新增）
            if hasattr(module, 'param_quantizers') and 'bias' in module.param_quantizers:
                bias_quantizer = module.param_quantizers['bias']
                if bias_quantizer is not None:
                    changes = []
                    
                    # 位宽
                    bias_bw = layer_config.get('bias_bitwidth')
                    if bias_bw:
                        bias_quantizer.bitwidth = bias_bw
                        changes.append(f"{bias_bw}-bit")
                    
                    # bias 通常使用对称量化
                    bias_sym = layer_config.get('bias_symmetric')
                    if bias_sym is not None:
                        bias_quantizer.symmetric = bias_sym
                        changes.append(f"{'sym' if bias_sym else 'asym'}")
                    
                    if changes and verbose:
                        if match_type == 'pattern':
                            print(f"  ✅ [pattern: {matched_pattern}] {name}.bias: {', '.join(changes)}")
                        else:
                            print(f"  ✅ [{match_type}] {name}.bias: {', '.join(changes)}")
            
            # 设置通用参数量化器（param_bitwidth / param_symmetric）
            # 作用于所有非 weight/bias 的 param_quantizers 键，如 Snake2d 的 alpha
            param_bw = layer_config.get('param_bitwidth')
            param_sym = layer_config.get('param_symmetric')
            if (param_bw is not None or param_sym is not None) and hasattr(module, 'param_quantizers'):
                for param_key, param_quantizer in module.param_quantizers.items():
                    if param_key in ('weight', 'bias'):
                        continue  # weight/bias 已由专用字段处理
                    if param_quantizer is not None:
                        changes = []
                        if param_bw is not None:
                            param_quantizer.bitwidth = param_bw
                            if hasattr(param_quantizer, 'symmetric') and param_quantizer.symmetric:
                                param_quantizer.qmin = -(2 ** (param_bw - 1))
                                param_quantizer.qmax = 2 ** (param_bw - 1) - 1
                            else:
                                param_quantizer.qmin = 0
                                param_quantizer.qmax = 2 ** param_bw - 1
                            changes.append(f"{param_bw}-bit")
                        if param_sym is not None:
                            param_quantizer.symmetric = param_sym
                            changes.append(f"{'sym' if param_sym else 'asym'}")
                        if changes and verbose:
                            prefix = f"pattern: {matched_pattern}" if match_type == 'pattern' else match_type
                            print(f"  ✅ [{prefix}] {name}.{param_key}: {', '.join(changes)}")

            # 设置输入量化参数
            if hasattr(module, 'input_quantizers') and len(module.input_quantizers) > 0:
                input_bw = layer_config.get('input_bitwidth')
                input_sym = layer_config.get('input_symmetric')
                for idx, input_quantizer in enumerate(module.input_quantizers):
                    # 若 slot 为 None 但配置了 input_bitwidth，且是第 0 个输入（主输入），则创建新 quantizer
                    # （如 QuantizedVar / QuantizedSnake2d 因无 ONNX 映射，无法由 JSON config 自动创建）
                    if input_quantizer is None and idx == 0 and input_bw is not None:
                        try:
                            from aimet_torch.v2.quantization.affine import QuantizeDequantize
                            import torch.nn as _nn
                            sym = input_sym if input_sym is not None else True
                            new_q = QuantizeDequantize(shape=(), bitwidth=input_bw, symmetric=sym)
                            # nn.ModuleList 不支持直接用索引赋值 None 元素，需重建整个列表
                            slots = list(module.input_quantizers)
                            slots[idx] = new_q
                            module.input_quantizers = _nn.ModuleList(slots)
                            input_quantizer = new_q
                            if verbose:
                                label = f"pattern: {matched_pattern}" if match_type == 'pattern' else match_type
                                print(f"  ➕ [{label}] {name}.input[{idx}]: 创建新 quantizer {input_bw}-bit {'sym' if sym else 'asym'}")
                        except Exception as _e:
                            if verbose:
                                print(f"  ⚠️  {name}.input[{idx}]: 创建 quantizer 失败 - {_e}")

                    if input_quantizer is not None:
                        changes = []

                        if input_bw:
                            input_quantizer.bitwidth = input_bw
                            if hasattr(input_quantizer, 'symmetric') and input_quantizer.symmetric:
                                input_quantizer.qmin = -(2 ** (input_bw - 1))
                                input_quantizer.qmax = 2 ** (input_bw - 1) - 1
                            else:
                                input_quantizer.qmin = 0
                                input_quantizer.qmax = 2 ** input_bw - 1
                            changes.append(f"{input_bw}-bit")

                        if input_sym is not None:
                            input_quantizer.symmetric = input_sym
                            bw = input_bw if input_bw else input_quantizer.bitwidth
                            if input_sym:
                                input_quantizer.qmin = -(2 ** (bw - 1))
                                input_quantizer.qmax = 2 ** (bw - 1) - 1
                            else:
                                input_quantizer.qmin = 0
                                input_quantizer.qmax = 2 ** bw - 1
                            changes.append(f"{'sym' if input_sym else 'asym'}")

                        if verbose and changes:
                            label = f"pattern: {matched_pattern}" if match_type == 'pattern' else match_type
                            if len(module.input_quantizers) > 1:
                                print(f"  ✅ [{label}] {name}.input[{idx}]: {', '.join(changes)}")
                            else:
                                print(f"  ✅ [{label}] {name}.input: {', '.join(changes)}")
            
            # 设置输出量化参数
            if hasattr(module, 'output_quantizers') and len(module.output_quantizers) > 0:
                output_quantizer = module.output_quantizers[0]
                output_bw = layer_config.get('output_bitwidth')
                output_sym = layer_config.get('output_symmetric')
                if output_quantizer is None and output_bw is not None:
                    try:
                        from aimet_torch.v2.quantization.affine import QuantizeDequantize
                        import torch.nn as _nn
                        sym = output_sym if output_sym is not None else True
                        new_q = QuantizeDequantize(shape=(), bitwidth=output_bw, symmetric=sym)
                        slots = list(module.output_quantizers)
                        slots[0] = new_q
                        module.output_quantizers = _nn.ModuleList(slots)
                        output_quantizer = new_q
                        if verbose:
                            label = f"pattern: {matched_pattern}" if match_type == 'pattern' else match_type
                            print(f"  ➕ [{label}] {name}.output[0]: 创建新 quantizer {output_bw}-bit {'sym' if sym else 'asym'}")
                    except Exception as _e:
                        if verbose:
                            print(f"  ⚠️  {name}.output[0]: 创建 quantizer 失败 - {_e}")
                if output_quantizer is not None:
                    changes = []
                    
                    # 位宽
                    if output_bw:
                        output_quantizer.bitwidth = output_bw
                        # 🔧 FIX: 更新 qmin 和 qmax 以匹配新的位宽
                        if hasattr(output_quantizer, 'symmetric') and output_quantizer.symmetric:
                            # 对称量化
                            output_quantizer.qmin = -(2 ** (output_bw - 1))
                            output_quantizer.qmax = 2 ** (output_bw - 1) - 1
                        else:
                            # 非对称量化
                            output_quantizer.qmin = 0
                            output_quantizer.qmax = 2 ** output_bw - 1
                        changes.append(f"{output_bw}-bit")
                    
                    # 对称性
                    if output_sym is not None:
                        output_quantizer.symmetric = output_sym
                        # 🔧 FIX: 如果改变了对称性，需要重新计算 qmin/qmax
                        # 使用配置的位宽或当前的位宽
                        bw = output_bw if output_bw else output_quantizer.bitwidth
                        if output_sym:
                            output_quantizer.qmin = -(2 ** (bw - 1))
                            output_quantizer.qmax = 2 ** (bw - 1) - 1
                        else:
                            output_quantizer.qmin = 0
                            output_quantizer.qmax = 2 ** bw - 1
                        changes.append(f"{'sym' if output_sym else 'asym'}")
                    
                    if verbose and changes:
                        # 只在没有设置权重时打印，避免重复
                        if not (hasattr(module, 'param_quantizers') and 'weight' in module.param_quantizers):
                            print(f"  ✅ [{match_type}] {name}.output: {', '.join(changes)}")
                    
                    # 统计不同类型的层（只统计有输出量化器的层）
                    if 'Tanh' in module_type or 'Sigmoid' in module_type or 'ReLU' in module_type:
                        stats['activation_count'] += 1
                    elif 'Add' in module_type:
                        stats['add_count'] += 1
                    elif 'Multiply' in module_type:
                        stats['multiply_count'] += 1
                    elif 'Subtract' in module_type:
                        stats['subtract_count'] += 1
    
    # 处理 QuantGRU 模块：调用 load_bitwidth_config，并按 use_quantization 区分开/关
    if _QuantGRUType is not None:
        quant_gru_total = 0
        quant_gru_enabled = 0
        quant_gru_disabled = 0
        for name, module in sim_model.named_modules():
            if not isinstance(module, _QuantGRUType):
                continue
            quant_gru_total += 1

            # 加载位宽配置（已校准的 QuantGRU 会被 load_bitwidth_config 内部跳过）
            was_calibrated = module.is_calibrated()
            try:
                module.load_bitwidth_config(config_file, verbose=verbose)
                if verbose and not was_calibrated:
                    print(f"  ✅ [QuantGRU] {name}: 已从配置文件加载位宽设置")
            except Exception as e:
                if verbose:
                    print(f"  ⚠️  [QuantGRU] {name}: 加载配置失败 - {e}")

            # 直接读 QuantGRU 自身的 use_quantization 属性判断量化是否启用
            if bool(getattr(module, 'use_quantization', False)):
                quant_gru_enabled += 1
            else:
                quant_gru_disabled += 1

        if quant_gru_total > 0:
            # 兼容旧字段（保留 quant_gru_count 作为总数），同时新增明确的开/关计数
            stats['quant_gru_count'] = quant_gru_total
            stats['quant_gru_enabled'] = quant_gru_enabled
            stats['quant_gru_disabled'] = quant_gru_disabled
    
    # 打印统计
    if verbose:
        print("="*70)
        print(f"📊 位宽设置统计:")
        print(f"  • Conv 层:             {stats['conv_count']} 个")
        print(f"  • Linear 层:           {stats['linear_count']} 个")
        print(f"  • Add 层:              {stats['add_count']} 个")
        print(f"  • Multiply 层:         {stats['multiply_count']} 个")
        print(f"  • Subtract 层:         {stats['subtract_count']} 个")
        print(f"  • 激活函数层:          {stats['activation_count']} 个")
        if 'quant_gru_count' in stats:
            print(f"  • QuantGRU 层:         {stats['quant_gru_count']} 个 "
                  f"(启用量化: {stats['quant_gru_enabled']}, 未启用: {stats['quant_gru_disabled']})")
        print(f"  • 禁用量化层:          {stats['disabled_count']} 个")
        print(f"  • 按名称匹配:          {stats['name_matched_count']} 个")
        print(f"  • 按类型匹配:          {stats['type_matched_count']} 个")
        print(f"  • 使用默认值:          {stats['default_count']} 个")
        print("="*70)

    return stats

def _disable_quantization(module, name: str, verbose: bool):
    """
    禁用模块的量化（辅助函数）
    
    Args:
        module: 要禁用量化的模块
        name: 模块名称
        verbose: 是否打印详细信息
    """
    # 移除输出量化器
    if hasattr(module, 'output_quantizers') and len(module.output_quantizers) > 0:
        for idx in range(len(module.output_quantizers)):
            if module.output_quantizers[idx] is not None:
                if verbose:
                    print(f"  🚫 {name}: 禁用输出量化器 [{idx}]")
                module.output_quantizers[idx] = None
    
    # 移除输入量化器
    if hasattr(module, 'input_quantizers') and len(module.input_quantizers) > 0:
        for idx in range(len(module.input_quantizers)):
            if module.input_quantizers[idx] is not None:
                if verbose:
                    print(f"  🚫 {name}: 禁用输入量化器 [{idx}]")
                module.input_quantizers[idx] = None
    
    # 移除参数量化器
    if hasattr(module, 'param_quantizers'):
        for param_name in list(module.param_quantizers.keys()):
            if module.param_quantizers[param_name] is not None:
                if verbose:
                    print(f"  🚫 {name}: 禁用参数量化器 [{param_name}]")
                module.param_quantizers[param_name] = None



def check_bn_params_status(model, verbose: bool = True) -> Dict[str, Any]:
    """
    检查 BatchNorm 参数的状态（是否可训练、是否存在）
    
    这个函数用于诊断 BatchNorm affine 参数的状态，帮助排查 QAT 训练问题。
    
    Args:
        model: PyTorch 模型或 AIMET QuantizationSimModel.model
        verbose: 是否打印详细信息
    
    Returns:
        结果字典，包含：
        - bn_modules: BatchNorm 模块数量
        - trainable_affine_params: 可训练的 affine 参数列表
        - frozen_affine_params: 冻结的 affine 参数列表
        - total_affine_params: affine 参数总数
    """
    import torch.nn as nn
    
    # 常见 BN 模块名称模式
    bn_name_patterns = ['_bn', 'bn_', 'pre_bn', 'rnn2d_bn', 'batchnorm', 'batch_norm']
    
    trainable_affine_params = []
    frozen_affine_params = []
    bn_modules = []
    
    # 策略 1: 通过类型识别 BatchNorm 模块
    try:
        from aimet_torch.quantizable_batchnorm import QuantizableBatchNorm1d, QuantizableBatchNorm2d
        custom_bn_types = (QuantizableBatchNorm1d, QuantizableBatchNorm2d)
    except ImportError:
        custom_bn_types = ()
    
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d) + custom_bn_types
    
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            bn_modules.append((name, module))
    
    # 策略 2: 检查参数状态
    for name, param in model.named_parameters():
        # 检查是否是 affine 参数
        is_affine_param = name.endswith('.weight') or name.endswith('.bias')
        if not is_affine_param:
            continue
        
        # 检查是否属于 BatchNorm
        name_lower = name.lower()
        is_bn_param = any(pattern in name_lower for pattern in bn_name_patterns)
        
        # 检查是否在识别的 BN 模块中
        for bn_name, bn_module in bn_modules:
            if name.startswith(bn_name + '.'):
                is_bn_param = True
                break
        
        if is_bn_param:
            if param.requires_grad:
                trainable_affine_params.append(name)
            else:
                frozen_affine_params.append(name)
    
    if verbose:
        print("\n" + "="*70)
        print("BatchNorm 参数状态检查")
        print("="*70)
        print(f"发现 BatchNorm 模块:     {len(bn_modules)} 个")
        
        if bn_modules:
            print("\nBatchNorm 模块列表:")
            for bn_name, bn_module in bn_modules:
                affine_status = "affine=True" if hasattr(bn_module, 'weight') and bn_module.weight is not None else "affine=False"
                training_status = "train" if bn_module.training else "eval"
                print(f"  - {bn_name}: {type(bn_module).__name__} ({affine_status}, {training_status})")
        
        print(f"\n可训练的 affine 参数:   {len(trainable_affine_params)} 个")
        if trainable_affine_params:
            print("  ⚠️  警告：以下 BN affine 参数仍可训练（可能导致 QAT 不稳定）:")
            for param_name in trainable_affine_params:
                print(f"      - {param_name}")
        
        print(f"\n冻结的 affine 参数:     {len(frozen_affine_params)} 个")
        if frozen_affine_params and len(frozen_affine_params) <= 10:
            print("  ✅ 已冻结的参数:")
            for param_name in frozen_affine_params:
                print(f"      - {param_name}")
        
        if trainable_affine_params:
            print("\n💡 建议：在 QAT 训练前调用 freeze_quantizer_parameters(sim.model, freeze_bn_affine=True)")
        else:
            print("\n✅ 所有 BatchNorm affine 参数已正确冻结")
        
        print("="*70 + "\n")
    
    return {
        'bn_modules': bn_modules,
        'bn_module_count': len(bn_modules),
        'trainable_affine_params': trainable_affine_params,
        'frozen_affine_params': frozen_affine_params,
        'total_affine_params': len(trainable_affine_params) + len(frozen_affine_params)
    }


def _align_conv_bias_scale(model, verbose: bool = True) -> Dict[str, Any]:
    """
    将所有卷积层的 bias scale 对齐到 Sx * Sw
    
    目标量化公式：Sy*Qy = Sx*Sw*(Qx*Qw + Qb)
    通过设置 Sb = Sx*Sw，使得 bias 可以在整数域与乘积结果直接相加
    
    适用场景：
    - 权重对称量化（zp_w = 0）
    - 输入可能非对称量化（zp_x 可能非零，但不影响 Sb 的设置）
    
    Args:
        model: 量化后的模型（已执行 compute_encodings 和 Power-of-2）
        verbose: 是否打印详细信息
        
    Returns:
        stats: {
            'total_conv': 检查的卷积层总数,
            'modified_conv': 成功修改的数量,
            'skipped': 跳过的层列表及原因,
            'scale_changes': 每层的 scale 变化详情
        }
    """
    import torch
    import torch.nn as nn
    from aimet_torch.v2.nn import QuantizationMixin
    
    stats = {
        'total_conv': 0,
        'modified_conv': 0,
        'skipped': [],
        'scale_changes': []
    }
    
    # 支持的卷积层类型
    conv_types = (
        nn.Conv1d, nn.Conv2d, nn.Conv3d,
        nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d
    )
    
    for name, module in model.named_modules():
        # 只处理卷积层
        if not isinstance(module, conv_types):
            continue
        
        stats['total_conv'] += 1
        
        # 检查是否有 bias
        if not hasattr(module, 'bias') or module.bias is None:
            stats['skipped'].append((name, 'no bias'))
            if verbose:
                print(f"⏭️  {name}: 跳过（无 bias）")
            continue
        
        # 检查是否是量化模块
        if not isinstance(module, QuantizationMixin):
            stats['skipped'].append((name, 'not quantized'))
            if verbose:
                print(f"⏭️  {name}: 跳过（未量化）")
            continue
        
        # 获取参数量化器
        if not hasattr(module, 'param_quantizers') or not module.param_quantizers:
            stats['skipped'].append((name, 'no param_quantizers'))
            if verbose:
                print(f"⏭️  {name}: 跳过（无参数量化器）")
            continue
        
        # 获取权重量化器（ModuleDict 访问方式）
        weight_quantizer = None
        if 'weight' in module.param_quantizers:
            weight_quantizer = module.param_quantizers['weight']
        
        if weight_quantizer is None or not weight_quantizer.is_initialized():
            stats['skipped'].append((name, 'weight quantizer not initialized'))
            if verbose:
                print(f"⏭️  {name}: 跳过（权重量化器未初始化）")
            continue
        
        # 获取 bias 量化器（ModuleDict 访问方式）
        bias_quantizer = None
        if 'bias' in module.param_quantizers:
            bias_quantizer = module.param_quantizers['bias']
        
        if bias_quantizer is None or not bias_quantizer.is_initialized():
            stats['skipped'].append((name, 'bias quantizer not initialized'))
            if verbose:
                print(f"⏭️  {name}: 跳过（bias 量化器未初始化）")
            continue
        
        # 获取输入量化器（从输入量化器列表）
        input_quantizer = None
        if hasattr(module, 'input_quantizers') and module.input_quantizers:
            input_quantizer = module.input_quantizers[0]  # 通常第一个是数据输入
        
        if input_quantizer is None or not input_quantizer.is_initialized():
            stats['skipped'].append((name, 'input quantizer not found or not initialized'))
            if verbose:
                print(f"⏭️  {name}: 跳过（输入量化器未找到或未初始化）")
            continue
        
        # ===== 步骤 1: 提取量化参数 =====
        try:
            # 确保所有量化器使用 scale-offset 参数化
            input_quantizer._reparametrize_to_scale_offset()
            weight_quantizer._reparametrize_to_scale_offset()
            bias_quantizer._reparametrize_to_scale_offset()
            
            # 获取 scale（直接访问属性）
            Sx = input_quantizer.scale
            Sw = weight_quantizer.scale
            Sb_old = bias_quantizer.scale
            
            # 获取 zero-point（用于记录）
            zp_x = input_quantizer.offset if hasattr(input_quantizer, 'offset') and input_quantizer.offset is not None else 0
            if isinstance(zp_x, torch.Tensor):
                zp_x = float(zp_x.item()) if zp_x.numel() == 1 else zp_x
            
            # 转换为标量（如果是 tensor）
            if isinstance(Sx, torch.Tensor):
                Sx = float(Sx.item()) if Sx.numel() == 1 else Sx
            if isinstance(Sb_old, torch.Tensor):
                Sb_old = float(Sb_old.item()) if Sb_old.numel() == 1 else Sb_old
            
            # 检查是否是 per-channel 权重量化
            is_per_channel = isinstance(Sw, torch.Tensor) and Sw.numel() > 1
            
        except Exception as e:
            stats['skipped'].append((name, f'error getting scales: {e}'))
            if verbose:
                print(f"⏭️  {name}: 跳过（获取 scale 失败: {e}）")
            continue
        
        # ===== 步骤 2: 计算新的 bias scale =====
        if is_per_channel:
            # Per-channel: Sb_new 是向量
            # Sx 是标量，Sw 是向量
            if isinstance(Sx, torch.Tensor):
                Sx_val = float(Sx.item())
            else:
                Sx_val = float(Sx)
            
            Sb_new_raw = Sx_val * Sw  # 可能是 [out_channels] 或 [out_channels, 1, 1, 1]
            
            # 🔑 关键：如果 Sw 是多维的（例如 [16, 1, 1, 1]），需要展平为 1D
            if isinstance(Sb_new_raw, torch.Tensor) and Sb_new_raw.dim() > 1:
                # 找到非1的维度
                non_one_dims = [i for i, s in enumerate(Sb_new_raw.shape) if s > 1]
                if len(non_one_dims) == 1:
                    # 只有一个非1维度，squeeze 掉其他维度
                    Sb_new = Sb_new_raw.squeeze()
                else:
                    # 多个非1维度，保持原样
                    Sb_new = Sb_new_raw
            else:
                Sb_new = Sb_new_raw
        else:
            # Per-tensor: Sb_new 是标量
            Sw_scalar = float(Sw.item()) if isinstance(Sw, torch.Tensor) else float(Sw)
            if isinstance(Sx, torch.Tensor):
                Sx_val = float(Sx.item())
            else:
                Sx_val = float(Sx)
            Sb_new = Sx_val * Sw_scalar
        
        # ===== 步骤 3: 计算新的浮点表示范围 =====
        try:
            # 获取 bias 位宽
            bias_bitwidth = bias_quantizer.bitwidth
            
            # 计算量化整数范围（对称量化）
            Qmax = 2 ** (bias_bitwidth - 1) - 1
            Qmin = -2 ** (bias_bitwidth - 1)
            
            # 🔑 使用展平后的 Sb_new 计算范围
            if is_per_channel:
                fmin_new = Qmin * Sb_new  # 向量，现在应该是 1D
                fmax_new = Qmax * Sb_new  # 向量，现在应该是 1D
            else:
                fmin_new = Qmin * Sb_new
                fmax_new = Qmax * Sb_new
            
        except Exception as e:
            stats['skipped'].append((name, f'error computing new range: {e}'))
            if verbose:
                print(f"⏭️  {name}: 跳过（计算新范围失败: {e}）")
            continue
        
        # ===== 步骤 4: 更新 bias 量化器 =====
        try:
            # 确保量化器使用 scale-offset 参数化
            bias_quantizer._reparametrize_to_scale_offset()
            
            # 🔑 关键：保持与原始 scale 相同的形状
            original_scale_shape = bias_quantizer.scale.shape
            
            if verbose:
                print(f"\n🔧 准备更新 bias 量化器:")
                print(f"  原始 scale 形状: {original_scale_shape}")
            
            # 转换 Sb_new 为 tensor（如果是标量）
            if not isinstance(Sb_new, torch.Tensor):
                Sb_new_tensor = torch.tensor(Sb_new, dtype=bias_quantizer.scale.dtype, device=bias_quantizer.scale.device)
            else:
                Sb_new_tensor = Sb_new.clone().to(dtype=bias_quantizer.scale.dtype, device=bias_quantizer.scale.device)
            
            if verbose:
                print(f"  Sb_new 原始形状: {Sb_new_tensor.shape}")
            
            # 确保形状匹配：如果原始 scale 有特殊形状，需要调整
            if Sb_new_tensor.shape != original_scale_shape:
                if Sb_new_tensor.numel() == 1:
                    # 标量情况：扩展到目标形状
                    Sb_new_tensor = Sb_new_tensor.view(1).expand(original_scale_shape).contiguous()
                elif original_scale_shape == torch.Size([]):
                    # 目标是标量
                    Sb_new_tensor = Sb_new_tensor.squeeze()
                else:
                    # Per-channel 情况：尝试 reshape
                    # 首先确保元素数量匹配
                    if Sb_new_tensor.numel() == original_scale_shape.numel():
                        Sb_new_tensor = Sb_new_tensor.reshape(original_scale_shape)
                    else:
                        # 如果元素数量不匹配，可能需要广播
                        # 例如 [16] -> [16, 1, 1, 16] 是不合理的，应该只是 [16]
                        # 尝试保持 1D 形状
                        if len(original_scale_shape) > 1 and original_scale_shape[0] == Sb_new_tensor.numel():
                            # 假设第一维是通道维度
                            target_shape = [original_scale_shape[0]] + [1] * (len(original_scale_shape) - 1)
                            Sb_new_tensor = Sb_new_tensor.view(target_shape).expand(original_scale_shape).contiguous()
                        else:
                            if verbose:
                                print(f"  ⚠️  形状不兼容: {Sb_new_tensor.shape} -> {original_scale_shape}")
            
            if verbose:
                print(f"  Sb_new 调整后形状: {Sb_new_tensor.shape}")
            
            # 修改 scale（使用 copy_ 避免破坏梯度图）
            with torch.no_grad():
                bias_quantizer.scale.copy_(Sb_new_tensor)
            
            # offset 保持 0（对称量化，无需修改）
            # bias_quantizer.offset 已经是 0，不需要改
            
            # 更新 min/max 范围（也需要匹配形状）
            if is_per_channel:
                fmin_new_tensor = fmin_new.clone().to(dtype=bias_quantizer.scale.dtype, device=bias_quantizer.scale.device)
                fmax_new_tensor = fmax_new.clone().to(dtype=bias_quantizer.scale.dtype, device=bias_quantizer.scale.device)
            else:
                fmin_new_tensor = torch.tensor(fmin_new, dtype=bias_quantizer.scale.dtype, device=bias_quantizer.scale.device)
                fmax_new_tensor = torch.tensor(fmax_new, dtype=bias_quantizer.scale.dtype, device=bias_quantizer.scale.device)
            
            # 确保 min/max 也匹配原始形状
            if hasattr(bias_quantizer, 'min') and bias_quantizer.min is not None:
                original_min_shape = bias_quantizer.min.shape
                if fmin_new_tensor.shape != original_min_shape:
                    if fmin_new_tensor.numel() == 1:
                        fmin_new_tensor = fmin_new_tensor.view(1).expand(original_min_shape).contiguous()
                    elif original_min_shape == torch.Size([]):
                        fmin_new_tensor = fmin_new_tensor.squeeze()
                    else:
                        if fmin_new_tensor.numel() == original_min_shape.numel():
                            fmin_new_tensor = fmin_new_tensor.reshape(original_min_shape)
                        elif len(original_min_shape) > 1 and original_min_shape[0] == fmin_new_tensor.numel():
                            target_shape = [original_min_shape[0]] + [1] * (len(original_min_shape) - 1)
                            fmin_new_tensor = fmin_new_tensor.view(target_shape).expand(original_min_shape).contiguous()
            
            if hasattr(bias_quantizer, 'max') and bias_quantizer.max is not None:
                original_max_shape = bias_quantizer.max.shape
                if fmax_new_tensor.shape != original_max_shape:
                    if fmax_new_tensor.numel() == 1:
                        fmax_new_tensor = fmax_new_tensor.view(1).expand(original_max_shape).contiguous()
                    elif original_max_shape == torch.Size([]):
                        fmax_new_tensor = fmax_new_tensor.squeeze()
                    else:
                        if fmax_new_tensor.numel() == original_max_shape.numel():
                            fmax_new_tensor = fmax_new_tensor.reshape(original_max_shape)
                        elif len(original_max_shape) > 1 and original_max_shape[0] == fmax_new_tensor.numel():
                            target_shape = [original_max_shape[0]] + [1] * (len(original_max_shape) - 1)
                            fmax_new_tensor = fmax_new_tensor.view(target_shape).expand(original_max_shape).contiguous()
            
            if verbose:
                print(f"  fmin 形状: {fmin_new_tensor.shape}")
                print(f"  fmax 形状: {fmax_new_tensor.shape}")
            
            # 使用 set_range() 方法更新 min/max
            bias_quantizer.set_range(fmin_new_tensor, fmax_new_tensor)
            
            if verbose:
                print(f"  ✅ 更新成功")
            
            stats['modified_conv'] += 1
            
            # 记录修改信息
            # 转换为可序列化的格式
            Sx_display = float(Sx) if not isinstance(Sx, torch.Tensor) else float(Sx.item()) if Sx.numel() == 1 else f'tensor shape={Sx.shape}'
            Sb_old_display = float(Sb_old) if not isinstance(Sb_old, torch.Tensor) else float(Sb_old.item()) if Sb_old.numel() == 1 else 'tensor'
            
            scale_change_info = {
                'layer': name,
                'Sx': Sx_display,
                'Sw': float(Sw_scalar) if not is_per_channel else f'per-channel (shape={Sw.shape})',
                'Sb_old': Sb_old_display,
                'Sb_new': float(Sb_new) if not is_per_channel else f'per-channel (shape={Sb_new.shape})',
                'scale_ratio': float(Sb_new / Sb_old_display) if not is_per_channel and isinstance(Sb_old_display, (int, float)) else 'varies',
                'bias_bitwidth': bias_bitwidth,
                'new_float_range': [float(fmin_new), float(fmax_new)] if not is_per_channel else 'per-channel',
                'zp_x': int(zp_x) if isinstance(zp_x, (int, float, torch.Tensor)) and zp_x is not None else 0,
                'is_per_channel': is_per_channel
            }
            stats['scale_changes'].append(scale_change_info)
            
            # 打印详细信息
            if verbose:
                print(f"\n📊 {name}:")
                print(f"  类型: {type(module).__name__}")
                
                # 打印 Sx
                Sx_display = float(Sx) if not isinstance(Sx, torch.Tensor) else float(Sx.item()) if Sx.numel() == 1 else Sx
                if isinstance(Sx_display, torch.Tensor):
                    print(f"  Sx (输入 scale):   per-channel, shape={Sx_display.shape}")
                else:
                    print(f"  Sx (输入 scale):   {Sx_display:.8f}")
                
                # 打印 Sw
                if is_per_channel:
                    # 显示原始形状和 Sb_new 的形状
                    Sb_new_shape = Sb_new.shape if isinstance(Sb_new, torch.Tensor) else "scalar"
                    if isinstance(Sb_new, torch.Tensor) and Sw.shape != Sb_new.shape:
                        print(f"  Sw (权重 scale):   per-channel, 原始形状={Sw.shape} → Sb_new形状={Sb_new.shape}")
                    else:
                        print(f"  Sw (权重 scale):   per-channel, shape={Sw.shape}")
                    print(f"                     min={Sw.min().item():.8f}, max={Sw.max().item():.8f}")
                else:
                    print(f"  Sw (权重 scale):   {Sw_scalar:.8f}")
                
                # 打印 Sb_old
                Sb_old_display = float(Sb_old) if not isinstance(Sb_old, torch.Tensor) else float(Sb_old.item()) if Sb_old.numel() == 1 else Sb_old
                if isinstance(Sb_old_display, torch.Tensor):
                    print(f"  Sb_old (旧 bias scale): per-channel")
                else:
                    print(f"  Sb_old (旧 bias scale): {Sb_old_display:.8f}")
                
                # 打印 Sb_new
                if is_per_channel:
                    print(f"  Sb_new (新 bias scale): per-channel, shape={Sb_new.shape}")
                    print(f"                          min={Sb_new.min().item():.8f}, max={Sb_new.max().item():.8f}")
                    # 计算平均变化倍数
                    ratio_tensor = Sb_new / Sb_old_display if isinstance(Sb_old_display, torch.Tensor) else Sb_new / Sb_old_display
                    print(f"  变化倍数:          平均 {ratio_tensor.mean().item():.4f}x (范围: {ratio_tensor.min().item():.4f}x - {ratio_tensor.max().item():.4f}x)")
                else:
                    print(f"  Sb_new (新 bias scale): {Sb_new:.8f}")
                    print(f"  变化倍数:          {Sb_new / Sb_old_display:.4f}x")
                
                print(f"  Bias 位宽:         {bias_bitwidth}-bit")
                print(f"  量化范围:          [{Qmin}, {Qmax}]")
                
                if is_per_channel:
                    print(f"  新浮点范围:        per-channel")
                    print(f"                     min范围: [{fmin_new.min().item():.6f}, {fmin_new.max().item():.6f}]")
                    print(f"                     max范围: [{fmax_new.min().item():.6f}, {fmax_new.max().item():.6f}]")
                else:
                    print(f"  新浮点范围:        [{fmin_new:.6f}, {fmax_new:.6f}]")
                
                print(f"  offset:            0 (对称量化)")
                
                if isinstance(zp_x, (int, float)) and zp_x != 0:
                    print(f"  输入 zero-point:   {zp_x} (非对称量化)")            
        except Exception as e:
            stats['skipped'].append((name, f'error updating quantizer: {e}'))
            if verbose:
                print(f"❌ {name}: 更新失败: {e}")
            continue
    
    # 打印汇总
    if verbose:
        print("\n" + "="*80)
        print("Bias Scale 对齐汇总")
        print("="*80)
        print(f"检查的卷积层总数:   {stats['total_conv']}")
        print(f"成功修改的层数:     {stats['modified_conv']}")
        print(f"跳过的层数:         {len(stats['skipped'])}")
        
        if stats['skipped'] and len(stats['skipped']) <= 10:
            print("\n跳过的层:")
            for layer_name, reason in stats['skipped']:
                print(f"  - {layer_name}: {reason}")
        elif len(stats['skipped']) > 10:
            print(f"\n跳过的层（仅显示前 10 个）:")
            for layer_name, reason in stats['skipped'][:10]:
                print(f"  - {layer_name}: {reason}")
            print(f"  ... 还有 {len(stats['skipped']) - 10} 个")
        
        if stats['modified_conv'] > 0:
            print(f"\n✅ 成功对齐 {stats['modified_conv']} 个卷积层的 bias scale")
        else:
            print("\n⚠️  没有成功修改任何层")
        
        print("="*80)
    
    return stats


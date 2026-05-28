"""
多阶段量化管理器
支持任意多个阶段的量化流程
"""
import json
import os
from typing import Dict, List, Any, Optional
from pathlib import Path


class StagedQuantizationManager:
    """
    多阶段量化管理器
    
    支持任意多个阶段的量化流程，每个阶段可以：
    - 指定要量化的层类型
    - 自动加载之前阶段的参数
    - 保存当前阶段的参数
    
    Examples:
        # 创建管理器
        manager = StagedQuantizationManager(
            config_file="multi_stage_config.json",
            output_dir="./staged_quantization_outputs"
        )
        
        # 执行阶段 1
        manager.prepare_stage(sim.model, stage_id=1)
        # ... 校准和QAT ...
        manager.save_stage(sim.model, stage_id=1)
        
        # 执行阶段 2
        manager.prepare_stage(sim.model, stage_id=2)
        # ... 校准和QAT ...
        manager.save_stage(sim.model, stage_id=2)
    """
    
    def __init__(
        self,
        config_file: str,
        output_dir: str = "./staged_quantization_outputs",
        verbose: bool = True
    ):
        """
        初始化多阶段量化管理器
        
        Args:
            config_file: 多阶段配置文件路径
            output_dir: 输出目录，用于保存每个阶段的参数
            verbose: 是否打印详细信息
        """
        self.config_file = config_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        
        # 加载配置
        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.stages = self.config.get('stages', [])
        self.global_config = self.config.get('global', {})
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"📋 多阶段量化管理器")
            print(f"{'='*80}")
            print(f"配置文件: {config_file}")
            print(f"输出目录: {output_dir}")
            print(f"总阶段数: {len(self.stages)}")
            print(f"{'='*80}\n")
    
    def get_stage_config(self, stage_id: int) -> Dict[str, Any]:
        """获取指定阶段的配置"""
        if stage_id < 1 or stage_id > len(self.stages):
            raise ValueError(f"无效的阶段 ID: {stage_id}，有效范围: 1-{len(self.stages)}")
        return self.stages[stage_id - 1]
    
    def get_stage_bitwidth_config_file(self, stage_id: int) -> Optional[str]:
        """获取指定阶段的位宽配置文件路径"""
        stage_config = self.get_stage_config(stage_id)
        bitwidth_config = stage_config.get('bitwidth_config_file')
        if bitwidth_config:
            # 如果是相对路径，基于配置文件目录解析
            if not os.path.isabs(bitwidth_config):
                config_dir = os.path.dirname(self.config_file)
                bitwidth_config = os.path.join(config_dir, bitwidth_config)
        return bitwidth_config
    
    def get_stage_encodings_file(self, stage_id: int) -> str:
        """获取指定阶段的编码文件路径"""
        return str(self.output_dir / f"stage{stage_id}_encodings.json")
    
    def get_stages_to_load(self, stage_id: int) -> List[int]:
        """
        获取当前阶段需要加载的所有之前阶段
        
        默认只加载上一个阶段（累积式保存），除非配置中指定了 load_from_stages
        
        累积式保存策略：
        - 每个阶段保存所有已量化的层（包括之前阶段的）
        - 下一个阶段只需加载上一个阶段的文件
        """
        stage_config = self.get_stage_config(stage_id)
        load_from = stage_config.get('load_from_stages')
        
        if load_from is None:
            # 默认只加载上一个阶段（累积式保存）
            if stage_id > 1:
                return [stage_id - 1]
            else:
                return []
        elif isinstance(load_from, list):
            # 指定加载哪些阶段
            return load_from
        elif load_from == "previous":
            # 显式指定只加载上一个阶段
            return [stage_id - 1] if stage_id > 1 else []
        elif load_from == "all":
            # 加载所有之前的阶段（非累积式保存时使用）
            return list(range(1, stage_id))
        elif load_from == "none":
            # 不加载任何阶段
            return []
        else:
            raise ValueError(f"无效的 load_from_stages 配置: {load_from}")
    
    def prepare_stage(
        self,
        sim_model,
        stage_id: int,
        apply_bitwidth: bool = True
    ) -> Dict[str, Any]:
        """
        准备指定阶段的量化
        
        1. 应用位宽配置（如果有）
        2. 加载之前阶段的量化参数
        
        Args:
            sim_model: AIMET QuantizationSimModel.model
            stage_id: 阶段 ID（从 1 开始）
            apply_bitwidth: 是否应用位宽配置
        
        Returns:
            准备结果统计
        """
        from aimet_torch.utils_rx import apply_mixed_precision_bitwidth
        from aimet_torch.staged_quantization_utils import load_quantizer_encodings
        
        stage_config = self.get_stage_config(stage_id)
        stage_name = stage_config.get('name', f'Stage {stage_id}')
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"🚀 准备阶段 {stage_id}: {stage_name}")
            print(f"{'='*80}")
        
        results = {
            'stage_id': stage_id,
            'stage_name': stage_name,
            'bitwidth_applied': False,
            'loaded_stages': [],
            'total_loaded': 0
        }
        
        # 1. 应用位宽配置
        if apply_bitwidth:
            bitwidth_config_file = self.get_stage_bitwidth_config_file(stage_id)
            if bitwidth_config_file:
                if self.verbose:
                    print(f"\n📝 应用位宽配置: {bitwidth_config_file}")
                stats = apply_mixed_precision_bitwidth(
                    sim_model,
                    config_file=bitwidth_config_file,
                    verbose=self.verbose
                )
                results['bitwidth_applied'] = True
                results['bitwidth_stats'] = stats
        
        # 2. 加载之前阶段的参数
        stages_to_load = self.get_stages_to_load(stage_id)
        if stages_to_load:
            if self.verbose:
                print(f"\n📥 加载之前阶段的参数: {stages_to_load}")
            
            # 获取当前阶段要排除的层类型
            exclude_types = stage_config.get('quantize_layer_types', [])
            if exclude_types and self.verbose:
                print(f"   排除层类型（本阶段要重新量化）: {', '.join(exclude_types)}")
            
            total_loaded = 0
            for prev_stage_id in stages_to_load:
                prev_encodings_file = self.get_stage_encodings_file(prev_stage_id)
                if os.path.exists(prev_encodings_file):
                    if self.verbose:
                        print(f"\n   加载阶段 {prev_stage_id}: {prev_encodings_file}")
                    
                    load_stats = load_quantizer_encodings(
                        sim_model,
                        load_path=prev_encodings_file,
                        exclude_layer_types=exclude_types if exclude_types else None,
                        skip_if_not_found=True,
                        verbose=self.verbose
                    )
                    total_loaded += load_stats['loaded']
                    results['loaded_stages'].append(prev_stage_id)
                else:
                    if self.verbose:
                        print(f"   ⚠️  阶段 {prev_stage_id} 的编码文件不存在，跳过")
            
            results['total_loaded'] = total_loaded
            if self.verbose:
                print(f"\n✅ 总共加载了 {total_loaded} 个量化器参数")
        
        if self.verbose:
            print(f"{'='*80}\n")
        
        return results
    
    def save_stage(
        self,
        sim_model,
        stage_id: int,
        save_all: bool = True
    ) -> Dict[str, Any]:
        """
        保存当前阶段的量化参数
        
        Args:
            sim_model: AIMET QuantizationSimModel.model
            stage_id: 阶段 ID（从 1 开始）
            save_all: 是否保存所有已量化的层（推荐）
        
        Returns:
            保存结果统计
        """
        from aimet_torch.staged_quantization_utils import save_quantizer_encodings
        
        stage_config = self.get_stage_config(stage_id)
        stage_name = stage_config.get('name', f'Stage {stage_id}')
        encodings_file = self.get_stage_encodings_file(stage_id)
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"💾 保存阶段 {stage_id}: {stage_name}")
            print(f"{'='*80}")
        
        # 确定要保存的层类型
        if save_all:
            layer_types = None  # 保存所有
        else:
            layer_types = stage_config.get('quantize_layer_types', [])
        
        encodings = save_quantizer_encodings(
            sim_model,
            save_path=encodings_file,
            layer_types=layer_types,
            verbose=self.verbose
        )
        
        if self.verbose:
            print(f"{'='*80}\n")
        
        return {
            'stage_id': stage_id,
            'stage_name': stage_name,
            'encodings_file': encodings_file,
            'saved_count': len(encodings)
        }
    
    def print_stage_summary(self, stage_id: int):
        """打印阶段摘要"""
        stage_config = self.get_stage_config(stage_id)
        stage_name = stage_config.get('name', f'Stage {stage_id}')
        quantize_types = stage_config.get('quantize_layer_types', [])
        
        print(f"\n{'='*80}")
        print(f"📊 阶段 {stage_id} 摘要: {stage_name}")
        print(f"{'='*80}")
        print(f"本阶段量化的层类型: {', '.join(quantize_types) if quantize_types else '所有'}")
        print(f"位宽配置文件: {self.get_stage_bitwidth_config_file(stage_id) or '无'}")
        print(f"加载之前阶段: {self.get_stages_to_load(stage_id)}")
        print(f"编码保存路径: {self.get_stage_encodings_file(stage_id)}")
        print(f"{'='*80}\n")
    
    def print_all_stages_summary(self):
        """打印所有阶段的摘要"""
        print(f"\n{'='*80}")
        print(f"📋 所有阶段摘要")
        print(f"{'='*80}")
        
        for i, stage in enumerate(self.stages, 1):
            stage_name = stage.get('name', f'Stage {i}')
            quantize_types = stage.get('quantize_layer_types', [])
            print(f"\n阶段 {i}: {stage_name}")
            print(f"  量化层类型: {', '.join(quantize_types) if quantize_types else '所有'}")
        
        print(f"\n{'='*80}\n")


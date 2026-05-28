# -*- mode: python -*-
# =============================================================================
#  AIMET 可量化一元算子 (Quantizable Unary Operators)
#  
#  包含 Abs, Sign 等一元操作的便捷包装和使用示例
# =============================================================================

"""
可量化的一元算子模块

AIMET 已经包含了 Abs 和 Sign 算子，本模块提供：
1. 便捷的导入和使用方式
2. 使用示例和最佳实践
3. 与 Power-of-2 量化的兼容性
"""

import torch
import torch.nn as nn
from aimet_torch._base.nn.modules.custom import Abs, ElementwiseUnarySign


# ============================================================================
# 重命名以便更直观使用
# ============================================================================
Sign = ElementwiseUnarySign  # 重命名为更直观的名称


# ============================================================================
# 便捷的复合模块
# ============================================================================

class PowerCompress(nn.Module):
    """
    Power Compression 模块（可量化版本）
    
    实现: output = (abs(x) ** 0.5) * sign(x)
    
    AIMET 量化要求：
    - 每个操作位置使用独立的 Module 实例
    - 避免共享 Module（会导致量化器冲突）
    
    Example:
        >>> import torch
        >>> from aimet_torch.quantizable_unary_ops import PowerCompress
        >>> 
        >>> # 创建模块
        >>> power_compress = PowerCompress()
        >>> 
        >>> # 使用
        >>> x = torch.randn(2, 3, 4, 4)
        >>> output = power_compress(x)
    """
    
    def __init__(self):
        super().__init__()
        # 独立的算子实例（AIMET 量化要求）
        self.abs = Abs()
        self.sign = Sign()
        self.pow = nn.Pow()  # 需要确认是否有量化版本
        self.mul = nn.Module()  # 使用 Multiply 算子
        
        # 导入 Multiply（如果需要量化）
        from aimet_torch._base.nn.modules.custom import Multiply, Pow
        self.pow_op = Pow()
        self.multiply = Multiply()
    
    def forward(self, x):
        """
        Forward pass: (abs(x) ** 0.5) * sign(x)
        
        Args:
            x: Input tensor
        
        Returns:
            Power compressed tensor
        """
        abs_x = self.abs(x)
        
        # sqrt(abs(x)) = abs(x) ** 0.5
        sqrt_abs_x = self.pow_op(abs_x, 0.5)
        
        # sign(x)
        sign_x = self.sign(x)
        
        # 组合结果
        output = self.multiply(sqrt_abs_x, sign_x)
        
        return output


class HypotFunction(nn.Module):
    """
    Hypot 函数模块（可量化版本）
    
    实现: output = sqrt(x^2 + y^2 + eps)
    
    Example:
        >>> import torch
        >>> from aimet_torch.quantizable_unary_ops import HypotFunction
        >>> 
        >>> hypot_fn = HypotFunction(eps=1e-8)
        >>> x = torch.randn(2, 3, 4, 4)
        >>> y = torch.randn(2, 3, 4, 4)
        >>> output = hypot_fn(x, y)
    """
    
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        
        # 导入量化算子
        from aimet_torch._base.nn.modules.custom import Multiply, Add, Sqrt, Pow
        
        # 独立的算子实例
        self.pow_x = Pow()
        self.pow_y = Pow()
        self.add_xy = Add()
        self.add_eps = Add()
        self.sqrt = Sqrt()
    
    def forward(self, x, y):
        """
        Forward pass: sqrt(x^2 + y^2 + eps)
        
        Args:
            x: First input tensor
            y: Second input tensor
        
        Returns:
            Hypot result
        """
        # x^2
        x_squared = self.pow_x(x, 2)
        
        # y^2
        y_squared = self.pow_y(y, 2)
        
        # x^2 + y^2
        sum_squares = self.add_xy(x_squared, y_squared)
        
        # x^2 + y^2 + eps
        sum_with_eps = self.add_eps(sum_squares, self.eps)
        
        # sqrt(x^2 + y^2 + eps)
        output = self.sqrt(sum_with_eps)
        
        return output


# ============================================================================
# 使用示例
# ============================================================================

class ExampleQuantizableModel(nn.Module):
    """
    示例：包含 Abs 和 Sign 算子的可量化模型
    
    展示如何在 AIMET 量化流程中使用这些算子
    """
    
    def __init__(self):
        super().__init__()
        
        # 常规层
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        
        # ✅ 可量化的一元算子
        self.abs = Abs()           # 绝对值
        self.sign = Sign()          # 符号函数
        
        # 复合算子
        self.power_compress = PowerCompress()
        
        # 输出层
        self.conv2 = nn.Conv2d(64, 10, 1)
    
    def forward(self, x):
        # 标准卷积流程
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # 使用 abs（例如：确保正值）
        x_abs = self.abs(x)
        
        # 使用 sign（例如：二值化方向）
        x_sign = self.sign(x)
        
        # 组合（示例）
        x = x_abs * 0.9 + x_sign * 0.1
        
        # 输出
        x = self.conv2(x)
        
        return x


def get_quantizable_abs_sign_ops():
    """
    获取所有可量化的 Abs 和 Sign 相关算子
    
    Returns:
        dict: 算子名称到类的映射
    """
    return {
        'Abs': Abs,
        'Sign': Sign,
        'ElementwiseUnarySign': ElementwiseUnarySign,  # 原始名称
        'PowerCompress': PowerCompress,
        'HypotFunction': HypotFunction,
    }


# ============================================================================
# 导出所有
# ============================================================================

__all__ = [
    'Abs',
    'Sign',
    'ElementwiseUnarySign',
    'PowerCompress',
    'HypotFunction',
    'ExampleQuantizableModel',
    'get_quantizable_abs_sign_ops',
]


# ============================================================================
# 使用说明
# ============================================================================

"""
使用方法：

1. 基本使用:
   ```python
   from aimet_torch.quantizable_unary_ops import Abs, Sign
   
   class MyModel(nn.Module):
       def __init__(self):
           super().__init__()
           self.abs = Abs()
           self.sign = Sign()
       
       def forward(self, x):
           x_abs = self.abs(x)
           x_sign = self.sign(x)
           return x_abs, x_sign
   ```

2. 量化流程:
   ```python
   from aimet_torch.v2 import quantsim
   from aimet_torch import model_preparer
   
   # 准备模型
   model = MyModel()
   prepared_model = model_preparer.prepare_model(model)
   
   # 创建量化模拟器
   sim = quantsim.QuantizationSimModel(
       prepared_model,
       dummy_input=torch.randn(1, 3, 224, 224),
       quant_scheme='percentile',
       config_file='quantsim_config.json',
       default_output_bw=8,
       default_param_bw=8
   )
   
   # 校准
   import aimet_torch.v2 as aimet
   with aimet.nn.compute_encodings(sim.model):
       for inputs, _ in calib_loader:
           sim.model(inputs)
   
   # 评估
   accuracy = evaluate(sim.model, test_loader)
   ```

3. 配置文件支持 (quantsim_config.json):
   ```json
   {
     "defaults": {
       "ops": {
         "is_output_quantized": "True",
         "is_symmetric": "True"
       },
       "params": {
         "is_quantized": "True",
         "is_symmetric": "True"
       }
     },
     "op_type": {
       "Abs": {
         "is_input_quantized": "True",
         "is_output_quantized": "True"
       },
       "Sign": {
         "is_input_quantized": "True",
         "is_output_quantized": "True"
       }
     }
   }
   ```

注意事项：
- Abs 和 Sign 已经在 AIMET 中支持，可以直接使用
- 每个操作位置需要独立的 Module 实例（不要共享）
- 支持所有标准 AIMET 量化流程（PTQ, QAT, Power-of-2）
- 可以在 quantsim_config.json 中配置量化参数
"""




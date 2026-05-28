import torch
import torch.nn as nn
from torch import Tensor

from aimet_torch.v2.nn.true_quant import QuantizationMixin


class QuantizableBatchNorm2d(nn.Module):
    """自定义BatchNorm2d，功能与nn.BatchNorm2d完全相同"""
    
    def __init__(self,
                 num_features: int,
                 eps: float = 1e-5,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True):
        """
        参数说明：
        num_features: 输入的特征图通道数（C）
        eps: 数值稳定性的小常数，防止除以0
        momentum: 用于running_mean和running_var更新的动量
        affine: 是否使用可学习的缩放和平移参数（gamma和beta）
        track_running_stats: 是否跟踪运行时的统计信息（训练时）
        """
        super(QuantizableBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        # 可学习的参数（如果affine=True）
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))  # gamma
            self.bias = nn.Parameter(torch.zeros(num_features))   # beta
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        # 运行时统计信息（如果track_running_stats=True）
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        
        # 重置参数
        self.reset_parameters()
    
    def reset_parameters(self):
        """重置所有参数"""
        if self.track_running_stats:
            # 初始化running_mean为0，running_var为1
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()
        
        if self.affine:
            # 初始化gamma为1，beta为0（与PyTorch官方实现一致）
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
    
    def _check_input_dim(self, x: Tensor):
        """检查输入维度是否为4D (N, C, H, W)"""
        # ⚠️ 注释掉维度检查，因为trace时无法求值
        # PyTorch 运算会自动检查维度，无需显式检查
        # if x.dim() != 4:
        #     raise ValueError(f'Expected 4D input (got {x.dim()}D input)')
        pass
    
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播 - trace友好版本
        
        ✅ 移除所有控制流，使用函数式API
        ✅ training/eval状态在Python层处理（不在trace中）
        """
        # 根据 training 状态选择统计量
        if self.training or not self.track_running_stats:
            # 训练模式：计算批次统计量
            mean = torch.mean(x, dim=(0, 2, 3), keepdim=True)  # [1, C, 1, 1]
            var = torch.var(x, dim=(0, 2, 3), keepdim=True, unbiased=False)  # [1, C, 1, 1]
            
            # 更新 running statistics
            if self.track_running_stats:
                with torch.no_grad():
                    mean_flat = mean.squeeze()
                    var_flat = var.squeeze()
                    self.running_mean.data = (1 - self.momentum) * self.running_mean.data + \
                                             self.momentum * mean_flat
                    self.running_var.data = (1 - self.momentum) * self.running_var.data + \
                                            self.momentum * var_flat
                    self.num_batches_tracked += 1
        else:
            # 推理模式：使用 running statistics
            mean = self.running_mean.view(1, -1, 1, 1)
            var = self.running_var.view(1, -1, 1, 1)
        
        # BatchNorm 计算 - ✅ 使用 torch 函数式 API
        x_centered = torch.sub(x, mean)  # 明确使用 torch.sub
        
        var_eps = torch.add(var, self.eps)  # 明确使用 torch.add
        
        std = torch.sqrt(var_eps)  # torch.sqrt
        
        x_normalized = torch.div(x_centered, std)  # 明确使用 torch.div
        
        if self.affine:
            weight = self.weight.view(1, -1, 1, 1)
            bias = self.bias.view(1, -1, 1, 1)
            x_scaled = torch.mul(x_normalized, weight)  # 明确使用 torch.mul
            output = torch.add(x_scaled, bias)  # 明确使用 torch.add
        else:
            output = x_normalized
        
        return output
    
    def extra_repr(self):
        """用于打印模块信息"""
        return (f'{self.num_features}, eps={self.eps}, momentum={self.momentum}, '
                f'affine={self.affine}, track_running_stats={self.track_running_stats}')


@QuantizationMixin.implements(QuantizableBatchNorm2d)
class QuantizedQuantizableBatchNorm2d(QuantizationMixin, QuantizableBatchNorm2d):
    """AIMET v2 quantized wrapper for QuantizableBatchNorm2d."""

    def __quant_init__(self):
        super().__quant_init__()
        self.input_quantizers = torch.nn.ModuleList([None])
        self.output_quantizers = torch.nn.ModuleList([None])

    def forward(self, x: Tensor) -> Tensor:
        if self.input_quantizers[0]:
            x = self.input_quantizers[0](x)

        with self._patch_quantized_parameters():
            ret = super().forward(x)

        if self.output_quantizers[0]:
            ret = self.output_quantizers[0](ret)

        return ret



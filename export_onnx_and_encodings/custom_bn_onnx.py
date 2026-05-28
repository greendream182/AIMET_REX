"""
自定义 BatchNorm2d 的 ONNX 导出支持。

职责：
- 提供占位模块 `ExportQuantizableBatchNorm2d`，
  用于把展开版 QuantizableBatchNorm2d 替换为单节点导出
- 提供 `replace_quantizable_batchnorm_modules` 函数，递归替换模型中所有 QuantizableBatchNorm2d

设计原理：
  QuantizableBatchNorm2d 将 BatchNorm 展开为若干基本算子（sub/add/sqrt/div/mul）以便
  AIMET 量化每个子算子。导出 ONNX 时，用标准 nn.BatchNorm2d 替换，PyTorch 原生
  exporter 会自动将其映射为单个 ONNX BatchNormalization 节点，无需注册自定义算子。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from aimet_torch.quantizable_batchnorm import QuantizableBatchNorm2d


class ExportQuantizableBatchNorm2d(nn.BatchNorm2d):
    """
    导出占位模块：把展开版 QuantizableBatchNorm2d 替换为标准 nn.BatchNorm2d。

    继承自 nn.BatchNorm2d，PyTorch ONNX exporter 会将其识别为标准 BN
    并导出为单个 ONNX BatchNormalization 节点。
    """

    def __init__(self, src: QuantizableBatchNorm2d) -> None:
        super().__init__(
            num_features=src.num_features,
            eps=src.eps,
            momentum=src.momentum if src.momentum is not None else 0.1,
            affine=src.affine,
            track_running_stats=src.track_running_stats,
        )
        with torch.no_grad():
            if src.affine and src.weight is not None and src.bias is not None:
                self.weight.copy_(src.weight)
                self.bias.copy_(src.bias)
            if src.track_running_stats and src.running_mean is not None and src.running_var is not None:
                self.running_mean.copy_(src.running_mean)
                self.running_var.copy_(src.running_var)
                if src.num_batches_tracked is not None:
                    self.num_batches_tracked.copy_(src.num_batches_tracked)


def replace_quantizable_batchnorm_modules(module: nn.Module) -> nn.Module:
    """递归替换模型中的 QuantizableBatchNorm2d 为导出占位模块。"""
    for name, child in list(module.named_children()):
        if isinstance(child, QuantizableBatchNorm2d):
            setattr(module, name, ExportQuantizableBatchNorm2d(child))
        else:
            replace_quantizable_batchnorm_modules(child)
    return module


__all__ = [
    "replace_quantizable_batchnorm_modules",
    "ExportQuantizableBatchNorm2d",
]

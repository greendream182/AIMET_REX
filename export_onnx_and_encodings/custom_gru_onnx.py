"""
自定义 GRU 的 ONNX 导出支持。

职责：
- 注册 torch.library 自定义算子 `custom_gru::custom_gru`
- 注册 legacy ONNX symbolic，把自定义算子导出为标准 ONNX `GRU` 节点
- 提供占位模块 `ExportOptimizedQuantizableGRU`，用于把 Python 展开版 GRU 替换为“单节点导出”
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.onnx import register_custom_op_symbolic, symbolic_helper

from aimet_torch.optimized_quantizable_gru import OptimizedQuantizableGRU

# torch.library 必须全局持有，否则 op 可能“消失”
_CUSTOM_GRU_LIB: Optional[torch.library.Library] = None


def ensure_custom_gru_op_registered(opset: int = 18) -> None:
    """注册 runtime op + ONNX symbolic。必须在导出 ONNX 前调用一次。"""
    if getattr(ensure_custom_gru_op_registered, "_done", False):
        return

    global _CUSTOM_GRU_LIB
    if _CUSTOM_GRU_LIB is None:
        _CUSTOM_GRU_LIB = torch.library.Library("custom_gru", "DEF")

    _CUSTOM_GRU_LIB.define(
        # 约定对齐 ONNX GRU：x 为 time-major [T,B,I]
        # 为了导出为标准 ONNX::GRU，我们在这里直接传入已打包/重排好的 W/R/B（ONNX 格式）。
        "custom_gru(Tensor x, Tensor h0, Tensor W, Tensor R, Tensor B, "
        "int hidden_size, int num_layers) -> (Tensor, Tensor)"
    )

    # BiGRU：导出为单个 ONNX GRU(direction=bidirectional)
    # 约定：x 为 time-major [T,B,I]，h0 为 [2, B, H]，W/R/B 为 ONNX bidirectional 格式：
    #   W[2,3H,I], R[2,3H,H], B[2,6H]
    _CUSTOM_GRU_LIB.define(
        "custom_bigru(Tensor x, Tensor h0, Tensor W, Tensor R, Tensor B, "
        "int hidden_size, int num_layers) -> (Tensor, Tensor)"
    )

    def _custom_gru_cpu(x, h0, W, R, B, hidden_size, num_layers):
        t, b = x.shape[0], x.shape[1]
        out = x.new_zeros(t, b, hidden_size)
        h_n = x.new_zeros(num_layers, b, hidden_size)
        return out, h_n

    try:
        _CUSTOM_GRU_LIB.impl("custom_gru", _custom_gru_cpu, dispatch_key="CPU")
    except TypeError:
        _CUSTOM_GRU_LIB.impl("custom_gru", "CPU", _custom_gru_cpu)

    def _custom_bigru_cpu(x, h0, W, R, B, hidden_size, num_layers):
        # 输出对齐 ExportOptimizedQuantizableBiGRU：out 为 [T,B,2H]，h_n 为 [2, B, H]
        t, b = x.shape[0], x.shape[1]
        out = x.new_zeros(t, b, 2 * hidden_size)
        h_n = x.new_zeros(2 * num_layers, b, hidden_size)
        return out, h_n

    try:
        _CUSTOM_GRU_LIB.impl("custom_bigru", _custom_bigru_cpu, dispatch_key="CPU")
    except TypeError:
        _CUSTOM_GRU_LIB.impl("custom_bigru", "CPU", _custom_bigru_cpu)

    def _symbolic_custom_gru(g, x, h0, W, R, B, hidden_size, num_layers):
        hidden_size_i = symbolic_helper._maybe_get_const(hidden_size, "i")
        num_layers_i = symbolic_helper._maybe_get_const(num_layers, "i")
        if hidden_size_i is None or num_layers_i is None:
            raise RuntimeError("custom_gru attributes must be constants for ONNX export")

        # 导出为标准 ONNX GRU（domain=""，op_type="GRU"）
        # 输入格式（ONNX）：X[T,B,I], W[1,3H,I], R[1,3H,H], B[1,6H], sequence_lens(optional), initial_h(optional)
        Y, Y_h = g.op(
            "GRU",
            x,
            W,
            R,
            B,
            symbolic_helper._optional_input_placeholder_tensor(g),
            h0,
            hidden_size_i=int(hidden_size_i),
            direction_s="forward",
            linear_before_reset_i=1,
            outputs=2,
        )

        # Y: [T, 1, B, H] -> squeeze axis=1 -> [T,B,H]
        axes = g.op("Constant", value_t=torch.tensor([1], dtype=torch.long))
        out = g.op("Squeeze", Y, axes)
        return out, Y_h

    register_custom_op_symbolic("custom_gru::custom_gru", _symbolic_custom_gru, opset)

    def _symbolic_custom_bigru(g, x, h0, W, R, B, hidden_size, num_layers):
        hidden_size_i = symbolic_helper._maybe_get_const(hidden_size, "i")
        num_layers_i = symbolic_helper._maybe_get_const(num_layers, "i")
        if hidden_size_i is None or num_layers_i is None:
            raise RuntimeError("custom_bigru attributes must be constants for ONNX export")

        # ONNX GRU bidirectional:
        #   Y: [T, 2, B, H], Y_h: [2, B, H]
        Y, Y_h = g.op(
            "GRU",
            x,
            W,
            R,
            B,
            symbolic_helper._optional_input_placeholder_tensor(g),
            h0,
            hidden_size_i=int(hidden_size_i),
            direction_s="bidirectional",
            linear_before_reset_i=1,
            outputs=2,
        )

        # 变换为 [T, B, 2H]（与 PyTorch BiGRU 的 feature concat 对齐）
        # [T,2,B,H] -> [T,B,2,H]
        Yp = g.op("Transpose", Y, perm_i=[0, 2, 1, 3])
        # reshape -> [T,B,2H]；0 代表继承维度
        shape = g.op("Constant", value_t=torch.tensor([0, 0, -1], dtype=torch.long))
        out = g.op("Reshape", Yp, shape)
        return out, Y_h

    register_custom_op_symbolic("custom_gru::custom_bigru", _symbolic_custom_bigru, opset)
    ensure_custom_gru_op_registered._done = True


class ExportOptimizedQuantizableGRU(nn.Module):
    """
    导出占位模块：统一处理单向/双向 OptimizedQuantizableGRU。

    - 单向：替换为标准 nn.GRU（走 PyTorch 官方 GRU symbolic）
    - 双向：替换为单节点 custom_bigru（导出为 ONNX GRU(direction=bidirectional)）
    """

    def __init__(self, src: OptimizedQuantizableGRU):
        super().__init__()
        assert src.num_layers == 1
        self.is_bidirectional = bool(getattr(src, "bidirectional", False))
        # 输出布局：与源模块一致
        self.batch_first = bool(getattr(src, "batch_first", False))
        # 输入布局：若源模块支持 input_batch_first，则以它为准；否则默认与 batch_first 一致
        self.input_batch_first = bool(getattr(src, "input_batch_first", self.batch_first))
        self.hidden_size = int(src.hidden_size)
        self.num_layers = int(src.num_layers)

        if not self.is_bidirectional:
            # 单向：统一内部用 batch-first 的标准 GRU，并拷贝原始参数（保持 PyTorch gate 布局 r,z,n）。
            cell = src.cells[0]
            self.gru = nn.GRU(
                input_size=int(src.input_size),
                hidden_size=self.hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=False,
            )
            with torch.no_grad():
                self.gru.weight_ih_l0.copy_(cell.weight_ih.weight)
                self.gru.weight_hh_l0.copy_(cell.weight_hh.weight)
                self.gru.bias_ih_l0.copy_(cell.weight_ih.bias)
                self.gru.bias_hh_l0.copy_(cell.weight_hh.bias)
            return

        # 双向：导出为 custom_bigru -> ONNX GRU(direction=bidirectional)
        H = int(self.hidden_size)

        def _reorder_rzn_to_zrh(t: torch.Tensor) -> torch.Tensor:
            assert t.shape[0] == 3 * H
            r, z, n = t[:H], t[H : 2 * H], t[2 * H :]
            return torch.cat([z, r, n], dim=0)

        cell_f = src.cells[0]
        reverse_cells = getattr(src, "reverse_cells", None)
        if reverse_cells is None or len(reverse_cells) == 0:
            raise RuntimeError("bidirectional OptimizedQuantizableGRU 缺少 reverse_cells，无法导出 BiGRU")
        cell_b = reverse_cells[0]

        W_ih_f = _reorder_rzn_to_zrh(cell_f.weight_ih.weight.detach())
        R_hh_f = _reorder_rzn_to_zrh(cell_f.weight_hh.weight.detach())
        b_ih_f = _reorder_rzn_to_zrh(cell_f.weight_ih.bias.detach())
        b_hh_f = _reorder_rzn_to_zrh(cell_f.weight_hh.bias.detach())

        W_ih_b = _reorder_rzn_to_zrh(cell_b.weight_ih.weight.detach())
        R_hh_b = _reorder_rzn_to_zrh(cell_b.weight_hh.weight.detach())
        b_ih_b = _reorder_rzn_to_zrh(cell_b.weight_ih.bias.detach())
        b_hh_b = _reorder_rzn_to_zrh(cell_b.weight_hh.bias.detach())

        self.register_buffer("W", torch.stack([W_ih_f, W_ih_b], dim=0).contiguous())
        self.register_buffer("R", torch.stack([R_hh_f, R_hh_b], dim=0).contiguous())
        self.register_buffer(
            "B",
            torch.stack(
                [
                    torch.cat([b_ih_f, b_hh_f], dim=0),
                    torch.cat([b_ih_b, b_hh_b], dim=0),
                ],
                dim=0,
            ).contiguous(),
        )

    def forward(self, x, h0=None):
        if not self.is_bidirectional:
            # 单向：统一到 batch-first，再调用标准 nn.GRU。
            x_bf = x if self.input_batch_first else x.transpose(0, 1)
            if h0 is None:
                h0 = x_bf.new_zeros((1, x_bf.size(0), self.hidden_size))

            out_bf, h_n = self.gru(x_bf, h0)
            return (out_bf if self.batch_first else out_bf.transpose(0, 1)), h_n

        # 双向：调用 custom_bigru
        x_tb = x.transpose(0, 1) if self.input_batch_first else x  # [T,B,I]
        if h0 is None:
            b_dim = x_tb.shape[1]
            try:
                b = int(b_dim)
                h0 = torch.zeros((2 * self.num_layers, b, int(self.hidden_size)), device=x.device, dtype=x.dtype)
            except Exception:
                b = x_tb.size(1)
                h0 = x.new_zeros((2 * self.num_layers, b, int(self.hidden_size)))

        out_tb, h_n = torch.ops.custom_gru.custom_bigru(
            x_tb,
            h0,
            self.W,
            self.R,
            self.B,
            int(self.hidden_size),
            int(self.num_layers),
        )  # out_tb: [T,B,2H]

        return (out_tb.transpose(0, 1) if self.batch_first else out_tb), h_n


def replace_optimized_gru_modules(module: nn.Module) -> nn.Module:
    """递归替换模型中的 OptimizedQuantizableGRU 为导出占位模块。"""
    for name, child in module.named_children():
        if isinstance(child, OptimizedQuantizableGRU):
            setattr(module, name, ExportOptimizedQuantizableGRU(child))  # type: ignore[arg-type]
        else:
            replace_optimized_gru_modules(child)
    return module


__all__ = [
    "ensure_custom_gru_op_registered",
    "replace_optimized_gru_modules",
    "ExportOptimizedQuantizableGRU",
]



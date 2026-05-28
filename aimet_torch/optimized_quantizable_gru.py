import torch
import torch.nn as nn
from aimet_torch._base.nn.modules.custom import Add, Multiply, Subtract


class OptimizedQuantizableGRUCell(nn.Module):
    """
    可量化 GRU Cell - 独立算术 Module（AIMET 量化专用）
    
    为什么使用独立 Module：
    - AIMET 量化要求每个操作位置有独立的量化器
    - 共享 Module 会导致量化器冲突，产生 NaN
    - 独立 Module 确保数值稳定和校准准确
    
    Args:
        input_size: 输入特征维度
        hidden_size: 隐藏状态维度
    """
    
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # ========== Linear 层 ==========
        # 注意：保持 PyTorch GRU gate 顺序 (r, z, n) 的参数布局，便于与 nn.GRU/导出逻辑兼容。
        # 在计算流程层面，我们会按 Haste QDQ 的“融合 Linear 输出 + z->r->g”顺序组织算子。
        self.weight_ih = nn.Linear(input_size, 3 * hidden_size, bias=True)  # W*x + bw（融合）
        self.weight_hh = nn.Linear(hidden_size, 3 * hidden_size, bias=True)  # R*h + br（融合）
        
        # ========== 激活函数 ==========
        # 为了让 encodings/internal_ops 的名字与 Haste QDQ 对齐，这里为每个门单独建激活模块。
        self.update_gate_output = nn.Sigmoid()
        self.reset_gate_output = nn.Sigmoid()
        self.new_gate_output = nn.Tanh()
        
        # ========== 独立的算术 Module（与 Haste QDQ 命名对齐）==========
        # ✅ 每个操作位置使用独立的 Module，避免量化器冲突，同时让 encodings 字段名可直接对齐 quant_params
        self.update_gate_input = Add()        # update_gate_input = ih_z + hh_z
        self.reset_gate_input = Add()         # reset_gate_input  = ih_r + hh_r
        self.mul_reset_hidden = Multiply()    # mul_reset_hidden  = reset_gate_output * hh_n
        self.new_gate_input = Add()           # new_gate_input    = ih_n + mul_reset_hidden

        self.sub_one_minus_update = Subtract()     # (1 - update_gate_output)
        self.mul_old_contribution = Multiply()     # mul_old_contribution = update_gate_output * h
        self.mul_new_contribution = Multiply()     # mul_new_contribution = (1-update_gate_output) * new_gate_output
        self.add_final_hidden = Add()              # h_new = mul_old_contribution + mul_new_contribution
        
        self.reset_parameters()

    # -------- 兼容旧命名（避免外部代码/旧 postprocess 依赖属性名时崩）--------
    @property
    def weight_ih_linear(self):
        return self.weight_ih

    @property
    def weight_hh_linear(self):
        return self.weight_hh

    @property
    def add_u_gate(self):
        return self.update_gate_input

    @property
    def add_r_gate(self):
        return self.reset_gate_input

    @property
    def add_n_gate(self):
        return self.new_gate_input

    @property
    def mul_reset(self):
        return self.mul_reset_hidden

    @property
    def add_final(self):
        return self.add_final_hidden

    @property
    def sub_one_minus(self):
        return self.sub_one_minus_update
    
    def reset_parameters(self):
        """使用与 PyTorch GRU 相同的初始化方式"""
        std = 1.0 / (self.hidden_size ** 0.5)
        for weight in self.parameters():
            weight.data.uniform_(-std, std)
    
    def forward(self, input, hidden):
        """
        Args:
            input: [batch, input_size]
            hidden: [batch, hidden_size]
        
        Returns:
            new_h: [batch, hidden_size]
        """

        # ========== 融合 Linear（与 Haste QDQ 对齐：weight_*_linear 已包含各自 bias）==========
        # weight_ih_linear = W*x + bw
        # weight_hh_linear = R*h + br
        weight_ih_linear = self.weight_ih(input)   # ✅ Linear 层输出被量化
        weight_hh_linear = self.weight_hh(hidden)  # ✅ Linear 层输出被量化
        
        # ========== 分割为三个门 ==========
        # PyTorch 参数布局是 (r, z, n)。为了让计算流程与 Haste QDQ（z, r, n）一致，
        # 这里在“使用阶段”做一次视图重排（不改变参数本身的存储布局/含义）。
        ih_r, ih_z, ih_n = weight_ih_linear.chunk(3, dim=1)
        hh_r, hh_z, hh_n = weight_hh_linear.chunk(3, dim=1)
        
        # ========== Update Gate (z 门) ==========
        update_gate_input = self.update_gate_input(ih_z, hh_z)          # ih_z + hh_z（已融合 bias）
        update_gate_output = self.update_gate_output(update_gate_input) # sigmoid

        # ========== Reset Gate (r) ==========
        reset_gate_input = self.reset_gate_input(ih_r, hh_r)            # ih_r + hh_r（已融合 bias）
        reset_gate_output = self.reset_gate_output(reset_gate_input)    # sigmoid
        
        # ========== New Gate (g 门 / Candidate) ==========
        # mul_reset_hidden = reset_gate * hh_n（hh_n 已含 br_n）
        mul_reset_hidden = self.mul_reset_hidden(reset_gate_output, hh_n)
        # new_gate_input = ih_n + mul_reset_hidden（ih_n 已含 bw_n）
        new_gate_input = self.new_gate_input(ih_n, mul_reset_hidden)
        new_gate_output = self.new_gate_output(new_gate_input)  # tanh
        
        # ========== 计算新的隐藏状态 ==========
        mul_old_contribution = self.mul_old_contribution(update_gate_output, hidden)
        one_minus_update = self.sub_one_minus_update(update_gate_output.new_tensor(1.0), update_gate_output)
        mul_new_contribution = self.mul_new_contribution(one_minus_update, new_gate_output)
        new_h = self.add_final_hidden(mul_old_contribution, mul_new_contribution)
        
        return new_h


class OptimizedQuantizableGRU(nn.Module):
    """
    可量化多层 GRU - 独立算术 Module（AIMET 量化专用）
    
    每个 Cell 包含 8 个独立的算术 Module，确保 AIMET 量化时：
    - 每个操作位置有独立的量化器
    - 避免量化器冲突和 NaN 错误
    - 校准速度正常，数值稳定
    
    Args:
        input_size: 输入特征维度
        hidden_size: 隐藏状态维度
        num_layers: GRU 层数
        bidirectional: 是否启用双向 GRU
        batch_first: **输出**布局。True -> 输出为 [B,T,H]；False -> 输出为 [T,B,H]
        input_batch_first: **输入**布局。None 表示与 batch_first 保持一致（兼容旧行为）。
            - True  : forward 接受 [B,T,I]
            - False : forward 接受 [T,B,I]
        dropout: 层间 dropout 概率
    """
    
    def __init__(self, input_size, hidden_size, num_layers=1, 
                 batch_first=True, dropout=0.0, input_batch_first=None, bidirectional=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bool(bidirectional)
        self.num_directions = 2 if self.bidirectional else 1
        self.batch_first = batch_first
        # 输入布局与输出布局解耦：默认与 batch_first 一致（完全兼容旧行为）
        self.input_batch_first = batch_first if input_batch_first is None else bool(input_batch_first)
        self.dropout = dropout
        
        # 创建多层 GRU Cell。
        # 为了兼容旧代码，self.cells 继续表示“正向”各层 Cell。
        self.cells = nn.ModuleList()
        self.reverse_cells = nn.ModuleList() if self.bidirectional else None
        for i in range(num_layers):
            cell_input_size = input_size if i == 0 else hidden_size * self.num_directions
            self.cells.append(OptimizedQuantizableGRUCell(cell_input_size, hidden_size))
            if self.bidirectional:
                self.reverse_cells.append(OptimizedQuantizableGRUCell(cell_input_size, hidden_size))
        
        if dropout > 0 and num_layers > 1:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = None
    
    def forward(self, input, h_0=None):
        """
        Args:
            input:
              - input_batch_first=True  : [B, T, I]
              - input_batch_first=False : [T, B, I]
            h_0:
              - 单向: [num_layers, batch, hidden_size]
              - 双向: [num_layers * 2, batch, hidden_size]
        
        Returns:
            output:
              - 单向: [*, *, H]
              - 双向: [*, *, 2H]
            h_n:
              - 单向: [num_layers, batch, hidden_size]
              - 双向: [num_layers * 2, batch, hidden_size]
        """
        # 统一内部计算布局为 batch-first: [B,T,*]
        if not self.input_batch_first:
            input = input.transpose(0, 1)
        
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        
        # 初始化隐藏状态
        if h_0 is None:
            h_0 = input.new_zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size)
        
        # 存储每层的输出
        layer_output = input
        h_n_list = []
        
        # 遍历每一层
        for layer_idx, forward_cell in enumerate(self.cells):
            forward_h_t = h_0[layer_idx * self.num_directions]
            forward_outputs = []
            
            # 正向时间步
            for t in range(seq_len):
                x_t = layer_output[:, t, :]
                forward_h_t = forward_cell(x_t, forward_h_t)
                forward_outputs.append(forward_h_t.unsqueeze(1))
            
            forward_output = torch.cat(forward_outputs, dim=1)

            if self.bidirectional:
                reverse_cell = self.reverse_cells[layer_idx]
                reverse_h_t = h_0[layer_idx * self.num_directions + 1]
                reverse_outputs = [None] * seq_len

                # 反向时间步，但输出位置仍然按原始时间轴对齐
                for t in range(seq_len - 1, -1, -1):
                    x_t = layer_output[:, t, :]
                    reverse_h_t = reverse_cell(x_t, reverse_h_t)
                    reverse_outputs[t] = reverse_h_t.unsqueeze(1)

                reverse_output = torch.cat(reverse_outputs, dim=1)
                layer_output = torch.cat([forward_output, reverse_output], dim=2)
                h_n_list.extend([forward_h_t.unsqueeze(0), reverse_h_t.unsqueeze(0)])
            else:
                layer_output = forward_output
                h_n_list.append(forward_h_t.unsqueeze(0))
            
            # 应用 dropout
            if self.dropout_layer is not None and layer_idx < self.num_layers - 1:
                layer_output = self.dropout_layer(layer_output)
        
        # 拼接所有层的最终隐藏状态
        h_n = torch.cat(h_n_list, dim=0)
        
        # 输出布局按 batch_first 控制
        if not self.batch_first:
            layer_output = layer_output.transpose(0, 1)
        
        return layer_output, h_n
    
    @classmethod
    def from_gru(cls, gru_module):
        """从 PyTorch nn.GRU 创建并迁移权重"""
        qgru = cls(
            input_size=gru_module.input_size,
            hidden_size=gru_module.hidden_size,
            num_layers=gru_module.num_layers,
            batch_first=gru_module.batch_first,
            dropout=gru_module.dropout if gru_module.num_layers > 1 else 0.0,
            bidirectional=gru_module.bidirectional,
        )
        
        # 迁移权重
        with torch.no_grad():
            for layer_idx in range(gru_module.num_layers):
                direction_specs = [("", qgru.cells[layer_idx])]
                if gru_module.bidirectional:
                    direction_specs.append(("_reverse", qgru.reverse_cells[layer_idx]))

                for suffix, cell in direction_specs:
                    weight_ih_name = f'weight_ih_l{layer_idx}{suffix}'
                    weight_hh_name = f'weight_hh_l{layer_idx}{suffix}'
                    bias_ih_name = f'bias_ih_l{layer_idx}{suffix}'
                    bias_hh_name = f'bias_hh_l{layer_idx}{suffix}'

                    weight_ih = getattr(gru_module, weight_ih_name)
                    weight_hh = getattr(gru_module, weight_hh_name)
                    bias_ih = getattr(gru_module, bias_ih_name)
                    bias_hh = getattr(gru_module, bias_hh_name)

                    cell.weight_ih.weight.copy_(weight_ih)
                    cell.weight_hh.weight.copy_(weight_hh)
                    cell.weight_ih.bias.copy_(bias_ih)
                    cell.weight_hh.bias.copy_(bias_hh)
        
        return qgru
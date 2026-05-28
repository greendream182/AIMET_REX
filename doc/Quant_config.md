# AIMET 量化配置说明

## 📐 双层配置架构

```
创建 QuantSim 时                应用混合精度时
    ↓                              ↓
基础配置 (JSON 1)            阶段配置 (JSON 2)
控制：是否量化               控制：位宽 + 对称性
  • is_quantized              • weight_bitwidth
  • per_channel               • input_bitwidth  
  • is_input_quantized        • output_bitwidth
                              • input_symmetric
                              • output_symmetric
```

## 🎯 快速配置

### 步骤 1: 创建 QuantSim（基础配置）

```python
sim = quantsim.QuantizationSimModel(
    prepared_model,
    dummy_input=sample_input.to(device),
    quant_scheme='percentile',
    config_file="mrnn_quantsim_config_custom_mixed_precision_v2.json",  # 基础配置
    default_output_bw=8,
    default_param_bw=8
)
```

**基础配置示例：**`mrnn_quantsim_config_custom_mixed_precision_v2.json`
```json
{
  "defaults": {
    "ops": {"is_output_quantized": "True"},
    "params": {"is_quantized": "True", "is_symmetric": "True"},
    "strict_symmetric": "False",
    "per_channel_quantization": "True"
  },
  "params": {
    "bias": {"is_quantized": "True"}
  },
  "op_type": {
    "Conv": {"is_input_quantized": "True"},
    "Gemm": {"is_input_quantized": "True"},
    "Add": {"is_input_quantized": "True"}
  }
}
```

#### 📝 基础配置参数详解

**1. `defaults` - 全局默认配置**

| 字段 | 值 | 说明 |
|------|---|------|
| `ops.is_output_quantized` | `"True"` | 量化所有算子的**输出激活值**（默认所有层输出都量化） |
| `params.is_quantized` | `"True"` | 量化所有**权重参数**（conv/linear 的 weight） |
| `params.is_symmetric` | `"True"` | 权重使用**对称量化**（范围 [-127,127]，零点=0） |
| `strict_symmetric` | `"False"` | False=-128 到127，True=-127到127 |
| `per_channel_quantization` | `"True"` | 权重使用**逐通道量化**（每个输出通道独立 scale） |

**2. `params` - 特定参数配置（覆盖 defaults）**

| 字段 | 值 | 说明 |
|------|---|------|
| `bias.is_quantized` | `"True"` | 偏置是否量化（通常建议 False 或使用高位宽） |



**3. `op_type` - 算子类型配置**

| 字段 | 值 | 说明 |
|------|---|------|
| `{op}.is_input_quantized` | `"True"` | 该算子的**输入是否量化** |
| `{op}.is_output_quantized` | `"True"` | 该算子的**输出是否量化** |
| `{op}.per_channel_quantization` | `"True"/"False"` | 该算子是否使用逐通道量化 |

⚠️ **重要**: `is_input_quantized` **不能**放在 `defaults` 中，必须在 `op_type` 中为每种算子单独配置！

#### 🏷️ op_type 命名规则

**op_type 使用 ONNX 算子名称**，而非 PyTorch 层名称：

| PyTorch 层 | ONNX 算子名 | 说明 |
|-----------|------------|------|
| `nn.Conv2d` | `Conv` | 2D 卷积 |
| `nn.ConvTranspose2d` | `ConvTranspose` | 转置卷积（上采样） |
| `nn.Linear` | `Gemm` | 全连接层（矩阵乘法） |
| `torch.add()` | `Add` | 加法算子 |
| `torch.mul()` | `Mul` | 乘法算子 |
| `torch.sub()` | `Sub` | 减法算子 |
| `torch.div()` | `Div` | 除法算子 |
| `torch.matmul()` | `MatMul` | 矩阵乘法 |
| `torch.pow()` | `Pow` | 幂运算 |
| `torch.sqrt()` | `Sqrt` | 平方根 |
| `torch.mean()` | `ReduceMean` | 均值（reduce 操作） |
| `F.pad()` | `Pad` | 填充 |
| `torch.sigmoid()` | `Sigmoid` | Sigmoid 激活 |
| `torch.tanh()` | `Tanh` | Tanh 激活 |

💡 **查找方法**: 
1. 导出 ONNX 模型查看算子名称
2. 参考 [ONNX 算子文档](https://github.com/onnx/onnx/blob/main/docs/Operators.md)
3. 在 `prepare_model()` 后，AIMET 会将 PyTorch 函数转换为带 ONNX 名称的 Module

### 步骤 2: 应用混合精度（阶段配置）

```python
from aimet_torch.utils_rx import apply_mixed_precision_bitwidth

apply_mixed_precision_bitwidth(
    sim.model,
    config_file="kws_config.json",  # 阶段配置
    verbose=True
)
```

**阶段配置示例：**`kws_config.json`
```json
{
  "default_bitwidth": {
    "weight": 8,
    "output": 8
  },
  "layer_type_config": {
    "QuantizedConv2d": {
      "weight_bitwidth": 2,
      "bias_bitwidth": 32,
      "input_bitwidth": 2,
      "output_bitwidth": 2,
      "input_symmetric": false,
      "output_symmetric": false
    }
  },
  "layer_name_config": {
    "conv_in": {
      "weight_bitwidth": 8,
      "input_bitwidth": 8,
      "output_bitwidth": 8
    }
  },
  "default_config": {
    "disable_quantization": true
  }
}
```

#### 📝 阶段配置参数详解

**1. `default_bitwidth` - 默认位宽**

| 字段 | 值 | 说明 |
|------|---|------|
| `weight` | 8 | 未匹配层的**权重默认位宽**（8-bit） |
| `output` | 8 | 未匹配层的**输出默认位宽**（8-bit） |

💡 这是兜底配置，只有在 `layer_type_config` 和 `layer_name_config` 都没有匹配时才使用。

**2. `layer_type_config` - 按层类型配置**

使用 **AIMET 量化后的类名**（与基础配置的 ONNX 名称不同！）（可以从print(sim.model)中去找）：

| 字段 | 值 | 说明 |
|------|---|------|
| `QuantizedConv2d` | {...} | 所有 Conv2d 层的配置 |
| `QuantizedLinear` | {...} | 所有 Linear 层的配置 |
| `QuantizedGRU` | {...} | 所有 GRU 层的配置 |
| `QuantizedAdd` | {...} | 所有 Add 算子的配置 |
| `QuantizedMultiply` | {...} | 所有 Multiply 算子的配置 |

**每个层类型可配置的字段：**

| 字段 | 类型 | 示例 | 说明 |
|------|-----|------|------|
| `weight_bitwidth` | int | 2, 4, 8, 16 | 权重量化位宽 |
| `bias_bitwidth` | int | 16, 32 | 偏置量化位宽（建议 ≥16） |
| `input_bitwidth` | int | 2, 4, 8, 16 | 输入激活值位宽 |
| `output_bitwidth` | int | 2, 4, 8, 16 | 输出激活值位宽 |
| `input_symmetric` | bool | true/false | 输入对称量化 |
| `output_symmetric` | bool | true/false | 输出对称量化 |
| `disable_quantization` | bool | true/false | 禁用该类型层的量化 |

**示例解读：**
```json
"QuantizedConv2d": {
  "weight_bitwidth": 2,          // 权重用 2-bit（极致压缩）
  "bias_bitwidth": 32,           // bias 用 32-bit（保持精度）
  "input_bitwidth": 2,           // 输入用 2-bit
  "output_bitwidth": 2,          // 输出用 2-bit
  "input_symmetric": false,      // 输入 [0, 3]
  "output_symmetric": false      // 输出 [0, 3]
}
```

**3. `layer_name_config` - 按层名称配置（精确匹配或通配符）**

使用模型中的**实际层名称**（通过 `sim.model.named_modules()` 查看）：

| 示例 | 说明 |
|------|------|
| `"conv_in"` | 精确匹配名为 `conv_in` 的层 |
| `"*.seq_t.*"` | 通配符匹配所有包含 `seq_t` 的层 |
| `"enc_seqs.0.*"` | 匹配 `enc_seqs.0` 下的所有子层 |

**可配置字段与 `layer_type_config` 相同。**

**示例解读：**
```json
"conv_in": {
  "weight_bitwidth": 8,    // 第一个卷积层用 8-bit（保持精度）
  "input_bitwidth": 8,     // 输入 8-bit（模型输入通常需要高精度）
  "output_bitwidth": 8     // 输出 8-bit
}
```

💡 **技巧**: 第一层和最后一层通常用高位宽（8-bit），中间层可以用低位宽（2-bit/4-bit）。

**4. `default_config` - 未匹配层的默认行为**

| 字段 | 值 | 说明 |
|------|---|------|
| `disable_quantization` | true | 禁用所有未匹配层的量化 |
| `disable_quantization` | false | 未匹配层使用 `default_bitwidth` |

**应用场景：**

```json
// 场景 1: 保守策略 - 只量化明确配置的层
{
  "default_config": {
    "disable_quantization": true  // 未配置的层全部禁用量化
  }
}

// 场景 2: 激进策略 - 全部量化，除了明确禁用的
{
  "default_config": {
    "disable_quantization": false  // 未配置的层使用 default_bitwidth
  }
}
```



## 📊 配置优先级

```
精确名称 > 模式匹配(*) > 层类型 > 默认位宽 > 默认配置
```

---

## 🎯 多阶段量化配置

### 📖 什么是多阶段量化？

**多阶段量化**是逐步启用不同层的量化，避免一次性量化所有层导致的精度崩溃。

```
阶段 1: 量化 Conv2d              → 校准 → QAT → 保存
  ↓
阶段 2: 量化 Conv2d + GRU        → 加载阶段1 → 校准 → QAT → 保存
  ↓
阶段 3: 量化 Conv2d + GRU + BN   → 加载阶段2 → 校准 → QAT → 保存
```

### 🔑 核心机制

1. **累积式配置**: 每个阶段的配置包含前一阶段的所有层
2. **参数继承**: 加载前一阶段的量化参数，避免重新校准已训练好的层
3. **权重继承**: 加载前一阶段的模型权重，保持 QAT 训练结果

---

### 📋 阶段 1: 量化 Conv2d

**配置文件: `stage1_conv_config.json`**

```json
{
  "layer_type_config": {
    "QuantizedConv2d": {
      "weight_bitwidth": 8,
      "input_bitwidth": 8,
      "output_bitwidth": 8,
      "disable_quantization": false
    }
  },
  "layer_name_config": {
    "conv_in": {
      "weight_bitwidth": 8,
      "input_bitwidth": 8,
      "output_bitwidth": 8
    }
  },
  "default_config": {
    "disable_quantization": true  // 其他层全部禁用
  }
}
```

**执行流程:**

```python
from aimet_torch.utils_rx import apply_mixed_precision_bitwidth
from aimet_torch.staged_quantization_utils import save_quantizer_encodings

# 1. 应用阶段1配置
apply_mixed_precision_bitwidth(
    sim.model,
    config_file="stage1_conv_config.json",
    verbose=True
)

# 2. 校准
run_calibration(sim.model, calib_loader, device, max_batches=100)

# 3. Power-of-2 量化 (可选)
apply_power_of_2_workflow(sim.model, verbose=False)

# 4. QAT 训练
run_qat(sim.model, train_loader, device, epochs=5)

# 5. 保存量化参数
save_quantizer_encodings(
    sim.model,
    save_path="stage1_encodings.json",
    layer_types=None,  # 保存所有已量化的层
    verbose=True
)

# 6. 保存模型权重
torch.save(sim.model.state_dict(), "stage1_weights.pth")
```

---

### 📋 阶段 2: 量化 Conv2d + Haste GRU

**配置文件: `stage2_gru_config.json`**

> **⚠️ 重要说明**: 本项目使用 **Haste GRU**（`QuantGRU`），这是一种优化的 C++ 实现，内部已包含量化逻辑。配置方式与标准 PyTorch GRU 不同。

```json
{
  "layer_type_config": {
    "QuantizedConv2d": {
      "weight_bitwidth": 2,
      "bias_bitwidth": 32,
      "input_bitwidth": 2,
      "output_bitwidth": 2,
      "input_symmetric": false,
      "output_symmetric": false,
      "disable_quantization": false
    }
  },
  "layer_name_config": {
    "conv_in": {
      "weight_bitwidth": 8,
      "input_bitwidth": 8,
      "output_bitwidth": 8,
      "comment": "第一个卷积层使用标准8位"
    },
    "fc0": {
      "disable_quantization": true,
      "comment": "暂时禁用全连接层的量化"
    }
  },
  "default_config": {
    "disable_quantization": true,
    "comment": "默认禁用所有其他层（Linear, Add, Multiply 等）"
  },
  "GRU_config": {
    "default_config": {
      "disable_quantization": false,
      "comment": "启用 Haste GRU 量化"
    },
    "operator_config": {
      "input.x": {
        "bitwidth": 8,
        "is_symmetric": false,
        "comment": "输入序列 x"
      },
      "input.h": {
        "bitwidth": 8,
        "is_symmetric": false,
        "comment": "隐藏状态 h"
      },
      "weight.W": {
        "bitwidth": 8,
        "is_symmetric": true,
        "comment": "输入权重 W"
      },
      "weight.R": {
        "bitwidth": 8,
        "is_symmetric": true,
        "comment": "循环权重 R"
      },
      "weight.bx": {
        "bitwidth": 8,
        "is_symmetric": true,
        "comment": "输入偏置 bx"
      },
      "weight.br": {
        "bitwidth": 8,
        "is_symmetric": true,
        "comment": "循环偏置 br"
      },
      "matmul.Wx": {
        "bitwidth": 8,
        "is_symmetric": false,
        "comment": "矩阵乘法 W @ x 结果"
      },
      "matmul.Rh": {
        "bitwidth": 8,
        "is_symmetric": false,
        "comment": "矩阵乘法 R @ h 结果"
      },
      "gate.z_pre": {
        "bitwidth": 8,
        "is_symmetric": false,
        "comment": "更新门 sigmoid 激活前"
      },
      "gate.z_out": {
        "bitwidth": 8,
        "is_symmetric": false,
        "comment": "更新门 sigmoid 激活后 [0,1]"
      },
      "gate.r_pre": {
        "bitwidth": 8,
        "is_symmetric": false,
        "comment": "重置门 sigmoid 激活前"
      },
      "gate.r_out": {
        "bitwidth": 8,
        "is_symmetric": false,
        "comment": "重置门 sigmoid 激活后 [0,1]"
      },
      "gate.g_pre": {
        "bitwidth": 8,
        "is_symmetric": false,
        "comment": "候选门 tanh 激活前"
      },
      "gate.g_out": {
        "bitwidth": 8,
        "is_symmetric": false,
        "comment": "候选门 tanh 激活后 [-1,1]"
      },
      "op.Rh_add_br": {
        "bitwidth": 8,
        "is_symmetric": false,
        "comment": "Rh + br 加法运算"
      },
      "op.rRh": {
        "bitwidth": 8,
        "is_symmetric": false,
        "comment": "r * Rh 元素乘法"
      },
      "op.old_contrib": {
        "bitwidth": 8,
        "is_symmetric": false,
        "comment": "z * h 旧状态贡献"
      },
      "op.new_contrib": {
        "bitwidth": 8,
        "is_symmetric": false,
        "comment": "(1-z) * g 新状态贡献"
      }
    }
  }
}
```

**配置说明:**

1. **标准 AIMET 配置**: 用于 Conv2d 等标准层
2. **特殊 `GRU_config` 配置**: 专门用于 Haste GRU (QuantGRU)
   - `default_config.disable_quantization: false` - 启用 GRU 量化
   - `operator_config` - 精细控制 GRU 内部所有算子的量化参数：
     - **输入/输出**: `input.x`, `input.h`
     - **权重/偏置**: `weight.W`, `weight.R`, `weight.bx`, `weight.br`
     - **矩阵乘法**: `matmul.Wx`, `matmul.Rh`
     - **门控单元**: `gate.z_*`, `gate.r_*`, `gate.g_*`（更新门、重置门、候选门）
     - **中间运算**: `op.Rh_add_br`, `op.rRh`, `op.old_contrib`, `op.new_contrib`

**执行流程:**

```python
from aimet_torch.staged_quantization_utils import load_quantizer_encodings
from quick_start_zxc import calibrate_quant_gru  # Haste GRU 专用校准函数

# 1. 应用阶段2配置
apply_mixed_precision_bitwidth(
    sim.model,
    config_file="stage2_gru_config.json",
    verbose=True
)

# 2. 🔄 加载阶段1的模型权重
sim.model.load_state_dict(
    torch.load("stage1_weights.pth", map_location=device)
)

# 3. 校准所有启用的层（Conv2d + Haste GRU）
# 3.1 常规层校准
run_calibration(sim.model, calib_loader, device, max_batches=100)

# 3.2 Haste GRU 专用校准（收集内部算子的激活范围）
calibrate_quant_gru(
    sim.model, 
    calib_loader, 
    device, 
    max_batches=100,
    calibration_method='histogram',
    verbose=True
)

# 4. 🔑 关键：加载阶段1的量化参数，覆盖 Conv2d 的校准结果
load_quantizer_encodings(
    sim.model,
    load_path="stage1_encodings.json",
    verbose=True
)
# ✅ 此时：Conv2d 保持阶段1的参数，Haste GRU 使用新校准的参数

# 5. Power-of-2 量化 (可选)
apply_power_of_2_workflow(sim.model, verbose=False)

# 6. QAT 训练
run_qat(sim.model, train_loader, device, epochs=5)

# 7. 保存量化参数（Conv2d + Haste GRU）
save_quantizer_encodings(
    sim.model,
    save_path="stage2_encodings.json",
    layer_types=None,
    verbose=True
)

# 8. 保存模型权重
torch.save(sim.model.state_dict(), "stage2_weights.pth")
```

**Haste GRU vs 标准 PyTorch GRU:**

| 特性 | 标准 PyTorch GRU | Haste GRU (QuantGRU) |
|------|------------------|----------------------|
| 实现方式 | Python + PyTorch | C++ + CUDA |
| 量化配置 | 使用 `layer_name_config` 匹配内部算子 | 使用专门的 `GRU_config.operator_config` |
| 校准方式 | 标准 `run_calibration()` | 需要额外调用 `calibrate_quant_gru()` |
| 量化粒度 | 粗粒度（整体权重、输入、输出） | 细粒度（内部每个算子独立配置） |
| 性能 | 标准速度 | 高性能优化 |

---

### 📋 阶段 3: 量化 Conv2d + Haste GRU + BatchNorm + fc0

**配置文件: `stage3_var_config.json`**

> **📌 说明**: 阶段3在阶段2的基础上，新增 BatchNorm 和 fc0 的量化。Haste GRU 的配置会通过加载阶段2的 encodings 继承。

```json
{
  "layer_type_config": {
    "QuantizedConv2d": {
      "weight_bitwidth": 8,
      "bias_bitwidth": 16,
      "input_bitwidth": 8,
      "output_bitwidth": 8,
      "disable_quantization": false,
      "comment": "Conv2d继承阶段1的配置"
    },
    "QuantizedLinear": {
      "weight_bitwidth": 8,
      "bias_bitwidth": 16,
      "input_bitwidth": 8,
      "output_bitwidth": 8,
      "disable_quantization": false,
      "comment": "Linear层（包括 fc0）"
    },
    "QuantizedAdd": {
      "input_bitwidth": 8,
      "output_bitwidth": 8,
      "disable_quantization": false,
      "comment": "Add操作（Haste GRU内部）继承阶段2"
    },
    "QuantizedMultiply": {
      "input_bitwidth": 8,
      "output_bitwidth": 8,
      "disable_quantization": false,
      "comment": "Multiply操作（Haste GRU内部）继承阶段2"
    },
    "QuantizedSubtract": {
      "input_bitwidth": 8,
      "output_bitwidth": 8,
      "disable_quantization": false,
      "comment": "Subtract操作（Haste GRU内部）继承阶段2"
    },
    "QuantizedBatchNorm2d": {
      "weight_bitwidth": 8,
      "input_bitwidth": 8,
      "output_bitwidth": 8,
      "disable_quantization": false,
      "comment": "BatchNorm2d在阶段3新增量化"
    },
    "QuantizedMean": {
      "input_bitwidth": 8,
      "output_bitwidth": 8,
      "disable_quantization": false,
      "comment": "Mean操作（BatchNorm内部）"
    }
  },
  "layer_name_config": {
    "fc0": {
      "weight_bitwidth": 8,
      "bias_bitwidth": 16,
      "input_bitwidth": 8,
      "output_bitwidth": 8,
      "disable_quantization": false,
      "comment": "输出全连接层在阶段3启用量化"
    },
    "pre_bn": {
      "weight_bitwidth": 8,
      "input_bitwidth": 8,
      "output_bitwidth": 8,
      "comment": "预处理 BatchNorm"
    },
    "enc_seqs.*.rnn2d_bn": {
      "weight_bitwidth": 8,
      "output_bitwidth": 8,
      "comment": "RNN2D BatchNorm（支持通配符）"
    }
  },
  "default_config": {
    "disable_quantization": true,
    "comment": "默认禁用所有其他层"
  }
}
```

> **⚠️ 注意**: 阶段3配置中**不需要**显式包含 `GRU_config`，因为 Haste GRU 的量化参数会通过 `load_quantizer_encodings()` 从阶段2继承。如果需要修改 Haste GRU 的配置，应该返回阶段2重新训练。

**执行流程:**

```python
# 1. 应用阶段3配置
apply_mixed_precision_bitwidth(
    sim.model,
    config_file="stage3_var_config.json",
    verbose=True
)

# 2. 🔄 加载阶段2的模型权重
sim.model.load_state_dict(
    torch.load("stage2_weights.pth", map_location=device)
)

# 3. 校准所有启用的层（Conv2d + Haste GRU + BN + fc0）
# 3.1 常规层校准
run_calibration(sim.model, calib_loader, device, max_batches=100)

# 3.2 Haste GRU 校准（如果配置文件中有 GRU_config）
# 注意：如果从阶段2加载 encodings，这步可以跳过
# calibrate_quant_gru(sim.model, calib_loader, device, max_batches=100)

# 4. 🔑 关键：加载阶段2的量化参数，覆盖 Conv2d + Haste GRU 的校准结果
load_quantizer_encodings(
    sim.model,
    load_path="stage2_encodings.json",
    verbose=True
)
# ✅ 此时：Conv2d + Haste GRU 保持阶段2的参数，BN + fc0 使用新校准的参数

# 5. Power-of-2 量化 (可选)
apply_power_of_2_workflow(sim.model, verbose=False)

# 6. QAT 训练
run_qat(sim.model, train_loader, device, epochs=5)

# 7. 保存最终量化参数
save_quantizer_encodings(
    sim.model,
    save_path="stage3_encodings.json",
    layer_types=None,
    verbose=True
)

# 8. 保存最终模型权重
torch.save(sim.model.state_dict(), "stage3_weights.pth")
```

---

### 🛠️ 工具函数说明

#### 1. `save_quantizer_encodings()` - 保存量化参数

```python
from aimet_torch.staged_quantization_utils import save_quantizer_encodings

save_quantizer_encodings(
    model,                          # AIMET 量化模型
    save_path="encodings.json",     # 保存路径
    layer_types=None,               # None=保存所有层，或指定 ["QuantizedConv2d"]
    verbose=True                    # 打印保存信息
)
```

**保存的内容:**
- 每个量化器的 `min`, `max`, `scale`, `offset`, `bitwidth`, `symmetric`
- JSON 格式，易于查看和调试

**示例输出:**
```json
{
  "conv_in.param_quantizers.weight": {
    "min": -0.5,
    "max": 0.5,
    "scale": 0.00390625,
    "offset": 0,
    "bitwidth": 8,
    "symmetric": true
  },
  "conv_in.input_quantizers.0": {
    "min": 0.0,
    "max": 2.0,
    "scale": 0.0078125,
    "offset": 0,
    "bitwidth": 8,
    "symmetric": false
  }
}
```

#### 2. `load_quantizer_encodings()` - 加载量化参数

```python
from aimet_torch.staged_quantization_utils import load_quantizer_encodings

load_quantizer_encodings(
    model,                          # AIMET 量化模型
    load_path="encodings.json",     # 加载路径
    verbose=True                    # 打印加载信息
)
```

**工作机制:**
- 从 JSON 文件读取量化参数
- 只恢复文件中存在的层（其他层不受影响）
- 覆盖校准结果，保持已训练好的层参数不变
- **特别注意**: Haste GRU 的内部算子量化参数也会被保存和恢复

#### 3. `calibrate_quant_gru()` - Haste GRU 专用校准

```python
from quick_start_zxc import calibrate_quant_gru

calibrate_quant_gru(
    model,                          # AIMET 量化模型
    calib_loader,                   # 校准数据加载器
    device,                         # 设备 (cuda/cpu)
    max_batches=100,                # 最大校准批次
    calibration_method='histogram', # 校准方法: 'histogram' 或 'min-max'
    verbose=True                    # 打印详细信息
)
```

**说明:**
- **必须在 `run_calibration()` 之后调用**，用于校准 Haste GRU 内部算子
- 使用前向钩子（forward hooks）收集 GRU 内部所有算子的激活值范围
- 支持两种校准方法：
  - `histogram`: 使用直方图统计（推荐，更准确）
  - `min-max`: 使用最小-最大值（更快）
- 校准的算子包括：
  - 输入/输出: `input.x`, `input.h`
  - 权重/偏置: `weight.W`, `weight.R`, `weight.bx`, `weight.br`
  - 矩阵乘法: `matmul.Wx`, `matmul.Rh`
  - 门控单元: `gate.z_*`, `gate.r_*`, `gate.g_*`
  - 中间运算: `op.Rh_add_br`, `op.rRh`, `op.old_contrib`, `op.new_contrib`

---

### ⚠️ 关键要点

#### 1. 为什么需要同时保存权重和量化参数？

| 文件类型 | 保存内容 | 作用 |
|---------|---------|------|
| `.pth` 权重文件 | 模型参数（weight, bias） | 保持 QAT 训练结果 |
| `.json` 量化参数 | scale, offset, bitwidth, symmetric | 保持校准结果 |

**两者缺一不可！**

#### 2. 加载顺序很重要

```python
# ✅ 正确顺序
sim.model.load_state_dict(...)       # 1. 先加载权重
run_calibration(...)                  # 2. 校准常规层
calibrate_quant_gru(...)              # 3. 校准 Haste GRU（如果有）
load_quantizer_encodings(...)         # 4. 覆盖已训练好的层     
```

#### 3. 配置文件的累积性

每个阶段的配置必须**包含前面所有阶段的层**：

```
阶段1: {Conv2d}
阶段2: {Conv2d, Haste GRU}       ← 包含阶段1
阶段3: {Conv2d, Haste GRU, BN, fc0}  ← 包含阶段1+2
```

#### 4. Haste GRU 的特殊性

**与标准 PyTorch GRU 的区别：**

1. **配置方式**:
   - 标准 GRU: 使用 `layer_name_config` + 通配符匹配内部算子
   - Haste GRU: 使用专门的 `GRU_config.operator_config`

2. **校准流程**:
   - 标准 GRU: 只需 `run_calibration()`
   - Haste GRU: 需要 `run_calibration()` + `calibrate_quant_gru()`

3. **量化粒度**:
   - 标准 GRU: 粗粒度（整体权重、输入、输出）
   - Haste GRU: 细粒度（内部每个算子独立配置，共 18 个量化点）

4. **性能**:
   - Haste GRU 使用 C++ + CUDA 实现，性能远超标准 PyTorch GRU

#### 5. 如何选择阶段划分？

**按敏感性从低到高:**

```
低敏感 → 高敏感
Conv2d → Linear/FC → Haste GRU → BatchNorm → Attention
```

**推荐策略:**
- 阶段1: 卷积层（最稳定）
- 阶段2: + Haste GRU（需要专门校准）
- 阶段3: + BatchNorm + 全连接层
- 阶段4: + 注意力机制（如果有）

---

### 📊 多阶段量化精度对比

| 阶段 | 启用的层 | 精度 | 说明 |
|------|---------|------|------|
| 浮点 | - | 84.89% | 基准 |
| 阶段1 | Conv2d | 82.5% | 卷积量化影响较小 |
| 阶段2 | + Haste GRU | 80.8% | GRU 量化有一定影响 |
| 阶段3 | + BN + fc0 | 81.7% | 最终精度恢复 |

**对比一次性全量化:**
- 一次性全量化: 1.70% ❌（精度崩溃）
- 多阶段量化: 81.74% ✅（精度保持）

---

### 💡 调试技巧

#### 1. 验证配置是否生效

```python
apply_mixed_precision_bitwidth(..., verbose=True)

# 输出示例：
# ✅ [type] QuantizedConv2d: 8 个
# ✅ [name] conv_in.weight: 8-bit
# 🚫 禁用量化层: 57 个
```

#### 2. 检查保存的量化参数

```python
import json

with open("stage2_encodings.json", "r") as f:
    encodings = json.load(f)
    
print(f"保存了 {len(encodings)} 个量化器")

# 检查 Haste GRU 的量化器
gru_quantizers = [k for k in encodings.keys() if 'seq_t' in k]
print(f"Haste GRU 量化器: {len(gru_quantizers)} 个")
for key in gru_quantizers[:5]:
    print(f"  - {key}")
```

#### 3. 对比阶段间的精度变化

```python
stage_accuracies = {}

# 阶段1
stage_accuracies['stage1'] = evaluate(sim.model, test_loader, device)

# 阶段2
stage_accuracies['stage2'] = evaluate(sim.model, test_loader, device)

# 打印对比
print(f"阶段1: {stage_accuracies['stage1']*100:.2f}%")
print(f"阶段2: {stage_accuracies['stage2']*100:.2f}%")
print(f"变化: {(stage_accuracies['stage2'] - stage_accuracies['stage1'])*100:.2f}%")
```

---

## 📚 参考资料

### 代码文件
- **完整示例（标准 PyTorch GRU）**: `quick_start.py` 第 492-520 行
- **完整示例（Haste GRU）**: `quick_start_zxc.py`
- **Haste GRU 校准函数**: `quick_start_zxc.py` 第 518-650 行（`calibrate_quant_gru`）

### 配置文件
- **基础量化配置**: `config/mrnn_quantsim_config_custom_mixed_precision_v2.json`
- **阶段1配置（Conv2d）**: `config/stage1_conv_config.json`
- **阶段2配置（Haste GRU）**: `config/stage2_gru_config.json`
- **阶段3配置（BatchNorm + fc0）**: `config/stage3_var_config.json`

### 文档
- **Haste GRU 用户手册**: `docs/QuantGRU_user_manual.md`
- **ONNX 算子文档**: https://github.com/onnx/onnx/blob/main/docs/Operators.md


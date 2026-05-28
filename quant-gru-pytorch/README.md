# quant-gru-pytorch（外部依赖）

> 仓库：[**CX9898/quant-gru-pytorch**](https://github.com/CX9898/quant-gru-pytorch)
>
> 本目录仅作说明，不包含源码, 请见上游 README。

## 与本仓库的关系

`aimet_rx` 用 `QuantGRU` 替代 `nn.GRU` 来获得 NPU 部署所需的细粒度量化能力

## 主要优点

- **算子级量化粒度**：把 GRU 内部每个算子（input/output、weight_ih/weight_hh、各门激活前后、中间乘加）独立暴露，可按 JSON 逐项配置位宽、对称性、`PER_TENSOR/PER_GATE/PER_CHANNEL`，远比 `nn.GRU` 整层量化精细。
- **CUDA + C++ 实现**：QAT 训练阶段也能跑得动，不会因为时序展开过慢卡训练。
- **接口兼容 `nn.GRU`**：直接替换即可。
- **与 AIMET 互通**：无缝接入 `aimet_rx` 的量化流程。

## 安装步骤

```bash
git clone https://github.com/CX9898/quant-gru-pytorch.git
cd quant-gru-pytorch

# 1. 编译 C++/CUDA 库
mkdir build && cd build
cmake ..
make -j$(nproc)
cd ..

# 2. 安装 Python 扩展
cd pytorch
pip install -e . --no-deps --no-build-isolation
```

环境要求：Python ≥ 3.9、PyTorch ≥ 2.0（含 CUDA）、CUDA Toolkit ≥ 11.0、C++17 编译器、CMake ≥ 3.18、OpenMP。

## 验证

跑一次最小 forward，CUDA 环境与 `.so` 加载会被一并验证（CPU 版 PyTorch 会在 `.cuda()` 处自动报错）：

```bash
python <<'PY'
import torch
from quant_gru import QuantGRU
gru = QuantGRU(8, 16, batch_first=True).cuda()
print(gru(torch.randn(2, 5, 8).cuda())[0].shape)
PY
```

输出 `torch.Size([2, 5, 16])` 即说明安装成功, 完整功能回归测试可在上游仓库内运行 `python pytorch/test_quant_gru.py`。

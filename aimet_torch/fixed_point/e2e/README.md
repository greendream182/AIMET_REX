# Fixed-point INT16 e2e — 接入新模型指南

> 本文档面向**首次接入**本套 INT16 fixed-point e2e 流水线的工程师（含外部客户）。
> 临时版本 v0：随实际接入案例增多会持续更新。

## 1. 三层架构（必读）

```
┌─────────────────────────────────────────────────────────────────┐
│  模型层（按模型族一文件，导出 build_prepared_* + sim wrapper） │
│  现有：mobilenet_v2.py                                          │
│  参考骨架（v0 临时）：attention.py / yolo.py / audio.py         │
├─────────────────────────────────────────────────────────────────┤
│  输出层（evaluator / QAT loop —— 与输出语义强耦合）             │
│  分类（单 logits + MSE）：int16_vs_fp32_cosine / train_int16_qat│
│                          — 见 mobilenet_v2.py，attention 可直接复用│
│  检测 / 序列任务：客户自己写（YOLO head / CTC / 嵌入差异太大）  │
├─────────────────────────────────────────────────────────────────┤
│  输入层（采样器，与输出语义无关，按张量形状划分）               │
│  inputs.py：image / audio_1d / melspec 三类 sampler             │
├─────────────────────────────────────────────────────────────────┤
│  PTQ 骨架（模型族无关，所有客户共用）                           │
│  sim_builder.build_calibrated_v2_sim                            │
│    CLE → BN fold → 经验偏差校正 → AdaRound                      │
│    → v2 QuantizationSim → ensure_output_quantizers_for_int16_eval│
│    → compute_encodings                                          │
└─────────────────────────────────────────────────────────────────┘
```

**复用边界**：

- **几乎所有模型**：复用 `build_calibrated_v2_sim` + 一个 `inputs.py` sampler。
- **分类模型（含 attention / ViT / Swin）**：可直接复用 `mobilenet_v2.int16_vs_fp32_cosine` 和 `mobilenet_v2.train_int16_qat`。
- **检测 / 语音任务**：PTQ 部分仍复用，但 evaluator / QAT 必须自己写（输出语义不同）。

## 2. 接入新模型的 5 步

> 假设你要接入一个名为 `MyModel` 的新模型。

### 步骤 1 — 准备模型 + dummy_input

```python
from aimet_torch.model_preparer import prepare_model

model = MyModel().eval()
model = prepare_model(model)            # 替换 functional + / mean 为 nn 模块
dummy_input = torch.randn(1, ...)        # 与模型 forward 真实输入同 shape
```

注意：

- `prepare_model` 对 BN-based CNN 必跑；对 ViT / Transformer 可跑可不跑（LN 无需 fold）。
- `dummy_input` 决定后续 BN-fold 与 AdaRound 的输入形状，**必须**与你 forward 时一致。

### 步骤 2 — 选 sampler（决定校准数据怎么生成）

| 你的输入是 … | 用这个 sampler |
|---|---|
| `(B, 3, H, W)` 图像（含 ViT 也走这条） | `make_image_sampler(H)` |
| `(B, C, T)` 1D 波形 | `make_audio_1d_sampler(T, channels=C)` |
| `(B, 1, n_mels, n_frames)` mel 谱 | `make_melspec_sampler(n_mels, n_frames)` |
| 其它（多模态 / 自定义） | 自己写：`def sampler() -> torch.Tensor:` |

```python
from aimet_torch.fixed_point.e2e.inputs import make_image_sampler

sampler = make_image_sampler(input_size=224, in_channels=3, batch=4)
```

### 步骤 3 — 调 PTQ 骨架拿 sim

```python
from aimet_torch.fixed_point.e2e.sim_builder import build_calibrated_v2_sim

bundle = build_calibrated_v2_sim(
    model,
    dummy_input,
    calibration_sampler=sampler,
    calibration_iters=4,           # 4 个 batch 通常够
    # 按模型族开关 PTQ 步骤：
    apply_cle=False,               # CNN 才开；ViT 关
    apply_bn_fold=True,            # CNN/audio-CNN 开；ViT 关（无 BN）
    bias_correction_data=None,     # 需要 BC 时传 DataLoader
    adaround_loader=None,          # 需要 AdaRound 时传 list[(images, labels)]
    adaround_filename_prefix="mymodel",
)
sim = bundle.sim                   # 这就是 v2 QuantizationSimModel
```

`bundle.n_oq_patched` 是 INT16 dispatch 修补的输出 quantizer 个数；> 0 才能跑 INT16_FIXED_EVAL。

### 步骤 4 — 写 evaluator（INT16 vs FP32 比对）

#### 4.1 你的模型输出是**单个 logits 张量**（分类 / 嵌入回归）

直接复用：

```python
from aimet_torch.fixed_point.e2e.mobilenet_v2 import int16_vs_fp32_cosine

cos = int16_vs_fp32_cosine(sim, x)     # x 是任意 batch
assert cos >= 0.99
```

#### 4.2 你的模型输出是**多 head**（YOLO / 多任务）

参考 `yolo.py` 里的 TODO 注释，自己写：

```python
def int16_vs_fp32_detection(sim, x) -> dict:
    with quant_execution_mode(ExecutionMode.FP32_QDQ):
        ref_outs = sim.model(x)        # tuple/list of (cls, box, obj, ...)
    with quant_execution_mode(ExecutionMode.INT16_FIXED_EVAL):
        int_outs = sim.model(x)
    return {
        f"head_{i}_cosine": cosine(ref, intr.to_float())
        for i, (ref, intr) in enumerate(zip(ref_outs, int_outs))
    }
```

#### 4.3 你的模型是 CTC / 嵌入

参考 `audio.py` 里的 TODO 注释，按你的 loss 设计 evaluator（典型：在 final embedding / logits 上取 cosine，或在 alignment 上取 WER 差）。

### 步骤 5 — 写 QAT loop（如需要）

#### 5.1 单 logits 输出：可直接复用 `train_int16_qat`

```python
from aimet_torch.fixed_point.e2e.mobilenet_v2 import train_int16_qat

losses = train_int16_qat(
    sim, teacher=fp32_teacher, input_size=224, epochs=10, lr=1e-3,
)
```

但注意：`train_int16_qat` 内部写死 `(CALIB_BATCH, 3, input_size, input_size)`。
若你的输入不是这个形状，必须自己写 QAT loop（见 `yolo.py` / `audio.py`）。

#### 5.2 其它情况

按 `INT16_FIXED_QAT_SIM` 模式 forward + 任意 loss 即可，详见 `yolo.py` / `audio.py` 模板。

## 3. 三类模型的特殊注意事项

### Attention（ViT / Swin / 自研 Transformer）

| 项 | 处理 |
|---|---|
| BN | 通常无（用 LN）→ `apply_bn_fold=False` |
| CLE | 不适用 → `apply_cle=False` |
| 经验偏差校正 | 一般不需要；若开，输入数据用 image sampler 即可 |
| AdaRound | 一般有效 |
| evaluator | 直接复用 `int16_vs_fp32_cosine` |
| QAT | 直接复用 `train_int16_qat`（输入是 `(B, 3, H, W)` 且分类） |

### YOLO（YOLOv5 / v8 / 自研 detector）

| 项 | 处理 |
|---|---|
| BN | 大部分用 BN → `apply_bn_fold=True` |
| CLE | 一般有效 |
| 经验偏差校正 | 注意 `correct_bias` 内部假设单 head 输出，YOLO 多 head 时可能需要自定义 |
| AdaRound | 有效（label 不被 AdaRound 使用） |
| evaluator | **必须自己写**（多 head 输出） |
| QAT | **必须自己写**（检测 loss 或加权 per-head MSE） |

### 音频（Wav2Vec / CRNN / 自研 1D-CNN）

| 项 | 处理 |
|---|---|
| 输入形状 | 1D 波形：`(B, C, T)`；mel：`(B, 1, n_mels, n_frames)` |
| BN | BN1d 也能 fold（`fold_all_batch_norms` 支持） |
| CLE | 视模型结构，一般有效 |
| 经验偏差校正 | 输入数据需匹配模型输入形状（用 audio_1d / melspec sampler） |
| AdaRound | 有效 |
| evaluator | 视任务：分类用 `int16_vs_fp32_cosine`；CTC / 嵌入按需自写 |
| QAT | 视任务，分类可复用（需改输入形状），CTC 必须自写 |

## 4. 推荐的接入工作流

1. **复制临时骨架**：从 `attention.py` / `yolo.py` / `audio.py` 中选最接近你模型族的一份，cp 到你客户的代码库（或本 repo 的新文件）。
2. **替换 dummy 模型**：把骨架里的 `_MinimalXxx` 替换为你的真实模型构造。
3. **跑 PTQ smoke**：先确保 `build_calibrated_v2_sim` 能成功 + `bundle.n_oq_patched > 0`。
4. **写 evaluator**：先实现最简单的"vs FP32 单 batch cosine"，验证 INT16 不爆。
5. **跑 PTQ 回归**：用客户真实数据测 PTQ 后 INT16 vs FP32 的 top-1 drop / box mAP drop / WER 差。
6. **如需 QAT**：再写 QAT loop。

## 5. INT16 后端算子覆盖度（接入时大概率会遇到）

**保证可跑**：FP32_QDQ 与 FP16_QDQ 路径只要 v2 QuantizationSim 能构出来就能跑。

**可能失败**：INT16_FIXED_EVAL 与 INT16_FIXED_QAT_SIM 依赖 fixed-point kernel 的覆盖度，
任何模型族都可能命中"某个算子无 INT16 实现"的报错，例如：

```
RuntimeError: INT16 fixed-point execution is not implemented for
              QuantizedAdaptiveAvgPool2d. Add a fixed-point kernel/adapter
              path or run in fp32_qdq/fp16_qdq mode.
```

或

```
AttributeError: 'Int16QuantizedTensor' object has no attribute 'flatten'
```

应对策略（按优先级）：

1. **先用 FP32_QDQ 把 PTQ 流水线跑通**（CLE/fold/BC/AdaRound 都能在 FP32_QDQ 下验证）；
2. **检查 `bundle.n_oq_patched`**：必须 > 0，否则 INT16_FIXED_EVAL 一定不能 dispatch；
3. **缺什么算子向 e2e 维护者反馈**：在 issue / PR 里附上你的模型 + 报错 traceback，
   维护者会评估补 INT16 kernel（在 `aimet_torch/fixed_point/kernels/` 下）或提供 adapter；
4. **临时绕过**：能改模型的就替换缺失算子（如 `AdaptiveAvgPool2d(1)` 替换为
   `AvgPool2d(kernel_size=H)` 或 `mean(dim=[-2,-1], keepdim=True)`），不能改的就先停在
   FP32_QDQ + FP16_QDQ 验证阶段，等 kernel 补齐。

> 当前三个临时骨架（attention/yolo/audio）执行 ``python -m aimet_torch.fixed_point.e2e.<name>``
> 时的实测结论：yolo 完全通过；attention 因 `flatten` 卡住；audio 因
> `AdaptiveAvgPool` 卡住 —— 都是"backend 算子覆盖度"问题，**不代表骨架结构错**，
> 真实客户模型按需上报算子覆盖即可。

## 6. 不要做的事

- **不要直接编辑 `sim_builder.py` 的 `build_calibrated_v2_sim` 加模型族专属逻辑**。任何"我的模型需要在 PTQ 之前/之后做 X"都应在你的 wrapper 里做，保持骨架干净。
- **不要在没有 evaluator 的情况下就上 QAT**。先确保 PTQ 阶段 INT16 vs FP32 的差距已知，QAT 才有可观测的目标。
- **不要把临时骨架（attention.py / yolo.py / audio.py）当生产代码**。当某类模型有真实接入后，请：
  1. 在 `aimet_torch/fixed_point/e2e/` 内新建以真实模型命名的文件（如 `vit_b_16.py`、`yolov8.py`、`wav2vec.py`）；
  2. 把对应的 dummy / TODO 替换为生产实现；
  3. 删除或精简临时骨架（避免出现两份 attention.py 长期并存）。

## 7. 在哪里 PR / 改谁

- PTQ 骨架（`sim_builder.py`）与输入工具（`inputs.py`）：本套 e2e 维护者 review。
- 新模型 wrapper（`<model>.py`）：客户工程师自行 owner，e2e 维护者只 review 接口规范。
- 临时骨架（attention.py / yolo.py / audio.py）：v0 临时文件，谁接入谁更新；过 2 个真实接入后整体清理。

"""
AIMET 量化演示 — 标准用法

本脚本演示 AIMET v2 在 SpeechCommands + MRNN（关键词识别）上的完整量化流程：
    FP 训练 → prepare_model → QuantizationSimModel + 混合精度位宽
        → compute_encodings 校准 (PTQ)
        → apply_power_of_2_workflow (NPU 友好的 Po2 量化)
        → freeze_quantizer_parameters + QAT 微调
        → state_dict + ONNX + .encodings 三件套保存
        → 重建 sim → load_state_dict → load_quantizer_encodings → 兜底 calib → 精度对比

主流程见文件末尾的 main()，所有标准 AIMET API 调用都直接内联在 main 里，
便于直接照抄；前面的函数只负责"非 AIMET"的辅助逻辑（数据集、模型定义、训练循环）。
"""

# ============================================================================
# 基础导入
# ============================================================================
import os
import math
import functools
import random
import time
import contextlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# AIMET 安装包公共 API
import aimet_torch.v2 as aimet
from aimet_torch import model_preparer
from aimet_torch.v2 import quantsim
from aimet_torch.utils_rx import (
    apply_mixed_precision_bitwidth,
    apply_power_of_2_workflow,
    freeze_quantizer_parameters,
    set_train_mode_freeze_bn,
)
from aimet_torch.fixed_point import ExecutionMode, quant_execution_mode
from aimet_torch.staged_quantization_utils import load_quantizer_encodings
from aimet_torch.quantizable_batchnorm import QuantizableBatchNorm2d
from quant_gru import QuantGRU

# 仓库自定义的 ONNX 导出工具（含 QuantGRU / QuantizableBatchNorm2d 的自定义符号化逻辑），
# 已随安装包一起发布
from export_onnx_and_encodings.export_onnx_json import export_onnx_json

# 当前 demo 的本地辅助模块（与本脚本同目录），运行 demo 时请进入 examples/ 后再启动
from common.fft2band import BandConverter
from common.torch_stft import STFT


# ============================================================================
# 全局配置
# ============================================================================
SEED = 42
EPS = 1e-8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = "/home/llq/workspace/data/speech_commands"
if not os.path.isdir(DATA_ROOT):
    DATA_ROOT = "/mnt/data8t/share/datasets/speech_commands/SpeechCommands/speech_commands_v0.02"
NUM_CLASSES = 35
BATCH_SIZE = 64
NUM_WORKERS = 4

FP_EPOCHS = 30
QAT_EPOCHS = 1
FP_LR = 1e-3
QAT_LR = 1e-4
FP_LR_MIN = 1e-5

# 量化方案 —— AIMET QuantizationSimModel.quant_scheme 接受以下输入：
#   字符串别名:  "min_max" | "tf" | "tf_enhanced" | "percentile"   ("tf" 是 "min_max" 的别名)
#   枚举:        QuantScheme.min_max | QuantScheme.post_training_tf_enhanced
#                | QuantScheme.post_training_percentile
# 注：QuantGRU 校准方法会从该值自动推断（minmax / sqnr / percentile）。
QUANT_SCHEME = "percentile"
PERCENTILE_VALUE = 99.99       # 仅 QUANT_SCHEME == "percentile" 时生效（其他 scheme 下 set_percentile_value 是空操作）
DEFAULT_BW = 8
MAX_CALIB_BATCHES = 100

_HERE = Path(__file__).resolve().parent
CONFIG_FILE = _HERE / "config" / "mrnn_quantsim_config_custom_mixed_precision_v2.json"
BITWIDTH_CONFIG_FILE = _HERE / "config" / "quick_start_full_quant.json"
OUTPUT_DIR = _HERE / "output" / "quick_start"
FP_MODEL_PATH = _HERE / "model_fp.pth"
# §2.3 CLZ encoding：sign bypass（校准前）+ reciprocal/power_2（Po2 后）
MRNN_CLZ_ENCODING_FIX = os.environ.get("MRNN_CLZ_ENCODING_FIX", "0") == "1"


# ============================================================================
# 复现性 / 环境辅助
# ============================================================================
def set_seed(seed: int = SEED) -> None:
    """统一设置 random / numpy / torch / cuDNN 的随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def worker_init_fn(worker_id: int) -> None:
    seed = SEED + worker_id
    random.seed(seed)
    np.random.seed(seed)


def setup_audio_backend() -> None:
    """torchaudio 后端选择，避免对 torchcodec 的硬依赖。"""
    for backend in ("soundfile", "sox_io"):
        try:
            torchaudio.set_audio_backend(backend)
            return
        except Exception:
            continue


# ============================================================================
# 数据集（非 AIMET 标准用法，与量化无关）
# ============================================================================
def list_from_txt(path: str) -> list:
    with open(path, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]


class SpeechCommands(Dataset):
    """从 speech_commands_v0.02 目录构建 train/val/test。返回 (sequence[T, 1], label_idx)。"""

    def __init__(self, root, split="train", sample_rate=16000,
                 add_noise_p=0.6, time_shift_max=0.1, target_dur=1.0):
        self.root = Path(root)
        assert self.root.exists(), f"Data root not found: {root}"
        self.split = split
        self.sr = sample_rate
        self.target_len = int(target_dur * sample_rate)
        self.time_shift_max = time_shift_max
        self.add_noise_p = add_noise_p

        self.labels = sorted(
            d.name for d in self.root.iterdir()
            if d.is_dir() and not d.name.startswith("_")
        )
        self.label_to_idx = {c: i for i, c in enumerate(self.labels)}

        val_list = set(list_from_txt(self.root / "validation_list.txt"))
        test_list = set(list_from_txt(self.root / "testing_list.txt"))

        all_items = []
        for label in self.labels:
            for wav in (self.root / label).glob("*.wav"):
                rel = f"{label}/{wav.name}"
                if rel in test_list:
                    sp = "test"
                elif rel in val_list:
                    sp = "val"
                else:
                    sp = "train"
                all_items.append((wav, label, sp))

        self.items = [(p, l) for (p, l, sp) in all_items if sp == split]
        if len(self.items) == 0:
            raise RuntimeError(f"No items for split={split} at {root}")

        self.bg_noises = []
        noise_dir = self.root / "_background_noise_"
        if noise_dir.exists():
            for w in noise_dir.glob("*.wav"):
                wav, sr = torchaudio.load(w)
                if sr != self.sr:
                    wav = torchaudio.functional.resample(wav, sr, self.sr)
                self.bg_noises.append(wav.squeeze(0))

    def __len__(self):
        return len(self.items)

    def _pad_or_crop(self, wav, train=True):
        L = wav.shape[-1]
        if L < self.target_len:
            wav = torch.nn.functional.pad(wav, (0, self.target_len - L))
        elif L > self.target_len:
            if self.split == "train" and train:
                start = random.randint(0, L - self.target_len)
            else:
                start = (L - self.target_len) // 2
            wav = wav[:, start:start + self.target_len]
        return wav

    def _time_shift(self, wav):
        if self.time_shift_max <= 0:
            return wav
        max_shift = int(self.target_len * self.time_shift_max)
        shift = random.randint(-max_shift, max_shift)
        return torch.roll(wav, shifts=shift, dims=-1)

    def _mix_bg_noise(self, wav):
        if not self.bg_noises or random.random() > self.add_noise_p:
            return wav
        noise = random.choice(self.bg_noises)
        if noise.numel() < self.target_len:
            rep = (self.target_len // noise.numel()) + 1
            noise = noise.repeat(rep)
        start = random.randint(0, noise.numel() - self.target_len)
        noise_seg = noise[start:start + self.target_len].unsqueeze(0)

        snr_db = random.uniform(-3.0, 15.0)
        sig_pow = wav.pow(2).mean()
        noi_pow = noise_seg.pow(2).mean() + 1e-9
        k = math.sqrt(sig_pow / (noi_pow * (10 ** (snr_db / 10.0))))
        return torch.clamp(wav + k * noise_seg, -1.0, 1.0)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)

        train_mode = (self.split == "train")
        wav = self._pad_or_crop(wav, train=train_mode)
        if train_mode:
            wav = self._time_shift(wav)
            wav = self._mix_bg_noise(wav)

        wav_tc = wav.T  # [1, T] -> [T, 1]
        y = torch.tensor(self.label_to_idx[label], dtype=torch.long)
        return wav_tc, y


def collate(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0)


def build_dataloaders(root: str):
    """构建 train / test / val / calib 四个 DataLoader（带固定种子 generator）。"""
    train_ds = SpeechCommands(root, split="train", add_noise_p=0.0)
    test_ds = SpeechCommands(root, split="test", add_noise_p=0.0)
    val_ds = SpeechCommands(root, split="val", add_noise_p=0.0)

    g_train = torch.Generator(); g_train.manual_seed(SEED)
    g_calib = torch.Generator(); g_calib.manual_seed(SEED)

    def _make(ds, *, shuffle, generator=None, drop_last=False):
        return DataLoader(
            ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=NUM_WORKERS,
            pin_memory=True, collate_fn=collate, drop_last=drop_last,
            worker_init_fn=worker_init_fn if shuffle else None,
            generator=generator,
        )

    return {
        "train": _make(train_ds, shuffle=True, generator=g_train, drop_last=True),
        "test":  _make(test_ds,  shuffle=False),
        "val":   _make(val_ds,   shuffle=False),
        "calib": _make(test_ds,  shuffle=True, generator=g_calib, drop_last=True),
    }


def fresh_calib_loader(loader: DataLoader) -> DataLoader:
    """独立 calib DataLoader（同 SEED、epoch 1），避免多 sim 串行 build 时迭代器偏移。"""
    g = torch.Generator()
    g.manual_seed(SEED)
    return DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        shuffle=True,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        collate_fn=loader.collate_fn,
        drop_last=loader.drop_last,
        worker_init_fn=worker_init_fn,
        generator=g,
    )


# ============================================================================
# 模型定义（非 AIMET 标准用法，与量化无关）
# ============================================================================
def _fp16_safe_forward(forward_fn):
    """fp16_qdq 兜底：preserved stateless module 收到 fp16 输入时，cast 到 fp32 算后再 cast 回。

    动机：``stateless_modules_to_preserve`` 的模块不被 v2 wrapper 接管，没有 fp16 promote；
    而 EPS=1e-8 在 fp16 下是 subnormal（fp16 normal min ≈ 6e-5），``clamp(min=EPS)``
    + ``sqrt`` / ``div`` 容易触发 NaN/Inf。这里只在 fp16_qdq 模式下显式升精度，FP32 / FIXED_SCALE_QDQ
    路径完全 no-op。
    """

    @functools.wraps(forward_fn)
    def wrapped(self, *args, **kwargs):
        in_fp16 = any(
            isinstance(a, torch.Tensor) and a.dtype == torch.float16 for a in args
        )
        if not in_fp16:
            return forward_fn(self, *args, **kwargs)
        promoted = tuple(
            a.float() if isinstance(a, torch.Tensor) and a.dtype == torch.float16 else a
            for a in args
        )
        out = forward_fn(self, *promoted, **kwargs)
        if isinstance(out, torch.Tensor) and out.is_floating_point():
            out = out.to(torch.float16)
        return out

    return wrapped


class PowerCompress(nn.Module):
    """Power compression: sqrt(|x|) * sign(x) — trace-friendly for INT16 decompose."""

    @_fp16_safe_forward
    def forward(self, x):
        return torch.mul(torch.sign(x), torch.sqrt(torch.abs(x)))


class HypotFun(nn.Module):
    """Hypot: sqrt(max(x^2 + y^2, EPS)) — trace-friendly; clamp 代替 +EPS 以利 INT16。"""

    @_fp16_safe_forward
    def forward(self, x, y):
        sum_sq = torch.add(torch.square(x), torch.square(y))
        return torch.sqrt(torch.clamp(sum_sq, min=EPS))


class CLN(nn.Module):
    """通道层归一化（按 (C, F) 维度对每个 (B, T) 做幅度归一化）。"""

    def __init__(self, factor=32):
        super().__init__()
        self.factor = factor

    @_fp16_safe_forward
    def forward(self, x):
        mean_sq = torch.mean(torch.square(x), dim=(1, 3), keepdim=True)
        std = torch.sqrt(torch.clamp(mean_sq, min=EPS))
        return torch.div(x, std)


class RNN2D(nn.Module):
    """
    使用 QuantGRU 的 RNN2D 模块（单向）。
    """

    def __init__(self, H):
        super().__init__()
        self.H = H
        self.seq_t = QuantGRU(input_size=H, hidden_size=H, batch_first=True,
                              num_layers=1, bidirectional=False)
        self.conv_t = nn.Conv2d(H, H, (1, 1))
        self.rnn2d_bn = QuantizableBatchNorm2d(H, affine=False, momentum=0.01)
        self.cln = CLN(factor=32)

    def forward(self, x):
        x = self.rnn2d_bn(x)
        x = self.cln(x)
        b, c, t, f = x.shape
        o = x.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        o, _ = self.seq_t(o)
        o = o.view(b, f, t, self.H).permute(0, 3, 2, 1).contiguous()
        return self.conv_t(o)


class FrequencyDownSampling(nn.Module):
    def __init__(self, in_size, out_size, stride=(1, 4), kernel_size=(1, 4)):
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels=in_size, out_channels=out_size,
            kernel_size=kernel_size, stride=stride,
            padding=(kernel_size[0] // 2, 0),
        )

    def forward(self, ipt):
        ipt = torch.clamp(ipt, -0, 2)
        out = self.conv2d(ipt)
        return torch.clamp(out, -2, 2)


class MRNN(nn.Module):
    """KWS 主模型：STFT → BandConverter → 多级 RNN2D + 频率下采样 → FC。"""

    def __init__(self, output_dim=NUM_CLASSES, NFFT=512, frame_size=160, fbank_num=240):
        super().__init__()
        channels = [120, 240, 320]
        freq_bins = NFFT // 2

        self.trans = STFT(filter_length=NFFT, hop_length=frame_size)
        self.pre_bn = QuantizableBatchNorm2d(freq_bins, affine=True, momentum=0.01)
        self.fft2band = BandConverter(band_num=fbank_num, freq_bins=freq_bins)

        self.power_compress_1 = PowerCompress()
        self.power_compress_2 = PowerCompress()
        self.hypot_fun = HypotFun()

        self.conv_in = nn.Conv2d(1, channels[0], (3, 3), stride=(1, 2), padding=(1, 1), groups=1)

        self.enc_seqs, self.freq_downs, self.neck_seqs = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        self.freq_downs.append(FrequencyDownSampling(channels[0], channels[0], kernel_size=(2, 4), stride=(2, 4)))
        self.enc_seqs.append(RNN2D(channels[0]))
        self.freq_downs.append(FrequencyDownSampling(channels[0], channels[1], kernel_size=(2, 5), stride=(2, 5)))
        self.enc_seqs.append(RNN2D(channels[1]))
        self.freq_downs.append(FrequencyDownSampling(channels[1], channels[2], kernel_size=(1, 6), stride=(1, 6)))
        self.neck_seqs.append(RNN2D(channels[-1]))
        self.neck_seqs.append(RNN2D(channels[-1]))

        self.fc0 = nn.Linear(channels[-1], output_dim)

    def forward(self, ipt):
        iptc = self.trans(ipt)                                    # (B C T F 2)
        iptc = self.power_compress_1(iptc[:, :, :, 1:, :])
        iptc = iptc.permute(0, 3, 2, 1, 4).contiguous().flatten(-2)
        iptc = self.pre_bn(iptc)
        iptc = torch.clamp(iptc, -4, 4)

        b, f, t, cr = iptc.shape
        c = cr // 2
        iptc = iptc.view(b, f, t, c, 2).permute(0, 3, 2, 1, 4).contiguous()
        mag = self.hypot_fun(iptc[..., 0], iptc[..., 1])
        mag = torch.clamp(mag, 0, 4)

        opt = self.fft2band(mag)
        opt = torch.clamp(opt, 0, 4)
        opt = self.power_compress_2(opt)
        opt = torch.clamp(opt, 0, 2)

        opt = self.conv_in(opt)
        opt = self.freq_downs[0](opt)
        for i in range(len(self.enc_seqs)):
            opt = self.enc_seqs[i](opt)
            opt = self.freq_downs[i + 1](opt)
        for layer in self.neck_seqs:
            opt = layer(opt)

        opt = opt.squeeze(-1).transpose(-1, -2)
        opt = self.fc0(opt)
        return torch.mean(opt, dim=1)


# ============================================================================
# 通用工具
# ============================================================================
def evaluate(model, loader, device) -> float:
    """返回 top-1 精度（[0, 1]）。"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            out = model(inputs)
            if hasattr(out, "to_float"):
                out = out.to_float()
            preds = out.max(1).indices
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
    return correct / total


# 三档 QDQ 主路径：用作 quick_start 的 PTQ/QAT 多模式精度对比（设计 §10.1 / 验收口径）。
_THREE_MODES = (
    ExecutionMode.FP32_QDQ,
    ExecutionMode.FP16_QDQ,
    ExecutionMode.FIXED_SCALE_QDQ,
)


def evaluate_three_modes(
    model,
    loader,
    device,
    *,
    title: str,
    pp_threshold: float = 3.0,
) -> dict[str, float | str]:
    """对 fp32_qdq / fp16_qdq / fixed_scale_qdq 三档逐一评估；fp32_qdq 作为 baseline。

    - try/except 包住每个模式：fp16/fixed_scale 任一挂掉只会被记成 ``"FAILED: ..."``，
      不影响 PTQ/QAT 主流程与其它模式的评估。
    - 自动相对 fp32_qdq 计算 ΔTop1，并按 ``< pp_threshold pp`` 阈值打 ✅/⚠️。
    """

    print("\n" + "-" * 70)
    print(f"{title}：3 模式精度对比（验收：与 fp32_qdq 误差 < {pp_threshold:.0f} pp）")
    print("-" * 70)

    results: dict[str, float | str] = {}
    for mode in _THREE_MODES:
        try:
            with quant_execution_mode(mode):
                acc = evaluate(model, loader, device)
            results[mode.value] = acc
            print(f"  {mode.value:22s} {acc * 100:7.2f}%")
        except Exception as exc:  # noqa: BLE001
            results[mode.value] = f"FAILED: {exc}"
            print(f"  {mode.value:22s} FAILED — {exc}")

    fp_ref = results.get(ExecutionMode.FP32_QDQ.value)
    if isinstance(fp_ref, float):
        for mode in _THREE_MODES:
            label = mode.value
            if label == ExecutionMode.FP32_QDQ.value:
                continue
            acc = results.get(label)
            if isinstance(acc, float):
                delta_pp = (acc - fp_ref) * 100
                marker = "✅" if abs(delta_pp) < pp_threshold else "⚠️"
                print(f"  Δ({label} − fp32_qdq) = {delta_pp:+.2f} pp {marker}")
    return results


@contextlib.contextmanager
def stage(name: str, timings: dict | None = None):
    """打印阶段标题 + 计时；不会折叠任何业务调用，保证 with 块里的代码原样可见。"""
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)
    t0 = time.time()
    try:
        yield
    finally:
        dur = time.time() - t0
        if timings is not None:
            timings[name] = dur
        if dur < 60:
            print(f"⏱️  耗时: {dur:.2f} 秒")
        else:
            print(f"⏱️  耗时: {dur:.2f} 秒 ({dur / 60:.2f} 分钟)")


# ============================================================================
# 训练循环（标准 PyTorch 训练，与 AIMET 解耦）
# ============================================================================
def train_floating_point(model, train_loader, test_loader, device,
                         epochs: int = FP_EPOCHS, lr: float = FP_LR,
                         save_path=FP_MODEL_PATH,
                         val_loader=None,
                         lr_min: float = FP_LR_MIN) -> float:
    """训练浮点模型；按 val Top-1 保存最优权重，返回 test Top-1。"""
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=max(epochs, 1), eta_min=lr_min,
    )
    val_loader = val_loader or test_loader
    best_val, best_state = -1.0, None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, batch_idx = 0.0, 0
        pbar = tqdm(train_loader, desc=f"FP Epoch {epoch}/{epochs}")
        for batch_idx, (x, y) in enumerate(pbar, 1):
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optim.step()
            running_loss += loss.item()
            pbar.set_postfix(
                loss=f"{running_loss / batch_idx:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )
        scheduler.step()

        val_acc = evaluate(model, val_loader, device)
        print(
            f"[FP] Epoch {epoch}/{epochs} - Loss: {running_loss / max(batch_idx, 1):.4f}, "
            f"Val: {val_acc * 100:.2f}%, LR: {scheduler.get_last_lr()[0]:.2e}"
        )
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    torch.save(model.state_dict(), save_path)
    test_acc = evaluate(model, test_loader, device)
    print(f"[FP] Best val {best_val * 100:.2f}% → saved {save_path}, test {test_acc * 100:.2f}%")
    return test_acc


def qat_finetune(
    sim,
    train_loader,
    device,
    *,
    epochs: int = QAT_EPOCHS,
    lr: float = QAT_LR,
    val_loader=None,
    max_batches_per_epoch: int | None = None,
    restore_best: bool = True,
) -> dict[str, float]:
    """
    QAT 训练循环（标准 PyTorch 训练循环 + AIMET ``set_train_mode_freeze_bn``）。
    调用前应已经通过 ``freeze_quantizer_parameters`` 冻结 quantizer / BN affine。
    """
    from aimet_torch.fixed_point import ExecutionMode, quant_execution_mode

    optim = torch.optim.Adam(
        [p for p in sim.model.parameters() if p.requires_grad], lr=lr,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    def _snapshot_state() -> dict:
        snap: dict = {}
        for k, v in sim.model.state_dict().items():
            if isinstance(v, torch.Tensor):
                snap[k] = v.detach().cpu().clone()
            else:
                snap[k] = v
        return snap

    best_val = -1.0
    best_state: dict | None = None
    stats: dict[str, float] = {}

    if val_loader is not None and restore_best:
        with quant_execution_mode(ExecutionMode.FP32_QDQ):
            best_val = evaluate(sim.model, val_loader, device)
        best_state = _snapshot_state()
        stats["val_before_qat"] = best_val

    for epoch in range(1, epochs + 1):
        set_train_mode_freeze_bn(sim.model)

        running_loss, valid = 0.0, 0
        pbar = tqdm(train_loader, desc=f"QAT Epoch {epoch}/{epochs}")
        with quant_execution_mode(ExecutionMode.FP32_QDQ):
            for batch_idx, (x, y) in enumerate(pbar):
                if max_batches_per_epoch is not None and batch_idx >= max_batches_per_epoch:
                    break
                x, y = x.to(device), y.to(device)
                optim.zero_grad()
                loss = loss_fn(sim.model(x), y)
                loss.backward()
                optim.step()
                running_loss += loss.item()
                valid += 1
                pbar.set_postfix(
                    loss=f"{running_loss / valid:.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )
        epoch_loss = running_loss / max(valid, 1)
        stats[f"train_loss_epoch_{epoch}"] = epoch_loss
        print(
            f"[QAT] Epoch {epoch}/{epochs} - Loss: {epoch_loss:.4f}, "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        if val_loader is not None:
            with quant_execution_mode(ExecutionMode.FP32_QDQ):
                val_acc = evaluate(sim.model, val_loader, device)
            stats[f"val_epoch_{epoch}"] = val_acc
            print(f"[QAT] Epoch {epoch}/{epochs} - Val Top-1: {val_acc * 100:.2f}%")
            if restore_best and val_acc >= best_val:
                best_val = val_acc
                best_state = _snapshot_state()

        scheduler.step()

    if restore_best and best_state is not None:
        sim.model.load_state_dict(best_state, strict=False)
        stats["val_restored"] = best_val
        print(f"[QAT] Restored best checkpoint (val Top-1: {best_val * 100:.2f}%)")

    return stats


# ============================================================================
# 总结打印
# ============================================================================
def print_accuracy_summary(fp_acc, ptq_acc, po2_acc, qat_acc, reload_acc):
    print("\n" + "=" * 70)
    print("精度汇总")
    print("=" * 70)
    print(f"  浮点模型精度:           {fp_acc * 100:.2f}%")
    print(f"  PTQ 量化精度:           {ptq_acc * 100:.2f}%")
    print(f"  Power-of-2 量化精度:    {po2_acc * 100:.2f}%")
    print(f"  QAT 微调后精度:         {qat_acc * 100:.2f}%")
    print(f"  重新加载后精度:         {reload_acc * 100:.2f}%  (差距 {abs(qat_acc - reload_acc) * 100:.3f}%)")
    print("=" * 70)


def print_stage_timings(timings: dict):
    print("\n" + "=" * 70)
    print("耗时汇总")
    print("=" * 70)
    total = sum(timings.values())
    for name, dur in timings.items():
        pct = (dur / total * 100) if total > 0 else 0
        if dur < 60:
            print(f"  {name:35s} {dur:8.2f} 秒 ({pct:5.1f}%)")
        else:
            print(f"  {name:35s} {dur:8.2f} 秒 ({dur / 60:6.2f} 分钟, {pct:5.1f}%)")
    print("-" * 70)
    if total < 3600:
        print(f"  {'总计':35s} {total:8.2f} 秒 ({total / 60:6.2f} 分钟)")
    else:
        print(f"  {'总计':35s} {total:8.2f} 秒 ({total / 3600:.2f} 小时)")
    print("=" * 70)


# ============================================================================
# 主流程：标准 AIMET 量化 + 加载验证
# ============================================================================
def main():
    set_seed(SEED)
    setup_audio_backend()
    timings: dict = {}

    print("=" * 70)
    print("AIMET 量化演示 — 标准用法（QuantGRU + 混合精度位宽 JSON）")
    print("=" * 70)
    print(f"使用设备: {DEVICE}")

    # ------------------------------------------------------------------
    # 步骤 1: 准备数据 + 浮点模型
    # ------------------------------------------------------------------
    with stage("步骤 1: 加载数据和模型", timings):
        loaders = build_dataloaders(DATA_ROOT)
        model = MRNN(output_dim=NUM_CLASSES).to(DEVICE)

    # ------------------------------------------------------------------
    # 步骤 2: 浮点训练 + 评估
    # ------------------------------------------------------------------
    with stage("步骤 2: 浮点训练 + 评估", timings):
        fp_accuracy = train_floating_point(
            model, loaders["train"], loaders["test"], DEVICE, val_loader=loaders["val"],
        )
        print(f"浮点精度: {fp_accuracy * 100:.2f}%")

    # ------------------------------------------------------------------
    # 步骤 3: prepare_model + 创建 QuantizationSimModel
    #   - prepare_model 把 forward 中的 functional 调用 / 复用算子重构成
    #     可被 sim 抓住的 nn.Module 节点；QuantGRU、nn.BatchNorm2d 当 leaf
    #     处理（不进入它们的内部 trace）。
    #   - QuantizationSimModel 给模型注入 input/output/param quantizer。
    #   - apply_mixed_precision_bitwidth 用 JSON 配置覆盖默认位宽（如对
    #     bias 用 16bit、把 FloorDivide/Pad 整体 disable 等）。
    # ------------------------------------------------------------------
    with stage("步骤 3: prepare_model + 创建 sim", timings):
        prepared_model = model_preparer.prepare_model(
            model,
            stateless_modules_to_preserve=[PowerCompress, HypotFun, CLN, QuantizableBatchNorm2d],
        )
        sample_input, _ = next(iter(loaders["train"]))
        dummy_input = sample_input.to(DEVICE)

        sim = quantsim.QuantizationSimModel(
            prepared_model,
            dummy_input=dummy_input,
            quant_scheme=QUANT_SCHEME,
            config_file=str(CONFIG_FILE),
            default_output_bw=DEFAULT_BW,
            default_param_bw=DEFAULT_BW,
        )
        sim.set_percentile_value(PERCENTILE_VALUE)
        apply_mixed_precision_bitwidth(
            sim.model, config_file=str(BITWIDTH_CONFIG_FILE), verbose=True,
        )
        if MRNN_CLZ_ENCODING_FIX:
            from common.mrnn_clz_encoding import apply_mrnn_clz_encoding_fixes

            apply_mrnn_clz_encoding_fixes(sim.model, verbose=True)
            print("MRNN CLZ encoding fix: sign input bypass（校准前）")

    # ------------------------------------------------------------------
    # 步骤 4: 校准 (PTQ)
    #   compute_encodings 是 AIMET 的核心 API：进入上下文时让 quantizer 进
    #   入"观察模式"，跑一批数据后退出时根据观察值推算 scale/zero-point。
    #   QuantGRU 的内部 calibrating 标志也会被同时切到 True / False。
    # ------------------------------------------------------------------
    with stage("步骤 4: 校准 (PTQ)", timings):
        sim.model.to(DEVICE).eval()
        with torch.no_grad(), aimet.nn.compute_encodings(sim.model):
            for idx, (x, _) in enumerate(tqdm(loaders["calib"], desc="校准", total=MAX_CALIB_BATCHES)):
                if idx >= MAX_CALIB_BATCHES:
                    break
                sim.model(x.to(DEVICE))

        ptq_accuracy = evaluate(sim.model, loaders["test"], DEVICE)
        print(f"PTQ 量化精度: {ptq_accuracy * 100:.2f}%")
        # 三档主路径精度对比（fp32_qdq / fp16_qdq / fixed_scale_qdq）。
        # 校准始终在 fp32_qdq 完成（设计 §4.2）；这里只切换 evaluate 上下文。
        evaluate_three_modes(sim.model, loaders["test"], DEVICE, title="PTQ 后")

    power2_float_fmax = None
    if MRNN_CLZ_ENCODING_FIX:
        from common.mrnn_clz_encoding import collect_power2_float_out_fmax

        power2_float_fmax = collect_power2_float_out_fmax(
            prepared_model, loaders["calib"], DEVICE, MAX_CALIB_BATCHES,
        )

    # ------------------------------------------------------------------
    # 步骤 5: Power-of-2 量化（NPU 友好的 scale 对齐）
    #   把所有 scale 调整到 2^n，便于在定点硬件上用移位实现 dequant。
    #   align_bias_scale=True 把 Conv bias 的 scale 对齐到 Sx*Sw。
    # ------------------------------------------------------------------
    with stage("步骤 5: Power-of-2 量化", timings):
        apply_power_of_2_workflow(
            sim.model,
            method="round",
            tolerance=0.02,
            align_bias_scale=True,
            verbose=True,
        )
        if MRNN_CLZ_ENCODING_FIX:
            from common.mrnn_clz_encoding import apply_mrnn_clz_encoding_fixes_post_calib

            stats = apply_mrnn_clz_encoding_fixes_post_calib(
                sim.model,
                power2_float_out_fmax=power2_float_fmax,
                verbose=True,
            )
            print(f"MRNN CLZ encoding fix（Po2 后）: {stats}")
        po2_accuracy = evaluate(sim.model, loaders["test"], DEVICE)
        print(f"Power-of-2 量化精度: {po2_accuracy * 100:.2f}%")

    # ------------------------------------------------------------------
    # 步骤 6: QAT 微调
    #   - freeze_quantizer_parameters: 冻结 quantizer 的 min/max 与 BN
    #     的 running mean/var、affine 参数；只让模型权重和 bias 参与
    #     反向传播。
    #   - 训练循环本身就是普通 PyTorch（见 qat_finetune），唯一变化
    #     是用 set_train_mode_freeze_bn 替代普通 model.train()。
    # ------------------------------------------------------------------
    with stage("步骤 6: QAT 微调", timings):
        freeze_quantizer_parameters(sim.model, verbose=True, freeze_bn_affine=True)
        qat_stats = qat_finetune(sim, loaders["train"], DEVICE, val_loader=loaders["val"])
        if qat_stats.get("val_restored") is not None:
            print(f"QAT val checkpoint: {qat_stats['val_restored'] * 100:.2f}%")
        qat_accuracy = evaluate(sim.model, loaders["test"], DEVICE)
        print(f"QAT 微调后精度: {qat_accuracy * 100:.2f}%")
        # QAT 训练在 fp32_qdq 下完成（设计 §4.2）；以下三档对比 = 训练后切 mode 评估。
        # 用法：fp32_qdq 主考核；fixed_scale_qdq 校准 (M,r) 自动 derive；fp16_qdq experimental。
        evaluate_three_modes(sim.model, loaders["test"], DEVICE, title="QAT 后")

    # ------------------------------------------------------------------
    # 步骤 7: 保存量化产物
    #   两类产物：
    #     1) 训练态权重    -> torch.save(sim.model.state_dict(), *.pth)
    #                        含 quantizer 的 min/max 等参数，下次重建 sim 后
    #                        load_state_dict(strict=False) 即可恢复
    #     2) 部署产物      -> export_onnx_json(sim, ...)
    #                        本仓库定制的 ONNX 导出工具，相比 sim.export 增加了
    #                        QuantGRU / QuantizableBatchNorm2d 等自定义算子的
    #                        符号化处理，一次性产出：
    #                          * <prefix>.onnx              部署用 ONNX 模型
    #                          * <prefix>.encodings         ONNX node 名格式
    #                          * <prefix>_torch.encodings   PyTorch 模块名格式
    #                                                       （配合 load_quantizer_encodings）
    #                        ONNX Runtime / QNN / NPU 工具链直接消费
    #
    # 该函数要求 sim.model 在 CPU，先搬回 CPU 再导出。
    # ------------------------------------------------------------------
    with stage("步骤 7: 保存量化产物", timings):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        qat_state_pth = OUTPUT_DIR / "mrnn_kws_qat.pth"
        torch.save({"model": sim.model.state_dict()}, qat_state_pth)

        sim.model.cpu().eval()
        onnx_path, enc_path, int16_path = export_onnx_json(
            sim,
            export_dir=OUTPUT_DIR,
            filename_prefix="mrnn_kws",
            dummy_input_shape=(1, 16000, 1),
            opset=18,
        )
        sim.model.to(DEVICE)
        print(f"训练态权重: {qat_state_pth}")
        print(f"ONNX:       {onnx_path}")
        print(f"encodings:  {enc_path}")
        if int16_path:
            print(f"INT16:      {int16_path}")

    # ------------------------------------------------------------------
    # 步骤 8: 重新加载并验证（生产侧标准复现流程）
    #   1) 重建 prepared_model + 重建 QuantizationSimModel（参数与训练时一致）
    #   2) load_state_dict(strict=False) 加载权重
    #   3) load_quantizer_encodings 加载量化参数
    #   4) 评估精度，与训练侧 QAT 精度做闭环对比
    # ------------------------------------------------------------------
    with stage("步骤 8: 重新加载并验证", timings):
        # (1) 重建模型 + sim
        fresh_model = MRNN(output_dim=NUM_CLASSES).to(DEVICE).eval()
        fresh_prepared = model_preparer.prepare_model(
            fresh_model,
            stateless_modules_to_preserve=[PowerCompress, HypotFun, CLN, QuantizableBatchNorm2d],
        )
        fresh_sim = quantsim.QuantizationSimModel(
            fresh_prepared,
            dummy_input=dummy_input,
            quant_scheme=QUANT_SCHEME,
            config_file=str(CONFIG_FILE),
            default_output_bw=DEFAULT_BW,
            default_param_bw=DEFAULT_BW,
        )
        fresh_sim.set_percentile_value(PERCENTILE_VALUE)
        apply_mixed_precision_bitwidth(
            fresh_sim.model, config_file=str(BITWIDTH_CONFIG_FILE), verbose=True,
        )

        # (2) 加载权重
        ckpt = torch.load(qat_state_pth, map_location=DEVICE, weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        fresh_sim.model.load_state_dict(state, strict=False)

        # (3) 加载量化参数
        load_quantizer_encodings(
            fresh_sim.model,
            load_path=str(enc_path),
            verbose=True,
        )

        # (4) 精度对比
        reload_accuracy = evaluate(fresh_sim.model, loaders["test"], DEVICE)
        gap = abs(qat_accuracy - reload_accuracy) * 100
        print(f"QAT 保存前精度:    {qat_accuracy * 100:.2f}%")
        print(f"重新加载后精度:    {reload_accuracy * 100:.2f}%")
        if gap < 0.5:
            print(f"✅ 精度差异 {gap:.3f}%，加载验证通过")
        else:
            print(f"⚠️  精度差异 {gap:.3f}% 偏大，请检查 sim 配置 / 权重 / encodings 一致性")

    # ------------------------------------------------------------------
    # 总结
    # ------------------------------------------------------------------
    print_accuracy_summary(fp_accuracy, ptq_accuracy, po2_accuracy, qat_accuracy, reload_accuracy)
    print_stage_timings(timings)
    print("\n量化流程完成（已通过加载验证）\n")


if __name__ == "__main__":
    main()

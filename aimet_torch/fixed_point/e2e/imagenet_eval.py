# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
"""ImageNet validation helpers for full MobileNet V2 fixed-point e2e checks."""

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
from zipfile import ZipFile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, IterableDataset, Subset

from aimet_torch.fixed_point import ExecutionMode, quant_execution_mode
from aimet_torch.fixed_point.e2e.mobilenet_v2 import (
    MobileNetV2SimBundle,
    build_calibrated_sim,
    build_prepared_mobilenet_v2,
)

IMAGENET_VAL_ENV = "AIMET_RX_IMAGENET_VAL"
IMAGENET_VAL_ZIP_ENV = "AIMET_RX_IMAGENET_VAL_ZIP"
IMAGENET_HF_DATASET_ENV = "AIMET_RX_IMAGENET_HF_DATASET"
HF_IMAGENET_DEFAULT = "ILSVRC/imagenet-1k"
IMAGENET_INPUT_SIZE = 224
IMAGENET_NUM_CLASSES = 1000

# Design v2 §10.1 (classification top-1 drop vs fp32_qdq reference).
IMAGENET_INT16_MAX_TOP1_DROP = 0.01
IMAGENET_FIXED_SCALE_MAX_TOP1_DROP = 0.001
IMAGENET_FP16_MAX_TOP1_DROP = 0.005
IMAGENET_INT16_MIN_COSINE = 0.99
# Spec 13 §108 actually defines an *op-level* gate: ``int16_fixed_eval`` vs
# ``fp32_qdq`` cosine >= 0.999 on a single op. The proof that this is
# *achievable* lives in :mod:`tests.fixed_point.kernels.test_conv2d_per_channel_diag`.
# At the *network* level, multi-layer accumulation pushes the realistic floor
# below 0.999 — top-1 drop (design v2 §9) is the production gate, not cosine.
#
# We still surface ``int16_fixed_eval`` vs ``fixed_scale_qdq`` cosine as a
# **diagnostic** metric (boundary-grid alignment health), with a relaxed floor
# matching the other modes — failing this only means the two ride different
# boundary grids, not that the integer kernel is wrong.
IMAGENET_INT16_VS_FIXED_SCALE_MIN_COSINE = 0.99
IMAGENET_LIGHT_DEFAULT_SAMPLES = 64
_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


def resolve_imagenet_val_dir() -> Optional[Path]:
    """Return ImageNet ``val/`` root if configured or found under common paths."""

    candidates: List[Path] = []
    env = os.environ.get(IMAGENET_VAL_ENV, "").strip()
    if env:
        candidates.append(Path(env).expanduser())
    home = Path.home()
    candidates.extend(
        [
            home / "datasets" / "imagenet" / "val",
            home / "data" / "imagenet" / "val",
            Path("/data/imagenet/val"),
            Path("/datasets/imagenet/val"),
            Path.home() / "workspace" / "llama.cpp" / "datasets" / "imagenet1K" / "imagenet_val",
        ]
    )
    for path in candidates:
        if path.is_dir() and any(path.iterdir()):
            return path
    return None


def resolve_imagenet_val_zip() -> Optional[Path]:
    """Return a readable ImageNet val zip if configured or found under common paths."""

    candidates: List[Path] = []
    env_zip = os.environ.get(IMAGENET_VAL_ZIP_ENV, "").strip()
    if env_zip:
        candidates.append(Path(env_zip).expanduser())
    env_val = os.environ.get(IMAGENET_VAL_ENV, "").strip()
    if env_val:
        env_path = Path(env_val).expanduser()
        if env_path.suffix.lower() == ".zip":
            candidates.append(env_path)

    home = Path.home()
    candidates.extend(
        [
            home / "datasets" / "imagenet" / "imagenet_val.zip",
            home / "data" / "imagenet" / "imagenet_val.zip",
            Path("/data/imagenet/imagenet_val.zip"),
            Path("/datasets/imagenet/imagenet_val.zip"),
            Path("/mnt/data8t/share/datasets/imagenet1K/imagenet_val.zip"),
        ]
    )
    for path in candidates:
        if path.is_file() and os.access(path, os.R_OK):
            return path
    return None


def imagenet_transform(input_size: int = IMAGENET_INPUT_SIZE):
    """Standard ImageNet eval transforms (Resize → CenterCrop → normalize)."""

    from torchvision import transforms

    resize = int(round(input_size / 0.875))
    return transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def hf_imagenet_access_help(dataset_name: str = HF_IMAGENET_DEFAULT) -> str:
    return (
        f"Failed to load Hugging Face dataset {dataset_name!r}.\n"
        "ImageNet is gated (research / non-commercial). Typical setup:\n"
        f"  1. Open https://huggingface.co/datasets/{dataset_name} and click \"Access repository\"\n"
        "  2. huggingface-cli login   (or export HF_TOKEN=...)\n"
        "  3. pip install datasets Pillow\n"
        "Then re-run with --source huggingface (or --source auto)."
    )


def _resolve_hf_dataset_name(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit.strip()
    env = os.environ.get(IMAGENET_HF_DATASET_ENV, "").strip()
    return env or HF_IMAGENET_DEFAULT


class _HFImageNetValMapDataset(Dataset):
    """Map-style wrapper over a materialized HF ``validation`` split."""

    def __init__(self, hf_dataset: Any, transform: Any) -> None:
        self._ds = hf_dataset
        self._transform = transform

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        row = self._ds[int(index)]
        image = row["image"]
        label = int(row["label"])
        return self._transform(image), label


class _HFImageNetValIterable(IterableDataset):
    """Stream HF validation images without downloading the full 50k set."""

    def __init__(
        self,
        dataset_name: str,
        transform: Any,
        *,
        max_samples: Optional[int] = None,
    ) -> None:
        self._dataset_name = dataset_name
        self._transform = transform
        self._max_samples = max_samples

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, int]]:
        from datasets import load_dataset

        stream = load_dataset(self._dataset_name, split="validation", streaming=True)
        count = 0
        for row in stream:
            if self._max_samples is not None and count >= self._max_samples:
                break
            yield self._transform(row["image"]), int(row["label"])
            count += 1


def load_hf_imagenet_validation(
    *,
    dataset_name: Optional[str] = None,
    max_samples: Optional[int] = 512,
    streaming: bool = False,
) -> Any:
    """Load ImageNet-1k validation via Hugging Face ``datasets`` (API / cache).

    Default ``ILSVRC/imagenet-1k`` (50k val images). Use ``max_samples`` to cap
    downloads; ``streaming=True`` avoids materializing the full split locally.
    """

    name = _resolve_hf_dataset_name(dataset_name)
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Hugging Face ImageNet requires: pip install datasets Pillow"
        ) from exc

    try:
        if streaming:
            return load_dataset(name, split="validation", streaming=True)
        split = "validation" if max_samples is None else f"validation[:{int(max_samples)}]"
        return load_dataset(name, split=split, streaming=False)
    except Exception as exc:
        raise RuntimeError(hf_imagenet_access_help(name)) from exc


def discover_image_paths(
    root: Path,
    *,
    max_samples: Optional[int] = None,
) -> List[Path]:
    """Collect image files under ``root`` (recursive); no class folders required."""

    root = Path(root)
    paths: List[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS:
            paths.append(path)
            if max_samples is not None and len(paths) >= max_samples:
                break
    return paths


class _UnlabeledImagesDataset(Dataset):
    """Load images from paths; label is a dummy ``0`` (unused for cosine-only checks)."""

    def __init__(self, image_paths: List[Path], transform: Any) -> None:
        self._paths = image_paths
        self._transform = transform

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        from PIL import Image

        with Image.open(self._paths[index]) as img:
            tensor = self._transform(img.convert("RGB"))
        return tensor, 0


class _ZipImageFolderDataset(Dataset):
    """ImageFolder-style dataset backed by a zip archive."""

    def __init__(
        self,
        zip_path: Path,
        transform: Any,
        *,
        max_samples: Optional[int] = None,
    ) -> None:
        self._zip_path = Path(zip_path)
        self._transform = transform
        with ZipFile(self._zip_path) as zf:
            image_names = [
                name
                for name in zf.namelist()
                if not name.endswith("/")
                and Path(name).suffix.lower() in _IMAGE_EXTENSIONS
                and len(Path(name).parts) >= 3
            ]
        if not image_names:
            raise FileNotFoundError(f"No ImageFolder-style images found in {zip_path}")

        classes = sorted({Path(name).parts[-2] for name in image_names})
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        samples = [
            (name, class_to_idx[Path(name).parts[-2]])
            for name in sorted(image_names)
        ]
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples[:max_samples] if max_samples is not None else samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        from PIL import Image

        name, label = self.samples[int(index)]
        with ZipFile(self._zip_path) as zf:
            with zf.open(name) as handle:
                data = handle.read()
        with Image.open(BytesIO(data)) as img:
            return self._transform(img.convert("RGB")), int(label)


class _SyntheticImageNetDataset(Dataset):
    """Deterministic ``[0,1]`` RGB + ImageNet normalize (no labels, no download)."""

    def __init__(
        self,
        num_samples: int,
        input_size: int = IMAGENET_INPUT_SIZE,
        *,
        seed: int = 0,
    ) -> None:
        self._num = int(num_samples)
        self._input_size = int(input_size)
        self._seed = int(seed)
        from torchvision import transforms

        self._normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __len__(self) -> int:
        return self._num

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        gen = torch.Generator().manual_seed(self._seed + int(index))
        rgb = torch.rand(3, self._input_size, self._input_size, generator=gen)
        return self._normalize(rgb), 0


def build_imagenet_val_loader_unlabeled(
    image_dir: Path,
    *,
    batch_size: int = 8,
    max_samples: Optional[int] = IMAGENET_LIGHT_DEFAULT_SAMPLES,
    input_size: int = IMAGENET_INPUT_SIZE,
    num_workers: int = 0,
) -> DataLoader:
    """DataLoader from a flat or nested folder of images (labels not used)."""

    paths = discover_image_paths(image_dir, max_samples=max_samples)
    if not paths:
        raise FileNotFoundError(f"No images found under {image_dir}")
    dataset: Dataset = _UnlabeledImagesDataset(paths, imagenet_transform(input_size))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )


def build_imagenet_val_loader_zip(
    zip_path: Path,
    *,
    batch_size: int = 8,
    max_samples: Optional[int] = IMAGENET_LIGHT_DEFAULT_SAMPLES,
    input_size: int = IMAGENET_INPUT_SIZE,
    num_workers: int = 0,
) -> DataLoader:
    """ImageFolder-style DataLoader backed by ``imagenet_val.zip``."""

    dataset: Dataset = _ZipImageFolderDataset(
        Path(zip_path),
        imagenet_transform(input_size),
        max_samples=max_samples,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )


def build_imagenet_val_loader_synthetic(
    *,
    batch_size: int = 8,
    max_samples: int = IMAGENET_LIGHT_DEFAULT_SAMPLES,
    input_size: int = IMAGENET_INPUT_SIZE,
    seed: int = 0,
    num_workers: int = 0,
) -> DataLoader:
    """Synthetic calibrated inputs (ImageNet normalize, no download, no labels)."""

    dataset = _SyntheticImageNetDataset(max_samples, input_size, seed=seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )


def build_imagenet_val_loader_hf(
    *,
    dataset_name: Optional[str] = None,
    batch_size: int = 8,
    max_samples: Optional[int] = 512,
    input_size: int = IMAGENET_INPUT_SIZE,
    num_workers: int = 0,
    streaming: bool = False,
) -> DataLoader:
    """DataLoader over HF ImageNet validation (same transforms as ``ImageFolder`` path)."""

    name = _resolve_hf_dataset_name(dataset_name)
    transform = imagenet_transform(input_size)
    if streaming:
        dataset: Any = _HFImageNetValIterable(
            name,
            transform,
            max_samples=max_samples,
        )
    else:
        hf_ds = load_hf_imagenet_validation(
            dataset_name=name,
            max_samples=max_samples,
            streaming=False,
        )
        dataset = _HFImageNetValMapDataset(hf_ds, transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )


def try_load_torchvision_imagenet_weights(model: nn.Module) -> bool:
    """Best-effort copy of torchvision MobileNetV2 ImageNet weights into ``model``."""

    try:
        from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
    except ImportError:
        return False

    try:
        ref = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    except Exception:
        try:
            ref = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        except Exception:
            ref = mobilenet_v2(weights=None)

    ref_sd = ref.state_dict()
    dst_sd = model.state_dict()
    matched = {
        k: v
        for k, v in ref_sd.items()
        if k in dst_sd and dst_sd[k].shape == v.shape
    }
    if len(matched) < 100:
        return False
    dst_sd.update(matched)
    model.load_state_dict(dst_sd)
    return True


def build_imagenet_val_loader(
    val_root: Path,
    *,
    batch_size: int = 8,
    max_samples: Optional[int] = None,
    input_size: int = IMAGENET_INPUT_SIZE,
    num_workers: int = 0,
    sample_seed: Optional[int] = None,
) -> DataLoader:
    """``ImageFolder`` loader with standard ImageNet normalization.

    If ``sample_seed`` is set and ``max_samples`` caps the dataset, select a
    deterministic random subset instead of the first N lexicographic samples.
    """

    from torchvision import datasets

    transform = imagenet_transform(input_size)
    dataset = datasets.ImageFolder(str(val_root), transform=transform)
    if max_samples is not None and max_samples < len(dataset):
        if sample_seed is None:
            indices = list(range(max_samples))
        else:
            gen = torch.Generator().manual_seed(int(sample_seed))
            indices = torch.randperm(len(dataset), generator=gen)[: int(max_samples)].tolist()
        dataset = Subset(dataset, indices)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )


def iter_image_batches(
    loader: DataLoader,
    max_batches: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Iterator[torch.Tensor]:
    """Yield image tensors from a ``(images, labels)`` loader."""

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        if isinstance(batch, (list, tuple)):
            images = batch[0]
        else:
            images = batch
        if device is not None:
            images = images.to(device, non_blocking=True)
        yield images


def _move_batch_to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, tuple):
        return tuple(_move_batch_to_device(item, device) for item in batch)
    if isinstance(batch, list):
        return [_move_batch_to_device(item, device) for item in batch]
    if isinstance(batch, dict):
        return {key: _move_batch_to_device(value, device) for key, value in batch.items()}
    return batch


class _DeviceLoader:
    """Re-iterable loader wrapper that moves each batch to ``device``."""

    def __init__(self, loader: DataLoader, device: torch.device):
        self.loader = loader
        self.device = device

    def __iter__(self) -> Iterator[Any]:
        for batch in self.loader:
            yield _move_batch_to_device(batch, self.device)

    def __len__(self) -> int:
        return len(self.loader)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.loader, name)


from aimet_torch.fixed_point.metrics import (  # noqa: E402
    dequantize_logits as _dequantize_logits,  # backward-compat private alias
    mean_logits_cosine_on_loader as _mean_logits_cosine_on_loader,
)
from aimet_torch.fixed_point.metrics.classification import (  # noqa: E402
    top1_accuracy as _top1_accuracy_core,
    top1_drop as _top1_drop_core,
)
from aimet_torch.fixed_point.metrics.logits import (  # noqa: E402
    _resolve_model_and_device as _logits_resolve_model_and_device,
)


def _model_device(sim: Any) -> torch.device:
    """Backward-compat shim — delegates to the generic resolver."""

    _, device = _logits_resolve_model_and_device(sim, None)
    return device


# Thin wrappers preserve the historical ``(sim, loader, mode, ...)`` signature
# and the JSON report keys produced by ``cosine_vs_fp32_on_loader``. The heavy
# lifting now lives in :mod:`aimet_torch.fixed_point.metrics.{logits,classification}`
# so non-ImageNet models can reuse those APIs directly.


def top1_accuracy(
    sim: Any,
    loader: DataLoader,
    mode: ExecutionMode,
    *,
    max_batches: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> float:
    """Top-1 accuracy on ``loader`` under ``mode`` — see :func:`metrics.top1_accuracy`."""

    return _top1_accuracy_core(
        sim, loader, mode, max_batches=max_batches, device=device
    )


def top1_drop(reference_acc: float, candidate_acc: float) -> float:
    """Backward-compat re-export of :func:`metrics.top1_drop`."""

    return _top1_drop_core(reference_acc, candidate_acc)


def logits_cosine_on_loader(
    sim: Any,
    loader: DataLoader,
    candidate_mode: ExecutionMode,
    *,
    max_batches: int = 2,
    device: Optional[torch.device] = None,
) -> float:
    """Mean batch cosine: ``candidate_mode`` vs ``FP32_QDQ`` — :func:`metrics.mean_logits_cosine_on_loader`."""

    return _mean_logits_cosine_on_loader(
        sim,
        loader,
        ref_mode=ExecutionMode.FP32_QDQ,
        cand_mode=candidate_mode,
        max_batches=max_batches,
        device=device,
    )


@torch.no_grad()
def per_layer_cosine_across_modes(
    sim: Any,
    loader: DataLoader,
    *,
    ref_mode: ExecutionMode = ExecutionMode.FP32_QDQ,
    cand_mode: ExecutionMode = ExecutionMode.INT16_FIXED_EVAL,
    max_batches: int = 1,
    top_k: int = 15,
    device: Optional[torch.device] = None,
) -> List[Dict[str, Any]]:
    """Thin loader-aware wrapper around :func:`per_layer_chained_cosine`.

    Pulls the first ``max_batches`` images from ``loader`` (only the first
    one is used — chained metrics only need a single batch) and forwards
    them to the model-agnostic core. The heavy lifting lives in
    :mod:`aimet_torch.fixed_point.metrics.chained`, so non-ImageNet models
    can reuse the same diagnostic with their own ``(model, inputs)`` pair.
    """

    from aimet_torch.fixed_point.metrics import per_layer_chained_cosine

    if device is None:
        device = _model_device(sim)

    images_batch: Optional[torch.Tensor] = None
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        images_batch = batch[0].to(device)
        break
    if images_batch is None:
        return []

    return per_layer_chained_cosine(
        sim.model,
        images_batch,
        ref_mode=ref_mode,
        cand_mode=cand_mode,
        top_k=top_k,
    )


@torch.no_grad()
def per_layer_isolated_cosine_on_loader(
    sim: Any,
    loader: DataLoader,
    *,
    cand_mode: ExecutionMode = ExecutionMode.FIXED_SCALE_QDQ,
    ref_mode: ExecutionMode = ExecutionMode.FP32_QDQ,
    max_batches: int = 1,
    top_k: int = 15,
    device: Optional[torch.device] = None,
) -> List[Dict[str, Any]]:
    """Thin loader-aware wrapper around :func:`per_layer_isolated_cosine`.

    Pulls the first ``max_batches`` images from ``loader`` and forwards them
    to the model-agnostic core. Keep the heavy lifting in
    :mod:`aimet_torch.fixed_point.metrics.isolated` so non-ImageNet models
    can reuse the diagnostic with their own inputs.
    """

    from aimet_torch.fixed_point.metrics import per_layer_isolated_cosine

    if device is None:
        device = _model_device(sim)

    images_batch: Optional[torch.Tensor] = None
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        images_batch = batch[0].to(device)
        break
    if images_batch is None:
        return []

    return per_layer_isolated_cosine(
        sim.model,
        images_batch,
        cand_mode=cand_mode,
        ref_mode=ref_mode,
        top_k=top_k,
    )


def per_layer_cosine_int16_vs_fixed_scale(
    sim: Any,
    loader: DataLoader,
    *,
    max_batches: int = 1,
    top_k: int = 15,
    device: Optional[torch.device] = None,
) -> List[Dict[str, Any]]:
    """Diagnostic alias for ``INT16_FIXED_EVAL`` vs ``FIXED_SCALE_QDQ``.

    Kept as a thin wrapper so existing CI scripts still work; new callers
    should prefer :func:`per_layer_cosine_across_modes` and explicitly pick
    ``ref_mode=FP32_QDQ`` (spec 13 §108) or ``ref_mode=FIXED_SCALE_QDQ``
    (boundary-grid diagnosis).
    """

    return per_layer_cosine_across_modes(
        sim,
        loader,
        ref_mode=ExecutionMode.FIXED_SCALE_QDQ,
        cand_mode=ExecutionMode.INT16_FIXED_EVAL,
        max_batches=max_batches,
        top_k=top_k,
        device=device,
    )


def logits_cosine_between_modes(
    sim: Any,
    loader: DataLoader,
    ref_mode: ExecutionMode,
    candidate_mode: ExecutionMode,
    *,
    max_batches: int = 2,
    device: Optional[torch.device] = None,
) -> float:
    """Mean batch cosine between two arbitrary modes — :func:`metrics.mean_logits_cosine_on_loader`.

    Used for spec 13 §108: directly compare ``int16_fixed_eval`` against
    ``fixed_scale_qdq`` (both share the fixed-scale grid) so any drop is
    attributable to the integer kernel / requant rounding rather than to
    the choice of quantization scales.
    """

    return _mean_logits_cosine_on_loader(
        sim,
        loader,
        ref_mode=ref_mode,
        cand_mode=candidate_mode,
        max_batches=max_batches,
        device=device,
    )


def cosine_vs_fp32_on_loader(
    sim: Any,
    loader: DataLoader,
    *,
    max_batches: int = 2,
    device: Optional[torch.device] = None,
) -> dict[str, float]:
    """Mean batch cosine vs ``FP32_QDQ`` for INT16 / fixed_scale / FP16.

    Backward-compat: keys remain ``"int16_fixed_eval" / "fixed_scale_qdq" /
    "fp16_qdq"`` (i.e. :attr:`ExecutionMode.value`) so existing JSON reports
    keep parsing. Use :func:`metrics.mean_logits_cosine_vs_fp32` for the
    generic sweep API.
    """

    from aimet_torch.fixed_point.metrics import mean_logits_cosine_vs_fp32

    return dict(
        mean_logits_cosine_vs_fp32(
            sim, loader, max_batches=max_batches, device=device
        )
    )


def build_imagenet_mobilenet_bundle(
    val_root: Optional[Path] = None,
    *,
    val_source: str = "local",
    image_dir: Optional[Path] = None,
    zip_path: Optional[Path] = None,
    hf_dataset_name: Optional[str] = None,
    hf_streaming: bool = False,
    synthetic_seed: int = 0,
    batch_size: int = 8,
    calib_max_batches: int = 4,
    calib_max_samples: Optional[int] = 256,
    eval_max_samples: Optional[int] = None,
    eval_seed: Optional[int] = None,
    load_pretrained: bool = True,
    variant: str = "torchvision",
    default_param_bw: int = 8,
    default_output_bw: int = 8,
    int16_eval_bw: int = 8,
    int16_eval_symmetric: bool = True,
    output_quantizer_overrides: Optional[Dict[str, Tuple[int, bool]]] = None,
    apply_bias_correction: bool = False,
    bias_correction_samples: int = 16,
    apply_adaround: bool = False,
    adaround_num_batches: int = 2,
    adaround_iterations: int = 80,
    adaround_export_dir: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> Tuple[MobileNetV2SimBundle, DataLoader]:
    """Prepare MobileNet V2 @ 224, calibrate on a val subset, return bundle + loader.

    ``val_source``:
      * ``local`` — ImageFolder (``val_root`` / ``AIMET_RX_IMAGENET_VAL``).
      * ``zip`` — ImageFolder-style zip (``zip_path`` / ``AIMET_RX_IMAGENET_VAL_ZIP``).
      * ``huggingface`` — HF ``datasets`` (labels present but optional for metrics).
      * ``unlabeled`` — any folder of images (``image_dir``); no synset layout.
      * ``synthetic`` — ``max_samples`` random normalized tensors (no download).

    For **quantization sim verification**, ``cosine_vs_fp32_on_loader`` only needs images;
    top-1 metrics need real labels (``local`` / ``zip`` / ``huggingface``).

    ``calib_max_samples`` caps how many images calibration touches.
    ``eval_max_samples`` (optional) caps the *returned* loader independently —
    when larger than ``calib_max_samples`` the loader yields up to
    ``eval_max_samples`` images and calibration still only consumes the first
    ``calib_max_batches`` of them (DataLoader iter restarts from 0 each pass).
    Use this to keep calibration cheap while running large top-1 / cosine eval
    without an extra calibration pass on the bigger set. ``None`` (default)
    preserves the legacy behavior: loader is capped at ``calib_max_samples``.

    Default ``variant='torchvision'`` uses ``torchvision.models.mobilenet_v2`` ImageNet weights
    (recommended for real val top-1). Use ``variant='full'`` for the in-repo MobileNetV2
    definition with optional ``try_load_torchvision_imagenet_weights``.
    """

    use_variant = "torchvision" if variant == "torchvision" else "full"
    model, dummy = build_prepared_mobilenet_v2(
        n_class=IMAGENET_NUM_CLASSES,
        input_size=IMAGENET_INPUT_SIZE,
        variant=use_variant,  # type: ignore[arg-type]
    )
    if load_pretrained and use_variant == "full":
        try_load_torchvision_imagenet_weights(model)
    if device is not None:
        model = model.to(device)
        dummy = dummy.to(device)

    # ``loader_max_samples`` is the cap on the *eval* loader returned to the
    # caller. Calibration always re-iterates the loader and only consumes the
    # first ``calib_max_batches`` batches, so a larger eval cap costs nothing
    # at calibration time (we just walk a longer dataset later).
    if eval_max_samples is not None:
        if calib_max_samples is None:
            loader_max_samples: Optional[int] = eval_max_samples
        else:
            loader_max_samples = max(int(calib_max_samples), int(eval_max_samples))
    else:
        loader_max_samples = calib_max_samples

    if val_source == "synthetic":
        n_samples = int(loader_max_samples or IMAGENET_LIGHT_DEFAULT_SAMPLES)
        loader = build_imagenet_val_loader_synthetic(
            batch_size=batch_size,
            max_samples=n_samples,
            input_size=IMAGENET_INPUT_SIZE,
            seed=synthetic_seed,
        )
    elif val_source == "unlabeled":
        root = image_dir or val_root
        if root is None:
            raise FileNotFoundError(
                "unlabeled source requires image_dir= or val_root= pointing at image files."
            )
        loader = build_imagenet_val_loader_unlabeled(
            Path(root),
            batch_size=batch_size,
            max_samples=loader_max_samples,
            input_size=IMAGENET_INPUT_SIZE,
        )
    elif val_source == "zip":
        archive = zip_path or val_root or resolve_imagenet_val_zip()
        if archive is None:
            raise FileNotFoundError(
                f"ImageNet val zip not found. Set {IMAGENET_VAL_ZIP_ENV} or "
                f"{IMAGENET_VAL_ENV} to imagenet_val.zip."
            )
        loader = build_imagenet_val_loader_zip(
            Path(archive),
            batch_size=batch_size,
            max_samples=loader_max_samples,
            input_size=IMAGENET_INPUT_SIZE,
        )
    elif val_source == "huggingface":
        loader = build_imagenet_val_loader_hf(
            dataset_name=hf_dataset_name,
            batch_size=batch_size,
            max_samples=loader_max_samples,
            input_size=IMAGENET_INPUT_SIZE,
            streaming=hf_streaming,
        )
    else:
        if val_root is None:
            resolved = resolve_imagenet_val_dir()
            if resolved is None:
                raise FileNotFoundError(
                    f"ImageNet val directory not found. Set {IMAGENET_VAL_ENV} or use "
                    "val_source='synthetic', 'unlabeled', or 'huggingface'."
                )
            val_root = resolved
        loader = build_imagenet_val_loader(
            val_root,
            batch_size=batch_size,
            max_samples=loader_max_samples,
            input_size=IMAGENET_INPUT_SIZE,
            sample_seed=eval_seed,
        )
    device_loader = _DeviceLoader(loader, device) if device is not None else loader
    calib_batches = list(
        iter_image_batches(loader, max_batches=calib_max_batches, device=device)
    )
    bundle = build_calibrated_sim(
        model,
        dummy,
        input_size=IMAGENET_INPUT_SIZE,
        variant=use_variant,  # type: ignore[arg-type]
        calibration_batches=calib_batches,
        default_param_bw=default_param_bw,
        default_output_bw=default_output_bw,
        int16_eval_bw=int16_eval_bw,
        int16_eval_symmetric=int16_eval_symmetric,
        output_quantizer_overrides=output_quantizer_overrides,
        apply_bias_correction=apply_bias_correction,
        bias_correction_loader=device_loader if apply_bias_correction else None,
        bias_correction_samples=bias_correction_samples,
        adaround_loader=device_loader if apply_adaround else None,
        adaround_num_batches=adaround_num_batches,
        adaround_iterations=adaround_iterations,
        adaround_export_dir=adaround_export_dir,
    )
    return bundle, loader

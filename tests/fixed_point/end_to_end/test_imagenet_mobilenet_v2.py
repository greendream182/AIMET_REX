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
"""ImageNet-style MobileNet V2 e2e (224x224, 1000 classes).

Requires a local ImageNet validation set for the full benchmark:

  export AIMET_RX_IMAGENET_VAL=/path/to/imagenet/val
  # or
  export AIMET_RX_IMAGENET_VAL_ZIP=/path/to/imagenet_val.zip

The directory must follow ``ImageFolder`` layout (synset subfolders).

Run:

  pytest tests/fixed_point/end_to_end/test_imagenet_mobilenet_v2.py -m imagenet -v

Without the env var, ImageNet-marked tests are skipped; a tiny synthetic val
tree is still exercised by ``test_imagenet_val_layout_smoke``.
"""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxscript")
pytest.importorskip("torchvision")

from aimet_torch.fixed_point import ExecutionMode  # noqa: E402
from aimet_torch.fixed_point.e2e.imagenet_eval import (  # noqa: E402
    IMAGENET_FIXED_SCALE_MAX_TOP1_DROP,
    IMAGENET_FP16_MAX_TOP1_DROP,
    IMAGENET_INT16_MAX_TOP1_DROP,
    IMAGENET_INT16_MIN_COSINE,
    build_imagenet_mobilenet_bundle,
    build_imagenet_val_loader,
    iter_image_batches,
    logits_cosine_on_loader,
    resolve_imagenet_val_dir,
    top1_accuracy,
    top1_drop,
)
from aimet_torch.fixed_point.e2e.mobilenet_v2 import int16_vs_fp32_cosine  # noqa: E402

EVAL_BATCHES = 16
CALIB_BATCHES = 4
CALIB_MAX_SAMPLES = 512


def _report_imagenet_metrics(case: str, **fields: float) -> None:
    """Print metrics on pass; use ``pytest -s`` or ``pytest -rP`` to see stdout."""

    parts = [f"[imagenet] {case}:"] + [f"{k}={v:.6g}" for k, v in fields.items()]
    print(" ".join(parts), flush=True)


def _make_tiny_imagenet_val(root: Path, *, n_classes: int = 4, per_class: int = 2) -> Path:
    """Minimal ImageFolder tree under ``root/val``."""

    val = root / "val"
    for class_idx in range(n_classes):
        class_dir = val / f"class_{class_idx:04d}"
        class_dir.mkdir(parents=True, exist_ok=True)
        for img_idx in range(per_class):
            img = torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8)
            from torchvision.io import write_png

            write_png(img, str(class_dir / f"img_{img_idx}.png"))
    return val


def _zip_imagenet_val(val_root: Path, zip_path: Path) -> Path:
    with ZipFile(zip_path, "w") as zf:
        for path in sorted(val_root.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(val_root.parent))
    return zip_path


@pytest.fixture(scope="module")
def imagenet_val_dir():
    root = resolve_imagenet_val_dir()
    if root is None:
        pytest.skip(
            "ImageNet val not found. Set AIMET_RX_IMAGENET_VAL to a directory "
            "with synset subfolders (ImageFolder layout)."
        )
    return root


@pytest.fixture(scope="module")
def imagenet_bundle(imagenet_val_dir):
    bundle, loader = build_imagenet_mobilenet_bundle(
        imagenet_val_dir,
        batch_size=8,
        calib_max_batches=CALIB_BATCHES,
        calib_max_samples=CALIB_MAX_SAMPLES,
        load_pretrained=True,
    )
    return {"bundle": bundle, "loader": loader}


def test_imagenet_val_layout_smoke(tmp_path):
    """Exercise ImageFolder + PTQ pipeline on a toy val tree (no real ImageNet)."""

    val_root = _make_tiny_imagenet_val(tmp_path, n_classes=4, per_class=2)
    bundle, loader = build_imagenet_mobilenet_bundle(
        val_root,
        batch_size=2,
        calib_max_batches=2,
        calib_max_samples=8,
        load_pretrained=False,
    )
    images = next(iter(iter_image_batches(loader, max_batches=1)))
    cos = int16_vs_fp32_cosine(bundle.sim, images)
    assert cos >= 0.95, f"toy val INT16 cosine={cos:.6f}"


def test_imagenet_val_zip_layout_smoke(tmp_path):
    """Exercise ImageFolder-style zip path without requiring extraction."""

    val_root = _make_tiny_imagenet_val(tmp_path, n_classes=4, per_class=2)
    zip_path = _zip_imagenet_val(val_root, tmp_path / "imagenet_val.zip")
    bundle, loader = build_imagenet_mobilenet_bundle(
        val_source="zip",
        zip_path=zip_path,
        batch_size=2,
        calib_max_batches=1,
        calib_max_samples=4,
        load_pretrained=False,
    )
    images = next(iter(iter_image_batches(loader, max_batches=1)))
    cos = int16_vs_fp32_cosine(bundle.sim, images)
    assert cos >= 0.95, f"toy zip val INT16 cosine={cos:.6f}"


@pytest.mark.slow
@pytest.mark.imagenet
def test_imagenet_mobilenet_int16_vs_fp32(imagenet_bundle):
    """INT16_FIXED_EVAL vs FP32_QDQ on ImageNet val (design §10.1)."""

    bundle = imagenet_bundle["bundle"]
    loader = imagenet_bundle["loader"]
    images = next(iter(iter_image_batches(loader, max_batches=1)))
    cos = int16_vs_fp32_cosine(bundle.sim, images)
    acc_ref = top1_accuracy(bundle.sim, loader, ExecutionMode.FP32_QDQ, max_batches=EVAL_BATCHES)
    acc_int = top1_accuracy(bundle.sim, loader, ExecutionMode.INT16_FIXED_EVAL, max_batches=EVAL_BATCHES)
    drop = top1_drop(acc_ref, acc_int)
    _report_imagenet_metrics(
        "INT16 vs FP32",
        cosine=cos,
        top1_fp32=acc_ref,
        top1_int16=acc_int,
        top1_drop=drop,
    )
    assert cos >= IMAGENET_INT16_MIN_COSINE, f"INT16 cosine={cos:.6f}"
    assert drop <= IMAGENET_INT16_MAX_TOP1_DROP, (
        f"INT16 top1 drop={drop:.4f} (ref={acc_ref:.4f}, int={acc_int:.4f})"
    )


@pytest.mark.slow
@pytest.mark.imagenet
def test_imagenet_mobilenet_fixed_scale_vs_fp32(imagenet_bundle):
    """fixed_scale_qdq vs FP32_QDQ top-1 drop on ImageNet val."""

    bundle = imagenet_bundle["bundle"]
    loader = imagenet_bundle["loader"]
    acc_ref = top1_accuracy(bundle.sim, loader, ExecutionMode.FP32_QDQ, max_batches=EVAL_BATCHES)
    acc_fix = top1_accuracy(bundle.sim, loader, ExecutionMode.FIXED_SCALE_QDQ, max_batches=EVAL_BATCHES)
    drop = top1_drop(acc_ref, acc_fix)
    cos = logits_cosine_on_loader(
        bundle.sim,
        loader,
        ExecutionMode.FIXED_SCALE_QDQ,
        max_batches=2,
    )
    _report_imagenet_metrics(
        "fixed_scale vs FP32",
        top1_fp32=acc_ref,
        top1_fixed=acc_fix,
        top1_drop=drop,
        mean_cosine=cos,
    )
    assert drop <= IMAGENET_FIXED_SCALE_MAX_TOP1_DROP, (
        f"fixed_scale top1 drop={drop:.4f} (ref={acc_ref:.4f}, fix={acc_fix:.4f})"
    )
    assert cos >= 0.999, f"fixed_scale mean cosine={cos:.6f}"


@pytest.mark.slow
@pytest.mark.imagenet
def test_imagenet_mobilenet_fp16_vs_fp32(imagenet_bundle):
    """FP16_QDQ vs FP32_QDQ top-1 drop on ImageNet val."""

    bundle = imagenet_bundle["bundle"]
    loader = imagenet_bundle["loader"]
    acc_ref = top1_accuracy(bundle.sim, loader, ExecutionMode.FP32_QDQ, max_batches=EVAL_BATCHES)
    acc_fp16 = top1_accuracy(bundle.sim, loader, ExecutionMode.FP16_QDQ, max_batches=EVAL_BATCHES)
    drop = top1_drop(acc_ref, acc_fp16)
    _report_imagenet_metrics(
        "FP16 vs FP32",
        top1_fp32=acc_ref,
        top1_fp16=acc_fp16,
        top1_drop=drop,
    )
    assert drop <= IMAGENET_FP16_MAX_TOP1_DROP, (
        f"fp16 top1 drop={drop:.4f} (ref={acc_ref:.4f}, fp16={acc_fp16:.4f})"
    )


def test_imagenet_val_loader_requires_class_subdirs(tmp_path):
    """``build_imagenet_val_loader`` requires at least one class subdirectory."""

    empty = tmp_path / "empty_val"
    empty.mkdir()
    with pytest.raises((FileNotFoundError, RuntimeError, ValueError)):
        build_imagenet_val_loader(empty, batch_size=1, max_samples=1)

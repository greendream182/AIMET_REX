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
"""Unit tests for ImageNet eval helpers (no full ImageNet val required)."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from aimet_torch.fixed_point.e2e.imagenet_eval import (  # noqa: E402
    IMAGENET_VAL_ENV,
    build_imagenet_val_loader,
    resolve_imagenet_val_dir,
    top1_drop,
    try_load_torchvision_imagenet_weights,
)


def _tiny_val(root: Path, n_classes: int = 2) -> Path:
    val = root / "val"
    for idx in range(n_classes):
        d = val / f"class_{idx:04d}"
        d.mkdir(parents=True, exist_ok=True)
        from torchvision.io import write_png

        write_png(torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8), str(d / "a.png"))
    return val


def test_resolve_imagenet_val_dir_from_env(tmp_path, monkeypatch):
    val = _tiny_val(tmp_path)
    monkeypatch.setenv(IMAGENET_VAL_ENV, str(val))
    assert resolve_imagenet_val_dir() == val


def test_resolve_imagenet_val_dir_missing(monkeypatch):
    monkeypatch.delenv(IMAGENET_VAL_ENV, raising=False)
    import aimet_torch.fixed_point.e2e.imagenet_eval as imagenet_eval

    monkeypatch.setattr(imagenet_eval, "_imagenet_val_fallback_candidates", lambda: [])
    assert resolve_imagenet_val_dir() is None


def test_top1_drop_non_negative():
    assert top1_drop(0.72, 0.70) == pytest.approx(0.02)
    assert top1_drop(0.70, 0.72) == 0.0


def test_build_imagenet_val_loader_max_samples(tmp_path):
    val = _tiny_val(tmp_path)
    loader = build_imagenet_val_loader(val, batch_size=2, max_samples=2, input_size=32)
    batch = next(iter(loader))
    assert batch[0].shape[0] <= 2
    assert batch[0].shape[-1] == 32


@pytest.mark.slow
def test_build_torchvision_mobilenet_smoke():
    """Prepared torchvision MobileNetV2 loads ImageNet weights (224)."""
    from aimet_torch.fixed_point.e2e.mobilenet_v2 import build_prepared_mobilenet_v2

    model, dummy = build_prepared_mobilenet_v2(
        n_class=1000,
        input_size=224,
        variant="torchvision",
    )
    assert dummy.shape == (1, 3, 224, 224)
    with torch.no_grad():
        y = model(dummy)
    assert y.shape[-1] == 1000


def test_try_load_torchvision_weights_on_mobilenet():
    from aimet_torch.examples.mobilenet import MobileNetV2

    model = MobileNetV2(n_class=1000, input_size=224).eval()
    loaded = try_load_torchvision_imagenet_weights(model)
    # Custom MobileNetV2 layout may not match torchvision keys; API must not raise.
    assert isinstance(loaded, bool)

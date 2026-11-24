"""Colour-histogram features for Baseline A (the linear SVM).

    from leafshift.features import split_features
    X, y = split_features("train")    # X: (N, 512) float32, each row sums to 1

    python -m leafshift.features      # run the self-checks below

Each image becomes a joint HSV histogram with BINS x BINS x BINS = 512 bins:
the fraction of the image's pixels whose hue, saturation and value fall in
each cell. HSV separates colour (hue) from brightness (value), so a brown
lesion stays in the same hue bins under different lighting, and fractions
make the feature independent of image size. Design in the decision log,
24 Nov 2026.

Padding is left out. build_cache.py pads non-square field photos with one
exact colour; PlantVillage is never padded. Counting the padding would give
every field image a spike in a bin the model never saw in training, and that
would show up as part of the lab-to-field gap. The pad colour is read from
cache.json so it cannot drift from what the cache used.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from leafshift.data import load_cache_meta, make_loader

BINS = 8  # per channel, so BINS ** 3 features
N_FEATURES = BINS ** 3


def pad_colour() -> tuple[int, int, int]:
    return tuple(load_cache_meta()["pad_colour"])


def colour_histogram(image: np.ndarray, pad: tuple[int, int, int]) -> np.ndarray:
    """One uint8 RGB image (H, W, 3) -> (N_FEATURES,) float32 summing to 1, padding excluded."""
    keep = ~np.all(image == np.array(pad, dtype=np.uint8), axis=-1)
    if not keep.any():
        raise ValueError("image is entirely padding")
    # PIL's HSV puts hue, saturation and value each on 0..255.
    hsv = np.asarray(Image.fromarray(image).convert("HSV"))[keep].astype(np.int64)
    h, s, v = (hsv * BINS // 256).T
    counts = np.bincount((h * BINS + s) * BINS + v, minlength=N_FEATURES)
    return (counts / keep.sum()).astype(np.float32)


def colour_histograms(images: torch.Tensor | np.ndarray, pad: tuple[int, int, int]) -> np.ndarray:
    """A batch of uint8 images, channels first (B, 3, H, W) as the loader serves
    them with normalize=False -> (B, N_FEATURES)."""
    batch = images.numpy() if isinstance(images, torch.Tensor) else images
    if batch.dtype != np.uint8:
        raise TypeError("expects raw uint8 pixels; build the loader with normalize=False")
    return np.stack([colour_histogram(img.transpose(1, 2, 0), pad) for img in batch])


def split_features(split: str, batch_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Features and labels for a whole split, in index-CSV order."""
    pad = pad_colour()
    xs, ys = [], []
    for images, labels in make_loader(split, batch_size=batch_size, shuffle=False, normalize=False):
        xs.append(colour_histograms(images, pad))
        ys.append(labels.numpy())
    return np.concatenate(xs), np.concatenate(ys)


def _self_check() -> None:
    pad = pad_colour()
    rng = np.random.default_rng(0)

    # 1. Padding is ignored: a picture padded to square has the same histogram as the picture alone.
    content = rng.integers(0, 256, size=(150, 224, 3), dtype=np.uint8)
    padded = np.full((224, 224, 3), pad, dtype=np.uint8)
    padded[37:187] = content
    same = np.allclose(colour_histogram(content, pad), colour_histogram(padded, pad))
    print(f"padding ignored: {same}")

    # 2. Every histogram is a set of fractions.
    hist = colour_histogram(content, pad)
    print(f"{hist.shape[0]} bins, sum {hist.sum():.6f}, min {hist.min():.3f}")

    # 3. On real data: how much of each split is masked as padding. PlantVillage
    #    is never padded, so anything above zero there would be real pixels that
    #    happen to match the pad colour.
    for split in ("train", "field_test"):
        masked, total = 0, 0
        for images, _ in make_loader(split, batch_size=512, shuffle=False, normalize=False):
            hwc = images.numpy().transpose(0, 2, 3, 1)
            masked += int(np.all(hwc == np.array(pad, dtype=np.uint8), axis=-1).sum())
            total += hwc.shape[0] * hwc.shape[1] * hwc.shape[2]
        print(f"{split:10s} pixels masked as padding: {masked:,} of {total:,} ({masked / total:.4%})")

    assert same and abs(hist.sum() - 1) < 1e-5


if __name__ == "__main__":
    _self_check()

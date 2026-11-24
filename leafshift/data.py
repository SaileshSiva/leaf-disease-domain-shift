"""Serve the cached images as PyTorch batches.

The cache is built by scripts/build_cache.py: one uint8 array per split,
shape (N, 224, 224, 3), with a CSV index giving each row's path, class and label.

    from leafshift.data import make_loader
    loader = make_loader("train", batch_size=64)
    for images, labels in loader:   # images: float32 (B, 3, 224, 224), labels: int64 (B,)
        ...

Per item, in this order:
    1. the cached pixels, uint8, channels first (3, 224, 224)
    2. the optional transform (augmentation, from Day 16), applied to uint8
    3. ImageNet normalisation, unless normalize=False

Every CNN uses ImageNet normalisation, including the one trained from scratch,
so the input is the same whichever model is being compared (decision log,
24 Nov 2026). The colour-histogram SVM passes normalize=False and gets the raw
uint8 pixels.
"""
from __future__ import annotations

import json
import random
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "data" / "cache"
SPLITS = ("train", "val", "test", "field_finetune", "field_test")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_MEAN = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
_STD = torch.tensor(IMAGENET_STD).view(3, 1, 1)


def load_cache_meta(cache_dir: Path = CACHE_DIR) -> dict:
    meta_path = cache_dir / "cache.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found. Run scripts/build_cache.py first.")
    return json.loads(meta_path.read_text())


def normalize(images: torch.Tensor) -> torch.Tensor:
    """uint8 (..., 3, H, W) -> float32 with ImageNet mean and std."""
    return (images.float() / 255.0 - _MEAN.to(images.device)) / _STD.to(images.device)


def denormalize(images: torch.Tensor) -> torch.Tensor:
    """Inverse of normalize, back to uint8. Used to inspect batches and draw overlays."""
    pixels = (images * _STD.to(images.device) + _MEAN.to(images.device)) * 255.0
    return pixels.round().clamp(0, 255).to(torch.uint8)


class CachedSplit(Dataset):
    """One split of the cache. Item i is (image, label) for row i of <split>.csv."""

    def __init__(self, split: str, cache_dir: Path = CACHE_DIR, normalize: bool = True,
                 transform: Callable[[torch.Tensor], torch.Tensor] | None = None) -> None:
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
        meta = load_cache_meta(cache_dir)
        self.split = split
        self.classes: list[str] = meta["classes"]
        self.normalize = normalize
        self.transform = transform
        self._npy = cache_dir / f"{split}.npy"

        index = pd.read_csv(cache_dir / f"{split}.csv")
        self.paths: list[str] = index["path"].tolist()
        self.labels = index["label"].to_numpy(dtype=np.int64)
        # The label numbers must mean the class names they sit next to.
        if not (index["class"].map(self.classes.index).to_numpy() == self.labels).all():
            raise ValueError(f"{split}.csv labels disagree with the class list in cache.json")

        shape = np.load(self._npy, mmap_mode="r").shape
        size = meta["size"]
        if shape != (len(index), size, size, 3):
            raise ValueError(f"{self._npy.name} has shape {shape}, expected ({len(index)}, {size}, {size}, 3)")
        # Opened on first use, so each DataLoader worker maps the file itself
        # rather than inheriting a handle from the parent process.
        self._images: np.ndarray | None = None

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_images"] = None
        return state

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        if self._images is None:
            self._images = np.load(self._npy, mmap_mode="r")
        image = torch.from_numpy(np.array(self._images[i])).permute(2, 0, 1)  # copy out of the map, HWC -> CHW
        if self.transform is not None:
            image = self.transform(image)
        if self.normalize:
            image = normalize(image)
        return image, int(self.labels[i])


def _seed_worker(worker_id: int) -> None:
    """Give numpy and random in each worker a seed derived from torch's, so any
    augmentation that uses them is reproducible too (PyTorch reproducibility notes)."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def make_loader(split: str, batch_size: int = 64, shuffle: bool | None = None, seed: int = 0,
                num_workers: int = 0, normalize: bool = True,
                transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
                drop_last: bool = False, cache_dir: Path = CACHE_DIR) -> DataLoader:
    """A DataLoader over one split.

    shuffle defaults to True for train and False for everything else, so
    evaluation always sees rows in index order. The shuffle order comes from a
    generator seeded with `seed`, so the same seed gives the same batches.
    """
    dataset = CachedSplit(split, cache_dir=cache_dir, normalize=normalize, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train") if shuffle is None else shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=num_workers,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        drop_last=drop_last,
    )

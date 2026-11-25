"""Sanity-check leafshift.data before anything is trained on it.

    python scripts/check_loader.py              # all checks, plus a labelled grid per split
    python scripts/check_loader.py --workers 4  # also time a training pass with 4 workers

Reads the cache built by scripts/build_cache.py and the counts in
data/splits/splits.json. For every split it checks:
    - batches are float32 (B, 3, 224, 224) and labels are valid class numbers
    - unshuffled, the labels come out in exactly the order of <split>.csv
    - a full pass yields every row once, with the per-class counts in splits.json
    - denormalising a batch gives back the cached pixels exactly
For the shuffled training loader it checks that each epoch is a permutation of
every row, that the same seed gives the same order and a different seed does not.

It writes data/cache/sheets/loader_<split>.jpg, one shuffled batch drawn back
from the loader with each image's class, for a by-eye check, and reports how
fast a training pass is so loading can be ruled out as a bottleneck.
Exits non-zero if any check fails.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torchvision.utils import make_grid

from leafshift.data import CACHE_DIR, REPO_ROOT, SPLITS, CachedSplit, denormalize, make_loader

SPLITS_JSON = REPO_ROOT / "data" / "splits" / "splits.json"
SHEETS_DIR = CACHE_DIR / "sheets"
BATCH = 256
GRID = 16  # images per by-eye grid
SEED = 0

failures: list[str] = []


def expect(ok: bool, what: str) -> None:
    print(f"  {'ok  ' if ok else 'FAIL'} {what}")
    if not ok:
        failures.append(what)


def check_split(split: str, counts: dict) -> None:
    print(f"\n{split}")
    dataset = CachedSplit(split)
    index = pd.read_csv(CACHE_DIR / f"{split}.csv")
    n_classes = len(dataset.classes)

    labels, shapes_ok, dtype_ok, first = [], True, True, None
    for images, y in make_loader(split, batch_size=BATCH, shuffle=False):
        shapes_ok &= images.shape[1:] == (3, 224, 224) and len(images) == len(y)
        dtype_ok &= images.dtype == torch.float32 and y.dtype == torch.int64
        labels.append(y)
        if first is None:
            first = images
    labels = torch.cat(labels)

    expect(shapes_ok, "every batch is (B, 3, 224, 224) with one label per image")
    expect(dtype_ok, "images float32, labels int64")
    expect(bool(((labels >= 0) & (labels < n_classes)).all()), f"labels within 0..{n_classes - 1}")
    expect(len(labels) == len(index), f"full pass yields {len(labels)} of {len(index)} rows")
    expect(torch.equal(labels, torch.from_numpy(index["label"].to_numpy())), "labels in the order of the index CSV")

    per_class = {c: int((labels == i).sum()) for i, c in enumerate(dataset.classes)}
    wanted = {c: counts[c][split] for c in dataset.classes}
    expect(per_class == wanted, "per-class counts match splits.json")

    cached = torch.from_numpy(np.array(np.load(CACHE_DIR / f"{split}.npy", mmap_mode="r")[:len(first)]))
    expect(torch.equal(denormalize(first).permute(0, 2, 3, 1), cached), "denormalised batch equals the cached pixels")

    write_grid(split, dataset.classes)


def check_shuffle() -> None:
    print("\ntrain, shuffled")
    n = len(CachedSplit("train"))

    def order(seed: int) -> list[int]:
        return list(iter(make_loader("train", batch_size=BATCH, seed=seed).sampler))

    a, b, c = order(SEED), order(SEED), order(SEED + 1)
    expect(sorted(a) == list(range(n)), "an epoch visits every row exactly once")
    expect(a != list(range(n)), "the order is actually shuffled")
    expect(a == b, "same seed, same order")
    expect(a != c, "different seed, different order")


def write_grid(split: str, classes: list[str]) -> None:
    """One shuffled batch, drawn back from the loader rather than the cache."""
    images, y = next(iter(make_loader(split, batch_size=GRID, shuffle=True, seed=SEED)))
    grid = make_grid(denormalize(images), nrow=4, padding=0).permute(1, 2, 0).numpy()
    sheet = Image.fromarray(grid)
    draw = ImageDraw.Draw(sheet)
    for k, label in enumerate(y.tolist()):
        x, top = (k % 4) * 224, (k // 4) * 224
        name = classes[label].replace("___", ": ").replace("_", " ")
        draw.rectangle((x, top, x + 224, top + 16), fill=(0, 0, 0))
        draw.text((x + 4, top + 3), f"{label} {name}", fill=(255, 255, 0))
    SHEETS_DIR.mkdir(parents=True, exist_ok=True)
    out = SHEETS_DIR / f"loader_{split}.jpg"
    sheet.save(out, quality=90)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def time_train_pass(workers: int) -> None:
    loader = make_loader("train", batch_size=64, num_workers=workers)
    start = time.perf_counter()
    n = sum(len(y) for _, y in loader)
    seconds = time.perf_counter() - start
    print(f"  {workers} workers: {n} images in {seconds:.1f} s ({n / seconds:,.0f} images/s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--workers", type=int, default=0, help="also time a training pass with this many workers")
    args = parser.parse_args()

    counts = json.loads(SPLITS_JSON.read_text())["counts"]
    for split in SPLITS:
        check_split(split, counts)
    check_shuffle()

    print("\nspeed of one training pass (batch 64)")
    time_train_pass(0)
    if args.workers:
        time_train_pass(args.workers)

    print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECK(S) FAILED'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

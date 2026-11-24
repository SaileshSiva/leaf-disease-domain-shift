"""Preprocess every image in the splits once and cache it, so training is not I/O bound.

    python scripts/build_cache.py            # build data/cache/ (skips if already up to date)
    python scripts/build_cache.py --force    # rebuild regardless
    python scripts/build_cache.py --check    # verify the cache against splits.csv
    python scripts/build_cache.py --sheets   # write contact sheets for the by-eye check

Reads data/splits/splits.csv, so run scripts/make_splits.py first.

Each image is, in order (design in the decision log, 23 Nov 2026):
    1. rotated as its EXIF orientation says, since some photos are stored sideways
    2. converted to RGB
    3. padded to square with the ImageNet mean colour, keeping the whole photo
    4. resized to SIZE x SIZE

PlantVillage images are already square, so step 3 only ever changes field photos.
Pixels are stored as uint8. Normalisation (ImageNet mean and std) happens at
load time, so the same cache serves the colour-histogram SVM and the CNNs.

Writes, under data/cache/:
    <split>.npy     uint8 array, shape (N, SIZE, SIZE, 3), memory-mappable
    <split>.csv     row -> path, class, label; same order as the array
    cache.json      parameters, class list and a fingerprint of splits.csv,
                    used to tell whether the cache is up to date
    sheets/         contact sheets for the by-eye check (with --sheets)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from tqdm import tqdm

from make_splits import CLASSES

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
SPLITS_CSV = REPO_ROOT / "data" / "splits" / "splits.csv"
CACHE_DIR = REPO_ROOT / "data" / "cache"
SHEETS_DIR = CACHE_DIR / "sheets"

SIZE = 224
PAD_COLOUR = (124, 116, 104)  # ImageNet mean (0.485, 0.456, 0.406) x 255, rounded
RESAMPLE = Image.Resampling.LANCZOS
SPLITS = ["train", "val", "test", "field_finetune", "field_test"]

SHEET_SAMPLES = 16  # image pairs per split on the contact sheet
SHEET_SEED = 0
CHECK_SAMPLES = 20  # images per split re-processed by --check


def preprocess(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        im = ImageOps.pad(im, (SIZE, SIZE), method=RESAMPLE, color=PAD_COLOUR)
        return np.asarray(im, dtype=np.uint8)


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_meta() -> dict:
    return {
        "size": SIZE,
        "pad_colour": list(PAD_COLOUR),
        "resample": RESAMPLE.name,
        "classes": CLASSES,
        "splits_csv_sha256": fingerprint(SPLITS_CSV),
    }


def load_splits() -> pd.DataFrame:
    if not SPLITS_CSV.exists():
        sys.exit(f"{SPLITS_CSV} not found. Run scripts/make_splits.py first.")
    df = pd.read_csv(SPLITS_CSV)
    unknown = set(df["class"]) - set(CLASSES)
    assert not unknown, f"splits.csv has classes outside the class set: {unknown}"
    return df.assign(label=df["class"].map(CLASSES.index))


def is_up_to_date() -> bool:
    meta_path = CACHE_DIR / "cache.json"
    if not meta_path.exists():
        return False
    meta = json.loads(meta_path.read_text())
    expected = expected_meta()
    same_params = all(meta.get(k) == v for k, v in expected.items())
    files_there = all((CACHE_DIR / f"{s}.npy").exists() and (CACHE_DIR / f"{s}.csv").exists() for s in SPLITS)
    return same_params and files_there


def build(df: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    counts, rotated = {}, 0
    for split in SPLITS:
        rows = df[df["split"] == split].reset_index(drop=True)
        tmp = CACHE_DIR / f"{split}.tmp.npy"
        arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(rows), SIZE, SIZE, 3))
        for i, path in enumerate(tqdm(rows["path"], desc=split, unit="img")):
            arr[i] = preprocess(RAW_DIR / path)
            with Image.open(RAW_DIR / path) as im:
                rotated += im.getexif().get(0x0112, 1) > 1  # EXIF orientation; 0 is invalid and means none
        arr.flush()
        del arr
        # Write to a temporary name and rename, so an interrupted run never
        # leaves a half-filled array that looks complete.
        tmp.replace(CACHE_DIR / f"{split}.npy")
        rows[["path", "class", "label"]].to_csv(CACHE_DIR / f"{split}.csv", index_label="row", lineterminator="\n")
        counts[split] = len(rows)

    meta = expected_meta() | {"counts": counts, "exif_rotated": int(rotated)}
    (CACHE_DIR / "cache.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"cached {sum(counts.values())} images: {counts}")
    print(f"{rotated} images needed an EXIF rotation")


def check(df: pd.DataFrame) -> bool:
    """Index matches splits.csv row for row, arrays have the right shape, and a
    random sample of rows re-processes to exactly the cached pixels."""
    ok = is_up_to_date()
    if not ok:
        print("cache.json is missing or does not match the current splits and parameters")
    rng = np.random.default_rng(SHEET_SEED)
    for split in SPLITS:
        rows = df[df["split"] == split].reset_index(drop=True)
        index = pd.read_csv(CACHE_DIR / f"{split}.csv")
        arr = np.load(CACHE_DIR / f"{split}.npy", mmap_mode="r")
        aligned = index[["path", "class", "label"]].equals(rows[["path", "class", "label"]])
        shaped = arr.shape == (len(rows), SIZE, SIZE, 3) and arr.dtype == np.uint8
        sample = rng.choice(len(rows), size=min(CHECK_SAMPLES, len(rows)), replace=False)
        same = all(np.array_equal(arr[i], preprocess(RAW_DIR / rows.at[i, "path"])) for i in sample)
        print(f"{split:15s} rows aligned: {aligned}  shape: {shaped}  {len(sample)} re-processed match: {same}")
        ok &= aligned and shaped and same
    print("cache OK" if ok else "CACHE PROBLEM")
    return ok


def short_name(cls: str) -> str:
    crop, disease = cls.split("___")
    return f"{crop}: {disease.replace('_', ' ')}"


def original_tile(path: Path) -> Image.Image:
    """The file as stored, before any rotation, fitted into a tile for comparison."""
    with Image.open(path) as im:
        return ImageOps.pad(im.convert("RGB"), (SIZE, SIZE), color=(40, 40, 40))


def sheet(items: list[tuple[Path, np.ndarray, str]], out: Path) -> None:
    """Tile (original file, cached pixels, caption) triples as original | cached pairs."""
    pairs_per_row, caption = 4, 18
    n_rows = -(-len(items) // pairs_per_row)
    canvas = Image.new("RGB", (pairs_per_row * 2 * SIZE, n_rows * (SIZE + caption)), "white")
    draw = ImageDraw.Draw(canvas)
    for k, (path, pixels, text) in enumerate(items):
        x = (k % pairs_per_row) * 2 * SIZE
        y = (k // pairs_per_row) * (SIZE + caption)
        canvas.paste(original_tile(path), (x, y))
        canvas.paste(Image.fromarray(pixels), (x + SIZE, y))
        draw.text((x + 4, y + SIZE + 3), text, fill="black")
    canvas.save(out, quality=90)
    print(f"wrote {out.relative_to(REPO_ROOT)}")


def contact_sheets(df: pd.DataFrame) -> None:
    """Per split: SHEET_SAMPLES random rows. Plus every EXIF-rotated image, since
    the original tile shows the file as stored, so the rotation can be judged."""
    SHEETS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SHEET_SEED)
    rotated = []
    for split in SPLITS:
        rows = df[df["split"] == split].reset_index(drop=True)
        arr = np.load(CACHE_DIR / f"{split}.npy", mmap_mode="r")
        picks = sorted(rng.choice(len(rows), size=min(SHEET_SAMPLES, len(rows)), replace=False))
        sheet([(RAW_DIR / rows.at[i, "path"], arr[i], f"#{i} {short_name(rows.at[i, 'class'])}") for i in picks],
              SHEETS_DIR / f"{split}.jpg")
        for i, path in enumerate(rows["path"]):
            with Image.open(RAW_DIR / path) as im:
                orientation = im.getexif().get(0x0112, 1)
            if orientation > 1:  # 0 is invalid and, like 1, means no rotation
                rotated.append((RAW_DIR / path, arr[i], f"{split} #{i} EXIF {orientation} {short_name(rows.at[i, 'class'])}"))
    if rotated:
        sheet(rotated, SHEETS_DIR / "exif_rotated.jpg")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--force", action="store_true", help="rebuild even if the cache is up to date")
    parser.add_argument("--check", action="store_true", help="verify the cache and exit")
    parser.add_argument("--sheets", action="store_true", help="write contact sheets for the by-eye check")
    args = parser.parse_args()

    df = load_splits()
    if args.check:
        sys.exit(0 if check(df) else 1)
    if args.force or not is_up_to_date():
        build(df)
    else:
        print("cache is up to date; use --force to rebuild")
    if args.sheets:
        contact_sheets(df)


if __name__ == "__main__":
    main()

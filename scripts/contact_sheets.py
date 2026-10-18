"""Tile a PlantDoc class into numbered contact sheets for the label hand check.

    python scripts/contact_sheets.py "Potato leaf early blight" "Potato leaf late blight"

Reads data/audit/images.csv, so run scripts/audit_data.py first. PlantDoc's
train/ and test/ folders are pooled, as everywhere else in the project, and each
distinct photo (by MD5) is shown once.

For each class this writes, under data/label_check/<class>/:
    sheet_NN.jpg   grids of thumbnails, each stamped with a running number
    index.csv      number -> md5, every path holding that photo, and the labels
                   PlantDoc gives it (more than one means a conflicted photo)

The numbers are what the written-up check refers to, and index.csv turns them
back into filenames for the exclusion list.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageOps

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
IMAGES_CSV = REPO_ROOT / "data" / "audit" / "images.csv"
OUT_DIR = REPO_ROOT / "data" / "label_check"

TILE = 256  # px per thumbnail side
COLS, ROWS = 5, 4  # 20 photos per sheet


def load_plantdoc() -> pd.DataFrame:
    df = pd.read_csv(IMAGES_CSV)
    df = df[(df["dataset"] == "plantdoc") & df["error"].isna()]
    # Every label PlantDoc gives each photo, across both of its folders.
    labels = df.groupby("md5")["class"].apply(lambda s: " | ".join(sorted(set(s))))
    return df.assign(labels=df["md5"].map(labels))


def thumbnail(path: Path, number: int) -> Image.Image:
    with Image.open(path) as im:
        tile = ImageOps.pad(im.convert("RGB"), (TILE, TILE), color=(40, 40, 40))
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, 44, 24), fill=(0, 0, 0))
    draw.text((5, 5), str(number), fill=(255, 255, 0))
    return tile


def build(df: pd.DataFrame, plantdoc_class: str) -> int:
    rows = df[df["class"] == plantdoc_class].sort_values("path")
    if rows.empty:
        sys.exit(f"No PlantDoc images found for class {plantdoc_class!r}")

    photos = (
        rows.groupby("md5", sort=False)
        .agg(paths=("path", " ; ".join), labels=("labels", "first"))
        .reset_index()
    )
    photos.insert(0, "number", range(1, len(photos) + 1))

    out = OUT_DIR / plantdoc_class
    out.mkdir(parents=True, exist_ok=True)
    photos.to_csv(out / "index.csv", index=False)

    per_sheet = COLS * ROWS
    for start in range(0, len(photos), per_sheet):
        sheet = Image.new("RGB", (COLS * TILE, ROWS * TILE), (255, 255, 255))
        for i, photo in enumerate(photos.iloc[start : start + per_sheet].itertuples()):
            first_path = RAW_DIR / photo.paths.split(" ; ")[0]
            sheet.paste(thumbnail(first_path, photo.number), ((i % COLS) * TILE, (i // COLS) * TILE))
        sheet.save(out / f"sheet_{start // per_sheet + 1:02d}.jpg", quality=90)
    return len(photos)


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    df = load_plantdoc()
    for plantdoc_class in sys.argv[1:]:
        n = build(df, plantdoc_class)
        print(f"{plantdoc_class}: {n} photos -> {OUT_DIR / plantdoc_class}")


if __name__ == "__main__":
    main()

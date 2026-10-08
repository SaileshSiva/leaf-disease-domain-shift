"""Pair PlantVillage classes with their PlantDoc counterparts and count what each
pair has to work with.

    python scripts/class_overlap.py      # print the overlap tables

Reads data/audit/images.csv, so run scripts/audit_data.py first. The output is
markdown tables for docs/class-mapping.md.

Counts are reported three ways, because the audit found exact duplicates and
photos filed under two labels:
    raw          files on disk
    unique       distinct photos (exact duplicates within the class counted once)
    unconflicted unique photos that do not also appear under another PlantDoc label

PlantDoc counts pool its train/ and test/ folders, because its own split is too
small and leaks (see docs/data-audit.md).

Inclusion rule: an in-scope class is kept if it has a PlantDoc pair with at least
MIN_FIELD_IMAGES unconflicted photos. The reasoning is in docs/class-mapping.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGES_CSV = REPO_ROOT / "data" / "audit" / "images.csv"

# Every PlantVillage class with a PlantDoc counterpart, matched by crop and disease.
# PlantDoc names a healthy leaf by the crop alone ("Tomato leaf"), so those map to
# the ___healthy classes.
PAIRS = {
    "Apple___Apple_scab": "Apple Scab Leaf",
    "Apple___Cedar_apple_rust": "Apple rust leaf",
    "Apple___healthy": "Apple leaf",
    "Blueberry___healthy": "Blueberry leaf",
    "Cherry_(including_sour)___healthy": "Cherry leaf",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Corn Gray leaf spot",
    "Corn_(maize)___Common_rust_": "Corn rust leaf",
    "Corn_(maize)___Northern_Leaf_Blight": "Corn leaf blight",
    "Grape___Black_rot": "grape leaf black rot",
    "Grape___healthy": "grape leaf",
    "Peach___healthy": "Peach leaf",
    "Pepper,_bell___Bacterial_spot": "Bell_pepper leaf spot",
    "Pepper,_bell___healthy": "Bell_pepper leaf",
    "Potato___Early_blight": "Potato leaf early blight",
    "Potato___Late_blight": "Potato leaf late blight",
    "Raspberry___healthy": "Raspberry leaf",
    "Soybean___healthy": "Soyabean leaf",
    "Squash___Powdery_mildew": "Squash Powdery mildew leaf",
    "Strawberry___healthy": "Strawberry leaf",
    "Tomato___Bacterial_spot": "Tomato leaf bacterial spot",
    "Tomato___Early_blight": "Tomato Early blight leaf",
    "Tomato___Late_blight": "Tomato leaf late blight",
    "Tomato___Leaf_Mold": "Tomato mold leaf",
    "Tomato___Septoria_leaf_spot": "Tomato Septoria leaf spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Tomato two spotted spider mites leaf",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato leaf yellow virus",
    "Tomato___Tomato_mosaic_virus": "Tomato leaf mosaic virus",
    "Tomato___healthy": "Tomato leaf",
}

IN_SCOPE_CROPS = ("Tomato", "Potato")

# Enough PlantDoc photos for a field test and a small fine-tuning slice per class.
MIN_FIELD_IMAGES = 50


def markdown(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |")
    return "\n".join(lines)


def section(title: str, body: str) -> None:
    print(f"\n## {title}\n\n{body}")


def crop(pv_class: str) -> str:
    return pv_class.split("___")[0]


def counts(df: pd.DataFrame) -> pd.DataFrame:
    """raw, unique and unconflicted counts per (dataset, class)."""
    # No photo appears in both datasets (data-audit.md), so md5 alone identifies a photo.
    labels_per_photo = df.groupby("md5")["class"].nunique()
    conflicted = set(labels_per_photo[labels_per_photo > 1].index)

    table = df.groupby(["dataset", "class"]).agg(raw=("path", "size"), unique=("md5", "nunique"))
    clean = df[~df["md5"].isin(conflicted)].groupby(["dataset", "class"])["md5"].nunique()
    table["unconflicted"] = clean.reindex(table.index, fill_value=0)
    return table


def crop_summary(pv_classes: list[str]) -> str:
    """How many PlantVillage classes each crop has, and how many have a PlantDoc pair."""
    rows = []
    for name in sorted({crop(c) for c in pv_classes}):
        own = [c for c in pv_classes if crop(c) == name]
        rows.append({"crop": name, "PlantVillage classes": len(own),
                     "with a PlantDoc pair": sum(c in PAIRS for c in own)})
    table = pd.DataFrame(rows).sort_values(["with a PlantDoc pair", "crop"], ascending=[False, True])
    return markdown(table)


def overlap(c: pd.DataFrame, pv_classes: list[str]) -> str:
    """One row per in-scope PlantVillage class, paired or not."""
    rows = []
    for pv in sorted(x for x in pv_classes if crop(x) in IN_SCOPE_CROPS):
        doc = PAIRS.get(pv)
        pv_n = c.loc[("plantvillage", pv)]
        doc_n = c.loc[("plantdoc", doc)] if doc else None
        rows.append({
            "PlantVillage class": pv,
            "PV unique": pv_n["unique"],
            "PlantDoc class": doc or "—",
            "PD raw": doc_n["raw"] if doc else 0,
            "PD unique": doc_n["unique"] if doc else 0,
            "PD unconflicted": doc_n["unconflicted"] if doc else 0,
        })
    table = pd.DataFrame(rows)
    table["kept"] = table["PD unconflicted"].map(lambda n: "yes" if n >= MIN_FIELD_IMAGES else "no")
    return markdown(table.sort_values("PD unconflicted", ascending=False))


def main() -> None:
    if not IMAGES_CSV.exists():
        sys.exit(f"{IMAGES_CSV} not found. Run scripts/audit_data.py first.")
    df = pd.read_csv(IMAGES_CSV)
    c = counts(df)
    pv_classes = sorted(df.loc[df["dataset"] == "plantvillage", "class"].unique())
    doc_classes = set(df.loc[df["dataset"] == "plantdoc", "class"])

    unknown = {d for d in PAIRS.values() if d not in doc_classes} | {p for p in PAIRS if p not in pv_classes}
    if unknown:
        sys.exit(f"PAIRS names classes not in the audit: {sorted(unknown)}")

    section("Overlap by crop", crop_summary(pv_classes))
    section("Tomato and potato class pairs", overlap(c, pv_classes))


if __name__ == "__main__":
    main()

"""Assign every in-scope image to exactly one split, reproducibly from a seed.

    python scripts/make_splits.py           # write data/splits/splits.csv and splits.json
    python scripts/make_splits.py --check   # rebuild in memory and confirm the files match

Reads data/audit/images.csv (run scripts/audit_data.py first), the label-check
exclusions in docs/label-exclusions.csv and the listed near-duplicates in
docs/near-duplicates.csv.

Splits (the design is recorded in the decision log, 23 Nov 2026):
    PlantVillage  train / val / test        70 / 15 / 15, stratified by class
    PlantDoc      field_finetune            FINETUNE_PER_CLASS photos per class
                  field_test                every other kept photo

Every row is labelled with the PlantVillage class name, so both datasets share
one label set. Paths are relative to data/raw/, as in the audit.

Leakage defences, each asserted before anything is written:
    - Exact duplicates (same MD5) are reduced to one copy, so a photo cannot sit
      on both sides of a split under two filenames.
    - PlantDoc near-duplicates (resized, re-saved, cropped or zoomed copies) are
      grouped and reduced to one copy, the one with the most pixels. Groups come
      from a difference hash plus the pairs listed in docs/near-duplicates.csv,
      which includes crops the hash cannot see. A crop shows the same leaf, so
      keeping both would count one leaf twice in the field score.
    - No path, MD5 or group appears in more than one split.
PlantVillage gets exact-duplicate removal only: its near-duplicates are repeat
photographs of the same leaf, which cannot be grouped without leaf IDs the
dataset does not provide. That is a stated limitation, not an oversight.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from class_overlap import PAIRS

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
IMAGES_CSV = REPO_ROOT / "data" / "audit" / "images.csv"
EXCLUSIONS_CSV = REPO_ROOT / "docs" / "label-exclusions.csv"
NEAR_DUPLICATES_CSV = REPO_ROOT / "docs" / "near-duplicates.csv"
OUT_DIR = REPO_ROOT / "data" / "splits"

SEED = 42
PV_VAL_FRACTION = 0.15
PV_TEST_FRACTION = 0.15
FINETUNE_PER_CLASS = 10
DHASH_MAX_BITS = 10  

CLASSES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

# Distinct PlantDoc photos kept per class after the label check (docs/label-check.md).
# The script refuses to split if what it reads disagrees with what was written up.
EXPECTED_FIELD_COUNTS = {
    "Potato___Early_blight": 62,
    "Potato___Late_blight": 88,
    "Tomato___Bacterial_spot": 57,
    "Tomato___Early_blight": 73,
    "Tomato___Late_blight": 103,
    "Tomato___Leaf_Mold": 68,
    "Tomato___Septoria_leaf_spot": 126,
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": 60,
    "Tomato___Tomato_mosaic_virus": 41,
    "Tomato___healthy": 45,
}

COLUMNS = ["path", "md5", "dataset", "class", "source_class", "group_id", "split"]


def load_audit() -> pd.DataFrame:
    if not IMAGES_CSV.exists():
        sys.exit(f"{IMAGES_CSV} not found. Run scripts/audit_data.py first.")
    df = pd.read_csv(IMAGES_CSV)
    return df[df["error"].isna()]


def one_copy_per_photo(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the alphabetically first path of each (class, MD5)."""
    return df.sort_values("path").drop_duplicates(["class", "md5"], keep="first")


def plantvillage(audit: pd.DataFrame) -> pd.DataFrame:
    pv = audit[(audit["dataset"] == "plantvillage") & audit["class"].isin(CLASSES)]
    pv = one_copy_per_photo(pv)
    shared = pv["md5"].duplicated(keep=False)
    assert not shared.any(), f"PlantVillage photos under two classes: {pv.loc[shared, 'path'].tolist()}"
    return pv.assign(source_class=pv["class"], group_id=pv["md5"])


def plantdoc(audit: pd.DataFrame) -> pd.DataFrame:
    to_pv = {PAIRS[c]: c for c in CLASSES}
    doc = audit[(audit["dataset"] == "plantdoc") & audit["class"].isin(to_pv)]

    # Exclude by path, not MD5: where PlantDoc files one photo under two labels,
    # the copy in the correct folder is kept and shares its MD5 with the excluded one.
    excluded = set(pd.read_csv(EXCLUSIONS_CSV)["path"])
    unknown = excluded - set(audit["path"])
    assert not unknown, f"label-exclusions.csv names files not in the audit: {sorted(unknown)[:5]}"
    doc = one_copy_per_photo(doc[~doc["path"].isin(excluded)])

    shared = doc["md5"].duplicated(keep=False)
    assert not shared.any(), f"Kept PlantDoc photos under two classes: {doc.loc[shared, 'path'].tolist()}"

    doc = doc.rename(columns={"class": "source_class"})
    doc["class"] = doc["source_class"].map(to_pv)
    counts = doc["class"].value_counts().to_dict()
    assert counts == EXPECTED_FIELD_COUNTS, f"Kept counts differ from label-check.md: {counts}"
    return doc


def dhash(path: Path) -> int:
    """64-bit difference hash: does each pixel of a 9x8 greyscale thumbnail get brighter to the right?"""
    with Image.open(path) as im:
        small = np.asarray(im.convert("L").resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        self.parent[self.find(i)] = self.find(j)


def near_duplicate_groups(doc: pd.DataFrame, audit: pd.DataFrame) -> pd.Series:
    """A group ID per PlantDoc row; rows sharing an ID must land in the same split."""
    doc = doc.reset_index(drop=True)
    hashes = np.array([dhash(RAW_DIR / p) for p in doc["path"]], dtype=np.uint64)
    uf = UnionFind(len(doc))

    hash_pairs = set()
    for i in range(len(doc)):
        distance = np.array([bin(int(x)).count("1") for x in hashes[i + 1:] ^ hashes[i]])
        for j in np.flatnonzero(distance <= DHASH_MAX_BITS) + i + 1:
            uf.union(i, int(j))
            hash_pairs.add((i, int(j)))

    # Listed pairs name whichever copy the label check numbered, which may be an
    # exact duplicate dropped above, so match them by MD5 rather than by path.
    row_of_md5 = {m: i for i, m in enumerate(doc["md5"])}
    md5_of_path = dict(zip(audit["path"], audit["md5"]))
    listed = pd.read_csv(NEAR_DUPLICATES_CSV)
    for pair in listed.itertuples(index=False):
        i, j = (row_of_md5.get(md5_of_path.get(p)) for p in (pair.path_a, pair.path_b))
        assert i is not None and j is not None, f"{pair.label_check_ref} names a photo that is not kept"
        found_by_hash = (min(i, j), max(i, j)) in hash_pairs or uf.find(i) == uf.find(j)
        # The hash is expected to miss crops; missing an identical photo would mean it is broken.
        if pair.relation == "same_photo" and not found_by_hash:
            print(f"note: {pair.label_check_ref} is the same photo but the hash missed it", file=sys.stderr)
        uf.union(i, j)

    roots = [uf.find(i) for i in range(len(doc))]
    # Name each group after the MD5 of its alphabetically first path, so the ID
    # does not depend on row order.
    first_md5 = doc.assign(root=roots).sort_values("path").groupby("root")["md5"].first()
    group_ids = pd.Series([first_md5[r] for r in roots], index=doc.index)

    cross_class = doc.groupby(group_ids)["class"].nunique()
    spans = cross_class[cross_class > 1].index
    assert not len(spans), "Near-duplicates kept under two classes:\n" + "\n".join(
        doc.loc[group_ids.isin(spans), "path"])
    return group_ids


def one_copy_per_group(doc: pd.DataFrame) -> pd.DataFrame:
    """Keep the largest image of each near-duplicate group (ties broken by path)."""
    return (doc.assign(pixels=doc["width"] * doc["height"])
               .sort_values(["pixels", "path"], ascending=[False, True])
               .drop_duplicates("group_id", keep="first")
               .drop(columns="pixels"))


def split_plantvillage(pv: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    parts = []
    for cls in CLASSES:
        rows = pv[pv["class"] == cls].sort_values("path")
        order = rng.permutation(len(rows))
        n_test = round(len(rows) * PV_TEST_FRACTION)
        n_val = round(len(rows) * PV_VAL_FRACTION)
        split = np.empty(len(rows), dtype=object)
        split[order[:n_test]] = "test"
        split[order[n_test:n_test + n_val]] = "val"
        split[order[n_test + n_val:]] = "train"
        parts.append(rows.assign(split=split))
    return pd.concat(parts)


def split_plantdoc(doc: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Walk each class's groups in shuffled order, taking any group that still fits
    in the fine-tuning slice. Everything not taken is field test."""
    parts = []
    for cls in CLASSES:
        rows = doc[doc["class"] == cls]
        sizes = rows.groupby("group_id").size().sort_index()
        taken, slice_groups = 0, set()
        for g in sizes.index[rng.permutation(len(sizes))]:
            if taken + sizes[g] <= FINETUNE_PER_CLASS:
                slice_groups.add(g)
                taken += sizes[g]
        assert taken == FINETUNE_PER_CLASS, f"{cls}: could only fill {taken} fine-tuning photos"
        split = np.where(rows["group_id"].isin(slice_groups), "field_finetune", "field_test")
        parts.append(rows.assign(split=split))
    return pd.concat(parts)


def check_no_leakage(splits: pd.DataFrame) -> None:
    assert not splits["path"].duplicated().any(), "a path appears twice"
    for key in ("md5", "group_id"):
        per_key = splits.groupby(key)["split"].nunique()
        assert (per_key == 1).all(), f"{key} in two splits: {per_key[per_key > 1].index[:5].tolist()}"


def build() -> tuple[pd.DataFrame, dict]:
    audit = load_audit()
    pv = plantvillage(audit)
    doc = plantdoc(audit).reset_index(drop=True)
    doc["group_id"] = near_duplicate_groups(doc, audit)
    n_groups = int((doc["group_id"].value_counts() > 1).sum())
    n_before = len(doc)
    doc = one_copy_per_group(doc)

    rng = np.random.default_rng(SEED)
    splits = pd.concat([split_plantvillage(pv, rng), split_plantdoc(doc, rng)])[COLUMNS]
    check_no_leakage(splits)
    splits = splits.sort_values(["dataset", "split", "class", "path"]).reset_index(drop=True)

    counts = splits.groupby(["class", "split"]).size().unstack(fill_value=0)
    counts = counts[["train", "val", "test", "field_finetune", "field_test"]]
    meta = {
        "seed": SEED,
        "plantvillage_fractions": {"train": 1 - PV_VAL_FRACTION - PV_TEST_FRACTION,
                                   "val": PV_VAL_FRACTION, "test": PV_TEST_FRACTION},
        "field_finetune_per_class": FINETUNE_PER_CLASS,
        "dhash_max_bits": DHASH_MAX_BITS,
        "plantdoc_after_label_check": n_before,
        "near_duplicate_groups": n_groups,
        "near_duplicates_removed": n_before - len(doc),
        "counts": {cls: {k: int(v) for k, v in row.items()} for cls, row in counts.iterrows()},
        "totals": {k: int(v) for k, v in counts.sum().items()},
    }
    return splits, meta


def render(splits: pd.DataFrame, meta: dict) -> tuple[str, str]:
    return splits.to_csv(index=False, lineterminator="\n"), json.dumps(meta, indent=2) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true",
                        help="rebuild in memory and confirm the files on disk are identical")
    args = parser.parse_args()

    csv_text, json_text = render(*build())
    csv_path, json_path = OUT_DIR / "splits.csv", OUT_DIR / "splits.json"

    if args.check:
        same = all(p.exists() and p.read_text() == t for p, t in ((csv_path, csv_text), (json_path, json_text)))
        print("splits regenerate identically" if same else "MISMATCH: splits on disk differ from a fresh build")
        sys.exit(0 if same else 1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(csv_text)
    json_path.write_text(json_text)
    meta = json.loads(json_text)
    print(pd.DataFrame(meta["counts"]).T.to_string())
    print(f"\ntotals: {meta['totals']}")
    print(f"near-duplicate groups in PlantDoc: {meta['near_duplicate_groups']}")
    print(f"wrote {csv_path.relative_to(REPO_ROOT)} and {json_path.name}")


if __name__ == "__main__":
    main()

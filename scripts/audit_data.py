"""Audit the raw datasets: images per class, formats, resolutions, corrupt files
and exact duplicates.

    python scripts/audit_data.py                     # print the audit
    python scripts/audit_data.py > audit-tables.md   # keep it

The output is markdown tables, so it can go straight into docs/data-audit.md.
Every image is fully decoded, not just opened, so a truncated file counts as
corrupt even when its header reads fine. One row per file is also written to
data/audit/images.csv, and every duplicate group to data/audit/duplicates.csv,
so any number in the audit can be traced back to filenames without rerunning.

Duplicates here are exact: files whose bytes have the same MD5. A resized or
re-encoded copy of the same photograph is not caught.
"""
from __future__ import annotations

import hashlib
import io
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW = REPO_ROOT / "data" / "raw"
AUDIT = REPO_ROOT / "data" / "audit"

# (dataset, split, folder whose subfolders are the classes)
SOURCES = [
    ("plantvillage", "all", RAW / "plantvillage" / "raw" / "color"),
    ("plantdoc", "train", RAW / "plantdoc" / "train"),
    ("plantdoc", "test", RAW / "plantdoc" / "test"),
]

# The file extensions each Pillow format name should have.
FORMAT_SUFFIXES = {
    "JPEG": {".jpg", ".jpeg"},
    "PNG": {".png"},
    "BMP": {".bmp"},
    "GIF": {".gif"},
}


def inspect(path: Path) -> dict:
    """Hash one file and fully decode it. Never raises; a failure goes in 'error'."""
    data = path.read_bytes()
    row = {
        "bytes": len(data),
        "md5": hashlib.md5(data).hexdigest(),
        "format": None, "width": None, "height": None, "mode": None, "error": None,
    }
    try:
        with Image.open(io.BytesIO(data)) as img:
            row.update(format=img.format, width=img.width, height=img.height, mode=img.mode)
            img.load()  # decodes the pixels; open() alone only reads the header
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def collect() -> pd.DataFrame:
    files = []
    for dataset, split, class_root in SOURCES:
        if not class_root.is_dir():
            sys.exit(f"{class_root.relative_to(REPO_ROOT)} not found; run scripts/download_data.py first")
        for class_dir in sorted(p for p in class_root.iterdir() if p.is_dir()):
            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and not path.name.startswith("."):
                    files.append((dataset, split, class_dir.name, path))

    rows = []
    for dataset, split, cls, path in tqdm(files, desc="auditing", unit="file"):
        rows.append({
            "dataset": dataset, "split": split, "class": cls,
            "path": str(path.relative_to(RAW)), "suffix": path.suffix.lower(),
            **inspect(path),
        })
    return pd.DataFrame(rows)


def markdown(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |")
    return "\n".join(lines)


def section(title: str, body: str) -> None:
    print(f"\n## {title}\n\n{body}")


def overview(df: pd.DataFrame) -> str:
    table = df.groupby(["dataset", "split"], sort=False).agg(
        classes=("class", "nunique"),
        files=("path", "size"),
        corrupt=("error", lambda s: int(s.notna().sum())),
        size_mb=("bytes", lambda s: round(s.sum() / 1024 / 1024)),
    )
    return markdown(table.reset_index())


def class_counts(df: pd.DataFrame) -> tuple[str, str]:
    pv = df[df["dataset"] == "plantvillage"].groupby("class").size().rename("images")

    # One column per PlantDoc split, so a class missing from test shows as 0.
    pd_rows = df[df["dataset"] == "plantdoc"]
    doc = pd.crosstab(pd_rows["class"], pd_rows["split"]).reindex(columns=["train", "test"], fill_value=0)
    doc["total"] = doc["train"] + doc["test"]
    return markdown(pv.reset_index()), markdown(doc.reset_index())


def formats(df: pd.DataFrame) -> str:
    fmt = df["format"].fillna("unreadable")
    table = pd.crosstab([df["dataset"], df["split"]], fmt)

    mismatched = df[[
        f is not None and f in FORMAT_SUFFIXES and s not in FORMAT_SUFFIXES[f]
        for f, s in zip(df["format"], df["suffix"])
    ]]
    note = f"\n\nFiles whose extension does not match their actual format: {len(mismatched)}"
    if len(mismatched):
        note += "\n\n" + markdown(mismatched[["path", "suffix", "format"]])
    return markdown(table.reset_index()) + note


def resolutions(ok: pd.DataFrame) -> str:
    def summarise(g: pd.DataFrame) -> pd.Series:
        sizes = g["width"].astype(int).astype(str) + "x" + g["height"].astype(int).astype(str)
        top = sizes.value_counts()
        return pd.Series({
            "min w": int(g["width"].min()), "median w": int(g["width"].median()), "max w": int(g["width"].max()),
            "min h": int(g["height"].min()), "median h": int(g["height"].median()), "max h": int(g["height"].max()),
            "distinct sizes": len(top),
            "most common": f"{top.index[0]} ({top.iloc[0] / len(g):.0%})",
        })

    table = ok.groupby(["dataset", "split"], sort=False).apply(summarise, include_groups=False)
    return markdown(table.reset_index())


def modes(ok: pd.DataFrame) -> str:
    return markdown(pd.crosstab([ok["dataset"], ok["split"]], ok["mode"]).reset_index())


def corrupt(df: pd.DataFrame) -> str:
    bad = df[df["error"].notna()]
    if bad.empty:
        return "None. Every file decoded."
    return markdown(bad[["path", "error"]])


def duplicates(df: pd.DataFrame) -> str:
    dup = df[df.duplicated("md5", keep=False)]
    if dup.empty:
        return "No exact duplicates."

    groups = []
    for md5, g in dup.groupby("md5"):
        groups.append({
            "md5": md5,
            "copies": len(g),
            "across datasets": g["dataset"].nunique() > 1,
            "across splits": g[g["dataset"] == "plantdoc"]["split"].nunique() > 1,
            "across classes": g["class"].nunique() > 1,
            "paths": " ; ".join(g["path"]),
        })
    groups = pd.DataFrame(groups)
    groups.to_csv(AUDIT / "duplicates.csv", index=False)

    # A group can be in more than one row of the summary, e.g. across splits and classes.
    kinds = {
        "all duplicate groups": pd.Series(True, index=groups.index),
        "within a single class only": ~groups[["across datasets", "across splits", "across classes"]].any(axis=1),
        "across classes (conflicting labels)": groups["across classes"],
        "across PlantDoc train/test (leakage)": groups["across splits"],
        "across PlantVillage and PlantDoc": groups["across datasets"],
    }
    summary = pd.DataFrame([
        {"kind": k, "groups": int(m.sum()), "redundant files": int((groups.loc[m, "copies"] - 1).sum())}
        for k, m in kinds.items()
    ])

    serious = groups[~kinds["within a single class only"]]
    body = markdown(summary)
    if len(serious):
        body += "\n\nGroups that cross a class, split or dataset boundary:\n\n"
        body += markdown(serious[["copies", "across datasets", "across splits", "across classes", "paths"]])
    return body + "\n\nAll groups, including within-class ones, are in data/audit/duplicates.csv."


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    df = collect()
    df.to_csv(AUDIT / "images.csv", index=False)
    ok = df[df["error"].isna()]

    pv_counts, doc_counts = class_counts(df)
    section("Overview", overview(df))
    section("PlantVillage images per class", pv_counts)
    section("PlantDoc images per class", doc_counts)
    section("File formats", formats(df))
    section("Resolutions (readable files only)", resolutions(ok))
    section("Colour modes (readable files only)", modes(ok))
    section("Corrupt or unreadable files", corrupt(df))
    section("Exact duplicates", duplicates(df))


if __name__ == "__main__":
    main()

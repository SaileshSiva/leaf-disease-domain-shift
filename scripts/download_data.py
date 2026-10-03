"""Download the two datasets this project compares.

    python scripts/download_data.py            # download anything missing
    python scripts/download_data.py --force    # delete and re-download both
    python scripts/download_data.py --report   # just report what is on disk

PlantVillage supplies the laboratory images used for training and the in-domain
test split. PlantDoc supplies the field images used as the out-of-domain test.
Neither is committed; data/ is git-ignored and this script recreates it.

Licence position, recorded because the report has to state it accurately:
  PlantVillage  No licence file and no licence statement in its README. It ships
                a CITATION.cff asking that the paper be cited
                (Mohanty, Hughes and Salathe 2016, doi 10.3389/fpls.2016.01419).
                Treat it as publicly distributed for research with a citation
                request, not as explicitly licensed.
  PlantDoc      CC-BY-4.0, declared in LICENSE.txt.

Only the standard library is used, so this runs before anything is installed.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW = REPO_ROOT / "data" / "raw"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

# PlantVillage holds three variants of the same images (color, grayscale,
# segmented) and this project only uses color, so it is cloned sparsely to avoid
# fetching roughly 1.4 GB that would never be read.
PLANTVILLAGE = {
    "name": "plantvillage",
    "url": "https://github.com/spMohanty/PlantVillage-Dataset.git",
    "sparse_path": "raw/color",
    "class_dir": "raw/color",
    "expected_classes": 38,
}

PLANTDOC = {
    "name": "plantdoc",
    "url": "https://github.com/pratikkayal/PlantDoc-Dataset.git",
    "sparse_path": None,
    "class_dir": "train",
    "expected_classes": 28,
}

DATASETS = [PLANTVILLAGE, PLANTDOC]


def run(args: list[str], cwd: Path | None = None) -> None:
    """Run a command, letting its output through, and stop on failure."""
    result = subprocess.run(args, cwd=cwd)
    if result.returncode != 0:
        sys.exit(f"command failed: {' '.join(args)}")


def count_classes(path: Path) -> int:
    return sum(1 for p in path.iterdir() if p.is_dir()) if path.is_dir() else 0


def count_images(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def directory_size_mb(path: Path) -> int:
    if not path.is_dir():
        return 0
    total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return round(total / 1024 / 1024)


def is_present(spec: dict) -> bool:
    """A dataset counts as present only if its class folders are actually there."""
    return count_classes(RAW / spec["name"] / spec["class_dir"]) > 0


def clone(spec: dict) -> None:
    target = RAW / spec["name"]
    print(f"\n[{spec['name']}] cloning {spec['url']}")

    if spec["sparse_path"]:
        # A sparse clone checks out nothing until a path is selected.
        run([
            "git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
            spec["url"], str(target),
        ])
        print(f"[{spec['name']}] selecting {spec['sparse_path']}")
        run(["git", "sparse-checkout", "set", spec["sparse_path"]], cwd=target)

        if count_classes(target / spec["class_dir"]) == 0:
            print(f"[{spec['name']}] working tree empty after sparse-checkout; re-applying")
            run(["git", "sparse-checkout", "set", spec["sparse_path"]], cwd=target)
    else:
        run(["git", "clone", "--depth", "1", spec["url"], str(target)])

    classes = count_classes(target / spec["class_dir"])
    if classes == 0:
        sys.exit(
            f"[{spec['name']}] {spec['class_dir']}/ is empty after cloning. "
            f"Inspect {target} before rerunning."
        )
    if classes != spec["expected_classes"]:
        print(
            f"[{spec['name']}] warning: found {classes} class folders, "
            f"expected {spec['expected_classes']}. The upstream dataset may have changed."
        )

    git_dir = target / ".git"
    if git_dir.exists():
        print(f"[{spec['name']}] removing {git_dir.relative_to(REPO_ROOT)}")
        shutil.rmtree(git_dir)


def report() -> None:
    print(f"\n{'dataset':<14}{'classes':>9}{'images':>10}{'size MB':>10}  path")
    for spec in DATASETS:
        base = RAW / spec["name"]
        class_dir = base / spec["class_dir"]
        print(
            f"{spec['name']:<14}{count_classes(class_dir):>9}{count_images(base):>10}"
            f"{directory_size_mb(base):>10}  {class_dir.relative_to(REPO_ROOT)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--force", action="store_true",
                        help="delete and re-download both datasets")
    parser.add_argument("--report", action="store_true",
                        help="report what is on disk and exit")
    args = parser.parse_args()

    if args.report:
        report()
        return

    if shutil.which("git") is None:
        sys.exit("git is required but was not found on PATH")

    RAW.mkdir(parents=True, exist_ok=True)

    for spec in DATASETS:
        target = RAW / spec["name"]
        if args.force and target.exists():
            print(f"[{spec['name']}] --force: removing {target.relative_to(REPO_ROOT)}")
            shutil.rmtree(target)
        if is_present(spec):
            print(f"[{spec['name']}] already present, skipping")
            continue
        clone(spec)

    report()


if __name__ == "__main__":
    main()

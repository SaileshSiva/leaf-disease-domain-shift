"""Draw the report figures for one run.

    python scripts/plot_results.py results/runs/baseline_a_seed0_<timestamp>

Reads the run's metrics.json and writes results/figures/<name>_confusion.png
and <name>_per_class.png, where <name> is the config name. A later run of the
same config replaces the figures, so plot the run the report quotes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES = REPO_ROOT / "results" / "figures"

LAB, FIELD = "#2a78d6", "#eb6834"
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "font.size": 9, "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": GRID,
    "xtick.color": INK_2, "ytick.color": INK_2, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "savefig.dpi": 200,
})


def short(cls: str) -> str:
    crop, disease = cls.split("___")
    disease = disease.replace("Tomato_", "").replace("_", " ")
    return f"{crop} {disease.lower()}" if disease != "healthy" else f"{crop} healthy"


def confusion_figure(results: dict, classes: list[str], title: str, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), constrained_layout=True)
    names = [short(c) for c in classes]
    for ax, split, label in zip(axes, ("test", "field_test"), ("PlantVillage test (lab)", "PlantDoc field test")):
        counts = np.array(results[split]["confusion_matrix"])
        share = counts / counts.sum(axis=1, keepdims=True)
        im = ax.imshow(share, cmap=BLUES, vmin=0, vmax=1)
        for i, j in zip(*np.nonzero(counts)):
            ax.text(j, i, counts[i, j], ha="center", va="center", fontsize=7,
                    color="#ffffff" if share[i, j] > 0.5 else INK_2)
        acc = results[split]["accuracy"]
        ax.set_title(f"{label}  ·  accuracy {acc:.1%}", loc="left", fontsize=10, color=INK)
        ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
        ax.set_yticks(range(len(names)), names)
        ax.set_xlabel("predicted class")
        ax.set_ylabel("true class")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    bar = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.01)
    bar.set_label("share of the true class")
    bar.outline.set_visible(False)
    fig.suptitle(f"{title}: confusion matrices (cell = count, colour = share of row)", x=0.01, ha="left",
                 fontsize=11, color=INK)
    fig.savefig(out)
    plt.close(fig)


def per_class_figure(results: dict, classes: list[str], title: str, out: Path) -> None:
    lab = np.array([results["test"]["per_class_accuracy"][c] for c in classes])
    field = np.array([results["field_test"]["per_class_accuracy"][c] for c in classes])
    y = np.arange(len(classes))[::-1]
    h = 0.38

    fig, ax = plt.subplots(figsize=(8, 5.2), constrained_layout=True)
    ax.barh(y + h / 2, lab, height=h - 0.03, color=LAB, label="PlantVillage test (lab)")
    ax.barh(y - h / 2, field, height=h - 0.03, color=FIELD, label="PlantDoc field test")
    for yi, v in zip(y - h / 2, field):
        ax.text(v + 0.01, yi, f"{v:.0%}", va="center", fontsize=8, color=INK_2)
    ax.axvline(0.1, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(0.1, len(classes) - 0.45, " chance 10%", color=MUTED, fontsize=8, va="bottom")

    ax.set_yticks(y, [short(c) for c in classes])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("accuracy within the class")
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=2, frameon=False, borderaxespad=0.2)
    gap = results["test"]["accuracy"] - results["field_test"]["accuracy"]
    fig.suptitle(f"{title}: accuracy per class  ·  overall gap {gap * 100:.1f} points", x=0.01, ha="left",
                 fontsize=11, color=INK)
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    data = json.loads((args.run_dir / "metrics.json").read_text())
    name = json.loads((args.run_dir / "run.json").read_text())["name"]
    results = data["results"]
    classes = list(results["test"]["per_class_accuracy"])
    title = data["model"].split(":")[0]
    FIGURES.mkdir(parents=True, exist_ok=True)

    for kind, draw in (("confusion", confusion_figure), ("per_class", per_class_figure)):
        out = FIGURES / f"{name}_{kind}.png"
        draw(results, classes, title, out)
        print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

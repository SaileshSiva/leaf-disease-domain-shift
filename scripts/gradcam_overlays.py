"""Grad-CAM overlays for the same field-test photos under three models.

    python scripts/gradcam_overlays.py

Uses seed 0 of each model (the committed run and its local best.pt) and writes
results/figures/gradcam_field.png: one row per photo, the photo first, then one
overlay per model labelled with true and predicted class. The photos were fixed
before any prediction was looked at.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from leafshift.cnn import build_model  # noqa: E402
from leafshift.data import CachedSplit, denormalize, load_cache_meta  # noqa: E402
from leafshift.gradcam import gradcam  # noqa: E402
from leafshift.run import find_source_run  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "results" / "figures" / "gradcam_field.png"

MODELS = {
    "baseline_b": "Baseline B\nsmall CNN (lab only)",
    "effnet_b0_finetune_aug": "EfficientNet-B0\nfine-tuned + aug (lab only)",
    "effnet_b0_field_finetune": "EfficientNet-B0\n+ aug + field fine-tune",
}
# field_test cache rows: one per class drawn with numpy default_rng(42); then 555 and 169, the first two
# with a visible watermark (checked by eye) among the stock-site-named files, shuffled with default_rng(42)
ROWS = [4, 111, 158, 199, 271, 370, 388, 520, 541, 569, 555, 169]

INK, INK_2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
HEAT = LinearSegmentedColormap.from_list("heat", ["#fdeee6", "#f6a77f", "#eb6834", "#b8461a", "#7a2a0a"])


def short(cls: str) -> str:
    crop, disease = cls.split("___")
    disease = disease.replace("Tomato_", "").replace("_", " ").lower()
    return f"{crop} {disease}"


def load(name: str, classes: list[str]) -> torch.nn.Module:
    source = find_source_run(name, 0)
    checkpoint = torch.load(source["checkpoint"], map_location="cpu")
    if checkpoint["classes"] != classes:
        raise ValueError(f"{source['run']} was trained on a different class list")
    model, _, _ = build_model(checkpoint["model"], len(classes))
    model.load_state_dict(checkpoint["state_dict"])
    return model


def overlay(photo: np.ndarray, cam: np.ndarray) -> np.ndarray:
    grey = photo.mean(axis=2, keepdims=True).repeat(3, axis=2) * 0.85 + 0.15
    heat = HEAT(cam)[..., :3]
    alpha = cam[..., None] * 0.75
    return np.clip(grey * (1 - alpha) + heat * alpha, 0, 1)


def main() -> None:
    classes = load_cache_meta()["classes"]
    field = CachedSplit("field_test")
    images = torch.stack([field[r][0] for r in ROWS])
    labels = [int(field.labels[r]) for r in ROWS]
    photos = denormalize(images).permute(0, 2, 3, 1).numpy() / 255.0

    results = {}
    for name in MODELS:
        cams, preds = gradcam(load(name, classes), images)
        results[name] = (cams.numpy(), preds.tolist())

    plt.rcParams.update({"font.size": 7, "text.color": INK, "figure.facecolor": SURFACE,
                         "savefig.facecolor": SURFACE, "savefig.dpi": 200})
    n_rows, n_cols = len(ROWS), 1 + len(MODELS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.9 * n_cols, 1.95 * n_rows), squeeze=False)
    for i in range(n_rows):
        axes[i, 0].imshow(photos[i])
        axes[i, 0].set_ylabel(f"true: {short(classes[labels[i]])}\nrow {ROWS[i]}", fontsize=6.5, color=INK_2)
        for j, name in enumerate(MODELS, start=1):
            cams, preds = results[name]
            axes[i, j].imshow(overlay(photos[i], cams[i]))
            mark = "correct" if preds[i] == labels[i] else "wrong"
            axes[i, j].set_xlabel(f"pred: {short(classes[preds[i]])}\n({mark})", fontsize=6, color=INK_2, labelpad=2)
    for j, title in enumerate(["field photo", *MODELS.values()]):
        axes[0, j].set_title(title, fontsize=7.5, pad=6)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.tight_layout(rect=(0, 0.02, 1, 1), h_pad=1.2, w_pad=0.6)
    bar = fig.add_axes((0.3, 0.006, 0.4, 0.004))
    fig.colorbar(plt.cm.ScalarMappable(cmap=HEAT), cax=bar, orientation="horizontal")
    bar.set_xlabel("Grad-CAM weight for the predicted class (0 to 1 per image)", fontsize=6.5, color=INK_2)
    bar.tick_params(labelsize=6, colors=INK_2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")
    for name in MODELS:
        print(name, "predictions", results[name][1])


if __name__ == "__main__":
    main()

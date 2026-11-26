"""Baseline A: colour-histogram features + linear SVM, scored on both test sets.

    python scripts/train.py configs/baseline_a.toml

Steps:
    1. 512-bin HSV histogram per image, padding excluded (leafshift.features)
    2. square root of each bin (the Hellinger transform); no standardisation
    3. for each C in the config's C_grid: train on train, score macro F1 on val
    4. keep the best C (ties go to the smaller, simpler model)
    5. score the in-domain test and the field test with that model

Why the square root rather than standardising: about 160 of the 512 bins are
(almost) never used in PlantVillage, so their training spread is near zero.
Field photos do contain those colours, and dividing by that spread sent their
values to around 10^5, which overflowed the solver. Square-rooted fractions stay
in [0, 1] whatever the image, every image's feature vector has length 1, and
the few large green bins no longer swamp the small lesion colours.

The test sets are never used to choose anything. No class weighting: the
imbalance is handled in reporting (macro F1 beside accuracy) and decided for
all models together later.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

from leafshift.evaluate import EVAL_SPLITS, evaluate
from leafshift.features import BINS, split_features


def make_model(c: float, seed: int, max_iter: int):
    return make_pipeline(FunctionTransformer(np.sqrt), LinearSVC(C=c, random_state=seed, max_iter=max_iter))


def fit(model, X, y) -> bool:
    """Fit, and report whether the solver converged rather than hiding the warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X, y)
    return not any(issubclass(w.category, ConvergenceWarning) for w in caught)


def train_and_evaluate(cfg: dict, classes: list[str], run_dir: Path) -> dict:
    """run_dir is unused: the SVM saves no weights. Every trainer takes it."""
    seed, m = cfg["seed"], cfg["model"]

    start = time.perf_counter()
    data = {s: split_features(s) for s in ("train", "val", *EVAL_SPLITS)}
    print(f"features extracted in {time.perf_counter() - start:.0f} s: "
          + ", ".join(f"{s} {len(y)}" for s, (_, y) in data.items()))

    X_train, y_train = data["train"]
    X_val, y_val = data["val"]

    print("\ntuning C on val (macro F1)")
    tuning, best = [], None
    for c in m["C_grid"]:
        model = make_model(c, seed, m["max_iter"])
        converged = fit(model, X_train, y_train)
        val_f1 = f1_score(y_val, model.predict(X_val), average="macro")
        tuning.append({"C": c, "val_macro_f1": float(val_f1), "converged": converged})
        print(f"  C={c:<7g} val macro F1 {val_f1:.4f}{'' if converged else '  (did not converge)'}")
        if best is None or val_f1 > best[1]:  # strict >, so a tie keeps the smaller C
            best = (c, val_f1, model, converged)
    best_c, _, model, converged = best
    print(f"chosen C = {best_c:g}")

    y_true = {s: data[s][1] for s in EVAL_SPLITS}
    y_pred = {s: model.predict(data[s][0]) for s in EVAL_SPLITS}
    return {
        "model": "Baseline A: colour histogram + linear SVM",
        "features": f"joint HSV histogram, {BINS}x{BINS}x{BINS} = {BINS ** 3} bins, padding excluded, square-rooted (Hellinger)",
        "seed": seed,
        "C_grid": m["C_grid"],
        "tuning": tuning,
        "chosen_C": best_c,
        "converged": converged,
        **evaluate(y_true, y_pred, classes, y_train),
    }

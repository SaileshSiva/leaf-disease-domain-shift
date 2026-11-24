"""Baseline A: colour-histogram features + linear SVM, scored on both test sets.

    python scripts/train_svm.py             # train, score, write results/baseline_a.json
    python scripts/train_svm.py --seed 1    # another seed (the SVM's solver shuffles data)

Steps (design in the decision log, 24 Nov 2026):
    1. 512-bin HSV histogram per image, padding excluded (leafshift.features)
    2. square root of each bin (the Hellinger transform); no standardisation
    3. for each C in C_GRID: train on train, score macro F1 on val
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
all models together later
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # find leafshift when run as a script

import numpy as np  # noqa: E402
import sklearn  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import FunctionTransformer  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

from leafshift.data import load_cache_meta  # noqa: E402
from leafshift.features import BINS, split_features  # noqa: E402

RESULTS = REPO_ROOT / "results" / "baseline_a.json"
C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
MAX_ITER = 20_000
EVAL_SPLITS = ("test", "field_test")


def make_model(c: float, seed: int):
    return make_pipeline(FunctionTransformer(np.sqrt), LinearSVC(C=c, random_state=seed, max_iter=MAX_ITER))


def fit(model, X, y) -> bool:
    """Fit, and report whether the solver converged rather than hiding the warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X, y)
    return not any(issubclass(w.category, ConvergenceWarning) for w in caught)


def scores(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict:
    labels = list(range(len(classes)))
    per_class = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class_accuracy": {c: float(a) for c, a in zip(classes, per_class)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def print_confusion(name: str, matrix: list[list[int]], classes: list[str]) -> None:
    short = [c.split("___")[0][0] + ":" + c.split("___")[1][:10] for c in classes]
    print(f"\nconfusion matrix, {name} (rows = true class, columns = predicted)")
    print(" " * 14 + " ".join(f"{i:>5d}" for i in range(len(classes))))
    for i, row in enumerate(matrix):
        print(f"{i} {short[i]:<12s}" + " ".join(f"{v:>5d}" for v in row))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    classes = load_cache_meta()["classes"]

    start = time.perf_counter()
    data = {s: split_features(s) for s in ("train", "val", *EVAL_SPLITS)}
    print(f"features extracted in {time.perf_counter() - start:.0f} s: "
          + ", ".join(f"{s} {len(y)}" for s, (_, y) in data.items()))

    X_train, y_train = data["train"]
    X_val, y_val = data["val"]

    print("\ntuning C on val (macro F1)")
    tuning, best = [], None
    for c in C_GRID:
        model = make_model(c, args.seed)
        converged = fit(model, X_train, y_train)
        val_f1 = f1_score(y_val, model.predict(X_val), average="macro")
        tuning.append({"C": c, "val_macro_f1": float(val_f1), "converged": converged})
        print(f"  C={c:<7g} val macro F1 {val_f1:.4f}{'' if converged else '  (did not converge)'}")
        if best is None or val_f1 > best[1]:  # strict >, so a tie keeps the smaller C
            best = (c, val_f1, model, converged)
    best_c, _, model, converged = best
    print(f"chosen C = {best_c:g}")

    results = {}
    for split in EVAL_SPLITS:
        X, y = data[split]
        results[split] = scores(y, model.predict(X), classes)

    # Floors: what guessing would score, so the SVM's numbers have something to beat.
    majority = int(np.bincount(y_train).argmax())
    floors = {"chance_accuracy": 1 / len(classes), "majority_class": classes[majority],
              **{f"majority_accuracy_{s}": float((data[s][1] == majority).mean()) for s in EVAL_SPLITS}}

    gap = {m: results["test"][m] - results["field_test"][m] for m in ("accuracy", "macro_f1")}
    output = {
        "model": "Baseline A: colour histogram + linear SVM",
        "features": f"joint HSV histogram, {BINS}x{BINS}x{BINS} = {BINS ** 3} bins, padding excluded, square-rooted (Hellinger)",
        "seed": args.seed,
        "C_grid": C_GRID,
        "tuning": tuning,
        "chosen_C": best_c,
        "converged": converged,
        "results": results,
        "gap_in_domain_minus_field": gap,
        "floors": floors,
        "versions": {"scikit-learn": sklearn.__version__, "numpy": np.__version__},
        "splits_csv_sha256": load_cache_meta()["splits_csv_sha256"],
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(output, indent=2) + "\n")

    print(f"\n{'':22s}{'accuracy':>10s}{'macro F1':>10s}")
    for split in EVAL_SPLITS:
        print(f"{split:22s}{results[split]['accuracy']:>10.4f}{results[split]['macro_f1']:>10.4f}")
    print(f"{'gap (lab - field)':22s}{gap['accuracy']:>10.4f}{gap['macro_f1']:>10.4f}")
    print(f"\nfloors: chance {floors['chance_accuracy']:.2f}; always '{floors['majority_class']}' "
          + ", ".join(f"{s} {floors[f'majority_accuracy_{s}']:.4f}" for s in EVAL_SPLITS))

    print("\nper-class accuracy")
    for c in classes:
        print(f"  {c:40s} test {results['test']['per_class_accuracy'][c]:.3f}   "
              f"field {results['field_test']['per_class_accuracy'][c]:.3f}")
    for split in EVAL_SPLITS:
        print_confusion(split, results[split]["confusion_matrix"], classes)
    print(f"\nwrote {RESULTS.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

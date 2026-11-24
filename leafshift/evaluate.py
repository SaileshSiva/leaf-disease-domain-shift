"""Score any model's predictions on both test sets and report the lab-to-field gap.

    from leafshift.evaluate import evaluate, print_report, save_results
    evaluation = evaluate(y_true, y_pred, classes, y_train)   # dicts keyed by split
    print_report(evaluation, classes)
    save_results(path, {"model": ..., **evaluation})
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

IN_DOMAIN = "test"
FIELD = "field_test"
EVAL_SPLITS = (IN_DOMAIN, FIELD)
METRICS = ("accuracy", "macro_f1")


def score(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict:
    labels = list(range(len(classes)))
    per_class = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class_accuracy": {c: float(a) for c, a in zip(classes, per_class)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def floors(y_train: np.ndarray, y_true: dict[str, np.ndarray], classes: list[str]) -> dict:
    majority = int(np.bincount(y_train, minlength=len(classes)).argmax())
    return {"chance_accuracy": 1 / len(classes), "majority_class": classes[majority],
            **{f"majority_accuracy_{s}": float((y_true[s] == majority).mean()) for s in EVAL_SPLITS}}


def evaluate(y_true: dict[str, np.ndarray], y_pred: dict[str, np.ndarray], classes: list[str],
             y_train: np.ndarray) -> dict:
    missing = [s for s in EVAL_SPLITS if s not in y_true or s not in y_pred]
    if missing:
        raise ValueError(f"both test sets are required; missing {missing}")
    results = {s: score(y_true[s], y_pred[s], classes) for s in EVAL_SPLITS}
    gap = {m: results[IN_DOMAIN][m] - results[FIELD][m] for m in METRICS}
    return {"results": results, "gap_in_domain_minus_field": gap, "floors": floors(y_train, y_true, classes)}


def print_report(evaluation: dict, classes: list[str]) -> None:
    results, gap, fl = evaluation["results"], evaluation["gap_in_domain_minus_field"], evaluation["floors"]

    print(f"\n{'':22s}{'accuracy':>10s}{'macro F1':>10s}")
    for s in EVAL_SPLITS:
        print(f"{s:22s}{results[s]['accuracy']:>10.4f}{results[s]['macro_f1']:>10.4f}")
    print(f"{'gap (lab - field)':22s}{gap['accuracy']:>10.4f}{gap['macro_f1']:>10.4f}")
    print(f"\nfloors: chance {fl['chance_accuracy']:.2f}; always '{fl['majority_class']}' "
          + ", ".join(f"{s} {fl[f'majority_accuracy_{s}']:.4f}" for s in EVAL_SPLITS))

    print("\nper-class accuracy")
    for c in classes:
        print(f"  {c:40s} lab {results[IN_DOMAIN]['per_class_accuracy'][c]:.3f}   "
              f"field {results[FIELD]['per_class_accuracy'][c]:.3f}")

    short = [c.split("___")[0][0] + ":" + c.split("___")[1][:10] for c in classes]
    for s in EVAL_SPLITS:
        print(f"\nconfusion matrix, {s} (rows = true class, columns = predicted)")
        print(" " * 14 + " ".join(f"{i:>5d}" for i in range(len(classes))))
        for i, row in enumerate(results[s]["confusion_matrix"]):
            print(f"{i} {short[i]:<12s}" + " ".join(f"{v:>5d}" for v in row))


def save_results(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")

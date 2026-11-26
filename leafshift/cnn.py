"""Baseline B: small CNN trained from scratch, scored on both test sets.

    python scripts/train.py configs/baseline_b.toml

The test sets are never used to choose anything.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from leafshift.data import make_loader
from leafshift.evaluate import EVAL_SPLITS, evaluate


class SmallCNN(nn.Module):
    def __init__(self, channels: list[int], n_classes: int, dropout: float) -> None:
        super().__init__()
        blocks, c_in = [], 3
        for c_out in channels:
            blocks += [nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out),
                       nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            c_in = c_out
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(c_in, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(model, loader, optimiser, loss_fn, device) -> float:
    model.train()
    total, n = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimiser.zero_grad()
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimiser.step()
        total += loss.item() * len(labels)
        n += len(labels)
    return total / n


@torch.inference_mode()
def predict(model, loader, device) -> np.ndarray:
    model.eval()
    return np.concatenate([model(images.to(device)).argmax(1).cpu().numpy() for images, _ in loader])


def train_and_evaluate(cfg: dict, classes: list[str], run_dir: Path) -> dict:
    seed, m = cfg["seed"], cfg["model"]
    device = pick_device()

    train = make_loader("train", batch_size=m["batch_size"], seed=seed)
    others = {s: make_loader(s, batch_size=m["batch_size"]) for s in ("val", *EVAL_SPLITS)}
    y_val = others["val"].dataset.labels

    model = SmallCNN(m["channels"], len(classes), m["dropout"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimiser = torch.optim.AdamW(model.parameters(), lr=m["lr"], weight_decay=m["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()
    print(f"device {device}, {n_params:,} parameters, train {len(train.dataset)}, val {len(y_val)}")

    history, best = [], {"val_macro_f1": -1.0}
    print(f"\n{'epoch':>5s}{'train loss':>12s}{'val macro F1':>14s}{'seconds':>9s}")
    for epoch in range(1, m["max_epochs"] + 1):
        start = time.perf_counter()
        train_loss = train_one_epoch(model, train, optimiser, loss_fn, device)
        val_f1 = float(f1_score(y_val, predict(model, others["val"], device), average="macro"))
        seconds = time.perf_counter() - start
        history.append({"epoch": epoch, "train_loss": train_loss, "val_macro_f1": val_f1, "seconds": seconds})
        improved = val_f1 > best["val_macro_f1"]
        print(f"{epoch:>5d}{train_loss:>12.4f}{val_f1:>14.4f}{seconds:>9.0f}{'  *' if improved else ''}")
        if improved:
            best = {"epoch": epoch, "val_macro_f1": val_f1,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        elif epoch - best["epoch"] >= m["patience"]:
            print(f"no improvement for {m['patience']} epochs; stopping")
            break
    print(f"best epoch {best['epoch']}, val macro F1 {best['val_macro_f1']:.4f}")

    model.load_state_dict(best["state"])
    torch.save({"state_dict": best["state"], "model": m, "classes": classes,
                "epoch": best["epoch"]}, run_dir / "best.pt")

    y_true = {s: others[s].dataset.labels for s in EVAL_SPLITS}
    y_pred = {s: predict(model, others[s], device) for s in EVAL_SPLITS}
    return {
        "model": "Baseline B: small CNN from scratch",
        "architecture": f"{len(m['channels'])} conv blocks {m['channels']}, global average pool, "
                        f"dropout {m['dropout']}, linear",
        "parameters": n_params,
        "device": str(device),
        "seed": seed,
        "history": history,
        "best_epoch": best["epoch"],
        "best_val_macro_f1": best["val_macro_f1"],
        "epochs_run": len(history),
        **evaluate(y_true, y_pred, classes, train.dataset.labels),
    }

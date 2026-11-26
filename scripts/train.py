"""Train and evaluate one model from a config file, in its own run folder.

    python scripts/train.py configs/baseline_a.toml
    python scripts/train.py configs/baseline_a.toml --seed 1   # override the config's seed

Writes results/runs/<config name>_seed<N>_<timestamp>/ (see leafshift.run):
config.toml, log.txt, run.json and metrics.json, plus best.pt for a CNN.
Every model is scored on both test sets.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from leafshift import cnn, svm
from leafshift.config import load_config
from leafshift.data import load_cache_meta
from leafshift.evaluate import print_report, save_results
from leafshift.run import seed_everything, start_run

TRAINERS = {"svm": svm.train_and_evaluate, "cnn": cnn.train_and_evaluate, "effnet_b0": cnn.train_and_evaluate}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--seed", type=int, help="overrides the seed in the config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    classes = load_cache_meta()["classes"]

    with start_run(cfg, args.config) as run_dir:
        seed_everything(cfg["seed"])
        metrics = TRAINERS[cfg["model"]["type"]](cfg, classes, run_dir)
        save_results(run_dir / "metrics.json", metrics)
        print_report(metrics, classes)


if __name__ == "__main__":
    main()

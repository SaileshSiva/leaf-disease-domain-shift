"""Starting a run: seeding, and a folder that records what the run was.

    from leafshift.run import seed_everything, start_run
    with start_run(cfg, config_path) as run_dir:
        seed_everything(cfg["seed"])
        ...                                   # train and evaluate; print freely
        save_results(run_dir / "metrics.json", metrics)

Each run gets results/runs/<config name>_seed<N>_<YYYYMMDD-HHMMSS>/, never
reused, holding:
    config.toml   the config file, copied byte for byte
    log.txt       everything the run printed, warnings and errors included
    run.json      seed actually used, start and end time, status, git commit
                  (and whether there were uncommitted changes), command,
                  library versions and the splits hash
    metrics.json  written by the training script
The seed in run.json is the one used; it differs from config.toml when the
command line overrode it.
"""
from __future__ import annotations

import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from leafshift.data import REPO_ROOT, load_cache_meta

RUNS_DIR = REPO_ROOT / "results" / "runs"
PACKAGES = ("numpy", "pandas", "Pillow", "scikit-learn", "torch", "torchvision")


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, numpy and torch (CPU, CUDA and MPS: torch.manual_seed covers all three).

    deterministic=True also asks torch for deterministic kernels. Ops without a
    deterministic version only warn, so a run never fails for this; the GPU can
    still differ from the CPU in the last digits. CUBLAS_WORKSPACE_CONFIG is
    what CUDA needs for deterministic matrix products (PyTorch reproducibility
    notes); it has no effect on a Mac.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)
        return len(text)

    def flush(self):
        for s in self.streams:
            s.flush()


@contextmanager
def start_run(cfg: dict, config_path: str | Path, runs_dir: Path = RUNS_DIR):
    started = datetime.now().astimezone()
    run_dir = runs_dir / f"{cfg['name']}_seed{cfg['seed']}_{started:%Y%m%d-%H%M%S}"
    info = {
        "name": cfg["name"],
        "seed": cfg["seed"],
        "config": cfg,
        "command": " ".join([Path(sys.executable).name, *sys.argv]),
        "started": started.isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_uncommitted_changes": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "splits_csv_sha256": load_cache_meta()["splits_csv_sha256"],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {p: version(p) for p in PACKAGES},
    }
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config_path, run_dir / "config.toml")

    clock = time.perf_counter()
    stdout, stderr = sys.stdout, sys.stderr
    with (run_dir / "log.txt").open("w") as log:
        sys.stdout, sys.stderr = _Tee(stdout, log), _Tee(stderr, log)
        print(f"run folder: {run_dir.relative_to(REPO_ROOT) if run_dir.is_relative_to(REPO_ROOT) else run_dir}")
        try:
            yield run_dir
            info["status"] = "finished"
        except BaseException:
            info["status"] = "failed"
            traceback.print_exc(file=log)  # the terminal gets it once, when the error reaches the top
            raise
        finally:
            sys.stdout, sys.stderr = stdout, stderr
            info["finished"] = datetime.now().astimezone().isoformat(timespec="seconds")
            info["duration_s"] = round(time.perf_counter() - clock, 1)
            (run_dir / "run.json").write_text(json.dumps(info, indent=2) + "\n")

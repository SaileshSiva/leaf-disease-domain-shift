"""Load and check a training config (TOML).

    from leafshift.config import load_config
    cfg = load_config("configs/baseline_a.toml")

    python -m leafshift.config configs/baseline_a.toml   # check a file and print it

Every config is checked before anything runs: a missing key, an unknown key
(usually a typo) or a value of the wrong type stops the run with a message,
rather than being ignored or filled in with a default. The run's name is the
file name, so the two cannot disagree.
"""
from __future__ import annotations

import sys
import tomllib
from pathlib import Path

NUMBER = (int, float)

TOP_LEVEL = {"seed": int, "model": dict}
MODELS = {
    "svm": {"type": str, "C_grid": [NUMBER], "max_iter": int},
    "cnn": {"type": str, "channels": [int], "dropout": NUMBER, "lr": NUMBER, "weight_decay": NUMBER,
            "batch_size": int, "max_epochs": int, "patience": int},
}


class ConfigError(ValueError):
    pass


def _type_ok(value, expected) -> bool:
    if isinstance(expected, list):
        return isinstance(value, list) and len(value) > 0 and all(_type_ok(v, expected[0]) for v in value)
    if isinstance(value, bool):  # bool is a subclass of int; true is never a valid seed or count
        return expected is bool
    return isinstance(value, expected)


def _check(section: dict, spec: dict, where: str) -> None:
    missing = sorted(set(spec) - set(section))
    unknown = sorted(set(section) - set(spec))
    if missing or unknown:
        problems = ([f"missing {missing}"] if missing else []) + ([f"unknown {unknown}"] if unknown else [])
        raise ConfigError(f"{where}: {', '.join(problems)}")
    for key, expected in spec.items():
        if not _type_ok(section[key], expected):
            raise ConfigError(f"{where}: {key} = {section[key]!r} is not {_describe(expected)}")


def _describe(expected) -> str:
    if isinstance(expected, list):
        return f"a non-empty list, each {_describe(expected[0])}"
    if expected is NUMBER:
        return "a number"
    return expected.__name__


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open("rb") as f:
        cfg = tomllib.load(f)
    _check(cfg, TOP_LEVEL, path.name)
    model_type = cfg["model"].get("type")
    if model_type not in MODELS:
        raise ConfigError(f"{path.name}: model.type {model_type!r} is not one of {sorted(MODELS)}")
    _check(cfg["model"], MODELS[model_type], f"{path.name}: model")
    return {"name": path.stem, **cfg}


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print(load_config(arg))

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
    "effnet_b0": {"type": str, "pretrained": bool, "freeze_backbone": bool, "dropout": NUMBER, "lr": NUMBER,
                  "weight_decay": NUMBER, "batch_size": int, "max_epochs": int, "patience": int},
}
# Optional section: absent means no augmentation. CNN models only.
AUGMENT = {"crop_scale": [NUMBER], "crop_ratio": [NUMBER], "hflip": NUMBER, "vflip": NUMBER,
           "rotation": NUMBER, "brightness": NUMBER, "contrast": NUMBER, "saturation": NUMBER, "hue": NUMBER,
           "erase_p": NUMBER, "erase_scale": [NUMBER], "erase_ratio": [NUMBER]}


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
    _check({k: v for k, v in cfg.items() if k != "augment"}, TOP_LEVEL, path.name)
    model_type = cfg["model"].get("type")
    if model_type not in MODELS:
        raise ConfigError(f"{path.name}: model.type {model_type!r} is not one of {sorted(MODELS)}")
    _check(cfg["model"], MODELS[model_type], f"{path.name}: model")
    if cfg["model"].get("freeze_backbone") and not cfg["model"]["pretrained"]:
        raise ConfigError(f"{path.name}: model: freeze_backbone = true needs pretrained = true")
    if "augment" in cfg:
        _check_augment(cfg["augment"], model_type, f"{path.name}: augment")
    return {"name": path.stem, **cfg}


def _check_augment(a, model_type: str, where: str) -> None:
    if model_type == "svm":
        raise ConfigError(f"{where}: the svm model takes no augmentation")
    if not isinstance(a, dict):
        raise ConfigError(f"{where}: must be a [augment] section")
    _check(a, AUGMENT, where)
    for key in ("crop_scale", "crop_ratio", "erase_scale", "erase_ratio"):
        if len(a[key]) != 2 or not 0 < a[key][0] <= a[key][1]:
            raise ConfigError(f"{where}: {key} = {a[key]!r} is not [low, high] with 0 < low <= high")
    for key in ("hflip", "vflip", "erase_p"):
        if not 0 <= a[key] <= 1:
            raise ConfigError(f"{where}: {key} = {a[key]!r} is not a probability")
    if not 0 <= a["hue"] <= 0.5:
        raise ConfigError(f"{where}: hue = {a['hue']!r} is not between 0 and 0.5")
    for key in ("rotation", "brightness", "contrast", "saturation"):
        if a[key] < 0:
            raise ConfigError(f"{where}: {key} = {a[key]!r} is negative")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print(load_config(arg))

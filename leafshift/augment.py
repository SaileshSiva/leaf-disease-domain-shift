"""Training-time augmentation, built from a config's [augment] section.

Works on uint8 (3, H, W) tensors, before ImageNet normalisation (see data.py).
Rotation corners and erased patches take the cache's padding colour.
"""
from __future__ import annotations

from torchvision.transforms import v2

from leafshift.data import load_cache_meta


def build_transform(a: dict) -> v2.Compose:
    meta = load_cache_meta()
    fill = list(meta["pad_colour"])
    return v2.Compose([
        v2.RandomResizedCrop(meta["size"], scale=tuple(a["crop_scale"]), ratio=tuple(a["crop_ratio"]),
                             antialias=True),
        v2.RandomHorizontalFlip(a["hflip"]),
        v2.RandomVerticalFlip(a["vflip"]),
        v2.RandomRotation(a["rotation"], fill=fill),
        v2.ColorJitter(a["brightness"], a["contrast"], a["saturation"], a["hue"]),
        v2.RandomErasing(a["erase_p"], scale=tuple(a["erase_scale"]), ratio=tuple(a["erase_ratio"]),
                         value=fill),
    ])


def describe(a: dict) -> str:
    return (f"crop scale {a['crop_scale']}, flips h{a['hflip']}/v{a['vflip']}, rotation ±{a['rotation']}°, "
            f"jitter b{a['brightness']}/c{a['contrast']}/s{a['saturation']}/h{a['hue']}, "
            f"erasing p{a['erase_p']}")

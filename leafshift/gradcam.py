"""Grad-CAM (Selvaraju et al., IJCV 2020) for the predicted class, on the output of model.features.

    maps, preds = gradcam(model, images)   # maps: (B, H, W) in [0, 1], one per image
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def gradcam(model: nn.Module, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    store = {}

    def keep(_module, _inputs, output):
        store["act"] = output
        output.register_hook(lambda grad: store.__setitem__("grad", grad))

    handle = model.features.register_forward_hook(keep)
    try:
        with torch.enable_grad():
            logits = model(images)
            preds = logits.argmax(1)
            model.zero_grad()
            logits.gather(1, preds[:, None]).sum().backward()
    finally:
        handle.remove()

    weights = store["grad"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * store["act"]).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=images.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    peak = cam.flatten(1).max(dim=1).values.clamp_min(1e-12)
    return (cam / peak[:, None, None]).detach(), preds.detach()

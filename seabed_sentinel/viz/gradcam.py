"""Grad-CAM for the detector's class heat-map head.

Explains *which pixels* drove a detection: for sonar this typically reveals
whether the network used the acoustic highlight, the shadow, or (undesirably)
background texture. Useful evidence when presenting ATR output to an operator.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: torch.nn.Module, layer: Optional[torch.nn.Module] = None):
        self.model = model
        self.layer = layer or model.decoder
        self._act = None
        self._grad = None
        self.layer.register_forward_hook(self._fwd)

    def _fwd(self, m, i, o):
        self._act = o
        o.register_hook(self._bwd)

    def _bwd(self, g):
        self._grad = g

    def __call__(self, image: torch.Tensor, class_id: int, center_xy: Tuple[int, int], radius: int = 2) -> np.ndarray:
        """image: [1,1,H,W]. Returns a [H,W] CAM in [0,1] for the given class at the given
        heat-map centre (input-pixel coordinates)."""
        self.model.eval()
        self.model.zero_grad()
        out = self.model(image.requires_grad_(True))
        hm = out["hm"][0, class_id]
        cx, cy = center_xy[0] // self.model.stride, center_xy[1] // self.model.stride
        y0, y1 = max(cy - radius, 0), min(cy + radius + 1, hm.shape[0])
        x0, x1 = max(cx - radius, 0), min(cx + radius + 1, hm.shape[1])
        score = hm[y0:y1, x0:x1].sum()
        score.backward()
        w = self._grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self._act).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam.detach().cpu().numpy()
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)

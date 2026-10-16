"""Anchor-free CenterNet-style sonar target detector with an auxiliary
highlight/shadow segmentation head and MC-dropout uncertainty.

Outputs (all at stride 4):
    hm  : [B, C, H, W]  per-class centre heat-map logits
    wh  : [B, 2, H, W]  box width / height (feature-map units)
    reg : [B, 2, H, W]  sub-pixel centre offset
    seg : [B, 3, H, W]  seabed / highlight / shadow logits (optional)

The shadow head is more than a regulariser: acoustic shadow carries most of
the shape information in SAS target recognition, and supervising it
explicitly forces the shared features to encode it.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import ConvBNAct, Encoder, FPNDecoder


class SonarCenterNet(nn.Module):
    def __init__(self, num_classes: int = 6, width: float = 1.0, fpn_ch: int = 64, aux_seg: bool = True,
                 dropout: float = 0.1, in_ch: int = 1):
        super().__init__()
        self.num_classes = num_classes
        self.stride = 4
        self.encoder = Encoder(in_ch, width)
        self.decoder = FPNDecoder(self.encoder.out_channels, fpn_ch)
        # one shared 3x3 "neck" followed by cheap 1x1 task heads keeps CPU cost low
        self.neck = nn.Sequential(ConvBNAct(fpn_ch, fpn_ch, 3), nn.Dropout2d(dropout) if dropout > 0 else nn.Identity())
        self.hm = nn.Conv2d(fpn_ch, num_classes, 1)
        self.wh = nn.Conv2d(fpn_ch, 2, 1)
        self.reg = nn.Conv2d(fpn_ch, 2, 1)
        self.seg = nn.Conv2d(fpn_ch, 3, 1) if aux_seg else None
        nn.init.constant_(self.hm.bias, -2.19)   # prior p ~ 0.1 for the focal loss

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.neck(self.decoder(self.encoder(x)))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        f = self.features(x)
        out = {"hm": self.hm(f), "wh": self.wh(f), "reg": self.reg(f)}
        if self.seg is not None:
            out["seg"] = self.seg(f)
        return out

    @torch.no_grad()
    def forward_mc(self, x: torch.Tensor, n: int = 8) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Monte-Carlo dropout: returns mean outputs and the per-pixel std of the
        max-class heat-map probability (epistemic uncertainty proxy)."""
        was_training = self.training
        self.eval()
        for m in self.modules():
            if isinstance(m, nn.Dropout2d):
                m.train()
        base = self.decoder(self.encoder(x))
        hms, whs, regs = [], [], []
        for _ in range(n):
            f = self.neck(base)
            hms.append(torch.sigmoid(self.hm(f)))
            whs.append(self.wh(f))
            regs.append(self.reg(f))
        hm = torch.stack(hms)
        out = {"hm_prob": hm.mean(0), "wh": torch.stack(whs).mean(0), "reg": torch.stack(regs).mean(0)}
        unc = hm.max(2).values.std(0)  # [B, H, W]
        self.train(was_training)
        return out, unc


def _nms_heat(heat: torch.Tensor, k: int = 3) -> torch.Tensor:
    hmax = F.max_pool2d(heat, k, stride=1, padding=k // 2)
    return heat * (hmax == heat).float()


def decode_detections(out: Dict[str, torch.Tensor], stride: int = 4, topk: int = 50, score_thresh: float = 0.1,
                      already_prob: bool = False) -> List[Dict[str, torch.Tensor]]:
    """Decode network outputs into per-image detections.

    Returns a list (len B) of dicts with ``boxes`` [N,4] (x0,y0,x1,y1 in input
    pixels), ``scores`` [N] and ``labels`` [N].
    """
    heat = out["hm_prob"] if already_prob else torch.sigmoid(out["hm"])
    heat = _nms_heat(heat)
    B, C, H, W = heat.shape
    results = []
    for b in range(B):
        scores, inds = heat[b].reshape(-1).topk(min(topk, C * H * W))
        keep = scores >= score_thresh
        scores, inds = scores[keep], inds[keep]
        labels = inds // (H * W)
        pix = inds % (H * W)
        ys, xs = (pix // W).float(), (pix % W).float()
        wh = out["wh"][b].reshape(2, -1)[:, pix]
        reg = out["reg"][b].reshape(2, -1)[:, pix]
        cx, cy = (xs + reg[0]) * stride, (ys + reg[1]) * stride
        w, h = wh[0].clamp(min=0) * stride, wh[1].clamp(min=0) * stride
        boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
        results.append({"boxes": boxes, "scores": scores, "labels": labels})
    return results

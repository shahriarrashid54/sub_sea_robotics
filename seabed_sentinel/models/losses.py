"""Losses for the CenterNet-style detector."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss(pred_logits: torch.Tensor, gt: torch.Tensor, alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    """Penalty-reduced pixel-wise focal loss (CenterNet Eq. 1)."""
    pred = torch.sigmoid(pred_logits).clamp(1e-4, 1 - 1e-4)
    pos = gt.eq(1).float()
    neg = 1 - pos
    neg_w = torch.pow(1 - gt, beta)
    pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos
    neg_loss = torch.log(1 - pred) * torch.pow(pred, alpha) * neg_w * neg
    num_pos = pos.sum()
    if num_pos == 0:
        return -neg_loss.sum()
    return -(pos_loss.sum() + neg_loss.sum()) / num_pos


class DetectionLoss(nn.Module):
    def __init__(self, w_hm: float = 1.0, w_wh: float = 0.1, w_reg: float = 1.0, w_seg: float = 0.5,
                 seg_class_weights=(0.3, 1.0, 1.0)):
        super().__init__()
        self.w_hm, self.w_wh, self.w_reg, self.w_seg = w_hm, w_wh, w_reg, w_seg
        self.register_buffer("seg_w", torch.tensor(seg_class_weights, dtype=torch.float32))

    def forward(self, out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        l_hm = focal_loss(out["hm"], batch["hm"])
        mask = batch["reg_mask"]
        n = mask.sum().clamp(min=1.0)
        l_wh = (F.l1_loss(out["wh"], batch["wh"], reduction="none") * mask).sum() / n
        l_reg = (F.l1_loss(out["reg"], batch["reg"], reduction="none") * mask).sum() / n
        total = self.w_hm * l_hm + self.w_wh * l_wh + self.w_reg * l_reg
        losses = {"hm": l_hm, "wh": l_wh, "reg": l_reg}
        if "seg" in out and "seg" in batch:
            l_seg = F.cross_entropy(out["seg"], batch["seg"], weight=self.seg_w)
            total = total + self.w_seg * l_seg
            losses["seg"] = l_seg
        losses["total"] = total
        return losses

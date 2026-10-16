"""Self-supervised pre-training of the encoder with a masked auto-encoder (MAE)
objective on *unlabelled* sonar tiles.

Why this matters for sonar: labelled SAS imagery is scarce (and often
classified) while unlabelled survey data is abundant. Pre-training the
encoder to reconstruct masked patches teaches it seabed texture, shadow
geometry and speckle statistics before any label is seen, improving
label-efficiency of the downstream detector.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.dataset import SonarDetectionDataset
from .backbone import ConvBNAct, Encoder, FPNDecoder


@dataclass
class MAEConfig:
    data_root: str = "data/sas_synth"
    out_dir: str = "runs/mae"
    epochs: int = 6
    batch_size: int = 16
    lr: float = 2e-3
    width: float = 1.0
    fpn_ch: int = 64
    crop: int = 256
    patch: int = 16
    mask_ratio: float = 0.6
    num_workers: int = 3
    seed: int = 0
    threads: Optional[int] = None


class MaskedAutoencoder(nn.Module):
    """Convolutional MAE: masked patches are zeroed and flagged with a mask
    channel; the encoder/FPN features are decoded back to the full image."""

    def __init__(self, width: float = 1.0, fpn_ch: int = 64):
        super().__init__()
        self.encoder = Encoder(in_ch=1, width=width)
        self.fpn = FPNDecoder(self.encoder.out_channels, fpn_ch)
        self.decoder = nn.Sequential(
            ConvBNAct(fpn_ch, fpn_ch, 3),
            nn.Upsample(scale_factor=2, mode="nearest"), ConvBNAct(fpn_ch, fpn_ch // 2, 3),
            nn.Upsample(scale_factor=2, mode="nearest"), ConvBNAct(fpn_ch // 2, fpn_ch // 4, 3),
            nn.Conv2d(fpn_ch // 4, 1, 1),
        )

    def forward(self, x_masked: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.fpn(self.encoder(x_masked)))


def random_patch_mask(shape, patch: int, ratio: float, device) -> torch.Tensor:
    B, _, H, W = shape
    gh, gw = H // patch, W // patch
    m = (torch.rand(B, 1, gh, gw, device=device) < ratio).float()
    return F.interpolate(m, scale_factor=patch, mode="nearest")


def pretrain_mae(cfg: MAEConfig) -> Dict:
    torch.manual_seed(cfg.seed)
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    os.makedirs(cfg.out_dir, exist_ok=True)
    ds = SonarDetectionDataset(os.path.join(cfg.data_root, "train"), crop=cfg.crop, train=True, with_masks=False)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True,
                        collate_fn=lambda b: torch.stack([x["image"] for x in b]))
    model = MaskedAutoencoder(cfg.width, cfg.fpn_ch)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.epochs * len(loader), pct_start=0.2)
    hist = []
    t_start = time.time()
    for ep in range(1, cfg.epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for x in loader:
            m = random_patch_mask(x.shape, cfg.patch, cfg.mask_ratio, x.device)
            x_in = x * (1 - m)
            rec = model(x_in)
            loss = (F.mse_loss(rec, x, reduction="none") * m).sum() / m.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
            nb += 1
        hist.append({"epoch": ep, "loss": tot / nb})
        print(f"[mae] epoch {ep}/{cfg.epochs} masked-MSE {tot / nb:.4f}", flush=True)
    torch.save({"encoder": model.encoder.state_dict(), "config": asdict(cfg)}, os.path.join(cfg.out_dir, "encoder.pt"))
    torch.save(model.state_dict(), os.path.join(cfg.out_dir, "mae_full.pt"))
    summary = {"history": hist, "time_min": (time.time() - t_start) / 60}
    with open(os.path.join(cfg.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


@torch.no_grad()
def reconstruct_examples(model_path: str, data_root: str, n: int = 4, patch: int = 16, ratio: float = 0.6,
                         width: float = 1.0, fpn_ch: int = 64, seed: int = 0):
    """Return (originals, masked inputs, reconstructions) for visualisation."""
    model = MaskedAutoencoder(width, fpn_ch)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    ds = SonarDetectionDataset(os.path.join(data_root, "val"), crop=256, train=False, with_masks=False)
    torch.manual_seed(seed)
    x = torch.stack([ds[i]["image"] for i in range(n)])
    m = random_patch_mask(x.shape, patch, ratio, x.device)
    rec = model(x * (1 - m))
    return x.numpy()[:, 0], (x * (1 - m)).numpy()[:, 0], rec.clamp(0, 1).numpy()[:, 0], m.numpy()[:, 0]

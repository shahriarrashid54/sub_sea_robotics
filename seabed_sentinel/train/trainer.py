"""Training loop (CPU-friendly) with CSV logging and best-checkpoint selection."""
from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.dataset import SonarDetectionDataset, collate_fn
from ..models.detector import SonarCenterNet
from ..models.losses import DetectionLoss
from .evaluate import compute_map, predict_dataset


@dataclass
class TrainConfig:
    data_root: str = "data/sas_synth"
    out_dir: str = "runs/detector"
    epochs: int = 30
    batch_size: int = 16
    lr: float = 2e-3
    weight_decay: float = 1e-4
    width: float = 1.0
    fpn_ch: int = 64
    aux_seg: bool = True
    dropout: float = 0.1
    crop: int = 256
    box_mode: str = "highlight"
    label_fraction: float = 1.0
    max_train_images: Optional[int] = None
    num_workers: int = 3
    seed: int = 0
    device: str = "cpu"
    init_from: Optional[str] = None       # path to encoder weights (e.g. MAE pre-training)
    eval_every: int = 1
    threads: Optional[int] = None
    loss_weights: Dict[str, float] = field(default_factory=lambda: {"w_hm": 1.0, "w_wh": 0.1, "w_reg": 1.0, "w_seg": 0.5})

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        import yaml
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        return cls(**d)


def build_model(cfg: TrainConfig, num_classes: int) -> SonarCenterNet:
    model = SonarCenterNet(num_classes=num_classes, width=cfg.width, fpn_ch=cfg.fpn_ch, aux_seg=cfg.aux_seg,
                           dropout=cfg.dropout)
    if cfg.init_from:
        sd = torch.load(cfg.init_from, map_location="cpu")
        sd = sd.get("encoder", sd)
        missing, unexpected = model.encoder.load_state_dict(sd, strict=False)
        print(f"[init] loaded encoder weights from {cfg.init_from} (missing={len(missing)}, unexpected={len(unexpected)})")
    return model


def train(cfg: TrainConfig) -> Dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    train_ds = SonarDetectionDataset(os.path.join(cfg.data_root, "train"), crop=cfg.crop, train=True,
                                     label_fraction=cfg.label_fraction, seed=cfg.seed, box_mode=cfg.box_mode,
                                     max_images=cfg.max_train_images)
    val_ds = SonarDetectionDataset(os.path.join(cfg.data_root, "val"), crop=None, train=False, box_mode=cfg.box_mode)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                              collate_fn=collate_fn, drop_last=True, persistent_workers=cfg.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                            collate_fn=collate_fn)
    model = build_model(cfg, train_ds.num_classes).to(cfg.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_params / 1e6:.2f} M parameters | train images {len(train_ds)} | val images {len(val_ds)}")
    crit = DetectionLoss(**cfg.loss_weights).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=max(steps, 1), pct_start=0.15,
                                                div_factor=20, final_div_factor=50)
    log_path = os.path.join(cfg.out_dir, "log.csv")
    history, best = [], -1.0
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "hm", "wh", "reg", "seg", "val_mAP50", "lr", "time_s"])
    t_start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        agg = {"total": 0.0, "hm": 0.0, "wh": 0.0, "reg": 0.0, "seg": 0.0}
        nb = 0
        for batch in train_loader:
            batch = {k: (v.to(cfg.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = model(batch["image"])
            losses = crit(out, batch)
            opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            sched.step()
            for k in agg:
                if k in losses:
                    agg[k] += float(losses[k].detach())
            nb += 1
        for k in agg:
            agg[k] /= max(nb, 1)
        val_map = float("nan")
        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            preds, gts, _ = predict_dataset(model, val_loader, device=cfg.device, desc="val")
            val_map = compute_map(preds, gts, train_ds.num_classes)["mAP@0.50"]
            if val_map > best:
                best = val_map
                torch.save({"model": model.state_dict(), "config": asdict(cfg), "classes": train_ds.classes,
                            "epoch": epoch, "val_mAP50": val_map}, os.path.join(cfg.out_dir, "best.pt"))
        dt = time.time() - t0
        row = [epoch, agg["total"], agg["hm"], agg["wh"], agg["reg"], agg["seg"], val_map, sched.get_last_lr()[0], dt]
        history.append(dict(zip(["epoch", "train_loss", "hm", "wh", "reg", "seg", "val_mAP50", "lr", "time_s"], row)))
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([f"{v:.5g}" if isinstance(v, float) else v for v in row])
        print(f"epoch {epoch:3d}/{cfg.epochs} loss {agg['total']:.3f} (hm {agg['hm']:.3f} wh {agg['wh']:.3f} "
              f"reg {agg['reg']:.3f} seg {agg['seg']:.3f}) val mAP50 {val_map:.3f} [{dt:.0f}s]", flush=True)
    torch.save({"model": model.state_dict(), "config": asdict(cfg), "classes": train_ds.classes, "epoch": cfg.epochs},
               os.path.join(cfg.out_dir, "last.pt"))
    summary = {"best_val_mAP50": best, "epochs": cfg.epochs, "params_M": n_params / 1e6,
               "train_time_min": (time.time() - t_start) / 60, "history": history}
    with open(os.path.join(cfg.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def load_checkpoint(path: str, device: str = "cpu") -> SonarCenterNet:
    ck = torch.load(path, map_location=device)
    cfg = TrainConfig(**{k: v for k, v in ck["config"].items() if k in TrainConfig.__dataclass_fields__})
    cfg.init_from = None
    model = build_model(cfg, len(ck["classes"]))
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model

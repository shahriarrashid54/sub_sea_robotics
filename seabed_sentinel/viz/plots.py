"""Plotting helpers (matplotlib, Agg backend)."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from ..sim.scene import CLASSES, MILCO_CLASSES  # noqa: E402

CLASS_COLOURS = {"cylinder": "#ff4d4d", "truncated_cone": "#ff9f1c", "wedge": "#ffe066", "sphere": "#ff6bd6",
                 "rock": "#4dd2ff", "debris": "#7cff7c"}


def sonar_cmap():
    """Copper-like 'sonar' colour map similar to those used in survey software."""
    return LinearSegmentedColormap.from_list("sonar", ["#000000", "#2b1a0a", "#7a4a14", "#c98a2e", "#f3d58a", "#ffffff"])


def plot_detections(ax, image: np.ndarray, dets: Optional[Dict] = None, gts: Optional[Dict] = None,
                    class_names: Sequence[str] = CLASSES, score_thr: float = 0.3, pixel_m: Optional[float] = None,
                    range_start_m: Optional[float] = None, cmap="gray", title: Optional[str] = None):
    H, W = image.shape
    extent = None
    sx = sy = 1.0
    x0 = 0.0
    if pixel_m is not None:
        x0 = range_start_m or 0.0
        extent = [x0, x0 + W * pixel_m, 0, H * pixel_m]
        sx = sy = pixel_m
    ax.imshow(image, cmap=cmap, vmin=0, vmax=1, origin="lower", extent=extent, interpolation="nearest")
    if gts is not None:
        for b in gts["boxes"]:
            bx0, by0, bx1, by1 = b
            ax.add_patch(plt.Rectangle((x0 + bx0 * sx, by0 * sy), (bx1 - bx0) * sx, (by1 - by0) * sy, fill=False,
                                       ec="white", lw=1.0, ls="--"))
    if dets is not None:
        for b, s, l in zip(dets["boxes"], dets["scores"], dets["labels"]):
            if s < score_thr:
                continue
            name = class_names[int(l)]
            col = CLASS_COLOURS.get(name, "lime")
            bx0, by0, bx1, by1 = b
            ax.add_patch(plt.Rectangle((x0 + bx0 * sx, by0 * sy), (bx1 - bx0) * sx, (by1 - by0) * sy, fill=False, ec=col, lw=1.6))
            ax.text(x0 + bx0 * sx, by1 * sy + 0.5 * sy, f"{name} {s:.2f}", color=col, fontsize=7, va="bottom",
                    bbox=dict(facecolor="black", alpha=0.5, pad=1, lw=0))
    if pixel_m is not None:
        ax.set_xlabel("ground range [m]")
        ax.set_ylabel("along-track [m]")
    else:
        ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)


def plot_pd_fa(ax, curves: Dict[str, Dict], title: str = "Detection performance"):
    for name, c in curves.items():
        fa = np.asarray(c["fa_per_km2"]); pd_ = np.asarray(c["pd"])
        ax.plot(fa, pd_, marker="o", ms=3, label=name)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel("false alarms per km$^2$")
    ax.set_ylabel("probability of detection")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    ax.set_title(title, fontsize=10)


def plot_confusion(ax, cm: np.ndarray, class_names: Sequence[str] = CLASSES, normalise: bool = True, title="Confusion"):
    cm = np.asarray(cm, dtype=float)
    if normalise:
        cm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalise else None)
    ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8); ax.set_yticklabels(class_names, fontsize=8)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.2f}" if normalise else f"{int(cm[i, j])}", ha="center", va="center",
                    fontsize=7, color="white" if cm[i, j] > 0.5 * (1 if normalise else cm.max()) else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)
    n_m = len([c for c in class_names if c in MILCO_CLASSES])
    ax.axhline(n_m - 0.5, color="red", lw=1); ax.axvline(n_m - 0.5, color="red", lw=1)
    return im


def plot_breakdowns(axes, metrics: Dict):
    items = [("by_range", "slant range [m]"), ("by_grazing", "grazing angle [deg]"),
             ("by_burial", "burial fraction"), ("by_aspect", "aspect angle [deg]")]
    for ax, (key, label) in zip(axes, items):
        d = metrics[key]
        edges = d["edges"]
        names = [f"{edges[i]:g}-{edges[i + 1]:g}" for i in range(len(edges) - 1)]
        rec = np.array(d["recall"]); cnt = np.array(d["count"])
        bars = ax.bar(names, rec, color="#3b7dd8")
        for b, n in zip(bars, cnt):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"n={n}", ha="center", fontsize=7)
        ax.set_ylim(0, 1.1); ax.set_ylabel("recall"); ax.set_xlabel(label)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="y", alpha=0.3)


def plot_training_curves(ax, history: List[Dict], label: str = ""):
    ep = [h["epoch"] for h in history]
    ax.plot(ep, [h["train_loss"] for h in history], label=f"train loss {label}".strip())
    ax2 = ax.twinx()
    ax2.plot(ep, [h["val_mAP50"] for h in history], color="tab:orange", marker="o", ms=3, label="val mAP@0.5")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax2.set_ylabel("val mAP@0.5")
    ax2.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")


def plot_calibration(ax, cal: Dict):
    conf = np.array(cal["bin_conf"], dtype=float); acc = np.array(cal["bin_acc"], dtype=float)
    ok = ~np.isnan(conf)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.plot(conf[ok], acc[ok], marker="o", color="#d62728")
    ax.set_xlabel("detection score"); ax.set_ylabel("fraction correct (TP & class)")
    ax.set_title(f"Reliability (ECE = {cal['ece']:.3f})", fontsize=10)
    ax.grid(alpha=0.3); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

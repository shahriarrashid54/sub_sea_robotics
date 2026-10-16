"""Evaluation suite with mine-countermeasures (MCM) standard metrics.

Besides COCO-style mAP, the suite reports what MCM operators actually care
about:

* **PD vs. FA/km^2** -- probability of detection against false alarms per
  square kilometre of surveyed seabed, computed from the known pixel size.
* **MILCO vs. NOMBO confusion** -- the classification error that drives
  re-acquisition cost (a rock called a mine) versus risk (a mine called a rock).
* **Breakdowns** by range, grazing angle, burial fraction and aspect angle.
* **Calibration** (expected calibration error) of the detection score.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.detector import decode_detections
from ..sim.scene import CLASSES, MILCO_CLASSES

# ---------------------------------------------------------------------------------
# basic geometry
# ---------------------------------------------------------------------------------

def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between [N,4] and [M,4] boxes in x0,y0,x1,y1."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix0 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy0 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix1 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.clip(aa[:, None] + ab[None, :] - inter, 1e-9, None)


def xywh_to_xyxy(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    return np.stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]], 1)


# ---------------------------------------------------------------------------------
# matching
# ---------------------------------------------------------------------------------

def match_image(pred_boxes, pred_scores, gt_boxes, iou_thr: float, pred_labels=None, gt_labels=None):
    """Greedy score-ordered matching. Returns (tp flags per pred, matched gt index per pred)."""
    order = np.argsort(-pred_scores)
    matched = np.zeros(len(gt_boxes), dtype=bool)
    tp = np.zeros(len(pred_boxes), dtype=bool)
    assign = np.full(len(pred_boxes), -1, dtype=int)
    if len(gt_boxes) == 0:
        return tp, assign
    ious = box_iou(pred_boxes, gt_boxes)
    for i in order:
        cand = ious[i].copy()
        if pred_labels is not None and gt_labels is not None:
            cand[gt_labels != pred_labels[i]] = -1
        cand[matched] = -1
        j = int(np.argmax(cand))
        if cand[j] >= iou_thr:
            matched[j] = True
            tp[i] = True
            assign[i] = j
    return tp, assign


def average_precision(tp: np.ndarray, scores: np.ndarray, n_gt: int) -> Tuple[float, np.ndarray, np.ndarray]:
    if n_gt == 0:
        return float("nan"), np.array([]), np.array([])
    if len(scores) == 0:
        return 0.0, np.array([0.0]), np.array([0.0])
    order = np.argsort(-scores)
    tp = tp[order].astype(np.float64)
    fp = 1 - tp
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap, recall, precision


def compute_map(preds: Dict[int, Dict], gts: Dict[int, Dict], num_classes: int,
                iou_thrs: Sequence[float] = (0.5,)) -> Dict:
    """preds/gts: image_id -> {boxes [N,4] xyxy, scores, labels}. Returns per-class AP and mAP."""
    out = {}
    for thr in iou_thrs:
        aps = []
        per_class = {}
        for c in range(num_classes):
            tps, scs, n_gt = [], [], 0
            for img_id, g in gts.items():
                gmask = g["labels"] == c
                n_gt += int(gmask.sum())
                p = preds.get(img_id, {"boxes": np.zeros((0, 4)), "scores": np.zeros(0), "labels": np.zeros(0, int)})
                pmask = p["labels"] == c
                tp, _ = match_image(p["boxes"][pmask], p["scores"][pmask], g["boxes"][gmask], thr)
                tps.append(tp)
                scs.append(p["scores"][pmask])
            ap, _, _ = average_precision(np.concatenate(tps) if tps else np.array([]),
                                         np.concatenate(scs) if scs else np.array([]), n_gt)
            per_class[CLASSES[c] if c < len(CLASSES) else str(c)] = ap
            if n_gt > 0:
                aps.append(ap)
        out[f"mAP@{thr:.2f}"] = float(np.mean(aps)) if aps else float("nan")
        out[f"AP_per_class@{thr:.2f}"] = per_class
    if len(iou_thrs) > 1:
        out["mAP@[.5:.95]"] = float(np.nanmean([out[f"mAP@{t:.2f}"] for t in iou_thrs]))
    return out


def pd_fa_curve(preds: Dict[int, Dict], gts: Dict[int, Dict], area_km2_per_image: Dict[int, float],
                iou_thr: float = 0.3, thresholds: Optional[np.ndarray] = None,
                milco_only: bool = False) -> Dict[str, np.ndarray]:
    """Class-agnostic detection PD vs. false alarms per km^2.

    With ``milco_only`` the operating picture is *mine-hunting*: PD is measured on
    MILCO ground truth and a false alarm is any detection classified as MILCO
    that does not match a MILCO ground truth (so correctly rejected rocks are
    not false alarms, but rocks called mines are).
    """
    total_area = float(sum(area_km2_per_image.values()))
    thresholds = np.linspace(0.02, 0.98, 49) if thresholds is None else thresholds
    all_tp, all_scores, n_gt = [], [], 0
    for img_id, g in gts.items():
        p = preds.get(img_id, {"boxes": np.zeros((0, 4)), "scores": np.zeros(0), "labels": np.zeros(0, int)})
        gb, pb, ps = g["boxes"], p["boxes"], p["scores"]
        if milco_only:
            gm = np.array([CLASSES[int(l)] in MILCO_CLASSES for l in g["labels"]], dtype=bool)
            pm = np.array([CLASSES[int(l)] in MILCO_CLASSES for l in p["labels"]], dtype=bool)
            gb, pb, ps = gb[gm], pb[pm], ps[pm]
        n_gt += len(gb)
        tp, _ = match_image(pb, ps, gb, iou_thr)
        all_tp.append(tp)
        all_scores.append(ps)
    tp = np.concatenate(all_tp) if all_tp else np.array([], bool)
    sc = np.concatenate(all_scores) if all_scores else np.array([])
    pd_, fa = [], []
    for t in thresholds:
        sel = sc >= t
        pd_.append(tp[sel].sum() / max(n_gt, 1))
        fa.append((~tp[sel]).sum() / max(total_area, 1e-9))
    return {"thresholds": thresholds, "pd": np.array(pd_), "fa_per_km2": np.array(fa), "n_gt": n_gt, "area_km2": total_area}


def classification_confusion(preds, gts, num_classes: int, iou_thr: float = 0.5, score_thr: float = 0.3) -> np.ndarray:
    """Confusion matrix over matched detections (rows = GT class, cols = predicted)."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for img_id, g in gts.items():
        p = preds.get(img_id)
        if p is None:
            continue
        sel = p["scores"] >= score_thr
        tp, assign = match_image(p["boxes"][sel], p["scores"][sel], g["boxes"], iou_thr)
        for i, j in enumerate(assign):
            if j >= 0:
                cm[int(g["labels"][j]), int(p["labels"][sel][i])] += 1
    return cm


def milco_confusion(cm: np.ndarray) -> Dict[str, float]:
    m = np.array([c in MILCO_CLASSES for c in CLASSES])
    tp = cm[m][:, m].sum(); fn = cm[m][:, ~m].sum(); fp = cm[~m][:, m].sum(); tn = cm[~m][:, ~m].sum()
    return {"milco_recall": float(tp / max(tp + fn, 1)), "nombo_rejection": float(tn / max(tn + fp, 1)),
            "milco_precision": float(tp / max(tp + fp, 1)), "n": int(cm.sum())}


def recall_breakdown(preds, gts, key: str, bins: Sequence[float], iou_thr: float = 0.5, score_thr: float = 0.3) -> Dict:
    """Recall of GT objects binned by an annotation attribute (range_m, burial, aspect_deg, grazing_deg)."""
    edges = np.asarray(bins)
    hit = np.zeros(len(edges) - 1); tot = np.zeros(len(edges) - 1)
    for img_id, g in gts.items():
        p = preds.get(img_id, {"boxes": np.zeros((0, 4)), "scores": np.zeros(0), "labels": np.zeros(0, int)})
        sel = p["scores"] >= score_thr
        _, assign = match_image(p["boxes"][sel], p["scores"][sel], g["boxes"], iou_thr)
        found = np.zeros(len(g["boxes"]), bool)
        found[assign[assign >= 0]] = True
        for j, ex in enumerate(g["extras"]):
            v = ex.get(key)
            if v is None:
                continue
            b = int(np.searchsorted(edges, v, side="right") - 1)
            if 0 <= b < len(tot):
                tot[b] += 1
                hit[b] += found[j]
    return {"edges": edges.tolist(), "recall": (hit / np.maximum(tot, 1)).tolist(), "count": tot.astype(int).tolist()}


def aspect_confusion_counts(preds, gts, num_classes: int, edges=(0, 30, 60, 90, 120, 150, 180.001),
                            iou_thr: float = 0.5, score_thr: float = 0.3) -> np.ndarray:
    """[bins, C, C] confusion counts of matched detections binned by GT aspect angle.
    Feeds the aspect-dependent likelihood model used by the re-acquisition planner."""
    edges = np.asarray(edges)
    out = np.zeros((len(edges) - 1, num_classes, num_classes), dtype=int)
    for img_id, g in gts.items():
        p = preds.get(img_id)
        if p is None:
            continue
        sel = p["scores"] >= score_thr
        _, assign = match_image(p["boxes"][sel], p["scores"][sel], g["boxes"], iou_thr)
        for i, j in enumerate(assign):
            if j < 0:
                continue
            a = g["extras"][j].get("aspect_deg")
            if a is None:
                continue
            b = int(np.clip(np.searchsorted(edges, a % 180.0, side="right") - 1, 0, len(edges) - 2))
            out[b, int(g["labels"][j]), int(p["labels"][sel][i])] += 1
    return out


def expected_calibration_error(preds, gts, iou_thr: float = 0.5, n_bins: int = 10) -> Dict:
    confs, correct = [], []
    for img_id, g in gts.items():
        p = preds.get(img_id)
        if p is None:
            continue
        tp, _ = match_image(p["boxes"], p["scores"], g["boxes"], iou_thr, p["labels"], g["labels"])
        confs.append(p["scores"]); correct.append(tp.astype(float))
    if not confs:
        return {"ece": float("nan")}
    c = np.concatenate(confs); y = np.concatenate(correct)
    edges = np.linspace(0, 1, n_bins + 1)
    ece, acc_b, conf_b, cnt_b = 0.0, [], [], []
    for i in range(n_bins):
        m = (c > edges[i]) & (c <= edges[i + 1])
        if m.any():
            ece += m.mean() * abs(y[m].mean() - c[m].mean())
            acc_b.append(float(y[m].mean())); conf_b.append(float(c[m].mean())); cnt_b.append(int(m.sum()))
        else:
            acc_b.append(float("nan")); conf_b.append(float("nan")); cnt_b.append(0)
    return {"ece": float(ece), "bin_acc": acc_b, "bin_conf": conf_b, "bin_count": cnt_b}


# ---------------------------------------------------------------------------------
# inference over a dataset
# ---------------------------------------------------------------------------------

@torch.no_grad()
def predict_dataset(model: torch.nn.Module, loader: DataLoader, score_thresh: float = 0.02, topk: int = 50,
                    device: str = "cpu", desc: str = "predict") -> Tuple[Dict[int, Dict], Dict[int, Dict], Dict[int, float]]:
    model.eval()
    preds, gts, areas = {}, {}, {}
    for batch in tqdm(loader, desc=desc, leave=False):
        x = batch["image"].to(device)
        out = model(x)
        dets = decode_detections(out, model.stride, topk, score_thresh)
        for i, d in enumerate(dets):
            iid = batch["image_id"][i]
            preds[iid] = {k: v.cpu().numpy() for k, v in d.items()}
            gb = batch["boxes"][i].numpy()
            gts[iid] = {"boxes": xywh_to_xyxy(gb), "labels": batch["labels"][i].numpy(), "extras": batch["extras"][i]}
            H, W = x.shape[-2:]
            areas[iid] = H * W * batch["pixel_m"][i] ** 2 / 1e6
    return preds, gts, areas


def evaluate_model(model, dataset, batch_size: int = 16, device: str = "cpu", out_dir: Optional[str] = None,
                   num_workers: int = 2, score_thr: float = 0.3) -> Dict:
    from ..data.dataset import collate_fn
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    preds, gts, areas = predict_dataset(model, loader, device=device)
    C = dataset.num_classes
    metrics = compute_map(preds, gts, C, iou_thrs=(0.5,))
    metrics.update({k: v for k, v in compute_map(preds, gts, C, iou_thrs=tuple(np.arange(0.5, 0.96, 0.05))).items()
                    if k == "mAP@[.5:.95]"})
    curve = pd_fa_curve(preds, gts, areas, iou_thr=0.3)
    curve_m = pd_fa_curve(preds, gts, areas, iou_thr=0.3, milco_only=True)
    cm = classification_confusion(preds, gts, C, score_thr=score_thr)
    metrics["pd_fa"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in curve.items()}
    metrics["pd_fa_milco"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in curve_m.items()}
    metrics["confusion"] = cm.tolist()
    metrics["milco_vs_nombo"] = milco_confusion(cm)
    metrics["by_range"] = recall_breakdown(preds, gts, "range_m", [0, 40, 70, 100, 140, 260], score_thr=score_thr)
    metrics["by_burial"] = recall_breakdown(preds, gts, "burial", [-0.01, 0.001, 0.2, 0.4, 0.6, 0.76], score_thr=score_thr)
    metrics["by_aspect"] = recall_breakdown(preds, gts, "aspect_deg", [0, 30, 60, 90, 120, 150, 180.1], score_thr=score_thr)
    metrics["by_grazing"] = recall_breakdown(preds, gts, "grazing_deg", [0, 4, 7, 12, 20, 90], score_thr=score_thr)
    metrics["calibration"] = expected_calibration_error(preds, gts)
    metrics["aspect_confusion"] = aspect_confusion_counts(preds, gts, C, score_thr=score_thr).tolist()
    metrics["aspect_edges"] = [0, 30, 60, 90, 120, 150, 180.001]
    metrics["n_images"] = len(gts)
    metrics["n_gt"] = int(sum(len(g["boxes"]) for g in gts.values()))
    metrics["area_km2"] = float(sum(areas.values()))
    # operating point summary at the given threshold
    idx = int(np.argmin(np.abs(curve["thresholds"] - score_thr)))
    metrics["operating_point"] = {"score_thr": score_thr, "pd": float(curve["pd"][idx]),
                                  "fa_per_km2": float(curve["fa_per_km2"][idx]),
                                  "pd_milco": float(curve_m["pd"][idx]), "fa_per_km2_milco": float(curve_m["fa_per_km2"][idx])}
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2, default=float)
        np.savez_compressed(os.path.join(out_dir, "predictions.npz"),
                            **{f"p{iid}_{k}": v for iid, p in preds.items() for k, v in p.items()})
    return metrics

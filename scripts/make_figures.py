"""Regenerate all README figures from the generated dataset, trained runs and demos.

Every section is optional: figures whose inputs are missing are skipped, so the
script can be re-run at any stage of the pipeline.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from seabed_sentinel.sim.scene import CLASSES  # noqa: E402
from seabed_sentinel.viz.plots import (  # noqa: E402
    CLASS_COLOURS,
    plot_breakdowns,
    plot_calibration,
    plot_confusion,
    plot_detections,
    plot_pd_fa,
    plot_training_curves,
)

torch.set_num_threads(2)


def fig_dataset_samples(data_root, out):
    from seabed_sentinel.data.dataset import SonarDetectionDataset
    ds = SonarDetectionDataset(os.path.join(data_root, "test"), crop=None, train=False)
    rng = np.random.default_rng(1)
    idx = [i for i in rng.permutation(len(ds)) if len(ds.anns_by_image[ds.images[i]["id"]]) > 0][:8]
    fig, axes = plt.subplots(2, 4, figsize=(17, 9))
    for ax, i in zip(axes.ravel(), idx):
        rec = ds.images[i]
        item = ds[i]
        img = item["image"][0].numpy()
        px, y0 = rec["pixel_m"], rec["range_start_m"]
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, origin="lower", extent=[y0, y0 + img.shape[1] * px, 0, img.shape[0] * px])
        for a in ds.anns_by_image[rec["id"]]:
            x, y, w, h = a["bbox"]
            ax.add_patch(plt.Rectangle((y0 + x * px, y * px), w * px, h * px, fill=False, ec=CLASS_COLOURS[CLASSES[a["category_id"]]], lw=1.5))
            if a.get("bbox_shadow"):
                x, y, w, h = a["bbox_shadow"]
                ax.add_patch(plt.Rectangle((y0 + x * px, y * px), w * px, h * px, fill=False, ec="cyan", lw=0.8, ls="--"))
            ax.text(y0 + a["bbox"][0] * px, (a["bbox"][1] + a["bbox"][3]) * px + 0.1, f"{CLASSES[a['category_id']]}" + (f" buried {a['burial']:.0%}" if a["burial"] > 0 else ""),
                    color="yellow", fontsize=7)
        ax.set_title(f"{rec['sediment']}, alt {rec['altitude_m']:.0f} m, grazing {np.degrees(np.arctan2(rec['altitude_m'], y0 + 4)):.1f}°, SNR {rec['snr_db']:.0f} dB, {rec['looks']} look(s)", fontsize=8)
        ax.set_xlabel("ground range [m]"); ax.set_ylabel("along-track [m]")
    fig.suptitle("Synthetic SAS tiles (3 cm pixels). Solid = highlight box (class colour), dashed cyan = acoustic shadow box", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(out, "dataset_samples.png"), dpi=100); plt.close(fig)

    # physics panel: height map, visibility, seg mask for one scene
    from seabed_sentinel.sim.renderer import RenderOptions, render_scene
    from seabed_sentinel.sim.scene import SceneConfig, make_random_scene
    scene = make_random_scene(1000003, SceneConfig(p_empty=0.0, max_objects=3))
    res = render_scene(scene, RenderOptions(p_dropout=0, p_striping=0))
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))
    g = scene.geometry
    ext = [g.range_start_m, g.range_start_m + g.range_extent_m, 0, g.along_extent_m]
    im = axes[0].imshow(res.height, cmap="terrain", origin="lower", extent=ext); axes[0].set_title("seabed + object height map [m]", fontsize=9); plt.colorbar(im, ax=axes[0], fraction=0.046)
    axes[1].imshow(res.visibility, cmap="gray", origin="lower", extent=ext); axes[1].set_title("ray-cast visibility (black = acoustic shadow)", fontsize=9)
    axes[2].imshow(res.seg_mask(), cmap="viridis", origin="lower", extent=ext, vmin=0, vmax=2); axes[2].set_title("semantic labels: seabed / highlight / shadow", fontsize=9)
    axes[3].imshow(res.image, cmap="gray", origin="lower", extent=ext, vmin=0, vmax=1); axes[3].set_title(f"rendered SAS image (SNR {res.meta['snr_db']:.0f} dB, {res.meta['looks']} look)", fontsize=9)
    for a in axes:
        a.set_xlabel("ground range [m]")
    axes[0].set_ylabel("along-track [m]")
    fig.tight_layout(); fig.savefig(os.path.join(out, "render_pipeline.png"), dpi=100); plt.close(fig)


def fig_eval(run_dir, data_root, out):
    mp = os.path.join(run_dir, "eval", "metrics.json")
    if not os.path.exists(mp):
        return None
    m = json.load(open(mp))
    hp = os.path.join(run_dir, "eval_hard", "metrics.json")
    hard = json.load(open(hp)) if os.path.exists(hp) else None
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    curves = {"test: all objects (MILCO + NOMBO)": m["pd_fa"], "test: mine-hunting (MILCO PD vs MILCO FA)": m["pd_fa_milco"]}
    if hard:
        curves["stress split: all objects"] = hard["pd_fa"]
        curves["stress split: mine-hunting"] = hard["pd_fa_milco"]
    plot_pd_fa(ax, curves, title=f"PD vs FA/km² — {m['n_images']} in-distribution test tiles ({m['area_km2'] * 1e4:.1f} ha)"
               + (f" and {hard['n_images']} out-of-distribution stress tiles" if hard else ""))
    for mm, mk in ((m, "k*"), (hard, "r*")):
        if mm is None:
            continue
        th = np.array(mm["pd_fa"]["thresholds"]); i = int(np.argmin(np.abs(th - 0.5)))
        ax.plot([mm["pd_fa"]["fa_per_km2"][i]], [mm["pd_fa"]["pd"][i]], mk, ms=12,
                label=f"score ≥ 0.5 operating point ({'stress' if mm is hard else 'test'})")
    ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(out, "pd_fa_curves.png"), dpi=110); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    plot_confusion(axes[0], np.array(m["confusion"]), title="Class confusion of matched detections (row-normalised)")
    mv = m["milco_vs_nombo"]
    axes[1].axis("off")
    txt = (f"mAP@0.5 = {m['mAP@0.50']:.3f}     mAP@[.5:.95] = {m['mAP@[.5:.95]']:.3f}\n\n"
           + "\n".join(f"  AP@0.5 {k:15s} {v:.3f}" for k, v in m["AP_per_class@0.50"].items())
           + f"\n\nMILCO recall (mine called mine)      {mv['milco_recall']:.3f}\n"
           f"NOMBO rejection (rock called rock)   {mv['nombo_rejection']:.3f}\n"
           f"MILCO precision                      {mv['milco_precision']:.3f}\n\n"
           f"Operating point (score ≥ {op['score_thr']}):\n  PD = {op['pd']:.3f}   FA = {op['fa_per_km2']:.0f} / km²\n"
           f"  mine-hunting PD = {op['pd_milco']:.3f}   MILCO FA = {op['fa_per_km2_milco']:.0f} / km²\n\n"
           f"Calibration ECE = {m['calibration']['ece']:.3f}")
    axes[1].text(0.0, 0.98, txt, family="monospace", fontsize=10, va="top")
    fig.tight_layout(); fig.savefig(os.path.join(out, "confusion.png"), dpi=110); plt.close(fig)

    rows = [("in-distribution test split", m)] + ([("out-of-distribution stress split (SNR 0–10 dB, burial ≤ 75 %, dense clutter)", hard)] if hard else [])
    fig, axes = plt.subplots(len(rows), 4, figsize=(18, 4.2 * len(rows)), squeeze=False)
    for r, (label, mm) in enumerate(rows):
        plot_breakdowns(axes[r], mm)
        axes[r, 0].set_title(label, loc="left", fontsize=9, fontweight="bold")
    fig.suptitle("Recall of visible objects by imaging condition (score ≥ 0.3, IoU ≥ 0.5)", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(out, "breakdowns.png"), dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    plot_calibration(ax, m["calibration"])
    fig.tight_layout(); fig.savefig(os.path.join(out, "calibration.png"), dpi=110); plt.close(fig)

    sp = os.path.join(run_dir, "summary.json")
    if os.path.exists(sp):
        s = json.load(open(sp))
        fig, ax = plt.subplots(figsize=(7, 4))
        plot_training_curves(ax, s["history"])
        ax.set_title(f"Training ({s['params_M']:.2f} M params, {s['train_time_min']:.0f} min on 4 CPU cores)", fontsize=10)
        fig.tight_layout(); fig.savefig(os.path.join(out, "training_curves.png"), dpi=110); plt.close(fig)
    return m


def fig_detections(run_dir, data_root, out, n=8):
    ck = os.path.join(run_dir, "best.pt")
    if not os.path.exists(ck):
        return
    from seabed_sentinel.data.dataset import SonarDetectionDataset
    from seabed_sentinel.models.detector import decode_detections
    from seabed_sentinel.train.evaluate import xywh_to_xyxy
    from seabed_sentinel.train.trainer import load_checkpoint
    from seabed_sentinel.viz.gradcam import GradCAM
    model = load_checkpoint(ck)
    ds = SonarDetectionDataset(os.path.join(data_root, "test"), crop=None, train=False)
    rng = np.random.default_rng(7)
    idx = [i for i in rng.permutation(len(ds)) if len(ds.anns_by_image[ds.images[i]["id"]]) > 0][:n]
    fig, axes = plt.subplots(2, 4, figsize=(17, 9))
    for ax, i in zip(axes.ravel(), idx):
        item = ds[i]
        rec = ds.images[i]
        with torch.no_grad():
            out_ = model(item["image"][None])
        det = {k: v.numpy() for k, v in decode_detections(out_, 4, 20, 0.05)[0].items()}
        gts = {"boxes": xywh_to_xyxy(item["boxes"].numpy()), "labels": item["labels"].numpy()}
        plot_detections(ax, item["image"][0].numpy(), det, gts, pixel_m=rec["pixel_m"], range_start_m=rec["range_start_m"],
                        score_thr=0.3, title=f"{rec['sediment']}, alt {rec['altitude_m']:.0f} m, y0 {rec['range_start_m']:.0f} m, SNR {rec['snr_db']:.0f} dB")
    fig.suptitle("Test-set detections (coloured = prediction with score, white dashed = ground truth)", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(out, "detections_test.png"), dpi=100); plt.close(fig)

    # Grad-CAM + MC-dropout uncertainty on 4 detections
    cam = GradCAM(model)
    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    k = 0
    for i in idx:
        if k >= 4:
            break
        item = ds[i]
        x = item["image"][None]
        with torch.no_grad():
            out_ = model(x)
        det = decode_detections(out_, 4, 5, 0.3)[0]
        if len(det["boxes"]) == 0:
            continue
        b = det["boxes"][0].numpy(); c = int(det["labels"][0]); s = float(det["scores"][0])
        cx, cy = int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2)
        heat = cam(x.clone(), c, (cx, cy))
        mc, unc = model.forward_mc(x, n=12)
        img = item["image"][0].numpy()
        axes[0, k].imshow(img, cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[0, k].add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=CLASS_COLOURS[CLASSES[c]], lw=2))
        axes[0, k].set_title(f"detection: {CLASSES[c]} ({s:.2f})", fontsize=9)
        axes[1, k].imshow(img, cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[1, k].imshow(heat, cmap="jet", alpha=0.45, origin="lower")
        axes[1, k].set_title("Grad-CAM: evidence used by the class head", fontsize=9)
        u = torch.nn.functional.interpolate(unc[None], size=img.shape, mode="bilinear", align_corners=False)[0, 0].numpy()
        axes[2, k].imshow(img, cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[2, k].imshow(u, cmap="magma", alpha=0.55, origin="lower", vmin=0, vmax=max(u.max(), 1e-3))
        axes[2, k].set_title(f"MC-dropout epistemic uncertainty (max {u.max():.2f})", fontsize=9)
        for r in range(3):
            axes[r, k].set_xticks([]); axes[r, k].set_yticks([])
        k += 1
    fig.tight_layout(); fig.savefig(os.path.join(out, "explainability.png"), dpi=100); plt.close(fig)


def fig_label_efficiency(runs_root, out):
    rows = []
    for name, label in (("detector_scratch_10pct", "from scratch"), ("detector_mae_10pct", "MAE pre-trained")):
        mp = os.path.join(runs_root, name, "eval", "metrics.json")
        if os.path.exists(mp):
            m = json.load(open(mp))
            rows.append((label, m["mAP@0.50"], m["operating_point"]["pd"]))
    full = os.path.join(runs_root, "detector_base", "eval", "metrics.json")
    if os.path.exists(full):
        m = json.load(open(full))
        rows.append(("100 % labels (reference)", m["mAP@0.50"], m["operating_point"]["pd"]))
    if len(rows) < 2:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.2))
    xs = np.arange(len(rows))
    ax.bar(xs - 0.18, [r[1] for r in rows], 0.36, label="mAP@0.5", color="#3b7dd8")
    ax.bar(xs + 0.18, [r[2] for r in rows], 0.36, label="PD @ score 0.3", color="#f28e2b")
    for x, r in zip(xs, rows):
        ax.text(x - 0.18, r[1] + 0.01, f"{r[1]:.2f}", ha="center", fontsize=8); ax.text(x + 0.18, r[2] + 0.01, f"{r[2]:.2f}", ha="center", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels([r[0] for r in rows]); ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    ax.set_title("Label efficiency: 10 % of labels (240 tiles), with / without self-supervised MAE pre-training", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(out, "label_efficiency.png"), dpi=110); plt.close(fig)
    mae = os.path.join(runs_root, "mae", "mae_full.pt")
    if os.path.exists(mae):
        from seabed_sentinel.models.ssl import reconstruct_examples
        x, xm, rec, mask = reconstruct_examples(mae, "data/sas_synth", n=4)
        fig, axes = plt.subplots(3, 4, figsize=(14, 10))
        for k in range(4):
            axes[0, k].imshow(x[k], cmap="gray", vmin=0, vmax=1, origin="lower"); axes[0, k].set_title("original", fontsize=9)
            axes[1, k].imshow(xm[k], cmap="gray", vmin=0, vmax=1, origin="lower"); axes[1, k].set_title("60 % of patches masked", fontsize=9)
            axes[2, k].imshow(rec[k] * mask[k] + x[k] * (1 - mask[k]), cmap="gray", vmin=0, vmax=1, origin="lower"); axes[2, k].set_title("MAE reconstruction (masked patches)", fontsize=9)
            for r in range(3):
                axes[r, k].set_xticks([]); axes[r, k].set_yticks([])
        fig.tight_layout(); fig.savefig(os.path.join(out, "mae_reconstruction.png"), dpi=100); plt.close(fig)
    return rows


def fig_stream(stream_dir, out):
    sp = os.path.join(stream_dir, "swath.npz")
    if not os.path.exists(sp):
        return None
    d = np.load(sp)
    rep = json.load(open(os.path.join(stream_dir, "stream_report.json")))
    img = d["image"]
    px, y0 = float(d["pixel_m"]), float(d["range_start_m"])
    H, W = img.shape
    fig, ax = plt.subplots(figsize=(8, 15))
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, origin="lower", extent=[y0, y0 + W * px, 0, H * px], aspect="equal")
    for b in d["gt_boxes"]:
        ax.add_patch(plt.Rectangle((y0 + b[0] * px, b[1] * px), (b[2] - b[0]) * px, (b[3] - b[1]) * px, fill=False, ec="white", lw=1.2, ls="--"))
    for b, s, l in zip(d["det_boxes"], d["det_scores"], d["det_labels"]):
        col = CLASS_COLOURS[CLASSES[int(l)]]
        ax.add_patch(plt.Rectangle((y0 + b[0] * px, b[1] * px), (b[2] - b[0]) * px, (b[3] - b[1]) * px, fill=False, ec=col, lw=1.6))
        ax.text(y0 + b[2] * px + 0.2, b[1] * px, f"{CLASSES[int(l)]} {s:.2f}", color=col, fontsize=7, bbox=dict(facecolor="black", alpha=0.5, pad=1, lw=0))
    ax.set_xlabel("ground range [m]"); ax.set_ylabel("along-track [m]")
    ax.set_title(f"Streaming ATR on a {H * px:.0f} m × {W * px:.0f} m swath: {rep['true_positives']}/{rep['n_gt_visible']} objects found, "
                 f"{rep['false_alarms']} false alarms\n{rep['measured_tiles_s']:.0f} tiles/s on CPU = {rep['realtime_factor']:.1f}× real time for a 200 m/side swath at 10 kn", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(out, "stream_swath.png"), dpi=100); plt.close(fig)
    return rep


def fig_plan(metrics_path, out, seed=3):
    from seabed_sentinel.autonomy.aspect_fusion import AspectConfusionModel, entropy
    from seabed_sentinel.autonomy.reacquire import Contact2D, plan_reacquisition
    rng = np.random.default_rng(seed)
    if os.path.exists(metrics_path):
        m = json.load(open(metrics_path))
        model = AspectConfusionModel(CLASSES, np.asarray(m["aspect_edges"]), np.asarray(m["aspect_confusion"], float))
        src = "aspect-binned confusion measured on the test set"
    else:
        model = AspectConfusionModel.synthetic(CLASSES); src = "synthetic prior"
    contacts = []
    for i in range(12):
        post = rng.dirichlet(np.ones(len(CLASSES)) * rng.uniform(0.3, 2.0))
        contacts.append(Contact2D(i, float(rng.uniform(-300, 300)), float(rng.uniform(50, 650)), post,
                                  first_aspect_deg=float(rng.uniform(0, 180)), est_heading_deg=float(rng.uniform(0, 360))))
    plan = plan_reacquisition(contacts, model, budget_m=2500.0)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    ax.plot(plan.path[:, 0], plan.path[:, 1], "-", color="#3b7dd8", lw=1.5, label="planned AUV path")
    for leg in plan.legs:
        ax.annotate("", xy=leg.end, xytext=leg.start, arrowprops=dict(arrowstyle="->", color="#1f4e99", lw=2.2))
    for c in contacts:
        h = entropy(c.posterior)
        ax.scatter(c.x, c.y, s=60 + 300 * h / np.log2(len(CLASSES)), c=[[1 - h / np.log2(len(CLASSES)), 0.2, h / np.log2(len(CLASSES))]], edgecolor="k", zorder=5)
        ax.text(c.x + 8, c.y + 8, f"#{c.id} H={h:.2f}b", fontsize=8)
    ax.scatter([0], [0], marker="s", s=90, c="green", label="start", zorder=6)
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_xlabel("east [m]"); ax.set_ylabel("north [m]")
    ax.set_title(f"Information-driven re-acquisition: {len(plan.legs)} re-look legs, {plan.total_length_m:.0f} m path (budget 2500 m), "
                 f"+{plan.total_gain_bits:.1f} bits expected\nmarker size/colour = posterior entropy after first pass; skipped contacts are already confident", fontsize=9)
    ax = axes[1]
    aspects = np.linspace(0, 179, 60)
    for cls_id in (0, 1, 4):
        post = np.full(len(CLASSES), 0.1); post[cls_id] = 0.5
        ax.plot(aspects, [model.expected_information_gain(post, a) for a in aspects], label=f"posterior peaked on '{CLASSES[cls_id]}'")
    ax.set_xlabel("re-look aspect angle [deg from along-track]"); ax.set_ylabel("expected information gain [bits]")
    ax.set_title(f"Aspect-dependent value of a new look\n({src})", fontsize=9); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "reacquisition_plan.png"), dpi=100); plt.close(fig)
    return plan.summary()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/sas_synth")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="assets")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    want = lambda k: args.only is None or k in args.only  # noqa: E731
    results = {}
    if want("dataset") and os.path.exists(os.path.join(args.data_root, "test", "annotations.json")):
        fig_dataset_samples(args.data_root, args.out); print("dataset figures done")
    if want("eval"):
        results["eval"] = fig_eval(os.path.join(args.runs, "detector_base"), args.data_root, args.out); print("eval figures done")
    if want("detections"):
        fig_detections(os.path.join(args.runs, "detector_base"), args.data_root, args.out); print("detection figures done")
    if want("label"):
        results["label_efficiency"] = fig_label_efficiency(args.runs, args.out); print("label-efficiency figures done")
    if want("stream"):
        results["stream"] = fig_stream(os.path.join(args.runs, "stream"), args.out); print("stream figure done")
    if want("plan"):
        results["plan"] = fig_plan(os.path.join(args.runs, "detector_base", "eval", "metrics.json"), args.out); print("plan figure done")
    with open(os.path.join(args.out, "figure_summary.json"), "w") as f:
        json.dump(results, f, indent=1, default=float)


if __name__ == "__main__":
    main()

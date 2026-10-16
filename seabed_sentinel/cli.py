"""Command-line interface: ``python -m seabed_sentinel <command>`` or ``sentinel <command>``."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import fields

import numpy as np


def _p(*a, **k):
    print(*a, **k, flush=True)


# ---------------------------------------------------------------------------------
def cmd_info(args):
    from .sim.acoustics import SASConfig
    cfg = SASConfig()
    _p("Default SAS configuration (AquaPix MINSAS-class, public figures):")
    for k, v in cfg.summary().items():
        _p(f"  {k:28s} {v:10.4g}")


def cmd_simulate(args):
    from .sim.dataset_gen import generate_dataset, generate_stress_split
    from .sim.scene import SceneConfig
    cfg = SceneConfig(n_along=args.tile, n_range=args.tile)
    t0 = time.time()
    if args.preset == "hard":
        generate_stress_split(args.root, name=args.split_name, n=args.n_test, base_cfg=cfg, workers=args.workers,
                              seed=args.seed + 9_000_000)
        _p(f"stress split '{args.split_name}' ({args.n_test} tiles) written to {args.root} in {(time.time() - t0) / 60:.1f} min")
        return
    generate_dataset(args.root, args.n_train, args.n_val, args.n_test, cfg=cfg, workers=args.workers, seed=args.seed)
    _p(f"dataset written to {args.root} in {(time.time() - t0) / 60:.1f} min")


def cmd_train(args):
    from .train.trainer import TrainConfig, train
    cfg = TrainConfig.from_yaml(args.config) if args.config else TrainConfig()
    for f in fields(TrainConfig):
        v = getattr(args, f.name.replace("-", "_"), None)
        if v is not None:
            setattr(cfg, f.name, v)
    summary = train(cfg)
    _p(f"best val mAP@0.5 = {summary['best_val_mAP50']:.3f}  ({summary['train_time_min']:.1f} min)")


def cmd_pretrain(args):
    from .models.ssl import MAEConfig, pretrain_mae
    cfg = MAEConfig(data_root=args.data_root, out_dir=args.out_dir, epochs=args.epochs, batch_size=args.batch_size,
                    num_workers=args.num_workers, threads=args.threads)
    pretrain_mae(cfg)


def cmd_evaluate(args):
    import torch

    from .data.dataset import SonarDetectionDataset
    from .train.evaluate import evaluate_model
    from .train.trainer import load_checkpoint
    if args.threads:
        torch.set_num_threads(args.threads)
    model = load_checkpoint(args.checkpoint)
    ds = SonarDetectionDataset(os.path.join(args.data_root, args.split), crop=None, train=False)
    m = evaluate_model(model, ds, batch_size=args.batch_size, out_dir=args.out_dir, num_workers=args.num_workers,
                       score_thr=args.score_thr)
    _p(json.dumps({k: m[k] for k in ("mAP@0.50", "mAP@[.5:.95]", "AP_per_class@0.50", "milco_vs_nombo", "operating_point")},
                  indent=1, default=float))
    _p(f"n_images={m['n_images']} n_gt={m['n_gt']} surveyed area={m['area_km2'] * 1e6:.0f} m^2  ECE={m['calibration']['ece']:.3f}")


def cmd_export(args):
    from .deploy.export_onnx import export_onnx, verify_onnx
    from .train.trainer import load_checkpoint
    model = load_checkpoint(args.checkpoint)
    export_onnx(model, args.out, tile=args.tile)
    rep = verify_onnx(model, args.out, tile=args.tile)
    _p(json.dumps(rep, indent=1))


def cmd_stream(args):
    """Simulate a long survey swath, stream it through the ONNX ATR, geo-reference contacts."""
    from .deploy.geo import TowfishNav, contacts_to_geojson
    from .deploy.stream import StreamingATR, benchmark_realtime
    from .sim.renderer import RenderOptions, render_scene
    from .sim.scene import CLASSES, SceneConfig, make_random_scene
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = SceneConfig(n_along=args.rows, n_range=args.cols, max_objects=args.n_objects, p_empty=0.0,
                      altitude_range=(12, 12), range_start_range=(args.range_start, args.range_start))
    scene = make_random_scene(args.seed, cfg)
    _p(f"rendering swath {args.rows}x{args.cols} px ({args.rows * cfg.pixel_m:.0f} m x {args.cols * cfg.pixel_m:.0f} m) ...")
    t0 = time.time()
    res = render_scene(scene, RenderOptions(p_dropout=0.0))
    _p(f"  rendered in {time.time() - t0:.1f}s with {len(scene.objects)} objects")
    atr = StreamingATR(args.onnx, tile=args.tile, overlap=args.overlap, score_thr=args.score_thr, n_threads=args.threads)
    strip = args.strip_rows
    t0 = time.time()
    for r in range(0, args.rows, strip):
        atr.push(res.image[r: r + strip])
    atr.finish()
    wall = time.time() - t0
    nav = TowfishNav.straight_line(args.rows, lat0=47.5615, lon0=-52.7126, heading_deg=35.0,
                                   ping_advance_m=cfg.pixel_m, altitude_m=12.0)
    gj = contacts_to_geojson(atr.contacts, nav, cfg.pixel_m, scene.geometry.range_start_m, CLASSES,
                             path=os.path.join(args.out_dir, "contacts.geojson"))
    bench = benchmark_realtime(atr, pixel_m=cfg.pixel_m, swath_m=args.swath_m, speed_kn=args.speed_kn)
    # ground truth matching for a quick quality figure
    from .train.evaluate import match_image, xywh_to_xyxy
    gt_boxes = xywh_to_xyxy(np.array([a.bbox_highlight for a in res.annotations if a.visible]).reshape(-1, 4))
    det_boxes = np.array([c.box for c in atr.contacts]).reshape(-1, 4)
    det_scores = np.array([c.score for c in atr.contacts])
    tp, _ = match_image(det_boxes, det_scores, gt_boxes, 0.3)
    report = {"n_contacts": len(atr.contacts), "n_gt_visible": int(len(gt_boxes)), "true_positives": int(tp.sum()),
              "false_alarms": int((~tp).sum()), "recall": float(tp.sum() / max(len(gt_boxes), 1)),
              "wall_time_s": wall, "tiles": atr.stats["tiles"], **bench,
              "swath_m2": args.rows * args.cols * cfg.pixel_m**2}
    with open(os.path.join(args.out_dir, "stream_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    np.savez_compressed(os.path.join(args.out_dir, "swath.npz"), image=res.image,
                        gt_boxes=gt_boxes, det_boxes=det_boxes, det_scores=det_scores,
                        det_labels=np.array([c.label for c in atr.contacts]),
                        gt_labels=np.array([a.class_id for a in res.annotations if a.visible]),
                        range_start_m=scene.geometry.range_start_m, pixel_m=cfg.pixel_m)
    _p(json.dumps(report, indent=1))
    _p(f"GeoJSON with {len(gj['features'])} contacts -> {os.path.join(args.out_dir, 'contacts.geojson')}")


def cmd_plan(args):
    """Plan an information-driven re-acquisition mission from a contact list."""
    from .autonomy.aspect_fusion import AspectConfusionModel
    from .autonomy.reacquire import Contact2D, plan_reacquisition
    from .sim.scene import CLASSES
    rng = np.random.default_rng(args.seed)
    if args.metrics and os.path.exists(args.metrics):
        m = json.load(open(args.metrics))
        model = AspectConfusionModel(CLASSES, np.asarray(m["aspect_edges"]), np.asarray(m["aspect_confusion"], float))
        src = args.metrics
    else:
        model = AspectConfusionModel.synthetic(CLASSES)
        src = "synthetic prior"
    contacts = []
    for i in range(args.n_contacts):
        post = rng.dirichlet(np.ones(len(CLASSES)) * rng.uniform(0.3, 2.0))
        contacts.append(Contact2D(i, float(rng.uniform(-300, 300)), float(rng.uniform(0, 600)), post,
                                  first_aspect_deg=float(rng.uniform(0, 180)), est_heading_deg=float(rng.uniform(0, 360))))
    plan = plan_reacquisition(contacts, model, budget_m=args.budget_m)
    _p(f"confusion model: {src}")
    _p(json.dumps(plan.summary(), indent=1))
    for k in plan.order:
        leg = plan.legs[k]
        _p(f"  contact {leg.contact_id:2d}  aspect {leg.aspect_deg:5.1f} deg  gain {leg.gain_bits:.3f} bits  heading {leg.heading_deg:6.1f}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"summary": plan.summary(), "path": plan.path.tolist(),
                       "legs": [leg.__dict__ for leg in plan.legs], "order": plan.order}, f, indent=1)


# ---------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("sentinel", description="Seabed Sentinel: SAS simulation + ATR toolkit")
    sp = ap.add_subparsers(dest="cmd", required=True)

    s = sp.add_parser("info", help="print the default sonar configuration"); s.set_defaults(fn=cmd_info)

    s = sp.add_parser("simulate", help="render a labelled synthetic SAS dataset")
    s.add_argument("--root", default="data/sas_synth"); s.add_argument("--n-train", type=int, default=2400)
    s.add_argument("--n-val", type=int, default=300); s.add_argument("--n-test", type=int, default=600)
    s.add_argument("--tile", type=int, default=288); s.add_argument("--workers", type=int, default=4)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--preset", choices=("default", "hard"), default="default",
                   help="'hard' renders only an out-of-distribution stress split (low SNR, heavy burial, dense clutter)")
    s.add_argument("--split-name", default="test_hard"); s.set_defaults(fn=cmd_simulate)

    s = sp.add_parser("train", help="train the detector")
    s.add_argument("--config", default=None)
    for name, typ in (("data_root", str), ("out_dir", str), ("epochs", int), ("batch_size", int), ("lr", float),
                      ("label_fraction", float), ("init_from", str), ("num_workers", int), ("threads", int), ("seed", int),
                      ("width", float), ("crop", int)):
        s.add_argument("--" + name.replace("_", "-"), dest=name, type=typ, default=None)
    s.set_defaults(fn=cmd_train)

    s = sp.add_parser("pretrain", help="MAE self-supervised pre-training")
    s.add_argument("--data-root", default="data/sas_synth"); s.add_argument("--out-dir", default="runs/mae")
    s.add_argument("--epochs", type=int, default=6); s.add_argument("--batch-size", type=int, default=16)
    s.add_argument("--num-workers", type=int, default=3); s.add_argument("--threads", type=int, default=None)
    s.set_defaults(fn=cmd_pretrain)

    s = sp.add_parser("evaluate", help="evaluate a checkpoint")
    s.add_argument("--checkpoint", required=True); s.add_argument("--data-root", default="data/sas_synth")
    s.add_argument("--split", default="test"); s.add_argument("--out-dir", default=None)
    s.add_argument("--batch-size", type=int, default=16); s.add_argument("--num-workers", type=int, default=2)
    s.add_argument("--score-thr", type=float, default=0.3); s.add_argument("--threads", type=int, default=None)
    s.set_defaults(fn=cmd_evaluate)

    s = sp.add_parser("export", help="export ONNX")
    s.add_argument("--checkpoint", required=True); s.add_argument("--out", required=True)
    s.add_argument("--tile", type=int, default=256); s.set_defaults(fn=cmd_export)

    s = sp.add_parser("stream", help="real-time streaming ATR demo on a simulated swath")
    s.add_argument("--onnx", required=True); s.add_argument("--out-dir", default="runs/stream")
    s.add_argument("--rows", type=int, default=2048); s.add_argument("--cols", type=int, default=1024)
    s.add_argument("--n-objects", type=int, default=14); s.add_argument("--range-start", type=float, default=40.0)
    s.add_argument("--tile", type=int, default=256); s.add_argument("--overlap", type=int, default=64)
    s.add_argument("--strip-rows", type=int, default=64); s.add_argument("--score-thr", type=float, default=0.35)
    s.add_argument("--threads", type=int, default=4); s.add_argument("--swath-m", type=float, default=200.0)
    s.add_argument("--speed-kn", type=float, default=10.0); s.add_argument("--seed", type=int, default=7)
    s.set_defaults(fn=cmd_stream)

    s = sp.add_parser("plan", help="information-driven re-acquisition planning demo")
    s.add_argument("--metrics", default="runs/detector_base/eval/metrics.json")
    s.add_argument("--n-contacts", type=int, default=12); s.add_argument("--budget-m", type=float, default=2500.0)
    s.add_argument("--seed", type=int, default=3); s.add_argument("--out", default=None)
    s.set_defaults(fn=cmd_plan)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())

"""Generate a labelled synthetic SAS dataset in COCO format.

Layout::

    <root>/<split>/images/000123.png     8-bit log-compressed SAS tile
    <root>/<split>/masks/000123.png      0 seabed, 1 highlight, 2 shadow
    <root>/<split>/annotations.json      COCO detection annotations (+ sonar extras)
    <root>/dataset_meta.json             generation config & class list

Each COCO annotation carries extra fields used by the evaluation suite:
``bbox_shadow``, ``bbox_union``, ``range_m``, ``grazing_deg``, ``aspect_deg``,
``burial``, ``is_milco``, ``visible``.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from multiprocessing import Pool
from typing import Dict, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

from .renderer import RenderOptions, render_scene
from .scene import CLASSES, SceneConfig, make_random_scene


def _worker(args):
    seed, split_dir, cfg_dict, opts_dict = args
    cfg = SceneConfig(**cfg_dict)
    opts = RenderOptions(**opts_dict)
    scene = make_random_scene(seed, cfg)
    res = render_scene(scene, opts)
    name = f"{seed:07d}"
    Image.fromarray((res.image * 255).round().astype(np.uint8)).save(os.path.join(split_dir, "images", name + ".png"))
    Image.fromarray(res.seg_mask()).save(os.path.join(split_dir, "masks", name + ".png"))
    g = scene.geometry
    img_rec = {
        "id": int(seed), "file_name": f"images/{name}.png", "mask_file": f"masks/{name}.png",
        "width": int(res.image.shape[1]), "height": int(res.image.shape[0]),
        "pixel_m": g.pixel_m, "altitude_m": g.altitude_m, "range_start_m": g.range_start_m,
        "sediment": scene.sediment, "snr_db": res.meta["snr_db"], "looks": res.meta["looks"],
        "n_clutter": len(scene.clutter),
    }
    anns = []
    for a in res.annotations:
        d = asdict(a)
        anns.append({
            "image_id": int(seed), "category_id": int(a.class_id), "bbox": list(a.bbox_highlight),
            "area": int(a.bbox_highlight[2] * a.bbox_highlight[3]), "iscrowd": 0,
            "bbox_shadow": list(a.bbox_shadow) if a.bbox_shadow else None,
            "bbox_union": list(a.bbox_union),
            **{k: d[k] for k in ("range_m", "grazing_deg", "aspect_deg", "burial", "is_milco", "visible",
                                 "length_m", "highlight_area_px", "shadow_area_px", "x_m", "y_m")},
        })
    return img_rec, anns


def generate_split(root: str, split: str, n: int, seed0: int, cfg: SceneConfig,
                   opts: Optional[RenderOptions] = None, workers: int = 4) -> Dict:
    opts = opts or RenderOptions()
    split_dir = os.path.join(root, split)
    os.makedirs(os.path.join(split_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(split_dir, "masks"), exist_ok=True)
    tasks = [(seed0 + i, split_dir, asdict(cfg), asdict(opts)) for i in range(n)]
    images, annotations = [], []
    with Pool(workers) as pool:
        for img_rec, anns in tqdm(pool.imap_unordered(_worker, tasks, chunksize=8), total=n, desc=f"render {split}"):
            images.append(img_rec)
            annotations.extend(anns)
    images.sort(key=lambda r: r["id"])
    for i, a in enumerate(annotations):
        a["id"] = i
    coco = {
        "info": {"description": f"Seabed Sentinel synthetic SAS dataset ({split})", "version": "1.0"},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i, "name": c, "supercategory": "MILCO" if c in ("cylinder", "truncated_cone", "wedge", "sphere") else "NOMBO"}
                       for i, c in enumerate(CLASSES)],
    }
    with open(os.path.join(split_dir, "annotations.json"), "w") as f:
        json.dump(coco, f)
    return coco


def generate_dataset(root: str, n_train: int, n_val: int, n_test: int, cfg: Optional[SceneConfig] = None,
                     opts: Optional[RenderOptions] = None, workers: int = 4, seed: int = 0) -> None:
    cfg = cfg or SceneConfig()
    opts = opts or RenderOptions()
    os.makedirs(root, exist_ok=True)
    splits = {"train": (n_train, seed + 1_000_000), "val": (n_val, seed + 2_000_000), "test": (n_test, seed + 3_000_000)}
    summary = {}
    for split, (n, s0) in splits.items():
        if n <= 0:
            continue
        coco = generate_split(root, split, n, s0, cfg, opts, workers)
        counts = {c["name"]: 0 for c in coco["categories"]}
        for a in coco["annotations"]:
            counts[CLASSES[a["category_id"]]] += 1
        summary[split] = {"n_images": len(coco["images"]), "n_annotations": len(coco["annotations"]), "per_class": counts}
    with open(os.path.join(root, "dataset_meta.json"), "w") as f:
        json.dump({"classes": CLASSES, "scene_config": asdict(cfg), "render_options": asdict(opts),
                   "seed": seed, "splits": summary}, f, indent=2)


HARD_SCENE = dict(p_burial=0.6, max_burial=0.75, rock_field_prob=0.7, small_rock_density=0.2, p_empty=0.1)
HARD_RENDER = dict(snr_db_range=(0.0, 10.0), p_dropout=0.3, p_striping=0.6, p_smear=0.5, smear_sigma_px=(1.0, 3.0),
                   tvg_residual_max=0.5)


def generate_stress_split(root: str, name: str = "test_hard", n: int = 600, seed: int = 9_000_000,
                          base_cfg: Optional[SceneConfig] = None, workers: int = 4) -> Dict:
    """Out-of-distribution stress split: low SNR, heavy burial, dense clutter, more artefacts."""
    cfg_d = asdict(base_cfg or SceneConfig())
    cfg_d.update(HARD_SCENE)
    opts = RenderOptions(**HARD_RENDER)
    return generate_split(root, name, n, seed, SceneConfig(**cfg_d), opts, workers)

import json
import os

import numpy as np
import torch

from seabed_sentinel.data.dataset import SonarDetectionDataset, collate_fn, encode_centernet_targets
from seabed_sentinel.models.detector import decode_detections
from seabed_sentinel.sim.renderer import RenderOptions, render_scene
from seabed_sentinel.sim.scene import CLASSES, SceneConfig, make_random_scene


def test_render_shapes_and_annotations():
    cfg = SceneConfig(n_along=128, n_range=128, p_empty=0.0, max_objects=2)
    scene = make_random_scene(5, cfg)
    res = render_scene(scene, RenderOptions())
    assert res.image.shape == (128, 128)
    assert res.image.min() >= 0 and res.image.max() <= 1
    assert res.seg_mask().max() <= 2
    assert len(res.annotations) == len(scene.objects)
    for a in res.annotations:
        x, y, w, h = a.bbox_highlight
        assert 0 <= x < 128 and 0 <= y < 128 and w >= 1 and h >= 1
        assert 0 <= a.aspect_deg < 180


def test_shadow_lies_at_farther_range_than_highlight():
    cfg = SceneConfig(n_along=160, n_range=160, p_empty=0.0, max_objects=1, p_burial=0.0, rock_field_prob=0.0)
    n_checked = 0
    for seed in range(8):
        scene = make_random_scene(seed, cfg)
        res = render_scene(scene, RenderOptions(p_dropout=0, p_striping=0, p_smear=0))
        for a in res.annotations:
            if a.bbox_shadow is None or not a.visible:
                continue
            # shadow box centre must be at larger column (range) than highlight centre
            assert a.bbox_shadow[0] + a.bbox_shadow[2] / 2 > a.bbox_highlight[0] + a.bbox_highlight[2] / 2 - 2
            n_checked += 1
    assert n_checked > 0


def test_coco_dataset_is_valid(tiny_dataset):
    coco = json.load(open(os.path.join(tiny_dataset, "train", "annotations.json")))
    assert len(coco["images"]) == 10
    assert [c["name"] for c in coco["categories"]] == CLASSES
    ids = {im["id"] for im in coco["images"]}
    for a in coco["annotations"]:
        assert a["image_id"] in ids
        assert len(a["bbox"]) == 4 and a["bbox"][2] > 0 and a["bbox"][3] > 0
        assert os.path.exists(os.path.join(tiny_dataset, "train", coco["images"][0]["file_name"]))
    meta = json.load(open(os.path.join(tiny_dataset, "dataset_meta.json")))
    assert meta["splits"]["train"]["n_images"] == 10


def test_dataset_item_and_collate(tiny_dataset):
    ds = SonarDetectionDataset(os.path.join(tiny_dataset, "train"), crop=128, train=True)
    item = ds[0]
    assert item["image"].shape == (1, 128, 128)
    assert item["hm"].shape == (len(CLASSES), 32, 32)
    assert "seg" in item and item["seg"].shape == (32, 32)
    if len(item["boxes"]):
        b = item["boxes"].numpy()
        assert (b[:, 0] >= 0).all() and (b[:, 1] >= 0).all()
        assert (b[:, 0] + b[:, 2] <= 128 + 1e-3).all() and (b[:, 1] + b[:, 3] <= 128 + 1e-3).all()
    batch = collate_fn([ds[i] for i in range(4)])
    assert batch["image"].shape == (4, 1, 128, 128)
    assert batch["seg"].shape == (4, 32, 32)


def test_label_fraction_subsamples_images(tiny_dataset):
    full = SonarDetectionDataset(os.path.join(tiny_dataset, "train"), crop=None, train=False)
    part = SonarDetectionDataset(os.path.join(tiny_dataset, "train"), crop=None, train=False, label_fraction=0.5)
    assert 0 < len(part) < len(full)


def test_centernet_encode_decode_roundtrip():
    boxes = np.array([[20, 30, 40, 24], [100, 60, 30, 50]], dtype=np.float32)
    labels = np.array([1, 4])
    tg = encode_centernet_targets(boxes, labels, (160, 160), 6, stride=4)
    assert tg["hm"].max() == 1.0 and tg["reg_mask"].sum() == 2
    out = {"hm": torch.logit(torch.from_numpy(tg["hm"]).clamp(1e-4, 1 - 1e-4))[None],
           "wh": torch.from_numpy(tg["wh"])[None], "reg": torch.from_numpy(tg["reg"])[None]}
    det = decode_detections(out, stride=4, topk=10, score_thresh=0.5)[0]
    got = det["boxes"].numpy()
    want = np.stack([boxes[:, 0], boxes[:, 1], boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]], 1)
    order = np.argsort(got[:, 0])
    assert np.allclose(got[order], want[np.argsort(want[:, 0])], atol=1.0)
    assert set(det["labels"].tolist()) == {1, 4}

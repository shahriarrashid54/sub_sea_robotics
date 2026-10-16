import os

import numpy as np
import torch

from seabed_sentinel.deploy.export_onnx import export_onnx, verify_onnx
from seabed_sentinel.deploy.geo import TowfishNav, contacts_to_geojson, latlon_to_enu, pixel_to_latlon
from seabed_sentinel.deploy.stream import Contact, StreamingATR, decode_numpy, nms
from seabed_sentinel.models import SonarCenterNet


def test_onnx_export_parity(tmp_path):
    torch.manual_seed(0)
    m = SonarCenterNet(num_classes=6, width=0.5).eval()
    p = str(tmp_path / "m.onnx")
    export_onnx(m, p, tile=64)
    rep = verify_onnx(m, p, tile=64)
    assert rep["max_abs_diff_hm"] < 1e-4 and rep["max_abs_diff_wh"] < 1e-3


def test_streaming_atr_runs_and_merges(tmp_path):
    torch.manual_seed(0)
    m = SonarCenterNet(num_classes=6, width=0.5).eval()
    p = str(tmp_path / "m.onnx")
    export_onnx(m, p, tile=64)
    atr = StreamingATR(p, tile=64, overlap=16, score_thr=0.0, n_threads=1)
    rng = np.random.default_rng(0)
    for _ in range(5):
        atr.push(rng.random((40, 128), dtype=np.float32))
    atr.finish()
    assert atr.stats["tiles"] > 0
    # contacts are in global coordinates and de-duplicated by NMS
    boxes = np.array([c.box for c in atr.contacts]).reshape(-1, 4)
    if len(boxes) > 1:
        from seabed_sentinel.train.evaluate import box_iou
        iou = box_iou(boxes, boxes)
        np.fill_diagonal(iou, 0)
        assert iou.max() <= atr.nms_iou + 1e-6


def test_nms_and_numpy_decode():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], float)
    keep = nms(boxes, np.array([0.9, 0.8, 0.7]), 0.5)
    assert sorted(keep.tolist()) == [0, 2]
    hm = np.zeros((2, 8, 8), np.float32); hm[1, 3, 4] = 0.9
    d = decode_numpy(hm, np.ones((2, 8, 8), np.float32) * 2, np.zeros((2, 8, 8), np.float32), 4, 0.5)
    assert len(d["boxes"]) == 1 and d["labels"][0] == 1
    assert np.allclose(d["boxes"][0], [16 - 4, 12 - 4, 16 + 4, 12 + 4])


def test_geo_referencing_roundtrip(tmp_path):
    nav = TowfishNav.straight_line(100, lat0=47.56, lon0=-52.71, heading_deg=90.0, ping_advance_m=0.5, altitude_m=12)
    lat, lon, gr = pixel_to_latlon(row=50, col=100, nav=nav, pixel_m=0.03, range_start_m=30.0, side="starboard")
    e, n = latlon_to_enu(np.array([lat]), np.array([lon]), nav.lat[50], nav.lon[50])
    # heading east, starboard = south: offset should be (0, -gr)
    assert abs(e[0]) < 0.05 and abs(n[0] + gr) < 0.05
    assert abs(gr - (30.0 + 100.5 * 0.03)) < 1e-6
    c = Contact(0, 50.0, 100.0, [95, 45, 105, 55], 0.8, 0)
    gj = contacts_to_geojson([c], nav, 0.03, 30.0, ["a", "b"], path=str(tmp_path / "c.geojson"))
    assert gj["features"][0]["properties"]["class"] == "a"
    assert os.path.exists(tmp_path / "c.geojson")

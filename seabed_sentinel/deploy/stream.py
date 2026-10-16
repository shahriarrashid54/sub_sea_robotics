"""Real-time streaming ATR over a sonar waterfall.

A towed SAS streams imagery continuously (KATFISH-class systems stream live
data to a shore / ship command centre). The :class:`StreamingATR` consumes
along-track strips as they arrive, keeps an overlap buffer so that targets on
strip boundaries are not missed, runs ONNX Runtime inference on overlapping
tiles, merges duplicate detections with NMS and emits a time-stamped contact
list. :func:`benchmark_realtime` compares throughput with the area coverage
rate of a given sonar / speed to answer the operational question: *can this
model keep up with the sensor on this CPU?*
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.5) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros(0, dtype=int)
    x0, y0, x1, y1 = boxes.T
    areas = (x1 - x0) * (y1 - y0)
    order = np.argsort(-scores)
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        ix0 = np.maximum(x0[i], x0[rest]); iy0 = np.maximum(y0[i], y0[rest])
        ix1 = np.minimum(x1[i], x1[rest]); iy1 = np.minimum(y1[i], y1[rest])
        inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
        iou = inter / np.clip(areas[i] + areas[rest] - inter, 1e-9, None)
        order = rest[iou < iou_thr]
    return np.array(keep, dtype=int)


def decode_numpy(hm: np.ndarray, wh: np.ndarray, reg: np.ndarray, stride: int = 4, score_thr: float = 0.3,
                 topk: int = 100, min_size_px: float = 2.0) -> Dict[str, np.ndarray]:
    """NumPy decode of one image's outputs ([C,H,W], [2,H,W], [2,H,W])."""
    from scipy.ndimage import maximum_filter
    C, H, W = hm.shape
    peak = hm * (maximum_filter(hm, size=(1, 3, 3), mode="constant") == hm)
    flat = peak.reshape(-1)
    idx = np.argpartition(-flat, min(topk, flat.size - 1))[:topk]
    idx = idx[flat[idx] >= score_thr]
    labels = idx // (H * W)
    pix = idx % (H * W)
    ys, xs = pix // W, pix % W
    cx = (xs + reg[0, ys, xs]) * stride
    cy = (ys + reg[1, ys, xs]) * stride
    w = np.clip(wh[0, ys, xs], 0, None) * stride
    h = np.clip(wh[1, ys, xs], 0, None) * stride
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    ok = (w >= min_size_px) & (h >= min_size_px)
    return {"boxes": boxes[ok].astype(np.float32), "scores": flat[idx][ok].astype(np.float32), "labels": labels[ok].astype(int)}


@dataclass
class Contact:
    id: int
    ping_index: float          # along-track row (global)
    range_px: float            # across-track column
    box: List[float]           # x0, y0, x1, y1 in global pixels
    score: float
    label: int
    t_detect: float = 0.0
    latlon: Optional[List[float]] = None
    class_probs: Optional[List[float]] = None


@dataclass
class StreamingATR:
    onnx_path: str
    tile: int = 256
    overlap: int = 64
    stride: int = 4
    score_thr: float = 0.3
    nms_iou: float = 0.4
    n_threads: int = 4
    _sess: object = field(default=None, repr=False)
    _buffer: Optional[np.ndarray] = field(default=None, repr=False)
    _rows_consumed: int = 0
    contacts: List[Contact] = field(default_factory=list)
    _pending: List[Dict] = field(default_factory=list, repr=False)
    stats: Dict[str, float] = field(default_factory=lambda: {"tiles": 0, "infer_s": 0.0, "rows": 0})

    def __post_init__(self):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = self.n_threads
        self._sess = ort.InferenceSession(self.onnx_path, so, providers=["CPUExecutionProvider"])

    # -- core inference -------------------------------------------------------
    def _infer_tiles(self, tiles: np.ndarray) -> List[Dict[str, np.ndarray]]:
        t0 = time.perf_counter()
        hm, wh, reg = self._sess.run(None, {"image": tiles[:, None].astype(np.float32)})
        self.stats["infer_s"] += time.perf_counter() - t0
        self.stats["tiles"] += len(tiles)
        return [decode_numpy(hm[i], wh[i], reg[i], self.stride, self.score_thr) for i in range(len(tiles))]

    def _process_block(self, block: np.ndarray, row_offset: int, batch: int = 8) -> None:
        """Tile a block [rows, cols] with overlap and queue detections in global coordinates."""
        R, Cc = block.shape
        step = self.tile - self.overlap
        r_starts = list(range(0, max(R - self.tile, 0) + 1, step))
        if r_starts[-1] + self.tile < R:
            r_starts.append(R - self.tile)
        c_starts = list(range(0, max(Cc - self.tile, 0) + 1, step))
        if c_starts[-1] + self.tile < Cc:
            c_starts.append(Cc - self.tile)
        coords, tiles = [], []
        for r in r_starts:
            for c in c_starts:
                t = block[r: r + self.tile, c: c + self.tile]
                if t.shape != (self.tile, self.tile):
                    pad = np.zeros((self.tile, self.tile), np.float32)
                    pad[: t.shape[0], : t.shape[1]] = t
                    t = pad
                tiles.append(t)
                coords.append((r, c))
        for b0 in range(0, len(tiles), batch):
            dets = self._infer_tiles(np.stack(tiles[b0: b0 + batch]))
            for d, (r, c) in zip(dets, coords[b0: b0 + batch]):
                if len(d["boxes"]) == 0:
                    continue
                bx = d["boxes"].copy()
                bx[:, [0, 2]] += c
                bx[:, [1, 3]] += r + row_offset
                self._pending.append({"boxes": bx, "scores": d["scores"], "labels": d["labels"]})

    def _flush_pending(self, upto_row: Optional[int] = None) -> List[Contact]:
        if not self._pending:
            return []
        boxes = np.concatenate([p["boxes"] for p in self._pending])
        scores = np.concatenate([p["scores"] for p in self._pending])
        labels = np.concatenate([p["labels"] for p in self._pending])
        keep = nms(boxes, scores, self.nms_iou)
        new = []
        remain = []
        for k in keep:
            cy = 0.5 * (boxes[k, 1] + boxes[k, 3])
            if upto_row is not None and cy > upto_row:
                remain.append(k)  # may still receive overlapping tiles -> keep pending
                continue
            # suppress duplicates of already-emitted contacts in the spatial neighbourhood
            dup = False
            for c in (c for c in self.contacts[-2000:] if abs(c.ping_index - cy) < 2 * self.tile):
                cb = np.array(c.box)
                inter = max(0, min(cb[2], boxes[k, 2]) - max(cb[0], boxes[k, 0])) * max(0, min(cb[3], boxes[k, 3]) - max(cb[1], boxes[k, 1]))
                a1 = (cb[2] - cb[0]) * (cb[3] - cb[1]); a2 = (boxes[k, 2] - boxes[k, 0]) * (boxes[k, 3] - boxes[k, 1])
                if inter / max(a1 + a2 - inter, 1e-9) > self.nms_iou:
                    dup = True
                    break
            if dup:
                continue
            ct = Contact(id=len(self.contacts), ping_index=float(cy), range_px=float(0.5 * (boxes[k, 0] + boxes[k, 2])),
                         box=boxes[k].tolist(), score=float(scores[k]), label=int(labels[k]), t_detect=time.time())
            self.contacts.append(ct)
            new.append(ct)
        self._pending = [{"boxes": boxes[remain], "scores": scores[remain], "labels": labels[remain]}] if remain else []
        return new

    # -- public API -----------------------------------------------------------
    def push(self, strip: np.ndarray) -> List[Contact]:
        """Push a new along-track strip [rows, cols] (float32 in [0,1]). Returns new contacts."""
        strip = strip.astype(np.float32)
        buf = strip if self._buffer is None else np.concatenate([self._buffer, strip], 0)
        start_row = self._rows_consumed - (0 if self._buffer is None else len(self._buffer))
        new: List[Contact] = []
        if len(buf) >= self.tile:
            self._process_block(buf, start_row)
            self.stats["rows"] += len(strip)
            # rows that can no longer be touched by future tiles are final
            self._buffer = buf[-self.overlap:].copy()
            self._rows_consumed = start_row + len(buf)
            new = self._flush_pending(upto_row=self._rows_consumed - self.overlap)
        else:
            self._buffer = buf
            self._rows_consumed = start_row + len(buf)
        return new

    def finish(self) -> List[Contact]:
        """Process any remaining buffered rows and flush all pending contacts."""
        if self._buffer is not None and len(self._buffer) > self.overlap:
            start_row = self._rows_consumed - len(self._buffer)
            self._process_block(self._buffer, start_row)
        new = self._flush_pending(None)
        self._buffer = None
        return new


def benchmark_realtime(atr: StreamingATR, pixel_m: float = 0.03, swath_m: float = 200.0, speed_kn: float = 10.0,
                       sides: int = 2) -> Dict[str, float]:
    """Compare the measured tile throughput with what the sonar produces.

    Area rate = sides * swath * speed. Pixel rate = area / pixel^2. Required
    tile rate accounts for the overlap re-processing factor.
    """
    speed = speed_kn * 0.514444
    area_rate = sides * swath_m * speed                           # m^2 / s
    pix_rate = area_rate / pixel_m**2
    eff = (atr.tile - atr.overlap) ** 2
    req_tiles_s = pix_rate / eff
    got_tiles_s = atr.stats["tiles"] / max(atr.stats["infer_s"], 1e-9)
    return {"area_rate_m2_s": area_rate, "pixel_rate_Mpix_s": pix_rate / 1e6, "required_tiles_s": req_tiles_s,
            "measured_tiles_s": got_tiles_s, "realtime_factor": got_tiles_s / max(req_tiles_s, 1e-9),
            "ms_per_tile": 1e3 / max(got_tiles_s, 1e-9)}
